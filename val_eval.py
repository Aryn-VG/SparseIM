from pytorch_lightning.metrics.functional import accuracy, auroc, average_precision
import torch
import numpy as np
import dgl
import time
import dgl.function as fn

def get_current_ts(args,pos_graph, neg_graph):
    with pos_graph.local_scope():
        pos_graph_ = dgl.add_reverse_edges(pos_graph, copy_edata=True)
        pos_graph_.update_all(fn.copy_e('timestamp', 'times'), fn.max('times','ts'))
        current_ts = pos_ts = pos_graph_.ndata['ts']
        num_pos_nodes = pos_graph_.num_nodes()
    if args.tasks == 'LP':
        with neg_graph.local_scope():
            neg_graph_ = dgl.add_reverse_edges(neg_graph)
            if args.neg_list_num>1:
                neg_ts=[]
                for i in pos_graph_.edata['timestamp']:
                    neg_ts+=i.repeat(args.neg_list_num)
                neg_graph_.edata['timestamp']=torch.tensor(neg_ts,device='cuda:0' if torch.cuda.is_available() else 'cpu')
            else:
                neg_graph_.edata['timestamp'] = pos_graph_.edata['timestamp']
            neg_graph_.update_all(fn.copy_e('timestamp', 'times'), fn.max('times', 'ts'))
            num_pos_nodes = torch.where(pos_graph_.ndata['ts'] > 0)[0].shape[0]
            pos_ts = pos_graph_.ndata['ts'][:num_pos_nodes]
            neg_ts = neg_graph_.ndata['ts'][num_pos_nodes:]
            current_ts = torch.cat([pos_ts, neg_ts])
    return current_ts, pos_ts, num_pos_nodes
def eval_epoch(args,g, dataloader, attn,decoder, loss_fcn, message_computer,memory_updater,raw_message_updater,device,num_samples,probe=None,memory_masker=None,baseline_vec=None):
    if args.tasks == 'LP':
        m_ap, m_auc, m_acc = [[], [], []]
    if args.tasks == 'NC':
        m_ap, m_auc, m_acc = [0,0,0]
        labels_all = torch.zeros((num_samples))
        logits_all = torch.zeros((num_samples))

    m_loss = []
    m_infer_time = []
    m_mrr=[]
    remain_feat_epoch = 0
    total_edge_epoch=0
    num_edge = 0
    with torch.no_grad():
        attn.eval()
        decoder.eval()
        for batch_idx, (input_nodes, pos_graph, neg_graph, blocks, frontier, current_ts) in enumerate(dataloader):
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            if args.tasks == 'NC':
                n_sample = pos_graph.num_edges()
                start_idx = batch_idx * n_sample
                end_idx = min(num_samples, start_idx + n_sample)
            for j in range(args.n_layers):
                blocks[j] = blocks[j].to(device)
            num_edge+=frontier.number_of_edges()
            current_ts, pos_ts, num_pos_nodes = get_current_ts(args,pos_graph, neg_graph)
            pos_graph.ndata['ts'] = current_ts

            start = time.time()
            if not args.use_no_memory:
                pos_graph = dgl.add_reverse_edges(pos_graph, copy_edata=True)
                pos_graph = message_computer.message_aggregating(pos_graph)
                pos_graph = memory_updater.update_memory(pos_graph)
                if memory_masker is not None:
                    pos_graph.ndata['masked_memory'], _, _, _,_ = memory_masker(pos_graph.ndata['memory'])
                    for k in range(len(blocks)):
                        blocks[k].srcdata['masked_h'][:blocks[-1].number_of_dst_nodes()] += pos_graph.ndata['masked_memory']
                        blocks[k].dstdata['masked_h'][:blocks[-1].number_of_dst_nodes()] += pos_graph.ndata['masked_memory']
                for k in range(len(blocks)):
                    blocks[k].srcdata['h'][:blocks[-1].number_of_dst_nodes()] += pos_graph.ndata['memory']
                    blocks[k].dstdata['h'][:blocks[-1].number_of_dst_nodes()] += pos_graph.ndata['memory']
            if probe is not None:
                if memory_masker is not None:
                    blocks, _, remain_feat_batch, total_edge_batch, _= attn.forward(blocks,analyser=probe,probe_train=False,memory_masked=True)
                    emb = blocks[-1].dstdata['masked_h']
                else:
                    blocks, _, remain_feat_batch, total_edge_batch, _= attn.forward(blocks,analyser=probe,probe_train=False,inference_baseline=baseline_vec)
                    emb = blocks[-1].dstdata['h']
            else:
                blocks = attn.forward(blocks,inference_baseline=baseline_vec)
                emb = blocks[-1].dstdata['h']
            if not args.use_no_memory:
                pos_graph = raw_message_updater.update_raw_message(pos_graph)
            logits, labels ,ranks= decoder(emb, pos_graph, neg_graph)

            end = time.time() - start
            m_infer_time.append(end)
            loss = loss_fcn(logits, labels)
            m_loss.append(loss.item())
            if probe is not None:
                remain_feat_epoch+=remain_feat_batch.item()
                total_edge_epoch+=float(total_edge_batch)
            if not args.use_no_memory:
                # pos_graph和全图id的联系
                idx_of_updated_node = torch.unique(
                    torch.cat(
                        [torch.unique(pos_graph.edges()[1]), torch.unique(pos_graph.edges()[0])]))  # pos_graph中节点的子图id
                nodeid = pos_graph.ndata[dgl.NID][idx_of_updated_node]  # pos_graph中节点对应的全图id
                g.ndata['memory'][nodeid] =pos_graph.ndata['memory'][idx_of_updated_node].cpu()
                if args.memory_mask:
                    g.ndata['masked_memory'][nodeid] = pos_graph.ndata['masked_memory'][idx_of_updated_node].cpu()
                g.ndata['raw_message'][nodeid] = pos_graph.ndata['raw_message'][idx_of_updated_node].cpu()
            #g.ndata['h'][idx_of_updated_node_block] =blocks[-1].dstdata['h'].cpu()
            g.ndata['last_update'][pos_graph.ndata[dgl.NID][:num_pos_nodes]] = pos_ts.to('cpu')#更新last_update

            if args.tasks == 'LP':
                pred = logits.sigmoid() > 0.5
                m_mrr.append(np.mean([1.0 / r for r in ranks]))
                m_ap.append(average_precision(logits, labels.int()).cpu().numpy())
                m_auc.append(auroc(logits.sigmoid(), labels.int()).cpu().numpy())
                m_acc.append(accuracy(pred, labels.int()).cpu().numpy())

            if args.tasks == 'NC':
                labels_all[start_idx:end_idx] = labels
                logits_all[start_idx:end_idx] = logits

    if args.tasks == 'LP':
        ap, auc, acc, mrr= np.mean(m_ap), np.mean(m_auc), np.mean(m_acc), np.mean(m_mrr)
    if args.tasks == 'NC':
        mrr=None
        pred_all = logits_all.sigmoid() > 0.5
        ap = average_precision(logits_all, labels_all).cpu().item()
        auc = auroc(logits_all, labels_all).cpu().item()
        acc = accuracy(pred_all, labels_all).cpu().item()


    time_c=np.sum(m_infer_time)
    attn.train()
    decoder.train()
    #print(baseline_vec)
    if probe is not None:
        remain_feat_rate=remain_feat_epoch/total_edge_epoch
        return ap, auc, acc,mrr, np.mean(m_loss),time_c,remain_feat_rate
    else:
        return ap, auc, acc,mrr, np.mean(m_loss),time_c,num_edge


