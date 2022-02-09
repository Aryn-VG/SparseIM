#from pytorch_lightning.metrics.functional import accuracy, auroc, average_precision
from torchmetrics.functional import accuracy, auroc, average_precision
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
def eval_epoch(args,g, dataloader, GNN_forward,decoder,loss_fcn=None,device=None,sparsifier=None):
    m_ap, m_auc, m_acc = [[], [], []]
    m_loss = []
    m_infer_time = []
    m_mrr=[]
    remaining_edge_epoch = 0
    total_edge_epoch=0
    with torch.no_grad():
        GNN_forward.eval()
        decoder.eval()
        for batch_idx, (input_nodes, pos_graph, neg_graph, blocks, frontier, current_ts) in enumerate(dataloader):
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            for j in range(args.n_layers):
                blocks[j] = blocks[j].to(device)
            current_ts, pos_ts, num_pos_nodes = get_current_ts(args,pos_graph, neg_graph)
            pos_graph.ndata['ts'] = current_ts

            start = time.time()
            if args.mode=='Pretrain':
                blocks = GNN_forward.forward(blocks)
                # loss
                emb = blocks[-1].dstdata['h']
                logits, labels, ranks = decoder(emb, pos_graph, neg_graph)
            else:
                # mask
                blocks, train_penalty_batch, remaining_edge_batch, total_edge_batch, s_ij = sparsifier.forward(blocks,
                                                                                                               training=True)  # block with new memory, ori emb and masked messages
                #blocks[-1].dstdata['h'] +=blocks[-1].dstdata['memory']
                blocks[-1].remove_edges(torch.where(s_ij == 0)[0])
                # undate emb
                blocks = GNN_forward.forward(blocks, sparsifier.manipulator)  # block with new emb
                emb = blocks[-1].dstdata['h']
                logits, labels, ranks = decoder(emb, pos_graph, neg_graph)
                # remaining edge rate
                remaining_edge_epoch += remaining_edge_batch.item()
                total_edge_epoch += float(total_edge_batch)
                g.ndata['memory'][blocks[-1].dstdata['_ID']] = blocks[-1].dstdata['memory'].cpu()
            inf_time = time.time() - start
            m_infer_time.append(inf_time)
            loss = loss_fcn(logits, labels)
            m_loss.append(loss.item())
            g.ndata['last_update'][pos_graph.ndata[dgl.NID][:num_pos_nodes]] = pos_ts.to('cpu').float()


            ## metric
            pred = logits.sigmoid() > 0.5
            m_mrr.append(np.mean([1.0 / r for r in ranks]))
            m_ap.append(average_precision(logits, labels.int()).cpu().numpy())
            m_auc.append(auroc(logits.sigmoid(), labels.int()).cpu().numpy())
            m_acc.append(accuracy(pred, labels.int()).cpu().numpy())

    ap, auc, acc, mrr= np.mean(m_ap), np.mean(m_auc), np.mean(m_acc), np.mean(m_mrr)


    time_c=np.sum(m_infer_time)
    if args.mode=='Pretrain':
        return ap, auc, acc, mrr, np.mean(m_loss), time_c
    else:
        remaining_edge_rate = remaining_edge_epoch / total_edge_epoch
        return ap, auc, acc, mrr, np.mean(m_loss), time_c, remaining_edge_rate



