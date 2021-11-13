from explainer import GraphMaskProbe,LagrangianOptimization
from dataloader import dataloader
from dgl.data.utils import load_graphs
from args import get_args
import torch
import logging
import numpy as np
import dgl
from val_eval import get_current_ts,eval_epoch
from Memory import GRUMemoryUpdater,Memory_updater
from Message import MLPMessageFunction,Message_computer,Raw_Message_Updater
from embeding import ATTN
from time_encode import TimeEncode
from decoder import  Decoder
def get_log(file_name):
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fh = logging.FileHandler(file_name, mode='a')
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
class GraphMaskAnalyser:
    def __init__(self,args):
        self.args=args
        self.device='cuda:0' if torch.cuda.is_available() else 'cpu'
        self.D_criterion = torch.nn.BCEWithLogitsLoss().to(self.device)

    def initialise_for_model(self,model_path=None):
        if not self.args.use_no_te:
            vec_dim=self.args.edge_feat_dim+self.args.time_dim+self.args.emb_dim
        else:
            vec_dim=self.args.edge_feat_dim+self.args.emb_dim
        parameters_list=[]
        self.probe = GraphMaskProbe(vec_dim, vec_dim // 2).to(self.device)
        parameters_list+=self.probe.parameters()
        if self.args.memory_mask:
            self.memory_masker=GraphMaskProbe(self.args.memory_dim,self.args.memory_dim//2).to(self.device)
            parameters_list += self.memory_masker.parameters()
        else:
            self.memory_masker=None
        if self.args.finetune and model_path is not None:
            model_CKPT = torch.load(model_path, map_location=torch.device(self.device))
            self.probe.load_state_dict(model_CKPT['prober'])
            for p in parameters_list:
                p.requires_grad = False
            self.optimizer = torch.optim.Adam(self.gnn_params, lr=1e-5)
        else:
            optimizer = torch.optim.Adam(parameters_list, lr=self.args.lagrangian_lr)

            self.lagrangian_optimization = LagrangianOptimization(optimizer,
                                                             self.device,
                                                             batch_size_multiplier=None)

    def load_checkpoint(self,model_path,g):
        model_CKPT = torch.load(model_path,map_location=torch.device(self.device))
        if not self.args.use_no_te:
            time_encoder = TimeEncode(self.args.time_dim).to(self.device)
            t0 = torch.zeros(g.number_of_nodes())
        else:
            self.args.time_dim = 0
            time_encoder = None
        self.time_encoder=time_encoder
        if not self.args.use_no_memory:
            raw_message_dim = 2 * self.args.message_dim + self.args.edge_feat_dim + self.args.time_dim
            message_func = MLPMessageFunction(raw_message_dim, self.args.message_dim).to(self.device)
            message_func.load_state_dict(model_CKPT['message_func'])

            memory_computer = GRUMemoryUpdater(self.args).to(self.device)
            memory_computer.load_state_dict(model_CKPT['memory_computer'])

            message_computer = Message_computer(time_encoder, message_func)
            memory_updater = Memory_updater(memory_computer)
            raw_message_updater = Raw_Message_Updater(time_encoder)
        else:
            message_func = None
            memory_computer = None
            message_computer = None
            memory_updater = None
            raw_message_updater = None
        emb_updater = ATTN(self.args, time_encoder)
        emb_updater.load_state_dict(model_CKPT['attn_emb_updater'])

        decoder = Decoder(self.args, self.args.emb_dim).to(self.device)
        decoder.load_state_dict(model_CKPT['decoder'])
        if not self.args.use_no_memory:
            total_params=list(message_func.parameters())+list(memory_computer.parameters())+list(decoder.parameters())+list(emb_updater.parameters())
        else:
            total_params =list(decoder.parameters()) + list(emb_updater.parameters())

        if not self.args.finetune:
            for p in total_params:
                p.requires_grad = False
        else:
            self.gnn_params=total_params

        return emb_updater,message_func,memory_computer,message_computer,memory_updater,raw_message_updater,decoder
    def explainer_train(self,args,g, model_path,probe_path=None):
        emb_updater, message_func, memory_computer, message_computer, memory_updater, raw_message_updater, decoder=self.load_checkpoint(model_path,g)
        self.initialise_for_model(probe_path)
        train_loader, val_loader, test_loader,inference_loader, val_enum, test_enum = dataloader(args, g,time_encoder=self.time_encoder,masker=self.probe)
        if not args.use_no_log:
            logger = get_log('log/'+args.prefix + args.data + '_' + args.tasks + '.txt')
            logger.info(args)
            logger.info("device:%s" % self.device)
        else:
            print(args)
            print("device:%s" % self.device)
        best_ap=0
        for epoch in range(self.args.ana_epochs):
            g = init_epoch(args, g)
            self.probe.train()
            m_penalty = []
            m_differ=[]
            masked_edge_epoch=0
            total_edge_epoch=0

            for batch_idx, (input_nodes, pos_graph, neg_graph, blocks, frontier, current_ts) in enumerate(train_loader):
                emb_updater.penalty = 0
                pos_graph = pos_graph.to(self.device)
                neg_graph = neg_graph.to(self.device)
                for j in range(args.n_layers):
                    blocks[j] = blocks[j].to(self.device)

                current_ts, pos_ts, num_pos_nodes = get_current_ts(args,pos_graph, neg_graph)
                pos_graph.ndata['ts'] = current_ts
                if not args.use_no_memory:
                    # 计算&聚合message
                    pos_graph = dgl.add_reverse_edges(pos_graph, copy_edata=True)
                    pos_graph = message_computer.message_aggregating(pos_graph)
                    # 更新memory
                    pos_graph = memory_updater.update_memory(pos_graph)
                    if args.memory_mask:
                        pos_graph.ndata['masked_memory'],_,_,_,_=self.memory_masker(pos_graph.ndata['memory'])
                        for k in range(len(blocks)):
                            blocks[k].srcdata['masked_h'][:blocks[-1].number_of_dst_nodes()] += pos_graph.ndata['masked_memory']
                            blocks[k].dstdata['masked_h'][:blocks[-1].number_of_dst_nodes()] += pos_graph.ndata['masked_memory']
                    for k in range(len(blocks)):
                        blocks[k].srcdata['h'][:blocks[-1].number_of_dst_nodes()] += pos_graph.ndata['memory']
                        blocks[k].dstdata['h'][:blocks[-1].number_of_dst_nodes()] += pos_graph.ndata['memory']

                ori_blocks = emb_updater.forward(blocks)
                masked_blocks,penalty,masked_edge_batch,total_edge_batch,baseline_vec=emb_updater.forward(blocks,analyser=self.probe,memory_masked=args.memory_mask)
                ori_emb = ori_blocks[-1].dstdata['h']
                if args.memory_mask:
                    masked_emb=masked_blocks[-1].dstdata['masked_h']
                else:
                    masked_emb = masked_blocks[-1].dstdata['h']
                if not args.use_no_memory:
                    pos_graph = raw_message_updater.update_raw_message(pos_graph)

                ori_logits, labels,ranks= decoder(ori_emb, pos_graph, neg_graph)

                # ori_pred=ori_logits.sigmoid() > 0.5
                # ori_acc=accuracy(ori_pred, labels.int())
                # ori_ap=average_precision(ori_logits, labels.int())

                masked_logits, _,_=decoder(masked_emb, pos_graph, neg_graph)
                differ=self.D_criterion(masked_logits,labels)



                m_penalty.append(penalty.item())
                m_differ.append(differ.item())
                masked_edge_epoch+=masked_edge_batch.item()
                total_edge_epoch+=float(total_edge_batch)

                if not self.args.finetune:
                    g_loss= torch.relu(differ - self.args.Lagrangian_allowance).mean()
                    f_loss=penalty* self.args.penalty_scaling
                    self.lagrangian_optimization.update(f_loss,g_loss)
                else:
                    self.optimizer.zero_grad()
                    differ.backward()
                    self.optimizer.step()

                if not args.use_no_memory:
                    # pos_graph和全图id的联系
                    idx_of_updated_node = torch.unique(
                        torch.cat(
                            [torch.unique(pos_graph.edges()[1]),
                             torch.unique(pos_graph.edges()[0])]))  # pos_graph中节点的子图id
                    nodeid = pos_graph.ndata[dgl.NID][idx_of_updated_node]  # pos_graph中节点对应的全图id
                    g.ndata['memory'][nodeid] = pos_graph.ndata['memory'][idx_of_updated_node].cpu()
                    if args.memory_mask:
                        g.ndata['masked_memory'][nodeid] = pos_graph.ndata['masked_memory'][idx_of_updated_node].cpu()
                    g.ndata['raw_message'][nodeid] = pos_graph.ndata['raw_message'][idx_of_updated_node].cpu()
                # g.ndata['h'][idx_of_updated_node_block] =blocks[-1].dstdata['h'].cpu()
                g.ndata['last_update'][pos_graph.ndata[dgl.NID][:num_pos_nodes]] = pos_ts.to('cpu')  # 更新last_update
            #val
            edge_penalty=np.mean(m_penalty)
            output_diff=np.mean(m_differ)
            train_remain_rate=masked_edge_epoch/total_edge_epoch

            ori_ap, ori_auc, ori_acc, ori_mrr,_,_,ori_edge_num = eval_epoch(args, g, val_loader, emb_updater, decoder,
                                                                    self.D_criterion, message_computer, memory_updater,
                                                                    raw_message_updater, self.device, val_enum)

            if args.remove_masked_edge:
                removed_ap,removed_auc,removed_acc,remove_mrr,_,_,remain_edge_rate=eval_epoch(args, g, val_loader, emb_updater, decoder,self.D_criterion, message_computer, memory_updater,
                                                                    raw_message_updater, self.device, val_enum,probe=self.probe,baseline_vec=baseline_vec)
                masked_ap, masked_auc, masked_acc, masked_mrr, masked_loss, masked_time_c, remain_feat_rate = eval_epoch(
                    args, g, val_loader, emb_updater, decoder,
                    self.D_criterion, message_computer, memory_updater,
                    raw_message_updater, self.device, val_enum, probe=self.probe, memory_masker=self.memory_masker)
                if not args.use_no_log:
                    logger.info("epoch:%d,ori_ap:%f,ori_acc:%f,penalty:%f,masked_ap:%f,removed_ap:%f,removed_acc:%f,train_remain_feat:%f,val_remain_feat:%f,val_remain_edge:%f" % (epoch, ori_ap, ori_acc,edge_penalty,masked_ap,removed_ap,removed_acc,train_remain_rate,remain_feat_rate,remain_edge_rate))
                else:
                    print("epoch:%d,ori_ap:%f,ori_acc:%f,penalty:%f,masked_ap:%f,removed_ap:%f,removed_acc:%f,train_remain_feat:%f,val_remain_feat:%f,val_remain_edge:%f" % (epoch, ori_ap, ori_acc,edge_penalty,masked_ap,removed_ap,removed_acc,train_remain_rate,remain_feat_rate,remain_edge_rate))

            else:
                masked_ap, masked_auc, masked_acc,masked_mrr, masked_loss, masked_time_c,val_masked_rate= eval_epoch(args, g, val_loader, emb_updater, decoder,
                                                                        self.D_criterion, message_computer,memory_updater,
                                                                            raw_message_updater, self.device, val_enum,probe=self.probe,memory_masker=self.memory_masker)
                if not args.use_no_log:
                    logger.info("epoch:%d,ori_ap:%f,masked_ap:%f,penalty:%f,differ:%f,train_masked:%f,val_masked:%f,time_consume:%f" % (epoch, ori_ap, masked_ap,edge_penalty,output_diff,train_remain_rate,val_masked_rate,masked_time_c))
                else:
                    print("epoch:%d,ori_ap:%f,masked_ap:%f,penaltyL::%f,differ:%f,train_masked:%f,val_masked:%f,time_consume:%f" % (epoch, ori_ap, masked_ap,edge_penalty,output_diff,train_remain_rate,val_masked_rate,masked_time_c))

            if remain_edge_rate<self.args.stop_proportion and self.args.random_drop==0:
                if best_ap< removed_ap:
                    best_ap=removed_ap
                    torch.save({'model_type': 'probe', 'prober': self.probe.state_dict()},'./checkpoint/exp_' + args.data +'.pth')
                    logger.info('current best ap:%f,and its prop:%f'%(best_ap,remain_edge_rate))
        if self.args.random_drop>0:
            pass
        else:
            best_model = torch.load('./checkpoint/exp_' + args.data +'.pth', map_location=torch.device(self.device))
            self.probe.load_state_dict(best_model['prober'])
        test_ap, test_auc, test_acc, test_mrr,_,_,test_remain_edge = eval_epoch(args, g, val_loader, emb_updater, decoder,self.D_criterion, message_computer, memory_updater,
                                                                    raw_message_updater, self.device, val_enum,probe=self.probe,baseline_vec=baseline_vec)
        if not args.use_no_log:
            logger.info("Test:remain_edge:%f,ap:%f,auc:%f,acc:%f,mrr:%f" % (
            test_remain_edge, test_ap, test_auc, test_acc, test_mrr))
        else:
            print("Test:remain_edge:%f,ap:%f,auc:%f,acc:%f,mrr:%f" % (
            test_remain_edge, test_ap, test_auc, test_acc, test_mrr))


def init_epoch(args,g):
    g.ndata['h']=torch.zeros((g.number_of_nodes(),args.emb_dim))
    if args.memory_mask:
        g.ndata['masked_h'] = torch.zeros((g.number_of_nodes(), args.emb_dim))
        g.ndata['masked_memory'] = torch.zeros((g.number_of_nodes(), args.memory_dim))
    if args.use_expire_edge:
        max_ts = torch.max(g.edata["timestamp"])
        g.edata['duration']=torch.ones(g.number_of_edges())*max_ts/g.number_of_nodes()
    if not args.use_no_te:
        g.ndata['last_update'] = torch.zeros(g.number_of_nodes())
    if not args.use_no_memory:
        g.ndata['memory'] = torch.zeros((g.number_of_nodes(), args.memory_dim))  # 初始化节点memory
        g.ndata['raw_message'] = torch.zeros((g.number_of_nodes(), args.memory_dim * 2 + args.time_dim + args.edge_feat_dim))
    return g

if __name__ == '__main__':
    args=get_args()
    analyser=GraphMaskAnalyser(args)
    if args.data == 'mooc':
        g = load_graphs('./data/mooc.bin')[0][0]
        g = dgl.add_reverse_edges(g, copy_edata=True)
        g.edata['timestamp'] = g.edata['timestamp'].float()
        g.edata['feat'] = g.edata['feats']
    elif args.data == 'wikipedia':
        g = load_graphs(f"./data/wikipedia.dgl")[0][0]
    elif args.data == 'reddit':
        g = load_graphs(f"./data/reddit.dgl")[0][0]
    else:
        print('invalid dataset')


    model_path='./checkpoint/TGAT_'+args.data+'.pth'
    #analyser.initialise_for_model()
    if args.finetune:
        probe_path='exp_' +args.data+'_' +''+ '.pth'
    else:
        probe_path=None
    analyser.explainer_train(args,g, model_path,probe_path)
    print('analyse complete!')


