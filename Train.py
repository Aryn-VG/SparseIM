import random
from data_preprocess import TemporalWikipediaDataset, TemporalRedditDataset, TemporalDataset
from args import get_args
from dataloader import dataloader
import torch

import dgl
from GNN_Model import ATTN
from decoder import  Decoder
from val_eval import get_current_ts,eval_epoch
import logging
from SparseIM import LagrangianOptimization,SparseIM
import numpy as np
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

def init_epoch(args,g):
    g.ndata['h']=torch.randn((g.number_of_nodes(),args.emb_dim))
    g.ndata['last_update'] = torch.zeros(g.number_of_nodes())
    g.ndata['memory'] = torch.randn((g.number_of_nodes(), args.memory_dim))
    return g
def choose_mode(args,GNN_model_path=None,Spar_model_path=None):
    if args.mode=="Pretrain":
        GNN_forward = ATTN(args).to(args.device)
        decoder = Decoder(args, args.emb_dim).to(args.device)

        optimizer = torch.optim.Adam(list(decoder.parameters()) + list(GNN_forward.parameters()), lr=args.GNN_lr,
                                     weight_decay=args.weight_decay)
        sparsifier=None

    elif args.mode=="SparseIM":
        ## fixed GNN model
        GNN_model_CKPT = torch.load(GNN_model_path, map_location=torch.device(args.device))
        GNN_forward = ATTN(args)
        GNN_forward.load_state_dict(GNN_model_CKPT['GNN_forward'])
        decoder = Decoder(args, args.emb_dim).to(args.device)
        decoder.load_state_dict(GNN_model_CKPT['decoder'])
        GNN_params = list(decoder.parameters()) + list(GNN_forward.parameters())
        for p in GNN_params:
            p.requires_grad = False

        ## sparsifier init
        sparsifier=SparseIM(args).to(args.device)
        optimizer = torch.optim.Adam(list(sparsifier.parameters()), lr=args.sparsifier_lr,
                                     weight_decay=args.weight_decay)

        #optimizer = LagrangianOptimization(Adam_optimizer,args.device,batch_size_multiplier=None)

    elif args.mode=="End2End":
        GNN_forward = ATTN(args).to(args.device)
        decoder = Decoder(args, args.emb_dim).to(args.device)
        sparsifier = SparseIM(args).to(args.device)
        optimizer = torch.optim.Adam(list(decoder.parameters()) + list(GNN_forward.parameters())+list(sparsifier.parameters()), lr=args.e2e_lr,
                                     weight_decay=args.weight_decay)
    else:
        GNN_forward=None
        decoder=None
        optimizer=None
        sparsifier=None


    return GNN_forward,decoder,optimizer,sparsifier



def Train(args,logger=None, g=None):

    init_epoch(args,g)
    ### dataloader
    train_loader, val_loader, test_loader, val_enum, test_enum,_ = dataloader(args, g)

    ### loss function
    loss_fcn = torch.nn.BCEWithLogitsLoss().to(args.device)
    ### choose mode
    GNN_model_path='./checkpoint/GNN_model/TGAT_' + args.dataset + '.pth'
    Spar_model_path='./checkpoint/Sparse_model/SparseIM_' + args.dataset + '.pth' if args.mode!='Pretrain' else None
    GNN_forward, decoder, optimizer, sparsifier=choose_mode(args,GNN_model_path=GNN_model_path,Spar_model_path=Spar_model_path)

    ### Training
    best_ap = 0
    last_prop = 1
    total_epochs=args.GNN_epochs if args.mode=='Pretrain' else args.n_epochs
    for epoch in range(total_epochs):
        if epoch>0:
            g = init_epoch(args, g)
        train_penalty = []
        train_differ = []
        total_edge_epoch=0
        remaining_edge_epoch=0
        loss_epoch=[]


        for batch_idx, (input_nodes, pos_graph, neg_graph, blocks, frontier, current_ts) in enumerate(train_loader):
            if batch_idx==0:
                continue
            pos_graph = pos_graph.to(args.device)
            neg_graph = neg_graph.to(args.device)
            for j in range(args.n_layers):
                blocks[j] = blocks[j].to(args.device)
            current_ts, pos_ts, num_pos_nodes = get_current_ts(args, pos_graph, neg_graph)
            pos_graph.ndata['ts'] = current_ts

            if args.mode=='Pretrain':
                blocks = GNN_forward.forward(blocks)
                # loss
                emb = blocks[-1].dstdata['h']
                logits, labels, ranks = decoder(emb, pos_graph, neg_graph)
                loss = loss_fcn(logits, labels)
                loss_epoch.append(loss.item())
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            else:
                # mask
                blocks,train_penalty_batch,remaining_edge_batch,total_edge_batch,s_ij=sparsifier.forward(blocks,training=True)# block with new memory, ori emb and masked messages
                for j in range(args.n_layers):
                    blocks[j].edata['feat'] = s_ij.unsqueeze(dim=1)*blocks[j].edata['feat']\
                                                 +(1-s_ij).unsqueeze(dim=1)*sparsifier.manipulator
                    blocks[j].edata['timestamp']=s_ij*blocks[j].edata['timestamp']
                #blocks[-1].dstdata['h'] +=blocks[-1].dstdata['memory']
                # undate emb
                blocks=GNN_forward.forward(blocks)# block with new emb

                # loss
                emb = blocks[-1].dstdata['h']
                logits, labels, ranks = decoder(emb, pos_graph, neg_graph)
                g_loss=loss_fcn(logits,labels)
                f_loss = train_penalty_batch
                if last_prop<=args.stop_proportion+0.03:
                    loss=g_loss
                else:
                    loss=g_loss+f_loss*args.penalty_scaling

                # backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sparsifier.parameters(), 5, norm_type=2)
                optimizer.step()



                train_penalty.append(train_penalty_batch.item())
                train_differ.append(g_loss.item())
                remaining_edge_epoch += remaining_edge_batch.item()
                total_edge_epoch += float(total_edge_batch)
                with torch.no_grad():
                    g.ndata['memory'][blocks[-1].dstdata['_ID']] = blocks[-1].dstdata['memory'] .cpu()
            with torch.no_grad():
                g.ndata['last_update'][pos_graph.ndata[dgl.NID][:num_pos_nodes]] = pos_ts.to('cpu').float()
        # Validation
        if args.mode=='Pretrain':
            ap, auc, acc, mrr,_,time = eval_epoch(args, g, val_loader,GNN_forward,decoder,loss_fcn,args.device)
            if args.use_log:
                logger.info(
                    "epoch:%d,loss:%f,ori_ap:%f,ori_auc:%f,ori_acc:%f,runing_time:%f"
                    % (epoch,np.mean(loss_epoch),ap,auc,acc,time))
            else:
                print(
                    "epoch:%d,loss:%f,ori_ap:%f,ori_auc:%f,ori_acc:%f,runing_time:%f"
                    % (epoch,np.mean(loss_epoch),ap,auc,acc,time))

            if ap>best_ap:
                best_ap=ap
                torch.save({'model_type': 'TGAT', 'GNN_forward': GNN_forward.state_dict(),'decoder':decoder.state_dict()},
                           './checkpoint/GNN_model/TGAT_' + args.dataset + '.pth')
                logger.info('current best ap:%f' % (best_ap)) if args.use_log else print('current best ap:%f' % (best_ap))
        else:
            ap, auc, acc, mrr, _,time, val_remaining_rate = eval_epoch(args, g, val_loader, GNN_forward, decoder,
                                                                      loss_fcn, args.device,sparsifier)
            if args.use_log:
                logger.info(
                    "epoch:%d,penalty:%f,sparse_ap:%f,sparse_auc:%f,sparse_acc:%f,train_remaining:%f,val_remaining_rate:%f,inf_time:%f"
                    % (epoch,np.mean(train_penalty),ap,auc,acc,remaining_edge_epoch/total_edge_epoch,val_remaining_rate,time))
            else:
                print(
                    "epoch:%d,penalty:%f,sparse_ap:%f,sparse_auc:%f,sparse_acc:%f,train_remaining:%f,val_remaining_rate:%f,inf_time:%f"
                    % (epoch,np.mean(train_penalty),ap,auc,acc,remaining_edge_epoch/total_edge_epoch,val_remaining_rate,time))

            if ap>best_ap and remaining_edge_epoch/total_edge_epoch<=args.stop_proportion+0.03:
                best_ap=ap
                last_prop=remaining_edge_epoch/total_edge_epoch

                if args.mode=='End2End':
                    torch.save({'model_type': 'Sparsifier', 'Sparsifier': sparsifier.state_dict()},
                               './checkpoint/Sparse_model/SparseIM_e2e_' + args.dataset + '.pth')
                    torch.save({'model_type': 'TGAT', 'GNN_forward': GNN_forward.state_dict(),'decoder':decoder.state_dict()},
                           './checkpoint/GNN_model/TGAT_e2e_' + args.dataset + '.pth')
                else:
                    torch.save({'model_type': 'Sparsifier', 'Sparsifier': sparsifier.state_dict()},
                               './checkpoint/Sparse_model/SparseIM' + args.dataset + '.pth')
                logger.info('current best ap:%f and its proportion:%f' % (best_ap,remaining_edge_epoch/total_edge_epoch)) if args.use_log else print('current best ap:%f and its proportion:%f' % (best_ap,remaining_edge_epoch/total_edge_epoch))
def Test(args,logger,g):
    ## Test phrase
    ## GNN model
    init_epoch(args,g)
    if args.mode=='Pretrain':
        GNN_model_path = './checkpoint/GNN_model/TGAT_' + args.dataset + '.pth'
    elif args.mode=='SparseIM':
        GNN_model_path = './checkpoint/GNN_model/TGAT_' + args.dataset + '.pth'
        Spar_model_path = './checkpoint/Sparse_model/SparseIM_' + args.dataset + '.pth'
    else:#end to end
        GNN_model_path = './checkpoint/GNN_model/TGAT_e2e_' + args.dataset + '.pth'
        Spar_model_path = './checkpoint/Sparse_model/SparseIM_e2e_' + args.dataset + '.pth'

    ## GNN load
    GNN_model_CKPT = torch.load(GNN_model_path, map_location=torch.device(args.device))
    GNN_forward = ATTN(args).to(args.device)
    GNN_forward.load_state_dict(GNN_model_CKPT['GNN_forward'])
    decoder = Decoder(args, args.emb_dim).to(args.device)
    decoder.load_state_dict(GNN_model_CKPT['decoder'])

    ## sparsifier load
    if args.mode!='Pretrain':
        sparsifier = SparseIM(args).to(args.device)
        Sparse_model_CKPT = torch.load(Spar_model_path, map_location=torch.device(args.device))
        sparsifier.load_state_dict(Sparse_model_CKPT['Sparsifier'])

    train_loader, val_loader, test_loader, val_enum, test_enum,_ = dataloader(args, g)
    if args.mode == 'pretrain':
        test_ap, test_auc, test_acc, test_mrr,_, test_time = eval_epoch(args, g, test_loader, GNN_forward, decoder, None, args.device)

        if args.use_log:
            logger.info(
                "test_ap:%f,test_auc:%f,test_acc:%f,test_time:%f"
                % (test_ap, test_auc, test_acc, test_time))
        else:
            print(
                "test_ap:%f,test_auc:%f,test_acc:%f,test_time:%f"
                % (test_ap, test_auc, test_acc, test_time))
    else:

        test_ap, test_auc, test_acc, test_mrr, _,test_time, test_remaining_rate = eval_epoch(args, g, test_loader, GNN_forward, decoder,
                                                                None, args.device, sparsifier)
        if args.use_log:
            logger.info(
                "Test_sparse_ap:%f,Test_sparse_auc:%f,Test_sparse_acc:%f,Test_remaining_rate:%f,inf_time:%f"
                % (test_ap, test_auc,test_acc,test_remaining_rate,
                   test_time))
        else:
            print(
                "Test_sparse_ap:%f,Test_sparse_auc:%f,Test_sparse_acc:%f,Test_remaining_rate:%f,inf_time:%f"
                % (test_ap, test_auc,test_acc,test_remaining_rate,
                   test_time))

if __name__ == '__main__':
    np.random.seed(1927)
    torch.manual_seed(1927)
    torch.cuda.manual_seed(1927)
    random.seed(1927)
    args = get_args()
    debug_mode=False

    ### load data
    if args.dataset == 'wikipedia':
        data = TemporalWikipediaDataset()
    elif args.dataset == 'reddit':
        data = TemporalRedditDataset()
    elif args.dataset == 'mooc':
        data=dgl.load_graphs('../data/mooc.bin')[0][0]
    else:
        print("Warning Using Untested Dataset: "+args.dataset)
        data = TemporalDataset(args.dataset)

    if debug_mode:
        if args.use_log:
            logger = get_log('log/test_' + args.mode + '.log')
            logger.info(args)
            logger.info("device:%s" % args.device)
        else:
            logger = None
            print(args)
            print("device:%s" % args.device)
        Train(args, logger=None,g=data)
    else:
        # tasks=[{'mode':'Pretrain','lr':1e-4,'ps':1.0,'note':'pre_attn'},
        #        {'mode':'End2End', 'lr':3e-4,'ps':1.0,'note':'e2e_ps1_lr3e4'}]
        # tasks=[{'mode':'End2End', 'lr':1e-4,'ps':1.0,'note':'e2e_ps1_lr1e4'}]
        # for task in tasks:
        #     args.mode=task['mode']
        #     args.penalty_scaling=task['ps']
        #     args.e2e_lr=task['lr']
        #
        #     if args.use_log:
        #         logger = get_log('log/'+args.dataset+'_'+task['note']+'.log')
        #         logger.info(args)
        #         logger.info("device:%s"%args.device)
        #     else:
        #         logger=None
        #         print(args)
        #         print("device:%s"%args.device)
        if args.use_log:
            logger = get_log('log/' + args.dataset + '_' + args.note + '.log')
            logger.info(args)
            logger.info("device:%s" % args.device)
        else:
            logger = None
            print(args)
            print("device:%s" % args.device)

        Train(args,logger,data)
        Test(args,logger,data)

















