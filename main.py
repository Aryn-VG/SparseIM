from args import get_args
from dataloader import dataloader
from dgl.data.utils import load_graphs
import torch
import dgl
from Memory import GRUMemoryUpdater,Memory_updater
from Message import MLPMessageFunction,Message_computer,Raw_Message_Updater
from embeding import ATTN
from time_encode import TimeEncode
from decoder import  Decoder
from val_eval import get_current_ts,eval_epoch
import logging
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
    g.ndata['h']=torch.zeros((g.number_of_nodes(),args.emb_dim))
    if args.use_expire_edge:
        max_ts = torch.max(g.edata["timestamp"])
        g.edata['duration']=torch.ones(g.number_of_edges())*max_ts/g.number_of_nodes()
    if not args.use_no_te:
        g.ndata['last_update'] = torch.zeros(g.number_of_nodes())
    if not args.use_no_memory:
        memory = torch.zeros((g.number_of_nodes(), args.memory_dim))
        g.ndata['memory'] = memory
        g.ndata['raw_message'] = torch.zeros((g.number_of_nodes(), args.memory_dim * 2 + args.time_dim + args.edge_feat_dim))
    return g


if __name__ == '__main__':
    args = get_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if not args.use_no_log:
        logger = get_log('log/'+args.prefix+args.data+'_'+args.tasks+'.txt')
        logger.info(args)
        logger.info("device:%s"%device)
    else:
        print(args)
        print("device:%s"%device)

    if args.data=='mooc':
        g = load_graphs('./data/mooc.bin')[0][0]
        g = dgl.add_reverse_edges(g, copy_edata=True)
        g.edata['timestamp'] = g.edata['timestamp'].float()
        g.edata['feat'] = g.edata['feats']
    elif args.data=='wikipedia':
        g = load_graphs(f"./data/wikipedia.dgl")[0][0]
    elif args.data=='reddit':
        g = load_graphs(f"./data/reddit.dgl")[0][0]
    else:
        print('ivalid dataset')




    train_loader, val_loader, test_loader,_, val_enum, test_enum = dataloader(args,g)

    if not args.use_no_te:
        time_encoder = TimeEncode(args.time_dim).to(device)
        t0 = torch.zeros(g.number_of_nodes())
    else:
        args.time_dim = 0
        time_encoder=None

    if not args.use_no_memory:
        raw_message_dim = 2 * args.message_dim + args.edge_feat_dim + args.time_dim
        message_func = MLPMessageFunction(raw_message_dim, args.message_dim).to(device)
        memory_computer = GRUMemoryUpdater(args).to(device)
        message_computer = Message_computer(time_encoder, message_func)
        memory_updater = Memory_updater(memory_computer)
        raw_message_updater = Raw_Message_Updater(time_encoder)
    else:
        message_func = None
        memory_computer = None
        message_computer =None
        memory_updater = None
        raw_message_updater=None
    emb_updater = ATTN(args, time_encoder).to(device)
    decoder=Decoder(args,args.emb_dim).to(device)
    loss_fcn = torch.nn.BCEWithLogitsLoss().to(device)
    if not args.use_no_memory:
        optimizer = torch.optim.Adam(list(message_func.parameters())+list(memory_computer.parameters())+list(decoder.parameters())+list(emb_updater.parameters()), lr=args.lr,
                                 weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(list(decoder.parameters()) + list(emb_updater.parameters()), lr=args.lr,weight_decay=args.weight_decay)


    args.memory_mask=False
    args.remove_masked_edge=False
    args.random_drop=False
    args.fine_tune=False
    args.spar_type='no'

    best_ap=0
    for i in range(args.n_epochs):
        g=init_epoch(args,g)
        if not args.use_no_memory:
            message_func.train()
            memory_computer.train()
        decoder.train()
        emb_updater.train()
        for batch_id, (input_nodes, pos_graph, neg_graph, blocks, frontier, current_ts) in enumerate(train_loader):
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            expire_loss=0
            for j in range(args.n_layers):
                blocks[j] = blocks[j].to(device)

            current_ts, pos_ts, num_pos_nodes = get_current_ts(args,pos_graph, neg_graph)
            pos_graph.ndata['ts'] = current_ts


            if not args.use_no_memory:

                pos_graph=dgl.add_reverse_edges(pos_graph, copy_edata=True)
                pos_graph=message_computer.message_aggregating(pos_graph)

                pos_graph=memory_updater.update_memory(pos_graph)

                for k in range(len(blocks)):
                    blocks[k].srcdata['h'][:blocks[-1].number_of_dst_nodes()] += pos_graph.ndata['memory']
                    blocks[k].dstdata['h'][:blocks[-1].number_of_dst_nodes()] += pos_graph.ndata['memory']

            blocks=emb_updater.forward(blocks)

            if not args.use_no_memory:
                pos_graph=raw_message_updater.update_raw_message(pos_graph)

            emb=blocks[-1].dstdata['h']

            logits, labels,_ = decoder(emb, pos_graph, neg_graph)

            if args.use_expire_edge:
                loss = loss_fcn(logits, labels)+args.expire_reg*expire_loss
            else:
                loss = loss_fcn(logits, labels)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            with torch.no_grad():
                if not args.use_no_memory:
                    idx_of_updated_node = torch.unique(
                        torch.cat([torch.unique(pos_graph.edges()[1]), torch.unique(pos_graph.edges()[0])])
                    )
                    nodeid = pos_graph.ndata[dgl.NID][idx_of_updated_node]
                    g.ndata['memory'][nodeid] =pos_graph.ndata['memory'][idx_of_updated_node].cpu()
                    g.ndata['raw_message'][nodeid] = pos_graph.ndata['raw_message'][idx_of_updated_node].cpu()
                if args.use_expire_edge:
                    for k in range(args.n_layers):
                        g.edata['duration'][blocks[k].edata['eid']] = blocks[k].edata['duration'].cpu()
                g.ndata['last_update'][pos_graph.ndata[dgl.NID][:num_pos_nodes]] = pos_ts.to('cpu')

        val_ap, val_auc, val_acc,val_mrr, val_loss,time_c,_ = eval_epoch(args,g, val_loader, emb_updater, decoder,
                                                        loss_fcn,message_computer,memory_updater,raw_message_updater,device,val_enum)
        if best_ap<val_ap:
            best_ap=val_ap
            if not args.use_no_memory:
                torch.save({'model_type': 'TGN', 'attn_emb_updater': emb_updater.state_dict(),
                            'decoder': decoder.state_dict(), 'message_func': message_func.state_dict(),
                            'memory_computer': memory_computer.state_dict()},
                           './checkpoint/' + 'TGN_' + args.data + args.n_degrees+ '.pth')
            else:
                torch.save({'model_type': 'TGAT', 'attn_emb_updater': emb_updater.state_dict(),
                            'decoder': decoder.state_dict(), 'message_func': None, 'memory_updater': None},
                           './checkpoint/' + 'TGAT_' + args.data + '.pth')

        if not args.use_no_log:
            logger.info("epoch:%d,loss:%f,ap:%f,auc:%f,mrr:%f,time_consume:%f" % (i, val_loss, val_ap,val_auc,val_mrr, time_c))
        else:
            print("epoch:%d,loss:%f,ap:%f,auc:%f,mrr:%f,time_consume:%f" % (i, val_loss, val_ap,val_auc,val_mrr, time_c))
    if not args.use_no_memory:
        model_CKPT=torch.load('./checkpoint/' + 'TGN_' + args.data + '.pth',map_location=torch.device(device))
        memory_computer.load_state_dict(model_CKPT['memory_computer'])
        message_func.load_state_dict(model_CKPT['message_func'])
    else:
        model_CKPT = torch.load('./checkpoint/' + 'TGAT_' + args.data + '.pth', map_location=torch.device(device))
    emb_updater.load_state_dict(model_CKPT['attn_emb_updater'])
    decoder.load_state_dict(model_CKPT['decoder'])

    test_ap, test_auc, test_acc,test_mrr,test_loss, test_time,_ = eval_epoch(args,g, test_loader, emb_updater, decoder,
                                                        loss_fcn,message_computer,memory_updater,raw_message_updater,device,test_enum)

    if not args.use_no_log:
        logger.info("Test:loss:%f,ap:%f,auc:%f,acc:%f,mrr:%f,time:%f" % (test_loss, test_ap, test_auc, test_acc,test_mrr,test_time))
    else:
        print("Test:loss:%f,ap:%f,auc:%f,acc:%f,mrr:%f,time:%f" % (test_loss, test_ap, test_auc, test_acc,test_mrr,test_time))
