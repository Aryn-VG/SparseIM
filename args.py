import argparse
import sys


def get_args():
    ### Argument and global variables
    parser = argparse.ArgumentParser('TGN')
    parser.add_argument('--prefix', type=str, default='TGAT', help='Prefix to name the checkpoints')
    parser.add_argument('--data', type=str,choices=["wikipedia", "reddit", "alipay","mooc"],
                        help='Dataset name (eg. wikipedia or reddit)',default='mooc')
    parser.add_argument('--tasks', type=str, default="LP", choices=["LP", "EC", "NC"],
                        help='task name link prediction, edge or node classification')
    parser.add_argument('--use_no_log', action='store_true', default=False,
                        help='do not use logger')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch_size')
    parser.add_argument('--n_workers', type=int, default=0, help='Number of network layers')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
    parser.add_argument('--seed', type=int, default=-1, help='Random Seed')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
    parser.add_argument('--warmup', action='store_true', help='')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--neg_list_num', type=int, default=1, help='the number of negtive graph')

    #################Parameters of TGN/TGAT###################
    parser.add_argument('--use_no_memory', action='store_true', default=True,
                        help='do not use memory')
    parser.add_argument('--use_no_te', action='store_true', default=False,
                        help='do not use time encoding')
    parser.add_argument('--memory_dim', type=int, default=100, help='memory_dimension')
    parser.add_argument('--message_dim', type=int, default=100, help='message_dimension')
    parser.add_argument('--emb_dim', type=int, default=100, help='emb_dimension')
    parser.add_argument('--time_dim', type=int, default=100, help='dimension of time-encoding')

    parser.add_argument('--n_degrees', type=int, default=10, help='Number of neighbors to sample')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of heads used in attention layer')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of network layers')
    parser.add_argument('--edge_feat_dim', type=int, default=4, help='Dimensions of the edge feature')
    parser.add_argument('--node_feat_dim', type=int, default=100, help='Dimensions of the node feature')
    parser.add_argument('--uniform', action='store_true',
                        help='take uniform sampling from temporal neighbors')
    parser.add_argument('--balance', action='store_true',
                        help='use fraud user sampling when doing EC or NC tasks')
    parser.add_argument('--pretrain', action='store_true',
                        help='use linkpred task model as pretrain model to learn EC or NC')
    parser.add_argument('--no_time', action='store_true',
                        help='do not use time embedding')
    parser.add_argument('--no_pos', action='store_true',
                        help='do not use position embedding')

    #################Parameters of annalyser###################
    parser.add_argument('--spar_type', type=str, choices=["neuralsparse", "dropedge", "sparseIM",'no'],
                        help='the type of sparsification method (eg. neuralsparse or immessage)', default='immessage')
    parser.add_argument('--ana_epochs', type=int, default=50, help='Number of analyse epochs')
    parser.add_argument('--Lagrangian_allowance', type=float, default=0.0, help='allowance of Lagrangian optimization')
    parser.add_argument('--penalty_scaling', type=float, default=0.02, help='the scaling rate of edgenumber penalty')
    parser.add_argument('--lagrangian_lr', type=float, default=3e-5, help='the learning rate of Lagrangian optimizer')
    parser.add_argument('--memory_mask', action='store_true', default=False, help='mask memory or not ')
    parser.add_argument('--remove_masked_edge', action='store_true', default=False, help='remove mask edge or not')
    parser.add_argument('--random_drop', type=float, default=0, help='remove mask edge randomly')
    parser.add_argument('--finetune', action='store_true', default=False, help='finetune or not ')
    parser.add_argument('--stop_proportion', type=float, default=0.3, help='the proportion of remain edges that our model need to meet ')
    parser.add_argument('--drop_training_edge', action='store_true', default=False, help='drop the edge on training phase')
    parser.add_argument('--im_message', action='store_true', default=False, help='use im message into aggregation or not ')





    try:
        args = parser.parse_args()
        assert args.n_workers == 0, "n_worker must be 0, etherwise dataloader will cause bug and results very bad performance (this bug will be fixed soon)"
        if args.data == 'alipay':
            args.feat_dim = 101
            args.lr *= 5
            args.bs *= 5
        elif args.data =='wikipedia' or args.data =='reddit':
            args.edge_feat_dim = 172
            args.node_feat_dim = 0
        elif args.data == 'mooc':
            args.edge_feat_dim = 4
            args.node_feat_dim = 0
        else:
            args.edge_feat_dim = 0
            args.node_feat_dim = 0
        if args.prefix=='TGN':
            args.use_no_memory=False
            args.n_layers=1
        else:
            args.use_no_memory = True
            args.n_layers=2
        args.no_time = True
        # if len(args.fanouts)!=args.n_layers:
        #     args.fanouts=[args.n_degrees for _ in range(args.n_layers)]
        # args.no_pos = True
        if args.spar_type=='sparseIM':
            args.remove_masked_edge=True
            args.drop_training_edge=False
            args.im_message=True
        elif args.spar_type== 'neuralsparse':
            args.remove_masked_edge = True
            args.drop_training_edge = True
            args.im_message = False
        elif args.spar_type=='dropedge':
            args.remove_masked_edge = True
            args.drop_training_edge = True
            args.im_message = False
            args.random_drop=args.stop_proportion
        else:
            args.remove_masked_edge = False
            args.random_drop=False
            args.finetune = False



    except:
        parser.print_help()
        sys.exit(0)

    return args

