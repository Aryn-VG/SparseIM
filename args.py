import argparse
import sys
import torch


def get_args():
    ### Argument and global variables
    parser = argparse.ArgumentParser('SparseIM')
    parser.add_argument('--prefix', type=str, default='SparseIM', help='Prefix to name the checkpoints')
    parser.add_argument('--mode', type=str, choices=["SparseIM", "Pretrain", "Finetune","End2End"],
                        help='SparseIM: training sparsifier, Pretrain: pretrain GNN model, Finetune: finetune GNN model'
                        , default='End2End')
    parser.add_argument('--note', type=str, default='', help='Prefix to name the checkpoints')
    parser.add_argument('--dataset', type=str,choices=["wikipedia", "reddit", "alipay","mooc"],
                        help='Dataset name (eg. wikipedia or reddit)',default='wikipedia')
    parser.add_argument('--tasks', type=str, choices=["LP", "NC", "LC", "GC"],
                        help='Task name (eg. link prediction or node classification)', default='LP')
    parser.add_argument('--use_log', action='store_true', default=False,
                        help='use logger or not')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch_size')
    parser.add_argument('--n_workers', type=int, default=0, help='Number of network layers')
    parser.add_argument('--GNN_lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
    parser.add_argument('--seed', type=int, default=-1, help='Random Seed')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
    parser.add_argument('--warmup', action='store_true', help='')
    parser.add_argument('--device', type=int, default=1, help='GPU or CPU')
    parser.add_argument('--GNN_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--neg_list_num', type=int, default=1, help='the number of negtive graph')

    #################Parameters of GNN###################
    parser.add_argument('--memory_dim', type=int, default=100, help='memory_dimension')
    parser.add_argument('--emb_dim', type=int, default=100, help='emb_dimension')
    parser.add_argument('--time_dim', type=int, default=100, help='dimension of time-encoding')

    parser.add_argument('--n_degrees', type=int, default=-1, help='Number of neighbors to sample')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of heads used in attention layer')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of network layers')
    parser.add_argument('--edge_feat_dim', type=int, default=4, help='Dimensions of the edge feature')
    parser.add_argument('--node_feat_dim', type=int, default=100, help='Dimensions of the node feature')
    parser.add_argument('--uniform', action='store_true',
                        help='take uniform sampling from temporal neighbors')
    parser.add_argument('--balance', action='store_true',
                        help='use fraud user sampling when doing EC or NC tasks')
    parser.add_argument('--no_time', action='store_true',
                        help='do not use time embedding')
    parser.add_argument('--no_pos', action='store_true',
                        help='do not use position embedding')
    #################Parameters of SparseIM###################
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of analyse epochs')
    parser.add_argument('--Lagrangian_allowance', type=float, default=0.0, help='allowance of Lagrangian optimization')
    parser.add_argument('--penalty_scaling', type=float, default=0.5, help='the scaling rate of edgenumber penalty')
    parser.add_argument('--sparsifier_lr', type=float, default=3e-4,help='the learning rate of Lagrangian optimizer')
    parser.add_argument('--stop_proportion', type=float, default=0.3, help='the proportion of remain edges that our model need to meet ')
    parser.add_argument('--use_mlp', type=bool, default=False,help='use mlp as interceptor ')
    parser.add_argument('--e2e_lr', type=float, default=3e-4, help='the learning rate of Lagrangian optimizer')





    try:
        args = parser.parse_args()
        assert args.n_workers == 0, "n_worker must be 0, etherwise dataloader will cause bug and results very bad performance (this bug will be fixed soon)"
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if args.dataset == 'alipay':
            args.feat_dim = 101
            args.lr *= 5
            args.bs *= 5
        elif args.dataset =='wikipedia' or args.dataset =='reddit':
            args.edge_feat_dim = 172
            args.node_feat_dim = 0
        elif args.dataset == 'mooc':
            args.edge_feat_dim = 4
            args.node_feat_dim = 0
        else:
            args.edge_feat_dim = 0
            args.node_feat_dim = 0
        args.no_time = True
        # if len(args.fanouts)!=args.n_layers:
        #     args.fanouts=[args.n_degrees for _ in range(args.n_layers)]
        # args.no_pos = True


    except:
        parser.print_help()
        sys.exit(0)

    return args

