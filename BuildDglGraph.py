import dgl
import torch
import argparse
import pandas as pd
import numpy as np
from dgl.data.utils import save_graphs

import dgl.function as fn

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('-d', '--data', type=str, choices=["wikipedia", "reddit", "alipay"], help='Dataset name (eg. wikipedia or reddit)',
                        default='reddit')
parser.add_argument('--new_node_count', action='store_true',
                        help='count how many nodes are not in training set')    
args = parser.parse_args()
args.new_node_count = True

graph_df = pd.read_csv('./data/{}.csv'.format(args.data))
edge_features = np.load('./data/{}.npy'.format(args.data))
nfeat_dim = edge_features.shape[1]


src = torch.tensor(graph_df.u.values)
dst = torch.tensor(graph_df.i.values)
label = torch.tensor(graph_df.label.values, dtype=torch.float32)
timestamp = torch.tensor(graph_df.ts.values, dtype=torch.float32)
edge_feat = torch.tensor(edge_features[1:], dtype=torch.float32)

# g = dgl.graph((torch.cat([src,dst]), torch.cat([dst,src])))
g = dgl.graph((src,dst))
len_event = src.shape[0]

g.edata['label'] = label.squeeze()
g.edata['timestamp'] = timestamp.squeeze()
g.edata['feat'] = edge_feat.squeeze()

print(g)

if args.new_node_count:
    origin_num_edges = g.num_edges()
    train_eid = torch.arange(0, int(0.7 * origin_num_edges)) # 训练集中的的eid
    un_train_eid = torch.arange(int(0.7 * origin_num_edges), origin_num_edges) # 非训练集中的eid

    train_g = dgl.graph(g.find_edges(train_eid)) # 训练集中所有的边组成的子图
    val_n_test_g = dgl.compact_graphs(dgl.graph(g.find_edges(un_train_eid))) # 消除0度的节点

    print(f'total nodes: {g.num_nodes()}, training nodes: {train_g.num_nodes()}, val_n_test nodes: {val_n_test_g.num_nodes()}')
    old_nodes = val_n_test_g.num_nodes()-g.num_nodes()+train_g.num_nodes()
    print(f'old nodes in val_n_test: {old_nodes} ({round((old_nodes)*100/val_n_test_g.num_nodes(),4)}%)')
    new_nodes = g.num_nodes()-train_g.num_nodes()
    print(f'new nodes in val_n_test: {new_nodes} ({round((new_nodes)*100/val_n_test_g.num_nodes(),4)}%)')
save_graphs("./data/{}.dgl".format(args.data), g)