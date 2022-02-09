import dgl
import torch
import random
import dgl.backend as F
import numpy as np
import copy

class TemporalEdgeCollator(dgl.dataloading.EdgeCollator):

    def __init__(self,args,g,eids,block_sampler,g_sampling=None,exclude=None,
                reverse_eids=None,reverse_etypes=None,negative_sampler=None):
        super(TemporalEdgeCollator,self).__init__(g,eids,block_sampler,
                                                 g_sampling,exclude,reverse_eids,reverse_etypes,negative_sampler)
        self.args=args
    def collate(self,items):
        current_ts=self.g.edata['timestamp'][items[0]]     #only sample edges before current timestamp
        self.block_sampler.ts=current_ts
        neg_pair_graph=None
        if self.negative_sampler is None:
            input_nodes,pair_graph,blocks=self._collate(items)
        else:
            input_nodes, pair_graph, neg_pair_graph, blocks = self._collate_with_negative_sampling(items)

        if self.args.n_layers>1:
            self.block_sampler.frontiers[0].add_edges(*self.block_sampler.frontiers[1].edges())
        frontier=dgl.reverse(self.block_sampler.frontiers[0])

        # if self.args.neg_list_num > 1:
        #     return input_nodes, pair_graph, neg_glist, blocks, frontier, current_ts
        # else:
        #     return input_nodes, pair_graph, neg_pair_graph, blocks, frontier, current_ts
        return input_nodes, pair_graph, neg_pair_graph, blocks, frontier, current_ts

class MultiLayerTemporalNeighborSampler(dgl.dataloading.BlockSampler):


    def __init__(self, args,fanouts, replace=False, return_eids=False):
        super().__init__(args.n_layers, return_eids)

        self.fanouts = fanouts
        self.replace = replace
        self.ts = 0
        self.args = args
        self.frontiers = [None for _ in range(args.n_layers)]

    def sample_frontier(self, block_id, g, seed_nodes):
        if self.fanouts is not None:
            fanout = self.fanouts[block_id]
        else:
            fanout = None
        # List of neighbors to sample per edge type for each GNN layer, starting from the first layer.
        g = dgl.in_subgraph(g, seed_nodes)
        g.remove_edges(torch.where(g.edata['timestamp'] > self.ts)[0])

        if fanout is None:
            frontier = g
        else:
            fanout = self.fanouts[block_id]
            if self.args.uniform:
                frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout)
            else:
                frontier = dgl.sampling.select_topk(g, fanout, 'timestamp', seed_nodes)
        self.frontiers[block_id] = frontier
        return frontier

def dataloader(args,g):
    TRAIN_SPLIT = 0.7
    VALID_SPLIT = 0.85
    # Pre-process data, mask new node in test set from original graph
    origin_num_edges = g.num_edges()
    origin_num_nodes=g.num_nodes()
    trainval_div = int(VALID_SPLIT * origin_num_edges)


    # Select new node from test set and remove them from entire graph
    test_split_ts = g.edata['timestamp'][trainval_div]
    test_nodes = torch.cat([g.edges()[0][trainval_div:], g.edges()[
                                                                1][trainval_div:]]).unique().numpy()
    test_new_nodes = np.random.choice(
        test_nodes, int(0.1 * len(test_nodes)), replace=False)

    in_subg = dgl.in_subgraph(g, test_new_nodes)
    out_subg = dgl.out_subgraph(g, test_new_nodes)
    # Remove edge who happen before the test set to prevent from learning the connection info
    new_node_in_eid_delete = in_subg.edata[dgl.EID][in_subg.edata['timestamp'] < test_split_ts]
    new_node_out_eid_delete = out_subg.edata[dgl.EID][out_subg.edata['timestamp'] < test_split_ts]
    new_node_eid_delete = torch.cat(
        [new_node_in_eid_delete, new_node_out_eid_delete]).unique()

    graph_new_node = copy.deepcopy(g)
    # relative order preseved
    graph_new_node.remove_edges(new_node_eid_delete)

    # Now for no new node graph, all edge id need to be removed
    in_eid_delete = in_subg.edata[dgl.EID]
    out_eid_delete = out_subg.edata[dgl.EID]
    eid_delete = torch.cat([in_eid_delete, out_eid_delete]).unique()

    graph_no_new_node = copy.deepcopy(g)
    graph_no_new_node.remove_edges(eid_delete)

    # graph_no_new_node and graph_new_node should have same set of nid

    train_eid = torch.arange(int(TRAIN_SPLIT*graph_no_new_node.num_edges()))
    val_eid = torch.arange(int(
        TRAIN_SPLIT*graph_no_new_node.num_edges()), trainval_div-new_node_eid_delete.size(0))
    test_eid = torch.arange(
        trainval_div-new_node_eid_delete.size(0), graph_no_new_node.num_edges())
    test_new_node_eid = torch.arange(
        trainval_div - new_node_eid_delete.size(0), graph_new_node.num_edges())

    # reverse_eids = torch.cat([torch.arange(origin_num_edges, 2 * origin_num_edges), torch.arange(0, origin_num_edges)])
    exclude, reverse_eids = None, None

    negative_sampler = dgl.dataloading.negative_sampler.Uniform(args.neg_list_num)
    fanouts = [args.n_degrees for _ in range(args.n_layers)] if args.n_degrees!=-1 else None
    train_sampler = MultiLayerTemporalNeighborSampler(args, fanouts, return_eids=True)
    val_sampler = MultiLayerTemporalNeighborSampler(args, fanouts, return_eids=True)


    train_collator = TemporalEdgeCollator(args,graph_no_new_node, train_eid, train_sampler, exclude=exclude, reverse_eids=reverse_eids,
                                          negative_sampler=negative_sampler)

    train_loader = torch.utils.data.DataLoader(
        train_collator.dataset, collate_fn=train_collator.collate,
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers)
    val_collator = TemporalEdgeCollator(args,graph_no_new_node, val_eid, val_sampler, exclude=exclude, reverse_eids=reverse_eids,
                                        negative_sampler=negative_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_collator.dataset, collate_fn=val_collator.collate,
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers)
    test_collator = TemporalEdgeCollator(args,graph_no_new_node, test_eid, val_sampler, exclude=exclude, reverse_eids=reverse_eids,
                                         negative_sampler=negative_sampler)
    test_loader = torch.utils.data.DataLoader(
        test_collator.dataset, collate_fn=test_collator.collate,
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers)

    inductive_test_collator = TemporalEdgeCollator(args, graph_new_node, test_new_node_eid, val_sampler, exclude=exclude,
                                         reverse_eids=reverse_eids,
                                         negative_sampler=negative_sampler)
    inductive_test_loader = torch.utils.data.DataLoader(
        inductive_test_collator.dataset, collate_fn=inductive_test_collator.collate,
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers)

    return train_loader, val_loader, test_loader, val_eid.shape[0], test_eid.shape[0],inductive_test_loader