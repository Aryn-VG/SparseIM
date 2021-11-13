import dgl
import torch
import random
import dgl.backend as F

class TemporalEdgeCollator(dgl.dataloading.EdgeCollator):
    '''
    整理边
    '''
    def __init__(self,args,g,eids,block_sampler,g_sampling=None,exclude=None,
                reverse_eids=None,reverse_etypes=None,negative_sampler=None):
        super(TemporalEdgeCollator,self).__init__(g,eids,block_sampler,
                                                 g_sampling,exclude,reverse_eids,reverse_etypes,negative_sampler)
        self.args=args
    def collate(self,items):    #items是什么?目标节点吗
        current_ts=self.g.edata['timestamp'][items[0]]     #only sample edges before current timestamp
        self.block_sampler.ts=current_ts    #当前block的最后一个ts，给下面的MultiLayerTemporalNeighborSampler定义的
        neg_pair_graph=None
        if self.negative_sampler is None:
            input_nodes,pair_graph,blocks=self._collate(items)
        else:
            input_nodes, pair_graph, neg_pair_graph, blocks = self._collate_with_negative_sampling(items)

        if self.args.n_layers>1:
            self.block_sampler.frontiers[0].add_edges(*self.block_sampler.frontiers[1].edges())#为什么要叠加？
        frontier=dgl.reverse(self.block_sampler.frontiers[0])#上行代码好像把边全部加到frontier[0]里面了
        #frontier：包含原图所有节点，但是只有在此层中有message passing的edge才被包含
        # if self.args.neg_list_num > 1:
        #     return input_nodes, pair_graph, neg_glist, blocks, frontier, current_ts
        # else:
        #     return input_nodes, pair_graph, neg_pair_graph, blocks, frontier, current_ts
        return input_nodes, pair_graph, neg_pair_graph, blocks, frontier, current_ts

class MultiLayerTemporalNeighborSampler(dgl.dataloading.BlockSampler):
    '''
    对边采样，返回block的frontier。
    '''

    def __init__(self, args,fanouts, replace=False, return_eids=False):
        super().__init__(args.n_layers, return_eids)

        self.fanouts = fanouts  # fanout应该是每层要采样的节点个数
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
        g = dgl.in_subgraph(g, seed_nodes)  # 只包含seed_node的g（不考虑邻居吗？）
        g.remove_edges(torch.where(g.edata['timestamp'] > self.ts)[0])  # 去掉ts之后的edge

        # torch.where(条件，符合条件设置为；不符合条件设置为)
        if fanout is None:  # 不采样就取全图
            frontier = g
        else:
            fanout = self.fanouts[block_id]
            if self.args.uniform:  # 如果是uniform采样
                frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout)
            else:
                frontier = dgl.sampling.select_topk(g, fanout, 'timestamp', seed_nodes)  # 选timestamp最大的【fanout】个点
        self.frontiers[block_id] = frontier  # 每层blocks采样的点存在frontier里面
        return frontier

class InferenceNeighborSampler(dgl.dataloading.BlockSampler):
    '''
    对边采样，返回block的frontier。
    '''

    def __init__(self, args,fanouts, replace=False, return_eids=False,time_encoder=None,masker=None):
        super().__init__(args.n_layers, return_eids)

        self.fanouts = fanouts  # fanout应该是每层要采样的节点个数
        self.replace = replace
        self.ts = 0
        self.args = args
        self.frontiers = [None for _ in range(args.n_layers)]
        self.time_encoder=time_encoder
        self.masker=masker
        self.device='cuda:0' if torch.cuda.is_available() else 'cpu'

    def sample_frontier(self, block_id, g, seed_nodes):
        if self.fanouts is not None:
            fanout = self.fanouts[block_id]
        else:
            fanout=None
        # List of neighbors to sample per edge type for each GNN layer, starting from the first layer.
        g = dgl.in_subgraph(g, seed_nodes)  # 只包含seed_node的g（不考虑邻居吗？）
        g.remove_edges(torch.where(g.edata['timestamp'] > self.ts)[0])  # 去掉ts之后的edge
        if self.args.use_no_te:
            mask=self.masker(g.edata['feat'])[-1]
        else:
            te=self.time_encoder(g.edata['timestamp'].to(self.device))
            mask = self.masker(torch.cat([g.edata['feat'].to(self.device),te],dim=1))[4].cpu()
        if self.args.random_drop:
            random.shuffle(mask)
        g.remove_edges(torch.where(mask==0)[0])
        # torch.where(条件，符合条件设置为；不符合条件设置为)

        if fanout is None:  # 不采样就取全图
            frontier = g
        else:
            if self.args.uniform:  # 如果是uniform采样
                frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout)
            else:
                frontier = dgl.sampling.select_topk(g, fanout, 'timestamp', seed_nodes)  # 选timestamp最大的【fanout】个点
        self.frontiers[block_id] = frontier  # 每层blocks采样的点存在frontier里面
        return frontier



def dataloader(args,g,time_encoder=None,masker=None):
    origin_num_edges = g.num_edges()

    train_eid = torch.arange(0, int(0.7 * origin_num_edges))
    val_eid = torch.arange(int(0.7 * origin_num_edges), int(0.85 * origin_num_edges))
    test_eid = torch.arange(int(0.85 * origin_num_edges), origin_num_edges)

    # reverse_eids = torch.cat([torch.arange(origin_num_edges, 2 * origin_num_edges), torch.arange(0, origin_num_edges)])
    exclude, reverse_eids = None, None

    negative_sampler = dgl.dataloading.negative_sampler.Uniform(args.neg_list_num)
    fanouts = [args.n_degrees for _ in range(args.n_layers)]
    train_sampler = MultiLayerTemporalNeighborSampler(args, fanouts, return_eids=True)
    val_sampler = MultiLayerTemporalNeighborSampler(args, fanouts, return_eids=True)


    train_collator = TemporalEdgeCollator(args,g, train_eid, train_sampler, exclude=exclude, reverse_eids=reverse_eids,
                                          negative_sampler=negative_sampler)

    train_loader = torch.utils.data.DataLoader(
        train_collator.dataset, collate_fn=train_collator.collate,
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers)
    val_collator = TemporalEdgeCollator(args,g, val_eid, val_sampler, exclude=exclude, reverse_eids=reverse_eids,
                                        negative_sampler=negative_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_collator.dataset, collate_fn=val_collator.collate,
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers)
    test_collator = TemporalEdgeCollator(args,g, test_eid, val_sampler, exclude=exclude, reverse_eids=reverse_eids,
                                         negative_sampler=negative_sampler)
    test_loader = torch.utils.data.DataLoader(
        test_collator.dataset, collate_fn=test_collator.collate,
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers)

    if args.remove_masked_edge:
        inference_sampler=InferenceNeighborSampler(args,fanouts,return_eids=False,time_encoder=time_encoder,masker=masker)
        inference_collator = TemporalEdgeCollator(args, g, val_eid, inference_sampler, exclude=exclude,
                                                  reverse_eids=reverse_eids,
                                                  negative_sampler=negative_sampler)
        inference_loader = torch.utils.data.DataLoader(
            inference_collator.dataset, collate_fn=inference_collator.collate,
            batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers)

    else:
        inference_loader=None
    return train_loader, val_loader, test_loader, inference_loader, val_eid.shape[0], test_eid.shape[0]