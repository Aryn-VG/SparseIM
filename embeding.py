import torch
import numpy as np

class ScaledDotProductAttention(torch.nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None,time_bias=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn)  # [n * b, l_q, l_k]
        if time_bias is not None:
            attn=attn+time_bias
        attn = self.dropout(attn)

        output = torch.bmm(attn, v) # [n*b,len_q,feat]

        return output, attn


class MultiHeadAttention(torch.nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, embed_dim,kdim,vdim,num_heads,dropout=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.kdim = kdim
        self.vdim = vdim
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.w_qs = torch.nn.Linear(embed_dim, num_heads * kdim, bias=False)
        self.w_ks = torch.nn.Linear(kdim, num_heads * kdim, bias=False)
        self.w_vs = torch.nn.Linear(vdim, num_heads * vdim, bias=False)

        self.attention = ScaledDotProductAttention(temperature=np.power(kdim, 0.5), attn_dropout=dropout)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

        self.fc = torch.nn.Linear(num_heads * vdim, embed_dim)
        torch.nn.init.normal_(self.w_qs.weight)
        torch.nn.init.normal_(self.w_ks.weight)
        torch.nn.init.normal_(self.w_vs.weight)
        torch.nn.init.xavier_normal_(self.fc.weight)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        '''
        q:[1,batchsize,featdim]
        k,v:[n_neighbors,batchsize,featdim]
        '''
        d_k, d_v, n_head = self.kdim, self.vdim, self.num_heads
        q,k,v = q.permute([1, 0, 2]),k.permute([1, 0, 2]),v.permute([1, 0, 2])
        # k,v:[batchsize,n_neighbors,featdim]
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, -1, d_k)
        k = self.w_ks(k).view(sz_b, len_k, -1, d_k)
        v = self.w_vs(v).view(sz_b, len_v, -1, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        #output:[n*b,len_q,feat],attn:[n*b,len_q,len_k]

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        # output = self.layer_norm(output)
        output=output.permute(1,0,2)
        return output, attn


class ATTN(torch.nn.Module):
    def __init__(self,args,time_encoder):
        super().__init__()
        self.n_layers = args.n_layers
        self.attnlayers = torch.nn.ModuleList()
        self.mergelayers = torch.nn.ModuleList()
        self.edge_feat_dim=args.edge_feat_dim
        self.n_heads = args.n_heads
        self.time_dim = args.time_dim
        self.emb_dim = args.emb_dim
        self.dropout = args.dropout
        self.args=args
        self.time_encoder=time_encoder
        self.device='cuda:0' if torch.cuda.is_available() else 'cpu'
        self.use_no_te=args.use_no_te
        self.use_expire_edge=args.use_expire_edge

        self.query_dim = self.emb_dim + self.time_dim  # 仅仅只是维
        self.key_dim = self.emb_dim + self.time_dim + self.edge_feat_dim
        self.alpha=args.alpha

        for i in range(0, self.n_layers):
            # self.attnlayers.append(torch.nn.MultiheadAttention(embed_dim=self.query_dim,
            #                                            kdim=self.key_dim,
            #                                            vdim=self.key_dim,
            #                                            num_heads=self.n_heads,
            #                                            dropout=self.dropout).to(self.device))
            self.attnlayers.append(MultiHeadAttention(embed_dim=self.query_dim,
                                                         kdim=self.key_dim,
                                                         vdim=self.key_dim,
                                                         num_heads=self.n_heads,
                                                         dropout=self.dropout).to(self.device))
            self.mergelayers.append(MergeLayer(self.query_dim, self.emb_dim, self.emb_dim, self.emb_dim).to(self.device))
    def C_compute(self,edges):
        #te_C=self.time_encoder(edges.data['timestamp']-edges.dst['last_update'])
        if not self.use_no_te:
            te_C = self.time_encoder(edges.data['timestamp'] - edges.src['last_update'])
        else:
            pass

        if not self.use_no_te:
            edge_info = torch.cat([edges.data['feat'], te_C], dim=1)
        else:
            edge_info=edges.data['feat']
        if self.memory_masked:
            C = torch.cat([edges.src['masked_h'], edge_info], dim=1)
        else:
            C = torch.cat([edges.src['h'], edge_info], dim=1)
        return {'C': C}

    def h_compute(self,nodes):
        C_ori=nodes.mailbox['C']
        C_ori=C_ori.permute([1,0,2])#[nodes_batch_size,num_neighbors,feat]->[num_neighbors,nodes_batch_size,feat]
        C=C_ori
        ## training phase
        if self.analyser is not None and self.inference_baseline is None:
            C,penalty,remain_edge_batch,total_edge_batch,train_mask,baseline=self.analyser(C,training=self.probe_train,shuffle_rate=0)
            self.penalty+=penalty*10/C.size(0)
            self.remain_edge_batch+=remain_edge_batch
            self.total_edge_batch+=total_edge_batch
            self.baseline=baseline
            if self.args.drop_training_edge and torch.sum(train_mask==0)>0:
                C[train_mask == 0] = torch.zeros((torch.sum(train_mask == 0).int(), C.size(-1)), device=self.device)


        ## inference phase
        if self.inference_baseline is not None:
            C,penalty,remain_edge_batch,total_edge_batch,mask,_=self.analyser(C,training=self.probe_train)
            self.penalty += penalty
            self.remain_edge_batch += remain_edge_batch
            self.total_edge_batch += total_edge_batch
            if self.args.random_drop>0:
                random_mask=torch.rand(mask.size(),device=self.device)
                random_mask=(random_mask<self.args.random_drop).int()
                C[random_mask == 0] = torch.zeros((torch.sum(random_mask == 0).int(), C.size(-1)), device=self.device)
            elif torch.sum(mask==0)>0:
                C[mask == 0] = torch.zeros((torch.sum(mask == 0).int(), C.size(-1)), device=self.device)
                if self.args.im_message:
                    global_vec=self.inference_baseline.view(1,1,-1).repeat(1,nodes.batch_size(),1)
                    C=torch.cat([C,global_vec],dim=0)
            elif self.args.im_message:
                global_vec = self.inference_baseline.view(1, 1, -1).repeat(1, nodes.batch_size(), 1)
                C = torch.cat([C, global_vec], dim=0)
            else:pass
        key = C.to(self.device)#[num_neighbors,nodes_batch_size,feat]
        if not self.use_no_te:
            te_q=self.time_encoder(torch.zeros(nodes.batch_size()).to(self.device))
            if self.memory_masked:
                query = torch.cat([nodes.data['masked_h'], te_q], dim=1).unsqueeze(dim=0)  # [1,nodes_batch_size,feat]
            else:
                query = torch.cat([nodes.data['h'], te_q],dim=1).unsqueeze(dim=0)#[1,nodes_batch_size,feat]
        else:
            if self.memory_masked:
                query = nodes.data['masked_h'].unsqueeze(dim=0)
            else:
                query=nodes.data['h'].unsqueeze(dim=0)
        h_before, _ = self.attnlayers[self.l](query, key, key)
        h_before=h_before.squeeze(0)
        if self.memory_masked:
            h = self.mergelayers[self.l](nodes.data['masked_h'], h_before)
        else:
            h= self.mergelayers[self.l](nodes.data['h'], h_before)
        if self.memory_masked:
            return {'masked_h':h}
        else:
            return {'h':h}

    def forward(self, blocks,analyser=None,probe_train=True,memory_masked=False,inference_baseline=None,local_probe=None): # x是h
        self.analyser = analyser
        self.memory_masked=memory_masked
        self.inference_baseline=inference_baseline
        if analyser is not None:
            self.penalty=0
            self.remain_edge_batch=0
            self.total_edge_batch=0
            self.probe_train=probe_train
        for l in range(self.n_layers):
            self.l=l
            blocks[l].update_all(self.C_compute,self.h_compute)
            if l!=self.n_layers-1:#如果不是最后一层，那么上一层的dst数据要同步到下一层的src
                if memory_masked:
                    blocks[l + 1].srcdata['masked_h'] = blocks[l].dstdata['masked_h']
                else:
                    blocks[l + 1].srcdata['h'] = blocks[l].dstdata['h']
        if analyser is not None:
            return blocks,self.penalty,self.remain_edge_batch,self.total_edge_batch,self.baseline
        else:
            return blocks

class MergeLayer(torch.nn.Module):
    '''(dim1+dim2)->dim3->dim4'''

    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)

