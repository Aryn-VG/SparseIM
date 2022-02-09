import torch.nn as nn
import torch
from torch import sigmoid
from torch.nn.parameter import Parameter
import numpy as np
from time_encode import TimeEncode
import math

class HardConcrete(torch.nn.Module):

    def __init__(self, beta=1 / 3, gamma=-0.2, zeta=1.0, fix_temp=True, loc_bias=3):
        super(HardConcrete, self).__init__()

        self.temp = beta if fix_temp else Parameter(torch.zeros(1).fill_(beta))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = np.math.log(-gamma / zeta)
        self.loc_bias = loc_bias

    def forward(self, input_element, summarize_penalty=True,training=True):
        input_element = input_element + self.loc_bias

        if training:
            u = torch.empty_like(input_element).uniform_(1e-6, 1.0-1e-6)

            s = sigmoid((torch.log(u) - torch.log(1 - u) + input_element) / self.temp)

            penalty = sigmoid(input_element - self.temp * self.gamma_zeta_ratio)
            penalty = penalty
        else:
            s = sigmoid(input_element)
            penalty = torch.zeros_like(input_element)
        if summarize_penalty:
            penalty = penalty.mean()
        #print(training,penalty)
        s = s * (self.zeta - self.gamma) + self.gamma
        clipped_s = self.clip(s)

        if True:
            hard_concrete = (clipped_s > 0.5).float()
            clipped_s = clipped_s + (hard_concrete - clipped_s).detach()

        return clipped_s, penalty

    def clip(self, x, min_val=0, max_val=1):
        return x.clamp(min_val, max_val)

class Squeezer(torch.nn.Module):
    def forward(self, x):
        return x.squeeze(dim=-1)

class SparseIM(nn.Module):
    def __init__(self,args):
        super(SparseIM,self).__init__()
        # self.message_dim=args.edge_feat_dim+args.memory_dim
        self.args=args
        if args.use_mlp:
            self.interceptor = torch.nn.Sequential(
                torch.nn.Linear(args.edge_feat_dim+args.memory_dim,args.memory_dim),
                torch.nn.LayerNorm(args.memory_dim),
                torch.nn.Linear(args.memory_dim, 1),
                Squeezer()
            )
            torch.nn.init.xavier_normal_(self.interceptor[0].weight)
            torch.nn.init.xavier_normal_(self.interceptor[1].weight)
        else:
            self.interceptor=torch.nn.Sequential(
                torch.nn.GRU(args.edge_feat_dim+args.memory_dim,args.memory_dim,batch_first=True),
                torch.nn.LayerNorm(args.memory_dim),
                torch.nn.Dropout(p=args.dropout, inplace=True),
                torch.nn.Linear(args.memory_dim,50),
                torch.nn.ReLU(),
                torch.nn.Linear(50, 1),
                Squeezer()
            )
            torch.nn.init.xavier_normal_(self.interceptor[3].weight)
            torch.nn.init.xavier_normal_(self.interceptor[5].weight)

        self.HCD = HardConcrete()
        manipulator = torch.FloatTensor(args.edge_feat_dim)
        stdv = 1. / math.sqrt(args.edge_feat_dim)
        manipulator.uniform_(-stdv, stdv)
        manipulator = torch.nn.Parameter(manipulator, requires_grad=True)
        self.manipulator = manipulator
        self.n_layers=args.n_layers
        self.device=args.device
        self.time_encoder=TimeEncode(args.edge_feat_dim)

    def send_message(self,edges):
        t_code = self.time_encoder(edges.data['timestamp'] - edges.dst['last_update'])
        messages=edges.data['feat']+t_code
        messages=torch.cat([messages,edges.dst['memory']],dim=1)
        return {'messages': messages,'ts':edges.data['timestamp'],'eid':edges.data['eid'],'src_d':edges.src['d']}

    def message_agg(self, nodes):
        messages = nodes.mailbox['messages']#[nodes_batch_size,num_neighbors,feat]
        ts = nodes.mailbox['ts']#[nodes_batch_size,num_neighbors]
        eids=nodes.mailbox['eid']#[nodes_batch_size,num_neighbors]


        if self.args.use_mlp:
            final_mem=self.interceptor[0](messages)
            gate_out = self.interceptor[1:](final_mem)  # [node_batch,num_neighbors]
            final_mem=torch.mean(final_mem,dim=1)
        else:
            ordered_messages = messages[0, ts.sort(dim=1)[1]]  # [nodes_batch_size,num_neighbors,feat]
            pre_mem = nodes.data['memory'].unsqueeze(dim=0)  # [1,nodes_batch_size,feat_dim]
            cur_mem,final_mem = self.interceptor[0](ordered_messages, pre_mem)  # GRU (input,h_0)=[node_batch,num_neighbors,mem_dim]
            final_mem=final_mem.squeeze(dim=0)
            gate_out = self.interceptor[1:](cur_mem)#[node_batch,num_neighbors]
        s_ij, penalty = self.HCD(gate_out, training=self.training)  # [batch,num_neighbors]]
        if self.training:
            self.penalty+=torch.mean(penalty/ts.size(1))
        self.remain_edge_batch += torch.sum(s_ij)
        self.total_edge_batch+=s_ij.numel()
        self.s_ij[eids.view(-1)]=s_ij.view(-1)# [batch,num_neighbors]


        return {'memory':final_mem}

    def forward(self, blocks, training=True):
        self.training = training
        self.penalty = 0
        self.remain_edge_batch = 0
        self.total_edge_batch = 0
        self.s_ij=torch.tensor([-1]*blocks[-1].number_of_edges()).float().to(self.device)

        blocks[-1].edata['eid']=torch.linspace(0,blocks[-1].number_of_edges()-1 ,blocks[-1].number_of_edges()).long().to(self.device)
        blocks[-1].srcdata['d']=blocks[-1].out_degrees()+1
        blocks[-1].update_all(self.send_message, self.message_agg)

        return blocks, self.penalty, self.remain_edge_batch, self.total_edge_batch, self.s_ij

class LagrangianOptimization:

    min_alpha = None
    max_alpha = None
    device = None
    original_optimizer = None
    batch_size_multiplier = None
    update_counter = 0

    def __init__(self, original_optimizer, device, init_alpha=0.55, min_alpha=-2, max_alpha=30, alpha_optimizer_lr=1e-2, batch_size_multiplier=None):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.device = device
        self.batch_size_multiplier = batch_size_multiplier
        self.update_counter = 0

        self.alpha = torch.tensor(init_alpha, device=device, requires_grad=True)
        self.optimizer_alpha = torch.optim.RMSprop([self.alpha], lr=alpha_optimizer_lr, centered=True)
        self.original_optimizer = original_optimizer

    def update(self, f, g):
        """
        L(x, lambda) = f(x) + lambda g(x)

        :param f_function:
        :param g_function:
        :return:
        """

        if self.batch_size_multiplier is not None and self.batch_size_multiplier > 1:
            if self.update_counter % self.batch_size_multiplier == 0:
                self.original_optimizer.zero_grad()
                self.optimizer_alpha.zero_grad()

            self.update_counter += 1
        else:
            self.original_optimizer.zero_grad()
            self.optimizer_alpha.zero_grad()

        loss = f + torch.nn.functional.softplus(self.alpha) * g
        loss.backward()


        if self.batch_size_multiplier is not None and self.batch_size_multiplier > 1:
            if self.update_counter % self.batch_size_multiplier == 0:
                self.original_optimizer.step()
                self.alpha.grad *= -1
                self.optimizer_alpha.step()
        else:
            self.original_optimizer.step()
            self.alpha.grad *= -1
            self.optimizer_alpha.step()

        if self.alpha.item() < -2:
            self.alpha.data = torch.full_like(self.alpha.data, -2)
        elif self.alpha.item() > 30:
            self.alpha.data = torch.full_like(self.alpha.data, 30)


