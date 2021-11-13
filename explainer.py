import torch
from torch.nn.parameter import Parameter
import numpy as np
from torch import sigmoid
import math
import random

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
        clipped_s = self.clip(s)    #使得s保持在[0,1]之间

        if True:
            hard_concrete = (clipped_s > 0.5).float()
            clipped_s = clipped_s + (hard_concrete - clipped_s).detach()

        return clipped_s, penalty

    def clip(self, x, min_val=0, max_val=1):
        return x.clamp(min_val, max_val)

class Squeezer(torch.nn.Module):
    def forward(self, x):
        return x.squeeze(dim=-1)

class GraphMaskProbe(torch.nn.Module):
    def __init__(self,vec_dim,hid_dim):
        torch.nn.Module.__init__(self)

        self.gate=torch.nn.Sequential(
            torch.nn.Linear(vec_dim,hid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim,1),
            Squeezer()
        )
        torch.nn.init.xavier_normal_(self.gate[0].weight)
        torch.nn.init.xavier_normal_(self.gate[2].weight)
        self.HCD=HardConcrete()
        baseline=torch.FloatTensor(vec_dim) # 应该是用来代替message的b
        stdv = 1. / math.sqrt(vec_dim)
        baseline.uniform_(-stdv, stdv)
        baseline=torch.nn.Parameter(baseline,requires_grad=True)
        self.baseline=baseline
    def forward(self,input_info,training=True,shuffle_rate=0):
        gate_out=self.gate(input_info)
        z_uv,penalty=self.HCD(gate_out,training=training)
        if training and shuffle_rate>0:
            r=random.random()
            if r<shuffle_rate:
                random.shuffle(z_uv)
        out_info=z_uv.unsqueeze(dim=2)*input_info+(1-z_uv).unsqueeze(dim=2)*self.baseline
        return out_info,penalty,torch.sum(z_uv),z_uv.numel(),z_uv,self.baseline

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
