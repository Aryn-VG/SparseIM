import torch.nn as nn
import torch





class GRUMemoryUpdater(nn.Module):
    def __init__(self,args):
        super(GRUMemoryUpdater,self).__init__()
        self.memory_updater = nn.GRUCell(input_size=args.message_dim+args.memory_dim,
                                         hidden_size=args.memory_dim)
    def forward(self,x):
        y=self.memory_updater(x)
        return y

class Memory_updater():
    '''用来计算&更新memory'''
    def __init__(self,memory_computer):
        super(Memory_updater,self).__init__()
        self.memory_computer = memory_computer

    def update_memory(self, g):
        g.apply_nodes(lambda nodes:{'memory':self.memory_computer(torch.cat([nodes.data['memory'],nodes.data['message']],dim=1))})
        return g