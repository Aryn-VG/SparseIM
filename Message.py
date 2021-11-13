import torch.nn as nn
import torch
import dgl.function as fn


class MLPMessageFunction(nn.Module):
    def __init__(self, raw_message_dim, message_dim):
        super(MLPMessageFunction, self).__init__()
        self.mlp = self.layers = nn.Sequential(nn.Linear(raw_message_dim, raw_message_dim // 2), nn.ReLU(),
                                               nn.Linear(raw_message_dim // 2, message_dim))

    def compute_message(self, raw_messages):
        messages = self.mlp(raw_messages)
        return messages

class Message_computer():
    def __init__(self,time_encoder,message_func):
        super(Message_computer,self).__init__()

        self.time_encoder = time_encoder
        self.message_func=message_func

    # def update_time(self,edge):
    #     return {'ts':edge.data['timestamp']}

    def message_computing(self,edge):
        raw_message=edge.dst['raw_message']
        message=self.message_func.compute_message(raw_message)
        # print(edge.src['memory'].size())
        # print(edge.data['feat'].size())
        return {'m':message}

    def message_aggregating(self,g):

        g.update_all(self.message_computing, fn.mean('m', 'message'))
        return g

class Raw_Message_Updater():

    def __init__(self,time_encoder):
        super(Raw_Message_Updater,self).__init__()
        self.time_encoder = time_encoder

    def update_raw_message_mf(self,edge):
        if self.time_encoder:
            te_m = edge.dst['ts'] - edge.dst['last_update']
            te_m = self.time_encoder(te_m)
            raw_message_new = torch.cat([edge.src['memory'], edge.dst['memory'], edge.data['feat'], te_m], dim=1)
        else:
            raw_message_new = torch.cat([edge.src['memory'], edge.dst['memory'], edge.data['feat']], dim=1)
        return {'rmn':raw_message_new}
    def update_raw_message(self,g):

        g.update_all(self.update_raw_message_mf, fn.mean('rmn', 'raw_message'))
        return g