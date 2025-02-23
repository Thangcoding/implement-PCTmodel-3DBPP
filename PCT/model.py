import torch 
from torch import nn
import torch.nn.functional as F 
import numpy as np 
from attention_model import AttentionModel
from envs import Env
import math 
from visual import visualize

class FullGlimpse(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim, bias=False)
        self.v = nn.Parameter(torch.Tensor(out_dim))

    def reset_parameters(self):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.dense.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.v, -bound, bound)

    def forward(self, ref):
        # Attention
        encoded_ref = self.dense(ref)
        scores = torch.sum(self.v * torch.tanh(encoded_ref), dim=-1)

        attention = F.softmax(scores, dim=-1)
        glimpse = ref * attention.unsqueeze(-1)
        glimpse = torch.sum(glimpse, dim=1)
        return glimpse

class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.glimpse = FullGlimpse(in_dim=input_dim, out_dim=input_dim)
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        x = x.unsqueeze(0)  # Thêm chiều batch
        glimpse = self.glimpse(x)
        output = self.fc(glimpse)
        return output

class PCT_Policy(nn.Module):
    def __init__(self,embed_dim, hidden,env,num_heads = 1):
        super(PCT_Policy,self).__init__()
        # actor 
        self.Actor = AttentionModel(embed_dim= embed_dim, 
                                    feed_forward_hidden = hidden, 
                                    num_heads = 1)
    
        # critic 
        self.critic = Critic(input_dim = embed_dim)

        # environment 
        self.env = env 


    def forward(self,item,bin_size):
        # update state 
        self.env.update_state(item)
        self.env.valid_leaf_node()
        self.env.update_graph()


        if torch.max(self.env.graph['isvalid_leaf_node']).item() != 0:
            graph_high_features,index_leaf_selected,prob = self.Actor(self.env.graph)

            if prob.item() != 0:

                curr_state = torch.concat((graph_high_features['internal_node'],graph_high_features['leaf_node'],graph_high_features['curr_item'].unsqueeze(0)), dim = 0)
                critic_val_curr = self.critic(curr_state)

            
                # add new item packed in internal node 
                internal = torch.concat((graph_high_features['internal_node'],graph_high_features['curr_item'].unsqueeze(0)), dim = 0)

                # remove packed place leaf node 
                leaf = torch.concat((graph_high_features['leaf_node'][:index_leaf_selected,:],graph_high_features['leaf_node'][index_leaf_selected + 1:,:]), dim = 0)
                next_state = torch.concat((internal,leaf), dim = 0)
                critic_val_next = self.critic(next_state)

                # reward
                curr_item = self.env.graph['curr_item']
                reward = curr_item[:,3]*curr_item[:,4]*curr_item[:,5]

                # packed item and update graph 
                self.env.packed(index_leaf_node = index_leaf_selected)
                self.env.update_graph()
            else:
                critic_val_curr = 0
                critic_val_next = 0
                reward = 0
        
            return critic_val_curr,critic_val_next, prob,reward.item()
        return None, None , None , None 


if __name__ == '__main__':
    pass 
