import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np
import math 


class SkipConnection(nn.Module):
    def __init__(self,module): 
        super(SkipConnection,self).__init__() 
        self.module = module   
    def forward(self,input):   
        return self.module(input) + input  

class GATLayer(nn.Module):
    def __init__(self,input_dim,embed_dim):
        super(GATLayer, self).__init__()
        self.W_query = nn.Linear(in_features = input_dim, out_features = embed_dim, bias = False)
        self.W_key = nn.Linear(in_features = input_dim, out_features = embed_dim, bias = False)
        self.W_value = nn.Linear(in_features = input_dim, out_features = embed_dim , bias = False)

        self.W_out = nn.Linear(in_features = embed_dim, out_features = input_dim)

    def forward(self,input):
        num_nodes = input.size(0)
        attention_score = torch.zeros(num_nodes,num_nodes, device = input.device)

        Q, K, V = self.W_query(input), self.W_key(input), self.W_value(input)
        scale = K.size(1)**(0.5)
        scores = torch.mm(Q, K.transpose(0,1)) / scale 
        scores = F.softmax(scores, dim = -1) 
        output = torch.mm(scores,V) 

        output = self.W_out(output) 

        return output 

class GAT(nn.Module):
    def __init__(self, input_dim ,embed_dim ,num_heads = 1, feed_forward_hidden =128):
        super(GAT,self).__init__()
        self.attention = SkipConnection(GATLayer(
            input_dim= input_dim, embed_dim= embed_dim
        ))
        self.feed_forward = nn.Sequential(  nn.Linear(in_features= input_dim , out_features = feed_forward_hidden ),
                                            nn.ReLU(),
                                            nn.Linear(in_features = feed_forward_hidden , out_features = input_dim)
                                            )

    def forward(self,input):
        output = self.attention(input)

        output = self.feed_forward(output)

        return output

class Pointer(nn.Module):
    def __init__(self, input_dim: int, hidden : int):
        super(Pointer, self).__init__()
        self.W = nn.Linear(input_dim*2,hidden) 
        self.v = nn.Linear(hidden,1)

    def forward(self,leaf_node : torch.tensor,curr_item: torch.tensor,mask : torch.tensor = None) -> torch.tensor:
        curr_item = torch.flatten(curr_item)
        curr_item = curr_item.unsqueeze(0).repeat(leaf_node.size(0),1)
        leaf_node_concat = torch.concat((leaf_node,curr_item), dim = 1)

        W_out = self.W(leaf_node_concat)
        scores = self.v(W_out)
        scores = scores.squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        prob = F.softmax(scores,dim = -1)
        return prob


# main attention model 
class AttentionModel(nn.Module):
    def __init__(self, embed_dim, feed_forward_hidden,num_heads = 1):
        super(AttentionModel,self).__init__()
        # embedding 
        self.embedding_internal_node = nn.Sequential(nn.Linear(in_features=6, out_features =32, bias= False),
                                                        nn.Linear(in_features=32,out_features = embed_dim, bias= False))

        self.embedding_leaf_node = nn.Sequential(nn.Linear(in_features=6, out_features = 32, bias= False),
                                                    nn.Linear(in_features=32, out_features = embed_dim, bias= False))

        self.embedding_curr_box = nn.Sequential(nn.Linear(in_features =6, out_features =32),
                                                    nn.Linear(in_features=32, out_features = embed_dim))
                        
        # graph attention model 
        self.GAT_model = GAT(input_dim = embed_dim,embed_dim=embed_dim*2, feed_forward_hidden= feed_forward_hidden)
        self.pointer = Pointer(input_dim = embed_dim, hidden = feed_forward_hidden)
    
    def leaf_node_selection(self,graph_high_features,graph):
        # compute the probability for selecting leaf nodes 
        leaf_nodes_high_feature = graph_high_features['leaf_node']
        curr_item_high_feature = graph_high_features['curr_item']
        mask = graph['isvalid_leaf_node']

        probs = self.pointer(leaf_nodes_high_feature, curr_item_high_feature, mask)
    
        leaf_node_selected = probs.argmax()

        prob = probs[:,leaf_node_selected]

        return prob,leaf_node_selected 

    def forward(self,graph):

        internal_node = graph['internal_node']
        leaf_node = graph['leaf_node']
        curr_item = graph['curr_item']
        num_internal = internal_node.size(0)
        num_leaf = leaf_node.size(0)

        embedding_internal_node = self.embedding_internal_node(internal_node)
        embedding_leaf_node = self.embedding_leaf_node(leaf_node)
        embedding_curr_item = self.embedding_curr_box(curr_item)


        graph_concat = torch.concat([embedding_internal_node,embedding_leaf_node,embedding_curr_item], dim = 0)

        output = self.GAT_model(graph_concat)

        graph_high_features = {'internal_node': None, 'leaf_node': None , 'curr_item': None}
        graph_high_features['internal_node'] = output[:num_internal,:]
        graph_high_features['leaf_node'] = output[num_internal: num_leaf + num_internal,:]
        graph_high_features['curr_item'] = output[-1,:]


        prob, index_leaf_selected = self.leaf_node_selection(graph_high_features,graph)


        return graph_high_features,index_leaf_selected,prob

if __name__ == '__main__':

    input = {'internal_node':torch.tensor([[1,2,3,4,5,6],[2,3,2,6,7,8],[5,6,7,3,4,5]], dtype = torch.float) ,
            'leaf_node': torch.tensor([[1,2,3,2,3,4],[7,8,9,2,3,4],[3,2,4,2,3,4]], dtype = torch.float), 'isvalid_leaf_node': torch.tensor([[0,1,1]], dtype = torch.float),
            'curr_item': torch.tensor([[0,0,0,2,3,4]], dtype = torch.float)}
    if not True:
        model = AttentionModel(
            embed_dim= 12,feed_forward_hidden = 64
        )

        output = model(input)
        print(output)

    if not True:
        model = Pointer(input_dim = 12, hidden = 32)

        output = model(input['leaf_node'], input['curr_item'],input['isvalid_leaf_node'])
        print(output)
    

