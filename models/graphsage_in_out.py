import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from torch import Tensor
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
                                    
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.utils import k_hop_subgraph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GNNStack_in_out(torch.nn.Module):
    def __init__(self, input_dim,hidden_dim, output_dim, params, emb=False):
        super(GNNStack_in_out, self).__init__()
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.gamma=2
        self.params=params
        self.output_dim=output_dim
        conv_model = self.build_conv_model(params['model_type'].split(' ')[0])
        self.convs = nn.ModuleList()
        
        self.convs.append(conv_model(input_dim, hidden_dim))
        
        for l in range(params['num_layers']-1):
            self.convs.append(conv_model(params['heads'] * hidden_dim, hidden_dim))
        self.dropout = params['dropout']
        self.num_layers = params['num_layers']
        # post-message-passing
        self.post_mp = nn.Sequential(nn.Linear(hidden_dim, output_dim),nn.Dropout(p=self.dropout))
        self.nn_loss=nn.Linear(self.output_dim*2,2)
        self.emb = emb

    def build_conv_model(self, model_type):
        if model_type == 'GraphSAGE':
            return GraphSage
        elif model_type == 'GAT':
            return GAT
   
    def forward(self):
        pass

    def encode(self,data,nodes,reverse=False): 
    
    
        idx=list(nodes.cpu().detach().numpy())
        if not reverse:
            
            central_nodes,edges,mapping,mask=k_hop_subgraph(node_idx=list(nodes.cpu().detach().numpy()),edge_index=data.edge_index,num_hops=2,relabel_nodes=True)
            
            self.map_id_index_in=dict(zip(idx, list(mapping.cpu().detach().numpy())))
            map=self.map_id_index_in
            
        else:
           
            ei = data.edge_index.clone()
            ei[0] =ei[1]
            ei[1] = data.edge_index[0]
            central_nodes,edges,mapping,mask=k_hop_subgraph(node_idx=list(nodes.cpu().detach().numpy()),edge_index=ei,num_hops=2,relabel_nodes=True)
            
            self.map_id_index_out=dict(zip(idx, list(mapping.cpu().detach().numpy())))
            map=self.map_id_index_out
           
        x, edge_index,score = data.x[central_nodes], edges, data.score[mask]
    
      
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index,score)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)

        x = self.post_mp(x)
        x = F.normalize(x, p=2)
        return x,map

    def contrastive_loss(self,z_in,z_out,map_in,map_out,edges,score,alpha=0.2, mode='cos'):
        
        score_ = self.cos(z_out[[edges[0].cpu().apply_(lambda x:map_out[x])]] , z_in[[edges[1].cpu().apply_(lambda x:map_in[x])]]) 
        labels = get_link_labels(score)
        gamma_1 = self.gamma
        gamma_2 = self.gamma
        b_1 = 0.1
        b_2 = 0.1
       
        return torch.mean(labels*(torch.log(1+torch.exp(-6*score_+1)))+(1-labels)*torch.log(1+torch.exp(6*score_+2)))
    
    def KG_bert_loss(self,z,edges,score,alpha=0.2, mode='cos'):
        concat_edges=torch.concat((z[[edges[0].cpu().apply_(lambda x:self.map_id_index[x])]] , z[[edges[1].cpu().apply_(lambda x:self.map_id_index[x])]]) ,1)
    
        score_=self.nn_loss(concat_edges)
        score_=nn.Softmax()(score_)
      
        labels = get_link_labels(score)
        return -torch.mean(labels*torch.log(score_[:,0])+(1-labels)*torch.log(score_[:,1]))
        

    def supervised_loss(self,z,edges,bins):
        loss_fn = nn.CrossEntropyLoss()
        out=self.post_mp(F.normalize(z[edges[0].cpu().apply_(lambda x:self.map_id_index[x])]*z[edges[1].cpu().apply_(lambda x:self.map_id_index[x])],p=2))
        
    
        
     
        
        loss=loss_fn(out,bins.long())
        
        correct= (torch.softmax(out, dim=1).argmax(dim=1) == bins.long()).float().sum()
        
        return loss,correct

def get_link_labels(scores):
    # returns a tensor:
    # [1,1,1,1,...,0,0,0,0,0,..] with the number of ones is equal to the lenght of pos_edge_index
    # and the number of zeros is equal to the length of neg_edge_index
    E = scores.size(0)
    link_labels = torch.zeros(E, dtype=torch.float,device=device)
    link_labels[scores>0] = 1.0
    return link_labels




class GraphSage(MessagePassing):
    
    def __init__(self, in_channels, out_channels, normalize = True,
                 bias = False, **kwargs):  
        super(GraphSage, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_l = None
        self.lin_r = None

        self.lin_l = nn.Linear(self.in_channels, self.out_channels)
        self.lin_r = nn.Linear(self.in_channels, self.out_channels)

        self.reset_parameters()


    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index,score, size = None):
        """"""
       
        score=score.expand(x.size(1),score.size(0)).T
        self.score=score
        
        
        prop = self.propagate(edge_index, x=(x, x), size=size)
        out = self.lin_l(x) + self.lin_r(prop)
        if self.normalize:
            out = F.normalize(out, p=2)

        return out

    def message(self, x_j):
       
        
        
        out = torch.mul(x_j, self.score)
      
        return out

    def aggregate(self, inputs, index, dim_size = None):
        node_dim = self.node_dim
        out = torch_scatter.scatter(inputs, index, node_dim, dim_size=dim_size, reduce='mean')


        return out


class GAT(MessagePassing):

    def __init__(self, in_channels, out_channels, heads = 1,normalize = True,
                 negative_slope = 0.2, dropout = 0., **kwargs):
        super(GAT, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.normalize = normalize

        self.lin_l = None
        self.lin_r = None
        self.att_l = None
        self.att_r = None

        self.lin_l = Linear(in_channels, out_channels*self.heads)

        self.lin_r = self.lin_l

        self.att_l = Parameter(torch.Tensor(1, self.heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, self.heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index,score, size = None):
        
        H, C = self.heads, self.out_channels

        wh_l = self.lin_l(x).view(-1, H, C)
        wh_r = self.lin_r(x).view(-1, H, C)
       

        alpha_l = torch.mul(self.att_l, wh_l)
        alpha_r = torch.mul(self.att_r, wh_r)
        

        prop = self.propagate(edge_index,x=(wh_l, wh_r), size=size, alpha=(alpha_l, alpha_r))
     
        out = prop.view(-1, H*C) #

        return out
        


    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):

        att = alpha_i + alpha_j
        

        att = F.leaky_relu(att, negative_slope=self.negative_slope)
        att = softmax(att, ptr if ptr else index)
        att = F.dropout(att, self.dropout)
       
        out = torch.mul(x_j, att)
        
        return out


    def aggregate(self, inputs, index, dim_size = None):

        out = torch_scatter.scatter(inputs, index = index, dim = self.node_dim, dim_size = dim_size, reduce = "sum")
    
        return out