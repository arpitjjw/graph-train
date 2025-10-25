import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class nn_str(torch.nn.Module):
    def __init__(self, input_dim=128,hidden_dim=100,output_dim=64,dropout=True,p=0.5):
        super(nn_str, self).__init__()
        self.drop = dropout
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim

        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.batchnorm=nn.BatchNorm1d(self.hidden_dim,device=device)
        self.dropout=nn.Dropout(p=p)
        self.linear_out = nn.Linear(self.hidden_dim, self.output_dim)


    def forward(self, x):
       
        x = self.linear1(x)
        x = F.relu(x)
        x=self.batchnorm(x)
        if self.drop:
            x = self.dropout(x)
        
        x = self.linear_out(x)
        return x

class nn_attr(torch.nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256,output_dim=64,dropout=True,p=0.5):
        super(nn_attr, self).__init__()
        self.drop=dropout
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.hidden_dim=hidden_dim
        self.batchnorm1=nn.BatchNorm1d(600,device=device)
        self.dropout1=nn.Dropout(p=p)
        self.linear1 = nn.Linear(self.input_dim, 600)
        self.batchnorm2=nn.BatchNorm1d(400,device=device)
        self.dropout2=nn.Dropout(p=p)
        self.linear2 = nn.Linear(600,400)
        self.linear_out = nn.Linear(400, self.output_dim)


    def forward(self, x):
        
        x = self.linear1(x)
        x = F.relu(x)
        x=self.batchnorm1(x)
        if self.drop:
            x =self.dropout1(x)
        x = self.linear2(x)
        x=self.batchnorm2(x)
        if self.drop:
            x =self.dropout2(x)
        
        x = self.linear_out(x)
        return x

class DEAL(torch.nn.Module):

    def __init__(self,skills_graph,embedding_str_layer,embedding_attr_layer,input_dim=768, hidden_dim=256,output_dim=64,gamma=2,b=0.1,p=0.5,drop=True,str='deal'):
        super(DEAL, self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.hidden_dim=hidden_dim
        
        self.attr_layer=nn_attr(self.input_dim,self.hidden_dim,self.output_dim,drop,p)
        self.node_layer=nn_str(128,self.hidden_dim,self.output_dim,drop,p)
        
        self.gamma=gamma
        self.b=b


        self.embed_attr_layer=embedding_attr_layer
        self.embed_str_layer=embedding_str_layer


        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    
        
    def link_forward(self, nodes):
        first_embs = self.embed_str_layer(nodes[:,0])
        sec_embs = self.embed_str_layer(nodes[:,1])
        
        first_embs = F.normalize(self.node_layer(first_embs), p=2)
        sec_embs = F.normalize(self.node_layer(sec_embs), p=2)

        return self.cos(first_embs,sec_embs)
    
    def attr_forward(self,nodes):

        first_embs = self.embed_attr_layer(nodes[:,0])
        sec_embs = self.embed_attr_layer(nodes[:,1])
  

        first_embs = F.normalize(self.attr_layer(first_embs), p=2)
        sec_embs =  F.normalize(self.attr_layer(sec_embs), p=2)

        return self.cos(first_embs,sec_embs)

    def align_forward(self,nodes):

        first_attr = self.embed_attr_layer(nodes[:,0])
        sec_attr = self.embed_attr_layer(nodes[:,1])
  
        first_attr = F.normalize(self.attr_layer(first_attr), p=2)
        sec_attr =  F.normalize(self.attr_layer(sec_attr), p=2)
        
        first_str = self.embed_str_layer(nodes[:,0])
        sec_str = self.embed_str_layer(nodes[:,1])
        
        first_str = F.normalize(self.node_layer(first_str), p=2)
        sec_str = F.normalize(self.node_layer(sec_str), p=2)

        return (self.cos(first_str,sec_attr)+self.cos(first_attr,sec_str))/2.0
    
    def forward(self):
        
        pass


    def RLL_loss(self,scores_graph,scores,labels,alpha=0.2, mode='cos'):
        gamma_1 = self.gamma
        gamma_2 = self.gamma
        b_1 = self.b
        b_2 = self.b

        return torch.mean(labels*(torch.log(1+torch.exp(-scores*gamma_1+b_1)))/gamma_1+ (1-labels)*torch.log(1+torch.exp(scores*gamma_2+b_2))/gamma_2)

    def default_loss(self,nodes,scores_graph,thetas=(0,1,1)):
        nodes=nodes.T
        # nodes shape=[batch_size,2] 
        labels=get_link_labels(scores_graph)
        loss_list = []

        # structure loss
        
        scores = self.link_forward(nodes) #nodes is of shape [batch size,2] 2 for pairs
        link_loss = self.RLL_loss(scores_graph,scores,labels)
        loss_list.append(link_loss*thetas[0])

        # attribute loss
        scores = self.attr_forward(nodes) #nodes is of shape [batch size,2] 2 for pairs
        attr_loss = self.RLL_loss(scores_graph,scores,labels)
        loss_list.append(attr_loss*thetas[1])
        
        # alignment loss
        scores = self.align_forward(nodes)
        align_loss = self.RLL_loss(scores_graph,scores,labels)
        loss_list.append(align_loss*thetas[1])

        losses = torch.stack(loss_list)
        self.losses = losses.data
        return losses.sum()

def get_link_labels(scores):
    # returns a tensor:
    # [1,1,1,1,...,0,0,0,0,0,..] with the number of ones is equel to the lenght of pos_edge_index
    # and the number of zeros is equal to the length of neg_edge_index
    E = scores.size(0)
    link_labels = torch.zeros(E, dtype=torch.float,device=device)
    link_labels[scores>0] = 1.0
    return link_labels