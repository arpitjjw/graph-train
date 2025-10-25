from models.graphsage_in_out import GNNStack_in_out
from models.graphsage import GNNStack
from models.deal import DEAL
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import train_test_split_edges

import torch.optim as optim
import torch
from numpy import dot
from numpy.linalg import norm
from torchviz import make_dot
import torch.nn as nn
from torchsummary import summary

import mlflow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_optimizer(params,model_params):
    weight_decay = params['decay']
    filter_fn = filter(lambda p : p.requires_grad, model_params)
    if params['opt'] == 'adam':
        optimizer = optim.Adam(filter_fn, lr=params['lr'], weight_decay=weight_decay)
    elif params['opt'] == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=params['lr'], momentum=0.95, weight_decay=weight_decay)
    elif params['opt'] == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=params['lr'], weight_decay=weight_decay)
    elif params['opt'] == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=params['lr'], weight_decay=weight_decay)
    if params['opt_scheduler']== 'none':
        return None, optimizer
    elif params['opt_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['step_size'])
    elif params['opt_scheduler'] == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['opt_restart'])
    return scheduler, optimizer

def train_graphsage(config,params,skills_graph,supervision_edges,supervision_bins,supervision_scores,validation_edges,validation_bins,validation_scores):
    
    skills_graph=skills_graph.to(device)
    supervision_edges=supervision_edges.to(device)
    supervision_bins=supervision_bins.to(device)
    supervision_scores=supervision_scores.to(device)
    validation_edges=validation_edges.to(device)
    validation_bins=validation_bins.to(device)
    validation_scores=validation_scores.to(device)


    params['num_layers']=params['n_channels']

    batch_size=params['batch_size']
    hidden_dim=params['hidden_dim']
    output_dim=params['output_dim']
    model_in = GNNStack_in_out(skills_graph.num_node_features,hidden_dim,output_dim,params).to(device)
    model_out= GNNStack_in_out(skills_graph.num_node_features,hidden_dim,output_dim,params).to(device)
    scheduler, opt = build_optimizer(params,list(model_in.parameters()) + list(model_out.parameters()))
    
    max_epochs=40
    val_checkpoint=100.0
    val_prev=100.0
    val_count_increase=0
    for epoch in range(max_epochs):
        total_loss = 0
        model_in.train()
        model_out.train()
        
       
        total=0
        print(f"Epoch {epoch+1}")
        
        
        n_batches=int(supervision_edges.shape[1]/batch_size)
        
        for i in range(n_batches):
            
            train_edges = supervision_edges[:,i*batch_size:(i+1)*batch_size]
            nodes=torch.unique(torch.unique(train_edges)  )
            z_in,map_in=model_in.encode(skills_graph,nodes)
            z_out,map_out=model_out.encode(skills_graph,nodes,reverse=True)
    
            
            # Local batches and labels
            
            train_scores= supervision_scores[i*batch_size:(i+1)*batch_size]
            train_bins= supervision_bins[i*batch_size:(i+1)*batch_size]
          
            
            loss=model_in.contrastive_loss(z_in,z_out,map_in,map_out,train_edges,train_scores)
           
            total+=batch_size
            
            loss.backward()
            opt.step()
            opt.zero_grad()
            total_loss += loss.item()
          
        
        scheduler.step()
        total_loss /= (n_batches*1.0)
        mlflow.log_metric(key='train_loss',value=total_loss,step=epoch+1)

        print(f"Train loss: {total_loss:.4f}")
        with torch.set_grad_enabled(False):
            model_in.eval()
            model_out.eval()
            
            nodes=torch.unique(torch.unique(validation_edges)  )
            z_in,map_in=model_in.encode(skills_graph,nodes)
            z_out,map_out=model_out.encode(skills_graph,nodes,reverse=True)
            
            
            val_loss = model_in.contrastive_loss(z_in,z_out,map_in,map_out,validation_edges,validation_scores)


        print(f"Val loss: {val_loss.item():.4f}")
        mlflow.log_metric(key='val_loss',value=val_loss.item(),step=epoch+1)
        if val_loss.item()<val_checkpoint:
            
            print(f'Saving model with val loss: {val_loss.item():.4f}')
            val_checkpoint=val_loss.item()
            state_dict = model_in.state_dict()
           
            mlflow.pytorch.log_state_dict(state_dict, artifact_path="model_in")
            state_dict = model_out.state_dict()
            mlflow.pytorch.log_state_dict(state_dict, artifact_path="model_out")
            
         
        if val_loss.item()<val_prev:
            val_count_increase=0
        if val_loss.item()>val_prev:
            val_count_increase+=1
        val_prev=val_loss.item()
        if val_count_increase==2:
            break
        
        
          
        
    
    return_dict = {
            'train_loss':total_loss
        }
    
    return return_dict

def train_deal(config,params,skills_graph,supervision_edges,supervision_bins,supervision_scores,validation_edges,validation_bins,validation_scores,negative_pairs,str='deal'):
   
    skills_graph=skills_graph.to(device)
    
    supervision_edges=supervision_edges.to(device)
    
    #supervision_bins=supervision_bins.to(device)
    supervision_scores=supervision_scores.to(device)
    validation_edges=validation_edges.to(device)
    #validation_bins=validation_bins.to(device)
    validation_scores=validation_scores.to(device)
    
    
    params['model_type']='DEAL'
    batch_size=params['batch_size']
    hidden_dim=params['hidden_dim']
    output_dim=params['output_dim']
    thetas=(params['theta_str'],params['theta_attr'],params['theta_align'])
    
    embedding_attr=torch.from_numpy(skills_graph.x).to(device).float()
    
    embedding_str=torch.from_numpy(skills_graph.node2vec).to(device).float()

    embed_attr_layer=nn.Embedding.from_pretrained(embedding_attr)
    embed_str_layer=nn.Embedding.from_pretrained(embedding_str)

    model = DEAL(skills_graph,embed_str_layer,embed_attr_layer,skills_graph.x.shape[1],hidden_dim,output_dim,params['gamma'],params['b'],params['dropout'],params['drop'],str).to(device)
    run_id=mlflow.active_run().info.run_id
    tags={'run_id':run_id}
    mlflow.set_tags(tags)
    list_summary = summary(model.attr_layer, input_size=(768,))
    textfile = open("architecture_attr.txt", "w")
    for string in list_summary:
        textfile.write(string + "\n")
    textfile.close()
    mlflow.log_artifact("architecture_attr.txt", artifact_path="architecture")
    list_summary = summary(model.node_layer, input_size=(128,))
    textfile = open("architecture_str.txt", "w")
    for string in list_summary:
        textfile.write(string + "\n")
    textfile.close()
    mlflow.log_artifact("architecture_str.txt", artifact_path="architecture")
    mlflow.log_artifact("src/models/deal.py", artifact_path="model_code")

    mlflow.log_param('loss function','torch.mean(labels*(torch.log(1+torch.exp(-scores*gamma_1+b_1)))/gamma_1+ (1-labels)*torch.log(1+torch.exp(scores*gamma_2+b_2))/gamma_2)')
    scheduler, opt = build_optimizer(params,model.parameters())
    
    max_epochs=30
    val_checkpoint=100.0
    val_prev=100.0
    val_count_increase=0

    

    for epoch in range(max_epochs):
        
        total_loss = 0
        model.train()
       
        total=0
        print(f"Epoch {epoch+1}")

        
        n_batches=int(supervision_edges.shape[1]/batch_size)+1
        for i in range(n_batches-1):
            
            train_edges = supervision_edges[:,i*batch_size:(i+1)*batch_size]
            
            nodes=torch.unique(torch.cat( (torch.unique(train_edges),torch.unique(validation_edges))  ))
            
            
            # Local batches and labels
            
            train_scores= supervision_scores[i*batch_size:(i+1)*batch_size]
          
            opt.zero_grad()
            
            loss=model.default_loss(train_edges,train_scores,thetas)
           
            total+=batch_size
            loss.backward()
            opt.step()
            opt.zero_grad()
          
            total_loss += loss.item()
          
        
        
        if params['scheduler']=='on':
            scheduler.step()


        total_loss /= (n_batches*1.0)
        mlflow.log_metric(key='train_loss',value=total_loss,step=epoch+1)
        print(f"Train loss: {total_loss:.4f}")

        with torch.set_grad_enabled(False):
            model.eval()
            val_loss = model.default_loss(validation_edges,validation_scores,thetas)

        print(f"Val loss: {val_loss.item():.4f}")
        mlflow.log_metric(key='val_loss',value=val_loss.item(),step=epoch+1)
        if val_loss.item()<val_checkpoint:
            print(f'Saving model with val loss: {val_loss.item():.4f}')
            val_checkpoint=val_loss.item()
            state_dict = model.state_dict()
            
            mlflow.pytorch.log_state_dict(state_dict, artifact_path="model")
        if val_loss.item()<val_prev:
            val_count_increase=0
        if val_loss.item()>val_prev:
            val_count_increase+=1
        val_prev=val_loss.item()

        
        
          
        
    
    return_dict = {
            'train_loss':total_loss
        }
    
    return return_dict


def cos_sim(a,b):
   return dot(a, b)/(norm(a)*norm(b))



