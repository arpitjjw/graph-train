from models.graphsage import GNNStack
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import train_test_split_edges

import torch.optim as optim
import torch
from numpy import dot
from numpy.linalg import norm

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

def train_model(params,skills_graph,supervision_edges,supervision_bins,supervision_scores,validation_edges,validation_bins,validation_scores):
    
    skills_graph=skills_graph.to(device)
    supervision_edges=supervision_edges.to(device)
    supervision_bins=supervision_bins.to(device)
    supervision_scores=supervision_scores.to(device)
    validation_edges=validation_edges.to(device)
    validation_bins=validation_bins.to(device)
    validation_scores=validation_scores.to(device)

    params['model_type']='GraphSage'
    params['num_layers']=params['n_channels']
    
    batch_size=params['batch_size']
    hidden_dim=params['hidden_dim']
    output_dim=params['output_dim']
    model = GNNStack(skills_graph.num_node_features,hidden_dim,output_dim,params).to(device)
    scheduler, opt = build_optimizer(params,model.parameters())
    
    max_epochs=20
    val_checkpoint=100.0
    val_prev=100.0
    val_acc_prev=0.0
    val_acc_checkpoint=0.0
    val_count_increase=0
    for epoch in range(max_epochs):
        total_loss = 0
        model.train()
       
        total=0
        correct=torch.tensor(0.0,device=device)
        print(f"Epoch {epoch+1}")

        
        n_batches=int(supervision_edges.shape[1]/batch_size)+1
        for i in range(n_batches):
            train_edges = supervision_edges[:,i*batch_size:(i+1)*batch_size]
            nodes=torch.unique(torch.cat( (torch.unique(train_edges),torch.unique(validation_edges))  ))
            z=model.encode(skills_graph,nodes)
            
            # Local batches and labels
            
            train_scores= supervision_scores[i*batch_size:(i+1)*batch_size]
            train_bins= supervision_bins[i*batch_size:(i+1)*batch_size]
          
            opt.zero_grad()
            loss=model.contrastive_loss(z,train_edges,train_scores)
           
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
            model.eval()
            val_loss = model.contrastive_loss(z,validation_edges,validation_scores)
        print(f"Val loss: {val_loss.item():.4f}")
        mlflow.log_metric(key='val_loss',value=val_loss.item(),step=epoch+1)
        if val_loss.item()<val_checkpoint:
            print(f'Saving model with val loss: {val_loss.item():.4f}')
            val_checkpoint=val_loss.item()
            state_dict = model.state_dict()
            mlflow.pytorch.log_state_dict(state_dict, artifact_path="model")
        if val_loss.item()<val_prev:
            val_count_increase=0
        if val_loss.item()+0.02>val_prev:
            val_count_increase+=1
        val_prev=val_loss.item()
        if val_count_increase==2:
            break
        
        
          
        
    
    return_dict = {
            'train_loss':total_loss
        }
    
    return return_dict


def cos_sim(a,b):
   return dot(a, b)/(norm(a)*norm(b))

def test_model(df):

    raise NotImplementedError

