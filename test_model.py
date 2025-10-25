import torch
import numpy as np
import os

from models.graphsage import GNNStack
from models.graphsage_in_out import GNNStack_in_out
from models.deal import DEAL

from numpy import dot, dtype
from numpy.linalg import norm
import mlflow

import math 
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import boto3
import datetime
from botocore.config import Config
import json 

config = Config(
	connect_timeout=5,
	read_timeout=5,
	retries = {
		'max_attempts': 2,
		'mode': 'standard'
	}
)
def get_file_from_s3(run_id):
	# Validate required environment variables
	required_vars = {
		'S3_BUCKET_NAME': os.environ.get('S3_BUCKET_NAME'),
		'AWS_REGION': os.environ.get('AWS_REGION', 'us-east-1'),
		'S3_MODEL_PATH_PREFIX': os.environ.get('S3_MODEL_PATH_PREFIX', 'mlflow-artifacts')
	}
	
	missing_vars = [k for k, v in required_vars.items() if not v and k != 'AWS_REGION' and k != 'S3_MODEL_PATH_PREFIX']
	if missing_vars:
		raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
	
	try:
		client = boto3.client('s3', region_name=required_vars['AWS_REGION'], config=config)
		input_object = client.get_object(
				Bucket = required_vars['S3_BUCKET_NAME'],
				Key = required_vars['S3_MODEL_PATH_PREFIX'] + f"/{run_id}/artifacts/model/state_dict.pth"
			)
		input_object_body = input_object['Body'].read()
		
		return input_object_body
	except boto3.exceptions.Boto3Error as e:
		raise RuntimeError(f"Failed to load model from S3: {str(e)}")
	except Exception as e:
		raise RuntimeError(f"Unexpected error loading model from S3: {str(e)}")



def cos_sim(a,b):
   return dot(a, b)/(norm(a)*norm(b))

def test_2(scores_dict,ground_truth,k,skill_id):
    count=0
    final_pred=set()
    for skill,score in scores_dict.items():
        count+=1
        final_pred.add(skill)
        if count==k:
            break
    common=final_pred.intersection(ground_truth[skill_id])
    total=len(ground_truth[skill])
    correct=len(common)
    return correct/total

def mean_rank(scores_dict,ground_truth,skill_id):
    total=len(ground_truth[skill_id])
    sum=0.0
    for rank,skill,score in enumerate(scores_dict.items()):
        if skill in ground_truth[skill_id]:
            sum+=rank
    return rank/total

def test_3(scores_dict,ground_truth,k,skill_id):
    #top k predictions, should be greater than 100
    ground_truth=ground_truth[skill_id]
    total=len(ground_truth)
    a=k/2
    sum=0.0
    position=0
    for i,k,v in enumerate(scores_dict.items()):
        if k in ground_truth:
            position+=1
            sum+=math.exp(position-(i+1))/a
    return sum/total

def pyg_dict(skills_graph):
    pyg_id_key=dict()
    for i,id in enumerate(skills_graph.id):
        pyg_id_key[id.item()]=i
    pyg_key_id=dict()
    for k,v in pyg_id_key.items():
        pyg_key_id[v]=k
    return pyg_id_key

def test_graphsage(params,config,ground_truth,skills_graph,skills_available,test_skills,run_id):
    params['num_layers']=params['n_channels']
    pyg_id_key=pyg_dict(skills_graph)
    
    model=GNNStack(768,config['hidden_dim'],config['output_dim'],params)
    
    # Model loading would need to be implemented here using get_file_from_s3 function
    
    model.eval()
    nodes=torch.unique(torch.unique(skills_graph.edge_index)  )
    z=model.encode(skills_graph,nodes)

    test_score2=0
    for i in test_skills:
        id=pyg_id_key[i]
        scores_dict=dict()
        for j in skills_available:
            id2=pyg_id_key[j]
            score=cos_sim(z[id2].detach().numpy() ,z[id].detach().numpy())
            scores_dict[j]=score

        scores_dict = dict( sorted(scores_dict.items(),
                                        key=lambda item: item[1],
                                        reverse=True))
        final_pred=set()
        count=0
        for k,v in scores_dict.items():
            count+=1
            final_pred.add(k)
            if count==100:
                break
        common=final_pred.intersection(ground_truth[i])
        total=len(ground_truth[i])
        correct=len(common)
        test_score2+=correct/total
        test_score2/=len(test_skills)
        mlflow.log_metric(key='test_score2',value=test_score2)

    return_dict = {
            'test2 score':test_score2
        }
    return return_dict
   

def test_graphsage_in_out(params,config,ground_truth,skills_graph,skills_available,test_skills,run_id):
    params['num_layers']=params['n_channels']

    pyg_id_key=pyg_dict(skills_graph)
    
    model_in=GNNStack_in_out(768,config['hidden_dim'],config['output_dim'],params)
    model_out=GNNStack_in_out(768,config['hidden_dim'],config['output_dim'],params)
    
    # Model loading would need to be implemented here using get_file_from_s3 function
    
    model_in.eval()
    model_out.eval()
    nodes=torch.unique(torch.unique(skills_graph.edge_index)  )
    z_in,map_in=model_in.encode(skills_graph,nodes)
    z_out,map_out=model_out.encode(skills_graph,nodes,reverse=True)
    

    test_score2=0
    for i in test_skills:
        id=pyg_id_key[i]
        scores_dict=dict()
        for j in skills_available:
            id2=pyg_id_key[j]
            score=cos_sim(z_in[id2].detach().numpy() ,z_out[id].detach().numpy())
            score2=cos_sim(z_out[id2].detach().numpy() ,z_in[id].detach().numpy())
            score=max(score,score2)
            scores_dict[j]=score
        scores_dict = dict( sorted(scores_dict.items(),
                        key=lambda item: item[1],
                        reverse=True))
        final_pred=set()
        count=0
        for k,v in scores_dict.items():
            count+=1
            final_pred.add(k)
            if count==100:
                break
        common=final_pred.intersection(ground_truth[i])
        total=len(ground_truth[i])
        correct=len(common)
        test_score2+=correct/total
        test_score2/=len(test_skills)
        mlflow.log_metric(key='test_score2',value=test_score2)
        




    return_dict = {
            'test2 score':test_score2
        }
    
    return return_dict
    
def test_deal(params,config,ground_truth,skills_graph,skills_available,test_skills,test_map_skill_embed):
    # Load test skill sets from config file
    test_skills_config = os.environ.get('TEST_SKILLS_CONFIG', 'test_skills.json')
    if os.path.exists(test_skills_config):
        with open(test_skills_config, 'r') as f:
            test_config = json.load(f)
            test_1 = set(test_config.get('test_1', []))
            test_2 = set(test_config.get('test_2', []))
    else:
        # Fallback to default values if config file doesn't exist
        test_1={1280,7909,1086,7728,604,5938}
        test_2={5690,976,4092,9843,2140,11129,11280,5739,515,11460,7878,26074,2434}
    skills_graph=skills_graph.to(device)
    pyg_id_key=pyg_dict(skills_graph)
    embedding_attr=torch.from_numpy(skills_graph.x).to(device).float()
        
    embedding_str=torch.from_numpy(skills_graph.node2vec).to(device).float()

    embed_attr_layer=nn.Embedding.from_pretrained(embedding_attr)
    embed_str_layer=nn.Embedding.from_pretrained(embedding_str)
    input_dim=768
    model= DEAL(skills_graph,embed_str_layer,embed_attr_layer,input_dim, config['hidden_dim'],config['output_dim'],str='deal').to(device)
    
    # Load model from S3 with error handling
    model_run_id = os.environ.get('MODEL_RUN_ID', 'latest')
    s3_bucket = os.environ.get('S3_BUCKET_NAME')
    s3_prefix = os.environ.get('S3_MODEL_PATH_PREFIX', 'mlflow-artifacts')
    
    if not s3_bucket:
        raise ValueError("S3_BUCKET_NAME environment variable is required")
    
    try:
        state_dict = mlflow.pytorch.load_state_dict(f's3://{s3_bucket}/{s3_prefix}/{model_run_id}/artifacts/model')
        model.load_state_dict(state_dict)
        mlflow.log_param('run_id', model_run_id)
        model.eval()
        params_model=mlflow.get_run(model_run_id).data.params
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
    for key,value in params_model.items():
        mlflow.log_param(f'MODEL_'+key,value)
    with torch.no_grad():
        attr_embs = model.embed_attr_layer.weight.data
        attr_embs = model.attr_layer(attr_embs)

    with torch.no_grad():
        str_embs = model.embed_str_layer.weight.data
        str_embs = model.node_layer(str_embs)

    

    
    
    embed_test=torch.from_numpy(np.stack(list(test_map_skill_embed.values()),axis=0)).to(device).float()

    with torch.no_grad():
            embed_test=model.attr_layer(embed_test)

    test_score2_manual=0
    test_score2_auto_1=0
    test_score2_auto_2=0
    for ind,i in enumerate(test_map_skill_embed.keys()):
        if (i in ground_truth.keys()):
            embed=embed_test[ind]
            
            scores_dict=dict()
            for j in skills_available:
        
                id2=pyg_id_key[j]
                score=params['lambda_attr']*cos_sim(attr_embs[id2].cpu().detach().numpy() ,embed.cpu().detach().numpy())+params['lambda_attr_str']*cos_sim(str_embs[id2].cpu().detach().numpy() ,embed.cpu().detach().numpy())
                
                scores_dict[j]=score
            scores_dict = dict( sorted(scores_dict.items(),
                            key=lambda item: item[1],
                            reverse=True))
            run_id=mlflow.active_run().info.run_id
        
            mlflow.log_dict(scores_dict,f'scores_{i}.json')
            final_pred=set()
            count=0
            for k,v in scores_dict.items():
                count+=1
                final_pred.add(k)
                if count==params['top_n']:
                    break
            common=final_pred.intersection(ground_truth[i])
            total=len(ground_truth[i])
            correct=len(common)
            if i in test_1:
                test_score2_auto_1+=correct/total
            elif i in test_2:
                test_score2_auto_2+=correct/total
            else:
                test_score2_manual+=correct/total
    test_score2_manual/=13
    test_score2_auto_1/=6
    test_score2_auto_2/=13
    print(f"Test scores - Manual: {test_score2_manual:.4f}, Auto-1: {test_score2_auto_1:.4f}, Auto-2: {test_score2_auto_2:.4f}")



    mlflow.log_metric(key='test_score2',value=test_score2_manual)
    mlflow.log_metric(key='test_score2_auto_1',value=test_score2_auto_1)
    mlflow.log_metric(key='test_score2_auto_2',value=test_score2_auto_2)
      




    return_dict = {
            'test2 score':test_score2_manual,
            'test_score2_auto_1':test_score2_auto_1,
            'test_score2_auto_2':test_score2_auto_2
        }
    
    return return_dict






  
