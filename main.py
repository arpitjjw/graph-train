import os
import json
import torch
from torch.utils.data import DataLoader
from matplotlib.pyplot import colorbar
import mlflow
import logging
import subprocess
import seaborn as sns
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from train_model import train_graphsage, train_deal
from models.attri2vec import train_attri2vec
from test_model import test_graphsage, test_graphsage_in_out, test_deal

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger("ontology-expansion")


def train(args, config, kwargs):
    mlflow.set_tag('model_name', config['model_name'])
    params = {
        'batch_size': config['batch_size'],
        'n_channels': config['n_channels'],
        'hidden_dim': config['hidden_dim'],
        'output_dim': config['output_dim'],
        'shuffle': config['shuffle'],
        'opt': config['opt'],
        'decay': config['decay'],
        'opt_scheduler': config['opt_scheduler'],
        'lr': config['lr'],
        'opt_restart': config['opt_restart'],
        'step_size': config['step_size'],
        'num_neighbors': config['num_neighbors'],
        'dropout': config['dropout'],
        'drop': config['drop'],
        'theta_str': config['theta_str'],
        'theta_attr': config['theta_attr'],
        'theta_align': config['theta_align'],
        'model_type': config['model_name'],
        'heads': config['heads'],
        'gamma': config['gamma'],
        'b': config['b'],
        'scheduler': config['scheduler']
    }
    for key, value in params.items():
        mlflow.log_param(key, value)
    
    data_path = config['data-path']['processed']
    
    if ('GraphSAGE' in config['model_name'] or 'GAT' in config['model_name']) and 'Directed' in config['model_name']:
        skills_graph = torch.load(os.path.join(data_path, 'skills_graph_directed.pt'))
        supervision_edges = torch.load(os.path.join(data_path, 'supervision_edges_directed.pt'))
        supervision_scores = torch.load(os.path.join(data_path, 'supervision_scores_directed.pt'))
        supervision_bins = torch.load(os.path.join(data_path, 'supervision_bins_directed.pt'))
        validation_edges = torch.load(os.path.join(data_path, 'validation_edges_directed.pt'))
        validation_scores = torch.load(os.path.join(data_path, 'validation_scores_directed.pt'))
        validation_bins = torch.load(os.path.join(data_path, 'validation_bins_directed.pt'))

        out = train_graphsage(config, params, skills_graph, supervision_edges, supervision_bins, 
                            supervision_scores, validation_edges, validation_bins, validation_scores)
    
    elif ('GraphSAGE' in config['model_name'] or 'GAT' in config['model_name']):
        skills_graph = torch.load(os.path.join(data_path, 'skills_graph.pt'))
        supervision_edges = torch.load(os.path.join(data_path, 'supervision_edges.pt'))
        supervision_scores = torch.load(os.path.join(data_path, 'supervision_scores.pt'))
        supervision_bins = torch.load(os.path.join(data_path, 'supervision_bins.pt'))
        validation_edges = torch.load(os.path.join(data_path, 'validation_edges.pt'))
        validation_scores = torch.load(os.path.join(data_path, 'validation_scores.pt'))
        validation_bins = torch.load(os.path.join(data_path, 'validation_bins.pt'))

        out = train_graphsage(config, params, skills_graph, supervision_edges, supervision_bins, 
                            supervision_scores, validation_edges, validation_bins, validation_scores)
    
    elif 'DEAL' in config['model_name']:
        skills_graph = torch.load(os.path.join(data_path, 'skills_graph_deal.pt'))
        supervision_edges = torch.load(os.path.join(data_path, 'supervision_edges_deal.pt'))
        supervision_scores = torch.load(os.path.join(data_path, 'supervision_scores_deal.pt'))
        validation_edges = torch.load(os.path.join(data_path, 'validation_edges_deal.pt'))
        validation_scores = torch.load(os.path.join(data_path, 'validation_scores_deal.pt'))
        negative_pairs = np.load(os.path.join(data_path, 'negative_pairs_deal.npy'), allow_pickle=True).item()
        supervision_bins = None
        validation_bins = None
        out = train_deal(config, params, skills_graph, supervision_edges, supervision_bins, 
                        supervision_scores, validation_edges, validation_bins, validation_scores, 
                        negative_pairs, 'deal')
    
    elif 'attri2vec' in config["model_name"]:
        ground_truth = np.load(os.path.join(data_path, 'ground_truth_high.npy'), allow_pickle=True).item()
        skills_available = np.load(os.path.join(data_path, 'skills_available.npy'), allow_pickle=True).item()
        test_skills = np.load(os.path.join(data_path, 'test_skills.npy'), allow_pickle=True).item()
        skills_graph = torch.load(os.path.join(data_path, 'skills_graph_attri2vec.pt'))
        out = train_attri2vec(config, skills_graph)

    return {}
    

def evaluate(args, config, kwargs):
    params = {
        'batch_size': config['batch_size'],
        'n_channels': config['n_channels'],
        'hidden_dim': config['hidden_dim'],
        'output_dim': config['output_dim'],
        'shuffle': config['shuffle'],
        'opt': config['opt'],
        'decay': config['decay'],
        'opt_scheduler': config['opt_scheduler'],
        'lr': config['lr'],
        'opt_restart': config['opt_restart'],
        'step_size': config['step_size'],
        'num_neighbors': config['num_neighbors'],
        'dropout': config['dropout'],
        'drop': config['drop'],
        'theta_str': config['theta_str'],
        'theta_attr': config['theta_attr'],
        'theta_align': config['theta_align'],
        'model_type': config['model_name'],
        'heads': config['heads'],
        'gamma': config['gamma'],
        'b': config['b'],
        'scheduler': config['scheduler'],
        'lambda_attr': config['lambda_attr'],
        'lambda_attr_str': config['lambda_attr_str'],
        'top_n': config['top_n']
    }
    
    keys = ['lambda_attr', 'lambda_attr_str', 'top_n']
    for key, value in params.items():
        if key in keys:
            mlflow.log_param(key, value)
    
    data_path = config['data-path']['processed']
    ground_truth = np.load(os.path.join(data_path, 'ground_truth_high.npy'), allow_pickle=True).item()
    skills_available = np.load(os.path.join(data_path, 'skills_available.npy'), allow_pickle=True).item()
    test_skills = np.load(os.path.join(data_path, 'test_skills.npy'), allow_pickle=True).item()
    test_map_skill_embed = np.load(os.path.join(data_path, 'test_map_skill_embed.npy'), allow_pickle=True).item()
    
    if "Directed" in config['model_name']:
        skills_graph = torch.load(os.path.join(data_path, 'skills_graph_test_directed.pt'))
    elif "DEAL" in config["model_name"]:
        skills_graph = torch.load(os.path.join(data_path, 'skills_graph_deal.pt'))
        test_deal(params, config, ground_truth, skills_graph, skills_available, test_skills, test_map_skill_embed)
        return {}
    else:
        skills_graph = torch.load(os.path.join(data_path, 'skills_graph_test.pt'))

    if 'in out' in config['model_name']:
        test_graphsage_in_out(params, config, ground_truth, skills_graph, skills_available, test_skills)
    else:
        test_graphsage(params, config, ground_truth, skills_graph, skills_available, test_skills)
            
    return {}


def predict(args, config, kwargs):
    raise NotImplementedError("Prediction functionality not yet implemented")
    

def prepare(args, config, kwargs):
    raise NotImplementedError("Data preparation functionality not yet implemented")


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Graph Neural Network Training')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'predict', 'prepare'],
                        required=True, help='Mode to run the script in')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to configuration file')
    parser.add_argument('--run-id', type=str, default=None, 
                        help='MLflow run ID for evaluation mode')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up MLflow
    if 'mlflow' in config:
        mlflow.set_tracking_uri(config['mlflow'].get('tracking_uri', 'file:./mlruns'))
        mlflow.set_experiment(config['mlflow'].get('experiment_name', 'graph-neural-networks'))
    
    # Run appropriate function
    if args.mode == 'train':
        with mlflow.start_run():
            train(args, config, {'run_id': args.run_id})
    elif args.mode == 'evaluate':
        with mlflow.start_run():
            evaluate(args, config, {'run_id': args.run_id})
    elif args.mode == 'predict':
        predict(args, config, {'run_id': args.run_id})
    elif args.mode == 'prepare':
        prepare(args, config, {'run_id': args.run_id})