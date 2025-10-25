# Graph Neural Network Training

This repository contains implementations of graph neural network models including GraphSAGE, DEAL, and Attri2Vec for skill graph analysis.

## Environment Configuration

Before running the code, set the following environment variables:

```bash
export AWS_REGION=your-aws-region              # Default: us-east-1
export S3_BUCKET_NAME=your-s3-bucket           # Required for S3 operations
export S3_MODEL_PATH_PREFIX=your-mlflow-path   # Default: mlflow-artifacts
export MODEL_RUN_ID=your-model-run-id          # Default: latest
export TEST_SKILLS_CONFIG=path-to-config.json  # Optional: test skills configuration file
```

## Project Structure

```
src/
├── models/
│   ├── graphsage.py         # GraphSAGE model implementation
│   ├── graphsage_in_out.py  # Bidirectional GraphSAGE
│   ├── deal.py              # DEAL model implementation
│   └── attri2vec.py         # Attri2Vec model implementation
├── train_model.py           # Training script for GraphSAGE
├── train_graphsage.py       # Training script for bidirectional GraphSAGE  
├── test_model.py            # Model evaluation functions
├── graph.py                 # Graph utility class
└── utils/
    └── sbert_embed_demo.py  # Sentence embedding utilities
```

## Requirements

- PyTorch
- PyTorch Geometric (torch_geometric)
- PyTorch Scatter (torch_scatter)
- PyTorch Sparse (torch_sparse)
- NetworkX
- Node2Vec
- Gensim
- MLflow
- NumPy
- Pandas
- scikit-learn
- boto3 (for S3 operations)
- sentence-transformers (for utils/sbert_embed_demo.py)
- NLTK
- stellargraph (for Attri2Vec)
- tensorflow/keras (for Attri2Vec)
- torchviz
- torchsummary

## Training

The project includes a main entry point (`main.py`) that handles training, evaluation, prediction, and data preparation.

### Usage

```bash
# Training
python main.py --mode train --config config.yaml

# Evaluation
python main.py --mode evaluate --config config.yaml --run-id <mlflow_run_id>

# Prediction (not yet implemented)
python main.py --mode predict --config config.yaml

# Data preparation (not yet implemented)
python main.py --mode prepare --config config.yaml
```

### Supported Models

- **GraphSAGE**: Standard GraphSAGE implementation
- **GraphSAGE Directed**: GraphSAGE for directed graphs  
- **GAT**: Graph Attention Networks
- **DEAL**: Dual Embedding Alignment model
- **attri2vec**: Attributed network embedding

### Configuration File

Create a YAML configuration file (see `config_example.yaml`) with the following sections:
- Model name and type
- Data paths for processed files
- Training hyperparameters
- Model-specific parameters
- MLflow tracking configuration

## Model Parameters

### Common Parameters:
- `hidden_dim`: Hidden layer dimension
- `output_dim`: Output embedding dimension
- `batch_size`: Training batch size
- `lr`: Learning rate
- `dropout`: Dropout probability
- `num_layers` / `n_channels`: Number of graph convolution layers
- `opt`: Optimizer type (adam, sgd, rmsprop, adagrad)
- `opt_scheduler`: Learning rate scheduler (none, step, cos)
- `decay`: Weight decay for optimizer
- `model_type`: Model architecture type (GraphSAGE, GAT)
- `heads`: Number of attention heads (for GAT)

### DEAL-specific Parameters:
- `gamma`: Gamma parameter for DEAL loss function (default: 2)
- `b`: Bias parameter for DEAL loss function (default: 0.1)
- `theta_str`: Weight for structure loss
- `theta_attr`: Weight for attribute loss
- `theta_align`: Weight for alignment loss
- `lambda_attr`: Weight for attribute similarity in testing
- `lambda_attr_str`: Weight for attribute-structure similarity in testing
- `scheduler`: Whether to use learning rate scheduler ('on'/'off')
- `drop`: Whether to use dropout (boolean)

### Test Parameters:
- `top_n`: Number of top predictions to consider (for DEAL testing)

## Evaluation

The `test_model.py` module provides evaluation functions for all models:
- `test_graphsage()`: Evaluate GraphSAGE model
- `test_graphsage_in_out()`: Evaluate bidirectional GraphSAGE
- `test_deal()`: Evaluate DEAL model

## MLflow Integration

All experiments are tracked using MLflow. Metrics logged include:
- Training loss
- Validation loss
- Test scores (manual, auto-1, auto-2)

Model checkpoints are saved automatically when validation loss improves.

## Data Files

The project expects processed data files in the directory specified by `data-path.processed` in the configuration file. Required files vary by model:

### For GraphSAGE/GAT:
- `skills_graph.pt` / `skills_graph_directed.pt`
- `supervision_edges.pt` / `supervision_edges_directed.pt`
- `supervision_scores.pt` / `supervision_scores_directed.pt`
- `supervision_bins.pt` / `supervision_bins_directed.pt`
- `validation_edges.pt` / `validation_edges_directed.pt`
- `validation_scores.pt` / `validation_scores_directed.pt`
- `validation_bins.pt` / `validation_bins_directed.pt`

### For DEAL:
- `skills_graph_deal.pt`
- `supervision_edges_deal.pt`
- `supervision_scores_deal.pt`
- `validation_edges_deal.pt`
- `validation_scores_deal.pt`
- `negative_pairs_deal.npy`

### For Evaluation:
- `ground_truth_high.npy`
- `skills_available.npy`
- `test_skills.npy`
- `test_map_skill_embed.npy`
- `skills_graph_test.pt` / `skills_graph_test_directed.pt` / `skills_graph_deal.pt`

### For Attri2Vec:
- `skills_graph_attri2vec.pt`
- `ground_truth_high.npy`
- `skills_available.npy`
- `test_skills.npy`

## Test Skills Configuration

If you want to use custom test skill sets, create a JSON file with the following format:

```json
{
  "test_1": [1280, 7909, 1086, 7728, 604, 5938],
  "test_2": [5690, 976, 4092, 9843, 2140, 11129, 11280, 5739, 515, 11460, 7878, 26074, 2434]
}
```

Then set the environment variable:
```bash
export TEST_SKILLS_CONFIG=/path/to/your/test_skills.json
```