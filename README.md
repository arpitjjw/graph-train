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
├── main.py                  # Main entry point for training/evaluation
├── models/
│   ├── graphsage.py         # GraphSAGE model implementation
│   ├── graphsage_in_out.py  # Bidirectional GraphSAGE
│   ├── deal.py              # DEAL model implementation
│   └── attri2vec.py         # Attri2Vec model implementation
├── train_model.py           # Training functions for GraphSAGE/DEAL
├── train_graphsage.py       # Training function for standard GraphSAGE  
├── test_model.py            # Model evaluation functions
├── graph.py                 # Graph utility class
├── utils/
│   └── sbert_embed_demo.py  # Sentence embedding utilities
├── data/
│   └── processed/           # Directory for processed data files
├── config_example.yaml      # Example configuration file
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd graph-neural-networks/src

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

1. **Prepare your data** in the format expected by the models (see Data Files section)
2. **Copy and modify** `config_example.yaml` to create your configuration:
   ```bash
   cp config_example.yaml config.yaml
   # Edit config.yaml with your parameters
   ```
3. **Set environment variables** if using S3:
   ```bash
   export S3_BUCKET_NAME=your-bucket
   export AWS_REGION=us-east-1
   ```
4. **Run training**:
   ```bash
   python main.py --mode train --config config.yaml
   ```
5. **View results** in MLflow UI:
   ```bash
   mlflow ui
   ```

## Requirements

Key dependencies include:
- PyTorch and PyTorch Geometric ecosystem
- MLflow for experiment tracking
- NetworkX and Node2Vec for graph operations
- TensorFlow/Keras and stellargraph (for Attri2Vec)
- boto3 for AWS S3 support
- See `requirements.txt` for complete list

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

Model selection is controlled by the `model_name` parameter in your configuration file:

- **GraphSAGE**: Standard GraphSAGE implementation
  - Set `model_name: "GraphSAGE"`
  - Uses `train_graphsage()` from `train_model.py`
  
- **GraphSAGE Directed**: GraphSAGE for directed graphs
  - Set `model_name: "GraphSAGE Directed"`
  - Uses directed graph data files
  
- **GAT**: Graph Attention Networks
  - Set `model_name: "GAT"`
  - Uses same training function as GraphSAGE
  
- **DEAL**: Dual Embedding Alignment model
  - Set `model_name: "DEAL"`
  - Uses `train_deal()` from `train_model.py`
  
- **attri2vec**: Attributed network embedding
  - Set `model_name: "attri2vec"`
  - Uses TensorFlow/Keras backend

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

Run evaluation with:
```bash
python main.py --mode evaluate --config config.yaml
```

The evaluation mode automatically selects the appropriate test function based on `model_name`:
- **GraphSAGE/GAT**: Uses `test_graphsage()`
- **Models with "in out"**: Uses `test_graphsage_in_out()` for bidirectional evaluation
- **DEAL**: Uses `test_deal()`

Note: The `--run-id` parameter is optional for evaluation mode.

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