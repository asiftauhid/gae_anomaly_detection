import torch
from .models import GAE


def save_model(model, filepath, hyperparameters=None, metrics=None):
    """
    Save GAE model with hyperparameters and metrics
    
    Args:
        model: GAE model to save
        filepath: path to save model
        hyperparameters: dict of hyperparameters
        metrics: dict of evaluation metrics
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'hyperparameters': hyperparameters or {},
        'metrics': metrics or {}
    }
    torch.save(checkpoint, filepath)


def load_model(filepath, input_dim, hidden_dim, latent_dim):
    """
    Load GAE model from checkpoint
    
    Args:
        filepath: path to saved model
        input_dim: input feature dimension
        hidden_dim: hidden layer dimension
        latent_dim: latent space dimension
    
    Returns:
        model: loaded GAE model
        checkpoint: full checkpoint dict
    """
    model = GAE(input_dim, hidden_dim, latent_dim)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint

