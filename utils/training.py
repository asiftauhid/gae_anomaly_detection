import random
import numpy as np
import torch


def train_gae(model, graph, num_epochs=100, learning_rate=0.01, verbose=True, print_every=10, seed=42):
    """
    Train GAE model using link prediction loss
    
    Args:
        model: GAE model
        graph: PyTorch Geometric Data object
        num_epochs: number of training epochs
        learning_rate: learning rate for Adam optimizer
        verbose: whether to print training progress
        print_every: print loss every N epochs
        seed: random seed for reproducibility
    
    Returns:
        trained model, list of losses
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    losses = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        z = model.encode(graph.x, graph.edge_index)
        loss = model.recon_loss(z, graph.edge_index)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if verbose and (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    return model, losses

