import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import negative_sampling


class GCNEncoder(nn.Module):
    """Two-layer Graph Convolutional Encoder"""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)
    
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        z = self.conv2(h, edge_index)
        return z


class GATEncoder(nn.Module):
    """Graph Attention Network Encoder"""
    def __init__(self, input_dim, hidden_dim, latent_dim, heads=4, dropout=0.6):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, 
                            concat=True, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, latent_dim, heads=1, 
                            concat=False, dropout=dropout)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.conv2(h, edge_index)
        return z


class InnerProductDecoder(nn.Module):
    """Inner Product Decoder"""
    def forward(self, z, edge_index, sigmoid=True):
        row, col = edge_index
        scores = (z[row] * z[col]).sum(dim=1)
        return torch.sigmoid(scores) if sigmoid else scores


class GAE(nn.Module):
    """Graph Autoencoder for Anomaly Detection"""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GAE, self).__init__()
        self.encoder = GCNEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = InnerProductDecoder()
    
    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)
    
    def decode(self, z, edge_index, sigmoid=True):
        return self.decoder(z, edge_index, sigmoid=sigmoid)
    
    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        pos_scores = self.decode(z, pos_edge_index, sigmoid=False)
        pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-15).mean()
        
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=z.size(0),
                num_neg_samples=pos_edge_index.size(1)
            )
        
        neg_scores = self.decode(z, neg_edge_index, sigmoid=False)
        neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-15).mean()
        
        return pos_loss + neg_loss
    
    def compute_anomaly_scores(self, z, edge_index):
        num_nodes = z.size(0)
        device = z.device
        
        anomaly_scores = torch.zeros(num_nodes, device=device)
        degree = torch.zeros(num_nodes, device=device)
        
        edge_probs = self.decode(z, edge_index, sigmoid=True)
        edge_errors = -torch.log(edge_probs + 1e-15)
        
        src_nodes = edge_index[0]
        anomaly_scores.scatter_add_(0, src_nodes, edge_errors)
        degree.scatter_add_(0, src_nodes, torch.ones_like(edge_errors))
        
        mask = degree > 0
        anomaly_scores[mask] = anomaly_scores[mask] / degree[mask]
        anomaly_scores[~mask] = 10.0
        
        return anomaly_scores


class GAE_GAT(nn.Module):
    """Graph Autoencoder with GAT Encoder"""
    def __init__(self, input_dim, hidden_dim, latent_dim, heads=4, dropout=0.6):
        super(GAE_GAT, self).__init__()
        self.encoder = GATEncoder(input_dim, hidden_dim, latent_dim, heads, dropout)
        self.decoder = InnerProductDecoder()
    
    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)
    
    def decode(self, z, edge_index, sigmoid=True):
        return self.decoder(z, edge_index, sigmoid=sigmoid)
    
    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        pos_scores = self.decode(z, pos_edge_index, sigmoid=False)
        pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-15).mean()
        
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=z.size(0),
                num_neg_samples=pos_edge_index.size(1)
            )
        
        neg_scores = self.decode(z, neg_edge_index, sigmoid=False)
        neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-15).mean()
        
        return pos_loss + neg_loss
    
    def compute_anomaly_scores(self, z, edge_index):
        num_nodes = z.size(0)
        device = z.device
        
        anomaly_scores = torch.zeros(num_nodes, device=device)
        degree = torch.zeros(num_nodes, device=device)
        
        edge_probs = self.decode(z, edge_index, sigmoid=True)
        edge_errors = -torch.log(edge_probs + 1e-15)
        
        src_nodes = edge_index[0]
        anomaly_scores.scatter_add_(0, src_nodes, edge_errors)
        degree.scatter_add_(0, src_nodes, torch.ones_like(edge_errors))
        
        mask = degree > 0
        anomaly_scores[mask] = anomaly_scores[mask] / degree[mask]
        anomaly_scores[~mask] = 10.0
        
        return anomaly_scores

