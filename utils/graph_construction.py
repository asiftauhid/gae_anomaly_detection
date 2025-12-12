import numpy as np
import torch
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from torch_geometric.data import Data


def build_knn_graph(features, k=10, add_self_loops=True, metric='cosine'):
    """
    Build k-NN graph from feature matrix
    
    Args:
        features: numpy array of shape (n_nodes, n_features)
        k: number of nearest neighbors
        add_self_loops: whether to add self-loops
        metric: distance metric ('cosine', 'euclidean', 'manhattan')
    
    Returns:
        edge_index: PyTorch tensor of shape (2, num_edges)
    """
    n_nodes = features.shape[0]
    start_time = time.time()
    
    print(f"  [1/4] Normalizing features...", end=" ", flush=True)
    t0 = time.time()
    if metric == 'cosine':
        features_norm = normalize(features, norm='l2', axis=1)
    else:
        features_norm = features
    print(f"({time.time()-t0:.1f}s)")
    
    print(f"  [2/4] Finding k={k} neighbors for {n_nodes:,} nodes...", end=" ", flush=True)
    t0 = time.time()
    nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric, n_jobs=-1)
    nbrs.fit(features_norm)
    distances, indices = nbrs.kneighbors(features_norm)
    print(f"({time.time()-t0:.1f}s)")
    
    print(f"  [3/4] Building edge list...", end=" ", flush=True)
    t0 = time.time()
    row_indices = np.repeat(np.arange(n_nodes), k)
    col_indices = indices[:, 1:].flatten()
    
    edges_src = np.stack([row_indices, col_indices], axis=0)
    edges_dst = np.stack([col_indices, row_indices], axis=0)
    edges = np.concatenate([edges_src, edges_dst], axis=1)
    
    if add_self_loops:
        self_loops = np.stack([np.arange(n_nodes), np.arange(n_nodes)], axis=0)
        edges = np.concatenate([edges, self_loops], axis=1)
    print(f"({time.time()-t0:.1f}s)")
    
    print(f"  [4/4] Removing duplicates...", end=" ", flush=True)
    t0 = time.time()
    edges = np.unique(edges, axis=1)
    edge_index = torch.from_numpy(edges).long()
    print(f"({time.time()-t0:.1f}s)")
    
    total_time = time.time() - start_time
    print(f"  Graph built: {n_nodes:,} nodes, {edge_index.shape[1]:,} edges (total: {total_time:.1f}s)")
    
    return edge_index


def create_graph_data(features, labels, k_neighbors=10):
    """
    Create PyTorch Geometric Data object from features and labels
    
    Args:
        features: numpy array (n_nodes, n_features)
        labels: numpy array (n_nodes,)
        k_neighbors: number of nearest neighbors for graph construction
    
    Returns:
        PyTorch Geometric Data object
    """
    edge_index = build_knn_graph(features, k=k_neighbors)
    
    graph = Data(
        x=torch.FloatTensor(features),
        edge_index=edge_index,
        y=torch.LongTensor(labels)
    )
    
    return graph

