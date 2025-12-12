"""Utility functions for rare pattern mining and integration with GAE."""

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import KBinsDiscretizer
from typing import List, Dict, Set, Tuple
import warnings


def prepare_transactions(features, feature_names=None, n_bins=5, strategy='quantile'):
    """
    Convert continuous features to transactions for rare pattern mining
    
    Args:
        features: numpy array (n_samples, n_features) - continuous features
        feature_names: list of feature names (optional)
        n_bins: number of bins for discretization
        strategy: 'quantile', 'uniform', or 'kmeans'
    
    Returns:
        transactions: list of sets, each set contains items like "feature_0_bin_2"
        item_to_idx: dict mapping item strings to indices
    """
    n_samples, n_features = features.shape
    
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(n_features)]
    
    print(f"Preparing transactions from {n_samples:,} samples, {n_features} features")
    print(f"   Discretization: {n_bins} bins, strategy={strategy}")
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        discretizer = KBinsDiscretizer(
            n_bins=n_bins, 
            encode='ordinal', 
            strategy=strategy,
            subsample=None
        )
        features_discrete = discretizer.fit_transform(features).astype(int)
    
    # Convert to transactions (each sample = one transaction)
    transactions = []
    item_set = set()
    
    for i in range(n_samples):
        transaction = set()
        for j in range(n_features):
            bin_id = features_discrete[i, j]
            # Create item name: "featureName_binX"
            item = f"{feature_names[j]}_bin{bin_id}"
            transaction.add(item)
            item_set.add(item)
        
        transactions.append(transaction)
    
    # Create item to index mapping
    item_to_idx = {item: idx for idx, item in enumerate(sorted(item_set))}
    
    print(f"Created {len(transactions):,} transactions")
    print(f"   Unique items: {len(item_to_idx):,}")
    print(f"   Avg items per transaction: {np.mean([len(t) for t in transactions]):.1f}")
    
    return transactions, item_to_idx, discretizer


def transactions_to_binary_matrix(transactions, item_to_idx):
    """
    Convert transactions to binary matrix for efficient processing
    
    Args:
        transactions: list of sets
        item_to_idx: dict mapping items to indices
    
    Returns:
        binary_matrix: numpy array (n_transactions, n_items)
    """
    n_transactions = len(transactions)
    n_items = len(item_to_idx)
    
    binary_matrix = np.zeros((n_transactions, n_items), dtype=bool)
    
    for i, transaction in enumerate(transactions):
        for item in transaction:
            if item in item_to_idx:
                binary_matrix[i, item_to_idx[item]] = True
    
    return binary_matrix


def build_rare_graph_fully_connected(rare_itemsets, transactions, node_features, labels):
    """
    Build rare graph by fully connecting processes that share rare patterns.
    
    Args:
        rare_itemsets: List of (itemset, support) tuples
        transactions: List of sets
        node_features: Torch tensor of node features
        labels: Torch tensor of labels
    
    Returns:
        rare_graph: PyTorch Geometric Data object
    """
    n_nodes = len(transactions)
    edges = []
    
    print(f"Building rare graph (fully connected mode)")
    print(f"   Processing {len(rare_itemsets):,} rare patterns...")
    
    for itemset, support in rare_itemsets:
        matching_nodes = []
        for node_id, transaction in enumerate(transactions):
            if itemset.issubset(transaction):
                matching_nodes.append(node_id)
        
        for i in range(len(matching_nodes)):
            for j in range(i + 1, len(matching_nodes)):
                edges.append([matching_nodes[i], matching_nodes[j]])
                edges.append([matching_nodes[j], matching_nodes[i]])
    
    if len(edges) == 0:
        print(f"Warning: No edges created! Try lower support threshold")
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edges = list(set(map(tuple, edges)))
        edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    rare_graph = Data(
        x=node_features,
        edge_index=edge_index,
        y=labels
    )
    
    print(f"Rare graph created:")
    print(f"   Nodes: {n_nodes:,}")
    print(f"   Edges: {rare_graph.num_edges:,}")
    
    return rare_graph


def build_rare_graph_virtual_nodes(rare_itemsets, transactions, node_features, labels):
    """
    Build rare graph with virtual rule nodes.
    
    Each rare pattern becomes a virtual node connected to matching processes.
    
    Args:
        rare_itemsets: List of (itemset, support) tuples
        transactions: List of sets
        node_features: Torch tensor of node features
        labels: Torch tensor of labels
    
    Returns:
        rare_graph: PyTorch Geometric Data object with virtual nodes
    """
    n_real_nodes = len(transactions)
    n_virtual_nodes = len(rare_itemsets)
    n_total_nodes = n_real_nodes + n_virtual_nodes
    
    print(f"Building rare graph (virtual nodes mode)")
    print(f"   Real nodes: {n_real_nodes:,}")
    print(f"   Virtual rule nodes: {n_virtual_nodes:,}")
    
    edges = []
    
    for rule_id, (itemset, support) in enumerate(rare_itemsets):
        virtual_node_id = n_real_nodes + rule_id
        
        for node_id, transaction in enumerate(transactions):
            if itemset.issubset(transaction):
                edges.append([node_id, virtual_node_id])
                edges.append([virtual_node_id, node_id])
    
    if len(edges) == 0:
        print(f"Warning: No edges created! Try lower support threshold")
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    virtual_node_features = torch.zeros((n_virtual_nodes, node_features.shape[1]))
    virtual_node_labels = torch.zeros(n_virtual_nodes, dtype=labels.dtype)
    
    all_features = torch.cat([node_features, virtual_node_features], dim=0)
    all_labels = torch.cat([labels, virtual_node_labels], dim=0)
    
    rare_graph = Data(
        x=all_features,
        edge_index=edge_index,
        y=all_labels,
        num_real_nodes=n_real_nodes,
        num_virtual_nodes=n_virtual_nodes
    )
    
    print(f"Rare graph created:")
    print(f"   Total nodes: {n_total_nodes:,} (real + virtual)")
    print(f"   Edges: {rare_graph.num_edges:,}")
    
    return rare_graph


def compute_rare_pattern_bonus(node_id, rare_graph, method='degree'):
    """
    Compute bonus score for a node based on rare graph connectivity.
    
    Args:
        node_id: Node index
        rare_graph: PyTorch Geometric Data object
        method: 'degree' or 'binary'
    
    Returns:
        bonus: Float score
    """
    edge_index = rare_graph.edge_index
    degree = (edge_index[0] == node_id).sum().item()
    
    if method == 'binary':
        return 1.0 if degree > 0 else 0.0
    elif method == 'degree':
        return float(degree)
    else:
        raise ValueError(f"Unknown method: {method}")


def boost_anomaly_scores(base_scores, rare_graph, alpha=1.0, method='degree'):
    """
    Boost GAE anomaly scores using rare pattern information.
    
    Args:
        base_scores: Torch tensor of GAE anomaly scores
        rare_graph: PyTorch Geometric Data object
        alpha: Boosting weight
        method: 'degree' or 'binary'
    
    Returns:
        boosted_scores: Torch tensor
    """
    n_nodes = len(base_scores)
    bonuses = torch.zeros(n_nodes, device=base_scores.device)
    
    for node_id in range(n_nodes):
        bonuses[node_id] = compute_rare_pattern_bonus(node_id, rare_graph, method)
    
    if bonuses.max() > 0:
        bonuses = bonuses / bonuses.max()
    
    boosted_scores = base_scores + alpha * bonuses
    
    print(f"Scores boosted with α={alpha}")
    print(f"   Nodes with rare patterns: {(bonuses > 0).sum().item():,}")
    print(f"   Avg bonus: {bonuses.mean():.4f}")
    
    return boosted_scores


def merge_graphs(base_graph, rare_graph, remove_duplicates=True):
    """
    Merge base k-NN graph with rare pattern graph.
    
    Args:
        base_graph: PyTorch Geometric Data (k-NN graph)
        rare_graph: PyTorch Geometric Data (rare pattern graph)
        remove_duplicates: Whether to remove duplicate edges
    
    Returns:
        merged_graph: PyTorch Geometric Data
    """
    print(f"Merging graphs...")
    print(f"   Base graph edges: {base_graph.num_edges:,}")
    print(f"   Rare graph edges: {rare_graph.num_edges:,}")
    
    merged_edges = torch.cat([base_graph.edge_index, rare_graph.edge_index], dim=1)
    
    if remove_duplicates:
        merged_edges = torch.unique(merged_edges, dim=1)
    
    merged_graph = Data(
        x=base_graph.x,
        edge_index=merged_edges,
        y=base_graph.y
    )
    
    print(f"Merged graph created:")
    print(f"   Nodes: {merged_graph.num_nodes:,}")
    print(f"   Edges: {merged_graph.num_edges:,}")
    print(f"   New edges added: {merged_graph.num_edges - base_graph.num_edges:,}")
    
    return merged_graph


def analyze_rare_pattern_distribution(transactions, rare_itemsets, labels):
    """
    Analyze distribution of rare patterns in normal vs attack processes.
    
    Args:
        transactions: List of sets
        rare_itemsets: List of (itemset, support) tuples
        labels: Numpy array of labels (0=normal, 1=attack)
    
    Returns:
        stats: Dict with statistics
    """
    n_rare_patterns_per_node = []
    
    for transaction in transactions:
        count = sum(1 for itemset, _ in rare_itemsets if itemset.issubset(transaction))
        n_rare_patterns_per_node.append(count)
    
    n_rare_patterns_per_node = np.array(n_rare_patterns_per_node)
    
    normal_mask = labels == 0
    attack_mask = labels == 1
    
    stats = {
        'avg_rare_patterns_normal': n_rare_patterns_per_node[normal_mask].mean(),
        'avg_rare_patterns_attack': n_rare_patterns_per_node[attack_mask].mean(),
        'std_rare_patterns_normal': n_rare_patterns_per_node[normal_mask].std(),
        'std_rare_patterns_attack': n_rare_patterns_per_node[attack_mask].std(),
        'pct_normal_with_rare': (n_rare_patterns_per_node[normal_mask] > 0).mean() * 100,
        'pct_attack_with_rare': (n_rare_patterns_per_node[attack_mask] > 0).mean() * 100,
    }
    
    print(f"\nRare Pattern Distribution Analysis:")
    print(f"   Normal processes:")
    print(f"      Avg rare patterns: {stats['avg_rare_patterns_normal']:.2f} ± {stats['std_rare_patterns_normal']:.2f}")
    print(f"      % with rare patterns: {stats['pct_normal_with_rare']:.1f}%")
    print(f"   Attack processes:")
    print(f"      Avg rare patterns: {stats['avg_rare_patterns_attack']:.2f} ± {stats['std_rare_patterns_attack']:.2f}")
    print(f"      % with rare patterns: {stats['pct_attack_with_rare']:.1f}%")
    
    if stats['avg_rare_patterns_attack'] > stats['avg_rare_patterns_normal']:
        print(f"   Attacks have MORE rare patterns (good signal!)")
    else:
        print(f"   Warning: Attacks have FEWER rare patterns (may need different threshold)")
    
    return stats
