import numpy as np
import torch
from sklearn.metrics import roc_curve, confusion_matrix


def compute_ndcg(anomaly_scores, y_true):
    """
    Compute nDCG (normalized Discounted Cumulative Gain)
    
    Args:
        anomaly_scores: numpy array of anomaly scores
        y_true: numpy array of true labels (0=normal, 1=attack)
    
    Returns:
        ndcg: nDCG score
        metrics: dict with additional metrics
    """
    ranked_indices = np.argsort(anomaly_scores)[::-1]
    ranked_labels = y_true[ranked_indices]
    
    true_positive_positions = np.where(ranked_labels == 1)[0] + 1
    
    num_attacks = (y_true == 1).sum()
    if num_attacks == 0:
        return 0.0, {}
    
    dcg = sum([1.0 / np.log2(i + 1) for i in true_positive_positions])
    maxdcg = sum([1.0 / np.log2(i + 1) for i in range(1, num_attacks + 1)])
    ndcg = dcg / maxdcg if maxdcg > 0 else 0.0
    
    metrics = {
        'ndcg': ndcg,
        'num_attacks': int(num_attacks),
        'best_position': int(true_positive_positions.min()),
        'worst_position': int(true_positive_positions.max()),
        'median_position': float(np.median(true_positive_positions)),
        'attack_positions': true_positive_positions.tolist()
    }
    
    return ndcg, metrics


def compute_optimal_threshold(anomaly_scores, y_true):
    """
    Find optimal threshold using Youden's J statistic
    
    Args:
        anomaly_scores: numpy array of anomaly scores
        y_true: numpy array of true labels
    
    Returns:
        threshold: optimal threshold
        metrics: dict with TPR, FPR, and confusion matrix
    """
    fpr, tpr, thresholds = roc_curve(y_true, anomaly_scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred = (anomaly_scores > optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'threshold': float(optimal_threshold),
        'tpr': float(tpr[optimal_idx]),
        'fpr': float(fpr[optimal_idx]),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }
    
    return optimal_threshold, metrics


def evaluate_gae(model, graph, compute_threshold=True):
    """
    Complete evaluation of GAE model
    
    Args:
        model: trained GAE model
        graph: PyTorch Geometric Data object
        compute_threshold: whether to compute optimal threshold
    
    Returns:
        results: dict with all evaluation metrics
    """
    model.eval()
    
    with torch.no_grad():
        z = model.encode(graph.x, graph.edge_index)
        anomaly_scores = model.compute_anomaly_scores(z, graph.edge_index)
    
    anomaly_scores_np = anomaly_scores.cpu().numpy()
    y_true = graph.y.cpu().numpy()
    
    ndcg, ndcg_metrics = compute_ndcg(anomaly_scores_np, y_true)
    
    results = {
        'ndcg': ndcg,
        **ndcg_metrics
    }
    
    if compute_threshold:
        threshold, threshold_metrics = compute_optimal_threshold(anomaly_scores_np, y_true)
        results.update(threshold_metrics)
    
    return results

