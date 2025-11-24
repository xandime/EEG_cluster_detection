"""
Internal validation metrics for clustering quality assessment.

These metrics evaluate cluster quality using only the data structure,
without requiring external labels.
"""

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score
)


def compute_silhouette_score(data, labels):
    """Compute average silhouette score for all samples.

    Range: [-1, 1]
    Interpretation: Closer to 1 = better separated clusters

    Args:
        data: numpy array of shape (N, d). Feature data.
        labels: numpy array of shape (N,). Cluster assignments.

    Returns:
        float: Average silhouette score.
    """
    if len(np.unique(labels)) < 2:
        return np.nan
    return silhouette_score(data, labels)


def compute_silhouette_samples(data, labels):
    """Compute silhouette score for each sample.

    Args:
        data: numpy array of shape (N, d). Feature data.
        labels: numpy array of shape (N,). Cluster assignments.

    Returns:
        numpy array of shape (N,): Silhouette score per sample.
    """
    if len(np.unique(labels)) < 2:
        return np.full(len(labels), np.nan)
    return silhouette_samples(data, labels)


def compute_davies_bouldin_score(data, labels):
    """Compute Davies-Bouldin index.

    Range: [0, ∞)
    Interpretation: Lower = better separation

    Args:
        data: numpy array of shape (N, d). Feature data.
        labels: numpy array of shape (N,). Cluster assignments.

    Returns:
        float: Davies-Bouldin index.
    """
    if len(np.unique(labels)) < 2:
        return np.nan
    return davies_bouldin_score(data, labels)


def compute_calinski_harabasz_score(data, labels):
    """Compute Calinski-Harabasz score (Variance Ratio Criterion).

    Range: [0, ∞)
    Interpretation: Higher = better defined clusters

    Args:
        data: numpy array of shape (N, d). Feature data.
        labels: numpy array of shape (N,). Cluster assignments.

    Returns:
        float: Calinski-Harabasz score.
    """
    if len(np.unique(labels)) < 2:
        return np.nan
    return calinski_harabasz_score(data, labels)


def compute_dunn_index(data, labels):
    """Compute Dunn index.

    Range: [0, ∞)
    Interpretation: Higher = better (min inter-cluster / max intra-cluster distance)
    Note: Computationally expensive for large datasets.

    Args:
        data: numpy array of shape (N, d). Feature data.
        labels: numpy array of shape (N,). Cluster assignments.

    Returns:
        float: Dunn index.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return np.nan

    # Compute cluster centers
    centers = np.array([data[labels == k].mean(axis=0) for k in unique_labels])

    # Min inter-cluster distance
    min_inter = np.inf
    for i, k1 in enumerate(unique_labels):
        for k2 in unique_labels[i+1:]:
            dist = np.linalg.norm(centers[i] - centers[np.where(unique_labels == k2)[0][0]])
            min_inter = min(min_inter, dist)

    # Max intra-cluster distance
    max_intra = 0
    for k in unique_labels:
        cluster_data = data[labels == k]
        if len(cluster_data) > 1:
            center = cluster_data.mean(axis=0)
            distances = np.linalg.norm(cluster_data - center, axis=1)
            max_intra = max(max_intra, distances.max())

    if max_intra == 0:
        return np.nan

    return min_inter / max_intra


def compute_within_cluster_variance(data, labels):
    """Compute within-cluster variance (inertia).


    Lower values indicate more compact (cohesive) clusters
    equals total inertia (sum_k sum_{x in C_k} ||x - mu_k||^2), and is sensitive to feature scaling and cluster count.

    Args:
        data: numpy array of shape (N, d). Feature data.
        labels: numpy array of shape (N,). Cluster assignments.

    Returns:
        float: Total within-cluster variance.
    """
    variance = 0.0
    for k in np.unique(labels):
        cluster_data = data[labels == k]
        center = cluster_data.mean(axis=0)
        variance += np.sum((cluster_data - center) ** 2)
    return variance


def compute_cluster_compactness(data, labels):
    """Compute average compactness per cluster.

    Compactness = average distance of points to their cluster center.
    Relevant for EEG data interpretation.

    Args:
        data: numpy array of shape (N, d). Feature data.
        labels: numpy array of shape (N,). Cluster assignments.

    Returns:
        dict: Compactness per cluster.
    """
    compactness = {}
    for k in np.unique(labels):
        cluster_data = data[labels == k]
        center = cluster_data.mean(axis=0)
        distances = np.linalg.norm(cluster_data - center, axis=1)
        compactness[k] = distances.mean()
    return compactness


def compute_all_metrics(data, labels, include_dunn=False):
    """Compute all internal validation metrics.

    Args:
        data: numpy array of shape (N, d). Feature data.
        labels: numpy array of shape (N,). Cluster assignments.
        include_dunn: bool. Whether to compute Dunn index (expensive).

    Returns:
        dict: All computed metrics.
    """
    metrics = {
        'silhouette_score': compute_silhouette_score(data, labels),
        'davies_bouldin_score': compute_davies_bouldin_score(data, labels),
        'calinski_harabasz_score': compute_calinski_harabasz_score(data, labels),
        'within_cluster_variance': compute_within_cluster_variance(data, labels),
        'cluster_compactness': compute_cluster_compactness(data, labels),
    }

    if include_dunn:
        metrics['dunn_index'] = compute_dunn_index(data, labels)

    return metrics


def print_metrics(metrics):
    """Print metrics in a readable format.

    Args:
        metrics: dict. Output from compute_all_metrics().
    """
    print("Internal Validation Metrics:")
    print("-" * 50)

    if 'silhouette_score' in metrics:
        print(f"  Silhouette Score:         {metrics['silhouette_score']:.4f}")
        print(f"    Range: [-1, 1], Higher is better")

    if 'davies_bouldin_score' in metrics:
        print(f"  Davies-Bouldin Index:     {metrics['davies_bouldin_score']:.4f}")
        print(f"    Range: [0, ∞), Lower is better")

    if 'calinski_harabasz_score' in metrics:
        print(f"  Calinski-Harabasz Score:  {metrics['calinski_harabasz_score']:.4f}")
        print(f"    Range: [0, ∞), Higher is better")

    if 'dunn_index' in metrics:
        print(f"  Dunn Index:               {metrics['dunn_index']:.4f}")
        print(f"    Range: [0, ∞), Higher is better")

    if 'within_cluster_variance' in metrics:
        print(f"  Within-Cluster Variance:  {metrics['within_cluster_variance']:.4f}")
        print(f"    Lower is better")

    if 'cluster_compactness' in metrics:
        print(f"  Cluster Compactness:")
        for k, v in metrics['cluster_compactness'].items():
            print(f"    Cluster {k}: {v:.4f}")

