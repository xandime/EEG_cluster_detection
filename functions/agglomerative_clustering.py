"""
Agglomerative (Hierarchical) Clustering implementation for EEG participant subtyping.

Agglomerative clustering is a bottom-up hierarchical clustering method that starts with
each data point as a separate cluster and progressively merges the closest pairs of
clusters until the desired number of clusters is reached.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from metrics.internal_metrics import (
    compute_silhouette_score,
    compute_davies_bouldin_score,
    compute_calinski_harabasz_score
)


def agglomerative_clustering(data, n_clusters=2, metric='euclidean',
                            linkage_method='ward', compute_full_tree=False):
    """
    Perform Agglomerative (Hierarchical) Clustering.

    Args:
        data: numpy array of shape (N, d). Data to cluster.
        n_clusters: int. Number of clusters to find.
        metric: str. Distance metric to use. Options include:
               - 'euclidean': Euclidean distance
               - 'manhattan': Manhattan (L1) distance
               - 'cosine': Cosine distance
               - 'l1', 'l2', 'cityblock', etc.
        linkage_method: str. Linkage criterion to use. Options:
                       - 'ward': Minimizes variance of clusters (only for euclidean)
                       - 'complete': Maximum distance between clusters
                       - 'average': Average distance between all pairs
                       - 'single': Minimum distance between clusters
        compute_full_tree: bool or str. Whether to compute the full dendrogram tree.
                          Can be 'auto' to automatically determine.

    Returns:
        dict with keys:
            - 'labels': Cluster labels for each sample.
            - 'n_clusters': Number of clusters.
            - 'model': Fitted AgglomerativeClustering object.
            - 'n_leaves': Number of leaves in the hierarchical tree.
            - 'n_components': Number of connected components.
            - 'children': Merge information (if compute_full_tree=True).
    """
    # Check if ward linkage is used with non-euclidean metric
    if linkage_method == 'ward' and metric != 'euclidean':
        raise ValueError("Ward linkage requires euclidean metric. "
                        "Use linkage_method='average', 'complete', or 'single' "
                        "for other metrics.")

    # For ward linkage, we must use euclidean (sklearn will handle this)
    # For other linkage methods, we can specify the metric
    if linkage_method == 'ward':
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method,
            compute_full_tree=compute_full_tree
        )
    else:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=metric,
            linkage=linkage_method,
            compute_full_tree=compute_full_tree
        )

    labels = clustering.fit_predict(data)

    result = {
        'labels': labels,
        'n_clusters': n_clusters,
        'model': clustering,
        'n_leaves': clustering.n_leaves_,
        'n_components': clustering.n_connected_components_,
    }

    # Add children information if full tree was computed
    if compute_full_tree:
        result['children'] = clustering.children_

    return result


def select_optimal_n_agglomerative(data, n_range, metric='euclidean',
                                  linkage_method='ward',
                                  selection_metric='silhouette',
                                  verbose=False):
    """
    Select optimal number of clusters for agglomerative clustering using internal metrics.

    Args:
        data: numpy array of shape (N, d). Data to cluster.
        n_range: list of int. Range of n_clusters to test.
        metric: str. Distance metric.
        linkage_method: str. Linkage criterion.
        selection_metric: str. Metric to use for selection:
                         'silhouette', 'davies_bouldin', or 'calinski_harabasz'.
        verbose: bool. Whether to print results for each n.

    Returns:
        optimal_n: int. Optimal number of clusters.
        results: dict. Results for each n tested, keyed by n.
    """
    results = {}
    scores = []

    for n in n_range:
        result = agglomerative_clustering(
            data,
            n_clusters=n,
            metric=metric,
            linkage_method=linkage_method
        )

        labels = result['labels']

        # Compute metrics
        metrics = {
            'silhouette': compute_silhouette_score(data, labels),
            'davies_bouldin': compute_davies_bouldin_score(data, labels),
            'calinski_harabasz': compute_calinski_harabasz_score(data, labels),
        }

        results[n] = {
            'labels': labels,
            'metrics': metrics,
            'model': result['model']
        }

        # Get score based on selection metric
        if selection_metric == 'silhouette':
            score = metrics['silhouette']
        elif selection_metric == 'davies_bouldin':
            score = metrics['davies_bouldin']
        elif selection_metric == 'calinski_harabasz':
            score = metrics['calinski_harabasz']
        else:
            raise ValueError(f"Unknown selection metric: {selection_metric}")

        scores.append(score)

        if verbose:
            print(f"n={n}: "
                  f"Silhouette={metrics['silhouette']:.4f}, "
                  f"Davies-Bouldin={metrics['davies_bouldin']:.4f}, "
                  f"Calinski-Harabasz={metrics['calinski_harabasz']:.2f}")

    # Select optimal (lower is better for Davies-Bouldin)
    if selection_metric == 'davies_bouldin':
        optimal_n = n_range[np.argmin(scores)]
    else:
        optimal_n = n_range[np.argmax(scores)]

    return optimal_n, results


def compare_linkage_methods(data, n_clusters, metric='euclidean', verbose=False):
    """
    Compare different linkage methods for agglomerative clustering.

    Args:
        data: numpy array. Data to cluster.
        n_clusters: int. Number of clusters.
        metric: str. Distance metric.
        verbose: bool. Whether to print comparison results.

    Returns:
        dict. Results for each linkage method with internal metrics.
    """
    # Ward only works with euclidean
    linkage_methods = ['complete', 'average', 'single']
    if metric == 'euclidean':
        linkage_methods.append('ward')

    results = {}

    for linkage in linkage_methods:
        result = agglomerative_clustering(
            data,
            n_clusters=n_clusters,
            metric=metric,
            linkage_method=linkage
        )

        labels = result['labels']

        metrics = {
            'silhouette': compute_silhouette_score(data, labels),
            'davies_bouldin': compute_davies_bouldin_score(data, labels),
            'calinski_harabasz': compute_calinski_harabasz_score(data, labels),
        }

        results[linkage] = {
            'labels': labels,
            'metrics': metrics,
            'model': result['model']
        }

        if verbose:
            print(f"Linkage={linkage}: "
                  f"Silhouette={metrics['silhouette']:.4f}, "
                  f"Davies-Bouldin={metrics['davies_bouldin']:.4f}, "
                  f"Calinski-Harabasz={metrics['calinski_harabasz']:.2f}")

    return results


def create_dendrogram_data(data, method='ward', metric='euclidean'):
    """
    Create hierarchical clustering dendrogram data using scipy.

    This is useful for visualization purposes. Use scipy's dendrogram function
    to actually plot the results.

    Args:
        data: numpy array of shape (N, d). Data to cluster.
        method: str. Linkage method ('ward', 'complete', 'average', 'single').
        metric: str. Distance metric.

    Returns:
        Z: numpy array. Linkage matrix that can be used with scipy.cluster.hierarchy.dendrogram.
    """
    Z = linkage(data, method=method, metric=metric)
    return Z


def cut_dendrogram(Z, n_clusters=None, height=None):
    """
    Cut a dendrogram to get flat cluster assignments.

    Args:
        Z: numpy array. Linkage matrix from scipy.cluster.hierarchy.linkage.
        n_clusters: int. Number of clusters to form (either this or height must be specified).
        height: float. Height at which to cut the dendrogram.

    Returns:
        labels: numpy array. Cluster labels for each sample.
    """
    from scipy.cluster.hierarchy import fcluster

    if n_clusters is not None:
        labels = fcluster(Z, n_clusters, criterion='maxclust')
    elif height is not None:
        labels = fcluster(Z, height, criterion='distance')
    else:
        raise ValueError("Either n_clusters or height must be specified")

    # Convert to 0-indexed labels
    return labels - 1

