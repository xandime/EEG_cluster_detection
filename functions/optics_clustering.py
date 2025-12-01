"""
OPTICS (Ordering Points To Identify the Clustering Structure) clustering implementation
for EEG participant subtyping.

OPTICS is a density-based clustering algorithm that doesn't require specifying the number
of clusters beforehand. It creates a reachability plot that shows the density-based
clustering structure of the data.
"""

import numpy as np
from sklearn.cluster import OPTICS
from metrics.internal_metrics import (
    compute_silhouette_score,
    compute_davies_bouldin_score,
    compute_calinski_harabasz_score
)


def optics_clustering(data, min_samples=5, max_eps=np.inf, metric='euclidean',
                     cluster_method='xi', xi=0.05, min_cluster_size=None,
                     algorithm='auto', leaf_size=30, n_jobs=None):
    """
    Perform OPTICS clustering.

    Args:
        data: numpy array of shape (N, d). Data to cluster.
        min_samples: int. Minimum number of samples in a neighborhood for a point
                    to be considered a core point.
        max_eps: float. Maximum distance between two samples for one to be considered
                as in the neighborhood of the other. Default is infinity.
        metric: str. Metric to use for distance computation (e.g., 'euclidean',
               'manhattan', 'cosine').
        cluster_method: str. Method to extract clusters ('xi' or 'dbscan').
        xi: float. Determines the minimum steepness on the reachability plot that
           constitutes a cluster boundary. Used if cluster_method='xi'.
        min_cluster_size: int or float. Minimum number of samples in a cluster.
                         If None, defaults to min_samples.
        algorithm: str. Algorithm used to compute nearest neighbors ('auto', 'ball_tree',
                  'kd_tree', 'brute').
        leaf_size: int. Leaf size for tree algorithms.
        n_jobs: int. Number of parallel jobs (-1 for all processors).

    Returns:
        dict with keys:
            - 'labels': Cluster labels (-1 for noise points).
            - 'reachability': Reachability distances per sample.
            - 'ordering': Indices of points in the order they were processed.
            - 'core_distances': Distance to the min_samples-th nearest neighbor.
            - 'predecessor': Index of the predecessor of each sample.
            - 'optics_model': Fitted OPTICS object.
            - 'n_clusters': Number of clusters found (excluding noise).
    """
    if min_cluster_size is None:
        min_cluster_size = min_samples

    optics = OPTICS(
        min_samples=min_samples,
        max_eps=max_eps,
        metric=metric,
        cluster_method=cluster_method,
        xi=xi,
        min_cluster_size=min_cluster_size,
        algorithm=algorithm,
        leaf_size=leaf_size,
        n_jobs=n_jobs
    )

    optics.fit(data)
    labels = optics.labels_

    # Count number of clusters (excluding noise labeled as -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    return {
        'labels': labels,
        'reachability': optics.reachability_,
        'ordering': optics.ordering_,
        'core_distances': optics.core_distances_,
        'predecessor': optics.predecessor_,
        'optics_model': optics,
        'n_clusters': n_clusters
    }


def tune_optics_min_samples(data, min_samples_range, max_eps=np.inf,
                           metric='euclidean', cluster_method='xi',
                           xi=0.05, verbose=False):
    """
    Tune the min_samples parameter for OPTICS clustering using internal metrics.

    Args:
        data: numpy array of shape (N, d). Data to cluster.
        min_samples_range: list of int. Range of min_samples values to test.
        max_eps: float. Maximum epsilon value.
        metric: str. Distance metric.
        cluster_method: str. Cluster extraction method.
        xi: float. Steepness threshold for cluster extraction.
        verbose: bool. Whether to print results for each configuration.

    Returns:
        best_min_samples: int. Optimal min_samples value.
        results: dict. Results for each min_samples value tested.
    """
    results = {}
    best_score = -np.inf
    best_min_samples = min_samples_range[0]

    for min_samples in min_samples_range:
        result = optics_clustering(
            data,
            min_samples=min_samples,
            max_eps=max_eps,
            metric=metric,
            cluster_method=cluster_method,
            xi=xi
        )

        labels = result['labels']
        n_clusters = result['n_clusters']

        # Only compute metrics if we have more than 1 cluster and not all points are noise
        if n_clusters > 1 and n_clusters < len(labels):
            # Filter out noise points for metric computation
            non_noise_mask = labels != -1
            if np.sum(non_noise_mask) > 0:
                filtered_data = data[non_noise_mask]
                filtered_labels = labels[non_noise_mask]

                metrics = {
                    'silhouette': compute_silhouette_score(filtered_data, filtered_labels),
                    'davies_bouldin': compute_davies_bouldin_score(filtered_data, filtered_labels),
                    'calinski_harabasz': compute_calinski_harabasz_score(filtered_data, filtered_labels),
                }
            else:
                metrics = {
                    'silhouette': -1,
                    'davies_bouldin': np.inf,
                    'calinski_harabasz': 0,
                }
        else:
            metrics = {
                'silhouette': -1,
                'davies_bouldin': np.inf,
                'calinski_harabasz': 0,
            }

        results[min_samples] = {
            'labels': labels,
            'n_clusters': n_clusters,
            'metrics': metrics
        }

        # Use silhouette score as the primary metric for selection
        if metrics['silhouette'] > best_score:
            best_score = metrics['silhouette']
            best_min_samples = min_samples

        if verbose:
            print(f"min_samples={min_samples}: "
                  f"n_clusters={n_clusters}, "
                  f"Silhouette={metrics['silhouette']:.4f}, "
                  f"Davies-Bouldin={metrics['davies_bouldin']:.4f}, "
                  f"Calinski-Harabasz={metrics['calinski_harabasz']:.2f}")

    return best_min_samples, results


def tune_optics_xi(data, xi_range, min_samples=5, max_eps=np.inf,
                  metric='euclidean', verbose=False):
    """
    Tune the xi parameter for OPTICS clustering.

    Args:
        data: numpy array of shape (N, d). Data to cluster.
        xi_range: list of float. Range of xi values to test (typically between 0.01 and 0.1).
        min_samples: int. Minimum samples parameter.
        max_eps: float. Maximum epsilon value.
        metric: str. Distance metric.
        verbose: bool. Whether to print results for each configuration.

    Returns:
        best_xi: float. Optimal xi value.
        results: dict. Results for each xi value tested.
    """
    results = {}
    best_score = -np.inf
    best_xi = xi_range[0]

    for xi in xi_range:
        result = optics_clustering(
            data,
            min_samples=min_samples,
            max_eps=max_eps,
            metric=metric,
            cluster_method='xi',
            xi=xi
        )

        labels = result['labels']
        n_clusters = result['n_clusters']

        # Only compute metrics if we have valid clustering
        if n_clusters > 1 and n_clusters < len(labels):
            non_noise_mask = labels != -1
            if np.sum(non_noise_mask) > 0:
                filtered_data = data[non_noise_mask]
                filtered_labels = labels[non_noise_mask]

                metrics = {
                    'silhouette': compute_silhouette_score(filtered_data, filtered_labels),
                    'davies_bouldin': compute_davies_bouldin_score(filtered_data, filtered_labels),
                    'calinski_harabasz': compute_calinski_harabasz_score(filtered_data, filtered_labels),
                }
            else:
                metrics = {
                    'silhouette': -1,
                    'davies_bouldin': np.inf,
                    'calinski_harabasz': 0,
                }
        else:
            metrics = {
                'silhouette': -1,
                'davies_bouldin': np.inf,
                'calinski_harabasz': 0,
            }

        results[xi] = {
            'labels': labels,
            'n_clusters': n_clusters,
            'metrics': metrics
        }

        if metrics['silhouette'] > best_score:
            best_score = metrics['silhouette']
            best_xi = xi

        if verbose:
            print(f"xi={xi:.3f}: "
                  f"n_clusters={n_clusters}, "
                  f"Silhouette={metrics['silhouette']:.4f}")

    return best_xi, results

