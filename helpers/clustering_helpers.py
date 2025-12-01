"""
Helper functions for clustering analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_metric_vs_k(k_values, scores, metric_name, title=None, higher_is_better=True):
    """
    Plot a metric vs number of clusters.

    Args:
        k_values: list. Number of clusters tested.
        scores: list. Metric scores corresponding to k_values.
        metric_name: str. Name of the metric for y-axis label.
        title: str. Plot title (optional).
        higher_is_better: bool. Whether higher metric values are better.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, scores, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel(metric_name)

    if title is None:
        direction = "higher is better" if higher_is_better else "lower is better"
        title = f'{metric_name} vs k ({direction})'
    plt.title(title)

    plt.grid(True)
    plt.xticks(k_values)
    plt.tight_layout()
    plt.show()


def plot_multiple_metrics(k_values, results_dict, metrics=None):
    """
    Plot multiple metrics in subplots.

    Args:
        k_values: list. Number of clusters tested.
        results_dict: dict. Results keyed by k, each containing metric scores.
        metrics: list. Metrics to plot.
    """
    if metrics is None:
        metrics = ['silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score']
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))

    if n_metrics == 1:
        axes = [axes]

    metric_names = {
        'silhouette_score': 'Silhouette',
        'davies_bouldin_score': 'Davies-Bouldin',
        'calinski_harabasz_score': 'Calinski-Harabasz'
    }

    for idx, metric in enumerate(metrics):
        scores = [results_dict[k]['metrics'][metric] for k in k_values]
        axes[idx].plot(k_values, scores, 'o-', linewidth=2, markersize=8)
        axes[idx].set_xlabel('k')
        axes[idx].set_ylabel(metric_names.get(metric, metric))
        axes[idx].set_title(f'{metric_names.get(metric, metric)} vs k')
        axes[idx].grid(True)
        axes[idx].set_xticks(k_values)

    plt.tight_layout()
    plt.show()


def create_cluster_summary(data, labels, feature_names=None):
    """
    Create a summary DataFrame with cluster statistics.

    Args:
        data: numpy array or DataFrame. Feature data.
        labels: array. Cluster labels.
        feature_names: list. Feature names (if data is numpy array).

    Returns:
        summary: DataFrame with cluster means, sizes, and stds.
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(data.shape[1])]
        df = pd.DataFrame(data, columns=feature_names)

    df['Cluster'] = labels

    cluster_summary = df.groupby('Cluster').agg(['mean', 'std', 'count'])

    return cluster_summary


def compute_cluster_sizes(labels):
    """
    Compute cluster sizes.

    Args:
        labels: array. Cluster labels.

    Returns:
        sizes: dict. Cluster ID -> count.
    """
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))


def filter_noise_points(data, labels, noise_label=-1):
    """
    Filter out noise points from data and labels.

    Args:
        data: numpy array. Feature data.
        labels: array. Cluster labels (with noise label for noise points).
        noise_label: int. Label used for noise points.

    Returns:
        data_clean: Data without noise points.
        labels_clean: Labels without noise.
    """
    mask = labels != noise_label
    data_clean = data[mask]
    labels_clean = labels[mask]

    return data_clean, labels_clean


def export_cluster_labels(subject_ids, labels, output_path='cluster_labels.csv'):
    """
    Export cluster labels to CSV.

    Args:
        subject_ids: array. Subject identifiers.
        labels: array. Cluster labels.
        output_path: str. Output file path.
    """
    df = pd.DataFrame({
        'subject_id': subject_ids,
        'cluster': labels
    })
    df.to_csv(output_path, index=False)
    print(f"Cluster labels exported to {output_path}")


def compare_clustering_results(results_dict, metric='silhouette'):
    """
    Compare clustering results from different algorithms/parameters.

    Args:
        results_dict: dict. Keys are algorithm names, values are dicts with 'labels' and 'metrics'.
        metric: str. Metric to compare ('silhouette', 'davies_bouldin', 'calinski_harabasz').

    Returns:
        comparison: DataFrame with comparison results.
    """
    data = []
    for name, result in results_dict.items():
        if 'metrics' in result:
            score = result['metrics'].get(metric, np.nan)
        else:
            score = result.get(metric, np.nan)

        n_clusters = len(np.unique(result['labels']))
        if -1 in result['labels']:
            n_clusters -= 1
            n_noise = list(result['labels']).count(-1)
        else:
            n_noise = 0

        data.append({
            'Algorithm': name,
            metric: score,
            'n_clusters': n_clusters,
            'n_noise': n_noise
        })

    comparison = pd.DataFrame(data)
    return comparison.sort_values(by=metric, ascending=(metric == 'davies_bouldin'))

