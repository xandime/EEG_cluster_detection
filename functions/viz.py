"""Visualization functions for clustering analysis."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from typing import List, Dict, Optional, Tuple


def get_pca_projections(data: np.ndarray, n_components: int = 6) -> Tuple[np.ndarray, PCA]:
    """Apply PCA to data for visualization purposes."""
    pca = PCA(n_components=min(n_components, data.shape[1]))
    transformed = pca.fit_transform(data)
    return transformed, pca


def plot_3d_clusters(
    data: np.ndarray,
    labels: np.ndarray,
    title: str = "3D Cluster Visualization",
    feature_indices: Tuple[int, int, int] = (0, 1, 2),
    feature_names: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "tab10"
) -> Optional[plt.Figure]:
    """
    Create a single 3D scatter plot of clustered data.

    Args:
        data: Feature data (N, d).
        labels: Cluster assignments.
        title: Plot title.
        feature_indices: Indices of 3 features to plot.
        feature_names: Names for the axes.
        ax: Existing 3D axes to plot on.
        cmap: Colormap name.

    Returns:
        Figure object if ax is None.
    """
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

    i, j, k = feature_indices
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(unique_labels)))

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        label_name = f"Cluster {label}" if label >= 0 else "Noise"
        ax.scatter(
            data[mask, i], data[mask, j], data[mask, k],
            c=[colors[idx]], label=label_name, alpha=0.7, s=30
        )

    if feature_names is not None and len(feature_names) > max(feature_indices):
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])
        ax.set_zlabel(feature_names[k])
    else:
        ax.set_xlabel(f"PC{i+1}")
        ax.set_ylabel(f"PC{j+1}")
        ax.set_zlabel(f"PC{k+1}")

    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)

    return fig


def plot_3d_triple(
    data: np.ndarray,
    labels: np.ndarray,
    algorithm_name: str,
    feature_names: Optional[List[str]] = None,
    projections: List[Tuple[int, int, int]] = None
) -> plt.Figure:
    """
    Create three 3D plots showing different feature projections.

    Args:
        data: Feature data (should have at least 6 dimensions for default projections).
        labels: Cluster assignments.
        algorithm_name: Name of the clustering algorithm.
        feature_names: Optional feature names.
        projections: List of 3 tuples specifying feature indices for each plot.

    Returns:
        Matplotlib figure.
    """
    if projections is None:
        max_dim = data.shape[1]
        projections = [
            (0, 1, 2),
            (min(3, max_dim-1), min(4, max_dim-1), min(5, max_dim-1)),
            (0, min(2, max_dim-1), min(4, max_dim-1))
        ]

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(f"{algorithm_name} - 3D Cluster Projections", fontsize=14, y=1.02)

    for idx, proj in enumerate(projections):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        plot_3d_clusters(
            data, labels,
            title=f"Projection {idx + 1}: PC{proj[0]+1}-{proj[1]+1}-{proj[2]+1}",
            feature_indices=proj,
            feature_names=feature_names,
            ax=ax
        )

    plt.tight_layout()
    return fig


def plot_all_algorithms_3d(
    results: Dict[str, Dict],
    data: np.ndarray,
    feature_names: Optional[List[str]] = None,
    projection: Tuple[int, int, int] = (0, 1, 2)
) -> plt.Figure:
    """
    Create side-by-side 3D comparison of all clustering algorithms.

    Args:
        results: Dictionary of algorithm results with 'labels' key.
        data: Feature data.
        feature_names: Optional feature names.
        projection: Feature indices for the 3D projection.

    Returns:
        Matplotlib figure.
    """
    n_algorithms = len(results)
    cols = min(3, n_algorithms)
    rows = (n_algorithms + cols - 1) // cols

    fig = plt.figure(figsize=(6 * cols, 5 * rows))
    fig.suptitle("Clustering Algorithm Comparison (3D)", fontsize=14, y=1.02)

    for idx, (name, result) in enumerate(results.items()):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        labels = result['labels']
        n_clusters = len(np.unique(labels[labels >= 0]))

        plot_3d_clusters(
            data, labels,
            title=f"{name} (k={n_clusters})",
            feature_indices=projection,
            feature_names=feature_names,
            ax=ax
        )

    plt.tight_layout()
    return fig


def plot_metrics_comparison(
    results: Dict[str, Dict],
    metrics: List[str] = None
) -> plt.Figure:
    """
    Create bar plots comparing metrics across algorithms.

    Args:
        results: Dictionary of algorithm results with 'metrics' key.
        metrics: List of metric names to plot.

    Returns:
        Matplotlib figure.
    """
    if metrics is None:
        metrics = ['silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score']

    algorithms = list(results.keys())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    metric_labels = {
        'silhouette_score': ('Silhouette Score', True),
        'davies_bouldin_score': ('Davies-Bouldin Index', False),
        'calinski_harabasz_score': ('Calinski-Harabasz Score', True),
        'within_cluster_variance': ('Within-Cluster Variance', False),
        'dunn_index': ('Dunn Index', True)
    }

    for ax, metric in zip(axes, metrics):
        values = []
        for alg in algorithms:
            if 'metrics' in results[alg] and metric in results[alg]['metrics']:
                val = results[alg]['metrics'][metric]
                values.append(val if not isinstance(val, dict) else np.nan)
            else:
                values.append(np.nan)

        bars = ax.bar(algorithms, values, color='steelblue', alpha=0.8)
        label, higher_is_better = metric_labels.get(metric, (metric, True))
        ax.set_ylabel(label)
        ax.set_title(f"{label}\n({'↑ higher is better' if higher_is_better else '↓ lower is better'})")
        ax.tick_params(axis='x', rotation=45)

        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, val),
                           ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def show_or_save(fig: plt.Figure, save_path: Optional[str] = None, show: bool = True):
    """Display or save a figure."""
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

