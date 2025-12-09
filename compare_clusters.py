"""
Compare multiple clustering algorithms on a given dataset.

Supports KMeans, Agglomerative, GMM, and OPTICS clustering methods.
Includes internal validation metrics and optional 3D visualizations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

from functions.k_means_clustering import kmeans, select_optimal_k
from functions.agglomerative_clustering import agglomerative_clustering, select_optimal_n_agglomerative
from functions.gmm_clustering import gmm_clustering, select_optimal_n_gmm
from functions.optics_clustering import optics_clustering, tune_optics_min_samples
from functions.viz import (
    get_pca_projections,
    plot_3d_triple,
    plot_all_algorithms_3d,
    plot_metrics_comparison,
    show_or_save
)
from metrics.internal_metrics import compute_all_metrics, print_metrics


@dataclass
class ClusterConfig:
    """Configuration for a clustering algorithm."""
    enabled: bool = True
    n_clusters: Optional[int] = None  # None = auto-detect
    k_range: List[int] = field(default_factory=lambda: list(range(2, 11)))
    selection_metric: str = 'silhouette'  # 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'elbow', 'bic', 'aic'
    params: Dict = field(default_factory=dict)


@dataclass
class CompareConfig:
    """Configuration for the comparison pipeline."""
    kmeans: ClusterConfig = field(default_factory=lambda: ClusterConfig())
    agglomerative: ClusterConfig = field(default_factory=lambda: ClusterConfig())
    gmm: ClusterConfig = field(default_factory=lambda: ClusterConfig(selection_metric='bic'))
    optics: ClusterConfig = field(default_factory=lambda: ClusterConfig(
        k_range=list(range(3, 15)),
        params={'min_samples': 5}
    ))

    visualize_3d: bool = False
    viz_mode: str = 'end'  # 'each' or 'end'
    include_dunn: bool = False
    random_state: int = 42


def run_kmeans(data: np.ndarray, config: ClusterConfig, random_state: int = 42, verbose: bool = False) -> Dict:
    """Execute KMeans clustering with optional auto k selection."""
    if config.n_clusters is None:
        if verbose:
            print(f"  Auto-detecting optimal k using {config.selection_metric}...")
        optimal_k, results = select_optimal_k(
            data,
            k_range=config.k_range,
            metric=config.selection_metric,
            random_state=random_state,
            **config.params
        )
        if verbose:
            print(f"  Optimal k: {optimal_k}")
        result = results[optimal_k]
        return {'labels': result['assignments'], 'centers': result['centers'], 'optimal_k': optimal_k, 'raw': result}
    else:
        result = kmeans(
            data,
            k=config.n_clusters,
            random_state=random_state,
            **config.params
        )
        return {'labels': result['assignments'], 'centers': result['centers'], 'raw': result}


def run_agglomerative(data: np.ndarray, config: ClusterConfig, verbose: bool = False) -> Dict:
    """Execute Agglomerative clustering with optional auto n selection."""
    if config.n_clusters is None:
        if verbose:
            print(f"  Auto-detecting optimal n using {config.selection_metric}...")
        optimal_n, results = select_optimal_n_agglomerative(
            data,
            n_range=config.k_range,
            selection_metric=config.selection_metric,
            **config.params
        )
        if verbose:
            print(f"  Optimal n: {optimal_n}")
        result = results[optimal_n]
        return {'labels': result['labels'], 'optimal_n': optimal_n, 'raw': result}
    else:
        result = agglomerative_clustering(
            data,
            n_clusters=config.n_clusters,
            **config.params
        )
        return {'labels': result['labels'], 'raw': result}


def run_gmm(data: np.ndarray, config: ClusterConfig, random_state: int = 42, verbose: bool = False) -> Dict:
    """Execute GMM clustering with optional auto n selection."""
    if config.n_clusters is None:
        if verbose:
            print(f"  Auto-detecting optimal n using {config.selection_metric}...")
        optimal_n, results = select_optimal_n_gmm(
            data,
            n_range=config.k_range,
            metric=config.selection_metric,
            random_state=random_state,
            **config.params
        )
        if verbose:
            print(f"  Optimal n: {optimal_n}")
        result = results[optimal_n]
        return {
            'labels': result['labels'],
            'probabilities': result['probabilities'],
            'optimal_n': optimal_n,
            'raw': result
        }
    else:
        result = gmm_clustering(
            data,
            n_components=config.n_clusters,
            random_state=random_state,
            **config.params
        )
        return {'labels': result['labels'], 'probabilities': result['probabilities'], 'raw': result}


def run_optics(data: np.ndarray, config: ClusterConfig, verbose: bool = False) -> Dict:
    """Execute OPTICS clustering with optional auto min_samples tuning."""
    min_samples = config.params.get('min_samples', None)

    if min_samples is None:
        if verbose:
            print(f"  Auto-tuning min_samples...")
        best_min_samples, results = tune_optics_min_samples(
            data,
            min_samples_range=config.k_range,
            **{k: v for k, v in config.params.items() if k != 'min_samples'}
        )
        if verbose:
            print(f"  Optimal min_samples: {best_min_samples}")
        result = results[best_min_samples]
        return {
            'labels': result['labels'],
            'n_clusters': result['n_clusters'],
            'optimal_min_samples': best_min_samples,
            'raw': result
        }
    else:
        params = {'min_samples': min_samples}
        params.update({k: v for k, v in config.params.items() if k != 'min_samples'})
        result = optics_clustering(data, **params)
        return {'labels': result['labels'], 'n_clusters': result['n_clusters'], 'raw': result}


def compare_clusters(
    data: Union[np.ndarray, pd.DataFrame],
    config: Optional[CompareConfig] = None,
    feature_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict:
    """
    Compare multiple clustering algorithms on a dataset.

    Args:
        data: Feature matrix (N samples, D features). Can be numpy array or DataFrame.
        config: Configuration object. Uses defaults if None.
        feature_names: Optional feature names for visualization.
        verbose: Print progress and metrics.

    Returns:
        Dictionary with results for each algorithm including labels and metrics.

    Example:
        >>> from compare_clusters import compare_clusters, CompareConfig, ClusterConfig
        >>> config = CompareConfig(
        ...     kmeans=ClusterConfig(n_clusters=4),
        ...     visualize_3d=True,
        ...     viz_mode='end'
        ... )
        >>> results = compare_clusters(data, config=config)
    """
    if config is None:
        config = CompareConfig()

    if isinstance(data, pd.DataFrame):
        if feature_names is None:
            feature_names = list(data.columns)
        data = data.values

    data = np.asarray(data, dtype=np.float64)

    results = {}
    algorithms = [
        ('KMeans', config.kmeans, run_kmeans),
        ('Agglomerative', config.agglomerative, run_agglomerative),
        ('GMM', config.gmm, run_gmm),
        ('OPTICS', config.optics, run_optics)
    ]

    pca_data = None
    if config.visualize_3d and data.shape[1] > 3:
        pca_data, _ = get_pca_projections(data, n_components=6)
    else:
        pca_data = data

    for name, alg_config, run_fn in algorithms:
        if not alg_config.enabled:
            continue

        if verbose:
            print(f"\n{'='*60}")
            print(f"Running {name}...")
            print('='*60)

        if name in ('KMeans', 'GMM'):
            result = run_fn(data, alg_config, config.random_state, verbose=verbose)
        else:
            result = run_fn(data, alg_config, verbose=verbose)

        labels = result['labels']

        valid_mask = labels >= 0
        if valid_mask.sum() > 0 and len(np.unique(labels[valid_mask])) > 1:
            metrics = compute_all_metrics(
                data[valid_mask], labels[valid_mask], include_dunn=config.include_dunn
            )
        else:
            metrics = {'error': 'Insufficient clusters for metrics'}

        result['metrics'] = metrics
        results[name] = result

        if verbose and isinstance(metrics, dict) and 'error' not in metrics:
            print_metrics(metrics)
            n_clusters = len(np.unique(labels[labels >= 0]))
            n_noise = (labels == -1).sum()
            print(f"\n  Clusters found: {n_clusters}")
            if n_noise > 0:
                print(f"  Noise points: {n_noise}")

        if config.visualize_3d and config.viz_mode == 'each':
            fig = plot_3d_triple(pca_data, labels, name, feature_names)
            show_or_save(fig)

    if config.visualize_3d and config.viz_mode == 'end':
        if verbose:
            print(f"\n{'='*60}")
            print("Generating comparison visualizations...")
            print('='*60)

        fig_compare = plot_all_algorithms_3d(results, pca_data, feature_names)
        show_or_save(fig_compare)

        fig_metrics = plot_metrics_comparison(results)
        show_or_save(fig_metrics)

        for name, result in results.items():
            fig = plot_3d_triple(pca_data, result['labels'], name, feature_names)
            show_or_save(fig)


    if verbose:
        print(f"\n{'='*60}")
        print("Summary Comparison")
        print('='*60)
        summary = create_summary_table(results)
        print(summary.to_string())

    return results


def create_summary_table(results: Dict) -> pd.DataFrame:
    """Create a summary DataFrame comparing all algorithms."""
    rows = []
    for name, result in results.items():
        labels = result['labels']
        n_clusters = len(np.unique(labels[labels >= 0]))
        n_noise = (labels == -1).sum()

        row = {
            'Algorithm': name,
            'Clusters': n_clusters,
            'Noise Points': n_noise
        }

        if 'metrics' in result and isinstance(result['metrics'], dict):
            metrics = result['metrics']
            if 'silhouette_score' in metrics:
                row['Silhouette'] = f"{metrics['silhouette_score']:.4f}"
            if 'davies_bouldin_score' in metrics:
                row['Davies-Bouldin'] = f"{metrics['davies_bouldin_score']:.4f}"
            if 'calinski_harabasz_score' in metrics:
                row['Calinski-Harabasz'] = f"{metrics['calinski_harabasz_score']:.2f}"

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == '__main__':
    np.random.seed(42)

    from sklearn.datasets import make_blobs
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=10, random_state=42)

    # Auto-detect optimal number of clusters
    config = CompareConfig(
        kmeans=ClusterConfig(n_clusters=None, k_range=list(range(2, 8))),
        agglomerative=ClusterConfig(n_clusters=None, k_range=list(range(2, 8))),
        gmm=ClusterConfig(n_clusters=None, k_range=list(range(2, 8)), selection_metric='bic'),
        optics=ClusterConfig(k_range=list(range(3, 10)), params={'min_samples': None}),
        visualize_3d=True,
        viz_mode='end'
    )

    results = compare_clusters(X, config=config, verbose=True)

