"""
Gaussian Mixture Model (GMM) clustering implementation for EEG participant subtyping.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from metrics.internal_metrics import (
    compute_silhouette_score,
    compute_davies_bouldin_score,
    compute_calinski_harabasz_score
)


def gmm_clustering(data, n_components, covariance_type='full', random_state=42,
                   max_iter=100, verbose=False):
    """
    Perform Gaussian Mixture Model clustering.

    Args:
        data: numpy array of shape (N, d). Data to cluster.
        n_components: int. Number of mixture components (clusters).
        covariance_type: str. Type of covariance parameters ('full', 'tied', 'diag', 'spherical').
        random_state: int. Random seed for reproducibility.
        max_iter: int. Maximum number of EM iterations.
        verbose: bool. Whether to print convergence information.

    Returns:
        dict with keys:
            - 'labels': Hard cluster assignments.
            - 'probabilities': Soft cluster probabilities.
            - 'gmm_model': Fitted GaussianMixture object.
            - 'bic': Bayesian Information Criterion.
            - 'aic': Akaike Information Criterion.
            - 'converged': Whether the algorithm converged.
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        max_iter=max_iter,
        verbose=verbose
    )

    gmm.fit(data)
    labels = gmm.predict(data)
    probabilities = gmm.predict_proba(data)

    return {
        'labels': labels,
        'probabilities': probabilities,
        'gmm_model': gmm,
        'bic': gmm.bic(data),
        'aic': gmm.aic(data),
        'converged': gmm.converged_
    }


def select_optimal_n_gmm(data, n_range, metric='bic', covariance_type='full',
                         random_state=42, verbose=False):
    """
    Select optimal number of components using BIC, AIC, or internal validation metrics.

    Args:
        data: numpy array of shape (N, d). Data to cluster.
        n_range: list of int. Range of n_components to test.
        metric: str. Metric for selection: 'bic', 'aic', 'silhouette',
                'davies_bouldin', or 'calinski_harabasz'.
        covariance_type: str. Type of covariance parameters.
        random_state: int. Random seed.
        verbose: bool. Print information per n.

    Returns:
        optimal_n: int. Optimal number of components.
        results: dict. Results for each n, keyed by n.
    """
    results = {}
    scores = []

    for n in n_range:
        result = gmm_clustering(
            data, n, covariance_type=covariance_type,
            random_state=random_state, verbose=verbose
        )
        results[n] = result

        if metric == 'bic':
            score = result['bic']
        elif metric == 'aic':
            score = result['aic']
        elif metric == 'silhouette':
            score = compute_silhouette_score(data, result['labels'])
        elif metric == 'davies_bouldin':
            score = compute_davies_bouldin_score(data, result['labels'])
        elif metric == 'calinski_harabasz':
            score = compute_calinski_harabasz_score(data, result['labels'])
        else:
            raise ValueError(f"Unknown metric: {metric}")

        scores.append(score)

        if verbose:
            print(f"n={n}: {metric}={score:.2f}")

    # Select optimal based on metric (lower is better for BIC/AIC/Davies-Bouldin)
    if metric in ['bic', 'aic', 'davies_bouldin']:
        optimal_n = n_range[np.argmin(scores)]
    else:
        optimal_n = n_range[np.argmax(scores)]

    return optimal_n, results


def compare_covariance_types(data, n_components, random_state=42):
    """
    Compare different covariance types for GMM.

    Args:
        data: numpy array. Data to cluster.
        n_components: int. Number of components.
        random_state: int. Random seed.

    Returns:
        dict. Results for each covariance type with BIC and AIC scores.
    """
    covariance_types = ['full', 'tied', 'diag', 'spherical']
    results = {}

    for cov_type in covariance_types:
        result = gmm_clustering(data, n_components, covariance_type=cov_type,
                               random_state=random_state)
        results[cov_type] = {
            'bic': result['bic'],
            'aic': result['aic'],
            'labels': result['labels'],
            'converged': result['converged']
        }

    return results

