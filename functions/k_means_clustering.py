"""
K-means clustering implementation for EEG participant subtyping.

This module provides functions for K-means clustering with support for:
- Preprocessed data
- PCA-transformed data (TODO)
- Interpretable feature space (TODO)
"""

import numpy as np


def initialize_clusters(data, k):
    """Randomly initialize the k cluster centers.

    Chooses k clusters from the data itself to ensure proper scale.

    Args:
        data: numpy array of shape (N, d). Original data.
        k: int. Number of clusters for the k-means algorithm.

    Returns:
        numpy array of shape (k, d) with the k initial cluster centers.
    """
    N, d = data.shape
    return data[np.random.choice(N, k, replace=False)]


def build_distance_matrix(data, mu):
    """Build a distance matrix of squared distances from data points to cluster centers.

    Args:
        data: numpy array of shape (N, d). Original data.
        mu: numpy array of shape (k, d). Each row is a cluster center.

    Returns:
        numpy array of shape (N, k) with squared distances.
        Entry (i, j) is the squared distance from data point i to cluster center j.
    """
    N, d = data.shape
    k, _ = mu.shape
    distance_matrix = np.zeros((N, k))
    for j in range(k):
        distance_matrix[:, j] = np.sum(np.square(data - mu[j, :]), axis=1)
    return distance_matrix


def update_kmeans_parameters(data, mu_old):
    """Compute one step of the k-means algorithm.

    Using mu_old, find which cluster each datapoint belongs to,
    then update the cluster centers.

    Args:
        data: numpy array of shape (N, d). Original data.
        mu_old: numpy array of shape (k, d). Each row is a cluster center.

    Returns:
        losses: numpy array of shape (N,). Squared distances of each data point
            to its assigned cluster mean (computed from mu_old).
        assignments: numpy array of shape (N,). Cluster assignment for each data point.
        mu: numpy array of shape (k, d). Updated cluster centers.
    """
    _, d = data.shape
    k, _ = mu_old.shape
    distance_matrix = build_distance_matrix(data, mu_old)
    losses = np.min(distance_matrix, axis=1)
    assignments = np.argmin(distance_matrix, axis=1)

    mu = np.zeros((k, d))
    for j in range(k):
        rows = np.where(assignments == j)[0]
        if len(rows) > 0:
            mu[j, :] = np.mean(data[rows, :], axis=0)
        else:
            mu[j, :] = mu_old[j, :]

    return losses, assignments, mu


def kmeans(data, k, max_iters=100, threshold=1e-5, verbose=False, random_state=None):
    """Run the k-means algorithm.

    Args:
        data: numpy array of shape (N, d). Data to cluster.
        k: int. Number of clusters.
        max_iters: int. Maximum number of iterations.
        threshold: float. Convergence threshold based on change in average loss.
        verbose: bool. Whether to print iteration information.
        random_state: int or None. Random seed for reproducibility.

    Returns:
        dict with keys:
            - 'assignments': numpy array of shape (N,). Final cluster assignments.
            - 'centers': numpy array of shape (k, d). Final cluster centers.
            - 'losses': list of average losses per iteration.
            - 'converged': bool. Whether the algorithm converged.
            - 'n_iterations': int. Number of iterations performed.
    """
    if random_state is not None:
        np.random.seed(random_state)

    mu_old = initialize_clusters(data, k)
    loss_list = []
    assignments = None
    mu = None

    for iteration in range(max_iters):
        losses, assignments, mu = update_kmeans_parameters(data, mu_old)
        average_loss = np.mean(losses)
        loss_list.append(average_loss)

        if verbose:
            print(f"Iteration {iteration}: average loss = {average_loss:.6f}")

        if iteration > 0 and np.abs(loss_list[-1] - loss_list[-2]) < threshold:
            if verbose:
                print(f"Converged after {iteration + 1} iterations")
            return {
                'assignments': assignments,
                'centers': mu,
                'losses': loss_list,
                'converged': True,
                'n_iterations': iteration + 1
            }

        mu_old = mu

    if verbose:
        print(f"Did not converge after {max_iters} iterations")

    return {
        'assignments': assignments,
        'centers': mu,
        'losses': loss_list,
        'converged': False,
        'n_iterations': max_iters
    }


def kmeans_multiple_runs(data, k, n_runs=10, max_iters=100, threshold=1e-5,
                        verbose=False, random_state=None):
    """Run k-means multiple times with different initializations.

    Returns the best result (lowest final loss) among all runs.

    Args:
        data: numpy array of shape (N, d). Data to cluster.
        k: int. Number of clusters.
        n_runs: int. Number of times to run k-means.
        max_iters: int. Maximum number of iterations per run.
        threshold: float. Convergence threshold.
        verbose: bool. Whether to print information.
        random_state: int or None. Random seed for reproducibility.

    Returns:
        dict with the best k-means result (same format as kmeans function).
    """
    if random_state is not None:
        np.random.seed(random_state)

    best_result = None
    best_loss = float('inf')

    for run in range(n_runs):
        if verbose:
            print(f"\n--- Run {run + 1}/{n_runs} ---")

        result = kmeans(data, k, max_iters=max_iters, threshold=threshold,
                       verbose=verbose, random_state=None)

        final_loss = result['losses'][-1]
        if final_loss < best_loss:
            best_loss = final_loss
            best_result = result
            best_result['best_run'] = run + 1

    if verbose:
        print(f"\nBest result from run {best_result['best_run']} with loss {best_loss:.6f}")

    return best_result


# TODO: Later apply to interpretable feature space
# TODO: Later apply to PCA-transformed data

