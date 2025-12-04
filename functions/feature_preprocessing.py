"""
Feature preprocessing utilities for EEG data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_preprocessed_data(filepath='dataset/Dortmund_features_preprocessed.csv'):
    """
    Load preprocessed features.

    Args:
        filepath: str. Path to preprocessed CSV file.

    Returns:
        data: DataFrame with subject IDs as index.
        features: numpy array of feature values.
        subject_ids: array of subject IDs.
    """
    data = pd.read_csv(filepath, index_col=0)
    features = data.values
    subject_ids = data.index.values

    return data, features, subject_ids


def load_pca_data(pca_filepath='dataset/Dortmund_pca.csv',
                  loadings_filepath='dataset/Dortmund_pca_loadings.csv'):
    """
    Load PCA-transformed data and related information.

    Args:
        pca_filepath: str. Path to PCA-transformed data.
        loadings_filepath: str. Path to PCA loadings.

    Returns:
        pca_data: DataFrame with PCA-transformed features.
        pca_features: numpy array of PCA features.
        loadings: DataFrame with PCA loadings.
        components_info: DataFrame with variance explained info.
    """
    pca_data = pd.read_csv(pca_filepath, index_col=0)
    pca_features = pca_data.values

    loadings = pd.read_csv(loadings_filepath, index_col=0)

    return pca_data, pca_features, loadings


def load_raw_features(filepath='dataset/Dortmund_features.csv'):
    """
    Load raw (unprocessed) feature data.

    Args:
        filepath: str. Path to raw features CSV.

    Returns:
        data: DataFrame with raw features.
        features: numpy array.
        subject_ids: array of subject IDs.
    """
    data = pd.read_csv(filepath, index_col=0)
    features = data.values
    subject_ids = data.index.values

    return data, features, subject_ids


def preprocess_features(data, var_threshold=1e-6, corr_threshold=0.95, z_clip=3.0):
    """
    Complete preprocessing pipeline: variance filtering, standardization, correlation filtering, outlier clipping.

    Args:
        data: DataFrame with raw features (subject IDs as index).
        var_threshold: float. Minimum variance threshold for feature retention.
        corr_threshold: float. Correlation threshold for dropping highly correlated features.
        z_clip: float. Z-score clipping threshold for outlier handling.

    Returns:
        preprocessed_data: DataFrame with preprocessed features.
        dropped_features: dict with info about dropped features.
    """
    dropped_features = {'low_variance': [], 'high_correlation': []}

    # 1. Drop low-variance features
    variances = data.var()
    low_var_cols = variances[variances < var_threshold].index.tolist()
    dropped_features['low_variance'] = low_var_cols
    data_filtered = data.drop(columns=low_var_cols)

    # 2. Standardization
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(
        scaler.fit_transform(data_filtered),
        columns=data_filtered.columns,
        index=data.index
    )

    # 3. Drop highly correlated features
    corr_matrix = data_scaled.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    dropped_features['high_correlation'] = to_drop
    data_nocorr = data_scaled.drop(columns=to_drop)

    # 4. Z-score clipping for outliers
    data_clean = data_nocorr.copy()
    for col in data_clean.columns:
        data_clean[col] = np.clip(data_clean[col], -z_clip, z_clip)

    return data_clean, dropped_features


def save_preprocessed_data(data, output_path='dataset/Dortmund_features_preprocessed.csv'):
    """
    Save preprocessed data to CSV.

    Args:
        data: DataFrame with preprocessed features.
        output_path: str. Output file path.
    """
    data.to_csv(output_path)
    print(f"Preprocessed features saved to {output_path}")


def get_feature_statistics(data):
    """
    Compute basic statistics for features.

    Args:
        data: DataFrame with features.

    Returns:
        stats: DataFrame with statistics (mean, std, min, max, etc.).
    """
    stats = data.describe()
    return stats


if __name__ == "__main__":
    # Example usage
    raw_data, _, _ = load_raw_features()
    preprocessed_data, dropped = preprocess_features(raw_data)
    save_preprocessed_data(preprocessed_data)
    stats = get_feature_statistics(preprocessed_data)
    print(stats)
    print("Dropped features:", dropped)

    raw_data_L, _, _ = load_raw_features('dataset/Lemon_features.csv')
    preprocessed_data_L, dropped_L = preprocess_features(raw_data_L)
    save_preprocessed_data(preprocessed_data_L, output_path='dataset/Lemon_features_preprocessed.csv')
    stats_L = get_feature_statistics(preprocessed_data_L)
    print(stats_L)
    print("Dropped features:", dropped_L)