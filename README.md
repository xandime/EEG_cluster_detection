[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/jZYLDMog)


# EEG Feature Clustering Project

This project investigates whether participants can be meaningfully grouped based on EEG-derived features. Analyses are performed on the **Dortmund dataset** (model development) and **LEMON dataset** (external validation). The focus is on assessing the quality, stability, and interpretability of feature clusters.

---

## Directory Overview

| Folder / File | Purpose |
|---------------|---------|
| `clustering_notebooks/` | Contains clustering algorithms for k-means, GMM, aggomerative and OPTICS Jupyter.|
| `dataset/` | Original datasets and exported PCA-transformed datasets (PCA).|
| `function/` | Functions used in clustering algorithms, and other reusable analysis routines. |
| `helpers/` | Helper files for data handling, plotting, and evaluation metrics. |
| `infos/` | Documentation, notes, and project infos. |
| `metrics/` | External and internal metric scripts to evaluate clustering performance (silhouette, Davies–Bouldin, etc.). |
| `requirements.txt` | -- |

---

## Notebooks Overview

### 1. `feature_preprocessing.ipynb`
**Purpose:** Preprocesses EEG features from Dortmund and LEMON datasets.  
**Steps:**
- Handling missing values
- Standardizing numeric features
- Optional feature selection for interpretable feature space  

**Output:** Cleaned datasets ready for clustering.

### 2. `PCA.ipynb`
**Purpose:** Perform PCA on standardized features to reduce dimensionality.  
**Steps:**
- Standardize numeric features
- Compute optimal number of PCs using cumulative explained variance (e.g., 80% threshold)
- Visualize PCA results (scatter plots, cumulative variance, scree plots)  

**Output:** PCA-transformed datasets and component loadings, exported for clustering.

### 3. `compare_clustering_methods.ipynb`
**Purpose:** Compare results of clustering approaches using metrics.  
**Steps:**
- Evaluate clusters with internal metrics (silhouette, Davies–Bouldin) and external validation (age, sex)
- Visualize cluster assignments and feature contributions  

**Output:** Cluster assignments, evaluation metrics, and visualizations for both datasets.

