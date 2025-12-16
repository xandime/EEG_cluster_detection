# EEG_cluster_detection

This repository investigates whether participants can be meaningfully grouped based on EEG-derived features. Analyses are performed on the **Dortmund dataset** (model development) and **LEMON dataset** (external validation). The focus is on assessing the quality, stability, and interpretability of feature clusters.

---

## Project overview

This repository compares multiple clustering algorithms (KMeans, GMM, Agglomerative, OPTICS) on EEG-derived features from two datasets (Dortmund and LEMON). The pipeline includes preprocessing, dimensionality reduction (PCA and an autoencoder), metric computation, visualization, and stability assessment. Reusable code is centralized in `functions/` and `helpers/`.

---

## Directory overview (top-level)

| Folder / File | Purpose |
|---------------|---------|
| `autoencoder/` | Autoencoder implementation and related scripts (`autoencoder_standard.py`)|
| `compare_cluster_notebooks (old)/` | Older exploratory notebooks retained for reference (archive). |
| `dataset/` | Raw and processed datasets, PCA outputs, and saved clustering label CSVs |
| `functions/` | Reusable functions for clustering experiments (k-means, GMM, agglomerative, OPTICS), comparison helpers, and plotting. |
| `helpers/` | Helper utilities used across notebooks (data handling, plotting, evaluation helpers). |
| `infos/` | Documentation, project notes, PDFs and design sketches. |
| `metrics/` | Internal and external metric implementations and documentation for cluster evaluation. |
| `clustering_notebooks/` | Collection of focused notebooks for each clustering method. |


---

## Quickstart — run the project (PowerShell)

Requirements
- Python 3.10+ is recommended (project code was developed with recent Python 3.10–3.12 stacks). See `requirements.txt` for a complete list of packages.




Steps (minimal)
1. Clone the repository and navigate into it:
```powershell
git clone [https://github.com/xandime/EEG_cluster_detection.git](https://github.com/yourusername/EEG_cluster_detection.git)
cd EEG_cluster_detection
```


2. Create and activate a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate; pip install -r requirements.txt
```

3. Start Jupyter Lab / Notebook and open one of the canonical notebooks. Example (Jupyter Lab):

```powershell
.\.venv\Scripts\activate; pip install jupyterlab; jupyter lab
```

4. Extract the dataset.zip file into the `dataset/` folder.

Read to run the notebooks!

---
## Analysis Pipeline

This project follows a linear pipeline. The methodology is implemented across three stages. Please run the notebooks in the following order:

### 1. Data Exploration
* **Notebook:** `compare_correlation_structure.ipynb`
    * **Method:** Performs a structural similarity check to ensure the external validation set (LEMON) is statistically comparable to the development set (Dortmund).


### 2. Dimensionality Reduction
To address the high dimensionality of EEG features, we apply two distinct reduction techniques:
* **Notebook:** `PCA.ipynb`
    * **Method:** Principal Component Analysis (PCA). We assess explained variance and use bootstrapping to verify component stability.
* **Notebook:** `autoencoder.ipynb`
    * **Method:** Deep Learning Autoencoder. Trains a neural network to compress features into a non-linear latent space.
* **Notebook:** `feature_preprocessing.ipynb`
  * **Method:** Reduces dimensions with a filter based approach.

### 3. Clustering & Validation
We apply four algorithms (**KMeans, GMM, Agglomerative, OPTICS**) to the reduced spaces and evaluate them using internal metrics (Silhouette, Davies-Bouldin) and external metadata (Age, Sex).

* **Notebook:** `compare_clusters_PCA_Dortmund.ipynb`
    * **Task:** Compare clusters via internal and external metrics on the PCA reduced Dortmund dataset.
* **Notebook:** `compare_clusters_PCA_Lemon.ipynb`
    * **Task:** Compare clusters via internal and external metrics on the PCA reduced LEMON dataset (external validation).
* **Notebook:** `compare_clusters_preprocessed.ipynb`
    * **Task:** Compare clusters via internal and external metrics on the filtered datasets.

---

## Main Notebooks

The repository contains a set of canonical notebooks designed to reproduce the main analyses. The canonical notebooks call the centralized functions in `functions/` and save their outputs to `dataset/`.

- `compare_correlation_structure.ipynb` — compares correlation structures between datasets as a preliminary assessment.
- `feature_preprocessing.ipynb` — routines to clean, impute and standardize original features.
- `PCA.ipynb` — PCA computation, explained-variance diagnostics, and component loadings export.
- `autoencoder.ipynb` — trains/inspects the autoencoder latent space used as an alternative dimensionality reduction.
- `compare_clusters_PCA_Dortmund.ipynb` — loads pca dataset and runs a comparison between clusters.
- `compare_clusters_PCA_Lemon.ipynb` — same pipeline for Lemon dataset (external validation).
- `compare_clusters_preprocessed.ipynb` — runs clustering on preprocessed features.

---

## Outputs and saved artifacts
- `dataset/*_preprocessed.csv` — preprocessed feature datasets.
- `dataset/*_pca.csv` and `dataset/*_pca_loadings.csv` — PCA-transformed data and loadings.
- Optional model artifacts (autoencoder weights) if generated by the autoencoder notebook/scripts.
- Optional labels from each clustering algorithm — saved cluster assignments.

---

## Troubleshooting tips
- If package installation fails: make sure you are using a compatible Python version and that pip is up-to-date: `python -m pip install --upgrade pip`.
- If a dataset file is missing, check `dataset/` for CSVs such as `Dortmund_pca.csv` and `LEMON_pca.csv`. If not present run the necessary preprocessing notebooks first.
