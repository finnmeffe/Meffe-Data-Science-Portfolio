# MLUnsupervisedApp

<kbd>
<img width="1487" height="812" alt="image" src="https://github.com/user-attachments/assets/33365598-d5f7-4613-a579-ffa5065b95c4" />
</kbd>

*Example model output using K-Means clustering*

This is an interactive unsupervised machine learning web app built with Python and Streamlit. Users can upload their own tabular dataset or explore the bundled Housing Prices sample (Kaggle), pick features, and tune three different unsupervised models: K-Means clustering, Agglomerative (hierarchical) clustering, and Principal Component Analysis.

## Live App

> **[Launch the app](https://finnmeffe-meffe-data-s-mlunsupervisedappunsupervised-app-aptbsi.streamlit.app/)**

## Project Overview

This app is the unsupervised counterpart to the classification app in `../MLStreamlitApp`. The goal is to make exploratory unsupervised learning accessible: you give it a dataset and a model, and the app produces the standard diagnostics (e.g. elbow plots, silhouette scores, dendrograms, and PCA scree/loadings) without writing any code.

The bundled sample is the Housing Prices Dataset by Yasser H. on Kaggle: 545 rows of residential property listings with both numeric (price, area, bedrooms, …) and categorical (mainroad, furnishingstatus, …) features. It is well suited to clustering because there are clear segments (small unfurnished houses vs. large furnished ones with parking and AC) that the algorithms should recover.

## App Features

### Data layer
- **Upload any CSV** to use your own tabular dataset
- **Housing sample from Kaggle** via the [`kagglehub`](https://github.com/Kaggle/kagglehub) API (`yasserh/housing-prices-dataset`); the download is cached locally so subsequent runs are instant and offline-friendly
- Automatic preprocessing: drops missing rows, maps `yes`/`no` columns to `1`/`0`, one-hot-encodes remaining categoricals, and standardizes features (toggleable)
- Data preview tab with row/column counts, missing-value summary, dtype inspection, and a numeric correlation heatmap

### Models supported

| Model | Key hyperparameters | Diagnostics produced |
|---|---|---|
| **K-Means Clustering** | `n_clusters`, `n_init`, `max_iter`, random seed, elbow scan range | Cluster scatter (PCA-reduced 2D), per-sample silhouette plot, elbow + silhouette curve across k, cluster sizes, mean-by-cluster profile table, downloadable labeled CSV |
| **Hierarchical Clustering** | `n_clusters`, linkage (`ward`/`complete`/`average`/`single`), distance metric, dendrogram leaf count | Cluster scatter, per-sample silhouette plot, truncated dendrogram, cluster sizes, downloadable labeled CSV |
| **Principal Component Analysis** | `n_components` | Explained variance bar + cumulative line plot, PC1 vs PC2 loading biplot, projection scatter, full loadings matrix, downloadable transformed CSV |

All implementations use `scikit-learn`; the dendrogram uses `scipy.cluster.hierarchy.linkage`.

### Layout
- **Sidebar** — data source, feature selection, model choice, hyperparameter sliders, "Run model" button
- **Tabs** — *Data preview* / *Model info* / *Results*. The *Model info* tab explains what each algorithm does and shows the currently selected hyperparameters as JSON, so users always know what is about to run.

### Hyperparameter tuning
All hyperparameters are exposed as sliders, dropdowns, or numeric inputs in the sidebar. Changes take effect on the next **Run model** click, which makes it easy to compare configurations side by side. The K-Means elbow plot scans a range of k values in one shot to help pick the right number of clusters.

## Sample dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Housing Prices Dataset by Yasser H.](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) |
| Rows | 545 |
| Columns | 13 (mix of numeric and categorical) |
| Numeric features | `price`, `area`, `bedrooms`, `bathrooms`, `stories`, `parking` |
| Binary features | `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea` |
| Categorical | `furnishingstatus` (furnished / semi-furnished / unfurnished) |

The CSV is downloaded on first launch via [`kagglehub`](https://github.com/Kaggle/kagglehub) and cached at `~/.cache/kagglehub/`, so the app works out of the box without manually placing any files. If the Kaggle API is unreachable (e.g. no network), the app automatically falls back to a synthetic sample with the same schema so it still runs.

## Instructions

### Prerequisites
- Python 3.10 or higher
- pip

### Run locally

```bash
# 1. Clone the portfolio repo
git clone https://github.com/finnmeffe/MEFFE-Data-Science-Portfolio.git
cd MEFFE-Data-Science-Portfolio/MLUnsupervisedApp

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app opens at a local host.

### Deploy to Streamlit Community Cloud
1. Push the repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), pick the repo, and set the main file to `MLUnsupervisedApp/app.py`.
3. Streamlit will install everything in `requirements.txt` automatically.

### Dependencies

| Library | Version | Purpose |
|---|---|---|
| streamlit | ≥ 1.32 | Web UI framework |
| pandas | ≥ 2.0 | Data loading and manipulation |
| numpy | ≥ 1.26 | Numeric operations |
| scikit-learn | ≥ 1.4 | KMeans, AgglomerativeClustering, PCA, silhouette |
| scipy | ≥ 1.11 | Hierarchical linkage and dendrogram |
| matplotlib | ≥ 3.8 | Plotting |
| seaborn | ≥ 0.13 | Heatmaps and color palettes |
| kagglehub | ≥ 0.3 | API download of the Housing sample dataset |

## How hyperparameters are selected

- **K in K-Means** — read the elbow plot for a "knee" in the inertia curve and the peak in the silhouette curve. The app overlays both on a single chart so the trade-off is visible.
- **Linkage in hierarchical clustering** — `ward` is a strong default for compact, similarly-sized clusters and is the only linkage that requires Euclidean distance. Use `complete`/`average`/`single` with custom metrics when cluster shape is non-spherical.
- **Number of components in PCA** — scan the cumulative variance curve and pick the smallest number of components that captures the variance you need (often 80–95 %).

## References
- [Kaggle Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)
- [scikit-learn: Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [scikit-learn: PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [scikit-learn: Silhouette analysis example](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
- [scipy: hierarchical clustering](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Community Cloud Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)
