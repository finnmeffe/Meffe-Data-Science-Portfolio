import os
import io
import glob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage


# Basic page configuration
st.set_page_config(
    page_title="Unsupervised Machine Learning Application",
    layout="wide",
)


# Kaggle dataset slug for the Housing Prices sample
KAGGLE_DATASET = "yasserh/housing-prices-dataset"


def _synthetic_housing_fallback():
    """Last-resort dummy data so the app still loads if the API is unreachable."""
    rng = np.random.default_rng(42)
    n = 545
    return pd.DataFrame({
        "price": rng.integers(1_750_000, 13_300_000, n),
        "area": rng.integers(1650, 16200, n),
        "bedrooms": rng.integers(1, 7, n),
        "bathrooms": rng.integers(1, 5, n),
        "stories": rng.integers(1, 5, n),
        "mainroad": rng.choice(["yes", "no"], n, p=[0.85, 0.15]),
        "guestroom": rng.choice(["yes", "no"], n, p=[0.18, 0.82]),
        "basement": rng.choice(["yes", "no"], n, p=[0.35, 0.65]),
        "hotwaterheating": rng.choice(["yes", "no"], n, p=[0.05, 0.95]),
        "airconditioning": rng.choice(["yes", "no"], n, p=[0.32, 0.68]),
        "parking": rng.integers(0, 4, n),
        "prefarea": rng.choice(["yes", "no"], n, p=[0.23, 0.77]),
        "furnishingstatus": rng.choice(
            ["furnished", "semi-furnished", "unfurnished"], n
        ),
    })


# Pull the Housing dataset directly from Kaggle via the kagglehub API.
# kagglehub caches the download under ~/.cache/kagglehub/, so subsequent runs
# re-use the local copy and don't hit the network.
@st.cache_data(show_spinner=False)
def load_housing_data():
    try:
        import kagglehub  # imported lazily so the app still starts if missing
        path = kagglehub.dataset_download(KAGGLE_DATASET)

        # The download path is a directory containing one or more CSVs; find
        # the Housing CSV regardless of exact filename / case.
        candidates = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
        if not candidates:
            raise FileNotFoundError(f"No CSV found inside {path}")

        # Prefer a file literally named "Housing.csv" if it's there
        preferred = [c for c in candidates if os.path.basename(c).lower() == "housing.csv"]
        csv_path = preferred[0] if preferred else candidates[0]
        return pd.read_csv(csv_path)
    except Exception as e:
        # Don't crash — let the caller know we're on the synthetic fallback.
        st.warning(
            "Could not download the Housing dataset from Kaggle "
            f"(`{KAGGLE_DATASET}`). Falling back to a synthetic sample so the "
            f"app still works.\n\n*Error:* `{e}`"
        )
        return _synthetic_housing_fallback()


# Encode categorical (yes/no -> 1/0, multi-class -> one-hot) and standardize
def preprocess(df: pd.DataFrame, feature_cols: list, scale: bool = True):
    sub = df[feature_cols].dropna().copy()
    # binary yes/no columns -> 1/0
    for col in sub.select_dtypes(include="object").columns:
        uniq = set(sub[col].astype(str).str.lower().unique())
        if uniq <= {"yes", "no"}:
            sub[col] = sub[col].astype(str).str.lower().map({"yes": 1, "no": 0})
    # one-hot any remaining object columns
    sub = pd.get_dummies(sub, drop_first=False)
    feature_names = sub.columns.tolist()
    X = sub.values.astype(float)
    if scale:
        X = StandardScaler().fit_transform(X)
    return X, feature_names


# 2-D projection used by every cluster scatter plot for consistency
def project_2d(X):
    pca = PCA(n_components=2, random_state=0)
    return pca.fit_transform(X), pca.explained_variance_ratio_


def plot_clusters_2d(X2d, labels, var_ratio, title):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    palette = sns.color_palette("tab10", n_colors=max(len(np.unique(labels)), 3))
    for i, lab in enumerate(np.unique(labels)):
        mask = labels == lab
        ax.scatter(X2d[mask, 0], X2d[mask, 1], s=22, alpha=0.8,
                   color=palette[i % len(palette)], label=f"Cluster {lab}")
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_elbow(X, k_range):
    inertias, sils = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        km.fit(X)
        inertias.append(km.inertia_)
        if k >= 2:
            sils.append(silhouette_score(X, km.labels_))
        else:
            sils.append(np.nan)
    fig, ax1 = plt.subplots(figsize=(5.5, 3.5))
    ax1.plot(list(k_range), inertias, "o-", color="#4c78a8", label="Inertia")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia (within-cluster SS)", color="#4c78a8")
    ax1.tick_params(axis="y", labelcolor="#4c78a8")
    ax2 = ax1.twinx()
    ax2.plot(list(k_range), sils, "s--", color="#f58518", label="Silhouette")
    ax2.set_ylabel("Silhouette score", color="#f58518")
    ax2.tick_params(axis="y", labelcolor="#f58518")
    ax1.set_title("Elbow & silhouette across k")
    fig.tight_layout()
    return fig, inertias, sils


def plot_silhouette(X, labels):
    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        return None, np.nan
    sil_avg = silhouette_score(X, labels)
    sample_sil = silhouette_samples(X, labels)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    y_lower = 10
    palette = sns.color_palette("tab10", n_colors=max(n_clusters, 3))
    for i, lab in enumerate(np.unique(labels)):
        vals = sample_sil[labels == lab]
        vals.sort()
        size = vals.shape[0]
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                         facecolor=palette[i % len(palette)], alpha=0.8)
        ax.text(-0.04, y_lower + 0.5 * size, str(lab))
        y_lower = y_upper + 10
    ax.axvline(sil_avg, color="red", linestyle="--",
               label=f"Mean = {sil_avg:.3f}")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.set_yticks([])
    ax.set_title("Per-sample silhouette plot")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig, sil_avg


def plot_dendrogram(X, method, max_leaves=30):
    Z = linkage(X, method=method)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=max_leaves,
        leaf_rotation=90.0,
        leaf_font_size=8.0,
        show_contracted=True,
        ax=ax,
    )
    ax.set_title(f"Dendrogram (linkage='{method}', truncated to last {max_leaves} merges)")
    ax.set_xlabel("Cluster size or sample index")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    return fig


def plot_pca_variance(pca):
    expl = pca.explained_variance_ratio_
    cum = np.cumsum(expl)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    xs = np.arange(1, len(expl) + 1)
    ax.bar(xs, expl, color="#4c78a8", alpha=0.85, label="Individual")
    ax.plot(xs, cum, "o-", color="#f58518", label="Cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA explained variance")
    ax.set_xticks(xs)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="center right")
    fig.tight_layout()
    return fig


def plot_pca_loadings(pca, feature_names):
    comps = pca.components_[:2]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for i, name in enumerate(feature_names):
        ax.arrow(0, 0, comps[0, i], comps[1, i],
                 head_width=0.02, head_length=0.02, fc="#4c78a8", ec="#4c78a8", alpha=0.8)
        ax.text(comps[0, i] * 1.12, comps[1, i] * 1.12, name, fontsize=8, ha="center")
    lim = float(np.abs(comps).max()) * 1.4
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("PC1 loading")
    ax.set_ylabel("PC2 loading")
    ax.set_title("PCA component loadings")
    fig.tight_layout()
    return fig


# -------------------- Sidebar --------------------
with st.sidebar:
    st.title("Unsupervised ML Project")
    st.caption("Finn Meffe")

    st.divider()
    st.subheader("1. Data source")
    data_source = st.radio(
        "Choose a data source",
        ["Sample: Housing Prices (Kaggle)", "Upload your own CSV"],
        label_visibility="collapsed",
    )

    df_raw = None
    if data_source == "Sample: Housing Prices (Kaggle)":
        with st.spinner("Loading Housing data…"):
            df_raw = load_housing_data()
        st.success(f"Loaded {len(df_raw):,} rows × {len(df_raw.columns)} columns")
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df_raw = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df_raw):,} rows × {len(df_raw.columns)} columns")

    if df_raw is not None:
        st.divider()
        st.subheader("2. Features")
        all_cols = df_raw.columns.tolist()
        # sensible defaults for the housing sample
        housing_defaults = [
            c for c in ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
            if c in all_cols
        ]
        default_feats = housing_defaults if housing_defaults else all_cols[: min(6, len(all_cols))]
        feature_cols = st.multiselect(
            "Feature columns (used for clustering / PCA)",
            all_cols,
            default=default_feats,
        )
        scale = st.checkbox("Standardize features (recommended)", value=True)

        st.divider()
        st.subheader("3. Model")
        model_name = st.selectbox(
            "Algorithm",
            ["K-Means Clustering", "Hierarchical Clustering", "Principal Component Analysis"],
        )

        st.divider()
        st.subheader("4. Hyperparameters")

        if model_name == "K-Means Clustering":
            n_clusters = st.slider("Number of clusters (k)", 2, 10, 3)
            n_init = st.slider("n_init (random restarts)", 1, 20, 10)
            max_iter = st.slider("Max iterations", 100, 1000, 300, 50)
            random_state = st.number_input("Random seed", value=42, step=1)
            elbow_max = st.slider("Elbow plot: scan k from 2 to …", 4, 12, 10)
            model_params = dict(
                n_clusters=n_clusters, n_init=n_init,
                max_iter=max_iter, random_state=int(random_state),
            )

        elif model_name == "Hierarchical Clustering":
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
            linkage_method = st.selectbox(
                "Linkage method", ["ward", "complete", "average", "single"]
            )
            metric = "euclidean" if linkage_method == "ward" else st.selectbox(
                "Distance metric", ["euclidean", "manhattan", "cosine"]
            )
            max_leaves = st.slider("Dendrogram: show last N merges", 10, 60, 30, 5)
            model_params = dict(
                n_clusters=n_clusters, linkage=linkage_method, metric=metric,
            )

        else:  # PCA
            max_components = max(2, min(len(feature_cols), 12) if feature_cols else 2)
            n_components = st.slider(
                "Number of components", 2, max_components, min(2, max_components)
            )
            model_params = dict(n_components=n_components)

        run_button = st.button("Run model", use_container_width=True, type="primary")


# -------------------- Main body --------------------
st.title("Unsupervised Machine Learning Application")
st.markdown(
    "Explore clustering and dimensionality reduction interactively. "
    "Choose the **Housing Prices** sample data (from Kaggle) or upload your own CSV, "
    "pick features, and tune the model from the sidebar."
    "Look at the Model info tab for detailed information explaining hyperparameters."
)

if df_raw is None:
    st.info("Please choose a data source in the sidebar.")
    st.stop()

tab_data, tab_model, tab_results = st.tabs(["Data preview", "Model info", "Results"])


# ---------- Data preview tab ----------
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df_raw):,}")
    c2.metric("Columns", len(df_raw.columns))
    c3.metric("Missing values", int(df_raw.isnull().sum().sum()))

    st.dataframe(df_raw.head(100), use_container_width=True)

    st.subheader("Column types & missing values")
    summary = pd.DataFrame({
        "dtype": df_raw.dtypes.astype(str),
        "missing": df_raw.isnull().sum(),
        "missing %": (df_raw.isnull().mean() * 100).round(1),
        "unique": df_raw.nunique(),
    })
    st.dataframe(summary, use_container_width=True)

    num_df = df_raw.select_dtypes(include="number")
    if not num_df.empty:
        st.subheader("Numeric feature correlation")
        fig_corr, ax_corr = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax_corr, cbar_kws={"shrink": 0.8})
        ax_corr.set_title("Correlation matrix")
        fig_corr.tight_layout()
        st.pyplot(fig_corr)
        plt.close(fig_corr)


# ---------- Model info tab ----------
with tab_model:
    # Quick orientation for users who are new to unsupervised learning
    with st.expander("Broad overview for beginners", expanded=False):
        st.markdown(
            "**Unsupervised learning** finds structure in data without a target/label column. "
            "Instead of predicting a known answer, the model groups similar rows together "
            "(**clustering**) or compresses many columns into a few summary axes "
            "(**dimensionality reduction**).\n\n"
            "- **K-Means** and **Hierarchical Clustering** attempt to find similarities across rows\n"
            "- **PCA** aims at discovering which combinations of features carry the most information\n\n"
            "Because there is no correct outcome or label to compare against, results are evaluated with "
            "internal metrics (silhouette score, inertia, explained variance) and by inspecting "
            "the clusters/components yourself to see whether they tell a sensible story."
        )

    # Long-form descriptions of each model. Kept inside expanders so the page
    # is approachable for newcomers but doesn't overwhelm experienced users.
    long_descs = {
        "K-Means Clustering": {
            "summary": (
                "Partitions samples into **k** clusters by iteratively assigning each point to the "
                "nearest centroid (cluster center) and updating centroids to the mean of their members. "
                "Choosing **k** is the main decision — use the **elbow plot** and **silhouette score** "
                "as guidance. K-Means is sensitive to feature scale, so standardization is on by default."
            ),
            "how": (
                "1. Pick **k** initial centroids (k-means++ chooses spread-out starting points).\n"
                "2. **Assign** each sample to its closest centroid (Euclidean distance).\n"
                "3. **Update** each centroid to the mean of the samples assigned to it.\n"
                "4. Repeat steps 2–3 until assignments stop changing or `max_iter` is hit.\n"
                "5. Because the result depends on the random starting centroids, scikit-learn runs "
                "the whole procedure `n_init` times and keeps the best one (lowest inertia)."
            ),
            "when": (
                "- You expect roughly **spherical, similarly-sized** clusters.\n"
                "- You have a rough idea of how many groups exist, or you're willing to scan a range of k.\n"
                "- Your features are numeric and on comparable scales after standardization."
            ),
            "watchouts": (
                "- Struggles with elongated, nested, or very uneven clusters.\n"
                "- Very sensitive to feature scaling — always standardize first.\n"
                "- Outliers can pull centroids; consider removing or winsorizing extreme values."
            ),
        },
        "Hierarchical Clustering": {
            "summary": (
                "Agglomerative clustering builds a **tree of merges**: every sample starts in its own "
                "cluster, and at each step the two closest clusters are joined. The **dendrogram** "
                "visualizes that merge history; \"cutting\" it at a chosen height (or number of "
                "clusters) gives the final assignment."
            ),
            "how": (
                "1. Treat each of the *n* samples as its own cluster.\n"
                "2. Compute pairwise distances between clusters using the chosen **linkage rule**.\n"
                "3. **Merge** the two closest clusters into one.\n"
                "4. Repeat until only one cluster remains, recording every merge in the dendrogram.\n"
                "5. Cut the tree at the level that produces the requested **n_clusters**."
            ),
            "when": (
                "- You want to see *structure at multiple scales* (the dendrogram), not just one fixed k.\n"
                "- Cluster shapes may be non-spherical, or you don't know k in advance.\n"
                "- Your dataset is small-to-medium (it's O(n²) in memory, so very large n is slow)."
            ),
            "watchouts": (
                "- `ward` linkage **requires Euclidean distance** — other metrics will be ignored or error.\n"
                "- `single` linkage can produce stringy \"chains\" of points; `complete` tends to make compact balls.\n"
                "- Once two clusters are merged they can never be split, so early mistakes propagate."
            ),
        },
        "Principal Component Analysis": {
            "summary": (
                "PCA is an unsupervised **linear transform** that finds new orthogonal axes "
                "(\"components\") capturing as much variance in the data as possible. PC1 captures "
                "the most variance, PC2 the next-most while being uncorrelated with PC1, and so on. "
                "Useful for visualization, noise reduction, and decorrelating features before another model."
            ),
            "how": (
                "1. **Center** (and usually standardize) the features so each has mean 0.\n"
                "2. Compute the **covariance matrix** of the features.\n"
                "3. Find its **eigenvectors** (the component directions) and **eigenvalues** (variance "
                "explained by each direction).\n"
                "4. Sort components from most to least variance and keep the top **n_components**.\n"
                "5. **Project** the original data onto those axes — each new column is a weighted "
                "combination of the originals (the **loadings**)."
            ),
            "when": (
                "- You have many correlated numeric features and want a compact summary.\n"
                "- You need a 2-D or 3-D view of high-dimensional data for plotting.\n"
                "- You want to feed decorrelated, lower-dimensional features into another model."
            ),
            "watchouts": (
                "- Components are **linear** combinations — non-linear structure may need t-SNE or UMAP.\n"
                "- Components are not always interpretable; check the loadings before naming them.\n"
                "- Standardize first if your features are on different scales (price vs. bedrooms)."
            ),
        },
    }

    st.subheader(model_name)
    info = long_descs[model_name]
    st.markdown(info["summary"])

    c_how, c_when = st.columns(2)
    with c_how:
        with st.expander("Algorithm steps", expanded=False):
            st.markdown(info["how"])
    with c_when:
        with st.expander("Use cases and potential issues", expanded=False):
            st.markdown("**Good fit indicators:**")
            st.markdown(info["when"])
            st.markdown("**Potential problems:**")
            st.markdown(info["watchouts"])

    # --- Hyperparameter guide --------------------------------------------------
    st.subheader("Hyperparameter guide")
    
    if model_name == "K-Means Clustering":
        hp_md = (
            "**`n_clusters` (k) — Number of clusters**  \n"
            "How many groups K-Means will carve the data into. This is the single most important "
            "choice. Too small and distinct groups get merged; too large and you fragment a real "
            "group into noise. Use the *Elbow & silhouette* plot in the **Results** tab to compare "
            "candidate values and look for a clear bend in the inertia curve and a high silhouette score.\n\n"
            "**`n_init` — Random restarts**  \n"
            "K-Means depends on its random initialization, so scikit-learn runs the whole fit "
            "`n_init` times with different starting centroids and keeps the best result. Higher "
            "values give a more stable answer at the cost of run time. `10` is the default and is "
            "usually plenty; bump it to `20` for noisy or high-dimensional data.\n\n"
            "**`max_iter` — Maximum iterations per run**  \n"
            "Caps how many assign-update cycles a single run will perform before giving up. The "
            "default `300` is almost always more than enough; convergence usually happens much "
            "earlier. Increase only if you see warnings about not converging.\n\n"
            "**`random_state` — Random seed**  \n"
            "Controls the random initialization so results are **reproducible**. Change it to test "
            "whether your clusters are stable across different starts — if labels jump around a lot, "
            "the clustering structure is weak.\n\n"
            "**`elbow_max` — Largest k to scan in the elbow plot**  \n"
            "Only affects the diagnostic *Elbow & silhouette* chart, not the final model. Pick a "
            "value a bit larger than the k you're considering so you can see the curve flatten."
        )

    elif model_name == "Hierarchical Clustering":
        hp_md = (
            "**`n_clusters` — Number of clusters**  \n"
            "Where to cut the dendrogram tree. The model itself doesn't really need this — it builds "
            "the full hierarchy regardless — but a single labeling has to pick a level. Use the "
            "dendrogram in the **Results** tab to look for a height where there is a *large jump* "
            "between successive merges; cutting just below that jump usually gives a natural number "
            "of clusters.\n\n"
            "**`linkage` — How cluster distance is measured**  \n"
            "Decides which two clusters are merged at each step:\n"
            "- `ward` minimizes the increase in within-cluster variance. Tends to produce "
            "compact, similarly-sized clusters. **Requires Euclidean distance.**\n"
            "- `complete` uses the *farthest* pair of points between clusters. Compact clusters, "
            "but sensitive to outliers.\n"
            "- `average` uses the mean pairwise distance. A balanced middle ground.\n"
            "- `single` uses the *nearest* pair. Can detect long, snaky shapes but is prone to "
            "\"chaining\" through noise.\n\n"
            "**`metric` — Distance metric**  \n"
            "How the distance between two individual points is measured (only configurable when "
            "linkage is not `ward`):\n"
            "- `euclidean` — straight-line distance, the usual default.\n"
            "- `manhattan` — sum of absolute differences; more robust to outliers.\n"
            "- `cosine` — angle between feature vectors; useful when *direction* matters more than "
            "magnitude (e.g. text or sparse data).\n\n"
            "**`max_leaves` — Dendrogram detail**  \n"
            "Only affects the visualization. The full dendrogram has *n* leaves which is unreadable "
            "for large datasets, so we show only the last *N* merges. Lower values are easier to "
            "read; higher values reveal more fine-grained structure."
        )

    else:  # PCA
        hp_md = (
            "**`n_components` — Number of components to keep**  \n"
            "How many new axes to compute. PCA produces up to `min(n_samples, n_features)` "
            "components, ranked from most to least variance explained. Common strategies:\n"
            "- **Visualization:** keep `2` (or `3`) so you can plot the data.\n"
            "- **Compression:** keep enough components to explain ~80–95% cumulative variance — "
            "see the cumulative-variance curve in the **Results** tab.\n"
            "- **Downstream modeling:** experiment; more components retain more signal but also "
            "more noise.\n\n"
            "Note that **standardization** (in the *Features* section of the sidebar) is also a "
            "critical decision for PCA: without it, features with large numeric ranges (like `price`) "
            "will dominate the components purely because of their scale."
        )

    st.markdown(hp_md)

    st.subheader("Current hyperparameter values")
    st.json(model_params)

    st.subheader("Selected features")
    if feature_cols:
        st.write(", ".join(f"`{f}`" for f in feature_cols))
        st.caption(
            f"{len(feature_cols)} feature(s) selected · "
            f"standardization is **{'on' if scale else 'off'}**."
        )
    else:
        st.warning("No features selected yet.")


# ---------- Results tab ----------
with tab_results:
    if not run_button:
        st.info("Configure the sidebar and click **Run model** to see results.")
        st.stop()

    if not feature_cols or len(feature_cols) < 2:
        st.error("Please select at least two feature columns.")
        st.stop()

    with st.spinner("Preprocessing and fitting…"):
        X, feature_names = preprocess(df_raw, feature_cols, scale=scale)

        if X.shape[0] < 5:
            st.error("Not enough rows after dropping missing values.")
            st.stop()

        X2d, var_ratio = project_2d(X)

    # ---- K-Means ----
    if model_name == "K-Means Clustering":
        km = KMeans(**model_params)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if len(np.unique(labels)) >= 2 else np.nan

        st.success("Clustering complete!")

        m1, m2, m3 = st.columns(3)
        m1.metric("k (clusters)", model_params["n_clusters"])
        m2.metric("Inertia", f"{km.inertia_:,.1f}")
        m3.metric("Silhouette", f"{sil:.3f}" if not np.isnan(sil) else "n/a")

        with st.expander("How to read these metrics", expanded=False):
            st.markdown(
                "- **k (clusters)** — the number of groups K-Means produced (your choice).\n"
                "- **Inertia** — the total squared distance from every point to its assigned "
                "centroid. Lower is tighter, though inertia always drops as k grows, so use it to "
                "compare different k values, not as an absolute quality score.\n"
                "- **Silhouette score** — runs from **−1 to 1**:\n"
                "  - `> 0.5` → strong, well-separated clusters.\n"
                "  - `0.25 – 0.5` → reasonable structure.\n"
                "  - `< 0.25` → weak or overlapping clusters; consider a different k or model.\n"
                "  - `< 0` → many points are closer to a *different* cluster than their own."
            )

        st.subheader("2-D cluster view & per-sample silhouette")
        col_a, col_b = st.columns(2)
        with col_a:
            st.pyplot(plot_clusters_2d(X2d, labels, var_ratio,
                                       f"K-Means clusters (k={model_params['n_clusters']})"))
            plt.close("all")
            st.caption(
                "Your features have been compressed to 2 dimensions with PCA so the clusters "
                "can be plotted. The percentages on each axis show how much of the original "
                "variance that axis preserves. Note that if they're low (e.g. < 40% combined), the "
                "picture only tells part of the story."
            )
        with col_b:
            sil_fig, _ = plot_silhouette(X, labels)
            if sil_fig is not None:
                st.pyplot(sil_fig)
                plt.close(sil_fig)
                st.caption(
                    "Each horizontal bar is one sample's silhouette coefficient. **Wide, "
                    "uniformly-tall blocks reaching to the right of the red mean line** are a "
                    "good sign. A cluster whose bars cross zero contains points that probably "
                    "belong elsewhere."
                )

        st.subheader("Elbow & silhouette across k")
        fig_elbow, inertias, sils = plot_elbow(X, range(2, elbow_max + 1))
        st.pyplot(fig_elbow)
        plt.close(fig_elbow)
        st.caption(
            "**Inertia** (blue) always decreases as k grows — look for an *elbow* where the "
            "drop sharply flattens; that k is often a good choice. **Silhouette** (orange) "
            "measures separation; pick a k where it peaks. When the two disagree, prefer the k "
            "where the silhouette is high *and* you're past the elbow."
        )

        st.subheader("Cluster sizes")
        sizes = pd.Series(labels).value_counts().sort_index()
        sizes.index = [f"Cluster {i}" for i in sizes.index]
        st.bar_chart(sizes)
        st.caption(
            "Roughly balanced clusters are common with K-Means. A very tiny cluster (a handful "
            "of points) often means outliers were grouped together and are worth inspecting. A "
            "single huge cluster swallowing most rows usually signals that k is too small."
        )

        st.subheader("Cluster profile (mean of each feature, original scale)")
        profile = (
            df_raw[feature_cols]
            .copy()
            .assign(cluster=labels)
            .groupby("cluster")
            .mean(numeric_only=True)
        )
        st.dataframe(profile.style.background_gradient(cmap="Blues"),
                     use_container_width=True)
        st.caption(
            "Each row is a cluster; each column is the **average value** of that feature for "
            "rows in the cluster (in the original units, before standardization). Compare rows "
            "to give each cluster a name. For example, *\"large, expensive homes with parking\"* "
            "vs. *\"small, budget homes with no AC\"*. Darker blue = higher value within the column."
        )

        out = df_raw.loc[df_raw[feature_cols].dropna().index].copy()
        out["cluster"] = labels
        st.download_button(
            "Download data with cluster labels (CSV)",
            out.to_csv(index=False).encode(),
            "kmeans_labeled.csv",
            "text/csv",
        )

    # ---- Hierarchical ----
    elif model_name == "Hierarchical Clustering":
        # AgglomerativeClustering uses different argument names by sklearn version
        try:
            agg = AgglomerativeClustering(
                n_clusters=model_params["n_clusters"],
                linkage=model_params["linkage"],
                metric=model_params["metric"],
            )
            labels = agg.fit_predict(X)
        except TypeError:
            agg = AgglomerativeClustering(
                n_clusters=model_params["n_clusters"],
                linkage=model_params["linkage"],
                affinity=model_params["metric"],
            )
            labels = agg.fit_predict(X)

        sil = silhouette_score(X, labels) if len(np.unique(labels)) >= 2 else np.nan

        st.success("Clustering complete!")

        m1, m2, m3 = st.columns(3)
        m1.metric("Clusters", model_params["n_clusters"])
        m2.metric("Linkage", model_params["linkage"])
        m3.metric("Silhouette", f"{sil:.3f}" if not np.isnan(sil) else "n/a")

        with st.expander("How to read these metrics", expanded=False):
            st.markdown(
                "- **Clusters** — number of groups produced by cutting the dendrogram.\n"
                "- **Linkage** — the rule used to decide which clusters were merged (see the "
                "*Model info* tab for what each option means).\n"
                "- **Silhouette score** — same scale as for K-Means: above `0.5` is strong, "
                "`0.25–0.5` is reasonable, below `0.25` indicates weak structure. Negative "
                "values mean many points sit closer to another cluster than their own."
            )

        st.subheader("2-D cluster view & per-sample silhouette")
        col_a, col_b = st.columns(2)
        with col_a:
            st.pyplot(plot_clusters_2d(X2d, labels, var_ratio,
                                       f"Hierarchical ({model_params['linkage']}, "
                                       f"{model_params['n_clusters']} clusters)"))
            plt.close("all")
            st.caption(
                "PCA is used to compress the features down to 2 axes for plotting. Look for "
                "clusters that occupy distinct regions, as overlap between colors is a sign that "
                "the chosen linkage or number of clusters may not match the data's structure."
            )
        with col_b:
            sil_fig, _ = plot_silhouette(X, labels)
            if sil_fig is not None:
                st.pyplot(sil_fig)
                plt.close(sil_fig)
                st.caption(
                    "Each bar is one sample's silhouette coefficient. Tall, uniformly wide "
                    "blocks pushed to the right of the red mean line indicate a healthy "
                    "cluster; narrow or negative bars suggest borderline assignments."
                )

        st.subheader("Dendrogram")
        # `ward` requires Euclidean distance for linkage as well
        link_method = model_params["linkage"]
        fig_dend = plot_dendrogram(X, link_method, max_leaves=max_leaves)
        st.pyplot(fig_dend)
        plt.close(fig_dend)
        st.caption(
            "The dendrogram shows the order in which clusters were merged from the bottom up. "
            "**The vertical height of each merge = the distance between the two clusters that "
            "joined.** A *long vertical line* means very different clusters were joined — those "
            "are natural \"cut points\". Drawing a horizontal line across the tree at any height "
            "and counting the lines it crosses tells you how many clusters you'd get at that "
            "level. Numbers in parentheses on the x-axis count how many original samples each "
            "leaf represents (because we truncated to the last "
            f"{max_leaves} merges for readability)."
        )

        st.subheader("Cluster sizes")
        sizes = pd.Series(labels).value_counts().sort_index()
        sizes.index = [f"Cluster {i}" for i in sizes.index]
        st.bar_chart(sizes)
        st.caption(
            "Hierarchical clustering, especially with `single` linkage, can produce **very "
            "uneven** cluster sizes (one giant cluster and several tiny ones). That isn't "
            "necessarily wrong, but it's worth checking whether the small clusters represent "
            "meaningful subgroups or just outliers."
        )

        out = df_raw.loc[df_raw[feature_cols].dropna().index].copy()
        out["cluster"] = labels
        st.download_button(
            "Download data with cluster labels (CSV)",
            out.to_csv(index=False).encode(),
            "hierarchical_labeled.csv",
            "text/csv",
        )

    # ---- PCA ----
    else:
        pca = PCA(n_components=model_params["n_components"], random_state=0)
        Z = pca.fit_transform(X)

        st.success("Dimensionality reduction complete!")

        m1, m2, m3 = st.columns(3)
        m1.metric("Components", model_params["n_components"])
        m2.metric("PC1 variance", f"{pca.explained_variance_ratio_[0]*100:.1f}%")
        cumv = float(pca.explained_variance_ratio_.sum()) * 100
        m3.metric("Cumulative variance", f"{cumv:.1f}%")

        with st.expander("How to read these metrics", expanded=False):
            st.markdown(
                "- **Components** — how many new axes you asked PCA to keep.\n"
                "- **PC1 variance** — the share of total variance captured by the *first* "
                "component alone. A high value (e.g. > 50%) means a single axis already "
                "summarizes most of what the features were saying.\n"
                "- **Cumulative variance** — total variance explained by *all* the components "
                "you kept. As a rule of thumb, **80–95%** is good for downstream modeling, "
                "and around **70%** is acceptable for visualization."
            )

        st.subheader("Variance explained & component loadings")
        col_a, col_b = st.columns(2)
        with col_a:
            st.pyplot(plot_pca_variance(pca))
            plt.close("all")
            st.caption(
                "**Bars** show the variance captured by each individual component; the **orange "
                "line** is the running total. Look for an *elbow* in the bars, as components after "
                "it add little new information. The cumulative line is what you compare to the "
                "70 / 80 / 95% targets above."
            )
        with col_b:
            st.pyplot(plot_pca_loadings(pca, feature_names))
            plt.close("all")
            st.caption(
                "Each arrow is one original feature, drawn in the PC1 (x) / PC2 (y) plane. "
                "**Arrow direction** is which combination of components that feature aligns with; "
                "**length** is how strongly the feature contributes. Arrows pointing the same way "
                "are positively correlated; arrows pointing opposite ways are negatively "
                "correlated; arrows at right angles are roughly independent."
            )

        st.subheader("Projection onto first two principal components")
        fig_proj, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(Z[:, 0], Z[:, 1], s=22, alpha=0.75, color="#4c78a8")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title("Samples projected onto PC1 vs PC2")
        fig_proj.tight_layout()
        st.pyplot(fig_proj)
        plt.close(fig_proj)
        st.caption(
            "Each dot is one row of your data, re-plotted in the new PC1/PC2 coordinate system. "
            "**Visible groupings, gradients, or outliers** here suggest natural structure that "
            "could be picked up by a clustering model. A diffuse blob with no pattern means PC1 "
            "and PC2 don't separate the data well. If this happens, try keeping more components, or look at the "
            "loadings to understand what PC1 and PC2 actually represent."
        )

        st.subheader("Component loadings")
        loadings = pd.DataFrame(
            pca.components_,
            columns=feature_names,
            index=[f"PC{i+1}" for i in range(pca.n_components_)],
        ).T
        st.dataframe(loadings.style.background_gradient(cmap="coolwarm", axis=None),
                     use_container_width=True)
        st.caption(
            "Each cell is the **weight** of an original feature (rows) in a principal component "
            "(columns). Values range from roughly −1 to +1: **large positive (red)** means the "
            "feature pushes the component up; **large negative (blue)** pushes it down; values "
            "near zero mean the feature barely contributes. Use this table to *name* each "
            "component — e.g. if PC1 has large positive weights on `price`, `area`, and "
            "`bathrooms`, you might call PC1 a *\"size & expense\"* axis."
        )

        out_idx = df_raw[feature_cols].dropna().index
        out = pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(pca.n_components_)],
                           index=out_idx)
        st.download_button(
            "Download principal components (CSV)",
            out.to_csv(index=True).encode(),
            "pca_components.csv",
            "text/csv",
        )
