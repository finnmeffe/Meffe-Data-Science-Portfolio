import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Set the basic page configuration
st.set_page_config(
    page_title="Machine Learning Application",
    layout="wide",
)

# Create some functions that will help with the app
@st.cache_data

# Function for loading data
def load_census_data():
    # Get a sample dataset of US Census data
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    )
    # Fetch all the columns that are needed to predict income
    cols = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
    ]
    try:
        df = pd.read_csv(url, names=cols, sep=",", skipinitialspace=True, na_values="?")
    except Exception:
        # Fallback that creates random data (recommended so app doesn't crash)
        rng = np.random.default_rng(50)
        n = 500
        df = pd.DataFrame({
            "age": rng.integers(18, 90, n),
            "education_num": rng.integers(1, 16, n),
            "hours_per_week": rng.integers(1, 99, n),
            "capital_gain": rng.integers(0, 100000, n),
            "capital_loss": rng.integers(0, 4000, n),
            "sex": rng.choice(["Male", "Female"], n),
            "race": rng.choice(["White", "Black", "Asian-Pac-Islander", "Other"], n),
            "workclass": rng.choice(["Private", "Self-emp", "Gov", "Other"], n),
            "income": rng.choice(["<=50K", ">50K"], n),
        })
    return df


# Process the loaded data. Ensure that NA observations are dropped and encode categorical data
def preprocess(df: pd.DataFrame, target_col: str, feature_cols: list):
    subset = df[feature_cols + [target_col]].dropna().copy()
    le = LabelEncoder()
    for col in subset.select_dtypes(include="object").columns:
        subset[col] = le.fit_transform(subset[col].astype(str))
    X = subset[feature_cols].values
    y = subset[target_col].values
    return X, y

# Confusion matrix plot for results page
def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    return fig

# ROC plot 
def plot_roc(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, auc

# Feature importance plot for each of the model types
def plot_feature_importance(model, feature_names, model_name):
    fig, ax = plt.subplots(figsize=(5, 3))
    if model_name == "Decision Tree":
        importances = model.feature_importances_
        ax.barh(feature_names, importances, color="#4c78a8")
        ax.set_xlabel("Importance")
        ax.set_title("Feature importances")
    elif model_name == "Logistic Regression":
        coefs = np.abs(model.coef_[0])
        ax.barh(feature_names, coefs, color="#4c78a8")
        ax.set_xlabel("|Coefficient|")
        ax.set_title("Feature coefficients (abs)")
    else:
        ax.text(0.5, 0.5, "Feature importance\nnot available for KNN",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    return fig


# Create an interactive sidebar
with st.sidebar:
    st.title("Machine Learning Project")
    st.caption("Finn Meffe")

    st.divider()
    st.subheader("1. Data source")
    data_source = st.radio("Choose a data source",
                           ["Sample: US Census Income Data", "Upload your own CSV"],
                           label_visibility="collapsed")
    
    # Add option so that users can upload their own CSV
    df_raw = None
    if data_source == "Sample: US Census Income Data":
        with st.spinner("Loading Census data…"):
            df_raw = load_census_data()
        st.success(f"Loaded {len(df_raw):,} rows")
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df_raw = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df_raw):,} rows × {len(df_raw.columns)} columns")

    # Allow users to select the target cols and predictors
    if df_raw is not None:
        st.divider()
        st.subheader("2. Target & features")
        all_cols = df_raw.columns.tolist()
        target_col = st.selectbox("Target column", all_cols,
                                  index=len(all_cols) - 1)
        default_features = [c for c in all_cols if c != target_col][:6]
        feature_cols = st.multiselect("Feature columns", [c for c in all_cols if c != target_col],
                                      default=default_features)
        

        # Three model types
        st.divider()
        st.subheader("3. Model")
        model_name = st.selectbox(
            "Algorithm",
            ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors"]
        )

        test_size = st.slider("Test set size (%)", 10, 40, 20, 5) / 100
        random_state = st.number_input("Random seed", value=42, step=1)

        st.divider()
        st.subheader("4. Hyperparameters")

        # options for model parameters, take solvers from sklearn
        if model_name == "Logistic Regression":
            C = st.select_slider("Regularization C", options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
            max_iter = st.slider("Max iterations", 100, 2000, 500, 100)
            solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
            model_params = dict(C=C, max_iter=max_iter, solver=solver, random_state=int(random_state))
            model_obj = LogisticRegression(**model_params)

        elif model_name == "Decision Tree":
            max_depth = st.slider("Max depth", 1, 20, 5)
            min_samples_split = st.slider("Min samples to split", 2, 50, 2)
            criterion = st.selectbox("Criterion", ["gini", "entropy"])
            model_params = dict(max_depth=max_depth, min_samples_split=min_samples_split,
                                criterion=criterion, random_state=int(random_state))
            model_obj = DecisionTreeClassifier(**model_params)

        else:  # KNN
            n_neighbors = st.slider("k (neighbors)", 1, 30, 5)
            weights = st.selectbox("Weight function", ["uniform", "distance"])
            metric = st.selectbox("Distance metric", ["euclidean", "manhattan", "minkowski"])
            model_params = dict(n_neighbors=n_neighbors, weights=weights, metric=metric)
            model_obj = KNeighborsClassifier(**model_params)

        train_button = st.button("Train model", use_container_width=True, type="primary")


# Main body of the application
st.title("Machine Learning Application")
st.markdown(
    "Choose to upload your own dataset or use the sample of US Census data. "
    "Model parameters can be selected using the the sidebar"
)

if df_raw is None:
    st.info("Please choose a data source in the sidebar.")
    st.stop()

tab_data, tab_model, tab_results = st.tabs(["Preview", "Model info", "Results"])

# Create a basic data preview tab
with tab_data:
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df_raw):,}")
    col2.metric("Columns", len(df_raw.columns))
    col3.metric("Missing values", df_raw.isnull().sum().sum())

    st.dataframe(df_raw.head(100), use_container_width=True)

    st.subheader("Column types & missing values")
    summary = pd.DataFrame({
        "dtype": df_raw.dtypes,
        "missing": df_raw.isnull().sum(),
        "missing %": (df_raw.isnull().mean() * 100).round(1),
        "unique": df_raw.nunique(),
    })
    st.dataframe(summary, use_container_width=True)

    if target_col and target_col in df_raw.columns:
        st.subheader(f"Target distribution — `{target_col}`")
        vc = df_raw[target_col].value_counts()
        fig_dist, ax_dist = plt.subplots(figsize=(5, 2.5))
        vc.plot(kind="bar", ax=ax_dist, color=["#4c78a8", "#f58518"])
        ax_dist.set_xlabel("")
        ax_dist.set_title("Class counts")
        plt.xticks(rotation=30)
        fig_dist.tight_layout()
        st.pyplot(fig_dist)
        plt.close(fig_dist)

# Model info tab
with tab_model:
    # basic descriptions of the model
    descs = {
        "Logistic Regression": (
            "A linear model that estimates the probability of class using the logistic "
            "function. Works well for linearly separable data. The regularization parameter **C** "
            "controls the trade-off between fitting the training data and keeping coefficients small "
            "(lower C = stronger regularization)."
        ),
        "Decision Tree": (
            "A tree-shaped classifier that splits data by feature thresholds. "
            "**Max depth** limits how deep the tree grows (prevents overfitting). **Min samples split** "
            "is the minimum number of samples required to split an internal node."
        ),
        "K-Nearest Neighbors": (
            "Classifies each point by majority vote among its **k** nearest neighbors. No explicit "
            "training phase, inference computes distances at prediction time. Sensitive to feature "
            "scale, so features are automatically standardized before training."
        ),
    }
    st.subheader(model_name)
    st.markdown(descs[model_name])

    # Read out the parameters directly for a double check
    st.subheader("Current hyperparameters")
    st.json(model_params)

    st.subheader("Selected features")
    if feature_cols:
        st.write(", ".join(f"`{f}`" for f in feature_cols))
    else:
        st.warning("No features selected yet.")

# Results tab
with tab_results:
    # catch some basic errors
    if not train_button:
        st.info("Set your options in the sidebar and click **Train model** to see results.")
        st.stop()

    if not feature_cols:
        st.error("Please select at least one feature column.")
        st.stop()

    with st.spinner("Preprocessing and training…"):
        X, y = preprocess(df_raw, target_col, feature_cols)

        if len(np.unique(y)) < 2:
            st.error("Target column must have at least 2 unique classes.")
            st.stop()

        # Scale for KNN and Logistic Regression
        if model_name in ("K-Nearest Neighbors", "Logistic Regression"):
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state), stratify=y
        )
        model_obj.fit(X_train, y_train)
        y_pred = model_obj.predict(X_test)

        classes = np.unique(y)
        binary = len(classes) == 2
        avg = "binary" if binary else "weighted"
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
        rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)

    st.success("Training complete!")

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  f"{acc:.3f}")
    m2.metric("Precision", f"{prec:.3f}")
    m3.metric("Recall",    f"{rec:.3f}")
    m4.metric("F1 score",  f"{f1:.3f}")

    # Following visuals are generated using the helper functions above (reference)
    # Charts
    col_cm, col_roc = st.columns(2)

    with col_cm:
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = plot_confusion_matrix(cm, classes)
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with col_roc:
        if binary:
            try:
                if hasattr(model_obj, "predict_proba"):
                    y_prob = model_obj.predict_proba(X_test)[:, 1]
                else:
                    y_prob = model_obj.decision_function(X_test)
                fig_roc, auc_val = plot_roc(y_test, y_prob)
                st.pyplot(fig_roc)
                plt.close(fig_roc)
                st.caption(f"AUC-ROC: **{auc_val:.4f}**")
            except Exception as e:
                st.info(f"ROC curve unavailable: {e}")
        else:
            st.info("ROC curve is shown for binary classification only.")

    # Feature importance
    st.subheader("Feature importance / coefficients")
    fig_fi = plot_feature_importance(model_obj, feature_cols, model_name)
    st.pyplot(fig_fi)
    plt.close(fig_fi)

    # Full classification report
    with st.expander("Full classification report"):
        report = classification_report(y_test, y_pred, zero_division=0)
        st.code(report)

    # Option to download predictions
    pred_df = pd.DataFrame({"actual": y_test, "predicted": y_pred})
    csv_bytes = pred_df.to_csv(index=False).encode()
    st.download_button("Download predictions CSV", csv_bytes,
                       "predictions.csv", "text/csv")
