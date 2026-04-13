# MLStreamlitApp

This is an interactive supervised machine learning web app built with Python and Streamlit. Users can upload their own dataset or explore the built-in Census sample dataset. The app offers three different models for the user to explore, and all parameters are adjustable within the app to suit different specifications.

## Live App

> **[Launch the app →](https://meffe-data-science-portfolio-9qwoe8cgyr3titw8wxd7ye.streamlit.app/)**  

## Project Overview

This app was built as a portfolio project to demonstrate machine learning skills, from data ingestion and preprocessing through model training, evaluation, and deployment. The goal was to make an accessible and interactive machine learning application for any user, regardless of coding background.

The built-in sample is the **UCI Adult Census Income dataset** — a real-world benchmark with demographic and employment features used to predict whether an individual earns above or below $50K per year.

## App Features

### Data layer
- **Upload any CSV** for custom datasets
- **Built-in Census sample** loaded automatically from the UCI ML Repository (48,842 rows, 14 features)
- Auto-detect column types; summary of missing values and unique counts
- Interactive target distribution bar chart

### Models supported

| Model | Key parameters |
|---|---|
| Logistic Regression | Regularization C, solver, max iterations |
| Decision Tree | Max depth, min samples to split, criterion (Gini / Entropy) |
| K-Nearest Neighbors | k (neighbors), weight function, distance metric |

All models use scikit-learn implementations. Features are automatically standardized (StandardScaler) for Logistic Regression and KNN; the Decision Tree works on raw values.

### Paremeter tuning
All paramter are controlled via sidebar sliders and dropdowns. Changes take effect on the next "Train model" click, allowing users to compare results across configurations.

### Performance metrics
- Accuracy, Precision, Recall, F1 score (binary or weighted average)
- Confusion matrix heatmap
- ROC curve + AUC score (binary classification)
- Feature importances / absolute coefficients bar chart
- Full `sklearn` classification report
- Download predictions as CSV

## Sample dataset description

The **Adult Income Dataset** comes from the 1994 US Census database.

| Property | Value |
|---|---|
| Source | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/adult) |
| Features | 14 (mix of numeric and categorical) |
| Target | `income` — `<=50K` or `>50K` |
| Class balance | ~76% ≤50K, ~24% >50K |

Selected features used by default: `age`, `education_num`, `hours_per_week`, `capital_gain`, `capital_loss`, `sex`, `race`, `workclass`.

## Instructions

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone the portfolio repo
git clone https://github.com/finnmeffe/MEFFE-Data-Science-Portfolio.git
cd MEFFE-Data-Science-Portfolio/MLStreamlitApp

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

### Dependencies

| Library | Version | Purpose |
|---|---|---|
| streamlit | ≥ 1.32 | Web UI framework |
| pandas | ≥ 2.0 | Data loading and manipulation |
| numpy | ≥ 1.26 | Numeric operations |
| scikit-learn | ≥ 1.4 | ML models and metrics |
| matplotlib | ≥ 3.8 | Plotting |
| seaborn | ≥ 0.13 | Heatmaps |

## References

- [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Community Cloud Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)