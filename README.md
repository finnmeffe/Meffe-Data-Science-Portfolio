# Data Science Portfolio

This collection of data science projects spanning exploratory data analysis, data cleaning, supervised machine learning, and unsupervised machine learning. Each project is self-contained with its own README and code. Together they trace a progression from foundational data wrangling through to interactive model deployment.

## Portfolio at a glance

| # | Project | Type | Packages used | 
|---|---|---|---|
| 1 | [Basic Streamlit App](./basic_streamlit_app) | Interactive EDA | Streamlit · pandas · Plotly |
| 2 | [Tidy Data Project](./TidyData-Project) | Data cleaning & analysis | pandas · matplotlib · Jupyter |
| 3 | [ML Streamlit App](./MLStreamlitApp) | Supervised ML | Streamlit · scikit-learn | 
| 4 | [ML Unsupervised App](./MLUnsupervisedApp) | Unsupervised ML | Streamlit · scikit-learn · scipy · kagglehub | 

---

## 1. [Basic Streamlit App — Palmer Penguins EDA](https://github.com/finnmeffe/Meffe-Data-Science-Portfolio/tree/main/basic_streamlit_app)

This project involved the construction of a basic interactive streamlit app to explore the "Palmer's Penguins" dataset. I focused on providing a detailed set of options from which to study the data, as understanding how data looks is a crucial step of data analysis. The goal was to make EDA accessible to non-coders by creating an intuitive UI to interact with the data. This also helped build the foundations of using a Streamlit app to present data analysis in a clean format to a broader audience.

---

## 2. [Tidy Data Project — Federal R&D Spending](https://github.com/finnmeffe/Meffe-Data-Science-Portfolio/tree/main/TidyData-Project)

The Tidy Data Project was created with the intention of applying tidy data principles to a raw dataset. The created notebook takes a deliberately messy dataset of US federal R&D spending (1976–2017) and reshapes it into a long, tidy format following Hadley Wickham's tidy-data principles. The cleaned data drives four progressively richer visualizations of how spending has evolved by department, including an animated `matplotlib.animation` bar-race. The ability to successfully tidy up data is a crucial part of the data science workflow, and this project helped me to understand and implement these principles.

---

## 3. [ML Streamlit App — Supervised Classification](https://github.com/finnmeffe/Meffe-Data-Science-Portfolio/tree/main/MLStreamlitApp)

**[Deployed App Link](https://meffe-data-science-portfolio-9qwoe8cgyr3titw8wxd7ye.streamlit.app/)**

The machine learning project involved the creation of a Streamlit app that provides an accessible platform to use basic machine learning models on user and sample data.  Users upload their own CSV or explore the bundled UCI Adult Census Income dataset, choose between logistic regression, decision tree, or K-nearest neighbors, tune every hyperparameter from the sidebar, and inspect performance through accuracy/precision/recall/F1, confusion matrices, ROC/AUC curves, and feature-importance bar charts. This app bridges multiple areas of data science, from creating and deploying an app to running and diagnosing machine learning models, and thus presents a step forward from previous projects in this repository.

---

## 4. [ML Unsupervised App — Clustering & Dimensionality Reduction](https://github.com/finnmeffe/Meffe-Data-Science-Portfolio/tree/main/MLStreamlitApp)

**[Deployed App Link](https://finnmeffe-meffe-data-s-mlunsupervisedappunsupervised-app-aptbsi.streamlit.app/)**

This is the unsupervised counterpart to the classification app. Users upload a CSV or explore the Kaggle Housing Prices sample (loaded live via the `kagglehub` API), pick features, and tune three different unsupervised models: K-Means clustering, hierarchical clustering, and PCA. A goal was to make unsupervised learning approachable, with every metric, plot, and table is explained inline. This helped teach me how to develop tooling around models, explaining when each algorithm is the right tool, what each knob does, and how to read the diagnostic plots.

## Repository structure

```
Meffe-Data-Science-Portfolio/
├── basic_streamlit_app/     # Palmer Penguins EDA app (Update 1)
├── TidyData-Project/        # Federal R&D tidy-data notebook (Update 2)
├── MLStreamlitApp/          # Supervised ML app (Update 3)
├── MLUnsupervisedApp/       # Unsupervised ML app (Update 4)
└── README.md                # This file
```
