# Palmer Penguins EDA App

An interactive **exploratory data analysis** web app built with Python, Streamlit, and Plotly. The app lets users filter the Palmer Penguins dataset on species, island, and sex, and explore the data through summary statistics, distributions, and feature relationships — entirely from a clean tabbed interface, with no code required.

## Live App

> **[Launch the app →](https://meffe-data-science-portfolio-9qwoe8cgyr3titw8wxd7ye.streamlit.app/)**

## Project Overview

This app was built as an introductory portfolio project to demonstrate end-to-end Streamlit development: ingesting a real-world dataset, building reactive sidebar filters, organizing analysis into tabs, and producing publication-quality interactive charts. The goal was to make exploratory data analysis accessible to any user — letting them filter and visualize the data themselves instead of reading static plots from a notebook.

The dataset is the **Palmer Penguins dataset**, a popular alternative to the classic Iris dataset for teaching data science. It contains body measurements for 344 penguins across three species (Adelie, Chinstrap, Gentoo) sampled from three islands in the Palmer Archipelago, Antarctica. Data were collected by **Dr. Kristen Gorman** and the **Palmer Station, Antarctica LTER**.

## App Features

### Data layer
- Loads the bundled `data/penguins.csv` (344 rows, 7 columns) at startup
- Robust to missing values: `dropna()` is applied per filter so toggles never break the app
- Real-time filtering: every chart and metric updates instantly when sidebar filters change

### Sidebar filters
- **Species** (Adelie / Chinstrap / Gentoo) — multi-select
- **Island** (Biscoe / Dream / Torgersen) — multi-select
- **Sex** (male / female) — multi-select
- Project description info box at the bottom of the sidebar

### Tabs

| Tab | What it shows |
|---|---|
| **Overview** | High-level metrics (total penguins, species count, island count), `df.describe()` summary statistics, and an optional toggle to view the full raw dataset |
| **Distributions** | Pie chart of the species mix in the current filter, plus a histogram (with box-plot marginal) of any selected numeric feature, colored by species |
| **Relationships** | Scatter plot of any two numeric features against each other, colored by species and shaped by sex — designed for spotting clusters and inter-species differences |

All visualizations are built with **Plotly Express**, so they support hover-tooltips, zoom, pan, and PNG export out of the box.

## Sample dataset

| Property | Value |
|---|---|
| Source | [Palmer Penguins Dataset](https://allisonhorst.github.io/palmerpenguins/) |
| Original collectors | Dr. Kristen Gorman & Palmer Station, Antarctica LTER |
| Rows | 344 |
| Numeric features | `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g` |
| Categorical features | `species`, `island`, `sex` |

| Column | Description |
|---|---|
| `species` | Penguin species (Adelie, Chinstrap, Gentoo) |
| `island` | Island where the penguin was observed (Biscoe, Dream, Torgersen) |
| `bill_length_mm` | Bill length in millimeters |
| `bill_depth_mm` | Bill depth in millimeters |
| `flipper_length_mm` | Flipper length in millimeters |
| `body_mass_g` | Body mass in grams |
| `sex` | Penguin sex (male / female) |

## Project Structure

```
basic_streamlit_app/
├── main.py            # Streamlit application
├── README.md
└── data/
    └── penguins.csv   # Palmer Penguins dataset (bundled)
```

## Instructions

### Prerequisites
- Python 3.8 or higher
- pip

### Run locally

```bash
# 1. Clone the portfolio repo
git clone https://github.com/finnmeffe/MEFFE-Data-Science-Portfolio.git
cd MEFFE-Data-Science-Portfolio/basic_streamlit_app

# 2. Install dependencies
pip install streamlit pandas plotly

# 3. Launch the app
streamlit run main.py
```

The app opens automatically at `http://localhost:8501`.

### Deploy to Streamlit Community Cloud
1. Push the repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), pick the repo, and set the main file to `basic_streamlit_app/main.py`.
3. Streamlit Cloud will install dependencies and host the app at a public URL.

### Dependencies

| Library | Version | Purpose |
|---|---|---|
| streamlit | ≥ 1.32 | Web UI framework |
| pandas | ≥ 2.0 | Data loading and filtering |
| plotly | ≥ 5.18 | Interactive charts (pie, histogram, scatter) |

## References

- [Palmer Penguins Dataset](https://allisonhorst.github.io/palmerpenguins/)
- [Gorman, Williams & Fraser (2014) — *Ecological Sexual Dimorphism…*](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0090081)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Plotly Express Documentation](https://plotly.com/python/plotly-express/)
- [Streamlit Community Cloud Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)
