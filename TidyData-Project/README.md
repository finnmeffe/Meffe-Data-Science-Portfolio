# Federal R&D Spending — Tidy Data Project

A data-cleaning and exploratory analysis project that takes a deliberately messy dataset of US federal research-and-development spending and reshapes it into a long, tidy format following Hadley Wickham's tidy-data principles. The cleaned data is then used to produce four visualizations of how R&D spending has evolved by department from 1976 to 2017.

## Project Overview

The raw dataset is a wide table with **one row per department** and **one column per year**, where the year columns also have the national GDP for that year mashed into the column name (e.g. `1976_gdp1790000000000.0`). This makes it nearly impossible to plot or aggregate the data without first cleaning it.

The notebook walks through a complete tidy-data workflow in detail:

1. Inspect the raw CSV and identify the structural problems.
2. Reshape the data with `pd.melt` so every row is a single department–year observation.
3. Split the encoded year/GDP column into separate `year` and `gdp` columns and cast each to its proper dtype.
4. Aggregate the tidy data with pivot tables and `groupby`.
5. Visualize budget trends with four charts that progressively reveal where federal R&D money goes.

### Tidy data principles applied

- Each variable is in its own column.  
- Each observation forms its own row.  
- Each type of observational unit forms its own table.

After cleaning, the dataset has exactly four columns — `department`, `year`, `budget`, `gdp` — with one row per department-year pair (588 rows total). This shape is what makes every downstream visualization considerably easier.

## Visualizations

| # | Chart | Description |
|---|---|---|
| 1 | **Total R&D budget by year** | A line chart of summed federal R&D spending across all departments, 1976–2017. Reveals the long climb through the 1990s, the post-2000 plateau, and the post-2015 dip. |
| 2 | **Total R&D budget by department** | A bar chart ranking departments by total spend across the full 42-year window. Shows how the **Department of Defense** dwarfs every other agency (~$2.6T), with HHS, NIH, and NASA following. |
| 3 | **Budget over time — top 5 departments** | A multi-line chart of how the top-5 departments' budgets evolved year-by-year. Demonstrates that DOD's volatility largely drives the aggregate trend in chart 1. |
| 4 | **Animated budget rankings** | A `matplotlib.animation` bar-race showing how department rankings shift across years — an alternative to a busy spaghetti plot for exploring the same data. |

## Dataset

| Property | Value |
|---|---|
| Source | [TidyTuesday — fed_r_d_spending.csv](https://github.com/rfordatascience/tidytuesday/blob/main/data/2019/2019-02-12/fed_r_d_spending.csv) |
| Local file | `data/fed_rd_year&gdp.csv` (an *un-tidied* version adapted for the cleaning exercise) |
| Years covered | 1976–2017 |
| Departments | 14 (DHS, DOC, DOD, DOE, DOT, EPA, HHS, Interior, NASA, NIH, NSF, Other, USDA, VA) |
| Tidy shape | 588 rows × 4 columns (`department`, `year`, `budget`, `gdp`) |

The local CSV is intentionally messier than the upstream TidyTuesday version so the project can demonstrate a non-trivial cleaning step rather than just pulling already-clean data.

## Project Structure

```
TidyData-Project/
├── tidydata.ipynb            # Notebook with all cleaning, analysis, and plots
├── README.md
└── data/
    └── fed_rd_year&gdp.csv   # Raw (untidy) federal R&D spending dataset
```

## Instructions

### Prerequisites
- Python 3.10 or higher
- pip
- A Jupyter-compatible environment (VS Code with the Python extension, classic Jupyter, JupyterLab, etc.)

### Run locally

```bash
# 1. Clone the portfolio repo
git clone https://github.com/finnmeffe/MEFFE-Data-Science-Portfolio.git
cd MEFFE-Data-Science-Portfolio/TidyData-Project

# 2. Install dependencies
pip install pandas matplotlib jupyter

# 3. Open the notebook
jupyter notebook tidydata.ipynb
```

In VS Code: open `tidydata.ipynb`, select your Python kernel, and run each cell from top to bottom. The animated bar-race in the final cell renders inline via `IPython.display.HTML`.

### Dependencies

| Library | Version | Purpose |
|---|---|---|
| pandas | ≥ 2.0 | Data loading, melting, pivoting, and grouping |
| matplotlib | ≥ 3.8 | Static plots and `matplotlib.animation` for the bar-race |
| jupyter / ipython | ≥ 8.0 | Notebook runtime and `HTML` display of the animation |

## References

- [Wickham, H. (2014). *Tidy Data*. Journal of Statistical Software.](https://www.jstatsoft.org/article/view/v059i10)
- [TidyTuesday — Federal R&D Spending (2019-02-12)](https://github.com/rfordatascience/tidytuesday/tree/main/data/2019/2019-02-12)
- [Pandas `melt` documentation](https://pandas.pydata.org/docs/reference/api/pandas.melt.html)
- [Pandas Cheat Sheet (PDF)](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [matplotlib.animation API reference](https://matplotlib.org/stable/api/animation_api.html)
