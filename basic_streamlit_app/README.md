# Palmer Penguins EDA Streamlit App
 
An interactive Exploratory Data Analysis (EDA) web app built with Streamlit, using the Palmer Penguins dataset. The app lets you filter and visualize penguin data across species, islands, and sex through a clean tabbed interface.
 
## Features
 
- Sidebar Filter: Filter the dataset by species, island, and sex using multi-select dropdowns. All charts and statistics update in real time.
- Overview tab: High-level summary metrics (total penguins, number of species, number of islands), descriptive statistics, and an optional view of the raw data.
- Distributions tab: A species population pie chart alongside a histogram (with box plot overlay) for any selected numeric feature.
- Relationships tab: An interactive scatter plot for exploring correlations between any two numeric features, colored by species and shaped by sex.

## Project Structure
```
├── app.py            # Main Streamlit application
└── data/
    └── penguins.csv  # Palmer Penguins dataset
```
 
## Requirements
 
- Python 3.8+
- streamlit
- pandas
- plotly
Install dependencies with:
 
```bash
pip install streamlit pandas plotly
```
 
## Running the App
 
```bash
streamlit run app.py
```
 
## Dataset
 
The Palmer Penguins dataset contains measurements for 344 penguins across three species (Adelie, Chinstrap, Gentoo) from three islands in the Palmer Archipelago, Antarctica. Data were collected and made available by **Dr. Kristen Gorman** and the **Palmer Station, Antarctica LTER**.
 
| Column | Description |
|---|---|
| `species` | Penguin species (Adelie, Chinstrap, Gentoo) |
| `island` | Island where the penguin was observed |
| `bill_length_mm` | Bill length in millimeters |
| `bill_depth_mm` | Bill depth in millimeters |
| `flipper_length_mm` | Flipper length in millimeters |
| `body_mass_g` | Body mass in grams |
| `sex` | Penguin sex (male/female) |
