import streamlit as st
import pandas as pd
import plotly.express as px

# set global options
st.set_page_config(layout="wide")
st.title("EDA using Palmer's Penguins Dataset")
df = pd.read_csv("data\penguins.csv")

# create filters on side
st.sidebar.header("Filters")
species_filter = st.sidebar.multiselect(
    "Species",
    options=df['species'].dropna().unique(),
    default=df['species'].dropna().unique()
)

island_filter = st.sidebar.multiselect(
    "Island",
    options=df['island'].dropna().unique(),
    default=df['island'].dropna().unique()
)

sex_filter = st.sidebar.multiselect(
    "Sex",
    options=df['sex'].dropna().unique(),
    default=df['sex'].dropna().unique()
)

# return filtered df according to selects above
df_filtered = df[
    (df['species'].isin(species_filter)) &
    (df['island'].isin(island_filter)) &
    (df['sex'].isin(sex_filter))
]

# create separate tabs for different types of analysis
tab1, tab2, tab3 = st.tabs(["Overview", "Distributions", "Relationships"])

with tab1:
    st.header("Dataset Overview")
    
    # give some basic information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Penguins", len(df_filtered))
    with col2:
        st.metric("Species", df_filtered['species'].nunique())
    with col3:
        st.metric("Islands", df_filtered['island'].nunique())
    
    st.subheader("Summary Statistics")
    st.dataframe(df_filtered.describe())

    if st.button("See complete raw data"):
        st.dataframe(df)

with tab2:
    st.header("Feature Distributions")

    col1, col2 = st.columns(2)

    # pie chart for population
    with col1: 
        species_counts = df_filtered['species'].value_counts()
        fig = px.pie(values=species_counts.values, names=species_counts.index, 
                 title="Species Population Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # histogram for different features
    with col2:
        feature = st.selectbox("Select Feature", 
                              ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])
        
        fig = px.histogram(df_filtered, x=feature, color='species',
                          marginal='box', 
                          title=f"Distribution of {feature}")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Relationships Between Features")
    st.text("This tab can be used to view differences in features across species")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox("X-axis", 
                            ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'],
                            key='x_var')
    with col2:
        y_var = st.selectbox("Y-axis", 
                            ['bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'bill_length_mm'],
                            key='y_var')
    
    # create a scatter plot that can demonstrate trends by species
    fig = px.scatter(df_filtered, x=x_var, y=y_var, 
                     color='species', symbol='sex',
                     title=f"{x_var} vs {y_var}")
    st.plotly_chart(fig, use_container_width=True)


# create basic description
st.sidebar.info(
    "This app explores the Palmer Penguins dataset, which contains various data on penguins collected in Antartica. Data were collected and made available by Dr. Kristen Gorman and the Palmer Station, Antarctica LTER"
)