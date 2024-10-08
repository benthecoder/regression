import os

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Explore",
    page_icon="📊",
)

st.write("# Explore your data!")

st.write("Choose a dataset to explore from the sidebar.")

datasets = os.listdir("datasets")
selected_dataset = st.selectbox("Select a dataset", datasets)


if selected_dataset == "gdp_per_capita.csv":
    st.write(
        """
        ## GDP per Capita

        Source: [World Bank](https://data.worldbank.org/indicator/NY.GDP.PCAP.CD)
        
        This dataset contains the GDP per capita of USA from 1960-2023."""
    )

    df = pd.read_csv(f"datasets/{selected_dataset}")
    st.write(df.head())
    st.line_chart(df, x="Year", y="GDP_per_capita")

if selected_dataset == "eco_fuel_consumption.csv":
    st.write(
        """
        ## 
        This synthetic dataset contains the EcoFuel consumption of a county's population."""
    )

    df = pd.read_csv(f"datasets/{selected_dataset}")
    st.write(df.head())
    st.line_chart(df, x="population", y="ecofuel_consumption")

if selected_dataset == "reactions.csv":
    st.write(
        """
        ## 
        This synthetic dataset contains the Reactions when a concentration is added to an experiment"""
    )

    df = pd.read_csv(f"datasets/{selected_dataset}")
    st.write(df.head())
    st.line_chart(df, x="concentration", y="reaction")

if selected_dataset == "infmort.csv":
    st.write(
        """
        ## Infant mortality and GDP per capita

        Source: [LearnR Walkthroughs](https://github.com/jgscott/learnR/blob/master/README.md#:~:text=Infant%20mortality%20and%20GDP)
        
        This dataset contains the GDP per capita in U.S. dollars from 1960-2023 across 207 countries."""
    )

    df = pd.read_csv(f"datasets/{selected_dataset}").dropna()
    st.write(df.head())
    st.line_chart(df, x="mortality", y="gdp")