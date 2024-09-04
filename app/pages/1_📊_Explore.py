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


if selected_dataset == "evpopulation.csv":
    st.write(
        """
        ## Electric Vehicle Population

        Source:  [Full Electric Vehicle Dataset 2024](https://www.kaggle.com/datasets/sahirmaharajj/electric-vehicle-population)
        
        This dataset contains the population of different countries over the years."""
    )

# fill for rest of datasets


df = pd.read_csv(f"datasets/{selected_dataset}")
st.write(df.head())
