import os
import pandas as pd
import numpy as np
import streamlit as st
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Explore", page_icon="📊", layout="wide")

st.write("# Explore your data!")

st.write("Choose a dataset to explore from the sidebar.")

datasets = os.listdir("datasets")
selected_dataset = st.selectbox("Select a dataset", datasets)

def preprocess_data(df, x_col, y_col):
    # Remove rows with inf or nan values
    df = df[(~df[x_col].isin([np.inf, -np.inf])) & (~df[y_col].isin([np.inf, -np.inf]))]
    df = df.dropna(subset=[x_col, y_col])
    
    # Ensure positive values for log transformation
    if (df[y_col] <= 0).any() or (df[x_col] <= 0).any():
        df[y_col] = df[y_col] - df[y_col].min() + 1
        df[x_col] = df[x_col] - df[x_col].min() + 1
    
    return df

def apply_transformations(df, x_col, y_col):
    X = df[x_col].values.reshape(-1, 1)
    y = df[y_col].values
    
    # LOG transformation
    y_log = np.log(y)
    model_log = sm.OLS(y_log, sm.add_constant(X)).fit()
    y_pred_log = model_log.predict(sm.add_constant(X))
    
    # EXPONENTIAL transformation
    y_exp = np.exp(y)
    model_exp = sm.OLS(y_exp, sm.add_constant(X)).fit()
    y_pred_exp = model_exp.predict(sm.add_constant(X))
    
    # POLYNOMIAL transformation
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    model_poly = LinearRegression().fit(X_poly, y)
    y_pred_poly = model_poly.predict(X_poly)
    
    # POWER LAW transformation
    X_power = np.log(X)
    y_power = np.log(y)
    model_power = sm.OLS(y_power, sm.add_constant(X_power)).fit()
    y_pred_power = np.exp(model_power.predict(sm.add_constant(X_power)))
    
    return y_log, y_pred_log, y_exp, y_pred_exp, y_pred_poly, X_power, y_power, y_pred_power

def plot_transformations(df, x_col, y_col):
    df = preprocess_data(df, x_col, y_col)
    
    try:
        y_log, y_pred_log, y_exp, y_pred_exp, y_pred_poly, X_power, y_power, y_pred_power = apply_transformations(df, x_col, y_col)
        
        fig = make_subplots(rows=2, cols=2, subplot_titles=("LOG", "EXPONENTIAL", "POLYNOMIAL", "POWER LAW"))
        
        # LOG
        fig.add_trace(go.Scatter(x=df[x_col], y=y_log, mode='markers', name='Data'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df[x_col], y=y_pred_log, mode='lines', name='LOG Fit'), row=1, col=1)
        fig.update_xaxes(title_text=f"{x_col}", row=1, col=1)
        fig.update_yaxes(title_text=f"log({y_col})", row=1, col=1)
        
        # EXPONENTIAL
        fig.add_trace(go.Scatter(x=df[x_col], y=y_exp, mode='markers', name='Data'), row=1, col=2)
        fig.add_trace(go.Scatter(x=df[x_col], y=y_pred_exp, mode='lines', name='EXPONENTIAL Fit'), row=1, col=2)
        fig.update_xaxes(title_text=x_col, row=1, col=2)
        fig.update_yaxes(title_text=f"exp({y_col})", row=1, col=2)
        
        # POLYNOMIAL
        fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='markers', name='Data'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df[x_col], y=y_pred_poly, mode='lines', name='POLYNOMIAL Fit'), row=2, col=1)
        fig.update_xaxes(title_text=x_col, row=2, col=1)
        fig.update_yaxes(title_text=y_col, row=2, col=1)
        
        # POWER LAW
        fig.add_trace(go.Scatter(x=np.log(df[x_col]), y=np.log(df[y_col]), mode='markers', name='Data'), row=2, col=2)
        fig.add_trace(go.Scatter(x=np.log(df[x_col]), y=np.log(y_pred_power), mode='lines', name='POWER LAW Fit'), row=2, col=2)
        fig.update_xaxes(title_text=f"log({x_col})", row=2, col=2)
        fig.update_yaxes(title_text=f"log({y_col})", row=2, col=2)
        
        fig.update_layout(height=800, width=1000, title_text=f"Regression Transformations for {selected_dataset}")
        
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred while processing the data: {str(e)}")
        st.write("Please check your data for any inconsistencies or try another dataset.")

if selected_dataset:
    df = pd.read_csv(f"datasets/{selected_dataset}")
    st.write(df.head())
    
    if selected_dataset == "gdp_per_capita.csv":
        st.write(
            """
            ## GDP per Capita
            Source: [World Bank](https://data.worldbank.org/indicator/NY.GDP.PCAP.CD)
            This dataset contains the GDP per capita of USA from 1960-2023."""
        )
        plot_transformations(df, "Year", "GDP_per_capita")
        
    elif selected_dataset == "eco_fuel_consumption.csv":
        st.write(
            """
            ## EcoFuel Consumption
            This synthetic dataset contains the EcoFuel consumption of a county's population."""
        )
        plot_transformations(df, "population", "ecofuel_consumption")
        
    elif selected_dataset == "reactions.csv":
        st.write(
            """
            ## Reactions
            This synthetic dataset contains the Reactions when a concentration is added to an experiment."""
        )
        plot_transformations(df, "concentration", "reaction")
        
    elif selected_dataset == "infmort.csv":
        st.write(
            """
            ## Infant mortality and GDP per capita
            Source: [LearnR Walkthroughs](https://github.com/jgscott/learnR/blob/master/README.md#:~:text=Infant%20mortality%20and%20GDP)
            This dataset contains the GDP per capita in U.S. dollars from 1960-2023 across 207 countries."""
        )
        plot_transformations(df, "mortality", "gdp")