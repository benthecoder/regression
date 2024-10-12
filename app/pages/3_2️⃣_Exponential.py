import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats

# Streamlit configuration
st.set_page_config(
    page_title="Exponential Transformation in Regression",
    page_icon="📊",
    layout="wide",
)

# Page Title and Introduction
st.write("# Exponential Transformation")

st.markdown(
    """
    Welcome to this interactive guide on exponential transformation in regression analysis! 

    ## Introduction

    Exponential transformation is a valuable technique in statistics and data science, particularly useful when dealing with:

    1. **Non-linear relationships**: It can linearize exponential growth or decay patterns, making it easier to model these types of relationships.
    2. **Rapidly changing data**: It helps handle variables that change exponentially over time, such as population growth or radioactive decay.
    3. **Modeling multiplicative effects**: It captures multiplicative interactions between variables, converting them to additive forms when needed.
    4. **Stabilizing variance**: By transforming exponential data, you can often reduce heteroscedasticity and stabilize variance in the residuals.

    In this guide, we'll explore how exponential transformation can enhance our regression model using synthetic data. 
    We'll compare a standard linear regression model with an exponentially transformed regression model to demonstrate 
    the advantages and practical applications of this technique.
    """
)


@st.cache_data
def load_data():
    data = pd.read_csv("datasets/eco_fuel_consumption.csv")
    return data

df = load_data()

st.write("## Data Preview")
st.dataframe(df.head())

st.markdown(
    """
    We can see a peek of our data above, but let's take a deeper dive.
    """
)

def plot_regression(x, y, y_pred, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, alpha=0.5)
    ax.plot(x, y_pred, color="red", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def plot_residuals(x, residuals, title, xlabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, residuals, alpha=0.5)
    ax.axhline(y=0, color="r", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Residuals")
    return fig


def plot_qq(residuals, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(title)
    return fig

# Regression Analysis
X = sm.add_constant(df['population'])
y = df['ecofuel_consumption']
model = sm.OLS(y, X).fit()
y_pred = model.predict(X)

df['exp_ecofuel_consumption'] = np.exp(df['ecofuel_consumption'])
exp_y = df['exp_ecofuel_consumption']
exp_model = sm.OLS(exp_y, X).fit()
exp_y_pred = exp_model.predict(X)


st.write("## Regression Plots: Before and After Exponential Transformation")
col1, col2 = st.columns(2)

with col1:
    st.pyplot(
        plot_regression(
            df['population'], 
            y, 
            y_pred, 
            'Standard Linear Regression',
            "Population (in millions)",
            "EcoFuel Consumption (in millions of gallons)"
        )
    )

with col2:
    st.pyplot(
        plot_regression(
            df['population'], 
            exp_y, 
            exp_y_pred, 
            'Exponentially Transformed Regression', 
            "Population (in millions)", 
            "Exp EcoFuel Consumption (in millions of gallons)"
        )
    )

st.write("## Model Performance and Interpretation")
col1, col2 = st.columns(2)

with col1:
    st.write("### Standard Linear Regression")
    st.write(f"R-squared: {r2_score(y, y_pred):.4f}")
    st.write(f"Mean Squared Error: {mean_squared_error(y, y_pred):.4f}")
    st.write(f"Mean Squared Total: {np.var(df['ecofuel_consumption'], ddof=1):.4f}")
    
    st.markdown(
        """
    #### Interpretation:
    In the standard model, the coefficient for Population represents the average increase in millions of gallons of EcoFuel consumption per 1 million people.
    
    Coefficient for Population: {:.4f}
    
    This means that, on average, EcoFuel consumption (in millions of gallons) increases by {:.4f} per 1 million people.
    """.format(model.params["population"], model.params["population"])
    )


with col2:
    st.write("### Exponentially Transformed Regression")
    st.write(f"R-squared: {r2_score(exp_y, exp_y_pred):.4f}")
    st.write(f"Mean Squared Error: {mean_squared_error(exp_y, exp_y_pred):.4f}")
    st.write(f"Mean Squared Total: {np.var(df['exp_ecofuel_consumption'], ddof=1):.4f}")

    coef = exp_model.params["population"]
    percentage_change = (np.exp(coef) - 1) * 100

    st.markdown(
        """
        #### Interpretation:
        In the transformed model, the coefficient for Population represents the approximate percentage change in EcoFuel consumption for each additional 1 million people.
        
        Coefficient for Population: {:.4f}
        
        This means that, on average, for each 1 million increase in population, EcoFuel consumption (in millions of gallons) increases by approximately {} percent.
        """.format(coef, int(percentage_change))
    )

st.markdown(
    """
    ### Why exponential transformation is better:
The exponentially transformed model often provides a better fit for data that follows an exponential growth or decay pattern because:

1. It captures rapid changes in the data more effectively than a linear model, making it suitable for modeling growth or decay over time.
2. It transforms multiplicative relationships into additive ones, simplifying the interpretation of interaction effects.
3. It can address issues of heteroscedasticity by stabilizing the variance in the residuals.

As shown by the metrics, the exponentially transformed model explains a larger proportion of the variance in the data (R-squared = 0.9691) compared to the standard linear regression (R-squared = 0.9050), indicating a significantly better fit. However, it is important to consider that the exponentially transformed model had a higher Mean Squared Error (0.4325) compared to the linear model (0.0327), suggesting potential sensitivity to extreme values or outliers in the dataset.

"""
)




st.write("---")

st.write("## Model Diagnostics")

st.write("### Residual Plots")

col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_residuals(df['population'], y - y_pred, "Residuals: Standard Model", "Population"))

with col2:
    st.pyplot(plot_residuals(df['population'], exp_y - exp_y_pred, "Residuals: Exponentially Transformed Model", "Population"))

st.markdown(
    """
    The residual plots help us assess the homoscedasticity assumption. Ideally, we want to see a random scatter of points 
    with consistent spread. The exponentially transformed model definitely shows a more random scatter of points, but homoscedasticity remains an issue. We may want to consider removing outliers.
    """
)


st.write("### Q-Q Plots")
col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_qq(y - y_pred, "Q-Q Plot: Standard Model"))

with col2:
    st.pyplot(plot_qq(exp_y - exp_y_pred, "Q-Q Plot: Exponentially Transformed Model"))

st.markdown(
    """
    Q-Q plots help us assess the normality of residuals. Points closer to the diagonal line indicate a better fit to 
    the normal distribution. The exponentially transformed model often shows improvement in normality of residuals with the exception of the points at the edges of our x range.
    """
)

st.markdown(
    """
    ## Conclusion

    In this analysis, we've seen how exponential transformation can improve the fit of a regression model:

    1. The exponentially transformed model captured the rapid changes in the data more effectively, particularly in cases of exponential growth patterns.
    2. The R-squared value improved significantly from 0.9050 (standard linear regression) to 0.9691 (exponentially transformed model), indicating a better overall fit to the data.
    3. The Q-Q plot for the exponentially transformed model showed residuals that were closer to the diagonal line, suggesting a better approximation to a normal distribution.

    Exponential transformation is often useful when dealing with data that changes exponentially over time, such as population growth or decay of a resource. It helps stabilize the variance and capture the multiplicative effects that are not easily represented by a linear model. However, it is crucial to be cautious of increased sensitivity to outliers, as indicated by the higher Mean Squared Error in the exponentially transformed model (0.4325) compared to the linear model (0.0327). Further refinement of the model or handling of outliers might be needed to optimize performance.
    """
)

st.write("## Appendix: Detailed Model Summaries")
col1, col2 = st.columns(2)

with col1:
    st.write("### Standard Model Summary")
    st.text(model.summary().as_text())

with col2:
    st.write("### Exponentially Transformed Model Summary")
    st.text(exp_model.summary().as_text())

st.markdown(
    """
    These summaries provide detailed statistical information about both models, including coefficient estimates, 
    standard errors, t-statistics, and p-values.
    """
)
