import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats

st.set_page_config(
    page_title="Power Law Transformation in Regression",
    page_icon="📊",
    layout="wide",
)

st.write("# Power Law Transformation")

st.markdown(
    """
    Welcome to this interactive guide on power law transformation in regression analysis! 
    
    ## Introduction
    
    Power law transformation is a powerful technique in statistics and data science, particularly useful when dealing with:
    
    1. **Non-linear relationships**: It can help linearize power law relationships.
    2. **Scale-invariant phenomena**: It's useful for data that follows power law distributions.
    3. **Heteroscedasticity**: It can stabilize variance in the residuals.
    4. **Wide range of values**: It can compress the range of variables spanning several orders of magnitude.

    In this guide, we'll explore how power law transformation can improve our regression model using simulated data. 
    We'll compare a standard linear regression with a log-log transformed model to demonstrate the benefits and 
    interpretations of this technique.
    """
)

# Generate simulated data
@st.cache_data
def load_data():
    data = pd.read_csv("datasets/infmort.csv")
    data["mortality"] = pd.to_numeric(data["mortality"], errors="coerce")
    data["gdp"] = pd.to_numeric(data["gdp"], errors="coerce")
    return data.dropna()


data = load_data()

st.write("## Data Preview")
st.dataframe(data.head())

st.markdown(
    """
    Above, we can see the first few rows of our simulated dataset. Let's start by visualizing this data 
    to understand why power law transformation might be beneficial.
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
X = sm.add_constant(data["mortality"])
y = data["gdp"]
model = sm.OLS(y, X).fit()
y_pred = model.predict(X)

data["log_mortality"] = np.log(data["mortality"])
data["log_gdp"] = np.log(data["gdp"])
X_log = sm.add_constant(data["log_mortality"])
model_log = sm.OLS(data["log_gdp"], X_log).fit()
y_pred_log = np.exp(model_log.predict(X_log))

# Plot before and after transformation side by side
st.write("## Regression Plots: Before and After Power Law Transformation")
col1, col2 = st.columns(2)

with col1:
    st.pyplot(
        plot_regression(
            data["mortality"],
            y,
            y_pred,
            "Standard Linear Regression",
            "Infant Mortality",
            "GDP per Capita",
        )
    )

with col2:
    st.pyplot(
        plot_regression(
            data["log_mortality"],
            data["log_gdp"],
            model_log.predict(X_log),
            "Power Law Transformed Regression",
            "Log Infant Mortality",
            "Log GDP per Capita",
        )
    )

st.write("## Model Performance and Interpretation")
col1, col2 = st.columns(2)

with col1:
    st.write("### Standard Linear Regression")
    st.write(f"R-squared: {r2_score(y, y_pred):.4f}")
    st.write(f"Mean Squared Error: {mean_squared_error(y, y_pred):.4f}")

    st.markdown(
        """
    #### Interpretation:
    In the standard model, the coefficient for x represents the average increase in y for each unit increase in x.
    
    Coefficient for x: {:.2f}
    
    This means that, on average, y increases by {:.2f} for each unit increase in x.
    """.format(model.params["mortality"], model.params["mortality"])
    )

with col2:
    st.write("### Power Law Transformed Regression")
    st.write(f"R-squared: {r2_score(data['log_gdp'], model_log.predict(X_log)):.4f}")
    st.write(f"Mean Squared Error: {mean_squared_error(data['log_gdp'], model_log.predict(X_log)):.4f}")

    x_coef = model_log.params["log_mortality"]

    st.markdown(
        """
    #### Interpretation:
    In the power law transformed model, the coefficient for log(x) represents the power in the relationship y = ax^b.
    
    Coefficient for log(x): {:.4f}
    
    This means that y is proportional to x^{:.4f}. In other words, when x increases by 1%, y increases by approximately {:.2f}%.
    """.format(x_coef, x_coef, x_coef)
    )

st.markdown(
    """
    ### Why power law transformation is better:
The power law transformed model often provides a better fit for data with non-linear, power law relationships because:

1. It captures the non-linear growth pattern typically seen in power law relationships.
2. It allows for interpretation in terms of percentage changes and elasticity.
3. It can help address issues of heteroscedasticity (uneven variance) in the residuals.

As we can see from the R-squared values, the power law transformed model explains a larger proportion of the variance in the data, indicating a better fit.

"""
)

st.write("---")

st.write("## Model Diagnostics")

st.write("### Residual Plots")
col1, col2 = st.columns(2)

with col1:
    st.pyplot(
        plot_residuals(data["mortality"], y - y_pred, "Residuals: Standard Model", "mortality")
    )

with col2:
    st.pyplot(
        plot_residuals(
            data["mortality"], y - y_pred_log, "Residuals: Power Law Transformed Model", "mortality"
        )
    )

st.markdown(
    """
    The residual plots help us assess the homoscedasticity assumption. Ideally, we want to see a random scatter of points 
    with consistent spread. The power law transformed model often shows improvement in this regard.
    """
)

st.write("### Q-Q Plots")
col1, col2 = st.columns(2)

with col1:
    st.pyplot(plot_qq(y - y_pred, "Q-Q Plot: Standard Model"))

with col2:
    st.pyplot(
        plot_qq(data["log_gdp"] - model_log.predict(X_log), "Q-Q Plot: Power Law Transformed Model")
    )

st.markdown(
    """
    Q-Q plots help us assess the normality of residuals. Points closer to the diagonal line indicate a better fit to 
    the normal distribution. The power law transformed model often shows improvement in normality of residuals.
    """
)

st.markdown(
    """
    ## Conclusion

    In this analysis, we've seen how power law transformation can improve the fit of a linear regression model:

    1. The residuals in the power law transformed model show less pattern and are more evenly distributed.
    2. The R-squared value improved after power law transformation, indicating a better fit.
    3. The Q-Q plot for the power law transformed model shows points closer to the diagonal line,
       suggesting that the residuals are closer to a normal distribution.

    Power law transformation is often useful when dealing with data that follows power law relationships,
    as it can help address issues of non-linearity and heteroscedasticity in the relationship between variables.
"""
)

st.write("## Appendix: Detailed Model Summaries")
col1, col2 = st.columns(2)

with col1:
    st.write("### Standard Model Summary")
    st.text(model.summary().as_text())

with col2:
    st.write("### Power Law Transformed Model Summary")
    st.text(model_log.summary().as_text())

st.markdown(
    """
    These summaries provide detailed statistical information about both models, including coefficient estimates, 
    standard errors, t-statistics, and p-values.
    """
)