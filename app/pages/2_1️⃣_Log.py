import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats

st.set_page_config(
    page_title="Log Transformation in Regression",
    page_icon="📊",
    layout="wide",
)

st.write("# Log Transformation")

st.markdown(
    """
    Welcome to this interactive guide on log transformation in regression analysis! 
    
    ## Introduction
    
    Log transformation is a powerful technique in statistics and data science, particularly useful when dealing with:
    
    1. **Skewed data**: It can help normalize right-skewed distributions.
    2. **Multiplicative relationships**: It converts multiplicative relationships to additive ones.
    3. **Heteroscedasticity**: It can stabilize variance in the residuals.
    4. **Percentage changes**: It allows us to interpret coefficients as percentage changes.

    In this guide, we'll explore how log transformation can improve our regression model using GDP per capita data. 
    We'll compare a standard linear regression with a log-transformed model to demonstrate the benefits and 
    interpretations of this technique.
    """
)


# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv("datasets/gdp_per_capita.csv")
    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
    data["GDP_per_capita"] = pd.to_numeric(data["GDP_per_capita"], errors="coerce")
    return data.dropna()


data = load_data()

st.write("## Data Preview")
st.dataframe(data.head())

st.markdown(
    """
    Above, we can see the first few rows of our GDP per capita dataset. Let's start by visualizing this data 
    to understand why log transformation might be beneficial.
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
X = sm.add_constant(data["Year"])
y = data["GDP_per_capita"]
model = sm.OLS(y, X).fit()
y_pred = model.predict(X)

data["log_GDP_per_capita"] = np.log(data["GDP_per_capita"])
model_log = sm.OLS(data["log_GDP_per_capita"], X).fit()
y_pred_log = model_log.predict(X)

# Plot before and after transformation side by side
st.write("## Regression Plots: Before and After Log Transformation")
col1, col2 = st.columns(2)

with col1:
    st.pyplot(
        plot_regression(
            data["Year"],
            y,
            y_pred,
            "Standard Linear Regression",
            "Year",
            "GDP per capita",
        )
    )

with col2:
    st.pyplot(
        plot_regression(
            data["Year"],
            data["log_GDP_per_capita"],
            y_pred_log,
            "Log-transformed Regression",
            "Year",
            "log(GDP per capita)",
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
    In the standard model, the coefficient for Year represents the average increase in GDP per capita for each year.
    
    Coefficient for Year: {:.2f}
    
    This means that, on average, GDP per capita increases by ${:.2f} each year.
    """.format(model.params["Year"], model.params["Year"])
    )

with col2:
    st.write("### Log-transformed Regression")
    st.write(
        f"R-squared: {r2_score(data['log_GDP_per_capita'], model_log.predict(X)):.4f}"
    )
    st.write(
        f"Mean Squared Error: {mean_squared_error(data['log_GDP_per_capita'], model_log.predict(X)):.4f}"
    )

    year_coef = model_log.params["Year"]
    percentage_change = (np.exp(year_coef) - 1) * 100

    st.markdown(
        """
    #### Interpretation:
    In the log-transformed model, the coefficient for Year represents the average percentage increase in GDP per capita for each year.
    
    Coefficient for Year: {:.4f}
    
    This means that, on average, GDP per capita increases by {:.4f}% each year.
    """.format(year_coef, percentage_change)
    )

st.write("---")

st.write("## Model Diagnostics")

st.write("### Residual Plots")
col1, col2 = st.columns(2)

with col1:
    st.pyplot(
        plot_residuals(data["Year"], y - y_pred, "Residuals: Standard Model", "Year")
    )

with col2:
    # Calculate residuals in the original space
    log_residuals = y - np.exp(y_pred_log)
    st.pyplot(
        plot_residuals(
            data["Year"],
            log_residuals,
            "Residuals: Log-transformed Model",
            "Year",
        )
    )

st.markdown(
    """
    Residual plots help assess homoscedasticity (constant variance of residuals):

    - **Standard Model**: Clear pattern with larger residuals in recent years, indicating heteroscedasticity.

    - **Log-transformed Model**: More consistent spread, but some patterns remain in the later years.

    The log-transformed model improves homoscedasticity, but doesn't fully capture all underlying data structures.
    """
)

st.write("### Q-Q Plots")
col1, col2 = st.columns(2)

with col1:
    st.pyplot(plot_qq(y - y_pred, "Q-Q Plot: Standard Model"))

with col2:
    st.pyplot(
        plot_qq(np.log(y) - model_log.predict(X), "Q-Q Plot: Log-transformed Model")
    )

st.markdown(
    """
    Q-Q plots help assess the normality of residuals. Points closer to the diagonal line indicate better normality.

    - **Standard Model**: Significant deviations from the diagonal, especially at the tails. Residuals are not normally distributed.

    - **Log-transformed Model**: Closer to the diagonal, especially in the middle. Some deviations remain at the tails.

    The log-transformed model improves residual normality, but some patterns persist, suggesting potential extreme values or unexplained trends in the data.
    """
)


st.markdown(
    """
    ## Conclusion

    Our analysis demonstrates how log transformation can significantly improve linear regression models for GDP per capita data:

    1. **Improved homoscedasticity in residuals**: The log-transformed model shows a more consistent spread of residuals across the years, addressing the heteroscedasticity observed in the standard model.
    
    2. **Better normality of residuals**: The Q-Q plot for the log-transformed model adheres more closely to the diagonal line, indicating a distribution closer to normal, especially in the middle range.
    
    3. **Higher R-squared value**: The log-transformed model explains a larger proportion of the variance in the data, increasing from {:.4f} to {:.4f}.

    Key benefits of log transformation for economic data:
    1. **Reveals exponential growth patterns**: By linearizing exponential trends, log transformation helps uncover and model the compound growth often present in economic data.
    
    2. **Allows interpretation of coefficients as percentage changes**: In our log-transformed model, we can interpret that GDP per capita grows by approximately {:.2f}% per year, providing a more intuitive understanding of economic growth.
    
    3. **Helps stabilize variance in residuals**: This addresses the issue of heteroscedasticity, which violates a key assumption of OLS regression.
    
    4. **Improves adherence to linear regression assumptions**: By addressing non-linearity, non-normality, and heteroscedasticity, log transformation helps meet the assumptions required for valid inference in linear regression.

    Limitations and areas for further investigation:
    - Some patterns in residuals and Q-Q plot deviations remain, particularly in the tails, suggesting that log transformation alone may not capture all complexities in the data.
    - MSE values are not directly comparable between models due to the change in scale. Alternative metrics like MAPE (Mean Absolute Percentage Error) could be considered for model comparison.
    - The simple time series model may not account for other factors influencing GDP per capita, such as economic cycles, technological advancements, or global events.

    While log transformation significantly improves the model, it's not a perfect solution. Further analysis could include:
    1. Exploring additional variables that might explain GDP per capita growth.
    2. Investigating non-linear trends that persist after log transformation.
    3. Considering more advanced time series techniques, such as ARIMA models or polynomial regression.
    4. Analyzing residuals for autocorrelation, which is common in time series data.
    5. Examining the impact of influential observations or outliers on the model.

    In conclusion, this analysis demonstrates the power of log transformation in improving linear regression models for economic data. It highlights the importance of carefully considering data characteristics and model assumptions in statistical analysis. While the log-transformed model provides valuable insights into GDP per capita growth, it also reveals the complexity of economic data and the need for ongoing refinement in modeling approaches.
    """.format(
        r2_score(y, y_pred),
        r2_score(data["log_GDP_per_capita"], model_log.predict(X)),
        percentage_change,
    )
)


st.write("## Appendix: Detailed Model Summaries")
col1, col2 = st.columns(2)

with col1:
    st.write("### Standard Model Summary")
    st.text(model.summary().as_text())

with col2:
    st.write("### Log-transformed Model Summary")
    st.text(model_log.summary().as_text())

st.markdown(
    """
    These summaries provide detailed statistical information about both models, including coefficient estimates, 
    standard errors, t-statistics, and p-values.
    """
)
