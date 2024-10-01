import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy import stats

st.set_page_config(
    page_title="Polynomial Transformation in Regression",
    page_icon="📊",
    layout="wide",
)

st.write("# Polynomia Transformation")

st.markdown(
    """
    
    ## Introduction
    
    Polynomial transformation is a versatile technique in statistics and data science, particularly useful when dealing with:

    1. **Non-linear relationships**: It enables the modeling of relationships that are not linear, capturing complex patterns.
    2. **Flexibility in fitting data**: It allows for better fitting of data that exhibit curved or more intricate trends.
    3. **Interactions among predictors**: Higher-degree terms can represent interactions between predictors and the response variable.
    4. **Improved model accuracy**: By adding polynomial terms, you can potentially improve the predictive power of your model.

    In this guide, we'll explore how polynomial transformation can enhance our regression model using a hypothetical dose-response dataset. We'll compare a standard linear regression model with a higher-degree polynomial regression model to demonstrate the advantages of capturing non-linear trends and making better predictions.
    """
)


# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv("datasets/reactions.csv")
    data["concentration"] = pd.to_numeric(data["concentration"], errors="coerce")
    data["reaction"] = pd.to_numeric(
        data["reaction"], errors="coerce")
    return data.dropna()


data = load_data()

st.write("## Data Preview")
st.dataframe(data.head())

st.markdown(
    """
    We cam see above a few sample of our data, lets take a closer look with some visualizations.
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
X = sm.add_constant(data["concentration"])
y = data["reaction"]
model = sm.OLS(y, X).fit()
y_pred = model.predict(X)

model_poly = LinearRegression()
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(data["concentration"].values.reshape(-1, 1))
model_poly.fit(X_poly, y)
X_range = np.linspace(0, 5, 100).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_pred_poly = model_poly.predict(X_poly)


# Plot before and after transformation side by side
st.write("## Regression Plots: Before and After Log Transformation")
col1, col2 = st.columns(2)

with col1:
    st.pyplot(
        plot_regression(
            data["concentration"],
            y,
            y_pred,
            "Standard Linear Regression",
            "Concentraction",
            "Reaction",
        )
    )

with col2:
    st.pyplot(
        plot_regression(
            data["concentration"],
            y,
            y_pred_poly,
            "Poly-transformed Regression",
            "Concentraction",
            "Reaction",
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
        f"R-squared: {r2_score(data['log_GDP_per_capita'],model_poly.predict(X)):.4f}"
    )
    st.write(
        f"Mean Squared Error: {mean_squared_error(data['log_GDP_per_capita'], model_poly.predict(X)):.4f}"
    )

    year_coef = model_poly.params["Year"]
    percentage_change = (np.exp(year_coef) - 1) * 100

    st.markdown(
        """
    #### Interpretation:
    In the log-transformed model, the coefficient for Year represents the average percentage increase in GDP per capita for each year.
    
    Coefficient for Year: {:.4f}
    
    This means that, on average, GDP per capita increases by {:.2f}% each year.
    """.format(year_coef, percentage_change)
    )

st.markdown(
    """
    ### Why log is better:
The log-transformed model often provides a better fit for economic data like GDP per capita because:

1. It captures the exponential growth pattern typically seen in economic indicators.
2. It allows for interpretation in terms of percentage changes, which is more intuitive for economic analysis.
3. It can help address issues of heteroscedasticity (uneven variance) in the residuals.

As we can see from the R-squared values, the log-transformed model explains a larger proportion of the variance in the data, indicating a better fit.

"""
)


st.write("---")

st.write("## Model Diagnostics")

st.write("### Residual Plots")
col1, col2 = st.columns(2)

with col1:
    st.pyplot(
        plot_residuals(data["Year"], y - y_pred,
                       "Residuals: Standard Model", "Year")
    )

with col2:
    st.pyplot(
        plot_residuals(
            data["Year"], y - y_pred_log, "Residuals: Log-transformed Model", "Year"
        )
    )

st.markdown(
    """
    The residual plots help us assess the homoscedasticity assumption. Ideally, we want to see a random scatter of points 
    with consistent spread. The log-transformed model often shows improvement in this regard.
    """
)

st.write("### Q-Q Plots")
col1, col2 = st.columns(2)

with col1:
    st.pyplot(plot_qq(y - y_pred, "Q-Q Plot: Standard Model"))

with col2:
    st.pyplot(
        plot_qq(np.log(y) - model_poly.predict(X),
                "Q-Q Plot: Log-transformed Model")
    )

st.markdown(
    """
    Q-Q plots help us assess the normality of residuals. Points closer to the diagonal line indicate a better fit to 
    the normal distribution. The log-transformed model often shows improvement in normality of residuals.
    """
)

st.markdown(
    """
    ## Conclusion

    In this analysis, we've seen how log transformation can improve the fit of a linear regression model:

    1. The residuals in the log-transformed model show less pattern and are more evenly distributed.
    2. The R-squared value improved after log transformation, indicating a better fit.
    3. The Q-Q plot for the log-transformed model shows points closer to the diagonal line,
       suggesting that the residuals are closer to a normal distribution.

    Log transformation is often useful when dealing with economic data like GDP, as it can help
    address issues of heteroscedasticity and non-linearity in the relationship between variables.
"""
)

st.write("## Appendix: Detailed Model Summaries")
col1, col2 = st.columns(2)

with col1:
    st.write("### Standard Model Summary")
    st.text(model.summary().as_text())

with col2:
    st.write("### Log-transformed Model Summary")
    st.text(model_poly.summary().as_text())

st.markdown(
    """
    These summaries provide detailed statistical information about both models, including coefficient estimates, 
    standard errors, t-statistics, and p-values.
    """
)
