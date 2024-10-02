import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from numpy.polynomial.legendre import Legendre
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
    We can see above a few sample of our data, lets take a closer look with some visualizations.
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
st.write("## Regression Plots: Before and After Polynomial Transformation")
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
    
    Coefficient for Concentration: {:.2f}
    
    This means that, on average, reaction increases by every {:.2f} amount of reagent added.
    """.format(model.params["concentration"], model.params["concentration"])
    )

with col2:
    st.write("### Poly-transformed Regression")
    st.write(
        f"R-squared: {r2_score(y, y_pred_poly):.4f}"
    )
    st.write(
        f"Mean Squared Error: {mean_squared_error(y, y_pred_poly):.4f}"
    )

    coefficients = model_poly.coef_
    powers = poly.powers_

    st.markdown(
        r"""
    #### Interpretation:
    In the poly-transformed model, the coefficient are a little harder to interprate but we can try.
    
    $\beta_0={:.4f}$<br>
    $\beta_1={:.4f}$<br>
    $\beta_2={:.4f}$<br>
    $\beta_3={:.4f}$<br>
    $\beta_4={:.4f}$<br>
    
    However this is hard to interperate because in our model we have..

    $X_i^1$<br>
    $X_i^2$<br>
    $X_i^3$<br>
    $X_i^4$<br>
    
    In our training data.
    """.format(coefficients[0],coefficients[1],coefficients[2],coefficients[3],coefficients[4]),
    unsafe_allow_html=True
    )

st.markdown(
    """
    ### Why poly-transform is better:
The 4th-degree polynomial transformation often provides a better fit for complex, non-linear relationships in data because:

1. It captures intricate patterns in the data that a simple linear model cannot, accommodating curves and multiple inflection points.
2. It increases the model's flexibility, allowing it to adapt to upward and downward trends within the dataset.
3. It can significantly improve model accuracy when the relationship between the variables is non-linear.

As seen from the R-squared values, the polynomial-transformed model better explains the variance in the data, indicating an improved fit over standard linear models. However, care should be taken to avoid overfitting, especially with higher-degree polynomials.

"""
)


st.write("---")

st.write("## Model Diagnostics")

st.write("### Residual Plots")
col1, col2 = st.columns(2)

with col1:
    st.pyplot(
        plot_residuals(data["concentration"], y - y_pred,
                       "Residuals: Standard Model", "Concentration")
    )

with col2:
    st.pyplot(
        plot_residuals(
            data["concentration"], y - y_pred_poly, "Residuals: Poly-transformed Model", "Concentration"
        )
    )

st.markdown(
    """
    The residual plots help us assess the homoscedasticity assumption. Ideally, we want to see a random scatter of points 
    with consistent spread. The poly-transformed model often shows improvement in this regard.
    """
)

st.write("### Q-Q Plots")
col1, col2 = st.columns(2)

with col1:
    st.pyplot(plot_qq(y - y_pred, "Q-Q Plot: Standard Model"))

with col2:
    st.pyplot(
        plot_qq(y - y_pred_poly,
                "Q-Q Plot: Poly-transformed Model")
    )

st.markdown(
    """
    While both Q-Q plots appear reasonably aligned with the theoretical quantiles, suggesting that the residuals are approximately normally distributed for both models, it is crucial to be cautious about relying solely on Q-Q plots for model evaluation.

Q-Q plots provide insight into the distribution of residuals but do not capture all aspects of model performance, such as:

1. Heteroscedasticity: Changes in the variance of residuals across the range of predictors.
2. Model Fit: How well the model captures the underlying relationship between the variables.
3. Outliers or Influential Points: These may not be fully apparent in the Q-Q plot.
    """
)

st.markdown(
    """
    ## Conclusion

In this analysis, we've seen how polynomial transformation can improve the fit of a regression model:

* The polynomial model captures the non-linear relationship between the variables more effectively than the standard linear model.
* The R-squared value improved after applying the polynomial transformation, indicating a better overall fit to the data.
* The Q-Q plot for the polynomial-transformed model shows residuals that are more aligned with the theoretical quantiles, suggesting that the assumptions of normality are reasonably met.

Polynomial transformation is often useful when dealing with data that exhibits curvature or complex patterns. It allows the model to adapt to non-linear trends, potentially leading to more accurate predictions and better representation of the underlying data structure. However, care must be taken to select an appropriate degree of the polynomial to avoid overfitting.
"""
)

st.write("## Extra details")
col1, col2 = st.columns(2)

# Step 1: Standardize the input data 'concentration' to [-1, 1]
# Assuming `data` is a DataFrame containing the column 'concentration'
X = data["concentration"].values
X_standardized = 2 * (X - X.min()) / (X.max() - X.min()) - 1  # scale to [-1, 1]

# Step 2: Generate Legendre polynomial features
degree = 4  # Assuming you want to use a 4th-degree Legendre polynomial
legendre_features = [Legendre.basis(d)(X_standardized) for d in range(degree + 1)]

# Stack the features into a design matrix
X_legendre = np.column_stack(legendre_features)

X_L_vif = pd.DataFrame(X_legendre, columns=[f'B_{i}' for i in range(X_legendre.shape[1])])
X_vif = pd.DataFrame(X_poly, columns=[f'P_{i}' for i in range(5)])

# Calculate VIF for each column
vif_L_data = pd.DataFrame()
vif_L_data['Feature'] = X_L_vif.columns
vif_L_data['VIF'] = [variance_inflation_factor(X_L_vif.values, i) for i in range(X_L_vif.shape[1])]

vif_data = pd.DataFrame()
vif_data['Feature'] = X_vif.columns
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

st.markdown(
    """
    It should be noted that there are other downsides to using a naive polynomial regression. Standard polynomial regression terms can be highly correlated and can lead to large coefficients, making the model prone to overfitting and sensitive to noise in the data.

    To check to see if we have colinearity within our data we can preform VIF tests. For the naive approuch our VIF tests are:
    """
    r"""
    $\beta_0={:.4f}$<br>
    $\beta_1={:.4f}$<br>
    $\beta_2={:.4f}$<br>
    $\beta_3={:.4f}$<br>
    $\beta_4={:.4f}$<br>
    """.format(vif_data['VIF'][0],vif_data['VIF'][1],vif_data['VIF'][2],vif_data['VIF'][3],vif_data['VIF'][4])
    +
    """ 
    We should be concered about the colinearity of our columns if the VIF value is above 10 for any predictor. With our values we should be very concerned.
    
    **Solution: Legendre Polynomial**
    The primary reasons to use Legendre polynomials in linear regression are to avoid multicollinearity, enhance numerical stability, and improve model generalization when dealing with non-linear relationships. Standard polynomial regression terms can be highly correlated and can lead to large coefficients, making the model prone to overfitting and sensitive to noise in the data. Legendre polynomials, being orthogonal and bounded, help alleviate these issues.
    """
    +
    r"""
    Legendre Polynomial are defined as 

    $P_n(x) = \frac{1}{2^n n!} \frac{d^n}{dx^n} \left[(x^2 - 1)^n\right]$

    where $P_n(x)$ is the $n$-th Legendre polynomial and x is the variable.
    """
    +
    r"""
    After calulating the new Legendre polynomials we can rerun the VIF test to see if we have improvment in the colinearity of the columns.


    $\beta_0={:.4f}$<br>
    $\beta_1={:.4f}$<br>
    $\beta_2={:.4f}$<br>
    $\beta_3={:.4f}$<br>
    $\beta_4={:.4f}$<br>

    By using Legendre polynomials, you improve the stability and interpretability of your polynomial regression model, particularly when dealing with higher degrees.
    """.format(vif_L_data['VIF'][0],vif_L_data['VIF'][1],vif_L_data['VIF'][2],vif_L_data['VIF'][3],vif_L_data['VIF'][4]),
    unsafe_allow_html=True
)
