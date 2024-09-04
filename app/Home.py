import streamlit as st

st.set_page_config(
    page_title="Regression Transformations",
    page_icon="📈",
)

st.write("# Transform your data!")

st.image("assets/transformations.jpg")

st.markdown(
    """

    Our app shows the effects of different transformations on your data.

    We have implemented the following transformations:
    - Logarithmic
    - Polynomial
    - Power Law
    - Exponential

    Click on the sidebar to select the transformation you want to apply!


    ### Team members:
    - Benedict Neo
    - Adrian Lam Lorn Hin
    - Nicholas Barsi-Rhyne  
    - Rebekah Zhou  
"""
)
