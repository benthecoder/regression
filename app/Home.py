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

    Click on the Explore page to see the effects of different transformations on different datasets.
    
    Click on the sidebar to select the transformation you want to apply!


    ### Team members:
    - [Adrian Lam Lorn Hin](https://www.linkedin.com/in/adrianlhlam/)
    - [Benedict Neo](https://www.linkedin.com/in/benedictneo/)
    - [Nicholas Barsi-Rhyne](https://www.linkedin.com/in/nicholas-barsi-rhyne-64a006204/)
    - [Rebekah Zhou](https://www.linkedin.com/in/rebekahzhou/)
"""
)
