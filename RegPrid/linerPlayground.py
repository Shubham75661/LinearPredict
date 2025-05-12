import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# App title
st.title("📈 LinearPredict: Simple & Multi Linear Regression Tool")

file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

# Load the data
if file is not None:
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.subheader("📊 Data Preview")
    st.write(df.head())

    # Feature selection
    st.subheader("🔧 Feature Selection")
    dependent_feature = st.selectbox("Select Dependent (Target) Feature", df.columns)
    independent_features = st.multiselect("Select Independent (Input) Features", [col for col in df.columns if col != dependent_feature])

    if len(independent_features) > 0:
        X = df[independent_features]
        y = df[dependent_feature]

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Predictions and R^2
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        st.subheader("📈 Model Evaluation")
        st.write(f"R² Score: **{r2:.4f}**")

        # Scatter plot
        st.subheader("🧮 Prediction Plot")
        fig, ax = plt.subplots()
        ax.scatter(y, y_pred, color='blue', alpha=0.6)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

        # Take custom input from user
        st.subheader("🔍 Make a Prediction")
        input_data = []
        for feature in independent_features:
            value = st.number_input(f"Enter value for {feature}", format="%.4f")
            input_data.append(value)

        if st.button("Predict"):
            input_array = np.array(input_data).reshape(1, -1)
            prediction = model.predict(input_array)
            st.success(f"📌 Predicted {dependent_feature}: **{prediction[0]:.4f}**")
