import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration
st.set_page_config(
    page_title="Movie Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load models and data
@st.cache_data
def load_data():
    data_path = "../Interim_Data/Final_Cleaned_Data.pkl"
    inference_results_path = "../Predictions/inference_results.csv"
    model_path = "../Model_outputs/champion_model.pkl"

    # Load the data and model
    data = pd.read_pickle(data_path)
    inference_results = pd.read_csv(inference_results_path)
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    # Feature Engineering: Add profit margin
    if "profit_margin" not in data.columns:
        data["profit_margin"] = (data["revenue"] - data["budget"]) / data["revenue"]
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=["profit_margin"], inplace=True)

    return data, inference_results, model


data, inference_results, model = load_data()

# Sidebar: Navigation
st.sidebar.title("Navigation")
pages = ["Overview", "EDA", "Model Prediction", "Inference Results"]
page_selection = st.sidebar.radio("Go to", pages)

# Helper function for Seaborn visualizations
def seaborn_plot(func):
    fig, ax = plt.subplots(figsize=(12, 6))
    func(ax)
    st.pyplot(fig)

# Helper function to clean numeric columns
def clean_numeric_column(data, column_name):
    data[column_name] = pd.to_numeric(data[column_name], errors="coerce")
    return data.dropna(subset=[column_name])

# Page 1: Overview
if page_selection == "Overview":
    st.title("Movie Analysis Dashboard")
    st.markdown("""
    ## Overview
    This dashboard provides insights into movie datasets, predictions of revenue based on input features, and performance evaluations of the predictive model. Key functionalities include:
    
    - **Exploratory Data Analysis (EDA):** Uncover patterns and trends in movie data.
    - **Model Prediction:** Predict movie revenue using a trained regression model.
    - **Inference Results:** Evaluate model performance and visualize predictions versus actual values.
    
    ### Data Sources
    - **Cleaned Dataset:** Movies with revenue, budget, genre, and ratings data.
    - **Machine Learning Model:** Trained on the cleaned data for revenue prediction.
    """)

# Page 2: EDA
elif page_selection == "EDA":
    st.title("Exploratory Data Analysis")

    # Visualization 1: Revenue vs Budget
    st.subheader("Revenue vs Budget")
    def plot_revenue_vs_budget(ax):
        sns.scatterplot(data=data, x="budget", y="revenue", ax=ax, alpha=0.6)
        ax.set_title("Revenue vs Budget")
        ax.set_xlabel("Budget")
        ax.set_ylabel("Revenue")
    seaborn_plot(plot_revenue_vs_budget)

    # Visualization 2: Distribution of Average Ratings
    st.subheader("Distribution of Average Ratings")
    def plot_avg_rating_dist(ax):
        sns.histplot(data=data, x="avg_rating", bins=20, kde=True, ax=ax, color="skyblue")
        ax.set_title("Distribution of Average Ratings")
        ax.set_xlabel("Average Rating")
        ax.set_ylabel("Frequency")
    seaborn_plot(plot_avg_rating_dist)

    # Visualization 3: Trends in Revenue Over Time
    st.subheader("Trends in Revenue Over Time")
    def plot_revenue_trends(ax):
        data["release_year"] = pd.to_numeric(data["release_year"], errors="coerce")
        data.dropna(subset=["release_year"], inplace=True)
        data["release_year"] = data["release_year"].astype(int)
        yearly_revenue = data.groupby("release_year")["revenue"].sum()
        sns.lineplot(x=yearly_revenue.index, y=yearly_revenue.values, ax=ax, marker="o")
        ax.set_title("Revenue Trends Over Time")
        ax.set_xlabel("Release Year")
        ax.set_ylabel("Revenue")
    seaborn_plot(plot_revenue_trends)

    # Visualization 4: Top Genres by Revenue
    st.subheader("Top Genres by Revenue")
    def plot_top_genres_by_revenue(ax):
        top_genres = (
            data.groupby("genres_x")["revenue"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        sns.barplot(data=top_genres, y="genres_x", x="revenue", ax=ax, palette="Blues_r")
        ax.set_title("Top Genres by Revenue")
        ax.set_xlabel("Average Revenue")
        ax.set_ylabel("Genres")
    seaborn_plot(plot_top_genres_by_revenue)

    # Visualization 5: Top Genres by Profit Margin
    st.subheader("Top Genres by Profit Margin")
    def plot_top_genres_by_profit_margin(ax):
        top_profit_genres = (
            data.groupby("genres_x")["profit_margin"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        sns.barplot(data=top_profit_genres, y="genres_x", x="profit_margin", ax=ax, palette="coolwarm")
        ax.set_title("Top Genres by Profit Margin")
        ax.set_xlabel("Average Profit Margin")
        ax.set_ylabel("Genres")
    seaborn_plot(plot_top_genres_by_profit_margin)

# Page 3: Model Prediction
elif page_selection == "Model Prediction":
    st.title("Predict Movie Revenue")

    # Input form for user input
    st.subheader("Input Features")
    budget = st.number_input("Budget ($)", min_value=1_000, max_value=1_000_000_000, step=10_000)
    avg_rating = st.number_input("Average Rating", min_value=0.0, max_value=10.0, step=0.1)
    rating_count = st.number_input("Rating Count", min_value=1, step=1)
    ROI = st.number_input("ROI", min_value=0.0, step=0.01)
    genres = st.text_input("Genres (comma-separated)", placeholder="Action, Drama, Comedy")

    if st.button("Predict Revenue"):
        # Preprocess user input
        input_data = pd.DataFrame(
            {
                "log_budget": [np.log1p(budget)],
                "avg_rating": [avg_rating],
                "rating_count": [rating_count],
                "ROI": [ROI],
                "genres_x": [genres],
            }
        )
        input_data = input_data.replace([np.inf, -np.inf], np.nan).dropna()

        # Prediction
        try:
            log_predicted_revenue = model.predict(input_data)[0]
            predicted_revenue = np.expm1(log_predicted_revenue)
            st.success(f"Predicted Revenue: ${predicted_revenue:,.2f}")
        except Exception as e:
            st.error("Error making prediction. Please check your input values.")

# Page 4: Inference Results
elif page_selection == "Inference Results":
    st.title("Inference Results")

    # Display inference data
    st.dataframe(inference_results)

    # Metrics: Calculate MSE and R²
    if "actual_revenue" in inference_results.columns and "predicted_revenue" in inference_results.columns:
        mse = mean_squared_error(
            inference_results["actual_revenue"], inference_results["predicted_revenue"]
        )
        r2 = r2_score(
            inference_results["actual_revenue"], inference_results["predicted_revenue"]
        )
        st.subheader("Champion Model Performance")
        st.metric("Mean Squared Error (MSE)", f"{mse:,.2f}")
        st.metric("R² Score", f"{r2:.2f}")

    # Visualization: Predicted vs Actual Revenue
    st.subheader("Predicted vs Actual Revenue")
    def plot_predicted_vs_actual(ax):
        sns.scatterplot(data=inference_results, x="actual_revenue", y="predicted_revenue", alpha=0.6, ax=ax)
        ax.set_title("Predicted vs Actual Revenue")
        ax.set_xlabel("Actual Revenue")
        ax.set_ylabel("Predicted Revenue")
    seaborn_plot(plot_predicted_vs_actual)
