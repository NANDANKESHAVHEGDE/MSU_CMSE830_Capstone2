import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

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
    model = joblib.load(model_path)

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

# Page 1: Overview
if page_selection == "Overview":
    st.title("Movie Analysis Dashboard")
    st.markdown("""
    ### Overview
    This dashboard provides insights into the movie dataset, explores trends, and predicts movie revenue.
    
    #### Data Sources:
    - [Kaggle Movie Dataset](https://www.kaggle.com/datasets)
    - [Netflix Open Data](https://data.netflix.com)
    
    #### Key Features:
    - Exploratory Data Analysis (EDA): Gain insights into key patterns and trends.
    - Revenue Prediction: Predict movie revenue using a trained machine learning model.
    - Inference Results: Analyze the model's predictions compared to actual revenue.

    #### Steps:
    1. **Data Cleaning**: Handle missing values and outliers.
    2. **EDA**: Visualize and understand the dataset.
    3. **Model Training**: Train models and select the champion model based on performance metrics.
    4. **Model Deployment**: Deploy the model for revenue prediction and analysis.

    Navigate through the sections to explore the insights and predictions!
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
    st.markdown("""
    **Insights**:
    - Positive correlation between budget and revenue.
    - High-budget movies don't always guarantee high revenue.
    """)

    # Visualization 2: Distribution of Average Ratings
    st.subheader("Distribution of Average Ratings")
    def plot_avg_rating_dist(ax):
        sns.histplot(data=data, x="avg_rating", bins=20, kde=True, ax=ax, color="skyblue")
        ax.set_title("Distribution of Average Ratings")
        ax.set_xlabel("Average Rating")
        ax.set_ylabel("Frequency")
    seaborn_plot(plot_avg_rating_dist)
    st.markdown("""
    **Insights**:
    - Most movies are rated between 6 and 8.
    - Few movies have exceptionally high or low ratings.
    """)

    # Visualization 3: Trends in Revenue Over Time
    st.subheader("Trends in Revenue Over Time")
    def plot_revenue_trends(ax):
        if "release_year" in data.columns:
            data["release_year"] = pd.to_numeric(data["release_year"], errors="coerce")
            cleaned_data = data.dropna(subset=["release_year"])
        else:
            st.error("Column `release_year` not found in the dataset.")
            return

        yearly_revenue = cleaned_data.groupby("release_year")["revenue"].mean().sort_index()
        sns.lineplot(x=yearly_revenue.index, y=yearly_revenue.values, ax=ax, marker="o")
        ax.set_title("Trends in Average Revenue Over the Years")
        ax.set_xlabel("Release Year")
        ax.set_ylabel("Average Revenue")
        ax.grid(True)
    seaborn_plot(plot_revenue_trends)
    st.markdown("""
    **Insights**:
    - Significant growth in revenue from the 2000s onward.
    - Possible decline in revenue in recent years due to industry shifts.
    """)

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
    st.markdown("""
    **Insights**:
    - Genres like "Adventure," "Action," and "Sci-Fi" dominate revenue.
    - More niche genres like "Documentary" generate lower average revenue.
    """)

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
    st.markdown("""
    **Insights**:
    - Genres with combinations like "Adventure|Children|Comedy|Musical" have very high profit margins.
    - Multi-genre movies catering to family audiences often yield higher returns.
    """)

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

    # Visualization: Predicted vs Actual Revenue
    st.subheader("Predicted vs Actual Revenue")
    def plot_predicted_vs_actual(ax):
        sns.scatterplot(data=inference_results, x="actual_revenue", y="predicted_revenue", alpha=0.6, ax=ax)
        ax.set_title("Predicted vs Actual Revenue")
        ax.set_xlabel("Actual Revenue")
        ax.set_ylabel("Predicted Revenue")
    seaborn_plot(plot_predicted_vs_actual)

    # Metrics: Champion model performance
    st.subheader("Champion Model Performance")
    mse = ((inference_results["actual_revenue"] - inference_results["predicted_revenue"]) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(inference_results["actual_revenue"] - inference_results["predicted_revenue"]).mean()
    r2 = 1 - (sum((inference_results["actual_revenue"] - inference_results["predicted_revenue"]) ** 2) /
              sum((inference_results["actual_revenue"] - inference_results["actual_revenue"].mean()) ** 2))
    mape = np.mean(np.abs((inference_results["actual_revenue"] - inference_results["predicted_revenue"]) /
                          inference_results["actual_revenue"])) * 100

    st.markdown(f"""
    - **Mean Squared Error (MSE)**: {mse:,.2f}
    - **Root Mean Squared Error (RMSE)**: {rmse:,.2f}
    - **Mean Absolute Error (MAE)**: {mae:,.2f}
    - **R² Score**: {r2:.2f}
    - **Mean Absolute Percentage Error (MAPE)**: {mape:.2f}%
    """)
