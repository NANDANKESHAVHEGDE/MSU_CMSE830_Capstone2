import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import os

# Set page configuration
st.set_page_config(
    page_title="Movie Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #4CAF50;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #333;
    }
    .data-source {
        font-size: 16px;
        margin-bottom: 15px;
        line-height: 1.6;
    }
    .metrics {
        font-size: 18px;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "../Interim_Data/Final_Cleaned_Data.pkl")
    inference_results_path = os.path.join(base_dir, "../Predictions/inference_results.csv")
    model_path = os.path.join(base_dir, "../Model_outputs/champion_model.pkl")

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

# Overview Section
if page_selection == "Overview":
    st.markdown("""
    <style>
        /* Main Container Background */
        .main-container {
            background: linear-gradient(to right, #e3f2fd, #fbe9e7);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 6px 15px rgba(0, 0, 0, 0.25);
        }
        /* Header Styles */
        .header {
            font-size: 38px;
            font-weight: bold;
            color: #1e88e5;
            text-align: center;
            margin-bottom: 20px;
        }
        .sub-header {
            font-size: 26px;
            font-weight: bold;
            color: #e53935;
            margin-top: 25px;
            display: flex;
            align-items: center;
        }
        .sub-header img {
            height: 30px;
            margin-right: 10px;
        }
        .content {
            font-size: 18px;
            color: #424242;
            line-height: 1.8;
            padding: 10px;
        }
        .content ul {
            list-style-type: none;
            padding-left: 0;
        }
        .content ul li {
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }
        .content ul li img {
            height: 20px;
            margin-right: 10px;
        }
        .divider {
            margin: 20px 0;
            border: 1px solid #c5cae9;
        }
        .link {
            color: #1e88e5;
            font-weight: bold;
            text-decoration: none;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    # Header
    st.markdown("<div class='header'>🎥 Movie Analysis Dashboard</div>", unsafe_allow_html=True)

    # Welcome Section
    st.markdown("""
    <div class='content'>
        Welcome to the Movie Analysis Dashboard, a comprehensive and user-friendly platform designed for analyzing and predicting movie performance. 
        This tool provides detailed insights into patterns in movie datasets, enables revenue prediction using machine learning models, and evaluates model performance with intuitive visualizations.
    </div>
    <hr class='divider'>
    """, unsafe_allow_html=True)

    # Objectives Section
    st.markdown("<div class='sub-header'><img src='https://img.icons8.com/color/48/null/goal.png'/> Objectives</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='content'>
        <ul>
            <li><img src='https://img.icons8.com/fluency/20/null/checkmark.png'/> Analyze patterns and trends in large-scale movie datasets.</li>
            <li><img src='https://img.icons8.com/fluency/20/null/checkmark.png'/> Predict movie revenue based on key attributes like budget, ratings, and genres.</li>
            <li><img src='https://img.icons8.com/fluency/20/null/checkmark.png'/> Evaluate machine learning models using intuitive metrics and visualizations.</li>
        </ul>
    </div>
    <hr class='divider'>
    """, unsafe_allow_html=True)

    # Features Section
    st.markdown("<div class='sub-header'><img src='https://img.icons8.com/color/48/null/features-list.png'/> Features</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='content'>
        <ul>
            <li><img src='https://img.icons8.com/fluency/20/null/presentation.png'/> <b>Exploratory Data Analysis (EDA):</b> Provides insights into trends like revenue vs. budget, rating distributions, and top-performing genres.</li>
            <li><img src='https://img.icons8.com/fluency/20/null/checkmark.png'/> <b>Model Prediction:</b> Predicts movie revenue based on input parameters like budget, ratings, and genre.</li>
            <li><img src='https://img.icons8.com/fluency/20/null/data-configuration.png'/> <b>Inference Results:</b> Displays model accuracy using metrics like R² and MAPE.</li>
        </ul>
    </div>
    <hr class='divider'>
    """, unsafe_allow_html=True)

    # Data Science Steps Section
    st.markdown("<div class='sub-header'><img src='https://img.icons8.com/color/48/null/artificial-intelligence.png'/> Data Science Steps</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='content'>
        <ul>
            <li><img src='https://img.icons8.com/color/20/null/database.png'/> <b>Data Collection:</b> Datasets from TMDB, IMDb, and MovieLens with key attributes like revenue, budget, genres, and ratings.</li>
            <li><img src='https://img.icons8.com/color/20/null/broom.png'/> <b>Data Cleaning:</b> Removed missing values, standardized columns, and filtered outliers.</li>
            <li><img src='https://img.icons8.com/color/20/null/settings.png'/> <b>Feature Engineering:</b> Added profit margin and log-transformed variables to improve model performance.</li>
            <li><img src='https://img.icons8.com/color/20/null/flow-chart.png'/> <b>Model Training:</b> Trained regression models to predict revenue based on key features.</li>
            <li><img src='https://img.icons8.com/fluency/20/null/checkmark.png'/> <b>Model Evaluation:</b> Evaluated using R², MAPE, and visual comparisons of actual vs. predicted revenues.</li>
            <li><img src='https://img.icons8.com/color/20/null/dashboard.png'/> <b>Dashboard Development:</b> Designed with Streamlit for data visualization and user interaction.</li>
        </ul>
    </div>
    <hr class='divider'>
    """, unsafe_allow_html=True)

    # Data Sources Section
    st.markdown("<div class='sub-header'><img src='https://img.icons8.com/color/48/null/source-code.png'/> Data Sources</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='content'>
        <ul>
            <li><img src='https://img.icons8.com/fluency/20/null/document.png'/> <b>TMDB Movies Dataset:</b> <a class='link' href='https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata' target='_blank'>TMDB Movie Metadata</a></li>
            <li><img src='https://img.icons8.com/fluency/20/null/document.png'/> <b>IMDb Extensive Dataset:</b> <a class='link' href='https://www.kaggle.com/datasets/stefanoleone992/imdb-extensive-dataset' target='_blank'>IMDb Dataset</a></li>
            <li><img src='https://img.icons8.com/fluency/20/null/document.png'/> <b>MovieLens Dataset:</b> <a class='link' href='https://grouplens.org/datasets/movielens/latest/' target='_blank'>MovieLens Dataset</a></li>
        </ul>
    </div>
    <hr class='divider'>
    """, unsafe_allow_html=True)

    # Technology Stack Section
    st.markdown("<div class='sub-header'><img src='https://img.icons8.com/color/48/null/technology.png'/> Technology Stack</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='content'>
        <ul>
            <li><img src='https://img.icons8.com/color/20/null/code.png'/> Streamlit: Interactive dashboard for data visualization and interaction.</li>
            <li><img src='https://img.icons8.com/fluency/20/null/document.png'/> Pandas: Data manipulation and cleaning.</li>
            <li><img src='https://img.icons8.com/color/20/null/line-chart.png'/> Seaborn & Matplotlib: Data visualization and trend analysis.</li>
            <li><img src='https://img.icons8.com/color/20/null/robot.png'/> Scikit-learn: Machine learning model development and evaluation.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif page_selection == "EDA":
    st.markdown("""
    <style>
        .eda-container {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.15);
        }
        .eda-header {
            font-size: 36px;
            font-weight: bold;
            color: #34495e;
            text-align: center;
            margin-bottom: 20px;
        }
        .eda-subheader {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 25px;
        }
        .eda-content {
            font-size: 18px;
            color: #555555;
            line-height: 1.8;
            margin-left: 20px;
        }
        .eda-content ul {
            list-style-type: none;
            padding-left: 0;
        }
        .eda-content ul li {
            margin-bottom: 10px;
        }
        .eda-content ul li:before {
            content: "• ";
            color: #3498db;
            font-size: 20px;
            margin-right: 5px;
        }
        .divider {
            margin: 20px 0;
            border: 1px solid #cccccc;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='eda-container'>", unsafe_allow_html=True)

    st.markdown("<div class='eda-header'>Exploratory Data Analysis</div>", unsafe_allow_html=True)

    # Visualization 1: Budget vs Revenue
    st.markdown("<div class='eda-subheader'>Relationship Between Budget and Revenue</div>", unsafe_allow_html=True)
    def plot_budget_vs_revenue(ax):
        sns.regplot(x='budget', y='revenue', data=data, scatter_kws={'alpha': 0.7}, line_kws={'color': 'red'}, ax=ax)
        ax.set_title('Relationship Between Budget and Revenue')
        ax.set_xlabel('Budget')
        ax.set_ylabel('Revenue')
    seaborn_plot(plot_budget_vs_revenue)
    st.markdown("""
    <div class='eda-content'>
        <ul>
            <li>There is a strong positive correlation between budget and revenue, showing that higher budgets typically lead to higher revenues.</li>
            <li>Excessively high budgets often face diminishing returns, where additional spending does not proportionally increase revenue.</li>
            <li>Outliers exist where low-budget movies achieve exceptional success due to factors like compelling storytelling or unique marketing strategies.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Visualization 2: Distribution of Ratings
    st.markdown("<div class='eda-subheader'>Distribution of Average Ratings</div>", unsafe_allow_html=True)
    def plot_avg_rating_dist(ax):
        sns.histplot(data['avg_rating'], bins=20, kde=True, color='blue', ax=ax)
        ax.set_title('Distribution of Average Ratings')
        ax.set_xlabel('Average Rating')
        ax.set_ylabel('Frequency')
    seaborn_plot(plot_avg_rating_dist)
    st.markdown("""
    <div class='eda-content'>
        <ul>
            <li>The majority of movies have ratings between 6 and 8, representing a clustering of moderately well-received films.</li>
            <li>Ratings below 4 or above 9 are rare, indicating that very poor or exceptional films are uncommon.</li>
            <li>The skew toward higher ratings may indicate better production quality or more lenient rating practices in recent years.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Visualization 3: Trends in Revenue Over the Years
    st.markdown("<div class='eda-subheader'>Trends in Revenue Over the Years</div>", unsafe_allow_html=True)
    def plot_revenue_trends(ax):
        data['release_year'] = pd.to_numeric(data['release_year'], errors='coerce')
        valid_data = data.dropna(subset=['release_year'])
        valid_data['release_year'] = valid_data['release_year'].astype(int)
        yearly_revenue = valid_data.groupby('release_year')['revenue'].mean()
        ax.plot(yearly_revenue.index, yearly_revenue.values, marker='o', color='#2980b9')
        ax.set_title('Trends in Average Revenue Over the Years')
        ax.set_xlabel('Release Year')
        ax.set_ylabel('Average Revenue')
    seaborn_plot(plot_revenue_trends)
    st.markdown("""
    <div class='eda-content'>
        <ul>
            <li>There has been a consistent rise in average revenue over the years, attributed to inflation, global markets, and blockbuster franchises.</li>
            <li>The recent decline in revenue may be due to increased streaming adoption or global economic changes.</li>
            <li>Significant spikes in certain years reflect the influence of major blockbuster releases.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Visualization 4: Top Genres by Revenue
    st.markdown("<div class='eda-subheader'>Top Genres by Average Revenue</div>", unsafe_allow_html=True)
    def plot_top_genres_by_revenue(ax):
        top_genres = data.groupby('genres_x')['revenue'].mean().sort_values(ascending=False).head(10)
        sns.barplot(x=top_genres.values, y=top_genres.index, palette="viridis", ax=ax)
        ax.set_title('Top Genres by Average Revenue')
        ax.set_xlabel('Average Revenue')
        ax.set_ylabel('Genres')
    seaborn_plot(plot_top_genres_by_revenue)
    st.markdown("""
    <div class='eda-content'>
        <ul>
            <li>Action, Adventure, and Sci-Fi lead in average revenue, supported by their global appeal and large fan bases.</li>
            <li>Genres like Animation and Fantasy perform well due to family-oriented themes and international reach.</li>
            <li>Comedy generates lower revenue on average, possibly due to smaller budgets and limited international appeal.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Visualization 5: Top Genres by Profit Margin
    st.markdown("<div class='eda-subheader'>Top Genres by Profit Margin</div>", unsafe_allow_html=True)
    def plot_top_genres_by_profit_margin(ax):
        data['profit_margin'] = (data['revenue'] - data['budget']) / data['revenue']
        profit_by_genre = data.groupby('genres_x')['profit_margin'].mean().sort_values(ascending=False).head(10)
        sns.barplot(x=profit_by_genre.values, y=profit_by_genre.index, palette="Blues_r", ax=ax)
        ax.set_title('Top Genres by Profit Margin')
        ax.set_xlabel('Average Profit Margin')
        ax.set_ylabel('Genres')
    seaborn_plot(plot_top_genres_by_profit_margin)
    st.markdown("""
    <div class='eda-content'>
        <ul>
            <li>Genres like Adventure and Musical excel in profit margins, balancing modest budgets with strong revenue potential.</li>
            <li>Family-oriented genres consistently perform well, reflecting their loyal audience base and cost-effective production.</li>
            <li>High profit margins for niche genres highlight their ability to achieve success with controlled spending.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Page 3: Model Prediction
elif page_selection == "Model Prediction":
    # CSS Styling
    st.markdown("""
    <style>
        .container {
            background: linear-gradient(to right, #e0f7fa, #e1bee7);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.25);
        }
        .header {
            font-size: 32px;
            font-weight: bold;
            color: #4caf50;
            text-align: center;
            margin-bottom: 20px;
        }
        .sub-header {
            font-size: 24px;
            font-weight: bold;
            color: #7b1fa2;
            margin-top: 25px;
        }
        .error-box {
            background-color: #ffcdd2;
            padding: 15px;
            border-radius: 8px;
            font-size: 18px;
            color: #d32f2f;
        }
        .success-box {
            background-color: #c8e6c9;
            padding: 15px;
            border-radius: 8px;
            font-size: 18px;
            color: #388e3c;
        }
        .button {
            background-color: #4caf50;
            color: white;
            font-size: 18px;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
        }
        .button:hover {
            background-color: #388e3c;
        }
    </style>
    """, unsafe_allow_html=True)

    # Load the champion model
    try:
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)
    except FileNotFoundError:
        st.markdown(f"<div class='error-box'>Model file not found. Please check the path: {model_path}</div>", unsafe_allow_html=True)
        pipeline = None
    except Exception as e:
        st.markdown(f"<div class='error-box'>Error loading model: {e}</div>", unsafe_allow_html=True)
        pipeline = None

    # Input form for user input
    st.markdown("<div class='container'>", unsafe_allow_html=True)
    st.markdown("<div class='header'>Predict Movie Revenue</div>", unsafe_allow_html=True)

    st.markdown("<div class='sub-header'>Input Features</div>", unsafe_allow_html=True)
    budget = st.number_input("Budget ($)", min_value=1_000, max_value=1_000_000_000, step=10_000)
    avg_rating = st.number_input("Average Rating", min_value=0.0, max_value=10.0, step=0.1)
    rating_count = st.number_input("Rating Count", min_value=1, step=1)
    ROI = st.number_input("ROI", min_value=0.0, step=0.01)
    genres = st.text_input("Genres (comma-separated)", placeholder="Action, Drama, Comedy")

    if st.button("Predict Revenue", key="predict_button"):
        if pipeline:
            try:
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

                # Make prediction
                log_predicted_revenue = pipeline.predict(input_data)[0]
                predicted_revenue = np.expm1(log_predicted_revenue)  # Convert from log scale

                st.markdown(f"<div class='success-box'>Predicted Revenue: ${predicted_revenue:,.2f}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"<div class='error-box'>Error making prediction: {e}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='error-box'>Model is not loaded correctly. Please check the champion_model.pkl file.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Page 3: Model Prediction
elif page_selection == "Model Prediction":
    # CSS Styling
    st.markdown("""
    <style>
        .container {
            background: linear-gradient(to right, #e8f5e9, #ede7f6);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.25);
        }
        .header {
            font-size: 32px;
            font-weight: bold;
            color: #4caf50;
            text-align: center;
            margin-bottom: 20px;
        }
        .sub-header {
            font-size: 24px;
            font-weight: bold;
            color: #7b1fa2;
            margin-top: 25px;
        }
        .error-box {
            background-color: #ffcdd2;
            padding: 15px;
            border-radius: 8px;
            font-size: 18px;
            color: #d32f2f;
        }
        .success-box {
            background-color: #c8e6c9;
            padding: 15px;
            border-radius: 8px;
            font-size: 18px;
            color: #388e3c;
        }
    </style>
    """, unsafe_allow_html=True)

    # Load the champion model
    try:
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)
    except FileNotFoundError:
        st.markdown(f"<div class='error-box'>Model file not found. Please check the path: {model_path}</div>", unsafe_allow_html=True)
        pipeline = None
    except Exception as e:
        st.markdown(f"<div class='error-box'>Error loading model: {e}</div>", unsafe_allow_html=True)
        pipeline = None

    # Input form for user input
    st.markdown("<div class='container'>", unsafe_allow_html=True)
    st.markdown("<div class='header'>Predict Movie Revenue</div>", unsafe_allow_html=True)

    st.markdown("<div class='sub-header'>Input Features</div>", unsafe_allow_html=True)
    budget = st.number_input("Budget ($)", min_value=1_000, max_value=1_000_000_000, step=10_000)
    avg_rating = st.number_input("Average Rating", min_value=0.0, max_value=10.0, step=0.1)
    rating_count = st.number_input("Rating Count", min_value=1, step=1)
    ROI = st.number_input("ROI", min_value=0.0, step=0.01)
    genres = st.text_input("Genres (comma-separated)", placeholder="Action, Drama, Comedy")

    if st.button("Predict Revenue", key="predict_button"):
        if pipeline:
            try:
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

                # Make prediction
                log_predicted_revenue = pipeline.predict(input_data)[0]
                predicted_revenue = np.expm1(log_predicted_revenue)  # Convert from log scale

                st.markdown(f"<div class='success-box'>Predicted Revenue: ${predicted_revenue:,.2f}</div>", unsafe_allow_html=True)

                # Visualization 1: Input Feature Importance
                st.subheader("Feature Breakdown")
                feature_values = {
                    "Budget": budget,
                    "Average Rating": avg_rating,
                    "Rating Count": rating_count,
                    "ROI": ROI
                }
                st.bar_chart(pd.DataFrame(feature_values, index=["Value"]).T)

                # Visualization 2: Distribution of Predicted Revenue
                st.subheader("Revenue Prediction Distribution")
                data['log_budget'] = np.log1p(data['budget'])
                predictions = pipeline.predict(data[["log_budget", "avg_rating", "rating_count", "ROI", "genres_x"]])
                all_predicted_revenue = np.expm1(predictions)

                fig, ax = plt.subplots()
                sns.histplot(all_predicted_revenue, bins=30, kde=True, color='blue', ax=ax)
                ax.axvline(predicted_revenue, color='red', linestyle='--', label="Your Prediction")
                ax.set_title("Predicted Revenue Distribution")
                ax.set_xlabel("Revenue ($)")
                ax.set_ylabel("Frequency")
                ax.legend()
                st.pyplot(fig)

            except Exception as e:
                st.markdown(f"<div class='error-box'>Error making prediction: {e}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='error-box'>Model is not loaded correctly. Please check the champion_model.pkl file.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Page 4: Inference Results
elif page_selection == "Inference Results":
    # CSS Styling
    st.markdown("""
    <style>
        .container {
            background: linear-gradient(to right, #e3f2fd, #ede7f6);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.25);
        }
        .header {
            font-size: 32px;
            font-weight: bold;
            color: #0288d1;
            text-align: center;
            margin-bottom: 20px;
        }
        .sub-header {
            font-size: 24px;
            font-weight: bold;
            color: #6a1b9a;
            margin-top: 25px;
        }
        .success-box {
            background-color: #c8e6c9;
            padding: 15px;
            border-radius: 8px;
            font-size: 18px;
            color: #388e3c;
        }
        .error-box {
            background-color: #ffcdd2;
            padding: 15px;
            border-radius: 8px;
            font-size: 18px;
            color: #d32f2f;
        }
    </style>
    """, unsafe_allow_html=True)

    # Load the inference results
    st.markdown("<div class='container'>", unsafe_allow_html=True)
    st.markdown("<div class='header'>Inference Results</div>", unsafe_allow_html=True)

    try:
        # Load inference data
        inference_results = pd.read_csv(inference_path)

        # Metrics: Mean Absolute Percentage Error (MAPE)
        st.markdown("<div class='sub-header'>Model Performance Metrics</div>", unsafe_allow_html=True)
        if 'actual_revenue' in inference_results.columns and 'predicted_revenue' in inference_results.columns:
            actual_revenue = inference_results['actual_revenue']
            predicted_revenue = inference_results['predicted_revenue']

            # Calculate MAPE
            mape = np.mean(np.abs((actual_revenue - predicted_revenue) / actual_revenue)) * 100

            st.markdown(f"<div class='success-box'>Mean Absolute Percentage Error (MAPE): {mape:.2f}%</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='error-box'>Missing columns for evaluation metrics.</div>", unsafe_allow_html=True)

        # Visualization 1: Predicted vs Actual Revenue
        st.markdown("<div class='sub-header'>Predicted vs Actual Revenue</div>", unsafe_allow_html=True)
        fig1, ax1 = plt.subplots()
        sns.scatterplot(x=inference_results['actual_revenue'], y=inference_results['predicted_revenue'], alpha=0.7, ax=ax1)
        ax1.plot([0, inference_results['actual_revenue'].max()], [0, inference_results['actual_revenue'].max()], color='red', linestyle='--', label='Perfect Prediction')
        ax1.set_title("Predicted vs Actual Revenue")
        ax1.set_xlabel("Actual Revenue")
        ax1.set_ylabel("Predicted Revenue")
        ax1.legend()
        st.pyplot(fig1)

        # Visualization 2: Distribution of Residuals
        st.markdown("<div class='sub-header'>Distribution of Residuals</div>", unsafe_allow_html=True)
        residuals = inference_results['actual_revenue'] - inference_results['predicted_revenue']
        fig2, ax2 = plt.subplots()
        sns.histplot(residuals, kde=True, color='purple', ax=ax2)
        ax2.set_title("Residual Distribution (Actual - Predicted)")
        ax2.set_xlabel("Residuals")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

        # Visualization 3: Actual Revenue Distribution
        st.markdown("<div class='sub-header'>Distribution of Actual Revenue</div>", unsafe_allow_html=True)
        fig3, ax3 = plt.subplots()
        sns.histplot(inference_results['actual_revenue'], bins=30, kde=True, color='green', ax=ax3)
        ax3.set_title("Actual Revenue Distribution")
        ax3.set_xlabel("Revenue")
        ax3.set_ylabel("Frequency")
        st.pyplot(fig3)

        # Display the first few rows of the inference results
        st.markdown("<div class='sub-header'>Sample Inference Results</div>", unsafe_allow_html=True)
        st.dataframe(inference_results.head())

    except FileNotFoundError:
        st.markdown(f"<div class='error-box'>Inference results file not found. Please check the path: {inference_path}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"<div class='error-box'>Error loading inference results: {e}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
