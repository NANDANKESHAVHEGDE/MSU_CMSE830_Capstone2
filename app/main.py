import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import os
import plotly.express as px
from sklearn.model_selection import train_test_split



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
pages = ["Overview", "Data Diagnostics", "EDA & Insights", "Model Training", "Model Inferencing","Model Prediction"]
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
    st.markdown("<div class='sub-header'><img src='https://img.icons8.com/color/20/null/settings.png'/> Features</div>", unsafe_allow_html=True)
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

    # Technology Stack Section with Fixed Icons
    st.markdown("<div class='sub-header'><img src='https://img.icons8.com/fluency/20/null/checkmark.png'/> Technology Stack</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='content'>
        <ul>
            <li><img src='https://img.icons8.com/fluency/20/source-code.png'/> Streamlit: Interactive dashboard for data visualization and interaction.</li>
            <li><img src='https://img.icons8.com/fluency/20/database.png'/> Pandas: Data manipulation and cleaning.</li>
            <li><img src='https://img.icons8.com/fluency/20/combo-chart.png'/> Seaborn & Matplotlib: Data visualization and trend analysis.</li>
            <li><img src='https://img.icons8.com/fluency/20/artificial-intelligence.png'/> Scikit-learn: Machine learning model development and evaluation.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # GitHub Repository Section
    st.markdown("<div class='sub-header'><img src='https://img.icons8.com/fluency/48/github.png'/> GitHub Repository</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="content">
        For more details and to explore the code, visit the <a class="link" href="https://github.com/NANDANKESHAVHEGDE/MSU_CMSE830_Capstone2" target="_blank">GitHub Repository</a>.
    </div>
    <hr class='divider'>
    """, unsafe_allow_html=True)

    # Footer Section with Enhanced Positioning
    st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 10px;
            width: 100%;
            background: linear-gradient(to right, #f9f9f9, #e0e0e0);
            padding: 10px 20px;
            font-size: 14px;
            color: #555555;
            text-align: center;
            border-top: 1px solid #dddddd;
            box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.1);
        }
        .footer img {
            vertical-align: middle;
            margin-right: 10px;
        }
        .footer a {
            font-weight: bold;
            color: #4CAF50;
            text-decoration: none;
        }
        .footer a:hover {
            color: #388E3C;
        }
    </style>
    <div class='footer'>
        <img src='https://img.icons8.com/ios-filled/50/000000/info.png' width='30' alt='Footer Icon' />
        This dashboard is a final project for the CMSE830 course of the MSDS program at Michigan State University. 
    </div>
    """, unsafe_allow_html=True)

# Data Diagnostics Page
elif page_selection == "Data Diagnostics":
    st.markdown("""
    <style>
        .diagnostics-container {
            background: linear-gradient(to right, #f3e5f5, #e8f5e9);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.25);
        }
        .diagnostics-header {
            font-size: 32px;
            font-weight: bold;
            color: #6a1b9a;
            text-align: center;
            margin-bottom: 20px;
        }
        .diagnostics-subheader {
            font-size: 24px;
            font-weight: bold;
            color: #4caf50;
            margin-top: 25px;
        }
        .diagnostics-content {
            font-size: 18px;
            color: #424242;
            line-height: 1.8;
        }
        .diagnostics-content ul {
            list-style-type: none;
            padding-left: 0;
        }
        .diagnostics-content ul li {
            margin-bottom: 10px;
        }
        .diagnostics-content ul li:before {
            content: "• ";
            color: #8e24aa;
            font-size: 20px;
            margin-right: 5px;
        }
        .divider {
            margin: 20px 0;
            border: 1px solid #cccccc;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='diagnostics-container'>", unsafe_allow_html=True)

    st.markdown("<div class='diagnostics-header'>Data Diagnostics</div>", unsafe_allow_html=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "../Interim_Data/Data_prep.pkl")
    data = pd.read_pickle(data_path)

    # Display Sample Data
    st.markdown("<div class='diagnostics-subheader'>Sample Data</div>", unsafe_allow_html=True)
    try:
        st.write(data.sample(n=10, random_state=42))  # Display a random sample of 10 rows
    except Exception as e:
        st.error(f"Error displaying sample data: {e}")

    # Correlation Heatmap
    st.markdown("<div class='diagnostics-subheader'>Correlation Heatmap</div>", unsafe_allow_html=True)
    try:
        numeric_columns = data.select_dtypes(include=[np.number])
        correlation = numeric_columns.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)
        st.markdown("""
        <div class='diagnostics-content'>
        <ul>
            <li><b>Purpose:</b> The heatmap shows how strongly numerical variables are related.</li>
            <li><b>Insights:</b> Strong positive correlation between budget and revenue, while average ratings show weak relationships with revenue.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error generating correlation heatmap: {e}")

    # Pairplot Analysis
    st.markdown("<div class='diagnostics-subheader'>Pairplot Analysis</div>", unsafe_allow_html=True)
    try:
        sampled_data = data.sample(n=min(500, len(data)), random_state=42)  # Sample data for performance
        pairplot_fig = sns.pairplot(sampled_data.select_dtypes(include=[np.number]))
        st.pyplot(pairplot_fig.fig)  # Use thread-safe figure object
        st.markdown("""
        <div class='diagnostics-content'>
        <ul>
            <li><b>Purpose:</b> Pairwise relationships between features help identify patterns and clusters.</li>
            <li><b>Insights:</b> Positive trends between budget and revenue, weak interactions with average ratings.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error generating pairplot: {e}")

    # Missing Values Analysis
    st.markdown("<div class='diagnostics-subheader'>Missing Values Analysis</div>", unsafe_allow_html=True)
    try:
        missing_values = data.isnull().sum()
        missing_df = pd.DataFrame({
            "Column": missing_values.index,
            "Missing Values": missing_values.values,
            "Percentage Missing": (missing_values.values / len(data)) * 100
        }).sort_values(by="Percentage Missing", ascending=False)

        st.dataframe(missing_df)

        st.markdown("""
        <div class='diagnostics-content'>
        <ul>
            <li><b>Purpose:</b> Helps identify missing values in the dataset for better data preprocessing.</li>
            <li><b>Insights:</b> Missing values found in budget, revenue, and runtime, requiring imputation techniques.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # Observations on Missingness
        st.markdown("<div class='diagnostics-subheader'>Observations from Missing Values Analysis</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='diagnostics-content'>
        <ul>
            <li><b>Low Correlations (Near Zero):</b> Weak correlations between missingness in budget, revenue, and runtime with other variables suggest data may be MCAR (Missing Completely at Random).</li>
            <li><b>NaN Correlations:</b> Overlapping missingness in columns prevents meaningful correlation calculations.</li>
            <li><b>Moderate Negative Correlation:</b> Negative correlation with rating_count (-0.32) indicates that less popular movies are more likely to have missing data.</li>
            <li><b>Imputation Guidance:</b>
                <ul>
                    <li><b>MCAR:</b> No pattern in missingness.</li>
                    <li><b>MAR:</b> Missingness correlates with observed variables; regression or KNN imputation methods are applicable.</li>
                    <li><b>MNAR:</b> Missingness depends on the variable itself, e.g., low-budget movies not reporting revenue. Domain-specific imputation is needed.</li>
                </ul>
            </li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error analyzing missing values: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Summary of Actions
    st.markdown("<div class='diagnostics-subheader'>Summary of Actions</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='summary-points'>
        <ul>
            <li><b>Data Loading:</b> Cleaned dataset loaded and prepared for analysis.</li>
            <li><b>Data Preparation:</b> Renamed columns, handled missing and infinite values.</li>
            <li><b>Correlation Analysis:</b> Analyzed correlations among numeric features with a heatmap.</li>
            <li><b>Pairplot Analysis:</b> Explored feature relationships visually using sampled data.</li>
            <li><b>Missing Values Analysis:</b> Identified missing values to guide data preprocessing steps.</li>
            <li><b>Error Handling:</b> Ensured graceful handling of exceptions for errors in data or visualizations.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Footer Section with Enhanced Positioning
    st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 10px;
            width: 100%;
            background: linear-gradient(to right, #f9f9f9, #e0e0e0);
            padding: 10px 20px;
            font-size: 14px;
            color: #555555;
            text-align: center;
            border-top: 1px solid #dddddd;
            box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.1);
        }
        .footer img {
            vertical-align: middle;
            margin-right: 10px;
        }
        .footer a {
            font-weight: bold;
            color: #4CAF50;
            text-decoration: none;
        }
        .footer a:hover {
            color: #388E3C;
        }
    </style>
    <div class='footer'>
        <img src='https://img.icons8.com/ios-filled/50/000000/info.png' width='30' alt='Footer Icon' />
        This dashboard is a final project for the CMSE830 course of the MSDS program at Michigan State University. 
    </div>
    """, unsafe_allow_html=True)

elif page_selection == "EDA & Insights":
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

    # Interactive Plot 1: Budget vs Revenue
    st.markdown("<div class='eda-subheader'>Interactive: Relationship Between Budget and Revenue</div>", unsafe_allow_html=True)
    try:
        fig = px.scatter(data, x='budget', y='revenue', color='avg_rating',
                         title="Interactive Scatterplot of Budget vs Revenue",
                         labels={'budget': 'Budget', 'revenue': 'Revenue', 'avg_rating': 'Average Rating'},
                         hover_data=['genres_x'])
        fig.update_traces(marker=dict(size=10, opacity=0.6))
        st.plotly_chart(fig)
        st.markdown("""
        <div class='eda-content'>
            <ul>
                <li>Interactive scatterplot allows users to hover over points to view details like genres and ratings.</li>
                <li>Visualizes the strong positive correlation between budget and revenue, with color indicating average ratings.</li>
                <li>Outliers with low-budget movies achieving high revenues are easily identified.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error creating interactive plot: {e}")

    # Interactive Plot 2: Trends in Revenue Over the Years
    st.markdown("<div class='eda-subheader'>Interactive: Trends in Revenue Over the Years</div>", unsafe_allow_html=True)
    try:
        # Clean and preprocess the data for release_year
        valid_data = data.copy()
        valid_data['release_year'] = pd.to_numeric(valid_data['release_year'], errors='coerce')  # Convert invalid entries to NaN
        valid_data = valid_data.dropna(subset=['release_year', 'revenue'])  # Drop rows with NaN values
        valid_data['release_year'] = valid_data['release_year'].astype(int)  # Ensure integer values
        yearly_revenue = valid_data.groupby('release_year')['revenue'].mean().reset_index()

        fig = px.line(yearly_revenue, x='release_year', y='revenue',
                      title="Interactive Line Plot of Revenue Trends Over the Years",
                      labels={'release_year': 'Release Year', 'revenue': 'Average Revenue'})
        fig.update_traces(line=dict(color='#2980b9', width=3))
        st.plotly_chart(fig)
        st.markdown("""
        <div class='eda-content'>
            <ul>
                <li>Interactive line chart enables zooming and panning to explore revenue trends in detail.</li>
                <li>Consistent rise in revenue over the years highlights global distribution and franchise growth.</li>
                <li>Recent dips may reflect streaming adoption and global economic changes.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error creating interactive plot: {e}")

    # Regular Visualization 3: Distribution of Ratings
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

    # Regular Visualization 4: Top Genres by Revenue
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

    # Regular Visualization 5: Top Genres by Profit Margin
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

    # Footer Section with Enhanced Positioning
    st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 10px;
            width: 100%;
            background: linear-gradient(to right, #f9f9f9, #e0e0e0);
            padding: 10px 20px;
            font-size: 14px;
            color: #555555;
            text-align: center;
            border-top: 1px solid #dddddd;
            box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.1);
        }
        .footer img {
            vertical-align: middle;
            margin-right: 10px;
        }
        .footer a {
            font-weight: bold;
            color: #4CAF50;
            text-decoration: none;
        }
        .footer a:hover {
            color: #388E3C;
        }
    </style>
    <div class='footer'>
        <img src='https://img.icons8.com/ios-filled/50/000000/info.png' width='30' alt='Footer Icon' />
        This dashboard is a final project for the CMSE830 course of the MSDS program at Michigan State University. 
    </div>
    """, unsafe_allow_html=True)


elif page_selection == "Model Training":
    # Page Styling
    st.markdown("""
    <style>
        .training-container {
            background: linear-gradient(to right, #f3f9ff, #f7f8fc);
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.1);
            margin: 20px;
        }
        .training-header {
            font-size: 40px;
            font-weight: 700;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .training-subheader {
            font-size: 26px;
            font-weight: 600;
            color: #34495e;
            margin-top: 30px;
            text-align: left;
        }
        .training-content {
            font-size: 18px;
            color: #4f4f4f;
            line-height: 1.8;
            margin-left: 20px;
        }
        .training-content ul, .training-content ol {
            padding-left: 30px;
            list-style: none;
        }
        .training-content ul li, .training-content ol li {
            margin-bottom: 10px;
            position: relative;
            padding-left: 20px;
        }
        .training-content ul li:before, .training-content ol li:before {
            content: "✔";
            color: #3498db;
            font-weight: bold;
            position: absolute;
            left: 0;
            top: 0;
        }
        .divider {
            margin: 30px 0;
            border: 1px solid #e0e0e0;
        }
        .image-container {
            text-align: center;
            margin-top: 20px;
            margin-bottom: 40px;
        }
        .image-caption {
            font-size: 16px;
            color: #7f8c8d;
            margin-top: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Main Container
    st.markdown("<div class='training-container'>", unsafe_allow_html=True)

    # Header
    st.markdown("<div class='training-header'>Model Training Process</div>", unsafe_allow_html=True)

    # Logical Steps Section
    st.markdown("<div class='training-subheader'>Logical Steps in Model Training</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='training-content'>
        <ol>
            <li><b>Feature Engineering:</b> Added calculated fields like profit margin, ROI, and applied log transformations to reduce skewness.</li>
            <li><b>Data Cleaning:</b> Handled missing or invalid values and removed rows with NaNs in critical columns.</li>
            <li><b>Feature and Target Selection:</b> Selected relevant features (e.g., budget, ratings) and the target (log revenue).</li>
            <li><b>Data Splitting:</b> Divided data into 80% training and 20% testing sets.</li>
            <li><b>Preprocessing:</b> Applied scaling for numerical data and encoding for categorical data.</li>
            <li><b>Model Training:</b> Compared XGBoost and Random Forest models using Cross-Validation MSE.</li>
            <li><b>Champion Model Selection:</b> Selected Random Forest as the best-performing model based on MSE.</li>
            <li><b>Model Evaluation:</b> Evaluated test set performance using MSE, RMSE, and MAE.</li>
            <li><b>Residual Analysis:</b> Analyzed prediction errors using a Residual Plot.</li>
            <li><b>Predicted vs. Actual:</b> Compared predicted and actual revenues using a scatterplot.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Results Section
    st.markdown("<div class='training-subheader'>Model Training Results</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='training-content'>
        <ul>
            <li><b>XGBoost:</b> Cross-Validation MSE = 0.64</li>
            <li><b>Random Forest:</b> Cross-Validation MSE = 0.06</li>
            <li><b>Champion Model:</b> Random Forest</li>
            <li><b>Test Performance:</b>
                <ul>
                    <li><b>MSE:</b> 0.0066</li>
                    <li><b>RMSE:</b> 0.0816</li>
                    <li><b>MAE:</b> 0.0167</li>
                </ul>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Visualizations Section
    st.markdown("<div class='training-subheader'>Residual Plot</div>", unsafe_allow_html=True)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plot1_path = os.path.join(base_dir, "../Model_Training/plot1.PNG")
    st.markdown("<div class='image-container'>", unsafe_allow_html=True)
    st.image(plot1_path, use_column_width=True)
    st.markdown("<div class='image-caption'>Residual Plot: Predicted vs. Residuals</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='training-subheader'>Predicted vs. Actual Revenue</div>", unsafe_allow_html=True)
    plot2_path = os.path.join(base_dir, "../Model_Training/plot2.PNG")
    st.markdown("<div class='image-container'>", unsafe_allow_html=True)
    st.image(plot2_path, use_column_width=True)
    st.markdown("<div class='image-caption'>Predicted vs. Actual Revenue</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # End of Container
    st.markdown("</div>", unsafe_allow_html=True)

    # Footer Section with Enhanced Positioning
    st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 10px;
            width: 100%;
            background: linear-gradient(to right, #f9f9f9, #e0e0e0);
            padding: 10px 20px;
            font-size: 14px;
            color: #555555;
            text-align: center;
            border-top: 1px solid #dddddd;
            box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.1);
        }
        .footer img {
            vertical-align: middle;
            margin-right: 10px;
        }
        .footer a {
            font-weight: bold;
            color: #4CAF50;
            text-decoration: none;
        }
        .footer a:hover {
            color: #388E3C;
        }
    </style>
    <div class='footer'>
        <img src='https://img.icons8.com/ios-filled/50/000000/info.png' width='30' alt='Footer Icon' />
        This dashboard is a final project for the CMSE830 course of the MSDS program at Michigan State University. 
    </div>
    """, unsafe_allow_html=True)

# Page 5: Inference Results
elif page_selection == "Model Inferencing":
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
        base_dir = os.path.dirname(os.path.abspath(__file__))
        inference_path = os.path.join(base_dir, "../Predictions/inference_results.csv")
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

    except FileNotFoundError:
        st.markdown(f"<div class='error-box'>Inference results file not found. Please check the path: {inference_path}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"<div class='error-box'>Error loading inference results: {e}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Footer Section with Enhanced Positioning
    st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 10px;
            width: 100%;
            background: linear-gradient(to right, #f9f9f9, #e0e0e0);
            padding: 10px 20px;
            font-size: 14px;
            color: #555555;
            text-align: center;
            border-top: 1px solid #dddddd;
            box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.1);
        }
        .footer img {
            vertical-align: middle;
            margin-right: 10px;
        }
        .footer a {
            font-weight: bold;
            color: #4CAF50;
            text-decoration: none;
        }
        .footer a:hover {
            color: #388E3C;
        }
    </style>
    <div class='footer'>
        <img src='https://img.icons8.com/ios-filled/50/000000/info.png' width='30' alt='Footer Icon' />
        This dashboard is a final project for the CMSE830 course of the MSDS program at Michigan State University. 
    </div>
    """, unsafe_allow_html=True)

elif page_selection == "Model Prediction":
    # CSS Styling
    st.markdown("""
    <style>
        .container {
            background: linear-gradient(to bottom right, #e3f2fd, #f1f8e9);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.2);
        }
        .header {
            font-size: 36px;
            font-weight: bold;
            color: #1a237e;
            text-align: center;
            margin-bottom: 25px;
        }
        .sub-header {
            font-size: 26px;
            font-weight: bold;
            color: #283593;
            margin-top: 20px;
            margin-bottom: 15px;
        }
        .input-label {
            font-size: 18px;
            font-weight: bold;
            color: #37474f;
        }
        .data-dictionary {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
            font-size: 16px;
            color: #424242;
            margin-bottom: 25px;
        }
        .data-dictionary h4 {
            color: #01579b;
            font-size: 20px;
            margin-bottom: 10px;
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
        .prediction-summary {
            background-color: #fff3e0;
            padding: 15px;
            border-radius: 8px;
            font-size: 18px;
            color: #ef6c00;
        }
    </style>
    """, unsafe_allow_html=True)

    # Load the champion model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "../Model_outputs/champion_model.pkl")
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
    
    col1, col2 = st.columns(2)
    with col1:
        budget = st.number_input("Budget ($)", min_value=1_000, max_value=1_000_000_000, step=10_000, value=1_000_000)
        avg_rating = st.number_input("Average Rating", min_value=0.0, max_value=10.0, step=0.1, value=7.0)
    with col2:
        rating_count = st.number_input("Rating Count", min_value=1, step=1, value=500)
        ROI = st.number_input("ROI (Expected Return on Investment)", min_value=0.0, step=0.01, value=1.5)
    
    genres = st.text_input("Genres (comma-separated)", placeholder="Action, Drama, Comedy", value="Action, Drama")

    # Add Data Dictionary
    st.markdown("<div class='sub-header'>Data Dictionary</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='data-dictionary'>
        <h4>Feature Explanations:</h4>
        <ul>
            <li><b>Budget ($):</b> Total production budget of the movie in USD.</li>
            <li><b>Average Rating:</b> The average viewer rating on a scale of 0 to 10.</li>
            <li><b>Rating Count:</b> The total number of ratings received for the movie.</li>
            <li><b>ROI (Return on Investment):</b> Expected return relative to the budget (e.g., 1.5 = 150% return).</li>
            <li><b>Genres:</b> Primary genres of the movie, separated by commas (e.g., Action, Drama).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Predict Revenue", key="predict_button"):
        if pipeline:
            try:
                # Validate input parameters
                if budget <= 0 or avg_rating <= 0 or rating_count <= 0 or ROI < 0:
                    st.markdown("<div class='error-box'>All inputs must be valid positive numbers.</div>", unsafe_allow_html=True)
                else:
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

                    # Display prediction
                    st.markdown(f"<div class='success-box'>Predicted Revenue: ${predicted_revenue:,.2f}</div>", unsafe_allow_html=True)

                    # Generate a dynamic summary
                    st.markdown("<div class='prediction-summary'>", unsafe_allow_html=True)
                    if predicted_revenue < 10_000_000:
                        st.markdown("The predicted revenue is below $10M, suggesting the movie might have limited commercial success. Consider focusing on improving marketability or reducing costs.", unsafe_allow_html=True)
                    elif 10_000_000 <= predicted_revenue <= 100_000_000:
                        st.markdown("The predicted revenue is between $10M and $100M, indicating moderate success. This aligns with a typical mid-budget movie.", unsafe_allow_html=True)
                    else:
                        st.markdown("The predicted revenue exceeds $100M, which suggests a potential blockbuster. Leverage aggressive marketing to maximize returns.", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Visualization 1: Input Feature Breakdown
                    st.markdown("<div class='sub-header'>Feature Breakdown</div>", unsafe_allow_html=True)
                    feature_values = {
                        "Budget ($)": budget,
                        "Average Rating": avg_rating,
                        "Rating Count": rating_count,
                        "ROI": ROI
                    }
                    st.bar_chart(pd.DataFrame(feature_values, index=["Value"]).T)

                    # Visualization 2: Distribution of Predicted Revenue
                    st.markdown("<div class='sub-header'>Predicted Revenue Distribution</div>", unsafe_allow_html=True)
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

     # Footer Section with Enhanced Positioning
    st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 10px;
            width: 100%;
            background: linear-gradient(to right, #f9f9f9, #e0e0e0);
            padding: 10px 20px;
            font-size: 14px;
            color: #555555;
            text-align: center;
            border-top: 1px solid #dddddd;
            box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.1);
        }
        .footer img {
            vertical-align: middle;
            margin-right: 10px;
        }
        .footer a {
            font-weight: bold;
            color: #4CAF50;
            text-decoration: none;
        }
        .footer a:hover {
            color: #388E3C;
        }
    </style>
    <div class='footer'>
        <img src='https://img.icons8.com/ios-filled/50/000000/info.png' width='30' alt='Footer Icon' />
        This dashboard is a final project for the CMSE830 course of the MSDS program at Michigan State University. 
    </div>
    """, unsafe_allow_html=True)