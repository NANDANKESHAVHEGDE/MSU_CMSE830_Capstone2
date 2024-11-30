Overview
The Movie Analysis Dashboard is an interactive and user-friendly tool designed to provide insights into the world of movies. From understanding data patterns to predicting revenues, the dashboard combines data science and visualization for an enhanced experience.

Features
Exploratory Data Analysis (EDA):

Understand relationships between budgets, revenues, and ratings.
Analyze trends and identify top-performing genres.
Uncover insights into profit margins and movie performance.
Revenue Prediction:

Predict revenue based on budget, ratings, ROI, and genres.
Leverage machine learning models for accurate predictions.

Model Evaluation:
Evaluate predictions using metrics like Mean Absolute Percentage Error (MAPE).
Visualize residuals, predicted vs. actual revenues, and more.

Data Sources:
TMDB Movies Dataset: Budgets, revenue, genres, and more.
IMDb Extensive Dataset: Ratings, director details, runtime.
MovieLens Dataset: User ratings, tags, and metadata.

Technology Stack:
Streamlit: For creating the interactive dashboard.
Pandas: For data manipulation and preparation.
Scikit-learn & XGBoost: For machine learning models.
Seaborn & Matplotlib: For visualizations.

Data Science Workflow:
Data Collection: Aggregated datasets from reliable sources like TMDB and IMDb.
Data Cleaning: Standardized features, removed outliers, and filled missing values.
Feature Engineering: Created profit margin, ROI, and log-transformed variables.
Model Training: Trained Random Forest and XGBoost regression models.
Evaluation: Measured model performance using MAPE, R², and residual analysis.