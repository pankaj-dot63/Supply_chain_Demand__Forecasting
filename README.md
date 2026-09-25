# 📦 Supply Chain Demand Forecasting Using Machine Learning

A Machine Learning-based **Supply Chain Demand Forecasting System** that predicts future product demand using historical sales, pricing, seasonal, economic, weather, and promotional data.

The project uses multiple regression algorithms and provides a **Streamlit web application** where users can enter business-related information and receive an instant demand prediction.

---

## 🚀 Project Overview

Accurate demand forecasting is important for supply-chain businesses because demand can change due to factors such as:

* Sales history
* Promotions
* Discounts
* Holidays
* Product prices
* Competitor pricing
* Weather conditions
* Economic conditions
* Seasonal patterns

Incorrect demand predictions can lead to:

* 📦 Overstocking
* 💰 Higher inventory and storage costs
* ❌ Stockouts
* 📉 Lost sales opportunities
* ⚠️ Poor inventory planning

This project builds an end-to-end Machine Learning pipeline to analyze historical data and predict **future product demand**.

---

## 🎯 Problem Statement

The objective of this project is to develop a data-driven system that can learn demand patterns from historical supply-chain data and predict future demand.

The problem is treated as a:

> **Supervised Machine Learning Regression Problem**

because the target variable `future_demand` is a continuous numerical value.

---

## 🛠️ Technologies Used

| Technology       | Purpose                 |
| ---------------- | ----------------------- |
| Python           | Programming Language    |
| Pandas           | Data Processing         |
| NumPy            | Numerical Computation   |
| Matplotlib       | Data Visualization      |
| Scikit-learn     | Machine Learning        |
| XGBoost          | Gradient Boosting       |
| Joblib           | Model Saving            |
| Streamlit        | Web Application         |
| Jupyter Notebook | Development / Analysis  |
| VS Code          | Development Environment |
| CSV              | Dataset                 |

The project uses Linear Regression, Decision Tree, Random Forest and XGBoost models.

---

## 📊 Dataset

The dataset contains approximately **5,000 historical records**.

### Target Variable

```text
future_demand
```

### Input Features

```text
sales_units
holiday_season
promotion_applied
competitor_price_index
economic_index
weather_impact
price
discount_percentage
region
store_type
category
date
```

`product_id` is treated as an identifier rather than a normal continuous numerical feature.

---

## 🔄 Project Workflow

The complete project follows this pipeline:

```text
CSV Dataset
     ↓
Data Cleaning
     ↓
Date Processing
     ↓
Feature Engineering
     ↓
Feature Selection
     ↓
Chronological Train-Test Split
     ↓
Model Training
     ↓
Hyperparameter Tuning
     ↓
Model Evaluation
     ↓
Model Saving
     ↓
Streamlit Application
     ↓
Demand Prediction
```

---

## 🧹 Data Preprocessing

The preprocessing pipeline includes:

1. Loading the CSV dataset using Pandas
2. Converting the `date` column into datetime format
3. Checking and preparing correct data types
4. Converting Boolean variables into `0/1` where required
5. Using already-encoded categorical variables
6. Selecting useful features
7. Separating features `X` and target `y`

### Feature Scaling

Tree-based models in this project do not require `StandardScaler`.

Decision Tree, Random Forest and XGBoost make predictions using threshold-based splits rather than distance-based calculations.

---

## 🕒 Feature Engineering

The raw `date` feature is transformed into useful time-based features.

For example:

```text
2026-08-15
```

is converted into:

```text
year
month
day
day_of_week
week_of_year
```

These features help the model identify:

* Monthly demand patterns
* Weekly demand patterns
* Weekday effects
* Seasonal behavior
* Time-related demand trends

---

## 📈 Exploratory Data Analysis

EDA is performed before model training to understand the dataset and demand patterns.

The analysis includes:

### Monthly Average Demand

Analyzes how average demand changes from month to month.

### Feature Relationships

Examines relationships between input variables and demand.

### Feature Importance

Helps identify which features contribute to model predictions.

### Demand Trends

Visualizes historical demand trends and seasonality.

Matplotlib is used for visualization.

---

# 🤖 Machine Learning Models

Four regression models are considered in the project.

## 1. Linear Regression

Linear Regression is used as a baseline model.

It attempts to model a linear relationship between input features and future demand.

### Advantages

* Simple
* Fast
* Easy to interpret
* Provides a baseline for comparison

---

## 2. Decision Tree Regressor

Decision Tree predicts demand through a sequence of decision rules.

Example:

```text
Price > X?
      ↓
Promotion = Yes?
      ↓
Holiday = Yes?
      ↓
Predict Demand
```

Important hyperparameters include:

```text
max_depth
min_samples_split
min_samples_leaf
```

---

## 3. Random Forest Regressor

Random Forest combines multiple Decision Trees and aggregates their predictions.

It can:

* Reduce dependence on a single tree
* Capture nonlinear relationships
* Handle complex feature interactions

Important parameters include:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
```

---

## 4. XGBoost Regressor

XGBoost stands for **Extreme Gradient Boosting**.

It builds trees sequentially, where each new tree attempts to correct errors made by previous trees.

Important parameters:

```text
n_estimators
learning_rate
max_depth
```

XGBoost was included to compare a powerful boosting algorithm against simpler regression and tree-based models.

---

# 🧪 Train-Test Split

Because this is a forecasting problem, the dataset is split **chronologically** instead of randomly.

```text
80% → Training Data
20% → Testing Data
```

The earlier records are used for training and later records are used for testing.

This prevents future information from influencing the training process.

---

# ⚙️ Hyperparameter Tuning

The project uses:

### RandomizedSearchCV

RandomizedSearchCV tests different combinations of hyperparameters and uses cross-validation to identify strong configurations.

### TimeSeriesSplit

`TimeSeriesSplit` maintains the chronological structure of the data during cross-validation.

This prevents validation from using data that would be considered "future" relative to the training data.

---

# 📏 Model Evaluation

The models are evaluated using:

## MAE — Mean Absolute Error

Measures the average absolute prediction error.

```text
Lower MAE = Better
```

## RMSE — Root Mean Squared Error

Penalizes larger prediction errors more strongly.

```text
Lower RMSE = Better
```

## R² Score

Measures how much of the variation in demand is explained by the model.

```text
Higher R² = Better
```

An R² value can also be negative when a model performs worse than a simple average-based baseline.

---

# 📊 Current Model Results

The currently recorded notebook results are:

| Model             |   MAE |  RMSE |     R² |
| ----------------- | ----: | ----: | -----: |
| Linear Regression | 46.87 | 54.72 | -0.021 |
| Random Forest     | 47.10 | 54.90 | -0.028 |
| XGBoost           | 49.18 | 57.84 | -0.141 |

These are **test-set regression metrics, not accuracy percentages**.

The current results show negative R² values for all three recorded models, indicating that the models currently explain the test-set variation poorly and that further feature engineering and model improvement would be useful.

> Note: The PPT records results for Linear Regression, Random Forest and XGBoost. A numerical result for Decision Tree is not provided in the current results slide.

---

# 💾 Model Saving

After training, the model is saved using **Joblib**.

```text
demand_forecasting_model.pkl
```

The feature list used during training is also saved.

This allows the Streamlit application to load the trained model without retraining it every time the application starts.

---

# 🌐 Streamlit Application

The project includes a web-based Streamlit application.

Users can enter:

```text
Sales Units
Holiday Season
Promotion
Competitor Price Index
Economic Index
Weather Impact
Price
Discount %
Region
Store Type
Category
Forecast Date
```

After entering the required values, the user can click:

```text
Predict Demand
```

The application loads the saved ML model and generates the predicted future demand.

---

# 📁 Suggested Project Structure

```text
Supply-Chain-Demand-Forecasting/
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   └── demand_forecasting.ipynb
│
├── models/
│   ├── demand_forecasting_model.pkl
│   └── feature_list.pkl
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

---

# ▶️ How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/Supply-Chain-Demand-Forecasting.git
```

## 2. Navigate to the Project

```bash
cd Supply-Chain-Demand-Forecasting
```

## 3. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Run Streamlit

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

---

# 📦 Requirements

Example `requirements.txt`:

```text
pandas
numpy
matplotlib
scikit-learn
xgboost
streamlit
joblib
```

---

# 🔮 Future Scope

The project can be further improved by adding:

* More historical data
* Lag features
* Rolling-average features
* Advanced time-series forecasting models
* Improved feature selection
* Further hyperparameter optimization
* Real-time data integration
* Cloud deployment
* Inventory optimization
* Automated reorder recommendations

These improvements are also identified as future scope in the project presentation.

---

# 💡 Key Learning Outcomes

Through this project, the following concepts are demonstrated:

* Data preprocessing with Pandas
* Exploratory Data Analysis
* Time-based feature engineering
* Supervised Machine Learning
* Regression algorithms
* Decision Trees
* Random Forest
* XGBoost
* Hyperparameter tuning
* Time-series-aware validation
* MAE, RMSE and R² evaluation
* Model persistence using Joblib
* Streamlit application development
* End-to-end ML pipeline development

---

# 👨‍💻 Project Summary

**Supply Chain Demand Forecasting** demonstrates an end-to-end Machine Learning workflow:

```text
Raw Data
   ↓
Preprocessing
   ↓
EDA
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Hyperparameter Tuning
   ↓
Evaluation
   ↓
Model Saving
   ↓
Streamlit Deployment
   ↓
Demand Prediction
```

The system transforms historical supply-chain data into a machine-learning-based demand prediction application.

---

## ⭐ Project Status

**Status:** Completed — Initial ML Prototype

**Application:** Streamlit

**ML Task:** Supervised Regression

**Target:** `future_demand`

**Models:** Linear Regression, Decision Tree, Random Forest, XGBoost

**Deployment:** Streamlit

---

## 📌 Note

The model performance reported in this README reflects the results currently recorded in the project notebook/presentation. The negative R² values indicate that the current prototype still has significant room for improvement through additional data, feature engineering, and model development.
