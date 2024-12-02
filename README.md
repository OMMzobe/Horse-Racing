# Horse Race Predictions

This repository contains a project that analyzes horse racing data from the year 2020. The goal is to uncover key factors that influence race outcomes and use this analysis to predict horse performance and winning probabilities. The project utilizes data cleaning, visualization, and machine learning techniques to provide actionable insights into the dynamics of horse racing.

## Table of Contents
1. [Project Overview](#project-overview)  
   1.1 [Introduction](#introduction)  
   1.2 [Problem Statement](#problem-statement)  
   1.3 [Aim](#aim)  
   1.4 [Objectives](#objectives)  
2. [Importing Packages](#importing-packages)  
3. [Loading Data](#loading-data)  
4. [Data Cleaning](#data-cleaning)  
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
6. [Machine Learning Model](#machine-learning-model)
7. [Interactive Visualizations and Predictions](#interactive-visualizations-and-predictions)

---

## Project Overview

### 1.1 Introduction
Horse racing is a competitive sport where horses, guided by jockeys, race to finish first. This project focuses on analyzing horse race data from the year 2020, using the `races_2020.csv` dataset. The dataset contains information about race distances, track conditions, prizes, and winning times. By conducting an in-depth analysis of these factors, this project provides data-driven insights to optimize race strategies.

### 1.2 Problem Statement
Predicting race outcomes is a complex task due to numerous influencing factors, such as race distance, track conditions, and horse attributes. This project applies a structured and data-driven approach to uncovering the factors that most influence race results, helping trainers and enthusiasts optimize race strategies.

### 1.3 Aim
The primary aim is to analyze horse racing data from 2020 and identify the most important factors that contribute to winning races. By examining attributes like race distance, track conditions, and age groups, the project provides actionable insights for race optimization.

### 1.4 Objectives
- **Data Analysis:** Perform a thorough analysis of horse racing data from 2020.
- **Examine Performance Factors:** Explore the impact of variables such as race distance, track type, and horse age.
- **Impact Assessment:** Assess how different features affect the likelihood of a horse winning a race.
- **Recommendations:** Provide data-driven recommendations to trainers, jockeys, and race organizers.

---

## 2. Importing Packages
This project uses various Python libraries for data manipulation, visualization, and machine learning. Below are some of the key libraries:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
```

---

## 3. Loading Data
The dataset (`races_2020.csv`) is loaded into the workspace using pandas for analysis.

```python
# Load the dataset
race_df = pd.read_csv("races_2020.csv")
print(race_df.head())
```

---

## 4. Data Cleaning
Data cleaning ensures the dataset is suitable for analysis. The steps include:

- **Removing Irrelevant Columns:** Drop unnecessary columns, such as `hurdles`.
- **Handling Missing Values:** Fill missing values in columns like `rclass`, `band`, and `currency`.
- **Renaming Columns:** Rename columns for clarity.
- **Handling Duplicates:** Check and handle duplicate rows.

```python
# Remove the 'hurdles' column
race_df = race_df.drop(columns=['hurdles'])

# Fill missing values in 'rclass' and 'band' columns
race_df['rclass'].fillna('Unknown', inplace=True)
race_df['band'].fillna('Unknown', inplace=True)

# Fill missing values in 'currency' with the most common currency
most_common_currency = race_df['currency'].mode()[0]
race_df['currency'].fillna(most_common_currency, inplace=True)

# Check for duplicates and rename columns
duplicate_values = race_df.duplicated().sum()
new_column_names = ["Course", "Time", "Class", "Band", "Age", "Distance", "Condition", "WinningTime", "Prizes"]
race_df.columns = new_column_names
```

---

## 5. Exploratory Data Analysis (EDA)
EDA helps visualize key trends and patterns in the data. It provides insights into relationships between variables such as:

- **Race Distance vs Winning Time:** Identify how distance impacts winning times.
- **Track Conditions:** Analyze how track conditions affect race outcomes.
- **Prize Distribution:** Examine prize distributions across races.

Example Visualization:

```python
# Visualize race distance vs winning time
sns.scatterplot(data=race_df, x="Distance", y="WinningTime")
plt.title("Race Distance vs Winning Time")
plt.show()
```

---

## 6. Machine Learning Model
A machine learning model, such as **XGBoost**, is used to predict the probability of a horse winning a race based on features like:

- Race distance
- Track conditions
- Age group

### Model Workflow:
1. **Data Preprocessing:** Encode categorical variables and scale numerical features.
2. **Model Training:** Train the model using the cleaned dataset.
3. **Evaluation:** Assess the model's performance using metrics like accuracy and F1-score.

```python
from xgboost import XGBClassifier
model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)
```

---

## 7. Interactive Visualizations and Predictions
The project includes an interactive web app built with **Streamlit**, allowing users to:

1. **Explore Visualizations:**
   - Distribution of track conditions (pie chart).
   - Correlation heatmap of race attributes.
   - Scatter plots of key variables like race distance and winning time.

2. **Make Predictions:**
   - Input race details (e.g., track condition, distance).
   - Predict the probability of a horse winning the race.

### Example Streamlit Features:
- **Pie Chart:** Track condition distribution.
```python
fig = px.pie(race_df, names='Condition', title='Track Condition Distribution')
st.plotly_chart(fig)
```

- **Prediction:**
```python
input_data = pd.DataFrame([{ "Distance": 1500, "Condition": "Good", "Age": 3 }])
prediction = model.predict_proba(input_data)[:, 1][0]
print(f"Probability of Winning: {prediction:.2f}")
```

---

## Conclusion
This project demonstrates how data analysis and machine learning can provide insights and predictions in horse racing. The interactive app enhances user engagement, making it easy for trainers, jockeys, and race organizers to explore data and optimize race strategies.

---

Feel free to explore the repository and contribute to the project!

