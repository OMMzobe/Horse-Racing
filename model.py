import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import joblib
from sklearn.preprocessing import LabelEncoder
import ast

# Load the dataset
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error("Dataset not found. Please check the file path.")
        return None

# Load the model
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please check the file path.")
        return None

# Initialize paths
dataset_path = r"C:\Users\masin\OneDrive\Documents\Explore AI\Workplace\Horse Racing\Horse-Racing\races_2020.csv"
model_path = r"C:\Users\masin\OneDrive\Documents\Explore AI\Workplace\Horse Racing\Horse-Racing\xgb_horse_model.pkl"

# Load data and model
race_df = load_data(dataset_path)
model = load_model(model_path)

st.title("Horse Race Predictor")

if race_df is not None:
    st.write("Dataset loaded successfully.")

    # Rename columns if necessary
    race_df.rename(columns={'winningTime': 'Winning_time', 'metric': 'Race_speed'}, inplace=True)

    # Parse 'prizes' column
    def parse_prizes_column(prizes):
        try:
            prize_list = ast.literal_eval(prizes)
            if isinstance(prize_list, list):
                return sum(prize_list)
            return float(prizes)
        except (ValueError, SyntaxError):
            return None

    if 'prizes' in race_df.columns:
        race_df['prizes'] = race_df['prizes'].apply(parse_prizes_column)

    # Tabs for Visualizations and Predictions
    tab2, tab3, tab4, tab5 = st.tabs([
        "Track Condition Analysis",
        "Race Speed vs Winning Time",
        "Track Condition Distribution",
        "Correlation Heatmap"
    ])

    # Tab 1: Track Condition Analysis (Pie Chart)
    with tab2:
        st.subheader("Track Condition Distribution")
        track_condition_counts = race_df['condition'].value_counts().reset_index()
        track_condition_counts.columns = ['condition', 'count']
        fig = px.pie(
            track_condition_counts,
            names='condition',
            values='count',
            title='Distribution of Track Conditions',
            hover_data=['count']
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig)

    # Tab 2: Race Speed vs Winning Time (Scatter Plot)
    with tab3:
        st.subheader("Race Speed vs Winning Time")
        if 'Winning_time' in race_df.columns and 'Race_speed' in race_df.columns:
            # Ensure data is clean
            race_df = race_df.dropna(subset=['Winning_time', 'Race_speed'])

            fig = px.scatter(
                race_df,
                x='Winning_time',
                y='Race_speed',
                title='Race Speed vs Winning Time',
                labels={'Winning_time': 'Winning Time (seconds)', 'Race_speed': 'Race Speed'},
                trendline='ols',
                hover_data=['condition']
            )
            fig.update_layout(xaxis_title='Winning Time (seconds)', yaxis_title='Race Speed')
            st.plotly_chart(fig)
        else:
            st.warning("The dataset does not contain 'Winning_time' and/or 'Race_speed'.")

    # Tab 3: Track Condition Distribution (Horizontal Bar Chart)
    with tab4:
        st.subheader("Track Condition Distribution (Sorted)")
        track_condition_counts = track_condition_counts.sort_values(by='count', ascending=True)
        fig = px.bar(
            track_condition_counts,
            x='count',
            y='condition',
            orientation='h',
            labels={'condition': 'Track Condition', 'count': 'Count'},
            title='Distribution of Track Conditions (Sorted by Count)'
        )
        st.plotly_chart(fig)

    # Tab 4: Correlation Heatmap
    with tab5:
        st.subheader("Correlation Heatmap")
        categorical_cols = ['course', 'rclass', 'condition', 'ages']
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            race_df[col] = le.fit_transform(race_df[col].astype(str))
            label_encoders[col] = le

        numeric_cols = race_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        all_numeric_cols = numeric_cols + categorical_cols
        correlation_matrix = race_df[all_numeric_cols].corr()

        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='Viridis'
        ))
        fig.update_layout(
            title='Correlation Heatmap of Numeric and Encoded Features',
            xaxis_nticks=36
        )
        st.plotly_chart(fig)

    # Sidebar for Prediction Inputs
    st.sidebar.header("Input Race Details for Prediction")
    race_course = st.sidebar.text_input("Course")  # Matches 'course'
    time = st.sidebar.text_input("Time")  # Matches 'time'
    race_class = st.sidebar.text_input("Race Class")  # Matches 'rclass'
    performance_bands = st.sidebar.text_input("Performance Bands")  # Matches 'band'
    age_group = st.sidebar.text_input("Age Group")  # Matches 'ages'
    race_distance = st.sidebar.number_input("Race Distance (miles)", min_value=0.0, step=0.1)  # Matches 'distance'
    track_condition = st.sidebar.text_input("Track Condition")  # Matches 'condition'
    temp_condition = st.sidebar.text_input("Temperature Condition")  # Matches 'ncond'
    class_indicator = st.sidebar.text_input("Class Indicator")  # Matches 'class'

    # Make Predictions
    if st.sidebar.button("Predict"):
        if model is not None:
            input_data = pd.DataFrame([{
                "course": race_course,
                "time": time,
                "rclass": race_class,
                "band": performance_bands,
                "ages": age_group,
                "distance": race_distance,
                "condition": track_condition,
                "ncond": temp_condition,
                "class": class_indicator
            }])

            # Placeholder for preprocessing if needed (e.g., encoding categorical variables)

            # Predict probabilities
            probabilities = model.predict_proba(input_data)
            probability_of_winning = probabilities[:, 1][0]

            st.sidebar.subheader("Prediction Result")
            st.sidebar.write(f"Probability of Winning: **{probability_of_winning:.2f}**")
else:
    st.error("Dataset could not be loaded. Please check the file path.")
