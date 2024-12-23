import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.title("House Price Prediction with Raw and Preprocessed Data")

# Step 1: Load Datasets from URLs
raw_data_path = "https://raw.githubusercontent.com/AbhayKiran/App/refs/heads/main/raw_data.csv"
preprocessed_data_path = "https://raw.githubusercontent.com/AbhayKiran/App/refs/heads/main/preprocessed_data.csv"

st.sidebar.header("Data Selection")
data_option = st.sidebar.radio("Choose Dataset to Load:", ["Raw Data", "Preprocessed Data"])

if data_option == "Raw Data":
    data_url = raw_data_path
    st.sidebar.success("Raw data selected.")
else:
    data_url = preprocessed_data_path
    st.sidebar.success("Preprocessed data selected.")

# Load the dataset
try:
    df = pd.read_csv(data_url)
    st.sidebar.success("Dataset successfully loaded!")
    
    # Display first few rows of the dataset
    st.header(f"Preview of the {data_option}")
    st.write(df.head())
except Exception as e:
    st.sidebar.error(f"Failed to load dataset. Error: {e}")
    st.stop()

# Step 2: Preprocess Data if Raw Data is Selected
if data_option == "Raw Data":
    df_cleaned = df.drop_duplicates()
    if 'NumRooms' in df_cleaned.columns:
        df_cleaned['NumRooms'] = df_cleaned['NumRooms'].fillna(df_cleaned['NumRooms'].mean())

    if 'Price' in df_cleaned.columns:
        # Remove outliers in Price using IQR method
        q1 = df_cleaned['Price'].quantile(0.25)
        q3 = df_cleaned['Price'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df_cleaned = df_cleaned[(df_cleaned['Price'] >= lower_bound) & (df_cleaned['Price'] <= upper_bound)]
else:
    df_cleaned = df  # Use the preprocessed dataset as is

# Step 3: Train-Test Split
if 'Price' not in df_cleaned.columns:
    st.error("The dataset must contain a 'Price' column for the target variable.")
    st.stop()

X = df_cleaned.drop(columns=['Price'])
y = df_cleaned['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Preprocessing Pipeline
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()

numerical_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Step 5: Model Training
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)

# Step 6: Evaluate Model
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

# Step 7: Visualizations
st.header(f"Data Distribution ({data_option})")
fig_data = px.histogram(df_cleaned, x='Price', nbins=20, title=f"{data_option} Distribution")
fig_data.update_layout(yaxis_title="Frequency")
st.plotly_chart(fig_data)

st.header("Model Performance Metrics")
import streamlit as st
import plotly.graph_objects as go

# Sample values for the metrics
mse_raw = 11422163191.58  # Replace with your actual value
mse_train = 11123913535.11  # Replace with your actual value
r2_raw = 0.8458036  # Replace with your actual value
r2_train = 0.847443  # Replace with your actual value

# Create the bar chart
fig_metrics = go.Figure()
fig_metrics.add_trace(go.Bar(
    name='MSE',
    x=['Raw Data', 'Preprocessed Data'],
    y=[mse_raw, mse_train]
))
fig_metrics.add_trace(go.Bar(
    name='R²',
    x=['Raw Data', 'Preprocessed Data'],
    y=[r2_raw, r2_train]
))
fig_metrics.update_layout(
    barmode='group',
    title="Model Performance Comparison",
    xaxis_title="Data Type",
    yaxis_title="Metrics",
    legend_title="Metric Type"
)

# Streamlit app
st.title("Model Metrics Comparison")

# Display the Plotly figure
st.plotly_chart(fig_metrics)


# Additional Visualizations
if 'Area' in df_cleaned.columns:
    st.header("Additional Visualization: Price vs Area")
    fig_area = px.scatter(df_cleaned, x='Area', y='Price', color='Location' if 'Location' in df_cleaned.columns else None, title="Price vs Area")
    st.plotly_chart(fig_area)


