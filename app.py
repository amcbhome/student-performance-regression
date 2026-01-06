import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Student Performance Regression", layout="wide")

st.title("📊 Student Habits & Exam Performance – Regression Analysis")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("student_habits_performance.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Target variable
target = st.selectbox(
    "Select target variable",
    options=["exam_score"],
    index=0
)

# Feature selection
features = st.multiselect(
    "Select predictor variables",
    options=[col for col in df.columns if col not in ["student_id", target]],
    default=[
        "study_hours_per_day",
        "attendance_percentage",
        "sleep_hours",
        "mental_health_rating"
    ]
)

if len(features) == 0:
    st.warning("Please select at least one predictor.")
    st.stop()

# Prepare data
X = df[features]
y = df[target]

# Encode categoricals
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("📈 Model Performance")
col1, col2 = st.columns(2)
col1.metric("R² Score", f"{r2:.3f}")
col2.metric("RMSE", f"{rmse:.2f}")

# Coefficients
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

st.subheader("📊 Regression Coefficients")
st.dataframe(coef_df)

# Actual vs Predicted
st.subheader("Actual vs Predicted Exam Scores")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y.min(), y.max()], [y.min(), y.max()], linestyle="--")
ax.set_xlabel("Actual Score")
ax.set_ylabel("Predicted Score")
st.pyplot(fig)

# Residuals
st.subheader("Residuals Distribution")
fig, ax = plt.subplots()
sns.histplot(y_test - y_pred, bins=20, ax=ax)
ax.set_xlabel("Residual")
st.pyplot(fig)

# Correlation heatmap
st.subheader("Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap="coolwarm", ax=ax)
st.pyplot(fig)
