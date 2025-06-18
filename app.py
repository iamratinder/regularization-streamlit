import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Tame Your Coefficients!", layout="wide")

st.title("Coefficient Playground: Ridge, Lasso & ElasticNet")
st.markdown("Choose a dataset and see how different regularization methods affect model coefficients!")

# Sidebar
st.sidebar.header("🔧 Model Settings")

# Dataset Selector
dataset_name = st.sidebar.selectbox("📁 Choose Dataset", ["California Housing", "Diabetes", "Synthetic Regression"])

@st.cache_data
def load_data(name):
    if name == "California Housing":
        data = fetch_california_housing()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target
    elif name == "Diabetes":
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target
    elif name == "Synthetic Regression":
        X, y = make_regression(n_samples=100, n_features=10, noise=10.0, random_state=42)
        X = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
    return X, y, X.columns

X, y, feature_names = load_data(dataset_name)

# Add function definition
def plot_coefficients(coefs, names, feature_names):
    fig, ax = plt.subplots(figsize=(10, 4))  # reduced from (12, 6)
    for coef, name in zip(coefs, names):
        ax.plot(feature_names, coef, marker='o', label=name)
    ax.set_title("Coefficient Comparison")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45)
    ax.set_ylabel("Coefficient Value")
    ax.grid(True)
    ax.legend()
    return fig

# Regularization Controls
alpha = st.sidebar.slider("Alpha (Regularization strength)", 0.001, 100.0, 1.0, step=0.1)
l1_ratio = st.sidebar.slider("L1 Ratio (ElasticNet only)", 0.0, 1.0, 0.5, step=0.05)

# Add feature selection for visualization
if X.shape[1] > 1:
    feature_to_plot = st.sidebar.selectbox("📊 Select feature to visualize", feature_names)
    feature_idx = list(feature_names).index(feature_to_plot)
else:
    feature_idx = 0

# Fit models including Linear Regression
linear = make_pipeline(StandardScaler(), LinearRegression())
ridge = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
lasso = make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=10000))
elastic = make_pipeline(StandardScaler(), ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000))

models = [linear.fit(X, y), ridge.fit(X, y), lasso.fit(X, y), elastic.fit(X, y)]
model_names = ["Linear", "Ridge", "Lasso", "ElasticNet"]
coefs = [model.named_steps[name.lower()].coef_ for name, model in zip(model_names[1:], models[1:])]


# Original coefficient plot
col1, col2 = st.columns([5, 1])
with col1:
    st.subheader("🔍 Coefficient Comparison")
    fig = plot_coefficients(coefs, model_names[1:], feature_names)
    st.pyplot(fig)

st.markdown("---")

# Model Performance Metrics
st.subheader("📈 Model Performance")
col1, col2 = st.columns(2)
with col1:
    st.write("**R² Score:**")
    for model, name in zip(models, model_names):
        r2 = r2_score(y, model.predict(X))
        st.write(f"{name}: {r2:.3f}")

with col2:
    st.write("**Mean Squared Error:**")
    for model, name in zip(models, model_names):
        mse = mean_squared_error(y, model.predict(X))
        st.write(f"{name}: {mse:.3f}")
        
st.markdown("---")

col1, col2 = st.columns([1.7, 1])
with col1:
    # Data Visualization
    st.subheader("📊 Data Visualization")
    fig_data, ax_data = plt.subplots(figsize=(8, 5))  # reduced from (10, 6)
    ax_data.scatter(X.iloc[:, feature_idx], y, alpha=0.5, label='Data points')

    # Plot predictions for each model
    X_plot = np.linspace(X.iloc[:, feature_idx].min(), X.iloc[:, feature_idx].max(), 100).reshape(-1, 1)
    X_plot_full = np.zeros((100, X.shape[1]))
    X_plot_full[:, feature_idx] = X_plot.ravel()

    for model, name in zip(models, model_names):
        y_pred = model.predict(X_plot_full)
        ax_data.plot(X_plot, y_pred, label=f'{name} fit', linewidth=2)

    ax_data.set_xlabel(feature_names[feature_idx])
    ax_data.set_ylabel("Target")
    ax_data.legend()
    ax_data.grid(True)
    st.pyplot(fig_data)

# Footer
st.markdown("---")
st.caption("Made with ❤️ OpenLearn")
