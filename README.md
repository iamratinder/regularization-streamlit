# Coefficient Playground: Ridge, Lasso & ElasticNet

This Streamlit app allows you to explore and visualize how different regularization methods (Ridge, Lasso, ElasticNet) affect model coefficients and performance on various regression datasets.

## Features

- Choose between California Housing, Diabetes, or a synthetic regression dataset.
- Adjust regularization strength (`alpha`) and L1 ratio (`l1_ratio` for ElasticNet).
- Visualize and compare model coefficients.
- See model performance metrics (R² Score, Mean Squared Error).
- Visualize predictions for a selected feature.

## Setup

1. **Clone the repository** (or download the code):

   ```
   git clone <repo-url>
   cd regularization_Streamlit
   ```

2. **Install dependencies** (preferably in a virtual environment):

   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:

```
streamlit run app.py
```

Then open the provided local URL in your browser.

## Requirements

- Python 3.7+
- See `requirements.txt` for Python package dependencies.

## License

MIT License.

---

Made with ❤️ by OpenLearn.
