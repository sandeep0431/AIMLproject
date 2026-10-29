# Maternal Health Risk Prediction

## Project overview

- Uses the UCI Maternal Health Risk dataset to compare machine learning models that classify records as low, mid, or high risk.
- Includes data validation, feature engineering, model training, evaluation reports, and a Streamlit interface for trying example inputs.
- This is an educational project. It is not validated for medical use and must not guide healthcare decisions.

## What the project does

- Checks the dataset columns, labels, missing values, and duplicate records.
- Adds six features derived from blood pressure, age, and blood sugar.
- Trains and compares nine model variants using a stratified 80/20 holdout split and five-fold cross-validation on the training data.
- Uses an ordinal cost-sensitive prediction option to increase recall for high-risk records.
- Saves the trained model, evaluation scores, classification reports, and charts.

## Folder structure

```text
maternal-health/
|-- app.py                         # Streamlit application
|-- requirements.txt               # Python dependencies
|-- pytest.ini                     # Pytest configuration
|-- pyproject.toml, pyrefly.toml    # Python tool configuration
|-- dataset/
|   \-- Maternal Health Risk Data Set.csv  # Source dataset
|-- data/
|   \-- maternal_health_risk_clean.csv    # Cleaned dataset
|-- models/
|   \-- best_model_bundle.joblib   # Model bundle loaded by the app
|-- reports/
|   |-- metrics_summary.txt        # Holdout scores in a readable table
|   |-- holdout_model_comparison.csv
|   |-- classification_reports.txt # Per-class scores and confusion matrices
|   |-- best_parameters.json       # Parameters selected during training CV
|   |-- data_audit.json, run_summary.txt
|   |-- ablation_results.csv, repeated_cv_summary.csv, per_class_metrics.csv
|   |-- REPORT.md                  # Detailed project report
|   |-- figures/                   # Evaluation charts
|   \-- screenshots/               # App screenshots
|-- scripts/
|   \-- capture_screenshots.py     # Screenshot capture helper
|-- src/
|   |-- data_utils.py              # Dataset loading and validation
|   |-- prepare_data.py            # Data audit and clean export
|   |-- features.py                # Feature engineering
|   |-- train.py                   # Model training and comparison
|   |-- evaluation.py              # Metrics and plots
|   |-- ordinal_cost.py            # Cost-sensitive prediction
|   \-- __init__.py
\-- tests/                         # Unit and smoke tests
```

## Setup and run

- Use Python 3.10 or newer.
- From this folder, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

- Start the app:

```powershell
streamlit run app.py
```

- Prepare the dataset and audit files:

```powershell
python -m src.prepare_data
```

- Train the models, print the full holdout metrics table, and regenerate reports:

```powershell
python -m src.train
```

- Run the tests:

```powershell
python -m pytest
```

Training performs parameter searches and may take a few minutes. It updates the saved model and report files.

## Model comparison

Recorded holdout results use 203 records from the stratified split with random seed 42. Accuracy and macro-F1 summarize overall classification; high-risk recall measures how many high-risk records were identified.

- **Tuned Gradient Boosting** had the highest recorded accuracy: **86.70%**, with macro-F1 **0.8737** and high-risk recall **0.9273**.
- **Ordinal Cost-Sensitive Tuned Random Forest**, the model saved for the app, had accuracy **84.24%**, macro-F1 **0.8463**, and high-risk recall **0.9818**.
- **Tuned Random Forest** without the cost-sensitive layer had accuracy **85.22%**, macro-F1 **0.8593**, and high-risk recall **0.9091**.
- **Baseline Decision Tree** had accuracy **83.74%**, macro-F1 **0.8468**, and high-risk recall **0.9091**.

The cost-sensitive option raises high-risk recall on this holdout set while lowering overall accuracy compared with the highest-accuracy model. Full results are in `reports/metrics_summary.txt` and `reports/holdout_model_comparison.csv`.

## Data and limitations

- Dataset: [UCI Maternal Health Risk](https://archive.ics.uci.edu/dataset/863/maternal+health+risk).
- Results describe this dataset and split; they do not show how the models perform on new patients or other populations.
- The dataset contains duplicate records and records with identical features but different labels. See `reports/data_audit.json` and `reports/REPORT.md` for details.
- This project is for coursework and exploration, not diagnosis or treatment.
