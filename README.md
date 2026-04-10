# Fraud Detection Project

A machine learning project for detecting potentially fraudulent credit card transactions using:
- **Random Forest** (supervised fraud classification)
- **Isolation Forest** (unsupervised anomaly detection)
- **Streamlit + Plotly** dashboard for interactive monitoring and predictions

## Project Structure

- `fraud_pipeline.py` — data preprocessing, feature engineering, model training, evaluation, and artifact export
- `app.py` — Streamlit dashboard for analysis, filtering, and single/batch predictions
- `fraud_with_predictions.csv` — transaction dataset with model outputs
- `fraud_enriched.csv` — engineered dataset created by the pipeline
- `metrics.json` — saved model performance metrics and metadata
- `requirements.txt` — Python dependencies
- Generated model files (after pipeline run):
  - `fraud_model.pkl`
  - `iso_model.pkl`
  - `scaler.pkl`
  - `le_location.pkl`

## Features

- Data cleaning and transformation
- Time-based and amount-based feature engineering
- Class-balanced Random Forest training
- Isolation Forest anomaly scoring
- Confusion matrix + precision/recall/F1/ROC-AUC tracking
- Interactive dashboard with:
  - KPI overview and fraud trends
  - Transaction explorer
  - Single transaction fraud prediction
  - Batch CSV prediction with downloadable results

## Requirements

- Python 3.9+ recommended
- Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

### 1) Train pipeline and generate artifacts

From the repository root:

```bash
python fraud_pipeline.py
```

This updates/creates:
- `fraud_enriched.csv`
- `fraud_with_predictions.csv`
- `metrics.json`
- model/scaler/encoder `.pkl` files

### 2) Launch dashboard

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Batch Prediction Input Format

For the dashboard batch tab, uploaded CSV must include:

- `Amount`
- `TransactionType` (`purchase` or `refund`)
- `Location`
- `MerchantID`
- `DayOfWeek` (must be one of: `Mon`, `Tue`, `Wed`, `Thu`, `Fri`, `Sat`, `Sun`)
- `Month` (3-letter abbreviation only, one of: `Jan`, `Feb`, `Mar`, `Apr`, `May`, `Jun`, `Jul`, `Aug`, `Sep`, `Oct`, `Nov`, `Dec`)

## Current Saved Metrics (from `metrics.json`)

- Precision: **0.1843**
- Recall: **0.1346**
- F1 Score: **0.1556**
- ROC-AUC: **0.6664**

## Notes

- The app expects model artifacts and metrics files in the project root.
- If you retrain models, rerun the pipeline before launching the dashboard.
