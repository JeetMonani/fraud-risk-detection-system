# Fraud Risk Detection System

An end-to-end machine learning application that detects potentially fraudulent financial transactions using a Random Forest classifier. The project combines a machine learning pipeline, a FastAPI inference service, and an interactive Streamlit dashboard to demonstrate how fraud risk predictions can be integrated into a transaction assessment workflow.

## Features

- **Machine Learning:** Random Forest classifier for transaction fraud assessment.
- **Class Imbalance Handling:** Uses stratified sampling and custom cost-weighting to address imbalanced fraud data.
- **REST API:** FastAPI backend that loads serialized preprocessing components and trained models to generate fraud risk probabilities from JSON requests.
- **Interactive Dashboard:** Streamlit interface for adjusting transaction parameters and inspecting predictions.
- **Risk-Based MFA Triggers:** Demonstrates automated multi-factor authentication (MFA) triggers based on transaction risk.
- **Model Interpretability:** Visualizes model decision paths to help users understand prediction behavior.

## Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn, Random Forest |
| API Backend | FastAPI |
| Dashboard | Streamlit |
| Data Processing | Pandas |
| Model Serving | Serialized model and feature-scaling artifacts |

## Architecture

The application follows an end-to-end inference workflow:

1. **Transaction Input:** Transaction parameters are submitted through the dashboard or API.
2. **Preprocessing:** Input features are processed using the saved preprocessing components.
3. **Fraud Prediction:** The trained Random Forest model evaluates the transaction.
4. **Risk Assessment:** The system returns a fraud risk probability for the submitted transaction.
5. **Dashboard Visualization:** The interface displays predictions and model decision-path information.
6. **MFA Triggering:** The application demonstrates risk-based MFA triggers according to its configured decision logic.
