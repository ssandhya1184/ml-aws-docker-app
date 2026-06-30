# Employee Attrition Prediction using Machine Learning
## Overview

This project predicts whether an employee is likely to leave the organization using a machine learning pipeline built with Scikit-learn.

The application includes:

- Data preprocessing pipeline
- Feature engineering
- SMOTE for handling class imbalance
- Random Forest classifier
- Interactive Streamlit UI
- Docker containerization
- Deployment on AWS EC2
- Container image stored in Amazon ECR

## Architecture

```text
Dataset
     │
     ▼
Preprocessing Pipeline
     │
     ▼
Random Forest Model
     │
     ▼
Pipeline Serialization (Joblib)
     │
     ▼
Streamlit UI
     │
     ▼
Docker
     │
     ▼
Amazon ECR
     │
     ▼
Amazon EC2
```

## Features
- Interactive Streamlit interface
- Automatic preprocessing using Scikit-learn Pipeline
- OneHotEncoding
- Standard Scaling
- Missing value handling
- SMOTE oversampling
- Probability-based prediction threshold (0.30)
- Docker deployment
- AWS EC2 deployment

## Technologies
- Python
- Scikit-learn
- Pandas
- NumPy
- Imbalanced-learn
- Streamlit
- Docker
- AWS EC2
- Amazon ECR
- uv Package Manager

## Model Pipeline
```text
Raw User Input
↓
Column Transformer
↓
StandardScaler
↓
OneHotEncoder
↓
SMOTE
↓
Random Forest
↓
Probability Prediction
↓
Threshold = 0.30
↓
Employee Attrition Prediction
```