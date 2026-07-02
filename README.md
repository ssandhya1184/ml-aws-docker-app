# Employee Attrition Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Container-blue?logo=docker)
![AWS EC2](https://img.shields.io/badge/AWS-EC2-orange?logo=amazonaws)
![Amazon ECR](https://img.shields.io/badge/AWS-ECR-orange?logo=amazonaws)
![uv](https://img.shields.io/badge/uv-Package_Manager-purple)
![License](https://img.shields.io/badge/License-MIT-green)

![Application](screenshots/home1.png)

An end-to-end ML application built with Scikit-learn, Streamlit, Docker, and AWS.
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

## Steps to run locally
uv sync

uv run streamlit run src/app/streamlit_app.py

## Docker
docker build -t ml-app .

docker run -p 8501:8501 ml-app

## AWS Deployment
```text
Docker Image
↓
Amazon ECR
↓
Amazon EC2
↓
Streamlit Application
```

# Application Screenshots

## Home Page

![Employee Details Section](screenshots/home1.png)

---
![Work Information Section](screenshots/home2.png)

---
![Other Details Section](screenshots/home3.png)

---

## High Risk Prediction

![High Risk](screenshots/prediction_yes.png)

---

## Low Risk Prediction

![Low Risk](screenshots/prediction_no.png)