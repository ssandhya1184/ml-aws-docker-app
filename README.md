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
                    Employee Data
                           │
                           ▼
                 Data Preprocessing
         (Imputer + Scaler + OneHotEncoder)
                           │
                           ▼
                       SMOTE
                           │
                           ▼
              Random Forest Classifier
                           │
                           ▼
                  attrition_pipeline.pkl
                           │
                           ▼
                    Streamlit UI
                           │
                           ▼
                  Docker Container
                           │
                           ▼
                     Amazon ECR
                           │
                           ▼
                     Amazon EC2
                           │
                           ▼
                      Web Browser
```