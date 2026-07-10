import joblib
import pandas as pd

def test_prediction_pipeline():
    model =  joblib.load("artifacts/attrition_pipeline.pkl")

    input_df = pd.DataFrame([{
                'YearsInCurrentRole'	:	2,
                'YearsWithCurrManager'	:	1,
                'StockOptionLevel'	:	1,
                'JobInvolvement'	:	2,
                'EnvironmentSatisfaction'	:	3,
                'Age'	:	27,
                'JobLevel'	:	1,
                'MonthlyIncome'	:	5000,
                'TotalWorkingYears'	:	3,
                'YearsAtCompany'	:	2,
                'Education'	:	2,
                'JobSatisfaction'	:	4,
                'BusinessTravel'	:	"Travel_Rarely",
                'Department'	:	"Sales",
                'EducationField'	:	"Marketing",
                'Gender'	:	"Male",
                'JobRole'	:	"Sales Executive",
                'MaritalStatus'	:	"Married",   
                'OverTime'	:	"No"
    }])

    probability = model.predict_proba(input_df)

    print("probability->",probability)
    print("shape->",probability.shape)

test_prediction_pipeline()