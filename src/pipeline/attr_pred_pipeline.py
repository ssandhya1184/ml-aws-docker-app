import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
print("Root Directory is ==>",root_dir)
import warnings
warnings.filterwarnings("ignore")


#Get the file path where the dataset is located docs/HR-Employee-Attrition.csv
file_path = root_dir + '/docs/HR-Employee-Attrition.csv'
print(f"CSV File path is ->{file_path}")

#Creating a dataframe
df = pd.read_csv(file_path)
print(df.head(3))

#Creating a copy of dataframe
df1 = df.copy()
df1.drop("EmployeeNumber",axis=1,inplace=True)
df1.drop("Over18",axis=1,inplace=True)

#Segregating categorical and Numerical Columns
cat_columns = df1.select_dtypes(exclude=np.number).columns.tolist()
cat_cols = cat_columns.copy()
cat_cols.remove("Attrition") #Removing attrition as it is the target column
print(f"Categorical columns: \n {cat_cols}")

num_cols = df1.select_dtypes(include=np.number).columns.tolist()
print(f"Numerical Columns: {num_cols}")

#Remove unwanted columns
df1.drop('EmployeeCount',axis=1,inplace=True)

#One Hot encoding of Target Column - Attrition to Column Attrition_Mapped
yesNoMap = {"Yes":1,"No":0}
df1['AttritionMapped'] = df1['Attrition'].map(yesNoMap)
print(df1.head(3))

num_cols.append('AttritionMapped')

print("Categorical Cols:")
print(cat_cols)
#Getting the correlation matrix
corr_matrix = df1.select_dtypes('number').corr()
print("Correlation of numerical columns with Attrition-->",corr_matrix['AttritionMapped'].sort_values())

num_cols_final =  ['YearsInCurrentRole','YearsWithCurrManager','StockOptionLevel','JobInvolvement',
                'EnvironmentSatisfaction','Age','JobLevel','MonthlyIncome','TotalWorkingYears','YearsAtCompany',
                'Education','JobSatisfaction']

# Numerical pipeline
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
# Categorical pipeline
cat_pipeline = Pipeline(steps=[
    #('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
# Combine both
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_cols_final),
    ('cat', cat_pipeline, cat_cols)
])

logistic_model_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', LogisticRegression())
])

y=df1['Attrition']
df1.drop(['Attrition','AttritionMapped'],axis=1,inplace=True)
#Selecting only main columns
final_cols = num_cols_final + cat_cols
df_final = df1[final_cols]
x=df_final

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)
logistic_model_pipeline.fit(x_train,y_train)
x_preds = logistic_model_pipeline.predict(x_train)

print("********************Logistic Regression - Report*******************")
print(classification_report(y_train, x_preds))

pipeline = ImbPipeline([
    ('preprocessing', preprocessor),  # convert to numeric
    ('smote', SMOTE()),               # now safe
    ('model', RandomForestClassifier(class_weight='balanced',n_estimators=1000))
])

pipeline.fit(x_train,y_train)

y_pred = pipeline.predict(x_test)
feature_names = pipeline.named_steps['preprocessing'].get_feature_names_out()
rf_model = pipeline.named_steps['model']
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print(feature_importance_df.head(20))

print(f"Feature Importances->{importances}")
print("********************Random Forest - Report*******************")


y_probas = pipeline.predict_proba(x_test)[:, 1]
prob_threshold = [0.2,0.3,0.4,0.5]
for i in prob_threshold:
    y_pred_news = (y_probas > i).astype(int)  
    y_pred_news = np.where(y_pred_news == 1, 'Yes', 'No')
    print(f"********************Random Forest - Report for Threshold -{i}*******************")
    print(classification_report(y_test, y_pred_news))

#Choosing 0.3 as the threshold
threshold = 0.3
y_pred_final = (y_probas > threshold).astype(int)

joblib.dump(pipeline, "artifacts/attrition_pipeline.pkl")
print("Pickling completed....")