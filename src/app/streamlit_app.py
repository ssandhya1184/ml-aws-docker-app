import streamlit as st
import joblib
import numpy as np
import pandas as pd 


# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
   return joblib.load("artifacts/attrition_pipeline.pkl")



# -----------------------
# Custom Styling
# -----------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Title Section
# -----------------------
st.title("📊 Employee Attrition Prediction")
st.markdown("### Predict the likelihood of an employee leaving the organization")


st.divider()



st.subheader("👤 Employee Details")

# -----------------------
# Layout (2 columns)
# -----------------------
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 60, 30)
    gender = st.selectbox("Gender", ["Female", "Male"])
    marital_status = st.selectbox("Marital Status",['Divorced','Married','Single'])
with col2:
    department = st.selectbox("Department", ["Sales", "Human Resources", "Research & Development"])
    education = st.selectbox("Education Level", [1,2,3,4,5])
    education_field = st.selectbox("Education Field", ['Human Resources','Life Sciences','Marketing','Medical', 'Other','Technical Degree'])
        
st.divider()

st.subheader("💼 Work Information")
col1, col2 = st.columns(2)
with col1:
    salary = st.number_input("Monthly Income", min_value=1000, max_value=20000, step=500)
    job_role = st.selectbox("Job Role", ['Healthcare Representative','Human Resources','Laboratory Technician','Manager','Manufacturing Director','Research Director','Research Scientist','Sales Executive','Sales Representative'])
    job_level = st.selectbox("Job Level", [1,2,3,4,5])    
with col2:
    years_at_company = st.slider("Years at Company", 0, 18, 5)
    years_in_curr_role = st.slider("Years in current role", 0, 18, 5)
    years_with_curr_mgr = st.slider("Years with Current Manager", 0, 17, 4)
    total_working_years = st.slider("Total Working Years", 0, 40, 5)

st.divider()
st.subheader("🧑🏻‍💻 Other Information")
col1, col2 = st.columns(2)
with col1:
    overtime = st.selectbox("Overtime", ["Yes", "No"])
    business_travel = st.selectbox("Busines Travel",['Non-Travel','Travel_Frequently','Travel_Rarely'])
    stock_option = st.selectbox("Stock Option Level", [1,2,3])    
with col2:
    env_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
    job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    job_involvement = st.slider("Job Involvement (1-4)", 1, 4, 3)

st.divider()

# -----------------------
# Prediction Button
# -----------------------
if st.button("🔍 Predict Attrition"):

 #'YearsInCurrentRole', 'YearsWithCurrManager', 'StockOptionLevel',
   #    'JobInvolvement', 'EnvironmentSatisfaction', 'Age', 'JobLevel',
   #    'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'Education',
     #  'JobSatisfaction', 'BusinessTravel', 'Department', 'EducationField',
     #  'Gender', 'JobRole', 'MaritalStatus', 'OverTime'"""
    
    input_df = pd.DataFrame([{
                'YearsInCurrentRole'	:	years_in_curr_role,
                'YearsWithCurrManager'	:	years_with_curr_mgr,
                'StockOptionLevel'	:	stock_option,
                'JobInvolvement'	:	job_involvement,
                'EnvironmentSatisfaction'	:	env_satisfaction,
                'Age'	:	age,
                'JobLevel'	:	job_level,
                'MonthlyIncome'	:	salary,
                'TotalWorkingYears'	:	total_working_years,
                'YearsAtCompany'	:	years_at_company,
                'Education'	:	education,
                'JobSatisfaction'	:	job_satisfaction,
                'BusinessTravel'	:	business_travel,
                'Department'	:	department,
                'EducationField'	:	education_field,
                'Gender'	:	gender,
                'JobRole'	:	job_role,
                'MaritalStatus'	:	marital_status,   
                'OverTime'	:	overtime
    }])
    print(input_df)
    st.divider()
    threshold = 0.3
    # -----------------------
    # Load Model
    # -----------------------
    model = load_model()
    prob = model.predict_proba(input_df)[0][1]
    prediction = "Yes" if prob > threshold else "No"
    

    

    # -----------------------
    # Result Section
    # -----------------------
    st.subheader("📈 Prediction Result")

    if prediction  == 'Yes':
        st.error("⚠️ High Risk: Employee likely to leave")

        st.markdown("""
        ### 💡 Suggested Actions:
        - Improve employee engagement
        - Review compensation structure
        - Conduct one-on-one feedback sessions
        """)

    else:
        st.success("✅ Low Risk: Employee likely to stay")

        st.markdown("""
        ### 👍 Insights:
        - Employee is stable and engaged
        - Maintain current work conditions
        """)

# -----------------------
# Footer
# -----------------------
st.markdown("---")
