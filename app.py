import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('heart_disease_model.pkl')



st.title("Heart Disease Prediction App  ❤️") 
st.markdown("""
This application predicts the risk of heart disease based on specific health parameters.  
Please fill out the fields in the **sidebar** and review the model's predictions and probabilities.  

""")


st.sidebar.header("Input Features")
def user_input_features():
    sex = st.sidebar.selectbox("Sex", options=["Male", "Female"])
    age = st.sidebar.slider('Age', 18, 85, 50)
    cp = st.sidebar.selectbox('Chest Pain Type', options=["Typical angina","Atypical angina","Non-anginal pain","Asymptomatic"])
    trestbps = st.sidebar.slider('Resting Blood Pressure', 94, 200, 120)
    chol = st.sidebar.slider('Serum Cholesterol', 126, 564, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ["Yes", "No"])
    restecg = st.sidebar.selectbox('Resting ECG Results', ["Normal", "Having ST-T wave abnormality", "Probable/Definite LVH (Estes' Criteria)"])
    thalach = st.sidebar.slider('Max Heart Rate Achieved', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', ["Yes", "No"])
    oldpeak = st.sidebar.slider('ST Depression', 0.0, 6.2, 1.0, 0.1)
    slope = st.sidebar.selectbox('Slope of Peak Exercise ST Segment', ["Upsloping", "Flat", "Downsloping"])
    ca = st.sidebar.slider('Number of Major Vessels', 0, 3, 0)
    thal = st.sidebar.selectbox('Thalassemia', ["Normal","Fixed defect","Reversable defect"])
    
    
    sex_numeric = 1 if sex == "Male" else 0
    cp_numeric = {"Typical angina": 0, "Atypical angina": 1, "Non-anginal pain": 2, "Asymptomatic": 3}[cp]
    fbs_numeric = 1 if fbs == "Yes" else 0
    restecg_numeric = {"Normal": 0, "Having ST-T wave abnormality": 1, "Probable/Definite LVH (Estes' Criteria)": 2}[restecg]
    thal_numeric = {"Normal": 1, "Fixed defect": 2, "Reversable defect": 3}[thal]
    exang_numeric = 1 if exang == "Yes" else 0
    slope_numeric = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope]
   

    data = np.array([age,sex_numeric, cp_numeric, trestbps, chol, fbs_numeric, restecg_numeric, thalach, exang_numeric, oldpeak, slope_numeric, ca, thal_numeric])
    return data.reshape(1, -1)


input_data = user_input_features()


prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)




st.subheader("Prediction Probability and Result")

df_predict_proba = pd.DataFrame(prediction_proba)
df_predict_proba.columns = ["Healthy Individual","Heart-Disease Patient"]
df_predict_proba.rename(columns={0:"Healthy Individual",
                                 1:"Heart-Disease Patient"})


st.dataframe(df_predict_proba,
             column_config={
               "Healthy Individual": st.column_config.ProgressColumn(
                 "Healthy Individual",
                 format='%.2f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               "Heart-Disease Patient": st.column_config.ProgressColumn(
                 "Heart-Disease Patient",
                 format='%.2f',
                 width='medium',
                 min_value=0,
                 max_value=1
               )
             }, hide_index=True)


if prediction[0] == 0:
    st.success("The model predicts that the patient **does not have heart disease**.")
else:
    st.error("The model predicts that the patient **has heart disease**.")


st.subheader("⚠️ Important Notice ⚠️ ")
st.warning('This model is for prediction purposes only and is not intended for diagnosis. **Please do not forget to consult a doctor!**')

