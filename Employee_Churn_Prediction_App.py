import streamlit as st
import pandas as pd
from joblib import load
import dill

# Load the pretrained model
with open('pipeline.pkl', 'rb') as file:
    model = dill.load(file)

# Load the feature schema
my_feature_dict = load('my_feature_dict.pkl')

# Function to predict churn
def predict_churn(data):
    prediction = model.predict(data)[0]
    return "Yes" if prediction == 1 else "No"

# App Title and Sub-header
st.title('Employee Churn Prediction App')
st.subheader('Created by Ahmed Hassaan Qadri')

# Display categorical features
st.subheader('Categorical Features')
categorical_input_vals = {}
categorical_data = my_feature_dict.get('CATEGORICAL')
for i, col in enumerate(categorical_data.get('Column Name').values()):
    categorical_input_vals[col] = st.selectbox(col, categorical_data.get('Members')[i], key=col)

# Display numerical features
st.subheader('Numerical Features')
numerical_input_vals = {}
numerical_data = my_feature_dict.get('NUMERICAL')
for col in numerical_data.get('Column Name'):
    numerical_input_vals[col] = st.number_input(col, key=col)

# Combine numerical and categorical inputs into a DataFrame
input_data = dict(list(categorical_input_vals.items()) + list(numerical_input_vals.items()))
input_df = pd.DataFrame([input_data])

# Predict churn on button click
if st.button('Predict'):
    prediction = predict_churn(input_df)
    translation_dict = {"Yes": "Expected", "No": "Not Expected"}
    prediction_translate = translation_dict.get(prediction)
    st.write(f'The Prediction is **{prediction}**, Hence employee is **{prediction_translate}** to leave.')