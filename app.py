import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('my_model.keras')

# Load the encoder and scaler
with open('one_hot_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('labelEncoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit inputs
geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input data for prediction
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Encode the geography column and get column names for one-hot encoding
geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

# Concatenate the geography encoding with the input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Fix column name formatting: Ensure that the input_data columns are named correctly (strip tuple formatting)
input_data.columns = [col[0] if isinstance(col, tuple) else col for col in input_data.columns]

# Print columns to debug the mismatch
print(f"Expected Columns from Scaler: {scaler.feature_names_in_}")
print(f"Input Data Columns: {input_data.columns}")

# Ensure the feature names match the ones used during model training
expected_columns = scaler.feature_names_in_

# Debug: Check if all expected columns are in input_data
missing_columns = [col for col in expected_columns if col not in input_data.columns]
extra_columns = [col for col in input_data.columns if col not in expected_columns]

if missing_columns:
    print(f"Missing Columns: {missing_columns}")
    # If any columns are missing, you may need to add them with a default value (e.g., 0 for one-hot encoded features)
    for col in missing_columns:
        input_data[col] = 0  # Add missing columns with 0 as a placeholder

if extra_columns:
    print(f"Extra Columns: {extra_columns}")
    # If there are extra columns, you can drop them
    input_data = input_data.drop(columns=extra_columns)

# Reorder the input_data columns to match the scaler's expected feature names
input_data = input_data[expected_columns]

# Scaling the input data
input_data_scaled = scaler.transform(input_data)

# Make the churn prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Display the results
st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
