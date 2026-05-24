import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="praghaav/tourism-sales-model",
    filename="tourism_sales_prediction_model_v1.joblib"
)
model = joblib.load(model_path)

# Streamlit UI for Insurance Charges Prediction
st.title("Wellness Tourism Package Sales Prediction App")
st.write("""
This application predicts the **Wellness Tourism Package Sales** based on customer lifestyle and details.
Please enter the required information below to get a prediction.
""")

# User inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)

type_of_contact = st.selectbox("TypeofContact", ["Self Enquiry", "Company Invited"])

city_tier = st.selectbox("CityTier", [1, 2, 3])

duration_of_pitch = st.number_input("DurationOfPitch", min_value=0.0, max_value=200.0, value=15.0, step=0.1)

occupation = st.selectbox("Occupation", ["Salaried","Small Business","Large Business","Free Lancer"])

gender = st.selectbox("Gender", ["Male","Female"])

num_of_person_visiting = st.number_input("NumberOfPersonVisiting", min_value=0, max_value=100, value=2, step=1)

num_of_followups = st.number_input("NumberOfFollowups", min_value=0, max_value=100, value=2, step=1)

product_pitched = st.selectbox("ProductPitched", ["Basic","Deluxe","Standard","Super Deluxe","King"])

preferred_property_star = st.number_input("PreferredPropertyStar", min_value=0, max_value=7, value=3, step=1)

marital_status = st.selectbox("MaritalStatus", ["Married","Divorced", "Unmarried", "Single"])

num_of_trips = st.number_input("NumberOfTrips", min_value=0, max_value=100, value=5, step=1)

passport = st.selectbox("Passport", ["Yes", "No"])

pitch_satisfaction_score = st.selectbox("PitchSatisfactionScore", [0, 1, 2, 3, 4, 5])

own_car = st.selectbox("OwnCar", ["Yes", "No"])

num_of_cld_visiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=10, value=1, step=1)

designation = st.selectbox("Designation", ["Executive","Manager","Senior Manager","AVP","VP"])

monthly_income = st.number_input("MonthlyIncome", min_value=0.0, value=20000.0, step=1)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_of_person_visiting,
    "NumberOfFollowups": num_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_of_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": num_of_cld_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Purchase" if prediction == 1 else "Not Purchase"
    st.write(f"Based on the information provided, the customer is likely to {result} the Wellness Tourism Package.")
