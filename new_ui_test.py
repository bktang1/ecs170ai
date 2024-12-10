import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

# Set page config
st.set_page_config(
    page_title="Stroke Risk Assessment",
    page_icon="🩺",
    layout="centered"
)

# Load the trained model and preprocessing tools
@st.cache_resource
def load_model_and_preprocessors():
    model = joblib.load('model.pkl')  # Load your trained model
    scaler = joblib.load('scaler.pkl')  # Load the scaler (if saved separately)
    preprocessor = joblib.load('preprocessor.pkl')  # Load the preprocessor (if saved separately)
    return model, scaler, preprocessor

model, scaler, preprocessor = load_model_and_preprocessors()

# Inject Custom CSS
def inject_custom_css():
    st.markdown(
        """
        <style>
        body {
            background-color: #f9f9f9;
            font-family: 'Arial', sans-serif;
        }

        h1, h2, h3 {
            color: #4CAF50;
        }

        .stButton {
        display: flex;
        justify-content: center;
        align-items: center;
        }

        .stButton > button {
            font-size: 18px;
            background-color: #007BFF;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
        }

        .stButton > button:hover {
            background-color: #0056b3;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Welcome page
def welcome_page():
    #st.title(":blue[Stroke Risk Assessment and Personalized Health Plan]")
    st.markdown('<h1 style="text-align: center; font-family:Bubblegum Sans; font-size:30px; color:#1fe0c3;">Stroke Risk Assessment and Personalized Health Plan</h1>', unsafe_allow_html=True)
    #st.write("Understand your stroke risk and get personalized health plans.")
    st.markdown(
    '<p style="text-align: center; font-family: Arial; font-size: 18px;">Understand your stroke risk and get personalized health plans.</p>',
    unsafe_allow_html=True
    )
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Assessment"):
            st.session_state.page = "input_form"
    
    with col2:
        if st.button("Information"):
            st.session_state.page = "info"

# Input Form Page
def input_form_page():
    st.title("Health Data Input Form")
    st.write("Please fill out the form below with your health details.")
    
    # Collect user inputs
    gender_option = st.radio("Gender", ["Male", "Female"])
    gender = 0 if gender_option == "Male" else 1

    age = st.number_input("Age", min_value=0, max_value=120, value=20, step=1)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    avg_glucose = st.number_input("Average Glucose Level (mg/dL)", min_value=0.0, max_value=500.0, value=100.0, step=0.1)

    hypertension_option = st.selectbox("History of Hypertension (High Blood Pressure)", ["No", "Yes"])
    hypertension = 0 if hypertension_option == "No" else 1

    heart_disease_option = st.selectbox("History of Heart Disease", ["No", "Yes"])
    heart_disease = 0 if heart_disease_option == "No" else 1
    
    residence_type_option = st.selectbox("Residence Type", ["Urban", "Rural"])
    Residence_type = 0 if residence_type_option == "Urban" else 1
    
    work_type_option = st.selectbox("Work Type", ["Child", "Never worked", "Self-Employed", "Private", "Government employed"])
    work_type = {
        "Child": 0,
        "Never worked": 1,
        "Self-Employed": 2,
        "Private": 3,
        "Government employed": 4
    }[work_type_option]

    smoking_status_option = st.selectbox("Smoking Status", ["Never smoked", "Formerly smoked", "Smokes", "Unknown"])
    smoking_status = {
        "Never smoked": 0,
        "Formerly smoked": 1,
        "Smokes": 2,
        "Unknown": 3
    }[smoking_status_option]

    # Save inputs
    user_data = pd.DataFrame({
        'age': [age],
        'avg_glucose_level': [avg_glucose],
        'bmi': [bmi],
        'gender': [gender],
        'work_type': [work_type],
        'Residence_type': [Residence_type],
        'smoking_status': [smoking_status],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease]
    })

    if st.button("Submit"):
        st.session_state.user_data = user_data
        st.session_state.page = "risk_analysis"

# Risk Analysis Page
def risk_analysis_page():
    st.title("Stroke Risk Analysis")
    user_data = st.session_state.user_data
    
    # Preprocess user input
    user_data = user_data.reindex(columns=['age', 'avg_glucose_level', 'bmi', 'gender', 'work_type', 'Residence_type', 'smoking_status', 'hypertension', 'heart_disease'])
    user_data = user_data.fillna(0)  # Fill NaN values with 0 or other suitable
    required_columns = ['age', 'avg_glucose_level', 'bmi', 'gender', 'work_type', 'Residence_type', 'smoking_status', 'hypertension', 'heart_disease']
    user_data = user_data[required_columns]  # Remove any extraneous columns

    # Preprocess user input
    try:
    # Validate and correct types
        user_data = user_data.astype({
            'age': 'int',
            'avg_glucose_level': 'float',
            'bmi': 'float',
            'gender': 'int',
            'work_type': 'int',
            'Residence_type': 'int',
            'smoking_status': 'int',
            'hypertension': 'int',
            'heart_disease': 'int'
        })
        # Transform data
        user_transformed = preprocessor.transform(user_data)

    except Exception as e:
        st.error(f"Error during transformation: {e}")

    user_transformed_df = pd.DataFrame(user_transformed.toarray() if hasattr(user_transformed, 'toarray') else user_transformed)
    
    # Predict stroke risk
    prediction_proba = model.predict_proba(user_transformed_df)[:, 1]
    prediction = (prediction_proba >= 0.3).astype(int)

    st.write("### Prediction:")
    if prediction[0] == 1:
        st.error(f"High Stroke Risk: {prediction_proba[0] * 100:.2f}%")
    else:
        st.success(f"Low Stroke Risk: {prediction_proba[0] * 100:.2f}%")
    
    if st.button("View Recommendations"):
        st.session_state.page = "recommendations"

# Recommendation Page
def recommendation_page():
    prediction_proba = st.session_state.get('prediction_proba', None)
    prediction = st.session_state.get('prediction', None)
    st.title("Personalized Health Plan")
    user_data = st.session_state.user_data
    if prediction == 1:
        st.write("Based on your response, here are some recommendations to lower your stroke risk.")
        st.subheader("Lifestyle Recommendations")
        if user_data['bmi'][0] > 25:
            st.write("Maintain a healthy weight by exercising regularly.")
        else:
            st.write("Great job maintaining a healthy weight! Keep it up with regular exercise and a balanced lifestyle.")
        if user_data['avg_glucose_level'][0] > 140:
            st.write("Monitor glucose levels and reduce sugar intake.")
        else:
            st.write("Your glucose levels are within a healthy range. Continue following a nutritious diet.")
        st.subheader("Dietary Guidance")
        st.write("Adopt a balanced diet with fruits, vegetables, and whole grains.")
        st.subheader("Medical Advice")
        st.write("Consult with a healthcare provider for regular check-ups.")
    else:
        st.write("Based on your response, here are some recommendations to lower your stroke risk.")
        st.subheader("Lifestyle Recommendations")
        if user_data['bmi'][0] > 25:
            st.write("Maintain a healthy weight by exercising regularly.")
        else:
            st.write("Great job maintaining a healthy weight! Keep it up with regular exercise and a balanced lifestyle.")
        if user_data['avg_glucose_level'][0] > 140:
            st.write("Monitor glucose levels and reduce sugar intake.")
        else:
            st.write("Your glucose levels are within a healthy range. Continue following a nutritious diet.")
        st.subheader("Dietary Guidance")
        st.write("Adopt a balanced diet with fruits, vegetables, and whole grains.")
        st.subheader("Medical Advice")
        st.write("Consult with a healthcare provider for regular check-ups.")

def info_page():
    st.markdown(
        """
        <h1 style="text-align: center; color: #4CAF50;">Project Overview</h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## Vitaly: An AI-driven Tool for Stroke Risk Assessment")
    st.write(
        "Vitaly is an AI-driven tool designed to assess stroke risk and provide personalized health recommendations. "
        "By combining advanced machine learning techniques with an intuitive user interface, it empowers users to proactively manage their health. "
        "The system predicts stroke risk based on key metrics like BMI, glucose levels, and hypertension, aiming to minimize false negatives and promote early intervention."
    )

    st.markdown("## Key AI Methodologies and Techniques")
    st.write(
        "- **Data Preprocessing**: Addressed missing values, performed feature selection, and engineered categorical variables into numerical formats.\n"
        "- **Model Development**:\n"
        "  - Transitioned from Decision Trees to Random Forests for better accuracy and handling of non-linear interactions.\n"
        "  - Addressed class imbalance using SMOTE and class weighting, enhancing the model's ability to predict rare stroke cases.\n"
        "  - Conducted hyperparameter tuning to optimize recall and maintain precision.\n"
        "- **User Interface**: Developed an interactive Streamlit app featuring intuitive input forms, risk visualizations, and personalized recommendations."
    )

    st.markdown("## Challenges Faced and Resolutions")
    st.write(
        "- **Class Imbalance**: Mitigated through SMOTE and class weighting, ensuring minority classes were adequately represented.\n"
        "- **Missing and Irrelevant Data**: Imputed missing BMI values and removed features like smoking status that lacked significant predictive power.\n"
        "- **Balancing Recall and Precision**: Prioritized recall to avoid false negatives, aligning with the project’s ethical focus on minimizing risk.\n"
        "- **User Engagement**: Enhanced UI interactivity with animations and a clear, color-coded risk visualization system.\n"
        "- **Model and UI compatibility**: The model was expecting more inputs than the UI was providing. We fixed this by changing how the preprocessor was working and removing the one-hot encoding we had initially put in the model."
    )


    col1, col2 = st.columns(2)
    # Back Button
    with col1:
        if st.button("Back"):
            st.session_state.page = "welcome"
    # URL Button
    with col2:
        st.markdown(
            """
            <a href="https://docs.google.com/document/d/1orC6raJHEq1lT5pdLkbE8osIPRaae67lzqd7nSONVMw/edit?tab=t.0" target="_blank">
                <button style="background-color:#007BFF; color:white; border:none; padding:10px 20px; border-radius:5px; font-size:16px; cursor:pointer;">
                    Additional Information
                </button>
            </a>
            """,
            unsafe_allow_html=True
        )



# Main Application Logic
def main():
    inject_custom_css()
    if "page" not in st.session_state:
        st.session_state.page = "welcome"
    if "user_data" not in st.session_state:
        st.session_state.user_data = None
    
    if st.session_state.page == "welcome":
        welcome_page()
    elif st.session_state.page == "info":
        info_page()
    elif st.session_state.page == "input_form":
        input_form_page()
    elif st.session_state.page == "risk_analysis":
        risk_analysis_page()
    elif st.session_state.page == "recommendations":
        recommendation_page()

if __name__ == "__main__":
    main()