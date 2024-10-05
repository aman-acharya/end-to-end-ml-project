import streamlit as st
import pandas as pd
import sys
from src.pipelines.predict_pipeline import PredictionPipeline, CustomData
from src.exception import CustomException

# set the page title
st.markdown("<h1 style='text-align: center;'>Student Performance Prediction</h1>", unsafe_allow_html=True)

try:
    with st.container():
        _, col1, _ = st.columns([0.5, 7, 0.5])
        
        with col1:
            gender = st.selectbox(
                label='**Gender** :',
                options=['', 'Male', 'Female']
            )

            race_ethnicity = st.selectbox(
                label='**Race/Ethnicity** :',
                options=['', 'group A', 'group B', 'group C', 'group D', 'group E']
            )

            parental_level_of_education = st.selectbox(
                label='**Parental Level of Education** :',
                options=['', 'Some High School', 'High School', 'Some College', "Associate's Degree", "Bachelor's Degree", "Master's Degree"]
            )

            lunch = st.selectbox(
                label='**Lunch** :',
                options=['', 'Standard', 'Free/Reduced']
            )

            test_preparation_course = st.selectbox(
                label='**Test Preparation Course** :',
                options=['', 'None', 'Completed']
            )

            math_score = st.number_input(
                label='**Math Score** :',
                min_value=0, max_value=100
            )

            reading_score = st.number_input(
                label='**Reading Score** :',
                min_value=0, max_value=100
            )

            if 'predict_output' not in st.session_state:
                st.session_state['predict_output'] = False

            if st.button('Predict', type='primary'):
                st.session_state['predict_output'] = not st.session_state['predict_output']

            if st.session_state['predict_output']:
                if all([gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course]):
                    data = CustomData(
                        gender=gender,
                        race_ethnicity=race_ethnicity,
                        parental_level_of_education=parental_level_of_education,
                        lunch=lunch,
                        test_preparation_course=test_preparation_course,
                        math_score=math_score,
                        reading_score=reading_score
                    )

                    data = data.get_data_as_dataframe()
                    pipeline = PredictionPipeline()
                    prediction = pipeline.predict(data)

                    st.write(f'**Predicted Score** : {prediction[0]}')
                else:
                    st.info(':warning: Please fill all the fields to predict the score.')


except Exception as e:
    raise CustomException(e, sys)
