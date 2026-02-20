import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.exception import CustomException

st.title("Sudent Exam Performance Prediction")
st.write("Fill the details below to predict the math score")

gender = st.selectbox("Gender",["Male", "Female"])
race_ethnicity = st.selectbox("Race / Ethnicity",["group A","group B","group C","group D","group E"])
parental_level_of_education = st.selectbox(
    "Parental Education",
    [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree"
    ]
)

lunch = st.selectbox(
    "Lunch Type",
    ["standard", "free/reduced"]
)

test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"]
)

writing_score = st.number_input("Writing score",min_value=0, max_value=100)
reading_score = st.number_input("Reading score",min_value=0, max_value=100)

if st.button("Predict Math Score"):

     data = CustomData(
        gender=gender,
        race_ethnicity=race_ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=float(reading_score),
        writing_score=float(writing_score)
     )


     pred_df = data.get_data_as_data_frame()
     predict_pipeline = PredictPipeline()
     results = predict_pipeline.predict(pred_df)

     st.success(f"Predicted Math Score: {results[0]}")


    

     
    


    
