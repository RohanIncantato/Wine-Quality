
#filename = 'D:/projects/PracticePRo/trained_model.sav'
#loaded_model = pickle.load(open(filename, 'rb'))

# Load the trained model
#filename = 'D:/projects/PracticePRo/trained_model.sav'

import streamlit as st
import pandas as pd
import numpy as np
import pickle

loaded_model = pickle.load(open('D:/projects/PracticePRo/Wine_Quality/trained_model.sav', 'rb'))


# Load the trained model
#filename = 'D:/projects/PracticePRo/trained_model.sav'
#loaded_model = pickle.load(open(filename, 'rb'))

# Streamlit App
def main():
    st.title("Wine Quality Prediction App")
    st.write("Enter the wine characteristics to predict its quality.")

    # User input
    fixed_acidity = st.number_input("Fixed Acidity:", min_value=0.0, max_value=15.0)
    volatile_acidity = st.number_input("Volatile Acidity:", min_value=0.0, max_value=2.0)
    citric_acid = st.number_input("Citric Acid:", min_value=0.0, max_value=1.0)
    residual_sugar = st.number_input("Residual Sugar:", min_value=0.0, max_value=30.0)
    chlorides = st.number_input("Chlorides:", min_value=0.0, max_value=1.0)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide:", min_value=0, max_value=100)
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide:", min_value=0, max_value=300)
    density = st.number_input("Density:", min_value=0.0, max_value=2.0)
    pH = st.number_input("pH:", min_value=0.0, max_value=4.0)
    sulphates = st.number_input("Sulphates:", min_value=0.0, max_value=2.0)
    alcohol = st.number_input("Alcohol:", min_value=8.0, max_value=15.0)

    if st.button("Predict"):
        # Reshape the input data for prediction
        input_data = np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                               free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]).reshape(1, -1)

        # Make prediction
        #prediction = loaded_model.predict(input_data)
        prediction = loaded_model.predict(input_data)

        # Display result
        if prediction[0] == 1:
            st.success('The wine is of Good Quality.')
        else:
            st.error('The wine is of Bad Quality.')

# Run the app
if __name__ == "__main__":
    main()
