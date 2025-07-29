
import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

st.title('Wine Quality Prediction')
st.write("Enter the wine properties to predict if it's Good or Not Good")

# Input fields
fixed_acidity = st.number_input('Fixed Acidity', 0.0, 20.0, 7.4)
volatile_acidity = st.number_input('Volatile Acidity', 0.0, 2.0, 0.7)
citric_acid = st.number_input('Citric Acid', 0.0, 1.0, 0.0)
residual_sugar = st.number_input('Residual Sugar', 0.0, 20.0, 1.9)
chlorides = st.number_input('Chlorides', 0.0, 0.2, 0.076)
free_sulfur = st.number_input('Free Sulfur Dioxide', 0, 100, 11)
total_sulfur = st.number_input('Total Sulfur Dioxide', 0, 300, 34)
density = st.number_input('Density', 0.0, 2.0, 0.9978)
pH = st.number_input('pH', 0.0, 14.0, 3.51)
sulphates = st.number_input('Sulphates', 0.0, 2.0, 0.56)
alcohol = st.number_input('Alcohol', 0.0, 20.0, 9.4)

if st.button('Predict'):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                             free_sulfur, total_sulfur, density, pH, sulphates, alcohol]])
    
    # Get probability of Good Quality
    prob = model.predict_proba(input_data)[0][1]
    
    if prob >= 0.4:  
        st.success(f'✅ Good Quality Wine (Confidence: {prob:.2f})')
    else:
        st.error(f'❌ Not Good Quality Wine (Confidence: {prob:.2f})')
