# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 22:28:31 2024

@author: lenovo
"""

from linear_regression import Linear_Regression
import numpy as np
import pickle

loaded_model = pickle.load(open('trained_model.sav','rb'))

import streamlit as st

def happiness_prediction(input_data):
    arr = np.array(input_data)
    prediction = loaded_model.predict(arr)
    return float(prediction)

def main():
    st.title("Happiness Score Prediction Web app")
    
    GDP_per_capita = st.number_input("Enter the GDP per capita of the country: ", format="%.3f")
    Social_support = st.number_input("Enter the Social support of the country: ", format="%.3f")
    Healthy_life_expectancy = st.number_input("Enter the Healthy life expectancy of the country: ", format="%.3f")
    
    score = ''
    
    if st.button("Predict the score"):
        score = happiness_prediction([GDP_per_capita, Social_support, Healthy_life_expectancy])
    st.success(score)
    
if __name__ == '__main__':
    main()
    