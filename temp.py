# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from linear_regression import Linear_Regression
import numpy as np
import pickle

loaded_model = pickle.load(open('trained_model.sav','rb'))

loaded_model = pickle.load(open("trained_model.sav","rb"))
arr = np.array([0.982, 0.79, 1.27])
y = loaded_model.predict(arr)
print(y)

