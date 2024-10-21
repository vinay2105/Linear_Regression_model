import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
class Linear_Regression():
    def __init__(self,learning_rate,no_of_iterations):
        self.no_of_iterations = no_of_iterations
        self.learning_rate = learning_rate
        

    def fit(self,X,Y):
        self.m, self.n = X.shape
        self.w = np.full(self.n, 0)
        self.b = 0
        self.X = X
        self.Y = Y
        for i in range(self.no_of_iterations):
            self.update_weights()
        

    def update_weights(self):
        Y_prediction = self.predict(self.X)
        dw = -(2 *  (self.X.T).dot(self.Y - Y_prediction))/self.m
        db = -(2 * np.sum(self.Y - Y_prediction))/self.m
        self.w = self.w - self.learning_rate*dw
        self.b = self.b - self.learning_rate*db

    def predict(self, X):
        return X.dot(self.w) + self.b
    pass
data = pd.read_csv("world happiness dataset.csv")
X = data.iloc[:,3:-3].values
Y = data["Score"].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 2)
model = Linear_Regression(learning_rate = 0.00001, no_of_iterations = 100000)
model.fit(X_train,Y_train)
import pickle
filename = "trained_model.sav"
pickle.dump(model, open(filename, 'wb') )