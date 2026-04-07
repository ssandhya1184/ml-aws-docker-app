from sklearn.linear_model import LinearRegression
import joblib
import numpy as np
import pandas as pd 
import os

def train():
    X = np.array([[1],[2],[3]])
    y = np.array([2,4,6])

    model = LinearRegression()
    model.fit(X,y)

    joblib.dump(model,"artifacts/model.pkl")
    print("Pkl done..")

if __name__ == "__main__":
    train()