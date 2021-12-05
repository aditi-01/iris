import joblib
import matplotlib as plt
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression

# Captures the path of current folder
curr_path = os.path.dirname(os.path.realpath(__file__))


feat_cols = ['SepalLengthCm', 'SepalWidthCm','PetalLengthCm','PetalWidthCm']

model_final = joblib.load(curr_path +"/model.joblib")


print(model_final)
def predict_class(attributes: np.ndarray):
    """ Returns Class value"""

    pred = model_final.predict(attributes)
    print("Class predicted")

    return int(pred[0])