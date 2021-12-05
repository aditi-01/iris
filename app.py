import streamlit as st
from model import predict_class
import numpy as np

st.set_page_config(page_title="Iris Flower Classification App",
                   page_icon="🌸", layout="wide")


with st.form("prediction_form"):

    st.header("Enter the Deciding Factors:")

    SepalLengthCm = st.number_input("Sepal Length: ")
    SepalWidthCm = st.number_input("Sepal Width: ")
    PetalLengthCm = st.number_input("Petal Length: ")
    PetalWidthCm = st.number_input("Petal Width: ")

    submit_val = st.form_submit_button("Predict Duration")

if submit_val:
    # If submit is pressed == True
    attribute = np.array([SepalLengthCm, SepalWidthCm, PetalLengthCm,
                        PetalWidthCm]).reshape(1,-1)


    if attribute.shape == (1,4):
        print("attrubutes valid")
        

        value = predict_class(attributes= attribute)
        if value==0:
            p='Iris-setosa'
        elif value==1:
            p='Iris-versicolor'
        else:
            p='Iris-verginica'


        st.header("Here are the results:")
        st.success(f"The class predicted is {p}")