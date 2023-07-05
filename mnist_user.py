import streamlit as st
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

st.title("MNIST Handwritten Digit Prediction")
st.write("Now you can run MNIST yourself!")
st.divider()

# IMPORTING THE MODEL
# Path starts at ~/users/user_name for some reasom
# Update: File path for opening in Streamlit and for running in normal python are DIFFERENT
filepath = os.path.abspath("./Desktop/youtube/streamlit/my_mnist.h5")
model = keras.models.load_model(filepath)

st.header("Model Overview")
model.summary(print_fn=lambda x: st.write(x))

selected = st.selectbox("Select Digit [0-9]", [i for i in range(10)])
