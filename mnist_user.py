import streamlit as st
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import urllib.request

# import os

st.title("MNIST Handwritten Digit Prediction")
st.write("Now you can run MNIST yourself!")
st.divider()

# IMPORTING THE MODEL
# Path starts at ~/users/user_name for some reasom
# Update: File path for opening in Streamlit and for running in normal python are DIFFERENT

# Load locally
# filepath = os.path.abspath("./Desktop/youtube/streamlit/my_mnist.h5")
# model = keras.models.load_model(filepath)

# Load from GitHub
# Note that the file URL needs to be for the `raw` file
url = "https://github.com/Purinat33/streamlit_stock_price/raw/master/my_mnist.h5"
file_path = keras.utils.get_file("my_mnist.h5", origin=url)
model = keras.models.load_model(file_path)
# Load image
img_url = (
    "https://github.com/Purinat33/streamlit_stock_price/raw/master/mnist_overview.png"
)
st.header("Model Overview")
urllib.request.urlretrieve(img_url, "mnist_overview.png")
img = Image.open("mnist_overview.png")
st.image(img)

# User Input
selected = st.selectbox("Select Digit [0-9]", [i for i in range(10)])
