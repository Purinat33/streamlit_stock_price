from keras.datasets import mnist
import numpy as np
import streamlit as st
from keras.models import Sequential, load_model
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import pandas as pd
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
import random

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
st.title("MNIST Handwritten Digit Classification App")
st.write("This application will demonstrate Deep Learning Modeling on an MNIST dataset")
st.divider()


# Example Data
st.header("Example Data:")
st.subheader("Training Data:")
ex_fig, ex_axis = plt.subplots()
ex_fig.supxlabel(f"Label: {Y_train[4]}")
ex_axis.imshow(X_train[4], cmap="gray")
st.pyplot(ex_fig)

st.subheader("Testing Data:")
ex_fig_test, ex_axis_test = plt.subplots()
ex_fig_test.supxlabel(f"Label: {Y_test[220]}")
ex_axis_test.imshow(X_test[220], cmap="gray")
st.pyplot(ex_fig_test)

st.divider()
st.header("Model Creation with Keras")
st.code("pip install tensorflow keras scikit-learn")

# Modifying the data and labels
X_train_flatten = X_train.reshape((60000, 784))  # To 1D array of pixels
X_test_flatten = X_test.reshape((10000, 784))

# Labels
Y_train_label = to_categorical(Y_train)
Y_test_label = to_categorical(Y_test)

# @st.cache_resource
# def fit_train(_model, _x_train, _y_train, _epochs):
#     history = _model.fit(_x_train, _y_train, _epochs)
#     return history

# DL model
model = Sequential(
    [
        Dense(units=784, input_shape=(784,), activation="sigmoid"),
        Dense(units=16, activation="relu"),
        Dense(units=16, activation="relu"),
        Dense(units=10, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
st.subheader("Model Summary")
model.summary(print_fn=lambda x: st.write(x))
# model = load_model("my_mnist.keras")
# st.write(model)
# history = model.history.history
# model.summary(print_fn=lambda x: st.write(x))
# Training and Testing
st.header("Training Process")
st.write("Please wait for the training process to finish")

# Set epoch in sidebar

epochs = st.sidebar.number_input(
    "Numbers of Epochs", min_value=1, max_value=500, step=1
)
st.sidebar.warning(
    "Too much epochs and you risk overfitting/long waiting time while too low risks inaccurate predictions"
)
history = model.fit(
    X_train_flatten,
    Y_train_label,
    epochs=epochs,
    verbose=1,
    validation_data=(X_test_flatten, Y_test_label),
)

# Plotting the data and epoch
# Credit on how to plot and get data on each epoch via:
# https://stackabuse.com/custom-keras-callback-for-saving-prediction-on-each-epoch-with-visualizations/
model_history = pd.DataFrame(history.history)
fig_model, ax_model = plt.subplots()
num_epochs = model_history.shape[0]

fig_model.suptitle("Loss and Accuracy of training data")
ax_model.grid()
fig_model.supxlabel("Epoch")
ax_model.plot(np.arange(0, num_epochs), model_history["loss"], label="Loss")
ax_model.plot(np.arange(0, num_epochs), model_history["accuracy"], label="Accuracy")
fig_model.legend(loc="upper right")
st.pyplot(fig_model)

# Training
st.write("Evaluating on train dataset")
loss, acc = model.evaluate(X_train_flatten, Y_train_label, verbose=2)
metrices = {"Loss": loss, "Accuracy (%)": acc}
train_metrics = pd.DataFrame(
    [loss, round(acc * 100, 3)], index=metrices.keys(), columns=["Value"]
)
st.write(train_metrics)

# Testing
st.header("Testing Process")
st.write("Evaluating on test dataset")
loss, acc = model.evaluate(X_test_flatten, Y_test_label, verbose=2)
metrices = {"Loss": loss, "Accuracy (%)": acc}

test_metrics = pd.DataFrame(
    [loss, round(acc * 100, 3)], index=metrices.keys(), columns=["Value"]
)
st.write(test_metrics)

# st.write(history)

# Accuracy Metrics (for both)
st.header("Model Accuracy")
fig_acc, ax_acc = plt.subplots()
fig_acc.suptitle("Model Accuracy")
ax_acc.plot(history.history["accuracy"], label="train")
ax_acc.plot(history.history["val_accuracy"], label="test")
fig_acc.legend(loc="upper right")
st.pyplot(fig_acc)

st.header("Model Loss")
fig_loss, ax_loss = plt.subplots()
fig_loss.suptitle("Model Loss")
ax_loss.plot(history.history["loss"], label="train")
ax_loss.plot(history.history["val_loss"], label="test")
fig_loss.legend(loc="upper right")
st.pyplot(fig_loss)

y_true = Y_test_label
y_true = y_true.argmax(axis=1)

y_pred = model.predict(X_test_flatten)
y_pred = y_pred.argmax(axis=1)
st.header("Confusion Matrix")
mnist_label = []
for i in range(10):
    mnist_label.append(str(i))

cm = pd.DataFrame(
    confusion_matrix(y_true, y_pred), index=mnist_label, columns=mnist_label
)

st.write(cm)


# def getImage(label, y_test):
#     breakPoint = 0
#     while True:
#         rand = random.randint(0, 9999)
#         if y_test[rand] == label:  # If the index = the label we want
#             index = rand
#             return index
#         breakPoint += 1
#         if breakPoint >= 100:
#             return 0


# st.header("Test Your Own")
# user_label = st.selectbox("Number (0-9)", [i for i in range(10)])
# index = getImage(user_label, Y_test)
# fig_user, ax_user = plt.subplots()
# fig_user.suptitle(f"Label {Y_test[index]}")
# ax_user.imshow(X_test[index], cmap="gray")
# st.pyplot(fig_user)


# Save model
if st.button("Save model"):
    model.save("my_mnist.h5", overwrite=True)
