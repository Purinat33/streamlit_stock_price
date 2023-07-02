# TODO: Predict Stock Price with some form of regression algorithm
import streamlit as st
import pandas as pd
import mplfinance as mpf
import pandas_datareader as web
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

st.title("Stock Price Prediction Regressor")
st.write("The following data has been retrieved using")
st.code("df = pandas_datareader.DataReader(symbol, src, start, end)", "python")

st.write(
    """
    Note that this prediction should not be used for major decision making. This app is only the result of a machine learning algorithm and does not factor in any external factors nor global events. The developer will not be held responsible should any losses occurred
         """
)
st.sidebar.title("User Input")
sym = "GOOGL"
start_d = dt.date(2023, 6, 20)
end = dt.date.today()  # Will probably be default anyway

sym = st.sidebar.text_input("Symbol", sym)
start = st.sidebar.date_input("Start Date", start_d)
if start >= end:
    start = end - dt.timedelta(days=14)

df = web.DataReader(sym, "stooq", start, end)
st.header("Present Data")

if not df.empty:
    fig, ax = mpf.plot(
        df,
        title=f"Stock Chart of {sym} from {start} to {end}",
        type="candle",
        returnfig=True,
    )
    st.pyplot(fig)

if st.button("Display Tabular Data"):
    st.header("Stock Data in Tabular Format")

    if not df.empty:
        st.write(df.iloc[::-1])

# Prediction
st.header("Prediction of tomorrow's prices")
model = DecisionTreeRegressor()

# We want to predict everything
x = df[["Open", "High", "Low", "Close", "Volume"]]
y = df[["Open", "High", "Low", "Close", "Volume"]]

x_train = x[:-1]  # All row except last row
y_train = y[:-1]

x_test = x[-1:]  # Today or Latest's values

model.fit(x_train, y_train)
predicted = model.predict(x_test)  # array
result = {
    "Open": predicted[:, 0],
    "High": predicted[:, 1],
    "Low": predicted[:, 2],
    "Close": predicted[:, 3],
    "Volume": predicted[:, 4],
}

# Setting the Index to be tomorrow's Date
new_index = [dt.date.today() + pd.DateOffset(days=1)]
results = pd.DataFrame(result, index=new_index)

st.write(results)
st.divider()
st.header("Charts of predicted Values")

data = pd.concat([df.iloc[::-1], results], axis=0)
# data.index[-1] = dt.date.today() + pd.DateOffset(days=1)
# Plotting plt charts
st.subheader("Open/Close")
fig_o, ax_o = plt.subplots()
fig_o.suptitle = "Open/Close"
ax_o.plot_date(data.index, data["Open"], ls="solid", label="Open")
ax_o.plot_date(data.index, data["Close"], ls="solid", label="Close")
fig_o.legend()
fig_o.autofmt_xdate()
st.pyplot(fig_o)

st.subheader("High/Low")
fig_h, ax_h = plt.subplots()
fig_h.suptitle = "High/Low"
ax_h.plot_date(data.index, data["High"], ls="solid", label="High")
ax_h.plot_date(data.index, data["Low"], ls="solid", label="Low")
fig_h.legend()
fig_h.autofmt_xdate()
st.pyplot(fig_h)

# Volume
st.subheader("Volume")
fig_v, ax_v = plt.subplots()
fig_v.suptitle("Volume")
ax_v.bar(data.index, data["Volume"])
fig_v.autofmt_xdate()
st.pyplot(fig_v)
