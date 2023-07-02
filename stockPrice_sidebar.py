# Wanted to try something new with the sidebars

import pandas_datareader as web
import streamlit as st
import pandas as pd
import datetime as dt
import mplfinance as mpf
import matplotlib.pyplot as plt
import seaborn as sns

symbol = "GOOGL"  # Default
d_end = dt.datetime.today() - dt.timedelta(days=1)
d_start = dt.datetime(2023, 6, 1)

st.title("A Stock Visualizing Application")
st.header("Visualizing Stock Chart")

# Sidebar
st.sidebar.header("Settings")
start = st.sidebar.date_input("Set Start Date", d_start)
end = st.sidebar.date_input("Set End Date", d_end)
symbol = st.sidebar.selectbox("Select Stock Symbol", [symbol, "TSLA", "META"])

if start >= end:
    start = end - dt.timedelta(days=1)

data = web.DataReader(name=symbol, data_source="stooq", start=start, end=end)
data.index = pd.to_datetime(data.index)

if data.empty:
    st.write(
        """

No stock data is available

             """
    )
else:
    fig, ax = mpf.plot(
        data,
        title=f"Showing Plot data for {symbol} from {start} to {end}",
        type="candle",
        returnfig=True,
        volume=True,
    )

    st.pyplot(fig)

    st.sidebar.info(
        """
Made by **Purinat33**

                    """
    )

    st.sidebar.write(
        """
    
        [Download File](https://www.google.com)
    
    """
    )

    if st.button("Display Data Details"):
        st.header("Tabular Data Format")
        st.write(data)

        st.subheader("Correlation Table")
        st.write(data.corr())

        st.subheader("Description")
        st.write(data.describe())

    if st.button("Display Graph"):
        fig_1, ax = plt.subplots()
        st.subheader("Heatmap of Correlation")
        fig_1.suptitle("Correlation")
        ax = sns.heatmap(data.corr(), annot=True)
        st.pyplot(fig_1)
