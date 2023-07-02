# TODO: Create and plot 2 different stocks on the same graph?
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import streamlit as st
import pandas_datareader as web
import datetime as dt
import numpy as np

st.title("Application for comparing two stocks")

sym_1 = "GOOGL"
sym_2 = "META"

start_d = dt.datetime(2023, 6, 20)
end_d = dt.datetime.today()


# Sidebar
st.sidebar.header("Configure Stock Data")

st.sidebar.subheader("Stock 1")
sym_1 = st.sidebar.text_input("Stock 1 Symbol", sym_1)

st.sidebar.subheader("Stock 2")
sym_2 = st.sidebar.text_input("Stock 2 Symbol", sym_2)

st.sidebar.divider()
st.sidebar.subheader("Configure Date")
start = st.sidebar.date_input("Start Date", start_d)
end = st.sidebar.date_input("End Date", end_d)

if start >= end:
    start = end - dt.timedelta(days=7)
# Sidebar

# Getting the data
d1 = web.DataReader(sym_1, "stooq", start, end)
d2 = web.DataReader(sym_2, "stooq", start, end)


def plot_title(name, start, end):
    return f"Data for {name} from {start} to {end}"


# Stock 1
st.header("Stock 1")
if d1.empty:
    st.write("""Invalid Stock Symbol or No Data""")
else:
    fig_1, ax_1 = mpf.plot(
        d1,
        title=plot_title(sym_1, start, end),
        type="candle",
        returnfig=True,
        volume=True,
    )
    st.pyplot(fig_1)

# Stock 2
st.header("Stock 2")
if d2.empty:
    st.write("""Invalid Stock Symbol or No Data Available""")

else:
    fig_2, ax_2 = mpf.plot(
        d2,
        title=plot_title(sym_2, start, end),
        type="candle",
        returnfig=True,
        volume=True,
    )
    st.pyplot(fig_2)

# Plot stuff on the same line
if st.button("Display Individual Values Comparison"):
    if d1.empty or d2.empty:
        st.write("Cannot display charts as stock 1 or 2 or both is invalid")
    else:
        # Open
        st.subheader("Open")
        fig_o, ax_o = plt.subplots()
        fig_o.suptitle("Opening Prices")
        ax_o.plot_date(d1.index, d1["Open"], ls="solid", label=f"{sym_1}")
        ax_o.plot_date(d2.index, d2["Open"], ls="solid", label=f"{sym_2}")
        fig_o.legend()
        st.pyplot(fig_o)

        # Close
        st.subheader("Close")
        fig_c, ax_c = plt.subplots()
        fig_c.suptitle("Closing Prices")
        ax_c.plot_date(d1.index, d1["Close"], ls="solid", label=f"{sym_1}")
        ax_c.plot_date(d2.index, d2["Close"], ls="solid", label=f"{sym_2}")
        fig_c.legend()
        st.pyplot(fig_c)

        # Volume?
        st.subheader("Volume")
        x_indexes = np.arange(len(d1.index))
        width = 0.2
        fig_v, ax_v = plt.subplots()
        fig_v.suptitle("Volume")
        ax_v.bar(x_indexes, d1["Volume"], width=width, label=f"{sym_1}")
        ax_v.bar(x_indexes + width, d2["Volume"], width=width, label=f"{sym_2}")
        ax_v.set_xticks(ticks=x_indexes, labels=d1.index)
        fig_v.autofmt_xdate()
        fig_v.legend()
        st.pyplot(fig_v)
