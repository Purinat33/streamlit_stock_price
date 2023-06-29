import streamlit as st
import pandas_datareader as web
from datetime import date, timedelta
import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import drange

# TODO: Sidebars, Prediction, styling etc.

st.write(
    """
# Stock Price Information
Dord
***
         """
)

####### Input #######
st.header("Input")

symbol = "GOOGL"
symbol = st.text_input("Symbol Here", "GOOGL")
symbol = symbol.upper().strip()

start = date.today() - timedelta(days=1)  # Yesterday
start = st.date_input(label="Start Date", value=start)
end = date.today()

if start >= end:
    st.write("""Date invalid""")
else:
    ##### OUTPUT ######
    st.header("Results")

    data = web.DataReader(symbol, "stooq", start, end)
    data.index = pd.to_datetime(data.index)

    if data.empty:
        st.write(
            """

            No data available
                 
                 """
        )
    else:
        st.subheader("Tabular Format")

        st.write(data)
        ####### Plotting #######
        df = pd.DataFrame(data)

        fig, ax = mpf.plot(
            data=df,
            title=f"{symbol} with {start} as start date",
            type="candle",
            # Need this setting for Streamlit, see source code (line 778) here:
            # https://github.com/matplotlib/mplfinance/blob/master/src/mplfinance/plotting.py
            returnfig=True,
            volume=True,
        )

        # For reference:
        # https://gist.github.com/asehmi/93c9e2934b26fc86b0a9283e4d7d0f5d

        st.subheader("Candlebar Format")
        st.pyplot(fig)

        # Drop data when there is no transaction (market closed)
        ### TODO: Don't plot dates with no market
        # Code here
        df = df[df["Open"] >= 0]
        #######

        # Date for plt.plot_date
        delta = timedelta(hours=24)
        dates = drange(start, end, delta)
        # dates = pd.date_range(start, end, freq="D")

        # High/Low
        st.subheader("Individual Value")

        fig1, ax1 = plt.subplots()
        fig1.suptitle("High/Low")
        fig1.supxlabel("Time")
        fig1.supylabel("Price")

        ax1.plot_date(df.index, df["High"], label="High", linestyle="solid")
        ax1.plot_date(df.index, df["Low"], label="Low", linestyle="solid")
        fig1.legend(loc="upper right")
        fig1.autofmt_xdate()
        st.pyplot(fig1)

        # Open/Close
        fig2, ax2 = plt.subplots()
        fig2.suptitle("Open/Close")
        fig2.supxlabel("Time")
        fig2.supylabel("Price")

        ax2.plot_date(df.index, df["Open"], label="Open", linestyle="solid")
        ax2.plot_date(df.index, df["Close"], label="Close", linestyle="solid")
        fig2.legend(loc="upper right")
        fig2.autofmt_xdate()
        st.pyplot(fig2)

        # Volumes
        fig3, ax3 = plt.subplots()
        fig3.suptitle("Volumes")
        fig3.supxlabel("Date")
        fig3.supylabel("Volumes")
        fig3.autofmt_xdate()

        ax3.bar(df.index, df["Volume"])
        st.pyplot(fig3)
