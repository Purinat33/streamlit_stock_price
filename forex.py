from forex_python.converter import CurrencyRates
import streamlit as st
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np

c = CurrencyRates()
today = dt.date.today()


st.title("Foreign Currency Exchange Market")
st.write(
    """
    ## How to read an exchange rate:
If the USD/CAD currency pair is 1.33, that means it costs 1.33 Canadian dollars to get 1 U.S. dollar.
In USD/CAD, the first currency listed (USD) always stands for one unit of that currency; 
the exchange rate shows how much of the second currency (CAD) is needed to purchase that one unit of the first (USD).
         """
)
st.divider()

# Getting list of currencies
exchange_rates = c.get_rates("THB")  # We get a dictionary excluding THB
currency = list(["THB"])
for key in exchange_rates.keys():
    currency.append(key)
########

st.sidebar.title("Settings")
base = st.sidebar.selectbox("Select Base Currency", options=currency)
dest = st.sidebar.multiselect("Select Destination Currencies", currency, "USD")
if st.sidebar.button("Add All"):
    x = currency.copy()
    dest = []

    for curr in currency:
        if curr != base:
            dest.append(curr)

st.header(f"Currency Exchange Rates of {today}")

if dest:  # If the destination currencies box is not empty
    # Store all exchange rates in a list
    results = []
    for curr in dest:
        results.append(c.get_rate(base_cur=base, dest_cur=curr, date_obj=today))

    # Create a DF to make it displayed in table in Streamlit
    # Make it so that the index is in the form of:
    # THB/USD
    # THB/JPY
    # THB/EUR
    # ...
    index = [f"{base}/{d}" for d in dest]
    df = pd.DataFrame(results, index=index, columns=["Exchange Rate Today"])
    # Add exchange rate of yesterday next to it

    prev_results = []
    for curr in dest:
        prev_results.append(
            c.get_rate(
                base_cur=base,
                dest_cur=curr,
                date_obj=today - dt.timedelta(days=1),  # Yesterday's Data
            )
        )

    # st.write(c.get_rate("THB", "USD", date_obj=today))
    # st.write(c.get_rate("THB", "USD", date_obj=today - dt.timedelta(days=1)))
    # df["Yesterday's Exchange Rate"] = pd.DataFrame(prev_results, index=index)
    df["Yesterday's Exchange Rate"] = prev_results

    df["Up/Down"] = np.where(
        df["Exchange Rate Today"] > df["Yesterday's Exchange Rate"],
        "+" + (df["Exchange Rate Today"] - df["Yesterday's Exchange Rate"]).astype(str),
        np.where(
            df["Exchange Rate Today"] < df["Yesterday's Exchange Rate"],
            (df["Exchange Rate Today"] - df["Yesterday's Exchange Rate"]).astype(str),
            "NONE",
        ),
    )

    st.dataframe(df)

    ##############################
    # This section is an entirely different section from the sidebar
    st.divider()
    st.header("View Currency Rates changes for a pair of currencies")
    st.subheader("Pair 1:")
    com_1_start = st.selectbox("Base", currency, key="base1")
    com_1_dest = st.selectbox("Destination", currency, key="dest1")

    # st.subheader("Pair 2:")
    # com_2_start = st.selectbox("Base", currency, key="base2")
    # com_2_dest = st.selectbox("Destination", currency, key="dest2")

    # Plotting last 7 days chart of both pair
    fig_1, ax_1 = plt.subplots()
    fig_1.suptitle(
        f"Previous 7 Days Exchange rate and today for {com_1_start}/{com_1_dest}"
    )
    fig_1.supxlabel("Date")
    # fig_1.supylabel("Rate")
    fig_1.autofmt_xdate()
    # Loop from 7 days ago to today
    # Work from today and go backwards, then reverse the list at the end
    dates = []
    rates_1 = []
    # rates_2 = []
    for i in range(8):  # 0 - 7 including today
        dates.append(today - dt.timedelta(days=i))
        rates_1.append(
            c.get_rate(
                base_cur=com_1_start,
                dest_cur=com_1_dest,
                date_obj=today - dt.timedelta(days=i),
            )
        )
        # rates_2.append(
        # c.get_rate(
        # base_cur=com_2_start,
        # dest_cur=com_2_dest,
        # date_obj=today - dt.timedelta(days=i),
        # )
        # )

    dates.reverse()  # Reverse the list (inplace)
    rates_1.reverse()

    dates_index = pd.DatetimeIndex(dates)
    # rates_2.reverse()
    values_1 = pd.DataFrame(rates_1, index=dates_index, columns=["Rate"])
    # Plot
    ax_1.plot_date(dates, rates_1, ls="solid")
    # ax_1.plot_date(dates, rates_2, ls="solid", label=f"{com_2_start}/{com_2_dest}")
    # fig_1.legend()
    ax_1.grid()
    st.pyplot(fig_1)
    # fig, ax = mpf.plot(
    #     values_1,
    #     type="candle",
    #     returnfig=True,
    # )

    # st.pyplot(fig)

    st.dataframe(values_1)

    # values_2 = pd.DataFrame(rates_2, index=dates, columns=["Rate"])
    # st.dataframe(values_2)
else:
    st.subheader("No Values Detected")
