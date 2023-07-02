from forex_python.converter import CurrencyRates
import streamlit as st
import matplotlib.pyplot as plt
import mplfinance as mpf
import datetime as dt
import pandas as pd

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
    df = pd.DataFrame(results, index=index, columns=["Exchange Rate"])
    st.dataframe(df)

    ##############################
    # Plotting last 7 days chart of the 1st pair?
    fig_1, ax_1 = plt.subplots()
    fig_1.suptitle(f"Previous 7 Days Exchange rate and today for {base}/{dest[0]}")
    fig_1.supxlabel("Date")
    # fig_1.supylabel("Rate")
    fig_1.autofmt_xdate()
    # Loop from 7 days ago to today
    # Work from today and go backwards, then reverse the list at the end
    dates = []
    rates = []
    for i in range(8):  # 0 - 7 including today
        dates.append(today - dt.timedelta(days=i))
        rates.append(
            c.get_rate(
                base_cur=base, dest_cur=dest[0], date_obj=today - dt.timedelta(days=i)
            )
        )

    dates.reverse()  # Reverse the list (inplace)
    rates.reverse()

    # Plot
    ax_1.plot_date(dates, rates, ls="solid")
    ax_1.grid()
    st.pyplot(fig_1)

    values = pd.DataFrame(rates, index=dates, columns=["Rate"])
    st.dataframe(values)


else:
    st.subheader("No Values Detected")
