import streamlit as st
from forex_python.bitcoin import BtcConverter
from forex_python.converter import CurrencyRates  # Used to get all the symbols
from forex_python.converter import CurrencyCodes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

b = BtcConverter()


# A function to return the symbol of the currency
def getSymbol(
    currency_name: str, currency_list: list[str], code_list: list[str]
) -> str:
    symbol = ""
    for i in range(len(currency_list)):
        if currency_list[i] == currency_name:
            symbol = code_list[i]
            break

    return symbol


def getIndex(currency_name: str, currency_list: list[str]) -> int:
    for i in range(len(currency_list)):
        if currency_list[i] == currency_name:
            return i

    return 0


# For getting all the currency
c = CurrencyRates()
currencies = ["USD"]
dict_currencies = c.get_rates(currencies[0])

for key in dict_currencies.keys():
    currencies.append(key)
# We now have the list of currencies in a currencies list()

# How about we get all the currency unicode symbol as well?
c = CurrencyCodes()
codes = []
for currency in currencies:
    codes.append(c.get_symbol(currency))
# We now have the same thing for the symbols

# Section 1
st.title("Bitcoin prices and exchange rate")
st.write(
    """
This app will show the current exchange rate of bitcoin (BTC).
         """
)
st.divider()

# Section 2

st.header("User Input")
# Sidebar settings (Update: looks like we do not need a sidebar just to do a quick 1 currency calculation)
selected_currency = st.selectbox("Select Currency: ", currencies, key="curr1")
# Get the index of the selected data and retrieve the symbol from the codes list
currency_code = getSymbol(selected_currency, currencies, codes)
st.write(f"Currency: {selected_currency} ({currency_code})")

# # Today Variable
# today = dt.date.today()
# date = st.date_input("(Optional) Select Date: ", today)

# if date > today:  # Choosing a date in the future
#     st.error("Could not select future dates")
#     date = dt.date.today()

# Current Exchange Rate
# How much is 1 bitcoin
st.header(f"Current Exchange Rate BTC/{selected_currency}")
st.subheader(
    f"1 BTC = :green[{b.get_latest_price(selected_currency)}]"
    + f" {selected_currency} ({(currency_code)})"
)

st.divider()

st.header("Custom Amount Exchange")
selected_currency = st.selectbox(
    "Select Currency: ",
    currencies,
    index=getIndex(selected_currency, currencies),
    key="curr2",
)
# Amount in selected currency
amount_curr = st.number_input(
    label=f"Amount in {selected_currency}",
    min_value=0.00,
    max_value=9999999.00,
    step=0.01,
)

# Section 3: convertion
# How much bitcoin can our amount variable get
st.subheader(
    f"{amount_curr} {selected_currency} ({currency_code}) = "
    + f":green[{b.convert_to_btc(amount_curr, selected_currency)}]"
    + " BTC"
)

st.divider()

# Section 4: Past data?
# st.header("Historical Prices of 1 BTC")
# selected_currency = st.selectbox(
#     "Select Currency: ",
#     currencies,
#     index=getIndex(selected_currency, currencies),
#     key="curr3",
# )
# # dates list of last 7 days + today
# today = dt.date.today()
# dates = []
# for i in range(8):
#     dates.append(today - dt.timedelta(days=i))  # today, today - 1, etc

# # Reverse the dates list so we get oldest first, then today at the end
# dates.reverse()
# dates_index = pd.DatetimeIndex(dates)

# st.write(b.get_previous_price(selected_currency, dates[1]))

# # Prices
# prices_1btc = []
# for i in range(8):
#     prices_1btc.append(b.get_previous_price(selected_currency, dates_index))

# # Same thing as dates
# prices_1btc.reverse()

# dates_index = pd.DatetimeIndex(dates)
# df = pd.DataFrame(
#     prices_1btc, index=dates_index, columns=[f"Prices ({selected_currency})"]
# )

# st.dataframe(df)
