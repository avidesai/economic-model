# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from fredapi import Fred
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

# set the end date to today
end_date = datetime.now()

# convert the end date to a string in the format 'YYYY-MM-DD'
end_date = end_date.strftime('%Y-%m-%d')

# Define the start date as 50 years ago
start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365*50)).strftime('%Y-%m-%d')

# Create an instance of the FRED class
fred = Fred(api_key='6a58db9c85745819db30352da7fef223')

# %% [markdown]
# <h3>Indicators</h3>
# <ul>
#     <li>Unemployment rate</li>
#     <li>Gross Domestic Product (GDP)</li>
#     <li>Consumer Price Index (CPI)</li>
#     <li>Industrial Production Index</li>
#     <li>Durable goods orders</li>
#     <li>Consumer Confidence Index</li>
#     <li>ISM Manufacturing Index</li>
# </ul>

# %%
# create a dictionary to store indicators
indicators = {
    'UNRATE': 'Unemployment rate',
    'CPIAUCNS': 'Consumer Price Index (CPI)',
    'INDPRO': 'Industrial Production Index',
    'DGORDER': 'Durable goods orders',
    'UMCSENT': 'Consumer Confidence Index',
    'MANEMP': 'Institute for Supply Management (ISM) Manufacturing Index',
    'HOUST': 'Housing Starts',
    'PERMIT': 'Building Permits',
}

# %%
# create an empty dictionary to store the dataframes
dfs = {}

# loop through the indicators
for key, value in indicators.items():
    try:
        # retrieve data using the key as the series identifier
        data = pd.DataFrame(fred.get_series(key, start_date=start_date, end_date=end_date))
        # add the data to the dfs dictionary using the value as the key
        dfs[value] = data
    except:
        print(f"{value} not found")

# Print the keys of the dfs dictionary
print(dfs.keys())

# %% [markdown]
# <h3>Statistical Analysis</h3>

# %%
# Perform descriptive statistics and cleanup on all dataframes
for key, value in dfs.items():
    # remove rows with missing values
    value.dropna(inplace=True)
    # reset the index
    value.reset_index(drop=True, inplace=True)
    # update the dfs dictionary with the cleaned data
    dfs[key] = value
    print(f"{key} statistics:")
    print(value.describe())

# %%
# # Visualize all dataframes
# for key, value in dfs.items():
#     value.plot()
#     plt.title(key)
#     plt.show()

# %% [markdown]
# <h3>Time Series Analysis</h3>    

# %%
interval = 5
# loop through the dataframes in the dfs dictionary
for key, df in dfs.items():
    adf_test = sm.tsa.stattools.adfuller(df)
    # print("ADF test p-value: ", adf_test[1])
    # fit the ARIMA model to the data
    model = ARIMA(df, order=(1, 0, 0))
    model_fit = model.fit()
    
    # make predictions
    predictions = model_fit.predict(start=len(df), end=len(df)+10, dynamic=False)
    
    # plot the predictions against the actual data
    plt.plot(df)
    plt.plot(predictions, color='red')
    plt.title(key)
    plt.show()

# %%
import streamlit as st
st.subheader("Key Economic Indicators + Projections")
st.markdown("Historical time-series data and ARIMA projections of economic data (charted in red)")

for i, (key, df) in enumerate(dfs.items()):
    st.write("")
    st.subheader(key)
    # fit the ARIMA model to the data
    model = ARIMA(df, order=(1, 0, 0))
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(df), end=len(df)+10, dynamic=False)
    # plot the predictions against the actual data
    fig, ax = plt.subplots()
    ax.plot(df)
    ax.plot(predictions, color='red')
    plt.title(key)
    st.pyplot(fig)
