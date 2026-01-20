import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go

st.title("Stock Market 30-Day Outlook")

# 1. User Input for Ticker
ticker = st.text_input("Enter Stock Ticker", "Ex: TSLA")

# 2. Fetch Live Data - Added multi_level_index=False to fix the TypeError
data = yf.download(ticker, period="2y", multi_level_index=False)

if not data.empty:
    data.reset_index(inplace=True)

    # 3. Prepare Data for Prophet
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    df_train['ds'] = df_train['ds'].dt.tz_localize(None)

    # 4. Model & Forecast
    m = Prophet(daily_seasonality=True)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)

    # 5. Visualize
    st.subheader(f"Next 30 Days Forecast for {ticker}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], name="Historical"))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted Outlook"))
    st.plotly_chart(fig)
else:
    st.error("No data found for this ticker. Please check the symbol.")
