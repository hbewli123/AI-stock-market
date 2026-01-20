import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go

st.title("Stock Market 30-Day Outlook")

ticker = st.text_input("Enter Stock Ticker", "Ex: TSLA")

data = yf.download(ticker, period="2y", multi_level_index=False)

if not data.empty:
    data.reset_index(inplace=True)

    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    df_train['ds'] = df_train['ds'].dt.tz_localize(None)

    m = Prophet(daily_seasonality=True)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)

    st.subheader(f"Next 30 Days Forecast for {ticker}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], name="Historical"))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted Outlook"))
    st.plotly_chart(fig)
else:
    st.error("No data found for this ticker. Please check the symbol.")


st.subheader(f"Next 30 Days Forecast for {ticker}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], name="Historical"))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted Outlook"))

# Essential: Add the config parameter here
st.plotly_chart(fig, config={'scrollZoom': True}, use_container_width=True)
