import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Stock Market 30-Day Stock Price Outlook")

# -------------------------------
# Sidebar Input
# -------------------------------
ticker = st.sidebar.text_input(
    "Enter Stock Ticker",
    placeholder="e.g. AAPL, TSLA, NVDA"
).upper()

# Fixed training window
period_history = 2  # YEARS (LOCKED)

# Stop execution if no ticker entered
if not ticker:
    st.info("👈 Enter a stock ticker in the sidebar to begin.")
    st.stop()

# -------------------------------
# Fetch Historical Data
# -------------------------------
data = yf.download(
    ticker,
    period=f"{period_history}y",
    multi_level_index=False
)

if data.empty:
    st.error("No data found. Please ensure the ticker symbol is correct.")
    st.stop()

data.reset_index(inplace=True)

# -------------------------------
# Prepare Data for Prophet
# -------------------------------
df_train = data[['Date', 'Close']].rename(
    columns={"Date": "ds", "Close": "y"}
)

# Remove timezone (required for Prophet)
df_train['ds'] = df_train['ds'].dt.tz_localize(None)

# -------------------------------
# Train Model & Forecast
# -------------------------------
with st.spinner("Generating forecast..."):
    model = Prophet(daily_seasonality=True)
    model.fit(df_train)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

# -------------------------------
# Visualization
# -------------------------------
st.subheader(f"📊 {ticker} — Past Outlook & 30-Day Forecast")

fig = go.Figure()

# Actual Historical Prices
fig.add_trace(go.Scatter(
    x=df_train['ds'],
    y=df_train['y'],
    name="Actual Price",
    line=dict(color="#1f77b4")
))

# Model's Past Outlook (historical predictions)
fig.add_trace(go.Scatter(
    x=forecast['ds'].iloc[:-30],
    y=forecast['yhat'].iloc[:-30],
    name="Model Past Outlook",
    line=dict(color="rgba(255,127,14,0.5)", dash="dash")
))

# Future 30-Day Forecast
fig.add_trace(go.Scatter(
    x=forecast['ds'].iloc[-31:],
    y=forecast['yhat'].iloc[-31:],
    name="30-Day Forecast",
    line=dict(color="#ff7f0e", dash="dot")
))

# Confidence Interval
fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat_upper'],
    line=dict(width=0),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat_lower'],
    fill='tonexty',
    line=dict(width=0),
    name="Confidence Interval",
    opacity=0.2
))

# Layout
fig.update_layout(
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Render Chart
st.plotly_chart(
    fig,
    use_container_width=True,
    config={"scrollZoom": True},
    key="stock_chart_main"
)

# -------------------------------
# Summary
# -------------------------------
current_price = data['Close'].iloc[-1]

st.write(
    f"The current price of **{ticker}** is approximately "
    f"**${current_price:.2f}**."
)

st.info(
    "💡 Tip: Scroll your mouse wheel while hovering over the chart to zoom in."
)
