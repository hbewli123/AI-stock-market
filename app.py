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
df_train['ds'] = df_train['ds'].dt.tz_localize(None)

# -------------------------------
# Train Model & Forecast (PROPHET)
# -------------------------------
with st.spinner("Generating forecast..."):
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.2
    )
    model.fit(df_train)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

# -------------------------------
# Trend Direction (Green / Red)
# -------------------------------
past_start = forecast['yhat'].iloc[0]
past_end = forecast['yhat'].iloc[len(df_train) - 1]
past_up = past_end > past_start

future_start = forecast['yhat'].iloc[len(df_train) - 1]
future_end = forecast['yhat'].iloc[-1]
future_up = future_end > future_start

past_color = "green" if past_up else "red"
future_color = "green" if future_up else "red"

# -------------------------------
# Percent Gain Calculations
# -------------------------------
current_price = data['Close'].iloc[-1]

def percent_gain(days):
    predicted_price = forecast['yhat'].iloc[len(df_train) - 1 + days]
    return ((predicted_price - current_price) / current_price) * 100

gain_5 = percent_gain(5)
gain_10 = percent_gain(10)
gain_30 = percent_gain(30)

# -------------------------------
# Visualization
# -------------------------------
st.subheader(f"📊 {ticker} — Past Outlook & 30-Day Forecast")

fig = go.Figure()

# Actual Price
fig.add_trace(go.Scatter(
    x=df_train['ds'],
    y=df_train['y'],
    name="Actual Price",
    line=dict(color="#1f77b4")
))

# Past Outlook
fig.add_trace(go.Scatter(
    x=forecast['ds'].iloc[:len(df_train)],
    y=forecast['yhat'].iloc[:len(df_train)],
    name="Past Outlook",
    line=dict(color=past_color, width=2)
))

# Future Forecast
fig.add_trace(go.Scatter(
    x=forecast['ds'].iloc[len(df_train) - 1:],
    y=forecast['yhat'].iloc[len(df_train) - 1:],
    name="30-Day Forecast",
    line=dict(color=future_color, width=2, dash="dot")
))

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

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"scrollZoom": True},
    key="stock_chart_main"
)

# -------------------------------
# Metrics Display
# -------------------------------
st.subheader("📈 Predicted Returns")

col1, col2, col3 = st.columns(3)

col1.metric(
    "5-Day Outlook",
    f"{gain_5:.2f}%",
    delta=f"{gain_5:.2f}%"
)

col2.metric(
    "10-Day Outlook",
    f"{gain_10:.2f}%",
    delta=f"{gain_10:.2f}%"
)

col3.metric(
    "30-Day Outlook",
    f"{gain_30:.2f}%",
    delta=f"{gain_30:.2f}%"
)

# -------------------------------
# Summary
# -------------------------------
st.write(
    f"The current price of **{ticker}** is approximately "
    f"**${current_price:.2f}**."
)

trend_text = "Bullish 📈" if future_up else "Bearish 📉"
st.info(f"Model Outlook: **{trend_text}** over the next 30 days.")

