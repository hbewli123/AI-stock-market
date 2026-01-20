import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go

# Set page title
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Stock Market 30-Day stock price Outlook")

# 1. Sidebar for User Inputs
ticker = st.sidebar.text_input(
    "Enter Stock Ticker",
    placeholder="e.g. AAPL, TSLA, NVDA"
).upper()

period_history = st.sidebar.slider("Years of History to Train On", 1, 5, 2)

# 2. Fetch Live Data
# multi_level_index=False is critical for modern yfinance compatibility
data = yf.download(ticker, period=f"{period_history}y", multi_level_index=False)

if not data.empty:
    data.reset_index(inplace=True)

    # 3. Prepare Data for Prophet
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    # Remove timezone so Prophet doesn't crash
    df_train['ds'] = df_train['ds'].dt.tz_localize(None)

    # 4. Model & Forecast
    # We use st.spinner so the user knows the AI is working
    with st.spinner('Generating 30-day outlook...'):
        m = Prophet(daily_seasonality=True)
        m.fit(df_train)
        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)

    # 5. Visualize (Single Graph Container)
    st.subheader(f"30-Day Prediction for {ticker.upper()}")
    
    fig = go.Figure()

    # Historical Data Line
    fig.add_trace(go.Scatter(
        x=df_train['ds'], 
        y=df_train['y'], 
        name="Historical Price",
        line=dict(color='#1f77b4')
    ))

    # Predicted Data Line
    fig.add_trace(go.Scatter(
        x=forecast['ds'].iloc[-31:], # Focus on the last 30 days
        y=forecast['yhat'].iloc[-31:], 
        name="Predicted Outlook",
        line=dict(color='#ff7f0e', dash='dot')
    ))

    # Update layout to support mouse-centered zooming
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Display the chart with scrollZoom enabled
    st.plotly_chart(
        fig, 
        config={'scrollZoom': True}, 
        use_container_width=True, 
        key="stock_chart_main"
    )

    # 6. Data Summary
    st.write(f"The current price of **{ticker}** is approximately **${data['Close'].iloc[-1]:.2f}**.")
    st.info("Tip: Use your **mouse scroll wheel** while hovering over the chart to zoom in on specific price action.")

else:
    st.error("No data found. Please ensure the ticker symbol (e.g., TSLA, NVDA) is correct.")
