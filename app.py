import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
import numpy as np

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Stock Market 30-Day Stock Price Outlook")

# -------------------------------
# Helper Functions
# -------------------------------
@st.cache_data(ttl=3600)
def get_news_sentiment(ticker):
    """Fetch recent news and calculate sentiment score"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news[:10] if hasattr(stock, 'news') and stock.news else []
        
        if not news:
            return 0.0
        
        sentiments = []
        for article in news:
            title = article.get('title', '')
            if title:
                blob = TextBlob(title)
                sentiments.append(blob.sentiment.polarity)
        
        return np.mean(sentiments) if sentiments else 0.0
    except:
        return 0.0

@st.cache_data(ttl=3600)
def get_market_indicators(ticker):
    """Get additional market indicators"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        indicators = {
            'rsi': 50,  # Default neutral
            'volume_trend': 0,
            'analyst_sentiment': 0
        }
        
        # Volume trend
        hist = stock.history(period="1mo")
        if not hist.empty and len(hist) > 1:
            recent_vol = hist['Volume'].iloc[-5:].mean()
            prev_vol = hist['Volume'].iloc[-20:-5].mean()
            indicators['volume_trend'] = (recent_vol - prev_vol) / prev_vol if prev_vol > 0 else 0
        
        # Analyst recommendations
        if 'recommendationKey' in info:
            rec_map = {
                'strong_buy': 1.0,
                'buy': 0.5,
                'hold': 0.0,
                'sell': -0.5,
                'strong_sell': -1.0
            }
            indicators['analyst_sentiment'] = rec_map.get(info['recommendationKey'], 0)
        
        return indicators
    except:
        return {'rsi': 50, 'volume_trend': 0, 'analyst_sentiment': 0}

def calculate_market_adjustment(sentiment_score, indicators):
    """Calculate price adjustment based on sentiment and market indicators"""
    # Weighted combination of factors
    adjustment = (
        sentiment_score * 0.4 +  # News sentiment (40%)
        indicators['analyst_sentiment'] * 0.3 +  # Analyst sentiment (30%)
        np.clip(indicators['volume_trend'], -0.5, 0.5) * 0.3  # Volume trend (30%)
    )
    
    # Scale adjustment to reasonable percentage (-10% to +10%)
    return np.clip(adjustment * 0.1, -0.1, 0.1)

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
with st.spinner("Fetching stock data..."):
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
# Fetch News Sentiment & Market Data
# -------------------------------
with st.spinner("Analyzing market sentiment and news..."):
    sentiment_score = get_news_sentiment(ticker)
    market_indicators = get_market_indicators(ticker)
    market_adjustment = calculate_market_adjustment(sentiment_score, market_indicators)

# -------------------------------
# Prepare Data for Prophet
# -------------------------------
df_train = data[['Date', 'Close']].rename(
    columns={"Date": "ds", "Close": "y"}
)
df_train['ds'] = df_train['ds'].dt.tz_localize(None)

# Add sentiment as a regressor
df_train['sentiment'] = sentiment_score
df_train['market_factor'] = market_adjustment

# -------------------------------
# Train Model & Forecast (PROPHET)
# -------------------------------
with st.spinner("Generating enhanced forecast..."):
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.2,
        seasonality_mode='multiplicative'
    )
    
    # Add regressors for sentiment and market factors
    model.add_regressor('sentiment')
    model.add_regressor('market_factor')
    
    model.fit(df_train)

    future = model.make_future_dataframe(periods=30)
    
    # Extend sentiment and market factors to future predictions
    future['sentiment'] = sentiment_score
    future['market_factor'] = market_adjustment
    
    forecast = model.predict(future)
    
    # Apply additional market adjustment to future predictions
    future_mask = future.index >= len(df_train)
    forecast.loc[future_mask, 'yhat'] *= (1 + market_adjustment)
    forecast.loc[future_mask, 'yhat_lower'] *= (1 + market_adjustment * 0.5)
    forecast.loc[future_mask, 'yhat_upper'] *= (1 + market_adjustment * 1.5)

# -------------------------------
# Trend Direction
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
# Percent Gain + Price Targets
# -------------------------------
current_price = data['Close'].iloc[-1]

def predicted_price(days):
    return forecast['yhat'].iloc[len(df_train) - 1 + days]

def percent_gain(days):
    return ((predicted_price(days) - current_price) / current_price) * 100

price_5, price_10, price_30 = (
    predicted_price(5),
    predicted_price(10),
    predicted_price(30)
)

gain_5, gain_10, gain_30 = (
    percent_gain(5),
    percent_gain(10),
    percent_gain(30)
)

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

# Confidence interval
fig.add_trace(go.Scatter(
    x=forecast['ds'].iloc[len(df_train) - 1:],
    y=forecast['yhat_upper'].iloc[len(df_train) - 1:],
    fill=None,
    mode='lines',
    line=dict(color='rgba(0,0,0,0)'),
    showlegend=False,
    hoverinfo='skip'
))

fig.add_trace(go.Scatter(
    x=forecast['ds'].iloc[len(df_train) - 1:],
    y=forecast['yhat_lower'].iloc[len(df_train) - 1:],
    fill='tonexty',
    mode='lines',
    line=dict(color='rgba(0,0,0,0)'),
    fillcolor='rgba(68, 68, 68, 0.1)',
    name='Confidence Interval',
    hoverinfo='skip'
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
# Metrics
# -------------------------------
st.subheader("📈 Predicted Returns")

col1, col2, col3 = st.columns(3)

col1.metric("5-Day Outlook", f"${price_5:.2f}", f"{gain_5:.2f}%")
col2.metric("10-Day Outlook", f"${price_10:.2f}", f"{gain_10:.2f}%")
col3.metric("30-Day Outlook", f"${price_30:.2f}", f"{gain_30:.2f}%")

# -------------------------------
# Market Sentiment Indicators
# -------------------------------
st.subheader("🎯 Market Intelligence")

col1, col2, col3 = st.columns(3)

# News Sentiment
sentiment_emoji = "🟢" if sentiment_score > 0.1 else "🔴" if sentiment_score < -0.1 else "🟡"
sentiment_label = "Positive" if sentiment_score > 0.1 else "Negative" if sentiment_score < -0.1 else "Neutral"
col1.metric("News Sentiment", sentiment_label, f"{sentiment_emoji}")

# Volume Trend
volume_trend = market_indicators['volume_trend']
volume_emoji = "📈" if volume_trend > 0.1 else "📉" if volume_trend < -0.1 else "➡️"
volume_label = f"{volume_trend*100:.1f}%"
col2.metric("Volume Trend", volume_label, f"{volume_emoji}")

# Analyst Sentiment
analyst_score = market_indicators['analyst_sentiment']
analyst_emoji = "👍" if analyst_score > 0.3 else "👎" if analyst_score < -0.3 else "👌"
analyst_map = {1.0: "Strong Buy", 0.5: "Buy", 0.0: "Hold", -0.5: "Sell", -1.0: "Strong Sell"}
analyst_label = analyst_map.get(analyst_score, "Hold")
col3.metric("Analyst Rating", analyst_label, f"{analyst_emoji}")

# -------------------------------
# Summary
# -------------------------------
st.write(
    f"The current price of **{ticker}** is approximately "
    f"**${current_price:.2f}**."
)

trend_text = "Bullish 📈" if future_up else "Bearish 📉"
confidence_text = "High" if abs(sentiment_score) > 0.2 or abs(market_adjustment) > 0.05 else "Moderate"
st.info(f"Model Outlook: **{trend_text}** over the next 30 days (Confidence: {confidence_text}).")

