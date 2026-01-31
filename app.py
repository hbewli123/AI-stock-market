import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Stock Market 1-Year Stock Price Outlook")

# -------------------------------
# Helper Functions
# -------------------------------
def simple_sentiment_analysis(text):
    """Simple keyword-based sentiment analysis"""
    if not text:
        return 0.0
    
    text = text.lower()
    
    # Positive keywords
    positive_words = [
        'surge', 'soar', 'rally', 'gain', 'profit', 'growth', 'rise', 'bull', 
        'upgrade', 'beat', 'strong', 'record', 'high', 'breakthrough', 'success',
        'optimistic', 'positive', 'jump', 'spike', 'boost', 'recover', 'expand'
    ]
    
    # Negative keywords
    negative_words = [
        'fall', 'drop', 'plunge', 'loss', 'decline', 'bear', 'downgrade', 
        'miss', 'weak', 'low', 'concern', 'worry', 'risk', 'threat', 'crash',
        'pessimistic', 'negative', 'slump', 'sink', 'tumble', 'cut', 'layoff'
    ]
    
    pos_count = sum(1 for word in positive_words if word in text)
    neg_count = sum(1 for word in negative_words if word in text)
    
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    
    return (pos_count - neg_count) / total

@st.cache_data(ttl=3600)
def get_news_sentiment(ticker):
    """Fetch recent news and calculate sentiment score"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news[:15] if hasattr(stock, 'news') and stock.news else []
        
        if not news:
            return 0.0
        
        sentiments = []
        for article in news:
            title = article.get('title', '')
            summary = article.get('summary', '')
            combined_text = f"{title} {summary}"
            
            if combined_text:
                sentiment = simple_sentiment_analysis(combined_text)
                sentiments.append(sentiment)
        
        return np.mean(sentiments) if sentiments else 0.0
    except Exception as e:
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
            'analyst_sentiment': 0,
            'price_momentum': 0
        }
        
        # Volume trend (last 5 days vs previous 15 days)
        hist = stock.history(period="1mo")
        if not hist.empty and len(hist) > 20:
            recent_vol = hist['Volume'].iloc[-5:].mean()
            prev_vol = hist['Volume'].iloc[-20:-5].mean()
            indicators['volume_trend'] = (recent_vol - prev_vol) / prev_vol if prev_vol > 0 else 0
            
            # Price momentum (comparing recent trend)
            recent_prices = hist['Close'].iloc[-5:].values
            prev_prices = hist['Close'].iloc[-10:-5].values
            if len(recent_prices) > 0 and len(prev_prices) > 0:
                recent_avg = np.mean(recent_prices)
                prev_avg = np.mean(prev_prices)
                indicators['price_momentum'] = (recent_avg - prev_avg) / prev_avg
        
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
    except Exception as e:
        return {'rsi': 50, 'volume_trend': 0, 'analyst_sentiment': 0, 'price_momentum': 0}

def calculate_market_adjustment(sentiment_score, indicators):
    """Calculate price adjustment based on sentiment and market indicators"""
    # Weighted combination of factors with reduced weights
    adjustment = (
        sentiment_score * 0.25 +  # News sentiment (25%, reduced from 35%)
        indicators['analyst_sentiment'] * 0.20 +  # Analyst sentiment (20%, reduced from 25%)
        np.clip(indicators['volume_trend'], -0.3, 0.3) * 0.15 +  # Volume trend (15%, reduced and clipped)
        np.clip(indicators['price_momentum'], -0.3, 0.3) * 0.15  # Price momentum (15%, reduced and clipped)
    )
    
    # Scale adjustment to much more conservative percentage (-3% to +3%)
    return np.clip(adjustment * 0.03, -0.03, 0.03)

# -------------------------------
# Sidebar Input
# -------------------------------
ticker = st.sidebar.text_input(
    "Enter Stock Ticker",
    placeholder="e.g. AAPL, TSLA, NVDA"
).upper()

# Fixed training window
period_history = 10  # YEARS (LOCKED)

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

# Create market strength indicator based on volume and momentum
df_train['market_strength'] = 0.0
recent_rows = min(30, len(df_train))
df_train.loc[df_train.index[-recent_rows:], 'market_strength'] = market_adjustment

# -------------------------------
# Train Model & Forecast (PROPHET)
# -------------------------------
with st.spinner("Generating enhanced forecast..."):
    model = Prophet(
        daily_seasonality=False,  # Disable daily seasonality for less noise
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,  # Reduced from 0.15 for more stable predictions
        seasonality_mode='additive',  # Changed from multiplicative for stability
        interval_width=0.80,  # Narrower confidence intervals
        changepoint_range=0.9  # Only fit changepoints to first 90% of data
    )
    
    # Add market strength regressor with lower prior scale
    model.add_regressor('market_strength', prior_scale=0.3)
    
    model.fit(df_train)

    future = model.make_future_dataframe(periods=365)
    
    # Extend market strength to future predictions
    future['market_strength'] = 0.0
    future.loc[future.index >= len(df_train), 'market_strength'] = market_adjustment
    
    forecast = model.predict(future)
    
    # Apply much more conservative graduated adjustment to future predictions
    for i in range(365):
        idx = len(df_train) + i
        if idx < len(forecast):
            # Much slower decay factor for stability
            decay = max(0.2, 1.0 - (i / 730.0))  # Decays slowly over 2 years, min 20%
            # Reduce adjustment impact significantly
            adjustment_factor = 1 + (market_adjustment * decay * 0.3)  # 30% of original impact
            
            forecast.loc[idx, 'yhat'] *= adjustment_factor
            forecast.loc[idx, 'yhat_lower'] *= (1 + market_adjustment * decay * 0.15)
            forecast.loc[idx, 'yhat_upper'] *= (1 + market_adjustment * decay * 0.45)

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

price_15, price_30, price_90, price_365 = (
    predicted_price(15),
    predicted_price(30),
    predicted_price(90),
    predicted_price(365)
)

gain_15, gain_30, gain_90, gain_365 = (
    percent_gain(15),
    percent_gain(30),
    percent_gain(90),
    percent_gain(365)
)

# -------------------------------
# Visualization
# -------------------------------
st.subheader(f"📊 {ticker} — Past Outlook & 1-Year Forecast")

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
    name="1-Year Forecast",
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

col1, col2, col3, col4 = st.columns(4)

col1.metric("15-Day Outlook", f"${price_15:.2f}", f"{gain_15:.2f}%")
col2.metric("30-Day Outlook", f"${price_30:.2f}", f"{gain_30:.2f}%")
col3.metric("90-Day Outlook", f"${price_90:.2f}", f"{gain_90:.2f}%")
col4.metric("1-Year Outlook", f"${price_365:.2f}", f"{gain_365:.2f}%")

# -------------------------------
# Market Sentiment Indicators
# -------------------------------
st.subheader("🎯 Market Intelligence")

col1, col2, col3, col4 = st.columns(4)

# News Sentiment
sentiment_emoji = "🟢" if sentiment_score > 0.15 else "🔴" if sentiment_score < -0.15 else "🟡"
sentiment_label = "Positive" if sentiment_score > 0.15 else "Negative" if sentiment_score < -0.15 else "Neutral"
col1.metric("News Sentiment", sentiment_label, f"{sentiment_emoji}")

# Volume Trend
volume_trend = market_indicators['volume_trend']
volume_emoji = "📈" if volume_trend > 0.1 else "📉" if volume_trend < -0.1 else "➡️"
volume_label = f"{volume_trend*100:.1f}%"
col2.metric("Volume Trend", volume_label, f"{volume_emoji}")

# Price Momentum
momentum = market_indicators['price_momentum']
momentum_emoji = "🚀" if momentum > 0.02 else "⬇️" if momentum < -0.02 else "↔️"
momentum_label = f"{momentum*100:.1f}%"
col3.metric("Price Momentum", momentum_label, f"{momentum_emoji}")

# Analyst Sentiment
analyst_score = market_indicators['analyst_sentiment']
analyst_emoji = "👍" if analyst_score > 0.3 else "👎" if analyst_score < -0.3 else "👌"
analyst_map = {1.0: "Strong Buy", 0.5: "Buy", 0.0: "Hold", -0.5: "Sell", -1.0: "Strong Sell"}
analyst_label = "Hold"
for score, label in analyst_map.items():
    if abs(analyst_score - score) < 0.26:
        analyst_label = label
        break
col4.metric("Analyst Rating", analyst_label, f"{analyst_emoji}")

# -------------------------------
# Summary
# -------------------------------
st.write(
    f"The current price of **{ticker}** is approximately "
    f"**${current_price:.2f}**."
)

trend_text = "Bullish 📈" if future_up else "Bearish 📉"

# Calculate confidence based on all signals alignment
signals = [
    1 if sentiment_score > 0.1 else -1 if sentiment_score < -0.1 else 0,
    1 if volume_trend > 0.1 else -1 if volume_trend < -0.1 else 0,
    1 if momentum > 0.02 else -1 if momentum < -0.02 else 0,
    1 if analyst_score > 0.3 else -1 if analyst_score < -0.3 else 0
]
alignment = abs(sum(signals))
confidence_text = "High" if alignment >= 3 else "Moderate" if alignment >= 2 else "Low"

st.info(f"Model Outlook: **{trend_text}** over the next year (Confidence: {confidence_text}).")

# Add disclaimer
st.caption("⚠️ This forecast incorporates news sentiment, analyst ratings, volume trends, and price momentum. Trained on 10 years of data. Not financial advice. Past performance doesn't guarantee future results.")

