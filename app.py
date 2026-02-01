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
def get_market_indicators(ticker, data):
    """Get additional market indicators and technical analysis"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        indicators = {
            'volume_trend': 0,
            'analyst_sentiment': 0,
            'price_momentum': 0,
            'rsi': 50,
            'ma_signal': 0,
            'volatility': 0
        }
        
        # Volume trend
        hist = stock.history(period="3mo")
        if not hist.empty and len(hist) > 20:
            recent_vol = hist['Volume'].iloc[-5:].mean()
            prev_vol = hist['Volume'].iloc[-20:-5].mean()
            indicators['volume_trend'] = (recent_vol - prev_vol) / prev_vol if prev_vol > 0 else 0
            
            # Price momentum
            recent_prices = hist['Close'].iloc[-10:].values
            prev_prices = hist['Close'].iloc[-30:-10].values
            if len(recent_prices) > 0 and len(prev_prices) > 0:
                recent_avg = np.mean(recent_prices)
                prev_avg = np.mean(prev_prices)
                indicators['price_momentum'] = (recent_avg - prev_avg) / prev_avg
            
            # Volatility
            returns = hist['Close'].pct_change().dropna()
            indicators['volatility'] = returns.std() if len(returns) > 0 else 0
        
        # Calculate RSI
        if len(data) >= 14:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['rsi'] = rsi.iloc[-1] if not rsi.empty else 50
        
        # Moving Average Signal
        if len(data) >= 50:
            ma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
            ma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
            current_price = data['Close'].iloc[-1]
            
            # Bullish if price > MA20 > MA50, bearish if opposite
            if current_price > ma_20 > ma_50:
                indicators['ma_signal'] = 0.5
            elif current_price < ma_20 < ma_50:
                indicators['ma_signal'] = -0.5
        
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
        return {'volume_trend': 0, 'analyst_sentiment': 0, 'price_momentum': 0, 
                'rsi': 50, 'ma_signal': 0, 'volatility': 0}

def calculate_composite_score(sentiment_score, indicators):
    """Calculate comprehensive market score combining all factors"""
    # Technical analysis weight
    technical_score = (
        indicators['ma_signal'] * 0.3 +
        ((indicators['rsi'] - 50) / 50) * 0.2 +  # Normalize RSI to -1 to 1
        np.clip(indicators['price_momentum'], -0.5, 0.5) * 0.25
    )
    
    # Fundamental/sentiment weight
    fundamental_score = (
        sentiment_score * 0.4 +
        indicators['analyst_sentiment'] * 0.35 +
        np.clip(indicators['volume_trend'], -0.3, 0.3) * 0.25
    )
    
    # Combine (60% technical, 40% fundamental for stocks)
    composite = technical_score * 0.6 + fundamental_score * 0.4
    
    # Adjust based on volatility (reduce impact if highly volatile)
    volatility_dampener = 1.0 - min(indicators['volatility'] * 10, 0.5)
    
    return composite * volatility_dampener

# -------------------------------
# Sidebar Input
# -------------------------------
ticker = st.sidebar.text_input(
    "Enter Stock Ticker",
    placeholder="e.g. AAPL, TSLA, NVDA"
).upper()

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
with st.spinner("Analyzing market sentiment and technical indicators..."):
    sentiment_score = get_news_sentiment(ticker)
    market_indicators = get_market_indicators(ticker, data)
    composite_score = calculate_composite_score(sentiment_score, market_indicators)

# -------------------------------
# Prepare Data for Prophet
# -------------------------------
df_train = data[['Date', 'Close']].rename(
    columns={"Date": "ds", "Close": "y"}
)
df_train['ds'] = df_train['ds'].dt.tz_localize(None)

# Add technical indicators as regressors
df_train['ma_7'] = data['Close'].rolling(window=7).mean().fillna(method='bfill')
df_train['ma_21'] = data['Close'].rolling(window=21).mean().fillna(method='bfill')
df_train['volume_norm'] = (data['Volume'] - data['Volume'].mean()) / data['Volume'].std()
df_train['volume_norm'] = df_train['volume_norm'].fillna(0)

# Add composite score to recent data
df_train['market_score'] = 0.0
recent_rows = min(60, len(df_train))
df_train.loc[df_train.index[-recent_rows:], 'market_score'] = composite_score

# -------------------------------
# Train Prophet Model with Custom Analysis
# -------------------------------
with st.spinner("Training hybrid AI model..."):
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_mode='additive',
        interval_width=0.85
    )
    
    # Add our custom regressors
    model.add_regressor('ma_7', prior_scale=0.3)
    model.add_regressor('ma_21', prior_scale=0.4)
    model.add_regressor('volume_norm', prior_scale=0.2)
    model.add_regressor('market_score', prior_scale=0.5)
    
    model.fit(df_train)

    # Create future dataframe
    future = model.make_future_dataframe(periods=365)
    
    # Extend regressors to future
    last_ma7 = df_train['ma_7'].iloc[-1]
    last_ma21 = df_train['ma_21'].iloc[-1]
    last_volume = df_train['volume_norm'].iloc[-1]
    
    future['ma_7'] = last_ma7
    future['ma_21'] = last_ma21
    future['volume_norm'] = last_volume
    future['market_score'] = 0.0
    
    # Apply market score to future predictions with decay
    for i in range(365):
        idx = len(df_train) + i
        if idx < len(future):
            decay = max(0.2, 1.0 - (i / 365.0))
            future.loc[idx, 'market_score'] = composite_score * decay
    
    # Generate forecast
    forecast = model.predict(future)

# Create our own adjustment layer on top of Prophet
with st.spinner("Applying technical analysis overlay..."):
    current_price = data['Close'].iloc[-1]
    
    for i in range(365):
        idx = len(df_train) + i
        if idx < len(forecast):
            # Apply gradual trend adjustment based on composite score
            days_out = i + 1
            decay = max(0.15, 1.0 - (days_out / 547.5))  # 1.5 years decay
            
            # Conservative adjustment (±3% max impact)
            adjustment = 1 + (composite_score * 0.03 * decay)
            
            forecast.loc[idx, 'yhat'] *= adjustment
            forecast.loc[idx, 'yhat_lower'] *= (adjustment * 0.8)
            forecast.loc[idx, 'yhat_upper'] *= (adjustment * 1.2)
            
            # Smooth extreme predictions
            if abs(forecast.loc[idx, 'yhat'] - current_price) / current_price > 0.5:
                # Cap at 50% change
                max_val = current_price * 1.5
                min_val = current_price * 0.5
                forecast.loc[idx, 'yhat'] = np.clip(forecast.loc[idx, 'yhat'], min_val, max_val)

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
st.subheader(f"📊 {ticker} — Hybrid AI Analysis & 1-Year Forecast")

fig = go.Figure()

# Actual Price
fig.add_trace(go.Scatter(
    x=df_train['ds'],
    y=df_train['y'],
    name="Actual Price",
    line=dict(color="#1f77b4", width=1.5)
))

# Prophet Base Model
fig.add_trace(go.Scatter(
    x=forecast['ds'].iloc[:len(df_train)],
    y=forecast['yhat'].iloc[:len(df_train)],
    name="Model Fit",
    line=dict(color=past_color, width=2, dash='dash'),
    opacity=0.6
))

# Enhanced Forecast (Prophet + Our Analysis)
fig.add_trace(go.Scatter(
    x=forecast['ds'].iloc[len(df_train) - 1:],
    y=forecast['yhat'].iloc[len(df_train) - 1:],
    name="Hybrid Forecast",
    line=dict(color=future_color, width=2.5)
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
    fillcolor='rgba(68, 68, 68, 0.15)',
    name='Confidence Range',
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
# Technical Analysis Dashboard
# -------------------------------
st.subheader("🎯 Technical Analysis & Market Intelligence")

col1, col2, col3, col4, col5 = st.columns(5)

# RSI Indicator
rsi_val = market_indicators['rsi']
rsi_emoji = "🔥" if rsi_val > 70 else "❄️" if rsi_val < 30 else "✅"
rsi_label = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
col1.metric("RSI (14)", f"{rsi_val:.1f}", rsi_label + f" {rsi_emoji}")

# Moving Average Signal
ma_signal = market_indicators['ma_signal']
ma_emoji = "📈" if ma_signal > 0.2 else "📉" if ma_signal < -0.2 else "➡️"
ma_label = "Bullish" if ma_signal > 0.2 else "Bearish" if ma_signal < -0.2 else "Neutral"
col2.metric("MA Signal", ma_label, f"{ma_emoji}")

# News Sentiment
sentiment_emoji = "🟢" if sentiment_score > 0.15 else "🔴" if sentiment_score < -0.15 else "🟡"
sentiment_label = "Positive" if sentiment_score > 0.15 else "Negative" if sentiment_score < -0.15 else "Neutral"
col3.metric("News Sentiment", sentiment_label, f"{sentiment_emoji}")

# Volume Trend
volume_trend = market_indicators['volume_trend']
volume_emoji = "📊" if volume_trend > 0.1 else "📉" if volume_trend < -0.1 else "➡️"
volume_label = f"{volume_trend*100:.1f}%"
col4.metric("Volume Trend", volume_label, f"{volume_emoji}")

# Analyst Rating
analyst_score = market_indicators['analyst_sentiment']
analyst_emoji = "👍" if analyst_score > 0.3 else "👎" if analyst_score < -0.3 else "👌"
analyst_map = {1.0: "Strong Buy", 0.5: "Buy", 0.0: "Hold", -0.5: "Sell", -1.0: "Strong Sell"}
analyst_label = "Hold"
for score, label in analyst_map.items():
    if abs(analyst_score - score) < 0.26:
        analyst_label = label
        break
col5.metric("Analyst Rating", analyst_label, f"{analyst_emoji}")

# -------------------------------
# Composite Score Indicator
# -------------------------------
st.subheader("📊 Composite Market Score")

# Visual indicator
score_normalized = (composite_score + 1) / 2 * 100  # Convert -1 to 1 → 0 to 100
score_color = "green" if composite_score > 0.2 else "red" if composite_score < -0.2 else "orange"

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.progress(score_normalized / 100)
    st.caption(f"Overall Market Score: {composite_score:.2f} (Range: -1.0 to +1.0)")
with col2:
    score_label = "Strongly Bullish" if composite_score > 0.4 else \
                  "Bullish" if composite_score > 0.2 else \
                  "Bearish" if composite_score < -0.2 else \
                  "Strongly Bearish" if composite_score < -0.4 else "Neutral"
    st.metric("Signal", score_label)
with col3:
    volatility_pct = market_indicators['volatility'] * 100
    vol_label = "High" if volatility_pct > 3 else "Low" if volatility_pct < 1 else "Moderate"
    st.metric("Volatility", vol_label, f"{volatility_pct:.2f}%")

# -------------------------------
# Summary
# -------------------------------
st.write(
    f"The current price of **{ticker}** is **${current_price:.2f}**."
)

trend_text = "Bullish 📈" if future_up else "Bearish 📉"

# Calculate confidence based on signal alignment
signals = [
    1 if sentiment_score > 0.1 else -1 if sentiment_score < -0.1 else 0,
    1 if volume_trend > 0.1 else -1 if volume_trend < -0.1 else 0,
    1 if market_indicators['price_momentum'] > 0.02 else -1 if market_indicators['price_momentum'] < -0.02 else 0,
    1 if analyst_score > 0.3 else -1 if analyst_score < -0.3 else 0,
    1 if ma_signal > 0.2 else -1 if ma_signal < -0.2 else 0,
    1 if rsi_val < 30 else -1 if rsi_val > 70 else 0
]
alignment = abs(sum(signals))
confidence_text = "High" if alignment >= 4 else "Moderate" if alignment >= 2 else "Low"

st.info(f"📊 **Hybrid Model Outlook:** {trend_text} over the next year | **Confidence:** {confidence_text} | **Composite Score:** {composite_score:.2f}")

# Methodology explanation
with st.expander("ℹ️ How This Works"):
    st.markdown("""
    **Hybrid AI Model = Prophet + Custom Technical Analysis**
    
    **Prophet Component:**
    - Time series forecasting with trend and seasonality
    - Trained on 10 years of historical data
    - Accounts for market cycles and patterns
    
    **Our Custom Analysis Layer:**
    - **Technical Indicators:** RSI, Moving Averages (7, 21-day), Volume trends
    - **Sentiment Analysis:** Real-time news keyword analysis
    - **Fundamental Data:** Analyst ratings, price momentum
    - **Composite Scoring:** Weighted combination of all signals
    
    **Final Prediction:** Prophet forecast adjusted by composite market score with time decay.
    Adjustments are capped at ±3% to prevent extreme volatility.
    """)

st.caption("⚠️ Hybrid model combining Prophet time series + technical analysis + market sentiment. Trained on 10 years of data. Not financial advice.")
