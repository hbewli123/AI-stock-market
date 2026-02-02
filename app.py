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
    
    positive_words = [
        'surge', 'soar', 'rally', 'gain', 'profit', 'growth', 'rise', 'bull', 
        'upgrade', 'beat', 'strong', 'record', 'high', 'breakthrough', 'success',
        'optimistic', 'positive', 'jump', 'spike', 'boost', 'recover', 'expand'
    ]
    
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
    """Get market indicators"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        indicators = {
            'volume_trend': 0,
            'analyst_sentiment': 0,
            'price_momentum': 0,
            'volatility': 0.02  # Default 2% daily volatility
        }
        
        hist = stock.history(period="3mo")
        if not hist.empty and len(hist) > 20:
            recent_vol = hist['Volume'].iloc[-5:].mean()
            prev_vol = hist['Volume'].iloc[-20:-5].mean()
            indicators['volume_trend'] = (recent_vol - prev_vol) / prev_vol if prev_vol > 0 else 0
            
            recent_prices = hist['Close'].iloc[-10:].values
            prev_prices = hist['Close'].iloc[-30:-10].values
            if len(recent_prices) > 0 and len(prev_prices) > 0:
                recent_avg = np.mean(recent_prices)
                prev_avg = np.mean(prev_prices)
                indicators['price_momentum'] = (recent_avg - prev_avg) / prev_avg
            
            # Calculate actual historical volatility
            returns = hist['Close'].pct_change().dropna()
            indicators['volatility'] = returns.std() if len(returns) > 0 else 0.02
        
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
        return {'volume_trend': 0, 'analyst_sentiment': 0, 'price_momentum': 0, 'volatility': 0.02}

# -------------------------------
# Sidebar Input
# -------------------------------
ticker = st.sidebar.text_input(
    "Enter Stock Ticker",
    placeholder="e.g. AAPL, TSLA, NVDA"
).upper()

period_history = 10  # YEARS

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
# Prepare Data for Prophet
# -------------------------------
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
df_train['ds'] = df_train['ds'].dt.tz_localize(None)

# -------------------------------
# Analyze Market Conditions
# -------------------------------
with st.spinner("Analyzing market conditions..."):
    sentiment_score = get_news_sentiment(ticker)
    market_indicators = get_market_indicators(ticker, data)
    
    # Calculate historical volatility from actual data
    recent_returns = data['Close'].pct_change().tail(60).dropna()
    historical_volatility = recent_returns.std() if len(recent_returns) > 0 else 0.02

# -------------------------------
# Prepare Data for Prophet
# -------------------------------
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
df_train['ds'] = df_train['ds'].dt.tz_localize(None)

# Calculate recent trend for comparison
recent_data = df_train.tail(90)
if len(recent_data) > 1:
    recent_trend = (recent_data['y'].iloc[-1] - recent_data['y'].iloc[0]) / recent_data['y'].iloc[0]
else:
    recent_trend = 0

# -------------------------------
# Train Prophet and Use Upper Range
# -------------------------------
with st.spinner("Generating forecast..."):
    # Train Prophet
    model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        seasonality_mode='multiplicative',
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95
    )
    
    model.fit(df_train)
    
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    
    # Extract future predictions - USE THE UPPER BAND
    future_forecast = forecast.iloc[len(df_train):].copy()
    current_price = data['Close'].iloc[-1]
    
    # Calculate ACTUAL trend from the stock's recent behavior
    last_month = data['Close'].tail(30)
    last_quarter = data['Close'].tail(90)
    last_year = data['Close'].tail(252)  # Trading days
    
    month_trend = (last_month.iloc[-1] - last_month.iloc[0]) / last_month.iloc[0] if len(last_month) > 1 else 0
    quarter_trend = (last_quarter.iloc[-1] - last_quarter.iloc[0]) / last_quarter.iloc[0] if len(last_quarter) > 1 else 0
    year_trend = (last_year.iloc[-1] - last_year.iloc[0]) / last_year.iloc[0] if len(last_year) > 1 else 0
    
    # Weight recent performance more heavily
    momentum_score = (month_trend * 0.5 + quarter_trend * 0.3 + year_trend * 0.2)
    
    # Combine with market signals
    signal_strength = (
        sentiment_score * 0.25 +
        market_indicators['analyst_sentiment'] * 0.25 +
        momentum_score * 0.35 +
        market_indicators['price_momentum'] * 0.15
    )
    
    # Use upper band but adjust based on signal strength
    final_predictions = []
    
    for i in range(len(future_forecast)):
        # Take the upper band value (this is the "optimistic" Prophet prediction)
        upper_value = future_forecast['yhat_upper'].iloc[i]
        base_value = future_forecast['yhat'].iloc[i]
        
        # Blend between base and upper based on signal strength
        # Positive signals → use more of upper band
        # Negative signals → use more of base prediction
        blend_factor = 0.7 + (signal_strength * 0.3)  # 0.4 to 1.0
        blend_factor = np.clip(blend_factor, 0.3, 1.0)
        
        predicted_value = base_value * (1 - blend_factor) + upper_value * blend_factor
        
        # Make sure it's always positive and reasonable
        predicted_value = max(predicted_value, current_price * 0.5)
        
        final_predictions.append(predicted_value)
    
    # Create forecast dataframe
    last_date = data['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=365, freq='D')
    
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Prediction': final_predictions
    })

# -------------------------------
# Metrics
# -------------------------------
def predicted_price(days):
    return final_predictions[days - 1]

def percent_gain(days):
    return ((predicted_price(days) - current_price) / current_price) * 100

price_15 = predicted_price(15)
price_30 = predicted_price(30)
price_90 = predicted_price(90)
price_365 = predicted_price(365)

gain_15 = percent_gain(15)
gain_30 = percent_gain(30)
gain_90 = percent_gain(90)
gain_365 = percent_gain(365)

# -------------------------------
# Determine trend
# -------------------------------
future_up = price_365 > current_price
future_color = "green" if future_up else "red"
past_color = "green" if data['Close'].iloc[-1] > data['Close'].iloc[0] else "red"

# -------------------------------
# Visualization
# -------------------------------
st.subheader(f"📊 {ticker} — Historical Performance & 1-Year Forecast")

fig = go.Figure()

# Actual Price
fig.add_trace(go.Scatter(
    x=data['Date'],
    y=data['Close'],
    name="Actual Price",
    line=dict(color="#2E86DE", width=2),
    mode='lines'
))

# Future Forecast - realistic with volatility
fig.add_trace(go.Scatter(
    x=forecast_df['Date'],
    y=forecast_df['Prediction'],
    name="Forecast",
    line=dict(color=future_color, width=2.5),
    mode='lines'
))

fig.update_layout(
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=500
)

st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

# -------------------------------
# Metrics
# -------------------------------
st.subheader("📈 Predicted Price Targets")

col1, col2, col3, col4 = st.columns(4)

col1.metric("15-Day", f"${price_15:.2f}", f"{gain_15:+.2f}%")
col2.metric("30-Day", f"${price_30:.2f}", f"{gain_30:+.2f}%")
col3.metric("90-Day", f"${price_90:.2f}", f"{gain_90:+.2f}%")
col4.metric("1-Year", f"${price_365:.2f}", f"{gain_365:+.2f}%")

# -------------------------------
# Market Intelligence
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
col2.metric("Volume Trend", f"{volume_trend*100:.1f}%", f"{volume_emoji}")

# Price Momentum
momentum = market_indicators['price_momentum']
momentum_emoji = "🚀" if momentum > 0.03 else "⬇️" if momentum < -0.03 else "↔️"
col3.metric("Price Momentum", f"{momentum*100:.1f}%", f"{momentum_emoji}")

# Volatility
vol_pct = historical_volatility * 100
vol_label = "High" if vol_pct > 2.5 else "Low" if vol_pct < 1.0 else "Moderate"
vol_emoji = "⚡" if vol_pct > 2.5 else "😌" if vol_pct < 1.0 else "📊"
col4.metric("Volatility", vol_label, f"{vol_pct:.2f}% {vol_emoji}")

# -------------------------------
# Summary
# -------------------------------
st.write(f"Current price of **{ticker}**: **${current_price:.2f}**")

trend_text = "Bullish 📈" if future_up else "Bearish 📉"
expected_change = abs(gain_365)

# Determine confidence based on signal alignment
signals_positive = sum([
    1 if sentiment_score > 0.1 else 0,
    1 if volume_trend > 0.1 else 0,
    1 if momentum > 0.02 else 0,
    1 if market_indicators['analyst_sentiment'] > 0.3 else 0
])

signals_negative = sum([
    1 if sentiment_score < -0.1 else 0,
    1 if volume_trend < -0.1 else 0,
    1 if momentum < -0.02 else 0,
    1 if market_indicators['analyst_sentiment'] < -0.3 else 0
])

confidence = "High" if max(signals_positive, signals_negative) >= 3 else "Moderate" if max(signals_positive, signals_negative) >= 2 else "Low"

st.info(f"**Model Outlook:** {trend_text} | **Expected 1-Year Change:** {expected_change:.1f}% | **Confidence:** {confidence}")

with st.expander("📖 Methodology"):
    st.markdown(f"""
    **How This Forecast Works:**
    
    - **Prophet Model:** Trained on 10 years of {ticker} historical data
    - **Upper Band Strategy:** Uses Prophet's optimistic range as the baseline prediction
    - **Signal Adjustment:** Blends base and upper predictions based on market strength
    - **Recent Performance Analysis:**
      - Last Month Trend: {month_trend*100:.1f}%
      - Last Quarter Trend: {quarter_trend*100:.1f}%
      - Last Year Trend: {year_trend*100:.1f}%
    - **Momentum Score:** {momentum_score*100:.1f}% (weighted recent performance)
    - **Market Signals:** News ({sentiment_score:.2f}), Analysts ({market_indicators['analyst_sentiment']:.2f})
    - **Combined Signal Strength:** {signal_strength:.2f}
    
    **Unique to {ticker}:**
    - Current price: ${current_price:.2f}
    - Historical volatility: {vol_pct:.2f}% daily
    - Momentum trend: {momentum_score*100:.1f}%
    
    The model learns {ticker}'s patterns and shows realistic upward/downward trends.
    """)

st.caption("⚠️ Prophet-based forecast using upper confidence range adjusted by market signals. Not financial advice.")
