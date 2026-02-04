import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Stock Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# FORCE LIGHT MODE + FIX FADED TEXT
# -------------------------------
st.markdown("""
<style>

/* Main app background */
.stApp {
    background-color: white !important;
    color: black !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #f8f9fa !important;
    color: black !important;
}

/* ALL text */
h1, h2, h3, h4, h5, h6, p, span, label, div, li {
    color: black !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: black !important;
}
[data-testid="stMetricDelta"] {
    color: black !important;
}

/* Inputs */
input, textarea {
    color: black !important;
    background-color: white !important;
}

/* Expander headers */
.streamlit-expanderHeader {
    color: black !important;
}

/* Tables */
table, th, td {
    color: black !important;
}

/* Plotly chart text */
.js-plotly-plot .plotly text {
    fill: black !important;
}

</style>
""", unsafe_allow_html=True)

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
    except Exception:
        return 0.0

@st.cache_data(ttl=3600)
def get_fundamental_data(ticker):
    """Get comprehensive fundamental analysis data"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        fundamentals = {
            'pe_ratio': info.get('trailingPE', 0) or 0,
            'forward_pe': info.get('forwardPE', 0) or 0,
            'peg_ratio': info.get('pegRatio', 0) or 0,
            'price_to_book': info.get('priceToBook', 0) or 0,
            'debt_to_equity': info.get('debtToEquity', 0) or 0,
            'current_ratio': info.get('currentRatio', 0) or 0,
            'roe': info.get('returnOnEquity', 0) or 0,
            'profit_margin': info.get('profitMargins', 0) or 0,
            'revenue_growth': info.get('revenueGrowth', 0) or 0,
            'earnings_growth': info.get('earningsGrowth', 0) or 0,
            'free_cash_flow': info.get('freeCashflow', 0) or 0,
            'operating_margin': info.get('operatingMargins', 0) or 0,
            'market_cap': info.get('marketCap', 0) or 0,
            'book_value': info.get('bookValue', 0) or 0,
            'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', 0) or 0,
        }
        
        # Earnings surprise
        try:
            earnings = stock.earnings_dates
            if earnings is not None and not earnings.empty:
                recent_earnings = earnings.head(4)
                eps_surprise = recent_earnings['EPS Estimate'].sub(recent_earnings['Reported EPS']).mean()
                fundamentals['eps_surprise'] = eps_surprise if not pd.isna(eps_surprise) else 0
            else:
                fundamentals['eps_surprise'] = 0
        except:
            fundamentals['eps_surprise'] = 0
        
        # Balance sheet strength
        try:
            balance_sheet = stock.balance_sheet
            if balance_sheet is not None and not balance_sheet.empty:
                total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else 0
                total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else 0
                fundamentals['asset_to_liability'] = total_assets / total_liabilities if total_liabilities > 0 else 1
            else:
                fundamentals['asset_to_liability'] = 1
        except:
            fundamentals['asset_to_liability'] = 1
        
        # Cash flow
        try:
            cashflow = stock.cashflow
            if cashflow is not None and not cashflow.empty:
                fundamentals['operating_cashflow'] = cashflow.loc['Operating Cash Flow'].iloc[0]
            else:
                fundamentals['operating_cashflow'] = 0
        except:
            fundamentals['operating_cashflow'] = 0
        
        return fundamentals
        
    except Exception:
        return {}

def calculate_fundamental_score(f):
    score = 0
    
    if 10 < f['pe_ratio'] < 25: score += 0.15
    elif f['pe_ratio'] >= 35: score -= 0.05
    
    if 0 < f['peg_ratio'] < 1: score += 0.15
    
    if f['roe'] > 0.15: score += 0.15
    
    if f['profit_margin'] > 0.2: score += 0.1
    
    if f['revenue_growth'] > 0.1: score += 0.15
    elif f['revenue_growth'] < 0: score -= 0.1
    
    if f['earnings_growth'] > 0.1: score += 0.15
    elif f['earnings_growth'] < 0: score -= 0.1
    
    if f['debt_to_equity'] < 50: score += 0.1
    
    if f['current_ratio'] > 1: score += 0.05
    
    if f.get('eps_surprise',0) > 0: score += 0.1
    
    return np.clip(score, -0.5, 0.5)

@st.cache_data(ttl=3600)
def get_market_indicators(ticker, data):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        indicators = {'volume_trend':0,'analyst_sentiment':0,'price_momentum':0,'volatility':0.02}
        
        if not hist.empty:
            returns = hist['Close'].pct_change().dropna()
            indicators['volatility'] = returns.std() if len(returns)>0 else 0.02
        
        info = stock.info
        if 'recommendationKey' in info:
            rec_map = {'strong_buy':1,'buy':0.5,'hold':0,'sell':-0.5,'strong_sell':-1}
            indicators['analyst_sentiment'] = rec_map.get(info['recommendationKey'],0)
        
        return indicators
    except:
        return {'volume_trend':0,'analyst_sentiment':0,'price_momentum':0,'volatility':0.02}

# -------------------------------
# Sidebar Input
# -------------------------------
ticker = st.sidebar.text_input("Enter Stock Ticker", placeholder="AAPL, TSLA, NVDA").upper()
if not ticker:
    st.stop()

# -------------------------------
# Fetch Historical Data
# -------------------------------
data = yf.download(ticker, period="10y", multi_level_index=False)
data.reset_index(inplace=True)
current_price = data['Close'].iloc[-1]

# -------------------------------
# Analyze Market Conditions
# -------------------------------
sentiment_score = get_news_sentiment(ticker)
market_indicators = get_market_indicators(ticker, data)

recent_returns = data['Close'].pct_change().tail(60).dropna()
historical_volatility = recent_returns.std() if len(recent_returns) > 0 else 0.02

fundamentals = get_fundamental_data(ticker)
fundamental_score = calculate_fundamental_score(fundamentals)

# -------------------------------
# Momentum
# -------------------------------
last_month = data['Close'].tail(30)
last_quarter = data['Close'].tail(90)
last_year = data['Close'].tail(252)

month_trend = (last_month.iloc[-1]-last_month.iloc[0])/last_month.iloc[0]
quarter_trend = (last_quarter.iloc[-1]-last_quarter.iloc[0])/last_quarter.iloc[0]
year_trend = (last_year.iloc[-1]-last_year.iloc[0])/last_year.iloc[0]

momentum_score = month_trend*0.5 + quarter_trend*0.3 + year_trend*0.2

signal_strength = (
    sentiment_score * 0.20 +
    market_indicators['analyst_sentiment'] * 0.20 +
    momentum_score * 0.25 +
    fundamental_score * 0.25
)

# -------------------------------
# Forecast Simulation
# -------------------------------
base_annual_growth = max(0.10, abs(signal_strength)*0.4)
daily_growth = (1+base_annual_growth)**(1/365)-1

np.random.seed(42)
final_predictions = []
for i in range(365):
    base = current_price*((1+daily_growth)**(i+1))
    daily_change = np.random.normal(0, historical_volatility)
    final_predictions.append(base*(1+daily_change))

future_dates = pd.date_range(start=data['Date'].iloc[-1]+timedelta(days=1), periods=365)

forecast_df = pd.DataFrame({"Date":future_dates,"Prediction":final_predictions})

# -------------------------------
# Plot
# -------------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data['Date'], y=data['Close'],
    name="Actual Price",
    line=dict(color="#2E86DE")
))

fig.add_trace(go.Scatter(
    x=forecast_df['Date'], y=forecast_df['Prediction'],
    name="Future Forecast",
    line=dict(color="green" if final_predictions[-1] > current_price else "red")
))

fig.update_layout(
    template="plotly_white",   # <-- LIGHT MODE FIX
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="black"),
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Metrics
# -------------------------------
def predicted_price(days): return final_predictions[days-1]
def percent_gain(days): return ((predicted_price(days)-current_price)/current_price)*100

st.subheader("📈 Predicted Price Targets")
c1,c2,c3,c4 = st.columns(4)
c1.metric("15-Day", f"${predicted_price(15):.2f}", f"{percent_gain(15):+.2f}%")
c2.metric("30-Day", f"${predicted_price(30):.2f}", f"{percent_gain(30):+.2f}%")
c3.metric("90-Day", f"${predicted_price(90):.2f}", f"{percent_gain(90):+.2f}%")
c4.metric("1-Year", f"${predicted_price(365):.2f}", f"{percent_gain(365):+.2f}%")

# -------------------------------
# Summary
# -------------------------------
trend_text = "Bullish 📈" if final_predictions[-1] > current_price else "Bearish 📉"
st.info(f"Model Outlook: {trend_text} | Signal Strength: {signal_strength:.2f}")
st.caption("⚠️ Optimistic growth model. Not financial advice.")

