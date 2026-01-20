import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Set page
st.title("Stock Predictor with News Sentiment")

# Input ticker
ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()

if not ticker:
    st.stop()

# Fetch price data
period_history = 2
data = yf.download(ticker, period=f"{period_history}y", progress=False)

if data.empty:
    st.error("No price data found for ticker.")
    st.stop()

data.reset_index(inplace=True)

# Fetch news headlines for last 60 days using NewsAPI (replace with your API key)
NEWS_API_KEY = "YOUR_NEWSAPI_KEY"  # You must get your own key from https://newsapi.org/
def fetch_news(ticker):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={ticker}&"
        f"from={(pd.Timestamp.today() - pd.Timedelta(days=60)).strftime('%Y-%m-%d')}&"
        f"sortBy=publishedAt&"
        f"language=en&"
        f"apiKey={NEWS_API_KEY}"
    )
    response = requests.get(url)
    if response.status_code != 200:
        return []
    articles = response.json().get("articles", [])
    return articles

news_articles = fetch_news(ticker)
st.write(f"Fetched {len(news_articles)} news articles")

# Prepare sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Aggregate sentiment scores by date
sentiment_by_date = {}

for article in news_articles:
    date = pd.to_datetime(article["publishedAt"]).date()
    score = analyzer.polarity_scores(article["title"])["compound"]
    if date not in sentiment_by_date:
        sentiment_by_date[date] = []
    sentiment_by_date[date].append(score)

# Average sentiment per day
sentiment_df = pd.DataFrame([
    {"ds": pd.to_datetime(date), "sentiment": sum(scores)/len(scores)}
    for date, scores in sentiment_by_date.items()
])

# Merge with price data dates
price_df = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
price_df['ds'] = pd.to_datetime(price_df['ds'])
price_df = price_df.sort_values('ds')

# Merge sentiment into price_df (fill missing sentiment with 0)
price_df = price_df.merge(sentiment_df, on='ds', how='left')
price_df['sentiment'] = price_df['sentiment'].fillna(0)

# Build Prophet model with sentiment regressor
model = Prophet()
model.add_regressor('sentiment')

with st.spinner("Training Prophet with sentiment regressor..."):
    model.fit(price_df)

# Make future dataframe and fill future sentiment with 0 (no news)
future = model.make_future_dataframe(periods=30)
future = future.merge(sentiment_df, on='ds', how='left')
future['sentiment'] = future['sentiment'].fillna(0)

forecast = model.predict(future)

# Plot results
import plotly.graph_objs as go

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=price_df['ds'],
    y=price_df['y'],
    mode='lines',
    name='Actual Price'
))

fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat'],
    mode='lines',
    name='Forecast with Sentiment'
))

st.plotly_chart(fig, use_container_width=True)

# Show sentiment examples
st.subheader("Sample News Sentiment Scores")

for i, article in enumerate(news_articles[:5]):
    headline = article["title"]
    date = article["publishedAt"]
    score = analyzer.polarity_scores(headline)["compound"]
    st.write(f"{date}: **{headline}** (Sentiment: {score:.2f})")

