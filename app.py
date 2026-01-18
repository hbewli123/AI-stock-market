import streamlit as st
import yfinance as yf

st.title("My Science Project: AI Stock Finder")
ticker = st.text_input("Type a stock (like AAPL):", "AAPL")
data = yf.download(ticker, period="1y")
st.line_chart(data['Close'])
st.write("This shows how the price changed over 1 year!")
