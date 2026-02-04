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

/* Main app */
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

/* Metric values */
[data-testid="stMetricV]()

