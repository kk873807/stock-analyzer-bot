# stock_bot_web.py

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from textblob import TextBlob
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

# Default global values to avoid reference errors
news_score = 0
risk_score = 5  # neutral
st.set_page_config(page_title="Stock Analyzer Bot", page_icon="📈", layout="wide")

st.markdown("""
    <style>
        .big-font {
            font-size:30px !important;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #2c7be5;
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">📊 Stock Analyzer Bot</p>', unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date, group_by="column", auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    data["MA50"] = data["Close"].rolling(window=50).mean()
    data["MA200"] = data["Close"].rolling(window=200).mean()
    return data

# ========== PHASE 1: Streamlit Setup & Portfolio Load ==========
with open("portfolio_data.json", "r") as f:
    portfolio_json = json.load(f)

if not portfolio_json or "stocks" not in portfolio_json or len(portfolio_json["stocks"]) == 0:
    st.error("No portfolio data found. Please add stocks to portfolio_data.json.")
    st.stop()

selected_stock = st.selectbox("Select a stock", portfolio_json["stocks"], format_func=lambda x: x["name"])
stock_symbol = selected_stock["symbol"]
st.set_page_config(page_title="📊 Stock Analyzer Bot", layout="centered")
st.title("📈 Stock Analysis Bot")
st.subheader("📁 Portfolio Overview")
total_stocks = len(portfolio_json["stocks"])
symbols = [stock["symbol"] for stock in portfolio_json["stocks"]]
st.write(f"Total Stocks: **{total_stocks}**")
st.write("Symbols in Portfolio:", ", ".join(symbols))

# ========== PHASE 2: Date Selection & Data Fetching ==========

date_range = st.date_input("📅 Select date range", [date.today() - timedelta(days=180), date.today()])
if len(date_range) != 2:
    st.warning("Please select a valid start and end date.")
    st.stop()
start_date, end_date = date_range

stock_data = fetch_stock_data(stock_symbol, start_date, end_date)


# 🔧 Flatten MultiIndex columns if present
if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data.columns = [col[0] for col in stock_data.columns]

if stock_data.empty:
    st.error("No data found for selected date range.")
    st.stop()

stock_data["MA50"] = stock_data["Close"].rolling(window=50).mean()
stock_data["MA200"] = stock_data["Close"].rolling(window=200).mean()


# ========== PHASE 3: Stock Price Visualization ==========

fig = px.line(stock_data.reset_index(), x="Date", y=["Close", "MA50", "MA200"],
              labels={"value": "Price", "variable": "Metric"},
              title=f"{stock_symbol} Price & Moving Averages")
st.plotly_chart(fig)

# ========== PHASE 4: Metrics and Investment Suggestion ==========

st.subheader("📊 Stock Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current Price", f"₹{stock_data['Close'].iloc[-1]:.2f}")
with col2:
    st.metric("52 Week High", f"₹{stock_data['High'].max():.2f}")
with col3:
    st.metric("52 Week Low", f"₹{stock_data['Low'].min():.2f}")

st.subheader("🤖 Investment Suggestion")
ma50 = stock_data['MA50'].iloc[-1]
ma200 = stock_data['MA200'].iloc[-1]
price = stock_data['Close'].iloc[-1]

if ma50 > ma200 and price > ma50:
    st.success("📈 Uptrend detected. Consider buying or holding.")
elif ma50 < ma200 and price < ma50:
    st.warning("📉 Downtrend detected. Consider avoiding or selling.")
else:
    st.info("⚖️ No strong trend detected. Monitor closely.")

# ========== PHASE 5: Macro Analysis ==========

macro_data = {
    "India": {"GDP_growth": 6.5, "Inflation": 5.4, "Interest_rate": 6.5, "Geopolitical_risk": "Low"},
    "United States": {"GDP_growth": 2.1, "Inflation": 3.4, "Interest_rate": 5.25, "Geopolitical_risk": "Moderate"},
    "China": {"GDP_growth": 5.0, "Inflation": 2.2, "Interest_rate": 3.6, "Geopolitical_risk": "High"}
}

def get_macro_analysis(symbol):
    if ".NS" in symbol:
        country = "India"
    elif symbol.upper() in ["AAPL", "TSLA", "MSFT", "AMZN"]:
        country = "United States"
    elif symbol.upper() in ["BABA", "JD"]:
        country = "China"
    else:
        country = "United States"

    data = macro_data.get(country)
    if not data:
        return {"country": country, "summary": ["No macro data available."]}

    summary = []

    if data["GDP_growth"] > 3:
        summary.append("✅ Strong GDP growth.")
    else:
        summary.append("⚠️ Weak GDP growth.")

    if data["Inflation"] > 6:
        summary.append("❌ High inflation risk.")
    elif data["Inflation"] > 4:
        summary.append("⚠️ Slightly elevated inflation.")
    else:
        summary.append("✅ Stable inflation.")

    if data["Interest_rate"] > 5:
        summary.append("⚠️ High interest rates.")
    else:
        summary.append("✅ Investor-friendly interest rates.")

    risk = data["Geopolitical_risk"]
    if risk == "Low":
        summary.append("✅ Low geopolitical risk.")
    elif risk == "Moderate":
        summary.append("⚠️ Moderate geopolitical risk.")
    else:
        summary.append("❌ High geopolitical risk.")

    return {"country": country, "summary": summary}

st.subheader("🌐 Macro Economic Insight")
macro = get_macro_analysis(stock_symbol)
st.markdown(f"**Country:** {macro['country']}")
for point in macro["summary"]:
    st.markdown(f"- {point}")

# ========== PHASE 6: News Sentiment Analysis ==========

API_KEY = "pub_aeb8f6aeaa614154856ab1817ffe14dd"

def get_news_sentiment(stock_name):
    url = f"https://newsdata.io/api/1/news?apikey={API_KEY}&q={stock_name}&language=en&country=in"
    try:
        response = requests.get(url)
        news_data = response.json()

        if news_data.get("status") != "success":
            return {"error": "Failed to fetch news."}

        articles = news_data.get("results", [])[:5]
        pos = neg = neu = 0
        detailed = []

        for article in articles:
            title = article.get("title", "")
            blob = TextBlob(title)
            polarity = blob.sentiment.polarity

            if polarity > 0.1:
                sentiment = "Positive"
                pos += 1
            elif polarity < -0.1:
                sentiment = "Negative"
                neg += 1
            else:
                sentiment = "Neutral"
                neu += 1

            detailed.append((title, sentiment))

        return {
            "Positive": pos,
            "Negative": neg,
            "Neutral": neu,
            "Details": detailed
        }

    except Exception as e:
        return {"error": str(e)}

st.subheader("📰 News Sentiment")
sentiment = get_news_sentiment(selected_stock["name"])
if "error" in sentiment:
    st.error(sentiment["error"])
else:
    st.write(f"**Positive:** {sentiment['Positive']} | **Negative:** {sentiment['Negative']} | **Neutral:** {sentiment['Neutral']}")
    for title, senti in sentiment["Details"]:
        st.markdown(f"- **{senti}**: {title}")

# ========== PHASE 7: Peer Comparison ==========

st.subheader("👥 Peer Comparison")
peer_dict = {
    "INFY.NS": ["TCS.NS", "WIPRO.NS", "TECHM.NS"],
    "TCS.NS": ["INFY.NS", "WIPRO.NS", "HCLTECH.NS"],
    "RELIANCE.NS": ["ONGC.NS", "IOC.NS", "BPCL.NS"],
    "AAPL": ["MSFT", "GOOGL", "AMZN"],
    "TSLA": ["F", "GM", "NIO"]
}

peers = peer_dict.get(stock_symbol.upper(), [])
peer_data = {}

for peer in peers:
    try:
        p_info = yf.Ticker(peer).info
        peer_data[peer] = {
            "Name": p_info.get("shortName", peer),
            "Price": p_info.get("regularMarketPrice", "NA"),
            "P/E Ratio": p_info.get("trailingPE", "NA"),
            "Market Cap": p_info.get("marketCap", "NA")
        }
    except Exception as e:
        peer_data[peer] = {"Name": peer, "Error": str(e)}

if peer_data:
    st.dataframe(pd.DataFrame.from_dict(peer_data, orient="index"))
else:
    st.info("No peers found for comparison.")

st.markdown("---")
st.markdown("📘 **Disclaimer**: This tool is for educational purposes only. Please do your own research before investing.")

# ========== PHASE 8: Risk Assessment ==========

st.subheader("⚠️ Risk Assessment")

# 1. Volatility Score
daily_returns = stock_data["Close"].pct_change().dropna()
volatility = daily_returns.std()
if volatility < 0.01:
    vol_score = 2
elif volatility < 0.02:
    vol_score = 5
else:
    vol_score = 8

# 2. P/E Ratio Score
try:
    pe_ratio = yf.Ticker(stock_symbol).info.get("trailingPE", None)
    if pe_ratio is None:
        pe_score = 5
        pe_text = "P/E data not available"
    elif pe_ratio < 15:
        pe_score = 2
        pe_text = f"Low P/E ({pe_ratio:.2f})"
    elif pe_ratio < 30:
        pe_score = 5
        pe_text = f"Moderate P/E ({pe_ratio:.2f})"
    else:
        pe_score = 8
        pe_text = f"High P/E ({pe_ratio:.2f})"
except:
    pe_score = 5
    pe_text = "Error fetching P/E"

# 3. Drawdown Score
max_price = stock_data["Close"].max()
current_price = stock_data["Close"].iloc[-1]
drawdown = (max_price - current_price) / max_price
if drawdown < 0.1:
    dd_score = 2
elif drawdown < 0.2:
    dd_score = 5
else:
    dd_score = 8

# Final Risk Score (average of 3)
risk_score = round((vol_score + pe_score + dd_score) / 3, 1)

# 📊 Display Breakdown
st.markdown(f"**Volatility:** {volatility:.4f} → Score: {vol_score}/10")
st.markdown(f"**{pe_text}** → Score: {pe_score}/10")
st.markdown(f"**Drawdown:** {drawdown:.2%} → Score: {dd_score}/10")

# 🚦 Display Final Risk
if risk_score < 4:
    st.success(f"🟢 **Low Risk** — Score: {risk_score}/10")
elif risk_score < 7:
    st.warning(f"🟡 **Moderate Risk** — Score: {risk_score}/10")
else:
    st.error(f"🔴 **High Risk** — Score: {risk_score}/10")

# ========== PHASE 9: Technical Indicators ==========
st.subheader("📉 Technical Indicators")

# Ensure enough data
if len(stock_data) < 100:
    st.warning("Not enough historical data to compute indicators.")
else:
    # RSI
    rsi_indicator = RSIIndicator(close=stock_data["Close"], window=14)
    stock_data["RSI"] = rsi_indicator.rsi()

    # MACD
    macd = MACD(close=stock_data["Close"])
    stock_data["MACD"] = macd.macd()
    stock_data["Signal"] = macd.macd_signal()

    # Bollinger Bands
    bb_indicator = BollingerBands(close=stock_data["Close"])
    stock_data["BB_High"] = bb_indicator.bollinger_hband()
    stock_data["BB_Low"] = bb_indicator.bollinger_lband()

    # Plot RSI
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=stock_data.index, y=stock_data["RSI"], name="RSI"))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    fig_rsi.update_layout(title="RSI (Relative Strength Index)", height=300)
    st.plotly_chart(fig_rsi)

    # Plot MACD
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=stock_data.index, y=stock_data["MACD"], name="MACD"))
    fig_macd.add_trace(go.Scatter(x=stock_data.index, y=stock_data["Signal"], name="Signal Line"))
    fig_macd.update_layout(title="MACD", height=300)
    st.plotly_chart(fig_macd)

    # Plot Bollinger Bands
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=stock_data.index, y=stock_data["Close"], name="Close Price"))
    fig_bb.add_trace(go.Scatter(x=stock_data.index, y=stock_data["BB_High"], name="BB High", line=dict(dash="dot")))
    fig_bb.add_trace(go.Scatter(x=stock_data.index, y=stock_data["BB_Low"], name="BB Low", line=dict(dash="dot")))
    fig_bb.update_layout(title="Bollinger Bands", height=300)
    st.plotly_chart(fig_bb)

# ========== PHASE 10: Final Recommendation ==========
st.subheader("🧠 Final Buy/Sell Recommendation")

recommendation_score = 0
reasons = []

# Use only if indicators were calculated (data >= 100)
if "RSI" in stock_data.columns:

    # RSI condition
    latest_rsi = stock_data["RSI"].iloc[-1]
    if latest_rsi < 30:
        recommendation_score += 1
        reasons.append("RSI indicates the stock is oversold (good buy opportunity).")
    elif latest_rsi > 70:
        recommendation_score -= 1
        reasons.append("RSI indicates the stock is overbought (might drop soon).")

    # MACD condition
    latest_macd = stock_data["MACD"].iloc[-1]
    latest_signal = stock_data["Signal"].iloc[-1]
    if latest_macd > latest_signal:
        recommendation_score += 1
        reasons.append("MACD shows bullish momentum.")
    else:
        recommendation_score -= 1
        reasons.append("MACD shows bearish momentum.")

    # Bollinger Bands
    latest_close = stock_data["Close"].iloc[-1]
    latest_bb_low = stock_data["BB_Low"].iloc[-1]
    latest_bb_high = stock_data["BB_High"].iloc[-1]

    if latest_close < latest_bb_low:
        recommendation_score += 1
        reasons.append("Price is near lower Bollinger Band (may rebound).")
    elif latest_close > latest_bb_high:
        recommendation_score -= 1
        reasons.append("Price is near upper Bollinger Band (may fall).")

# Add risk score
if 'risk_score' in locals():
    if risk_score < 3.5:
        recommendation_score += 1
        reasons.append("Low risk score indicates relatively safe investment.")
    elif risk_score > 6:
        recommendation_score -= 1
        reasons.append("High volatility, consider risk before investing.")

# News sentiment (optional)
if 'news_score' in locals() or 'news_score' in globals():
    if news_score > 0.1:
        recommendation_score += 1
        reasons.append("Positive sentiment in recent news.")
    elif news_score < -0.1:
        recommendation_score -= 1
        reasons.append("Negative sentiment in recent news.")

# Final verdict
if recommendation_score >= 3:
    final_recommendation = "✅ BUY"
elif recommendation_score <= -2:
    final_recommendation = "❌ SELL"
else:
    final_recommendation = "⏸️ HOLD"

st.markdown(f"### Final Recommendation: **{final_recommendation}**")
st.write("#### Reasoning:")
for reason in reasons:
    st.markdown(f"- {reason}")
