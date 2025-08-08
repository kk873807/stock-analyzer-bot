# stock_bot_web.py

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests
import numpy as np
from textblob import TextBlob
from countryinfo import CountryInfo

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def predict_stock_prices(stock_symbol, days=7):
    try:
        df = yf.download(stock_symbol, period="6mo", interval="1d")
        df = df[['Close']].dropna()
        df['Prediction'] = df[['Close']].shift(-days)

        # Features and Labels
        X = df.drop(['Prediction'], axis=1)[:-days]
        y = df['Prediction'][:-days]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict next 'days' days
        future_days = df.drop(['Prediction'], axis=1)[-days:]
        predictions = model.predict(future_days)

        return predictions, df
    except Exception as e:
        return None, None

# ====== 🗃 Portfolio File ======
PORTFOLIO_FILE = "portfolio_data.json"

def load_portfolio():
    try:
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f)

# ====== 🧠 App UI Starts Here ======
st.set_page_config(page_title="Stock Analyzer Bot", layout="centered")
st.title("📈 Stock Analyzer Bot")

stock_symbol = st.text_input("Enter stock symbol (e.g. INFY.NS for Infosys)", "INFY.NS")

# ====== 📊 Show Stock Info ======
if stock_symbol:
    stock_data = yf.Ticker(stock_symbol)
    stock_df = stock_data.history(period="6mo")

    st.subheader(f"📄 {stock_symbol} Summary")
    try:
        st.write(stock_data.info['longBusinessSummary'])
    except:
        st.write("Company summary not available.")

    st.subheader("📉 Stock Price Chart")
    st.line_chart(stock_df['Close'])

# ====== 💼 Buy/Sell Portfolio Tracking ======
st.subheader("💰 Buy / Sell Tracker")

portfolio = load_portfolio()
quantity = st.number_input("Enter quantity to buy/sell", min_value=1, step=1)

col1, col2 = st.columns(2)
with col1:
    if st.button("🟩 Buy"):
        if stock_symbol in portfolio:
            portfolio[stock_symbol] += quantity
        else:
            portfolio[stock_symbol] = quantity
        save_portfolio(portfolio)
        st.success(f"Bought {quantity} shares of {stock_symbol}")

with col2:
    if st.button("🟥 Sell"):
        if stock_symbol in portfolio and portfolio[stock_symbol] >= quantity:
            portfolio[stock_symbol] -= quantity
            if portfolio[stock_symbol] == 0:
                del portfolio[stock_symbol]
            save_portfolio(portfolio)
            st.success(f"Sold {quantity} shares of {stock_symbol}")
        else:
            st.error("You don't have enough shares to sell.")

# ====== 📂 Show Portfolio ======
st.subheader("📦 Your Portfolio")

if not portfolio:
    st.info("You don't own any stocks yet.")
else:
    for symbol, qty in portfolio.items():
        st.write(f"📌 {symbol}: {qty} shares")

# ====== 🔮 Stock Price Prediction ======
st.subheader("🔮 Stock Price Prediction (Next 7 Days)")

if st.button("📊 Predict Future Prices"):
    predictions, df = predict_stock_prices(stock_symbol)

    if predictions is not None:
        st.success("Prediction complete!")
        st.write("📈 **Next 7 Days Predicted Prices**")
        for i, price in enumerate(predictions, 1):
            st.write(f"Day {i}: ₹{price:.2f}")

        # Plot
        fig, ax = plt.subplots()
        df['Close'].plot(ax=ax, label='Historical Prices')
        future_index = list(range(len(df), len(df) + 7))
        ax.plot(future_index, predictions, label='Predicted', linestyle='--')
        ax.set_title(f"{stock_symbol} - 7 Day Forecast")
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("Prediction failed. Try a different stock.")
# Mocked data for macro indicators (in production, pull from APIs)
macro_data = {
    "India": {
        "GDP_growth": 6.5,
        "Inflation": 5.4,
        "Interest_rate": 6.5,
        "Geopolitical_risk": "Low"
    },
    "United States": {
        "GDP_growth": 2.1,
        "Inflation": 3.4,
        "Interest_rate": 5.25,
        "Geopolitical_risk": "Moderate"
    },
    "China": {
        "GDP_growth": 5.0,
        "Inflation": 2.2,
        "Interest_rate": 3.6,
        "Geopolitical_risk": "High"
    }
}

def get_macro_analysis(stock_symbol):
    # Guess country from ticker
    if ".NS" in stock_symbol:
        country = "India"
    elif stock_symbol.upper() in ["AAPL", "TSLA", "MSFT", "AMZN"]:
        country = "United States"
    elif stock_symbol.upper() in ["BABA", "JD"]:
        country = "China"
    else:
        country = "United States"

    data = macro_data.get(country, None)
    if not data:
        return {"error": "Country macro data not found."}

    analysis = []

    # GDP Growth
    if data["GDP_growth"] > 3:
        analysis.append("✅ GDP growth is strong.")
    else:
        analysis.append("⚠️ Low GDP growth. Might affect markets.")

    # Inflation
    if data["Inflation"] > 6:
        analysis.append("❌ High inflation may reduce profitability.")
    elif data["Inflation"] > 4:
        analysis.append("⚠️ Inflation is slightly elevated.")
    else:
        analysis.append("✅ Inflation is within safe range.")

    # Interest Rate
    if data["Interest_rate"] > 5:
        analysis.append("⚠️ High interest rates may reduce investments.")
    else:
        analysis.append("✅ Interest rates are investor-friendly.")

    # Geopolitical Risk
    risk = data["Geopolitical_risk"]
    if risk == "Low":
        analysis.append("✅ Geopolitical risk is minimal.")
    elif risk == "Moderate":
        analysis.append("⚠️ Moderate geopolitical risks present.")
    else:
        analysis.append("❌ High geopolitical risk. Proceed cautiously.")

    return {
        "country": country,
        "summary": analysis
    }


# ====== 🔑 Enter your NewsData.io API key here ======
API_KEY = "pub_aeb8f6aeaa614154856ab1817ffe14dd"  # <-- Replace this with your API key
# ================================================

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


peer_dict = {
    "INFY.NS": ["TCS.NS", "WIPRO.NS", "TECHM.NS"],
    "TCS.NS": ["INFY.NS", "WIPRO.NS", "HCLTECH.NS"],
    "RELIANCE.NS": ["ONGC.NS", "IOC.NS", "BPCL.NS"],
    "AAPL": ["MSFT", "GOOGL", "AMZN"],
    "TSLA": ["F", "GM", "NIO"]
}

st.set_page_config(page_title="📊 Stock Analyzer", layout="centered")
st.title(":chart_with_upwards_trend: Stock Analyzer Bot (Beginner Friendly)")

stock_symbol = st.text_input("Enter Stock Symbol (e.g., INFY.NS, TCS.NS, AAPL, TSLA)", value="INFY.NS")

if stock_symbol:
    try:
        stock = yf.Ticker(stock_symbol)
        info = stock.info

        st.subheader(f"📌 {info.get('longName', stock_symbol)}")
        current_price = info.get('currentPrice', None)
        high_52 = info.get('fiftyTwoWeekHigh', None)
        low_52 = info.get('fiftyTwoWeekLow', None)
        pe_ratio = info.get('trailingPE', None)

        st.markdown(f"**Current Price:** ₹{current_price}")
        st.markdown(f"**52-Week High:** ₹{high_52}")
        st.markdown(f"**52-Week Low:** ₹{low_52}")
        st.markdown(f"**P/E Ratio:** {pe_ratio}")
        st.markdown(f"**Market Cap:** {info.get('marketCap', 'N/A')}")
        st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")

        hist = stock.history(period="6mo")
        hist["50MA"] = hist["Close"].rolling(window=50).mean()

        hist["20MA"] = hist["Close"].rolling(window=20).mean()
        hist["200MA"] = hist["Close"].rolling(window=200).mean()
        hist["Returns"] = hist["Close"].pct_change()
        hist["Volatility"] = hist["Returns"].rolling(window=20).std() * np.sqrt(252)

        st.subheader(":chart_with_downwards_trend: Stock Price Trend (6 Months)")
        fig, ax = plt.subplots()
        ax.plot(hist.index, hist["Close"], label="Close Price")
        ax.plot(hist.index, hist["20MA"], label="20-Day MA", linestyle='--')
        ax.plot(hist.index, hist["50MA"], label="50-Day MA", linestyle='--')
        ax.plot(hist.index, hist["200MA"], label="200-Day MA", linestyle='--')

        ax.set_xlabel("Date")
        ax.set_ylabel("Price (₹)")
        ax.set_title(f"{stock_symbol.upper()} - Last 6 Months")
        ax.legend()
        st.pyplot(fig)

        st.subheader("🧠 Investment Suggestion:")
        suggestion = []

        if current_price and low_52 and current_price <= (low_52 * 1.1):
            suggestion.append("✅ Price is near 52-week low. May be undervalued.")
        elif current_price and high_52 and current_price >= (high_52 * 0.95):
            suggestion.append("⚠️ Price is near 52-week high. Wait for better entry.")

        if pe_ratio:
            if pe_ratio < 10:
                suggestion.append("✅ Very low P/E ratio. Likely undervalued.")
            elif 10 <= pe_ratio <= 25:
                suggestion.append("🟡 Fairly valued based on P/E ratio.")
            else:
                suggestion.append("🔺 High P/E ratio. Could be overvalued.")

        if "Close" in hist.columns:
            latest_50ma = hist["50MA"].iloc[-1]
            if current_price and current_price > latest_50ma:
                suggestion.append("📈 Price is above 50-day MA. Bullish signal.")
            elif current_price and current_price < latest_50ma:
                suggestion.append("📉 Price is below 50-day MA. Bearish signal.")

        if suggestion:
            for s in suggestion:
                st.markdown(f"- {s}")
        else:
            st.info("Not enough data to analyze stock.")

    except Exception as e:
        st.error(f"Failed to load data: {e}")

# =================== PHASE 6 ENHANCED LOGIC ====================
st.subheader("📊 Buy/Sell Signal Strength")

company_name = info.get("longName") or info.get("shortName") or stock_symbol
sentiment = get_news_sentiment(company_name)

strength_score = 0


# Near 52-week low?
if current_price and low_52 and current_price <= (low_52 * 1.1):
    strength_score += 2

# P/E Ratio
if pe_ratio:
    if pe_ratio < 10:
        strength_score += 2
    elif pe_ratio <= 25:
        strength_score += 1
    else:
        strength_score -= 1

# Moving Average Crossovers
latest_20ma = hist["20MA"].iloc[-1]
latest_50ma = hist["50MA"].iloc[-1]
latest_200ma = hist["200MA"].iloc[-1]

if current_price > latest_20ma > latest_50ma:
    strength_score += 2
elif current_price > latest_50ma:
    strength_score += 1
else:
    strength_score -= 1

# News Sentiment
if sentiment["Positive"] > sentiment["Negative"]:
    strength_score += 1
elif sentiment["Negative"] > sentiment["Positive"]:
    strength_score -= 1

# Volatility check
vol = hist["Volatility"].iloc[-1]
if vol and vol > 0.4:
    st.warning("⚠️ High market volatility. Extra caution advised.")
    strength_score -= 1
else:
    st.info("✅ Market volatility is within normal range.")

# Decision Based on Score
if strength_score >= 5:
    st.success("📈 Strong Buy Signal")
elif strength_score >= 3:
    st.info("🟢 Mild Buy Signal")
elif strength_score <= -1:
    st.error("🔻 Sell/Exit Signal")
else:
    st.warning("🟡 Hold / Wait — Signal is Neutral")

# ================= PEER COMPARISON ===================
st.subheader(":bar_chart: Peer Comparison")
peers = peer_dict.get(stock_symbol.upper(), [])

if not peers:
    st.info("No peer data available for this stock.")
else:
    data = []

    for peer in peers:
        try:
            p = yf.Ticker(peer)
            p_info = p.info
            name = p_info.get('shortName', peer)
            price = p_info.get('currentPrice', None)
            pe = p_info.get('trailingPE', None)
            high = p_info.get('fiftyTwoWeekHigh', None)
            low = p_info.get('fiftyTwoWeekLow', None)

            if price and high and low:
                discount = round((1 - price / high) * 100, 2)
                gain = round((price / low - 1) * 100, 2)
            else:
                discount = gain = None

            data.append({
                "Company": name,
                "Price": price,
                "P/E Ratio": pe,
                "% From High": f"{discount}%" if discount else "N/A",
                "% From Low": f"{gain}%" if gain else "N/A"
            })

        except:
            continue

    if data:
        df = pd.DataFrame(data)
        st.dataframe(df)

        undervalued = df[df["P/E Ratio"] < 20]
        if not undervalued.empty:
            best = undervalued.sort_values(by="P/E Ratio").iloc[0]
            st.success(f"📌 Among peers, **{best['Company']}** looks most undervalued (P/E = {best['P/E Ratio']}).")
        else:
            st.info("No significantly undervalued stock found among peers.")

# =============== NEWS SENTIMENT SECTION ===============
st.subheader("📵 News Sentiment Analysis")

company_name = info.get("longName") or info.get("shortName") or stock_symbol
sentiment = get_news_sentiment(company_name)

if "error" in sentiment:
    st.warning(f"Could not fetch news: {sentiment['error']}")
else:
    st.markdown(f"**Positive:** {sentiment['Positive']} | **Negative:** {sentiment['Negative']} | **Neutral:** {sentiment['Neutral']}")

    if sentiment["Positive"] > sentiment["Negative"]:
        st.success("✅ Overall news sentiment is positive.")
    elif sentiment["Negative"] > sentiment["Positive"]:
        st.error("⚠️ News sentiment is mostly negative. Be cautious.")
    else:
        st.info("ℹ️ Neutral news sentiment.")

    with st.expander("🔍 View News Headlines"):
        for text, label in sentiment["Details"]:
            st.markdown(f"- {text} — *{label}*")

# 🌍 MACRO & GEOPOLITICAL RISK
st.subheader("🌍 Macro & Geopolitical Analysis")

macro = get_macro_analysis(stock_symbol)
if "error" in macro:
    st.warning(macro["error"])
else:
    st.markdown(f"**Country:** {macro['country']}")
    for item in macro["summary"]:
        st.markdown(f"- {item}")
