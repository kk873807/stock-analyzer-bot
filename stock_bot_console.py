# stock_bot_console.py

import yfinance as yf

def fetch_stock_info(symbol):
    stock = yf.Ticker(symbol)
    info = stock.info
    
    print(f"\n📊 Basic Info for {symbol.upper()}")
    print(f"----------------------------------")
    print(f"Name: {info.get('longName', 'N/A')}")
    print(f"Current Price: ₹{info.get('currentPrice', 'N/A')}")
    print(f"Previous Close: ₹{info.get('previousClose', 'N/A')}")
    print(f"52-Week High: ₹{info.get('fiftyTwoWeekHigh', 'N/A')}")
    print(f"52-Week Low: ₹{info.get('fiftyTwoWeekLow', 'N/A')}")
    print(f"P/E Ratio: {info.get('trailingPE', 'N/A')}")
    print(f"Market Cap: {info.get('marketCap', 'N/A')}")
    print(f"Sector: {info.get('sector', 'N/A')}")

if __name__ == "__main__":
    print("📈 Welcome to Stock Bot (Console)")
    print("✅ Example NSE: INFY.NS, RELIANCE.NS")
    print("✅ Example NASDAQ: AAPL, TSLA, AMZN\n")

    stock_symbol = input("Enter Stock Symbol: ").strip().upper()
    fetch_stock_info(stock_symbol)
