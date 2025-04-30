# =============================================
# STOCK SENTIMENT ANALYZER (GitHub-Deployable Version)
# =============================================

import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime
import os  # <-- Add this if not already present
# Add to imports (top of file)
import logging
logging.basicConfig(level=logging.INFO)

def get_stock_data(ticker):
    """Enhanced stock fetcher with debug"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d", prepost=True)
        
        if hist.empty:
            logging.error(f"⚠️ No data for {ticker}: Empty history")
            return None
            
        # Debug print prices
        logging.info(f"\n{ticker} Price Data:")
        logging.info(f"Open: {hist['Open'].iloc[0]}")
        logging.info(f"Close: {hist['Close'].iloc[-1]}")
        logging.info(f"After Hours: {stock.fast_info.get('postMarketPrice')}")
        
        return {
            "symbol": ticker,
            "regular_change": (hist['Close'].iloc[-1] - hist['Open'].iloc[0]) / hist['Open'].iloc[0] * 100,
            "current_price": stock.fast_info.get('postMarketPrice', hist['Close'].iloc[-1])
        }
    except Exception as e:
        logging.error(f"🚨 Failed {ticker}: {str(e)}")
        return None

def run_analysis():
    print("🔍 Starting analysis with debug...")
    valid_stocks = {}
    
    for ticker in CONFIG["top_50_stocks"] + CONFIG["elon_stocks"]:
        data = get_stock_data(ticker)
        if data:
            valid_stocks[ticker] = data
        else:
            logging.warning(f"⚠️ Skipping {ticker} - no valid data")
    
    if not valid_stocks:
        logging.error("❌ All tickers failed - check network or API limits")
        return None
    
    # Rest of your analysis code...
def save_results(df):
    """Enhanced file saving with debug output"""
    try:
        df.to_json("results.json", orient='records')
        print("✅ Successfully saved results.json")
        print(f"File size: {os.path.getsize('results.json')} bytes")
        return True
    except Exception as e:
        print(f"❌ Failed to save results.json: {str(e)}")
        return False

# Configuration
CONFIG = {
    "top_50_stocks": [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "JNJ", "V",
        "PG", "HD", "MA", "DIS", "PYPL", "ADBE", "NFLX", "CRM", "BAC", "KO",
        "PEP", "INTC", "CMCSA", "XOM", "WMT", "ABT", "TMO", "AVGO", "CSCO", "PFE",
        "MRK", "ABBV", "ACN", "DHR", "VZ", "T", "UNH", "LIN", "ORCL", "NKE",
        "PM", "LLY", "AMD", "MDT", "BMY", "HON", "QCOM", "INTU", "AMGN", "COP"
    ],
    "elon_stocks": ["TSLA"],
    "api_keys": {
        "alpha_vantage": os.getenv('ALPHAVANTAGE_KEY', 'demo'),
        "newsapi": os.getenv('NEWSAPI_KEY', '')
    },
    "nlp_model": {
        "name": "distilbert-base-uncased-finetuned-sst-2-english",
        "max_length": 512
    }
}

# Initialize NLP model
try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=CONFIG["nlp_model"]["name"],
        device=-1
    )
except:
    def get_sentiment(text):
        positive = ["rise", "gain", "up", "bullish", "beat", "buy", "strong"]
        negative = ["fall", "drop", "down", "bearish", "miss", "sell", "weak"]
        score = sum(1 for w in positive if w in text.lower()) - \
                sum(1 for w in negative if w in text.lower())
        return "POSITIVE" if score > 0 else "NEGATIVE" if score < 0 else "NEUTRAL"
    sentiment_pipeline = None

def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d", prepost=True)
        
        if hist.empty:
            return {
                "symbol": ticker,
                "valid": False,
                "error": "No data"
            }
        
        open_p = hist['Open'].iloc[0]
        regular_close = hist['Close'].iloc[-1]
        post_market = stock.fast_info.get('postMarketPrice', regular_close)
        
        regular_pct = np.round((regular_close - open_p) / open_p * 100, 2)
        post_pct = np.round((post_market - regular_close) / regular_close * 100, 2)
        
        return {
            "symbol": ticker,
            "valid": True,
            "regular_change": float(regular_pct),
            "post_change": float(post_pct),
            "current_price": float(post_market)
        }
    except Exception as e:
        return {
            "symbol": ticker,
            "valid": False,
            "error": str(e)
        }

def fetch_all_stocks():
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(get_stock_data, CONFIG["top_50_stocks"] + CONFIG["elon_stocks"]))
    return {r["symbol"]: r for r in results if r["valid"]}

def fetch_news():
    news_items = []
    
    # Alpha Vantage
    if CONFIG['api_keys']['alpha_vantage'] not in ['', 'demo']:
        try:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=TSLA,AAPL&apikey={CONFIG['api_keys']['alpha_vantage']}"
            data = requests.get(url, timeout=10).json()
            news_items.extend([{
                "title": item["title"],
                "source": item.get("source", "AlphaVantage"),
                "tickers": [t["ticker"] for t in item.get("ticker_sentiment", [])]
            } for item in data.get("feed", [])[:10]])
        except:
            pass
    
    # NewsAPI Fallback
    if CONFIG['api_keys']['newsapi'] and len(news_items) < 5:
        try:
            url = f"https://newsapi.org/v2/everything?q=stocks&apiKey={CONFIG['api_keys']['newsapi']}"
            data = requests.get(url, timeout=10).json()
            news_items.extend([{
                "title": article["title"],
                "source": article["source"]["name"],
                "tickers": []
            } for article in data.get("articles", [])[:5]])
        except:
            pass
    
    # Hardcoded Fallback
    if not news_items:
        news_items = [{
            "title": "Tech stocks rally as AI boom continues",
            "source": "Fallback",
            "tickers": ["AAPL", "MSFT", "NVDA"]
        }]
    
    return news_items[:15]

def analyze_news(stock_data, news_items):
    results = []
    
    for item in news_items:
        try:
            sentiment = (sentiment_pipeline(item["title"][:512])[0]['label'] 
                       if sentiment_pipeline 
                       else get_sentiment(item["title"]))
            
            for ticker in set(item.get("tickers", [])):
                if ticker in stock_data:
                    results.append({
                        "Ticker": ticker,
                        "Company": yf.Ticker(ticker).info.get('shortName', ticker),
                        "Headline": item["title"],
                        "Sentiment": sentiment,
                        "Regular Change": stock_data[ticker]["regular_change"],
                        "After Hours": stock_data[ticker]["post_change"],
                        "Source": item["source"],
                        "Timestamp": datetime.now().isoformat()
                    })
        except:
            continue
    
    return pd.DataFrame(results) if results else None

def save_outputs(df):
    """Save results in multiple formats"""
    if df is not None:
        # JSON for web consumption
        df.to_json("results.json", orient="records")
        
        # HTML for debugging
        with open("results.html", "w") as f:
            f.write(df.to_html())
        
        # Simple text summary
        with open("summary.txt", "w") as f:
            f.write(f"Last run: {datetime.now()}\n")
            f.write(f"Stocks analyzed: {len(df['Ticker'].unique())}\n")
            f.write(f"Positive headlines: {sum(df['Sentiment'] == 'POSITIVE')}\n")
        
        return True
    return False

def run_analysis():
    print("🚀 Starting analysis...")
    stock_data = fetch_all_stocks()
    news_items = fetch_news()
    results_df = analyze_news(stock_data, news_items)
    
    if save_outputs(results_df):
        print("✅ Analysis complete - results saved")
        return results_df
    else:
        print("❌ Analysis failed")
        return None

if __name__ == "__main__":
    # Configure matplotlib for headless environments
    import matplotlib
    matplotlib.use('Agg')
    
    results = run_analysis()
    if results is not None:
        print("\nTop Movers:")
        print(results.sort_values("Regular Change", ascending=False).head())

if __name__ == "__main__":
    results = run_analysis()
    if results is not None:
        save_results(results)
