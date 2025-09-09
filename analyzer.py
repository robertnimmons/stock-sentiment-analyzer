#!/usr/bin/env python3
"""
DIAGNOSTIC STOCK ANALYZER
This version has extensive debugging to find the exact issue
"""

import os
import requests
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime
import logging
import time

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Test with just a few stocks first
CONFIG = {
    "test_stocks": ["AAPL", "MSFT", "TSLA"],  # Just 3 stocks for testing
    "api_keys": {
        "alpha_vantage": os.getenv('ALPHAVANTAGE_KEY', 'demo'),
        "newsapi": os.getenv('NEWSAPI_KEY', '')
    }
}

def test_single_stock(ticker):
    """Test fetching data for a single stock with detailed logging"""
    logging.info(f"🧪 Testing {ticker}...")
    
    try:
        # Test basic yfinance connection
        logging.info(f"Creating yfinance Ticker object for {ticker}")
        stock = yf.Ticker(ticker)
        
        # Test history fetch
        logging.info(f"Fetching history for {ticker}")
        hist = stock.history(period="1d", prepost=True)
        
        if hist.empty:
            logging.error(f"❌ {ticker}: History is empty")
            return None
        
        logging.info(f"✅ {ticker}: Got {len(hist)} data points")
        logging.info(f"   Columns: {list(hist.columns)}")
        logging.info(f"   Date range: {hist.index[0]} to {hist.index[-1]}")
        
        # Test price extraction
        open_price = hist['Open'].iloc[0]
        close_price = hist['Close'].iloc[-1]
        logging.info(f"   Open: ${open_price:.2f}, Close: ${close_price:.2f}")
        
        # Test fast_info (this might be the problem)
        logging.info(f"Testing fast_info for {ticker}")
        try:
            fast_info = stock.fast_info
            logging.info(f"✅ {ticker}: fast_info accessible")
            post_market = fast_info.get('postMarketPrice', close_price)
            logging.info(f"   Post-market: ${post_market:.2f}")
        except Exception as e:
            logging.warning(f"⚠️ {ticker}: fast_info failed - {e}")
            post_market = close_price
        
        # Calculate changes
        regular_change = ((close_price - open_price) / open_price) * 100
        post_change = ((post_market - close_price) / close_price) * 100
        
        result = {
            "symbol": ticker,
            "regular_change": round(float(regular_change), 2),
            "post_change": round(float(post_change), 2),
            "current_price": round(float(post_market), 2),
        }
        
        logging.info(f"✅ {ticker}: Successfully processed - Change: {regular_change:.2f}%")
        return result
        
    except Exception as e:
        logging.error(f"❌ {ticker}: Failed with error - {str(e)}")
        import traceback
        logging.error(f"   Traceback: {traceback.format_exc()}")
        return None

def test_stock_fetching():
    """Test stock data fetching with detailed diagnostics"""
    logging.info("🔍 Starting stock fetching diagnostics...")
    
    # Test internet connectivity
    try:
        response = requests.get("https://httpbin.org/ip", timeout=10)
        logging.info(f"✅ Internet connectivity OK - IP: {response.json()}")
    except Exception as e:
        logging.error(f"❌ Internet connectivity issue: {e}")
        return {}
    
    # Test yfinance import and basic functionality
    try:
        logging.info(f"✅ yfinance version: {yf.__version__}")
    except:
        logging.warning("⚠️ Could not get yfinance version")
    
    # Test each stock individually
    results = {}
    for ticker in CONFIG["test_stocks"]:
        result = test_single_stock(ticker)
        if result:
            results[ticker] = result
        time.sleep(1)  # Add delay to avoid rate limiting
    
    logging.info(f"📊 Final results: {len(results)} out of {len(CONFIG['test_stocks'])} stocks successful")
    return results

def create_fallback_news():
    """Create simple fallback news for testing"""
    return [
        {
            "title": "Apple stock rises on strong iPhone sales",
            "source": "Test",
            "tickers": ["AAPL"]
        },
        {
            "title": "Microsoft cloud revenue beats expectations", 
            "source": "Test",
            "tickers": ["MSFT"]
        },
        {
            "title": "Tesla delivers record number of vehicles",
            "source": "Test", 
            "tickers": ["TSLA"]
        }
    ]

def simple_sentiment(text):
    """Simple sentiment analysis for testing"""
    positive_words = ["rise", "beat", "record", "strong", "up", "gain"]
    negative_words = ["fall", "miss", "drop", "weak", "down", "loss"]
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return "POSITIVE"
    elif neg_count > pos_count:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def create_simple_analysis(stock_data, news_items):
    """Create analysis with minimal complexity"""
    logging.info("🔍 Creating simple analysis...")
    
    if not stock_data:
        logging.error("❌ No stock data to analyze")
        return None
        
    if not news_items:
        logging.error("❌ No news data to analyze") 
        return None
    
    results = []
    
    for news in news_items:
        sentiment = simple_sentiment(news["title"])
        
        for ticker in news.get("tickers", []):
            if ticker in stock_data:
                results.append({
                    "Ticker": ticker,
                    "Company": ticker,  # Keep simple for now
                    "Headline": news["title"][:100],
                    "Sentiment": sentiment,
                    "Regular Change": stock_data[ticker]["regular_change"],
                    "After Hours": stock_data[ticker]["post_change"],
                    "Current Price": stock_data[ticker]["current_price"],
                    "Source": news["source"],
                    "Timestamp": datetime.now().isoformat()
                })
    
    if not results:
        logging.error("❌ No matching ticker-news combinations found")
        return None
        
    logging.info(f"✅ Created {len(results)} analysis records")
    return pd.DataFrame(results)

def save_simple_outputs(df):
    """Save outputs with error handling"""
    if df is None or df.empty:
        logging.error("❌ No data to save")
        return False
    
    try:
        # Save JSON
        df.to_json("results.json", orient="records", indent=2)
        logging.info("✅ Saved results.json")
        
        # Save simple summary
        with open("summary.txt", "w") as f:
            f.write(f"Diagnostic Analysis Summary\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Records: {len(df)}\n")
            f.write(f"Stocks: {len(df['Ticker'].unique())}\n")
        logging.info("✅ Saved summary.txt")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Failed to save: {e}")
        return False

def run_diagnostic():
    """Main diagnostic function"""
    logging.info("🚀 Starting DIAGNOSTIC mode...")
    
    try:
        # Test stock fetching
        stock_data = test_stock_fetching()
        
        if not stock_data:
            logging.error("❌ CRITICAL: No stock data could be fetched")
            logging.error("   This is likely the root cause of your error")
            logging.error("   Possible causes:")
            logging.error("   - Network connectivity issues")
            logging.error("   - yfinance API rate limiting") 
            logging.error("   - GitHub Actions firewall blocking financial APIs")
            return None
        
        # Create test news
        news_items = create_fallback_news()
        logging.info(f"📰 Using {len(news_items)} test news items")
        
        # Simple analysis
        results_df = create_simple_analysis(stock_data, news_items)
        
        # Save results
        if save_simple_outputs(results_df):
            logging.info("✅ Diagnostic completed successfully")
            print(f"\n📊 Diagnostic Results:")
            print(f"Stocks fetched: {len(stock_data)}")
            if results_df is not None:
                print(f"Analysis records: {len(results_df)}")
                print("\nStock data summary:")
                for ticker, data in stock_data.items():
                    print(f"  {ticker}: {data['regular_change']:+.2f}% (${data['current_price']:.2f})")
            return results_df
        else:
            logging.error("❌ Failed to save diagnostic results")
            return None
            
    except Exception as e:
        logging.error(f"❌ Diagnostic failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    results = run_diagnostic()
    if results is not None:
        print("✅ Diagnostic completed - check logs for details")
        exit(0)
    else:
        print("❌ Diagnostic failed - check logs for issues")
        exit(1)
