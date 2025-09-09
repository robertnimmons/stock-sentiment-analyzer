#!/usr/bin/env python3
"""
MINIMAL WORKING STOCK ANALYZER
This version bypasses yfinance issues and creates mock data for testing
"""

import os
import pandas as pd
import json
from datetime import datetime
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_mock_stock_data():
    """Create realistic mock stock data since yfinance isn't working"""
    logging.info("📊 Creating mock stock data (yfinance bypass)...")
    
    stocks = {
        "AAPL": {"name": "Apple Inc.", "base_price": 175.00},
        "MSFT": {"name": "Microsoft Corp.", "base_price": 340.00},
        "TSLA": {"name": "Tesla Inc.", "base_price": 240.00},
        "GOOGL": {"name": "Alphabet Inc.", "base_price": 140.00},
        "AMZN": {"name": "Amazon.com Inc.", "base_price": 145.00}
    }
    
    mock_data = {}
    
    for ticker, info in stocks.items():
        # Generate realistic random changes (-5% to +5%)
        regular_change = round(random.uniform(-5.0, 5.0), 2)
        post_change = round(random.uniform(-2.0, 2.0), 2)
        
        current_price = info["base_price"] * (1 + regular_change/100)
        
        mock_data[ticker] = {
            "symbol": ticker,
            "company_name": info["name"],
            "regular_change": regular_change,
            "post_change": post_change,
            "current_price": round(current_price, 2)
        }
        
        logging.info(f"  {ticker}: {regular_change:+.2f}% (${current_price:.2f})")
    
    logging.info(f"✅ Created mock data for {len(mock_data)} stocks")
    return mock_data

def create_mock_news():
    """Create realistic news data"""
    logging.info("📰 Creating mock news data...")
    
    news_items = [
        {
            "title": "Apple reports record iPhone sales in latest quarter",
            "source": "TechNews",
            "tickers": ["AAPL"],
            "expected_sentiment": "POSITIVE"
        },
        {
            "title": "Microsoft cloud revenue surges as AI demand grows",
            "source": "BusinessDaily", 
            "tickers": ["MSFT"],
            "expected_sentiment": "POSITIVE"
        },
        {
            "title": "Tesla faces production challenges at new factory",
            "source": "AutoWorld",
            "tickers": ["TSLA"], 
            "expected_sentiment": "NEGATIVE"
        },
        {
            "title": "Google announces major AI breakthrough in search",
            "source": "TechReport",
            "tickers": ["GOOGL"],
            "expected_sentiment": "POSITIVE"
        },
        {
            "title": "Amazon Prime membership growth slows amid competition",
            "source": "MarketWatch",
            "tickers": ["AMZN"],
            "expected_sentiment": "NEGATIVE"
        }
    ]
    
    logging.info(f"✅ Created {len(news_items)} news items")
    return news_items

def analyze_sentiment(text):
    """Simple but effective sentiment analysis"""
    positive_words = [
        "surge", "record", "breakthrough", "growth", "strong", "beat", 
        "rise", "gain", "up", "bullish", "positive", "excellent", "soar"
    ]
    
    negative_words = [
        "challenge", "slow", "fall", "drop", "weak", "miss", "down", 
        "bearish", "negative", "decline", "loss", "struggle", "plunge"
    ]
    
    text_lower = text.lower()
    
    pos_score = sum(1 for word in positive_words if word in text_lower)
    neg_score = sum(1 for word in negative_words if word in text_lower)
    
    if pos_score > neg_score:
        return "POSITIVE"
    elif neg_score > pos_score:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def create_analysis(stock_data, news_items):
    """Create the final analysis combining stocks and news"""
    logging.info("🔍 Creating sentiment analysis...")
    
    results = []
    
    for news in news_items:
        sentiment = analyze_sentiment(news["title"])
        
        for ticker in news.get("tickers", []):
            if ticker in stock_data:
                stock_info = stock_data[ticker]
                
                results.append({
                    "Ticker": ticker,
                    "Company": stock_info["company_name"],
                    "Headline": news["title"],
                    "Sentiment": sentiment,
                    "Regular Change": stock_info["regular_change"],
                    "After Hours": stock_info["post_change"], 
                    "Current Price": stock_info["current_price"],
                    "Source": news["source"],
                    "Timestamp": datetime.now().isoformat()
                })
    
    if not results:
        logging.error("❌ No analysis results created")
        return None
        
    logging.info(f"✅ Created {len(results)} analysis records")
    return pd.DataFrame(results)

def save_results(df):
    """Save all output files"""
    if df is None or df.empty:
        logging.error("❌ No data to save")
        return False
    
    try:
        # Save JSON for web apps
        df.to_json("results.json", orient="records", indent=2)
        logging.info("✅ Saved results.json")
        
        # Save HTML for viewing
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Stock Sentiment Analysis Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .positive {{ background-color: #d4edda; }}
                .negative {{ background-color: #f8d7da; }}
                .neutral {{ background-color: #fff3cd; }}
            </style>
        </head>
        <body>
            <h1>Stock Sentiment Analysis Results</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Total Records: {len(df)}</p>
            
            <h2>Summary</h2>
            <ul>
                <li>Stocks Analyzed: {len(df['Ticker'].unique())}</li>
                <li>Positive Sentiment: {sum(df['Sentiment'] == 'POSITIVE')}</li>
                <li>Negative Sentiment: {sum(df['Sentiment'] == 'NEGATIVE')}</li>
                <li>Neutral Sentiment: {sum(df['Sentiment'] == 'NEUTRAL')}</li>
            </ul>
            
            <h2>Detailed Results</h2>
        """
        
        # Add table with color coding
        for _, row in df.iterrows():
            sentiment_class = row['Sentiment'].lower()
            html_content += f"""
            <div class="{sentiment_class}" style="margin: 10px 0; padding: 10px; border-radius: 5px;">
                <strong>{row['Ticker']} ({row['Company']})</strong><br>
                <em>"{row['Headline']}"</em><br>
                Sentiment: {row['Sentiment']} | 
                Change: {row['Regular Change']:+.2f}% | 
                Price: ${row['Current Price']:.2f}<br>
                <small>Source: {row['Source']}</small>
            </div>
            """
        
        html_content += "</body></html>"
        
        with open("results.html", "w") as f:
            f.write(html_content)
        logging.info("✅ Saved results.html")
        
        # Save summary
        with open("summary.txt", "w") as f:
            f.write("STOCK SENTIMENT ANALYSIS SUMMARY\n")
            f.write("=" * 40 + "\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Total Records: {len(df)}\n")
            f.write(f"Stocks Analyzed: {len(df['Ticker'].unique())}\n\n")
            
            f.write("SENTIMENT BREAKDOWN:\n")
            f.write(f"Positive: {sum(df['Sentiment'] == 'POSITIVE')}\n")
            f.write(f"Negative: {sum(df['Sentiment'] == 'NEGATIVE')}\n") 
            f.write(f"Neutral: {sum(df['Sentiment'] == 'NEUTRAL')}\n\n")
            
            f.write("TOP PERFORMERS:\n")
            top_performers = df.nlargest(3, 'Regular Change')
            for _, row in top_performers.iterrows():
                f.write(f"{row['Ticker']}: {row['Regular Change']:+.2f}% - {row['Sentiment']}\n")
            
            f.write("\nBOTTOM PERFORMERS:\n")
            bottom_performers = df.nsmallest(3, 'Regular Change')
            for _, row in bottom_performers.iterrows():
                f.write(f"{row['Ticker']}: {row['Regular Change']:+.2f}% - {row['Sentiment']}\n")
        
        logging.info("✅ Saved summary.txt")
        return True
        
    except Exception as e:
        logging.error(f"❌ Failed to save results: {e}")
        return False

def run_analysis():
    """Main function that orchestrates the entire analysis"""
    try:
        logging.info("🚀 Starting Stock Sentiment Analysis (Mock Data Mode)")
        
        # Create mock data instead of fetching real data
        stock_data = create_mock_stock_data()
        news_items = create_mock_news()
        
        # Perform analysis
        results_df = create_analysis(stock_data, news_items)
        
        if results_df is None:
            logging.error("❌ Analysis failed")
            return None
        
        # Save all outputs
        if save_results(results_df):
            logging.info("✅ Analysis completed successfully!")
            
            # Print summary to console
            print(f"\n📊 ANALYSIS COMPLETE")
            print(f"📈 Stocks: {len(results_df['Ticker'].unique())}")
            print(f"📰 Headlines: {len(results_df)}")
            print(f"😊 Positive: {sum(results_df['Sentiment'] == 'POSITIVE')}")
            print(f"😞 Negative: {sum(results_df['Sentiment'] == 'NEGATIVE')}")
            print(f"😐 Neutral: {sum(results_df['Sentiment'] == 'NEUTRAL')}")
            
            print(f"\n🏆 TOP MOVERS:")
            top_movers = results_df.nlargest(3, 'Regular Change')
            for _, row in top_movers.iterrows():
                print(f"  {row['Ticker']}: {row['Regular Change']:+.2f}% ({row['Sentiment']})")
            
            return results_df
        else:
            logging.error("❌ Failed to save results")
            return None
            
    except Exception as e:
        logging.error(f"❌ Analysis failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    # Set random seed for reproducible results in testing
    random.seed(42)
    
    results = run_analysis()
    if results is not None:
        print("✅ SUCCESS: Check the generated files!")
        exit(0)
    else:
        print("❌ FAILED: Check the logs above") 
        exit(1)
