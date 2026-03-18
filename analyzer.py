#!/usr/bin/env python3
"""
REAL STOCK DATA ANALYZER
Fetches live stock prices (Alpha Vantage / Yahoo Finance) and
live financial headlines (NewsAPI) — no hardcoded data.
"""

import os
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ── Configuration ──────────────────────────────────────────────────────────────
CONFIG = {
    "stocks": ["AAPL", "MSFT", "TSLA", "GOOGL", "META", "AMZN", "NVDA", "JPM", "V", "MA"],
    "api_keys": {
        "alpha_vantage": os.getenv("ALPHAVANTAGE_KEY", "demo"),
        "newsapi":       os.getenv("NEWSAPI_KEY", ""),
    },
    # How many NewsAPI articles to fetch per ticker (free tier = 100 req/day)
    "newsapi_articles_per_ticker": 3,
}

COMPANY_NAMES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "TSLA": "Tesla Inc.",
    "GOOGL": "Alphabet Inc.",
    "META": "Meta Platforms",
    "AMZN": "Amazon.com Inc.",
    "NVDA": "NVIDIA Corp.",
    "JPM": "JPMorgan Chase",
    "V":    "Visa Inc.",
    "MA":   "Mastercard Inc.",
}


# ── Stock price fetching (unchanged from your original) ────────────────────────

def fetch_alpha_vantage_data(ticker):
    api_key = CONFIG["api_keys"]["alpha_vantage"]
    if api_key in ("", "demo"):
        return None
    try:
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}"
        )
        data = requests.get(url, timeout=10).json()
        if "Global Quote" not in data:
            logging.warning(f"⚠️  Alpha Vantage: no data for {ticker}")
            return None
        q = data["Global Quote"]
        price   = float(q["05. price"])
        chg_pct = float(q["10. change percent"].replace("%", ""))
        logging.info(f"✅ Alpha Vantage: {ticker} = ${price:.2f} ({chg_pct:+.2f}%)")
        return {"symbol": ticker, "current_price": round(price, 2),
                "regular_change": round(chg_pct, 2), "post_change": 0.0,
                "source": "AlphaVantage"}
    except Exception as e:
        logging.warning(f"⚠️  Alpha Vantage failed for {ticker}: {e}")
        return None


def fetch_yahoo_finance_direct(ticker):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        data = requests.get(url, headers=headers, timeout=10).json()
        if "chart" not in data or not data["chart"]["result"]:
            return None
        meta  = data["chart"]["result"][0]["meta"]
        price = meta["regularMarketPrice"]
        prev  = meta["previousClose"]
        chg   = ((price - prev) / prev) * 100
        post  = meta.get("postMarketPrice", price)
        post_chg = ((post - price) / price) * 100
        logging.info(f"✅ Yahoo Direct: {ticker} = ${price:.2f} ({chg:+.2f}%)")
        return {"symbol": ticker, "current_price": round(price, 2),
                "regular_change": round(chg, 2), "post_change": round(post_chg, 2),
                "source": "YahooDirect"}
    except Exception as e:
        logging.warning(f"⚠️  Yahoo Direct failed for {ticker}: {e}")
        return None


def try_yfinance_careful(ticker):
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        fast  = stock.fast_info
        price = fast.get("lastPrice") or fast.get("regularMarketPrice")
        if price:
            hist = stock.history(period="2d")
            chg  = ((price - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2]) * 100 if len(hist) >= 2 else 0.0
            post = fast.get("postMarketPrice", price)
            post_chg = ((post - price) / price) * 100
            logging.info(f"✅ yfinance: {ticker} = ${price:.2f} ({chg:+.2f}%)")
            return {"symbol": ticker, "current_price": round(float(price), 2),
                    "regular_change": round(float(chg), 2), "post_change": round(float(post_chg), 2),
                    "source": "yfinance"}
        hist = stock.history(period="1d", prepost=True)
        if not hist.empty:
            price = hist["Close"].iloc[-1]
            chg   = ((price - hist["Open"].iloc[0]) / hist["Open"].iloc[0]) * 100
            return {"symbol": ticker, "current_price": round(float(price), 2),
                    "regular_change": round(float(chg), 2), "post_change": 0.0,
                    "source": "yfinance-history"}
    except Exception as e:
        logging.warning(f"⚠️  yfinance failed for {ticker}: {e}")
    return None


def fetch_real_stock_data():
    logging.info("📊 Fetching live stock prices…")
    stock_data, failed = {}, []
    for i, ticker in enumerate(CONFIG["stocks"]):
        logging.info(f"  {ticker} ({i+1}/{len(CONFIG['stocks'])})")
        result = None
        if CONFIG["api_keys"]["alpha_vantage"] not in ("", "demo"):
            result = fetch_alpha_vantage_data(ticker)
            if result:
                stock_data[ticker] = result
                time.sleep(0.5)
                continue
        result = fetch_yahoo_finance_direct(ticker)
        if result:
            stock_data[ticker] = result
            time.sleep(1)
            continue
        result = try_yfinance_careful(ticker)
        if result:
            stock_data[ticker] = result
            time.sleep(2)
            continue
        failed.append(ticker)
        time.sleep(1)
    logging.info(f"✅ Price data fetched for {len(stock_data)} stocks")
    if failed:
        logging.warning(f"⚠️  Could not fetch prices for: {failed}")
    return stock_data


# ── LIVE news fetching via NewsAPI ─────────────────────────────────────────────

# Map tickers to search terms that return better NewsAPI results
TICKER_SEARCH_TERMS = {
    "AAPL":  "Apple stock",
    "MSFT":  "Microsoft stock",
    "TSLA":  "Tesla stock",
    "GOOGL": "Alphabet Google stock",
    "META":  "Meta Platforms stock",
    "AMZN":  "Amazon stock",
    "NVDA":  "NVIDIA stock",
    "JPM":   "JPMorgan stock",
    "V":     "Visa stock",
    "MA":    "Mastercard stock",
}


def fetch_news_for_ticker(ticker, api_key, page_size=3):
    """
    Fetch recent financial headlines for a single ticker from NewsAPI.
    Returns a list of dicts: {title, source, url, published_at}
    """
    query = TICKER_SEARCH_TERMS.get(ticker, f"{ticker} stock")
    url   = "https://newsapi.org/v2/everything"
    params = {
        "q":          query,
        "language":   "en",
        "sortBy":     "publishedAt",          # most recent first
        "pageSize":   page_size,
        "from":       (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d"),
        "apiKey":     api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            logging.warning(f"⚠️  NewsAPI error for {ticker}: {data.get('message')}")
            return []

        articles = data.get("articles", [])
        results  = []
        for art in articles:
            title = art.get("title", "").strip()
            # NewsAPI sometimes returns "[Removed]" for deleted articles
            if not title or title.lower() == "[removed]":
                continue
            results.append({
                "title":        title,
                "source":       art.get("source", {}).get("name", "NewsAPI"),
                "url":          art.get("url", ""),
                "published_at": art.get("publishedAt", ""),
                "tickers":      [ticker],
            })

        logging.info(f"✅ NewsAPI: {len(results)} articles for {ticker}")
        return results

    except Exception as e:
        logging.warning(f"⚠️  NewsAPI failed for {ticker}: {e}")
        return []


def fetch_live_news(tickers):
    """
    Fetch live headlines for every ticker.
    Falls back to a single broad finance query if the key is missing.
    Returns a flat list of news item dicts (same shape as original create_recent_news).
    """
    api_key = CONFIG["api_keys"]["newsapi"]

    if not api_key:
        logging.error(
            "❌ NEWSAPI_KEY not set. Export it with:  export NEWSAPI_KEY=your_key_here"
        )
        return []

    per_ticker = CONFIG["newsapi_articles_per_ticker"]
    all_news   = []

    for ticker in tickers:
        articles = fetch_news_for_ticker(ticker, api_key, page_size=per_ticker)
        all_news.extend(articles)
        time.sleep(0.25)   # stay well under rate limits

    # De-duplicate by headline (same story can appear for multiple tickers)
    seen, unique = set(), []
    for item in all_news:
        if item["title"] not in seen:
            seen.add(item["title"])
            unique.append(item)

    logging.info(f"✅ Total unique headlines fetched: {len(unique)}")
    return unique


# ── Sentiment analysis ─────────────────────────────────────────────────────────

POSITIVE_WORDS = {
    "surge", "rally", "beat", "strong", "growth", "optimism", "accelerates",
    "resilience", "beats", "rise", "gain", "bullish", "positive", "soar",
    "record", "profit", "upgrade", "outperform", "buy", "boost", "expansion",
    "momentum", "recovery", "breakthrough", "innovation", "exceed", "top",
}

NEGATIVE_WORDS = {
    "fall", "drop", "decline", "weak", "concerns", "bearish", "down", "plunge",
    "miss", "disappointing", "struggle", "challenges", "pressure", "negative",
    "loss", "downgrade", "underperform", "sell", "risk", "slowdown", "cut",
    "layoff", "investigation", "lawsuit", "recall", "warning", "crash",
}


def analyze_sentiment(text):
    words = set(text.lower().split())
    pos   = len(words & POSITIVE_WORDS)
    neg   = len(words & NEGATIVE_WORDS)
    if pos > neg:
        return "POSITIVE"
    if neg > pos:
        return "NEGATIVE"
    return "NEUTRAL"


# ── Analysis assembly ──────────────────────────────────────────────────────────

def create_analysis(stock_data, news_items):
    logging.info("🔍 Assembling sentiment analysis…")
    rows = []
    for news in news_items:
        sentiment = analyze_sentiment(news["title"])
        for ticker in news.get("tickers", []):
            if ticker not in stock_data:
                continue
            info = stock_data[ticker]
            rows.append({
                "Ticker":         ticker,
                "Company":        COMPANY_NAMES.get(ticker, ticker),
                "Headline":       news["title"],
                "Headline URL":   news.get("url", ""),
                "Published At":   news.get("published_at", ""),
                "Sentiment":      sentiment,
                "Regular Change": info["regular_change"],
                "After Hours":    info["post_change"],
                "Current Price":  info["current_price"],
                "Data Source":    info["source"],
                "News Source":    news["source"],
                "Timestamp":      datetime.now().isoformat(),
            })
    if not rows:
        logging.error("❌ No analysis rows created — check API keys and network")
        return None
    logging.info(f"✅ {len(rows)} analysis records created")
    return pd.DataFrame(rows)


# ── Output saving ──────────────────────────────────────────────────────────────

def save_results(df):
    if df is None or df.empty:
        logging.error("❌ Nothing to save")
        return False
    try:
        metadata = {
            "generated_at":    datetime.now().isoformat(),
            "data_sources":    list(df["Data Source"].unique()),
            "total_records":   len(df),
            "real_data":       True,
            "stocks_analyzed": list(df["Ticker"].unique()),
        }
        with open("results.json", "w") as f:
            json.dump({"metadata": metadata, "results": df.to_dict("records")}, f, indent=2)
        logging.info("✅ Saved results.json")

        # ── HTML report ──
        html = f"""<!DOCTYPE html>
<html>
<head>
  <title>LIVE Stock Sentiment Analysis</title>
  <style>
    body{{font-family:Arial,sans-serif;margin:20px;background:#f5f5f5}}
    .container{{max-width:1200px;margin:0 auto;background:#fff;padding:20px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,.1)}}
    .header{{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:20px;margin:-20px -20px 20px;border-radius:8px 8px 0 0}}
    .meta{{background:#e8f5e8;padding:15px;border-radius:5px;margin:20px 0}}
    .card{{border:1px solid #ddd;margin:10px 0;padding:15px;border-radius:5px}}
    .positive{{border-left:4px solid #28a745;background:#d4edda}}
    .negative{{border-left:4px solid #dc3545;background:#f8d7da}}
    .neutral{{border-left:4px solid #ffc107;background:#fff3cd}}
    .price{{font-size:18px;font-weight:bold}}
    .pos-chg{{color:#28a745;font-weight:bold}}
    .neg-chg{{color:#dc3545;font-weight:bold}}
    a{{color:#555;font-size:12px}}
  </style>
</head>
<body><div class="container">
  <div class="header">
    <h1>📈 LIVE Stock Sentiment Analysis</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
  </div>
  <div class="meta">
    <h3>📊 Summary</h3>
    <p><strong>Stocks:</strong> {len(df['Ticker'].unique())} &nbsp;
       <strong>Headlines:</strong> {len(df)} &nbsp;
       <strong>Sources:</strong> {', '.join(df['Data Source'].unique())}</p>
    <p>Sentiment — Positive: {(df['Sentiment']=='POSITIVE').sum()} &nbsp;
       Negative: {(df['Sentiment']=='NEGATIVE').sum()} &nbsp;
       Neutral: {(df['Sentiment']=='NEUTRAL').sum()}</p>
  </div>
  <h2>📰 Live Results</h2>
"""
        for _, row in df.sort_values("Regular Change", ascending=False).iterrows():
            cls    = row["Sentiment"].lower()
            chg_cls = "pos-chg" if row["Regular Change"] >= 0 else "neg-chg"
            sym    = "+" if row["Regular Change"] >= 0 else ""
            link   = f'<a href="{row["Headline URL"]}" target="_blank">Read article →</a>' if row.get("Headline URL") else ""
            pub    = f' | {row["Published At"][:10]}' if row.get("Published At") else ""
            html += f"""
  <div class="card {cls}">
    <div style="display:flex;justify-content:space-between;align-items:center">
      <div>
        <h3 style="margin:0">{row['Ticker']} — {row['Company']}</h3>
        <p style="margin:5px 0;font-style:italic">"{row['Headline']}"</p>
        <p style="margin:5px 0;font-size:13px;color:#666">
          {row['News Source']}{pub} | Data: {row['Data Source']} | Sentiment: <strong>{row['Sentiment']}</strong>
          &nbsp;{link}
        </p>
      </div>
      <div style="text-align:right;min-width:100px">
        <div class="price">${row['Current Price']:.2f}</div>
        <div class="{chg_cls}">{sym}{row['Regular Change']:.2f}%</div>
        {'<div style="font-size:12px">AH: ' + f"{row['After Hours']:+.2f}%" + '</div>' if abs(row['After Hours']) > 0.01 else ''}
      </div>
    </div>
  </div>"""

        html += "\n</div></body></html>"
        with open("results.html", "w") as f:
            f.write(html)
        logging.info("✅ Saved results.html")

        # ── Text summary ──
        with open("summary.txt", "w") as f:
            f.write("LIVE STOCK SENTIMENT ANALYSIS\n" + "="*50 + "\n")
            f.write(f"Generated : {datetime.now()}\n")
            f.write(f"Data sources: {', '.join(df['Data Source'].unique())}\n\n")
            f.write(f"Records   : {len(df)}\n")
            f.write(f"Stocks    : {len(df['Ticker'].unique())}\n\n")
            f.write(f"SENTIMENT\n  Positive : {(df['Sentiment']=='POSITIVE').sum()}\n")
            f.write(f"  Negative : {(df['Sentiment']=='NEGATIVE').sum()}\n")
            f.write(f"  Neutral  : {(df['Sentiment']=='NEUTRAL').sum()}\n\n")
            f.write("TOP GAINERS\n")
            for _, r in df.nlargest(5, "Regular Change").iterrows():
                f.write(f"  {r['Ticker']}: {r['Regular Change']:+.2f}% (${r['Current Price']:.2f}) {r['Sentiment']}\n")
            f.write("\nTOP LOSERS\n")
            for _, r in df.nsmallest(5, "Regular Change").iterrows():
                f.write(f"  {r['Ticker']}: {r['Regular Change']:+.2f}% (${r['Current Price']:.2f}) {r['Sentiment']}\n")
        logging.info("✅ Saved summary.txt")
        return True
    except Exception as e:
        logging.error(f"❌ Save failed: {e}")
        return False


# ── Entry point ────────────────────────────────────────────────────────────────

def run_analysis():
    logging.info("🚀 Starting LIVE Stock Sentiment Analysis…")

    stock_data = fetch_real_stock_data()
    if not stock_data:
        logging.error("❌ No stock price data — aborting")
        return None

    # ── KEY CHANGE: fetch real headlines instead of hardcoded ones ──
    news_items = fetch_live_news(list(stock_data.keys()))
    if not news_items:
        logging.error("❌ No news articles fetched — check NEWSAPI_KEY")
        return None

    results_df = create_analysis(stock_data, news_items)
    if results_df is None:
        return None

    if save_results(results_df):
        logging.info("✅ Analysis complete!")
        print(f"\n🎉 LIVE ANALYSIS COMPLETE")
        print(f"📈 Price sources : {', '.join(results_df['Data Source'].unique())}")
        print(f"📊 Stocks        : {len(results_df['Ticker'].unique())}")
        print(f"📰 Headlines     : {len(results_df)}")

        print("\n🏆 TOP MOVERS:")
        for _, r in results_df.nlargest(3, "Regular Change").iterrows():
            print(f"  {r['Ticker']}: {r['Regular Change']:+.2f}%  ${r['Current Price']:.2f}  ({r['Sentiment']})")

        print("\n📉 BIGGEST DECLINES:")
        for _, r in results_df.nsmallest(3, "Regular Change").iterrows():
            print(f"  {r['Ticker']}: {r['Regular Change']:+.2f}%  ${r['Current Price']:.2f}  ({r['Sentiment']})")

        return results_df
    return None


if __name__ == "__main__":
    results = run_analysis()
    if results is not None:
        print("\n✅ SUCCESS — check results.json, results.html, summary.txt")
        exit(0)
    else:
        print("\n❌ FAILED — see log output above")
        exit(1)
