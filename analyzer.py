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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ── Configuration ──────────────────────────────────────────────────────────────
CONFIG = {
    "stocks": ["AAPL", "MSFT", "TSLA", "GOOGL", "META", "AMZN", "NVDA", "JPM", "V", "MA"],
    "api_keys": {
        "alpha_vantage": os.getenv("ALPHAVANTAGE_KEY", "demo"),
        "newsapi":       os.getenv("NEWSAPI_KEY", ""),
    },
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

# ── Specific search queries to avoid irrelevant results ────────────────────────
TICKER_SEARCH_TERMS = {
    "AAPL":  '"AAPL" OR "Apple stock" OR "Apple earnings" OR "Apple shares"',
    "MSFT":  '"MSFT" OR "Microsoft stock" OR "Microsoft earnings" OR "Microsoft shares"',
    "TSLA":  '"TSLA" OR "Tesla stock" OR "Tesla earnings" OR "Tesla shares"',
    "GOOGL": '"GOOGL" OR "Alphabet stock" OR "Google stock" OR "Alphabet earnings"',
    "META":  '"META stock" OR "Meta Platforms stock" OR "Meta earnings" OR "Meta shares"',
    "AMZN":  '"AMZN" OR "Amazon stock" OR "Amazon earnings" OR "Amazon shares"',
    "NVDA":  '"NVDA" OR "Nvidia stock" OR "Nvidia earnings" OR "Nvidia shares"',
    "JPM":   '"JPM" OR "JPMorgan stock" OR "JPMorgan earnings" OR "JPMorgan shares"',
    "V":     '"Visa stock" OR "Visa Inc earnings" OR "Visa Inc shares" OR "Visa payment"',
    "MA":    '"Mastercard stock" OR "Mastercard earnings" OR "Mastercard shares"',
}

# Low-quality domains to skip
EXCLUDED_DOMAINS = {
    "ozbargain.com.au", "naturalnews.com", "timesofindia.indiatimes.com",
    "indianexpress.com", "theecologist.org", "notebookcheck.net",
}


# ── Stock price fetching ───────────────────────────────────────────────────────

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


# ── Live news via NewsAPI ──────────────────────────────────────────────────────

def is_relevant_article(title, ticker):
    """Reject articles that clearly aren't about the stock."""
    title_lower = title.lower()

    finance_words = {
        "stock", "share", "shares", "earnings", "investor", "market", "trading",
        "nasdaq", "nyse", "quarter", "revenue", "profit", "loss", "analyst",
        "forecast", "dividend", "valuation", "ipo", "sec", "fund", "hedge",
        "rally", "decline", "surge", "drop", "beat", "miss", "guidance",
        "outlook", "upgrade", "downgrade", "buyback", "acquisition", "merger",
        ticker.lower(),
    }

    company_fragments = {
        "AAPL":  ["apple"],
        "MSFT":  ["microsoft"],
        "TSLA":  ["tesla", "musk", "elon"],
        "GOOGL": ["google", "alphabet", "deepmind"],
        "META":  ["meta", "facebook", "instagram", "zuckerberg", "whatsapp"],
        "AMZN":  ["amazon", "aws", "bezos", "prime"],
        "NVDA":  ["nvidia", "jensen huang", "gpu", "cuda"],
        "JPM":   ["jpmorgan", "jamie dimon", "chase"],
        "V":     ["visa"],
        "MA":    ["mastercard"],
    }

    words_in_title = set(title_lower.replace(",", " ").replace("'", " ").split())
    frags = set(company_fragments.get(ticker, []))

    has_finance_word = bool(words_in_title & finance_words)
    has_company_word = any(f in title_lower for f in frags)

    return has_finance_word or has_company_word


def fetch_news_for_ticker(ticker, api_key, page_size=5):
    """Fetch and filter recent headlines for a single ticker."""
    query = TICKER_SEARCH_TERMS.get(ticker, f'"{ticker}" stock')
    url   = "https://newsapi.org/v2/everything"
    params = {
        "q":        query,
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": page_size,
        "from":     (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d"),
        "apiKey":   api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            logging.warning(f"⚠️  NewsAPI error for {ticker}: {data.get('message')}")
            return []

        results = []
        for art in data.get("articles", []):
            title  = art.get("title", "").strip()
            source = art.get("source", {}).get("name", "NewsAPI")
            url_   = art.get("url", "")

            if not title or title.lower() == "[removed]":
                continue

            domain = url_.split("/")[2].replace("www.", "") if url_ else ""
            if domain in EXCLUDED_DOMAINS:
                continue

            if not is_relevant_article(title, ticker):
                logging.info(f"  ↳ Skipped irrelevant: {title[:60]}…")
                continue

            results.append({
                "title":        title,
                "source":       source,
                "url":          url_,
                "published_at": art.get("publishedAt", ""),
                "tickers":      [ticker],
            })

            if len(results) >= CONFIG["newsapi_articles_per_ticker"]:
                break

        logging.info(f"✅ NewsAPI: {len(results)} relevant articles for {ticker}")
        return results

    except Exception as e:
        logging.warning(f"⚠️  NewsAPI failed for {ticker}: {e}")
        return []


def fetch_live_news(tickers):
    api_key = CONFIG["api_keys"]["newsapi"]
    if not api_key:
        logging.error("❌ NEWSAPI_KEY not set")
        return []

    all_news = []
    for ticker in tickers:
        articles = fetch_news_for_ticker(ticker, api_key, page_size=5)
        all_news.extend(articles)
        time.sleep(0.25)

    seen, unique = set(), []
    for item in all_news:
        if item["title"] not in seen:
            seen.add(item["title"])
            unique.append(item)

    logging.info(f"✅ Total unique relevant headlines: {len(unique)}")
    return unique


# ── Sentiment analysis ─────────────────────────────────────────────────────────

POSITIVE_WORDS = {
    "surge", "surges", "surging", "rally", "rallies", "rallying", "soar", "soars", "soaring",
    "rise", "rises", "rising", "gain", "gains", "climb", "climbs", "jump", "jumps",
    "rebound", "rebounds", "recover", "recovery",
    "beat", "beats", "topped", "exceed", "exceeds", "outperform", "outperforms",
    "record", "profit", "profits", "strong", "strength", "robust", "solid",
    "growth", "grew", "accelerate", "accelerates", "expand", "expansion",
    "upgrade", "upgrades", "upgraded", "buy", "overweight", "bullish",
    "optimism", "optimistic", "confidence", "boost", "positive",
    "deal", "partnership", "wins", "awarded", "launches", "breakthrough",
    "innovation", "momentum", "resilience", "resilient",
}

NEGATIVE_WORDS = {
    "fall", "falls", "falling", "drop", "drops", "dropping", "decline", "declines",
    "declining", "plunge", "plunges", "plunging", "sink", "sinks", "tumble", "tumbles",
    "slide", "slides", "slump", "slumps", "crash", "crashes",
    "miss", "misses", "missed", "disappoint", "disappoints", "disappointing",
    "weak", "weakness", "loss", "losses", "struggle", "struggles", "shrink",
    "slowdown", "slowing", "cut", "cuts", "reduce", "reduction",
    "downgrade", "downgrades", "downgraded", "sell", "underweight", "bearish",
    "concern", "concerns", "risk", "risks", "pressure", "negative", "warn", "warning",
    "lawsuit", "investigation", "probe", "fine", "layoff", "layoffs", "recall",
    "delay", "delays", "halt", "suspended", "breach", "hack",
}


def analyze_sentiment(text):
    tokens = set(t.strip(".,!?\"'();:-").lower() for t in text.split())
    pos = len(tokens & POSITIVE_WORDS)
    neg = len(tokens & NEGATIVE_WORDS)
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
        logging.error("❌ No analysis rows — check API keys and network")
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
            cls     = row["Sentiment"].lower()
            chg_cls = "pos-chg" if row["Regular Change"] >= 0 else "neg-chg"
            sym     = "+" if row["Regular Change"] >= 0 else ""
            link    = f'<a href="{row["Headline URL"]}" target="_blank">Read article →</a>' if row.get("Headline URL") else ""
            pub     = f' | {row["Published At"][:10]}' if row.get("Published At") else ""
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

        with open("summary.txt", "w") as f:
            f.write("LIVE STOCK SENTIMENT ANALYSIS\n" + "="*50 + "\n")
            f.write(f"Generated : {datetime.now()}\n")
            f.write(f"Sources   : {', '.join(df['Data Source'].unique())}\n\n")
            f.write(f"Records   : {len(df)}\n")
            f.write(f"Stocks    : {len(df['Ticker'].unique())}\n\n")
            f.write("SENTIMENT\n")
            f.write(f"  Positive : {(df['Sentiment']=='POSITIVE').sum()}\n")
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
