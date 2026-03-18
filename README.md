# 📈 Stock Sentiment Analyzer

A fully automated pipeline that fetches live stock prices and real financial headlines twice daily, performs sentiment analysis on each headline, and publishes the results as a public website via GitHub Pages.

**Live site:** https://robertnimmons.github.io/stock-sentiment-analyzer/

---

## What it does

- Fetches real-time prices for 10 major stocks (AAPL, MSFT, TSLA, GOOGL, META, AMZN, NVDA, JPM, V, MA)
- Pulls recent financial headlines for each ticker from NewsAPI
- Classifies each headline as POSITIVE, NEGATIVE, or NEUTRAL using keyword-based sentiment analysis
- Publishes results as a live webpage, updated automatically on a schedule
- Archives historical JSON snapshots by date

---

## How it runs

A GitHub Actions workflow triggers automatically on weekdays:

| Schedule | Time (UTC) | Purpose |
|---|---|---|
| Morning run | 9:00 AM Mon–Fri | Regular market hours analysis |
| Evening run | 9:00 PM Mon–Fri | After-hours analysis |

You can also trigger it manually from the **Actions** tab in GitHub using the `workflow_dispatch` event.

---

## Project structure

```
├── analyzer.py                  # Main script — fetches prices, news, runs analysis
├── .github/
│   └── workflows/
│       └── stock-analysis.yml   # GitHub Actions workflow
├── results.html                 # Latest analysis (copied to index.html for Pages)
├── results.json                 # Latest results as structured JSON
├── latest.json                  # Symlink to most recent results
├── summary.txt                  # Plain text summary of latest run
├── index.html                   # GitHub Pages entry point
├── api/
│   └── data.json                # JSON API endpoint
└── data/
    └── YYYY/
        └── MM/
            └── YYYY-MM-DD.json  # Historical archive by date
```

---

## Setup

### 1. Fork or clone this repo

```bash
git clone https://github.com/robertnimmons/stock-sentiment-analyzer.git
cd stock-sentiment-analyzer
```

### 2. Add API keys as GitHub Secrets

Go to **Settings → Secrets and variables → Actions → New repository secret** and add:

| Secret name | Where to get it |
|---|---|
| `ALPHAVANTAGE_KEY` | [alphavantage.co](https://www.alphavantage.co/support/#api-key) — free tier |
| `NEWSAPI_KEY` | [newsapi.org/account](https://newsapi.org/account) — free tier |

### 3. Enable GitHub Pages

Go to **Settings → Pages** and set the source to **GitHub Actions**.

### 4. Run it

Either wait for the next scheduled run, or go to **Actions → Daily Stock Analysis with Website Update → Run workflow** to trigger it manually.

---

## Data sources

**Stock prices** — tried in this order per ticker:
1. **Alpha Vantage** (if `ALPHAVANTAGE_KEY` is set) — most reliable
2. **Yahoo Finance direct API** — free fallback, no key required
3. **yfinance library** — last resort fallback

**News headlines** — fetched from **NewsAPI** (`/v2/everything` endpoint), filtered to:
- Only articles published in the last 3 days
- Only articles that pass a relevance check (must contain finance keywords or company name fragments)
- Excluding known low-quality domains

---

## Sentiment analysis

Headlines are scored by matching tokens against two word lists:

- **POSITIVE** — surge, rally, beat, upgrade, profit, growth, rebound, breakthrough, etc.
- **NEGATIVE** — decline, drop, miss, downgrade, layoff, lawsuit, warning, crash, etc.

If positive matches > negative matches → `POSITIVE`  
If negative matches > positive matches → `NEGATIVE`  
Otherwise → `NEUTRAL`

---

## API

The latest results are available as JSON at:

```
https://robertnimmons.github.io/stock-sentiment-analyzer/api/data.json
```

Historical snapshots are available at:

```
https://robertnimmons.github.io/stock-sentiment-analyzer/data/YYYY/MM/YYYY-MM-DD.json
```

---

## API key limits

| Service | Free tier limit | Usage per run |
|---|---|---|
| Alpha Vantage | 25 requests/day | 10 (one per ticker) |
| NewsAPI | 100 requests/day | 10 (one per ticker) |

Two runs per day = 20 requests each, well within free tier limits for both services.

> **Note:** NewsAPI's free plan only allows requests from `localhost`. If you see 401 errors, verify your key at [newsapi.org/account](https://newsapi.org/account) and confirm the secret name matches exactly `NEWSAPI_KEY` in your repo settings.

---

## Local development

```bash
pip install yfinance==0.2.18 pandas requests

export ALPHAVANTAGE_KEY=your_key_here
export NEWSAPI_KEY=your_key_here

python analyzer.py
```

Results are written to `results.html`, `results.json`, and `summary.txt` in the current directory.

---

## Customizing the stock list

Edit the `CONFIG` dict at the top of `analyzer.py`:

```python
CONFIG = {
    "stocks": ["AAPL", "MSFT", "TSLA", ...],  # add or remove tickers here
    ...
}
```

Also add a matching entry to `TICKER_SEARCH_TERMS` and `COMPANY_NAMES` for best results.
