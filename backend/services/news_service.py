import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, Any, List

COMPANY_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet Google",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "JPM": "JPMorgan Chase",
    "KO": "Coca-Cola",
    "WMT": "Walmart",
    "DIS": "Walt Disney",
    "IBM": "IBM",
    "RELIANCE.NS": "Reliance Industries",
    "RELIANCE.BO": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services TCS",
    "TCS.BO": "Tata Consultancy Services TCS",
    "INFY.NS": "Infosys",
    "INFY.BO": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
}

POSITIVE_WORDS = {
    "surge", "surges", "gain", "gains", "jump", "jumps", "rally", "rallies",
    "profit", "profits", "beat", "beats", "record", "growth", "high", "highs",
    "upgrade", "upgrades", "bull", "bullish", "soar", "soars", "boost", "boosts",
    "dividend", "strong", "outperform", "opportunity", "expansion"
}

NEGATIVE_WORDS = {
    "drop", "drops", "fall", "falls", "plunge", "plunges", "loss", "losses",
    "miss", "misses", "downgrade", "downgrades", "bear", "bearish", "decline",
    "declines", "weak", "weakness", "slump", "slumps", "probe", "lawsuit",
    "selloff", "risk", "risks", "warning", "warnings", "cut", "cuts", "inflation"
}


class NewsService:
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 600  # 10 minutes

    def _get_search_query(self, symbol: str) -> str:
        clean = symbol.upper().strip()
        name = COMPANY_NAMES.get(clean, clean)
        # For Indian NSE/BSE stocks, append 'India stock'
        if clean.endswith(".NS") or clean.endswith(".BO"):
            base = clean.split(".")[0]
            return f"{name} {base} stock India"
        return f"{name} {clean} stock"

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        words = set(re.findall(r"\b[a-z]+\b", text.lower()))
        pos = len(words.intersection(POSITIVE_WORDS))
        neg = len(words.intersection(NEGATIVE_WORDS))

        if pos > neg:
            sentiment = "Bullish"
            badge_class = "badge-profit"
        elif neg > pos:
            sentiment = "Bearish"
            badge_class = "badge-loss"
        else:
            sentiment = "Neutral"
            badge_class = "badge-neutral"

        return {
            "sentiment": sentiment,
            "badge_class": badge_class,
            "score": pos - neg
        }

    def fetch_stock_news(self, symbol: str, limit: int = 6) -> Dict[str, Any]:
        now = time.time()
        cache_key = symbol.upper()
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if now - entry["cached_at"] < self._cache_ttl:
                return entry["data"]

        query = self._get_search_query(symbol)
        encoded_query = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

        articles: List[Dict[str, Any]] = []
        overall_score = 0

        try:
            req = urllib.request.Request(
                rss_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            with urllib.request.urlopen(req, timeout=6) as response:
                content = response.read()

            root = ET.fromstring(content)
            items = root.findall(".//item")

            for item in items[:limit]:
                raw_title = item.findtext("title") or "No title"
                link = item.findtext("link") or "#"
                pub_date = item.findtext("pubDate") or ""
                source = item.findtext("source") or ""

                # Extract source if formatted as "Title - Source"
                title = raw_title
                if " - " in raw_title and not source:
                    parts = raw_title.rsplit(" - ", 1)
                    title = parts[0]
                    source = parts[1]

                sentiment_meta = self._analyze_sentiment(title)
                overall_score += sentiment_meta["score"]

                # Clean pub date to shorter format
                display_date = pub_date
                if len(pub_date) > 16:
                    display_date = pub_date[:16]

                articles.append({
                    "title": title,
                    "source": source or "Financial News",
                    "link": link,
                    "published": display_date,
                    "sentiment": sentiment_meta["sentiment"],
                    "badge_class": sentiment_meta["badge_class"]
                })

        except Exception as err:
            # Fallback placeholder when offline or connection is blocked
            articles = [{
                "title": f"Live news currently unavailable for {symbol}. (Check internet connectivity)",
                "source": "Market Feed",
                "link": f"https://www.google.com/search?q={urllib.parse.quote(query)}",
                "published": "Recent",
                "sentiment": "Neutral",
                "badge_class": "badge-neutral"
            }]

        if overall_score > 0:
            market_sentiment = "Bullish"
            sentiment_class = "pred-profit"
        elif overall_score < 0:
            market_sentiment = "Bearish"
            sentiment_class = "pred-loss"
        else:
            market_sentiment = "Neutral"
            sentiment_class = "pred-neutral"

        result = {
            "symbol": symbol.upper(),
            "company_name": COMPANY_NAMES.get(symbol.upper(), symbol.upper()),
            "articles": articles,
            "overall_sentiment": market_sentiment,
            "sentiment_class": sentiment_class,
            "sentiment_score": overall_score,
            "count": len(articles),
            "source_type": "Google Financial RSS Feed (Real-Time)"
        }

        self._cache[cache_key] = {"data": result, "cached_at": now}
        return result


news_service = NewsService()
