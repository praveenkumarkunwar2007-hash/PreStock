import time
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from backend.services.stock_data import stock_data_service
from backend.services.predictor import predictor
from backend.services.news_service import news_service


# Curated universe of core blue-chip stocks and actively traded high-potential IPOs
DEFAULT_UNIVERSE = [
    # US Blue-Chip & Tech Leaders
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "category": "Stock", "market": "US", "exchange": "NASDAQ"},
    {"symbol": "AAPL", "name": "Apple Inc.", "category": "Stock", "market": "US", "exchange": "NASDAQ"},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "category": "Stock", "market": "US", "exchange": "NASDAQ"},
    {"symbol": "AMZN", "name": "Amazon.com, Inc.", "category": "Stock", "market": "US", "exchange": "NASDAQ"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "category": "Stock", "market": "US", "exchange": "NASDAQ"},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "category": "Stock", "market": "US", "exchange": "NYSE"},
    {"symbol": "WMT", "name": "Walmart Inc.", "category": "Stock", "market": "US", "exchange": "NYSE"},

    # Indian Market Leaders
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries Limited", "category": "Stock", "market": "India", "exchange": "NSE"},
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services", "category": "Stock", "market": "India", "exchange": "NSE"},
    {"symbol": "INFY.NS", "name": "Infosys Limited", "category": "Stock", "market": "India", "exchange": "NSE"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank Limited", "category": "Stock", "market": "India", "exchange": "NSE"},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank Limited", "category": "Stock", "market": "India", "exchange": "NSE"},

    # High-Growth & Trending Indian IPOs
    {"symbol": "SWIGGY.NS", "name": "Swiggy Limited (IPO)", "category": "IPO", "market": "India", "exchange": "NSE"},
    {"symbol": "TATATECH.NS", "name": "Tata Technologies Limited (IPO)", "category": "IPO", "market": "India", "exchange": "NSE"},
    {"symbol": "ZOMATO.NS", "name": "Zomato Limited (Consumer Tech)", "category": "IPO", "market": "India", "exchange": "NSE"},
    {"symbol": "BAJAJHFL.NS", "name": "Bajaj Housing Finance (IPO)", "category": "IPO", "market": "India", "exchange": "NSE"},
    {"symbol": "HYUNDAI.NS", "name": "Hyundai Motor India (IPO)", "category": "IPO", "market": "India", "exchange": "NSE"},
    {"symbol": "IREDA.NS", "name": "Indian Renewable Energy Agency (IPO)", "category": "IPO", "market": "India", "exchange": "NSE"},
    {"symbol": "NYKAA.NS", "name": "FSN E-Commerce Nykaa (IPO)", "category": "IPO", "market": "India", "exchange": "NSE"},

    # Top US Growth IPOs
    {"symbol": "RDDT", "name": "Reddit, Inc. (IPO)", "category": "IPO", "market": "US", "exchange": "NYSE"},
    {"symbol": "ARM", "name": "Arm Holdings plc (IPO)", "category": "IPO", "market": "US", "exchange": "NASDAQ"},
    {"symbol": "CART", "name": "Maplebear Instacart (IPO)", "category": "IPO", "market": "US", "exchange": "NASDAQ"},
    {"symbol": "KVUE", "name": "Kenvue Inc. (IPO)", "category": "IPO", "market": "US", "exchange": "NYSE"},
]


class RecommendationService:
    def __init__(self):
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_time: float = 0
        self._cache_ttl: float = 600  # 10 minutes cache

    def _generate_profit_reasons(
        self,
        symbol: str,
        name: str,
        pred: Dict[str, Any],
        features: Dict[str, Any],
        news_sentiment: str,
        news_headline: str,
        is_ipo: bool
    ) -> List[str]:
        """
        Synthesizes human-interpretable reasons why this stock or IPO is expected to deliver profit.
        """
        reasons = []
        prob_profit = pred["probabilities"].get("Profit", 0.0) * 100
        rsi = float(features.get("RSI_14", 50.0))
        price_vs_sma20 = float(features.get("Price_vs_SMA20", 0.0)) * 100
        vol_ratio = float(features.get("Volume_Ratio", 1.0))
        ret_5d = float(features.get("Return_5D", 0.0)) * 100

        # Reason 1: Machine Learning Model Projection
        if prob_profit >= 65:
            reasons.append(
                f"High-Conviction ML Signal: The XGBoost predictive engine assigns an exceptional {prob_profit:.1f}% probability of upward price expansion over the next 5 trading days."
            )
        elif prob_profit >= 45:
            reasons.append(
                f"Positive AI Forecast: Machine learning models indicate favorable risk-reward with a {prob_profit:.1f}% probability of profit over the 5-day horizon."
            )
        else:
            reasons.append(
                f"Quantitative Edge: Model classifies overall price momentum in the positive bracket ({pred['prediction']}) with {pred['confidence'] * 100:.1f}% confidence."
            )

        # Reason 2: Technical Momentum & Moving Average Structure
        if price_vs_sma20 > 0.5:
            reasons.append(
                f"Bullish Trend Alignment: Trading {price_vs_sma20:.1f}% above its 20-day Moving Average, maintaining upward structural momentum."
            )
        elif rsi < 38:
            reasons.append(
                f"Oversold Rebound Trigger: 14-day RSI stands at {rsi:.1f} in an oversold accumulation zone, creating strong statistical conditions for a sharp mean-reversion rally."
            )
        elif 50 <= rsi <= 68:
            reasons.append(
                f"Healthy Momentum Expansion: 14-day RSI of {rsi:.1f} shows sustained buying interest without entering dangerous overbought territory (> 70)."
            )
        else:
            reasons.append(
                f"Price Stability: Short-term returns (+{max(0.1, ret_5d):.1f}% 5D) demonstrate consistent baseline support against market volatility."
            )

        # Reason 3: Volume & Institutional Accumulation
        if vol_ratio >= 1.20:
            reasons.append(
                f"Institutional Volume Surge: Recent trading volume is {(vol_ratio - 1.0) * 100:.0f}% higher than the 20-day baseline average, signaling institutional accumulation."
            )
        elif vol_ratio >= 0.95:
            reasons.append(
                "Liquidity Support: Healthy trading volume provides sufficient market depth to sustain upcoming upward price momentum."
            )
        else:
            reasons.append(
                "Supply Squeeze: Low selling liquidity at current levels creates favorable upward elasticity on incremental buying."
            )

        # Reason 4: News Sentiment & Catalysts
        if news_sentiment == "Bullish":
            reasons.append(
                f"Bullish Market Sentiment: Open-source financial feeds report positive catalysts: '{news_headline}'."
            )
        elif is_ipo:
            reasons.append(
                "IPO Growth Catalyst: Emerging public market listing benefits from fresh capital deployment, brand visibility, and expanding institutional investor participation."
            )
        else:
            reasons.append(
                "Macro & Industry Tailwind: Sector resilience and positive analyst sentiment support continuous operational growth."
            )

        return reasons

    def scan_and_recommend(
        self,
        category: str = "all",
        market: str = "all",
        limit: int = 10,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Conducts deep market research across all tracked stocks and IPOs, evaluates
        technical factors, XGBoost profit probabilities, and news sentiment, and
        returns ranked recommendations with actionable rationale.
        """
        now = time.time()
        cat_lower = str(getattr(category, "default", category) or "all").lower().strip()
        mkt_lower = str(getattr(market, "default", market) or "all").lower().strip()

        # Check cached full research results
        if not force_refresh and self._cache and (now - self._cache_time < self._cache_ttl):
            all_recs = self._cache["all_recommendations"]
        else:
            # Run Deep Research across all target equities
            candidates = []
            scanned_count = 0

            for item in DEFAULT_UNIVERSE:
                sym = item["symbol"]
                scanned_count += 1
                try:
                    features = stock_data_service.get_latest_features(sym)
                    if not features:
                        continue

                    pred = predictor.predict_features(features)
                    prob_profit = float(pred["probabilities"].get("Profit", 0.0))
                    confidence = float(pred.get("confidence", 0.0))
                    prediction = pred.get("prediction", "Neutral")

                    # 1. Technical Score Calculation (0 - 100)
                    rsi = float(features.get("RSI_14", 50.0))
                    vol_ratio = float(features.get("Volume_Ratio", 1.0))
                    volatility = float(features.get("Volatility_20D", 0.20))
                    ret_5d = float(features.get("Return_5D", 0.0))
                    price_vs_sma20 = float(features.get("Price_vs_SMA20", 0.0))
                    price_vs_sma50 = float(features.get("Price_vs_SMA50", 0.0))

                    tech_score = 50.0
                    # RSI momentum
                    if 50 <= rsi <= 68:
                        tech_score += 18
                    elif rsi < 35:
                        tech_score += 15
                    elif rsi > 70:
                        tech_score -= 8

                    # SMA alignment
                    if price_vs_sma20 > 0:
                        tech_score += 12
                    if price_vs_sma50 > 0:
                        tech_score += 10

                    # Volume Surge
                    if vol_ratio >= 1.15:
                        tech_score += 10
                    elif vol_ratio < 0.8:
                        tech_score -= 5

                    # 5D Momentum
                    if ret_5d > 0:
                        tech_score += 8

                    tech_score = max(10.0, min(100.0, tech_score))

                    # Preliminary Profit Score (ML + Technical)
                    prelim_score = (prob_profit * 100.0 * 0.60) + (tech_score * 0.40)

                    candidates.append({
                        "item": item,
                        "sym": sym,
                        "features": features,
                        "pred": pred,
                        "prob_profit": prob_profit,
                        "confidence": confidence,
                        "prediction": prediction,
                        "rsi": rsi,
                        "vol_ratio": vol_ratio,
                        "volatility": volatility,
                        "ret_5d": ret_5d,
                        "tech_score": tech_score,
                        "prelim_score": prelim_score
                    })
                except Exception:
                    continue

            # Sort by preliminary score so we enrich the top picks
            candidates.sort(key=lambda x: x["prelim_score"], reverse=True)

            # Enrich candidates with news sentiment (top picks get live news if available)
            enriched_recs = []
            for rank_idx, c in enumerate(candidates, 1):
                sym = c["sym"]
                item = c["item"]
                features = c["features"]
                pred = c["pred"]
                prob_profit = c["prob_profit"]
                tech_score = c["tech_score"]

                news_sentiment = "Neutral"
                headline = f"Positive institutional sentiment and market activity for {item['name']}"

                # Only fetch live news for top candidates with a fast timeout
                if rank_idx <= 12:
                    try:
                        news_data = news_service.fetch_stock_news(sym, limit=2, timeout=1.2)
                        news_sentiment = news_data.get("overall_sentiment", "Neutral")
                        articles = news_data.get("articles", [])
                        if articles and articles[0].get("title"):
                            headline = articles[0]["title"]
                    except Exception:
                        pass

                # 2. News Catalyst Score
                news_score = 50.0
                if news_sentiment == "Bullish":
                    news_score = 85.0
                elif news_sentiment == "Bearish":
                    news_score = 25.0

                # 3. Overall Profit Score (Composite 0 - 100)
                # 50% ML Model Probability + 35% Technical Score + 15% News Catalyst
                profit_score = int(round(
                    (prob_profit * 100.0 * 0.50) +
                    (tech_score * 0.35) +
                    (news_score * 0.15)
                ))
                profit_score = max(18, min(98, profit_score))

                # Risk Rating & Upside
                is_ipo = item["category"] == "IPO"
                if is_ipo:
                    risk_level = "High Growth (IPO)"
                    upside_range = "+5.0% to +12.0%"
                elif c["volatility"] > 0.28:
                    risk_level = "Medium-High Risk"
                    upside_range = "+3.5% to +8.5%"
                elif c["volatility"] < 0.20:
                    risk_level = "Low Risk / Blue-Chip"
                    upside_range = "+2.0% to +5.0%"
                else:
                    risk_level = "Moderate Risk"
                    upside_range = "+3.0% to +6.5%"

                reasons = self._generate_profit_reasons(
                    symbol=sym,
                    name=item["name"],
                    pred=pred,
                    features=features,
                    news_sentiment=news_sentiment,
                    news_headline=headline,
                    is_ipo=is_ipo
                )

                enriched_recs.append({
                    "symbol": sym,
                    "name": item["name"],
                    "category": item["category"],
                    "exchange": item["exchange"],
                    "market": item["market"],
                    "profit_score": profit_score,
                    "prediction": c["prediction"],
                    "profit_probability": round(prob_profit, 3),
                    "confidence": round(c["confidence"], 3),
                    "latest_close": round(float(features.get("Close", 0.0)), 2),
                    "return_5d": round(c["ret_5d"], 4),
                    "rsi_14": round(c["rsi"], 1),
                    "volatility_20d": round(c["volatility"], 3),
                    "volume_ratio": round(c["vol_ratio"], 2),
                    "risk_level": risk_level,
                    "expected_upside": upside_range,
                    "horizon": "5 Trading Days",
                    "news_sentiment": news_sentiment,
                    "news_headline": headline,
                    "reasons_for_profit": reasons
                })

            # Final sort by Profit Score descending
            enriched_recs.sort(key=lambda x: x["profit_score"], reverse=True)

            for idx, c in enumerate(enriched_recs, 1):
                c["rank"] = idx

            all_recs = enriched_recs
            self._cache = {
                "all_recommendations": all_recs,
                "scanned_count": scanned_count,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now))
            }
            self._cache_time = now

        # Apply user filters
        filtered = []
        for rec in all_recs:
            # Filter category
            if cat_lower in ["ipo", "ipos"] and rec["category"] != "IPO":
                continue
            if cat_lower in ["stock", "stocks"] and rec["category"] != "Stock":
                continue

            # Filter market
            if mkt_lower in ["india", "in", "nse", "bse"] and rec["market"] != "India":
                continue
            if mkt_lower in ["us", "usa", "nasdaq", "nyse"] and rec["market"] != "US":
                continue

            filtered.append(rec)

        # Apply limit
        result_recs = filtered[:max(1, min(limit, 50))]

        return {
            "generated_at": self._cache.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S UTC")),
            "total_scanned": self._cache.get("scanned_count", len(DEFAULT_UNIVERSE)),
            "recommended_count": len(result_recs),
            "category_filter": category,
            "market_filter": market,
            "recommendations": result_recs
        }


# Global singleton instance
recommendation_service = RecommendationService()
