import time
import json
import logging
import urllib.request
import urllib.parse
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import yfinance as yf

from training.preprocess import calculate_features

import socket
import urllib.error

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"


class OnlineStockFetcher:
    """
    High-performance online stock market fetcher with multiple fallback sources:
    1. Direct Yahoo Finance v8 Chart JSON (instant sub-second OHLCV & real-time quotes)
    2. Groww Public Quotes API (for live Indian NSE/BSE equities & IPOs)
    3. yfinance engine (standard resilience fallback)
    """

    def __init__(self, cache_ttl_seconds: int = 180):
        self.cache_ttl = cache_ttl_seconds
        # In-memory live cache: symbol -> {"df": pd.DataFrame, "history": List[dict], "features": dict, "fetched_at": float}
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._circuit_open_until: float = 0.0

    def _get_from_cache(self, symbol: str) -> Optional[Dict[str, Any]]:
        clean = symbol.upper().strip()
        item = self._cache.get(clean)
        if item and (time.time() - item["fetched_at"] < self.cache_ttl):
            return item
        return None

    def fetch_live_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch latest online prices, complete OHLCV history, and compute all technical indicators.
        Returns dict with keys: 'df', 'history', 'features', 'meta', 'source'.
        """
        clean_sym = symbol.upper().strip()

        # Check in-memory TTL cache first for instant response
        cached = self._get_from_cache(clean_sym)
        if cached:
            return cached

        # If network was recently determined to be unreachable (e.g. offline sandbox), fast exit
        if time.time() < self._circuit_open_until:
            return None

        # Resolve candidates (e.g. RELIANCE -> RELIANCE.NS, RELIANCE.BO)
        candidates = [clean_sym]
        if "." not in clean_sym:
            # Common Indian equity aliases
            candidates.extend([f"{clean_sym}.NS", f"{clean_sym}.BO"])

        for cand in candidates:
            # Strategy 1: Direct Yahoo v8 Chart API (ultra fast & global coverage)
            data = self._fetch_via_yahoo_chart(cand)
            if data is not None:
                self._save_to_cache(clean_sym, cand, data)
                return data

            # If network error triggered circuit breaker, do not attempt slower fallbacks
            if time.time() < self._circuit_open_until:
                return None

            # Strategy 2: For Indian equities, try Groww + Yahoo fallback
            if cand.endswith(".NS") or "." not in clean_sym:
                base_ticker = clean_sym.replace(".NS", "").replace(".BO", "")
                groww_data = self._fetch_groww_quote(base_ticker)
                if groww_data:
                    yf_data = self._fetch_via_yfinance(cand)
                    if yf_data is not None:
                        if "Close" in yf_data["features"] and groww_data.get("ltp"):
                            yf_data["features"]["Close"] = float(groww_data["ltp"])
                            yf_data["features"]["Price_Source"] = "Groww / NSE Live"
                        self._save_to_cache(clean_sym, cand, yf_data)
                        return yf_data

            if time.time() < self._circuit_open_until:
                return None

            # Strategy 3: Standard yfinance fallback
            yf_data = self._fetch_via_yfinance(cand)
            if yf_data is not None:
                self._save_to_cache(clean_sym, cand, yf_data)
                return yf_data

        return None

    def _fetch_via_yahoo_chart(self, cand_sym: str) -> Optional[Dict[str, Any]]:
        """Fetch 6 months of daily candles directly from Yahoo v8 chart endpoint."""
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(cand_sym)}?interval=1d&range=6mo"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                raw_json = json.loads(resp.read().decode())

            result = raw_json.get("chart", {}).get("result")
            if not result or len(result) == 0:
                return None

            chart_data = result[0]
            meta = chart_data.get("meta", {})
            timestamps = chart_data.get("timestamp", [])
            indicators = chart_data.get("indicators", {})
            quote_list = indicators.get("quote", [])
            if not timestamps or not quote_list or len(quote_list) == 0:
                return None

            q = quote_list[0]
            opens = q.get("open", [])
            highs = q.get("high", [])
            lows = q.get("low", [])
            closes = q.get("close", [])
            volumes = q.get("volume", [])

            dates = [datetime.fromtimestamp(ts) for ts in timestamps]
            df = pd.DataFrame({
                "Date": dates,
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": closes,
                "Volume": volumes
            }).dropna()

            if df.empty or len(df) < 5:
                return None

            df["Adj Close"] = df["Close"]
            df["Symbol"] = cand_sym

            # Compute technical indicators using the model feature pipeline
            feat_df = calculate_features(df, verbose=False, min_rows=5)
            if feat_df.empty:
                return None

            # Get latest row
            last_row = feat_df.iloc[-1].to_dict()
            last_row["Symbol"] = cand_sym
            last_close = float(last_row.get("Close", 1.0) or 1.0)
            if last_close <= 0:
                last_close = 1.0

            # Override with real-time regularMarketPrice if available and recent
            live_price = meta.get("regularMarketPrice")
            if live_price and float(live_price) > 0:
                last_row["Close"] = float(live_price)
                last_close = float(live_price)

            # Normalized MACD features
            if "MACD_pct" not in last_row and "MACD" in last_row:
                last_row["MACD_pct"] = float(last_row["MACD"]) / last_close
            if "MACD_Signal_pct" not in last_row and "MACD_Signal" in last_row:
                last_row["MACD_Signal_pct"] = float(last_row["MACD_Signal"]) / last_close
            if "MACD_Histogram_pct" not in last_row and "MACD_Histogram" in last_row:
                last_row["MACD_Histogram_pct"] = float(last_row["MACD_Histogram"]) / last_close

            # Build formatted history records for Chart.js
            history_records = []
            for r in feat_df.to_dict("records"):
                d_str = str(r.get("Date", "")).split()[0]
                s20 = r.get("SMA_20")
                s50 = r.get("SMA_50")
                history_records.append({
                    "date": d_str,
                    "open": float(r.get("Open", 0.0) or 0.0),
                    "high": float(r.get("High", 0.0) or 0.0),
                    "low": float(r.get("Low", 0.0) or 0.0),
                    "close": float(r.get("Close", 0.0) or 0.0),
                    "volume": float(r.get("Volume", 0.0) or 0.0),
                    "sma_20": round(float(s20), 2) if pd.notna(s20) and s20 is not None else None,
                    "sma_50": round(float(s50), 2) if pd.notna(s50) and s50 is not None else None,
                    "rsi_14": round(float(r.get("RSI_14", 50.0) or 50.0), 2),
                    "daily_return": round(float(r.get("Daily_Return", 0.0) or 0.0), 4),
                    "return_5d": round(float(r.get("Return_5D", 0.0) or 0.0), 4),
                    "volatility_20d": round(float(r.get("Volatility_20D", 0.0) or 0.0), 4),
                    "volume_ratio": round(float(r.get("Volume_Ratio", 1.0) or 1.0), 2)
                })

            return {
                "df": feat_df,
                "history": history_records,
                "features": last_row,
                "meta": meta,
                "source": "Yahoo Finance (Online Live)",
                "fetched_at": time.time()
            }

        except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
            logger.debug(f"Yahoo v8 chart network/timeout error for {cand_sym}: {e}")
            self._circuit_open_until = time.time() + 15.0
            return None
        except Exception as e:
            logger.debug(f"Yahoo v8 chart failed for {cand_sym}: {e}")
            return None

    def _fetch_groww_quote(self, clean_symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch live price quote from Groww Public NSE/BSE endpoint."""
        if time.time() < self._circuit_open_until:
            return None
        url = f"https://groww.in/v1/api/stocks_data/v1/accord_points/exchange/NSE/segment/CASH/latest_prices_ohlc/{urllib.parse.quote(clean_symbol)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, socket.timeout, TimeoutError):
            self._circuit_open_until = time.time() + 15.0
            return None
        except Exception:
            return None

    def _fetch_via_yfinance(self, cand_sym: str) -> Optional[Dict[str, Any]]:
        """Fallback live downloader using yfinance."""
        try:
            ticker = yf.Ticker(cand_sym)
            hist = ticker.history(period="6mo", interval="1d").reset_index()
            if hist.empty or len(hist) < 5:
                return None

            if "Date" in hist.columns and hasattr(hist["Date"].dt, "tz") and hist["Date"].dt.tz is not None:
                hist["Date"] = hist["Date"].dt.tz_localize(None)

            if "Adj Close" not in hist.columns:
                hist["Adj Close"] = hist["Close"]
            hist["Symbol"] = cand_sym

            feat_df = calculate_features(hist, verbose=False, min_rows=5)
            if feat_df.empty:
                return None

            last_row = feat_df.iloc[-1].to_dict()
            last_row["Symbol"] = cand_sym
            last_close = float(last_row.get("Close", 1.0) or 1.0)
            if last_close <= 0:
                last_close = 1.0

            if "MACD_pct" not in last_row and "MACD" in last_row:
                last_row["MACD_pct"] = float(last_row["MACD"]) / last_close
            if "MACD_Signal_pct" not in last_row and "MACD_Signal" in last_row:
                last_row["MACD_Signal_pct"] = float(last_row["MACD_Signal"]) / last_close
            if "MACD_Histogram_pct" not in last_row and "MACD_Histogram" in last_row:
                last_row["MACD_Histogram_pct"] = float(last_row["MACD_Histogram"]) / last_close

            history_records = []
            for r in feat_df.to_dict("records"):
                d_str = str(r.get("Date", "")).split()[0]
                s20 = r.get("SMA_20")
                s50 = r.get("SMA_50")
                history_records.append({
                    "date": d_str,
                    "open": float(r.get("Open", 0.0) or 0.0),
                    "high": float(r.get("High", 0.0) or 0.0),
                    "low": float(r.get("Low", 0.0) or 0.0),
                    "close": float(r.get("Close", 0.0) or 0.0),
                    "volume": float(r.get("Volume", 0.0) or 0.0),
                    "sma_20": round(float(s20), 2) if pd.notna(s20) and s20 is not None else None,
                    "sma_50": round(float(s50), 2) if pd.notna(s50) and s50 is not None else None,
                    "rsi_14": round(float(r.get("RSI_14", 50.0) or 50.0), 2),
                    "daily_return": round(float(r.get("Daily_Return", 0.0) or 0.0), 4),
                    "return_5d": round(float(r.get("Return_5D", 0.0) or 0.0), 4),
                    "volatility_20d": round(float(r.get("Volatility_20D", 0.0) or 0.0), 4),
                    "volume_ratio": round(float(r.get("Volume_Ratio", 1.0) or 1.0), 2)
                })

            return {
                "df": feat_df,
                "history": history_records,
                "features": last_row,
                "meta": {},
                "source": "yfinance (Online Live)",
                "fetched_at": time.time()
            }
        except Exception:
            return None

    def _save_to_cache(self, clean_sym: str, cand: str, data: Dict[str, Any]):
        self._cache[clean_sym] = data
        self._cache[cand] = data
        if cand.endswith(".NS"):
            self._cache[cand[:-3]] = data


online_stock_fetcher = OnlineStockFetcher(cache_ttl_seconds=180)
