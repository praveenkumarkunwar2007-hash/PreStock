import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import yfinance as yf

from training.preprocess import calculate_features
from backend.services.online_fetcher import online_stock_fetcher

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "datasets"

INDIAN_FEATURES_FILE = DATA_DIR / "india" / "indian_features.csv"
US_FEATURES_FILE = DATA_DIR / "usa" / "us_features.csv"
IPO_FEATURES_FILE = DATA_DIR / "ipos" / "ipo_features.csv"


class StockDataService:
    def __init__(self):
        self._df: Optional[pd.DataFrame] = None
        self._last_mtimes: Dict[Path, float] = {}
        self._live_cache: Dict[str, pd.DataFrame] = {}
        self._symbol_map: Dict[str, pd.DataFrame] = {}
        self._cached_stock_list: List[Dict[str, Any]] = []
        self._latest_features_cache: Dict[str, Dict[str, Any]] = {}
        self._history_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _ensure_indexed(self):
        current_mtimes = {}
        for p in [INDIAN_FEATURES_FILE, US_FEATURES_FILE, IPO_FEATURES_FILE]:
            if p.exists():
                current_mtimes[p] = p.stat().st_mtime

        if self._df is not None and current_mtimes == self._last_mtimes and self._cached_stock_list:
            return

        dfs = []
        if INDIAN_FEATURES_FILE.exists():
            dfs.append(pd.read_csv(INDIAN_FEATURES_FILE))
        if US_FEATURES_FILE.exists():
            dfs.append(pd.read_csv(US_FEATURES_FILE))
        if IPO_FEATURES_FILE.exists():
            dfs.append(pd.read_csv(IPO_FEATURES_FILE))

        if not dfs:
            self._df = pd.DataFrame()
            self._symbol_map = {}
            self._cached_stock_list = []
            self._latest_features_cache = {}
            self._history_cache = {}
            return

        combined = pd.concat(dfs, ignore_index=True)
        combined["Date"] = pd.to_datetime(combined["Date"])
        self._df = combined
        self._last_mtimes = current_mtimes

        # Build in-memory fast index for sub-millisecond retrieval
        symbol_map: Dict[str, pd.DataFrame] = {}
        stock_list: List[Dict[str, Any]] = []
        latest_cache: Dict[str, Dict[str, Any]] = {}
        history_cache: Dict[str, List[Dict[str, Any]]] = {}

        for sym, group in combined.groupby("Symbol"):
            sym_str = str(sym).upper().strip()
            sorted_group = group.sort_values("Date")
            symbol_map[sym_str] = sorted_group

            last_row = sorted_group.iloc[-1].to_dict()
            last_row["Symbol"] = sym_str
            date_str = str(last_row.get("Date", "")).split()[0]
            try:
                close_price = float(last_row.get("Close", 1.0) or 1.0)
            except (ValueError, TypeError):
                close_price = 1.0
            if close_price <= 0:
                close_price = 1.0

            # Pre-calculate normalized MACD features
            if "MACD_pct" not in last_row and "MACD" in last_row:
                last_row["MACD_pct"] = float(last_row["MACD"]) / close_price
            if "MACD_Signal_pct" not in last_row and "MACD_Signal" in last_row:
                last_row["MACD_Signal_pct"] = float(last_row["MACD_Signal"]) / close_price
            if "MACD_Histogram_pct" not in last_row and "MACD_Histogram" in last_row:
                last_row["MACD_Histogram_pct"] = float(last_row["MACD_Histogram"]) / close_price

            latest_cache[sym_str] = last_row

            # Pre-compute formatted history list for instant slicing
            sym_records = []
            for r in sorted_group.to_dict("records"):
                d_val = str(r.get("Date", "")).split()[0]
                s20 = r.get("SMA_20")
                s50 = r.get("SMA_50")
                sym_records.append({
                    "date": d_val,
                    "open": float(r.get("Open", 0.0) or 0.0),
                    "high": float(r.get("High", 0.0) or 0.0),
                    "low": float(r.get("Low", 0.0) or 0.0),
                    "close": float(r.get("Close", 0.0) or 0.0),
                    "volume": float(r.get("Volume", 0.0) or 0.0),
                    "sma_20": round(float(s20), 2) if pd.notna(s20) and s20 is not None else None,
                    "sma_50": round(float(s50), 2) if pd.notna(s50) and s50 is not None else None,
                    "rsi_14": round(float(r.get("RSI_14", 0.0) or 0.0), 2),
                    "daily_return": round(float(r.get("Daily_Return", 0.0) or 0.0), 4),
                    "return_5d": round(float(r.get("Return_5D", 0.0) or 0.0), 4),
                    "volatility_20d": round(float(r.get("Volatility_20D", 0.0) or 0.0), 4),
                    "volume_ratio": round(float(r.get("Volume_Ratio", 1.0) or 1.0), 2)
                })
            history_cache[sym_str] = sym_records

            # If symbol ends in .NS, alias base symbol for instant resolution
            if sym_str.endswith(".NS"):
                base_sym = sym_str[:-3]
                if base_sym not in latest_cache:
                    latest_cache[base_sym] = last_row
                if base_sym not in symbol_map:
                    symbol_map[base_sym] = sorted_group
                if base_sym not in history_cache:
                    history_cache[base_sym] = sym_records

            stock_list.append({
                "symbol": sym_str,
                "exchange": str(last_row.get("Exchange", "")),
                "market": str(last_row.get("Market", "")),
                "latest_date": date_str,
                "latest_close": float(last_row.get("Close", 0.0)),
                "records_count": len(sorted_group)
            })

        self._symbol_map = symbol_map
        self._cached_stock_list = sorted(stock_list, key=lambda x: x["symbol"])
        self._latest_features_cache = latest_cache
        self._history_cache = history_cache

    def get_all_features_df(self) -> pd.DataFrame:
        self._ensure_indexed()
        return self._df

    def list_stocks(self) -> List[Dict[str, Any]]:
        self._ensure_indexed()
        stocks = []
        for s in self._cached_stock_list:
            item = dict(s)
            sym = item.get("symbol", "")
            # If fresh live quote is cached, reflect live close price and date
            cached_live = online_stock_fetcher._get_from_cache(sym)
            if cached_live and "features" in cached_live:
                feat = cached_live["features"]
                if "Close" in feat:
                    item["latest_close"] = float(feat["Close"])
                if "Date" in feat:
                    item["latest_date"] = str(feat["Date"]).split()[0]
                item["source"] = cached_live.get("source", "Live Feeds")
            stocks.append(item)
        return stocks

    def search_stocks(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for any stock or IPO company across global exchanges (NSE, BSE, NASDAQ, NYSE).
        Integrates Groww Public Search (for Indian IPOs & equities) and Yahoo Finance (Global).
        """
        q_clean = query.strip()
        if not q_clean:
            return []

        results: List[Dict[str, Any]] = []
        seen = set()

        # 1. Search local tracked stocks first
        local_df = self.get_all_features_df()
        if not local_df.empty:
            matches = local_df[local_df["Symbol"].astype(str).str.upper().str.contains(q_clean.upper())]
            for sym, group in matches.groupby("Symbol"):
                sym_str = str(sym).upper()
                last_r = group.iloc[-1]
                results.append({
                    "symbol": sym_str,
                    "name": f"{sym_str} (Tracked Stock)",
                    "exchange": str(last_r.get("Exchange", "Market")),
                    "market": str(last_r.get("Market", "Tracked")),
                    "type": "Tracked",
                    "is_local": True
                })
                seen.add(sym_str)

        # 2. Query Groww Public Search for Indian equities & recent IPOs (e.g. Swiggy, Hyundai, Waaree, Tata Tech)
        try:
            groww_url = f"https://groww.in/v1/api/search/v1/entity?app=false&entity_type=stocks&page=0&q={urllib.parse.quote(q_clean)}&size=8"
            req = urllib.request.Request(groww_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                g_data = json.loads(resp.read().decode())
                for item in g_data.get("content", []):
                    nse_code = item.get("nse_scrip_code")
                    bse_code = item.get("bse_scrip_code")
                    title = item.get("title") or ""
                    sym = f"{nse_code}.NS" if nse_code else (f"{bse_code}.BO" if bse_code else "")
                    if sym and sym not in seen:
                        results.append({
                            "symbol": sym,
                            "name": title,
                            "exchange": "NSE" if nse_code else "BSE",
                            "market": "India",
                            "type": "IPO / Live Indian Equity",
                            "is_local": False
                        })
                        seen.add(sym)
                        if len(results) >= 15:
                            break
        except Exception as e:
            logger.debug(f"Groww search failed for {q_clean}: {e}")

        # 3. Query open real-time stock & IPO search endpoint (Yahoo Finance Global)
        search_urls = [
            f"https://query2.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(q_clean)}&quotesCount=10&newsCount=0",
            f"https://query2.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(q_clean)}&quotesCount=10&newsCount=0&region=IN&lang=en-IN"
        ]

        for s_url in search_urls:
            if len(results) >= 15:
                break
            try:
                req = urllib.request.Request(s_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
                with urllib.request.urlopen(req, timeout=3) as resp:
                    data = json.loads(resp.read())

                for item in data.get("quotes", []):
                    sym = item.get("symbol", "").upper().strip()
                    if not sym or sym in seen:
                        continue
                    quote_type = item.get("quoteType", "EQUITY")
                    if quote_type not in ["EQUITY", "ETF", "MUTUALFUND"]:
                        continue

                    display_name = item.get("shortname") or item.get("longname") or sym
                    exchange = item.get("exchange", "Exchange")
                    market = "India" if sym.endswith(".NS") or sym.endswith(".BO") or exchange in ["NSI", "BSE"] else "US/Global"

                    results.append({
                        "symbol": sym,
                        "name": display_name,
                        "exchange": exchange,
                        "market": market,
                        "type": "IPO / Live Equity",
                        "is_local": False
                    })
                    seen.add(sym)
                    if len(results) >= 15:
                        break
            except Exception:
                pass

        return results

    def _fetch_and_compute_live(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        On-the-fly live historical download and technical indicator calculation
        for any newly searched or IPO stock. Supports young IPOs (>= 5 trading days)
        and automatic regional suffix fallback (.NS / .BO for Indian equities).
        """
        clean_sym = symbol.strip().upper()
        if clean_sym in self._live_cache:
            return self._live_cache[clean_sym]

        # Automatic suffix fallback: if no exchange suffix, test raw, then Indian NSE/BSE
        candidates = [clean_sym]
        if "." not in clean_sym:
            candidates.extend([f"{clean_sym}.NS", f"{clean_sym}.BO"])

        for cand in candidates:
            try:
                ticker = yf.Ticker(cand)
                hist = ticker.history(period="6mo", interval="1d").reset_index()
                if hist.empty or len(hist) < 5:
                    continue

                if "Date" in hist.columns and hasattr(hist["Date"].dt, "tz") and hist["Date"].dt.tz is not None:
                    hist["Date"] = hist["Date"].dt.tz_localize(None)

                if "Adj Close" not in hist.columns:
                    hist["Adj Close"] = hist["Close"]
                hist["Symbol"] = cand

                # Calculate indicators with adaptive IPO handling
                feat_df = calculate_features(hist, verbose=False, min_rows=5)
                if feat_df.empty:
                    continue

                # Cache under both query symbol and resolved ticker
                self._live_cache[clean_sym] = feat_df
                self._live_cache[cand] = feat_df
                return feat_df
            except Exception:
                continue

        return None

    def get_latest_features(self, symbol: str) -> Optional[Dict[str, Any]]:
        self._ensure_indexed()
        clean_sym = symbol.strip().upper()

        # 1. Primary path: Fetch real-time online data (Yahoo v8 Chart / Groww)
        try:
            live_data = online_stock_fetcher.fetch_live_data(clean_sym)
            if live_data and "features" in live_data and live_data["features"]:
                feat = dict(live_data["features"])
                feat["data_source"] = live_data.get("source", "Online Market Feeds")
                # Sync into in-memory cache
                self._latest_features_cache[clean_sym] = feat
                if "history" in live_data:
                    self._history_cache[clean_sym] = live_data["history"]
                return feat
        except Exception as e:
            logger.debug(f"Live online fetch failed for {clean_sym}: {e}")

        # 2. Resilient fallback path: In-memory cached features or local dataset
        candidates = [clean_sym]
        if "." not in clean_sym:
            candidates.extend([f"{clean_sym}.NS", f"{clean_sym}.BO"])

        for cand in candidates:
            if cand in self._latest_features_cache:
                f = dict(self._latest_features_cache[cand])
                f.setdefault("data_source", "Cached Market Data")
                return f

        # Check live cache
        for cand in candidates:
            if cand in self._live_cache:
                live_df = self._live_cache[cand]
                if not live_df.empty:
                    last_row = live_df.iloc[-1].to_dict()
                    last_row["Symbol"] = cand
                    last_row["data_source"] = "Local Cache"
                    return last_row

        # Fallback to yfinance downloader for IPO or new stocks
        live_df = self._fetch_and_compute_live(clean_sym)
        if live_df is not None and not live_df.empty:
            resolved_sym = str(live_df.iloc[-1].get("Symbol", clean_sym)).upper()
            last_row = live_df.iloc[-1].to_dict()
            last_row["Symbol"] = resolved_sym
            last_row["data_source"] = "Live Market Fallback"
            return last_row

        return None

    def get_stock_history(self, symbol: str, limit: int = 30) -> List[Dict[str, Any]]:
        self._ensure_indexed()
        safe_limit = max(1, min(int(limit), 500))
        clean_sym = symbol.strip().upper()

        # 1. Primary path: Fetch real-time online history (Yahoo v8 Chart / Groww)
        try:
            live_data = online_stock_fetcher.fetch_live_data(clean_sym)
            if live_data and "history" in live_data and live_data["history"]:
                self._history_cache[clean_sym] = live_data["history"]
                return live_data["history"][-safe_limit:]
        except Exception as e:
            logger.debug(f"Live online history failed for {clean_sym}: {e}")

        # 2. Resilient fallback path: In-memory pre-computed history cache
        candidates = [clean_sym]
        if "." not in clean_sym:
            candidates.extend([f"{clean_sym}.NS", f"{clean_sym}.BO"])

        # Instant O(1) retrieval from pre-computed history cache
        for cand in candidates:
            if cand in self._history_cache:
                return self._history_cache[cand][-safe_limit:]

        stock_df = None
        for cand in candidates:
            if cand in self._symbol_map:
                stock_df = self._symbol_map[cand]
                break

        # Fallback to live cache or live fetch
        if stock_df is None or stock_df.empty:
            for cand in candidates:
                if cand in self._live_cache:
                    stock_df = self._live_cache[cand]
                    break

        if stock_df is None or stock_df.empty:
            live_df = self._fetch_and_compute_live(clean_sym)
            if live_df is not None and not live_df.empty:
                stock_df = live_df

        if stock_df is None or stock_df.empty:
            return []

        tail = stock_df.tail(safe_limit)
        # Vectorized list comprehension
        records = []
        for row in tail.to_dict("records"):
            date_val = str(row.get("Date", "")).split()[0]
            sma_20_val = row.get("SMA_20")
            sma_50_val = row.get("SMA_50")
            records.append({
                "date": date_val,
                "open": float(row.get("Open", 0.0) or 0.0),
                "high": float(row.get("High", 0.0) or 0.0),
                "low": float(row.get("Low", 0.0) or 0.0),
                "close": float(row.get("Close", 0.0) or 0.0),
                "volume": float(row.get("Volume", 0.0) or 0.0),
                "sma_20": round(float(sma_20_val), 2) if pd.notna(sma_20_val) and sma_20_val is not None else None,
                "sma_50": round(float(sma_50_val), 2) if pd.notna(sma_50_val) and sma_50_val is not None else None,
                "rsi_14": round(float(row.get("RSI_14", 0.0) or 0.0), 2),
                "daily_return": round(float(row.get("Daily_Return", 0.0) or 0.0), 4),
                "return_5d": round(float(row.get("Return_5D", 0.0) or 0.0), 4),
                "volatility_20d": round(float(row.get("Volatility_20D", 0.0) or 0.0), 4),
                "volume_ratio": round(float(row.get("Volume_Ratio", 1.0) or 1.0), 2)
            })
        return records


stock_data_service = StockDataService()
