import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import yfinance as yf

from training.preprocess import calculate_features

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "datasets"

INDIAN_FEATURES_FILE = DATA_DIR / "india" / "indian_features.csv"
US_FEATURES_FILE = DATA_DIR / "usa" / "us_features.csv"


class StockDataService:
    def __init__(self):
        self._df: Optional[pd.DataFrame] = None
        self._last_mtimes: Dict[Path, float] = {}
        self._live_cache: Dict[str, pd.DataFrame] = {}

    def get_all_features_df(self) -> pd.DataFrame:
        current_mtimes = {}
        for p in [INDIAN_FEATURES_FILE, US_FEATURES_FILE]:
            if p.exists():
                current_mtimes[p] = p.stat().st_mtime

        if self._df is not None and current_mtimes == self._last_mtimes:
            return self._df

        dfs = []
        if INDIAN_FEATURES_FILE.exists():
            dfs.append(pd.read_csv(INDIAN_FEATURES_FILE))
        if US_FEATURES_FILE.exists():
            dfs.append(pd.read_csv(US_FEATURES_FILE))
        if not dfs:
            return pd.DataFrame()
        combined = pd.concat(dfs, ignore_index=True)
        combined["Date"] = pd.to_datetime(combined["Date"])
        self._df = combined
        self._last_mtimes = current_mtimes
        return self._df

    def list_stocks(self) -> List[Dict[str, Any]]:
        df = self.get_all_features_df()
        if df.empty:
            return []

        stocks = []
        for symbol, group in df.groupby("Symbol"):
            sorted_group = group.sort_values("Date")
            last_row = sorted_group.iloc[-1]
            date_str = str(last_row.get("Date", "")).split()[0]
            stocks.append({
                "symbol": str(symbol),
                "exchange": str(last_row.get("Exchange", "")),
                "market": str(last_row.get("Market", "")),
                "latest_date": date_str,
                "latest_close": float(last_row.get("Close", 0.0)),
                "records_count": len(sorted_group)
            })
        return sorted(stocks, key=lambda x: x["symbol"])

    def search_stocks(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for any stock or IPO company across global exchanges (NSE, BSE, NASDAQ, NYSE).
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

        # 2. Query open real-time stock & IPO search endpoint (Yahoo Finance)
        # Search both global and Indian regional feeds so newly listed IPOs (e.g. Swiggy, Tata Tech) appear
        search_urls = [
            f"https://query2.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(q_clean)}&quotesCount=12&newsCount=0",
            f"https://query2.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(q_clean)}&quotesCount=12&newsCount=0&region=IN&lang=en-IN"
        ]

        for s_url in search_urls:
            try:
                req = urllib.request.Request(s_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
                with urllib.request.urlopen(req, timeout=4) as resp:
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
            if len(results) >= 15:
                break

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
        clean_sym = symbol.strip().upper()
        df = self.get_all_features_df()

        candidates = [clean_sym]
        if "." not in clean_sym:
            candidates.extend([f"{clean_sym}.NS", f"{clean_sym}.BO"])

        stock_df = pd.DataFrame()
        resolved_sym = clean_sym
        if not df.empty:
            for cand in candidates:
                match = df[df["Symbol"].astype(str).str.upper() == cand]
                if not match.empty:
                    stock_df = match.sort_values("Date")
                    resolved_sym = cand
                    break

        # Fallback to live data for IPO or new stocks
        if stock_df.empty:
            live_df = self._fetch_and_compute_live(clean_sym)
            if live_df is not None and not live_df.empty:
                stock_df = live_df
                resolved_sym = str(live_df.iloc[-1].get("Symbol", clean_sym)).upper()

        if stock_df.empty:
            return None

        last_row = stock_df.iloc[-1].to_dict()
        last_row["Symbol"] = resolved_sym

        # Safe close price for normalization
        try:
            close_price = float(last_row.get("Close", 1.0) or 1.0)
        except (ValueError, TypeError):
            close_price = 1.0
        if close_price <= 0:
            close_price = 1.0

        if "MACD_pct" not in last_row and "MACD" in last_row:
            last_row["MACD_pct"] = float(last_row["MACD"]) / close_price
        if "MACD_Signal_pct" not in last_row and "MACD_Signal" in last_row:
            last_row["MACD_Signal_pct"] = float(last_row["MACD_Signal"]) / close_price
        if "MACD_Histogram_pct" not in last_row and "MACD_Histogram" in last_row:
            last_row["MACD_Histogram_pct"] = float(last_row["MACD_Histogram"]) / close_price

        return last_row

    def get_stock_history(self, symbol: str, limit: int = 30) -> List[Dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 500))
        clean_sym = symbol.strip().upper()
        df = self.get_all_features_df()

        candidates = [clean_sym]
        if "." not in clean_sym:
            candidates.extend([f"{clean_sym}.NS", f"{clean_sym}.BO"])

        stock_df = pd.DataFrame()
        if not df.empty:
            for cand in candidates:
                match = df[df["Symbol"].astype(str).str.upper() == cand]
                if not match.empty:
                    stock_df = match.sort_values("Date")
                    break

        # Fallback to live data for IPO or new stocks
        if stock_df.empty:
            live_df = self._fetch_and_compute_live(clean_sym)
            if live_df is not None and not live_df.empty:
                stock_df = live_df

        if stock_df.empty:
            return []

        tail = stock_df.tail(safe_limit)
        records = []
        for _, row in tail.iterrows():
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
