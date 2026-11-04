from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "datasets"

INDIAN_FEATURES_FILE = DATA_DIR / "india" / "indian_features.csv"
US_FEATURES_FILE = DATA_DIR / "usa" / "us_features.csv"


class StockDataService:
    def __init__(self):
        self._df: Optional[pd.DataFrame] = None
        self._last_mtimes: Dict[Path, float] = {}

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

    def get_latest_features(self, symbol: str) -> Optional[Dict[str, Any]]:
        df = self.get_all_features_df()
        if df.empty:
            return None

        clean_sym = symbol.strip().upper()
        stock_df = df[df["Symbol"].astype(str).str.upper() == clean_sym].sort_values("Date")
        if stock_df.empty:
            return None

        last_row = stock_df.iloc[-1].to_dict()

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
        df = self.get_all_features_df()
        if df.empty:
            return []

        safe_limit = max(1, min(int(limit), 500))
        clean_sym = symbol.strip().upper()
        stock_df = df[df["Symbol"].astype(str).str.upper() == clean_sym].sort_values("Date")
        if stock_df.empty:
            return []

        tail = stock_df.tail(safe_limit)
        records = []
        for _, row in tail.iterrows():
            date_val = str(row.get("Date", "")).split()[0]
            records.append({
                "date": date_val,
                "open": float(row.get("Open", 0.0) or 0.0),
                "high": float(row.get("High", 0.0) or 0.0),
                "low": float(row.get("Low", 0.0) or 0.0),
                "close": float(row.get("Close", 0.0) or 0.0),
                "volume": float(row.get("Volume", 0.0) or 0.0),
                "rsi_14": float(row.get("RSI_14", 0.0) or 0.0),
                "daily_return": float(row.get("Daily_Return", 0.0) or 0.0),
                "volatility_20d": float(row.get("Volatility_20D", 0.0) or 0.0)
            })
        return records


stock_data_service = StockDataService()
