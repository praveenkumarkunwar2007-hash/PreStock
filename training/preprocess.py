
import pandas as pd
import numpy as np
from pathlib import Path

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import (
    BollingerBands,
    AverageTrueRange
)
from ta.volume import OnBalanceVolumeIndicator


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets"

# Prediction label settings
FUTURE_DAYS = 5
PROFIT_THRESHOLD = 0.01
LOSS_THRESHOLD = -0.01


def calculate_features(df, verbose: bool = False, min_rows: int = 5):

    df = df.copy()

    # Ensure dates and numeric columns are valid
    df["Date"] = pd.to_datetime(df["Date"])
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    if "Symbol" not in df.columns:
        df["Symbol"] = "CUSTOM"

    numeric_columns = [
        "Open", "High", "Low",
        "Close", "Adj Close", "Volume"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df.sort_values("Date")
    df = df.drop_duplicates(
        subset=["Symbol", "Date"],
        keep="last"
    )

    df = df.dropna(
        subset=["Open", "High", "Low", "Close", "Volume"]
    )

    # Require at least min_rows (default 5 days) so young IPOs can be evaluated
    if len(df) < min_rows:
        return pd.DataFrame()

    # Use closing price for technical indicators
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    n_samples = len(close)

    # ==================================
    # 1. PRICE AND RETURN FEATURES
    # ==================================

    df["Daily_Return"] = close.pct_change().bfill().fillna(0.0)

    df["Return_5D"] = close.pct_change(5).bfill().fillna(0.0)

    df["Return_10D"] = close.pct_change(10).bfill().fillna(0.0)

    df["Return_20D"] = close.pct_change(20).bfill().fillna(0.0)

    # ==================================
    # 2. MOVING AVERAGES (Adaptive with min_periods=1)
    # ==================================

    df["SMA_10"] = close.rolling(10, min_periods=1).mean()

    df["SMA_20"] = close.rolling(20, min_periods=1).mean()

    df["SMA_50"] = close.rolling(50, min_periods=1).mean()

    df["EMA_10"] = close.ewm(
        span=10, min_periods=1, adjust=False
    ).mean()

    df["EMA_20"] = close.ewm(
        span=20, min_periods=1, adjust=False
    ).mean()

    df["EMA_50"] = close.ewm(
        span=50, min_periods=1, adjust=False
    ).mean()

    df["Price_vs_SMA20"] = (
        close / df["SMA_20"].replace(0, np.nan) - 1
    ).fillna(0.0)

    df["Price_vs_SMA50"] = (
        close / df["SMA_50"].replace(0, np.nan) - 1
    ).fillna(0.0)

    # ==================================
    # 3. RSI (Adaptive window for young IPOs)
    # ==================================

    rsi_window = min(14, max(2, n_samples - 1))
    df["RSI_14"] = RSIIndicator(
        close=close,
        window=rsi_window
    ).rsi().bfill().ffill().fillna(50.0)

    # ==================================
    # 4. MACD (Adaptive parameters)
    # ==================================

    fast_w = min(12, max(2, n_samples // 3))
    slow_w = min(26, max(fast_w + 1, min(n_samples - 1, 26)))
    sign_w = min(9, max(2, n_samples // 4))

    macd = MACD(
        close=close,
        window_fast=fast_w,
        window_slow=slow_w,
        window_sign=sign_w
    )

    df["MACD"] = macd.macd().bfill().ffill().fillna(0.0)

    df["MACD_Signal"] = macd.macd_signal().bfill().ffill().fillna(0.0)

    df["MACD_Histogram"] = (
        macd.macd_diff().bfill().ffill().fillna(0.0)
    )

    # ==================================
    # 5. BOLLINGER BANDS (Adaptive window)
    # ==================================

    bb_window = min(20, max(2, n_samples - 1))
    bollinger = BollingerBands(
        close=close,
        window=bb_window,
        window_dev=2
    )

    df["BB_High"] = bollinger.bollinger_hband().bfill().ffill().fillna(close * 1.02)

    df["BB_Low"] = bollinger.bollinger_lband().bfill().ffill().fillna(close * 0.98)

    df["BB_Middle"] = bollinger.bollinger_mavg().bfill().ffill().fillna(close)

    df["BB_Width"] = (
        (df["BB_High"] - df["BB_Low"])
        / df["BB_Middle"].replace(0, np.nan)
    ).bfill().ffill().fillna(0.04)

    df["BB_Position"] = (
        (close - df["BB_Low"])
        / (df["BB_High"] - df["BB_Low"]).replace(0, np.nan)
    ).bfill().ffill().fillna(0.5)

    # ==================================
    # 6. ATR VOLATILITY (Adaptive window)
    # ==================================

    atr_window = min(14, max(2, n_samples - 1))
    atr = AverageTrueRange(
        high=high,
        low=low,
        close=close,
        window=atr_window
    )

    df["ATR_14"] = atr.average_true_range().bfill().ffill().fillna(close * 0.015)

    df["ATR_Percent"] = (
        df["ATR_14"] / close.replace(0, np.nan)
    ).bfill().ffill().fillna(0.015)

    vol_window = min(20, max(2, n_samples - 1))
    df["Volatility_20D"] = (
        df["Daily_Return"].rolling(vol_window, min_periods=2).std()
        * np.sqrt(252)
    ).bfill().ffill().fillna(0.20)

    # ==================================
    # 7. VOLUME INDICATORS
    # ==================================

    df["Volume_SMA_20"] = (
        volume.rolling(20, min_periods=1).mean()
    )

    df["Volume_Ratio"] = (
        volume / df["Volume_SMA_20"].replace(0, np.nan)
    ).bfill().ffill().fillna(1.0)

    obv = OnBalanceVolumeIndicator(
        close=close,
        volume=volume
    )

    df["OBV"] = obv.on_balance_volume().bfill().ffill().fillna(0.0)

    # ==================================
    # 8. FUTURE PREDICTION TARGETS
    # ==================================

    # These columns are labels for training.
    # NEVER use them as model input features.

    df["Future_Return_5D"] = (
        close.shift(-FUTURE_DAYS) / close - 1
    )

    conditions = [
        df["Future_Return_5D"] > PROFIT_THRESHOLD,
        df["Future_Return_5D"] < LOSS_THRESHOLD,
    ]
    choices = [1.0, -1.0]
    df["Target_5D"] = np.select(conditions, choices, default=0.0)
    df.loc[df["Future_Return_5D"].isna(), "Target_5D"] = np.nan

    valid = df["Target_5D"].notna()

    if verbose:
        print("\n5-Day Target Distribution:")
        print(
            df.loc[valid, "Target_5D"]
            .value_counts()
            .sort_index()
        )
        print("\nTarget Distribution Percentage:")
        print(
            (
                df.loc[valid, "Target_5D"]
                .value_counts(normalize=True)
                .sort_index() * 100
            ).round(2)
        )

    # Keep rows with complete indicators
    df = df.replace(
        [np.inf, -np.inf], np.nan
    )

    feature_columns = [
        "SMA_50",
        "RSI_14",
        "MACD",
        "BB_Width",
        "ATR_Percent",
        "Volatility_20D"
    ]

    df = df.dropna(
        subset=feature_columns
    )

    return df


def process_market(input_file, output_file):

    input_path = DATA_DIR / input_file
    output_path = DATA_DIR / output_file

    print("\nProcessing:", input_file)

    if not input_path.exists():
        print("Input file not found:", input_path)
        return

    raw_data = pd.read_csv(input_path)

    processed_stocks = []

    for symbol, stock_data in raw_data.groupby("Symbol"):

        print("Processing stock:", symbol)

        try:
            result = calculate_features(stock_data)

            if not result.empty:
                processed_stocks.append(result)

                print(
                    "Features generated:",
                    symbol,
                    "Rows:",
                    len(result)
                )

        except Exception as error:
            print("Error:", symbol, error)

    if not processed_stocks:
        print("No processed data available.")
        return

    final_data = pd.concat(
        processed_stocks,
        ignore_index=True
    )

    final_data = final_data.sort_values(
        ["Symbol", "Date"]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    final_data.to_csv(
        output_path,
        index=False
    )

    print("\nSaved:", output_path)

    print("Total rows:", len(final_data))

    print(
        "Total stocks:",
        final_data["Symbol"].nunique()
    )

    print("\nAvailable columns:")

    print(final_data.columns.tolist())


if __name__ == "__main__":

    print("=" * 55)
    print("AI STOCK PREDICTOR - FEATURE ENGINEERING")
    print("=" * 55)

    process_market(
        "india/indian_stocks.csv",
        "india/indian_features.csv"
    )

    process_market(
        "usa/us_stocks.csv",
        "usa/us_features.csv"
    )

    print("\nFEATURE ENGINEERING COMPLETED")

