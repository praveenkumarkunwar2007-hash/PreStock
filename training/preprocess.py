
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


def calculate_features(df, verbose: bool = False):

    df = df.copy()

    # Ensure dates and numeric columns are valid
    df["Date"] = pd.to_datetime(df["Date"])

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

    if len(df) < 50:
        return pd.DataFrame()

    # Use closing price for technical indicators
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # ==================================
    # 1. PRICE AND RETURN FEATURES
    # ==================================

    df["Daily_Return"] = close.pct_change()

    df["Return_5D"] = close.pct_change(5)

    df["Return_10D"] = close.pct_change(10)

    df["Return_20D"] = close.pct_change(20)

    # ==================================
    # 2. MOVING AVERAGES
    # ==================================

    df["SMA_10"] = close.rolling(10).mean()

    df["SMA_20"] = close.rolling(20).mean()

    df["SMA_50"] = close.rolling(50).mean()

    df["EMA_10"] = close.ewm(
        span=10, adjust=False
    ).mean()

    df["EMA_20"] = close.ewm(
        span=20, adjust=False
    ).mean()

    df["EMA_50"] = close.ewm(
        span=50, adjust=False
    ).mean()

    df["Price_vs_SMA20"] = (
        close / df["SMA_20"] - 1
    )

    df["Price_vs_SMA50"] = (
        close / df["SMA_50"] - 1
    )

    # ==================================
    # 3. RSI
    # ==================================

    df["RSI_14"] = RSIIndicator(
        close=close,
        window=14
    ).rsi()

    # ==================================
    # 4. MACD
    # ==================================

    macd = MACD(
        close=close,
        window_fast=12,
        window_slow=26,
        window_sign=9
    )

    df["MACD"] = macd.macd()

    df["MACD_Signal"] = macd.macd_signal()

    df["MACD_Histogram"] = (
        macd.macd_diff()
    )

    # ==================================
    # 5. BOLLINGER BANDS
    # ==================================

    bollinger = BollingerBands(
        close=close,
        window=20,
        window_dev=2
    )

    df["BB_High"] = bollinger.bollinger_hband()

    df["BB_Low"] = bollinger.bollinger_lband()

    df["BB_Middle"] = bollinger.bollinger_mavg()

    df["BB_Width"] = (
        (df["BB_High"] - df["BB_Low"])
        / df["BB_Middle"]
    )

    df["BB_Position"] = (
        (close - df["BB_Low"])
        / (df["BB_High"] - df["BB_Low"]).replace(0, np.nan)
    )

    # ==================================
    # 6. ATR VOLATILITY
    # ==================================

    atr = AverageTrueRange(
        high=high,
        low=low,
        close=close,
        window=14
    )

    df["ATR_14"] = atr.average_true_range()

    df["ATR_Percent"] = (
        df["ATR_14"] / close
    )

    df["Volatility_20D"] = (
        df["Daily_Return"].rolling(20).std()
        * np.sqrt(252)
    )

    # ==================================
    # 7. VOLUME INDICATORS
    # ==================================

    df["Volume_SMA_20"] = (
        volume.rolling(20).mean()
    )

    df["Volume_Ratio"] = (
        volume / df["Volume_SMA_20"].replace(0, np.nan)
    )

    obv = OnBalanceVolumeIndicator(
        close=close,
        volume=volume
    )

    df["OBV"] = obv.on_balance_volume()

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


    # Remove rows without sufficient historical features.
    # Keep final rows with unavailable future labels for
    # later prediction/inference.

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

