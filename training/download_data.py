
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets"

# Historical data configuration
PERIOD = "5y"
INTERVAL = "1d"

# Indian stocks: NSE and BSE
INDIA_STOCKS = {
    "NSE": [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
    ],
    "BSE": [
        "RELIANCE.BO",
        "TCS.BO",
        "INFY.BO",
    ],
}

# US stocks: NASDAQ and NYSE
USA_STOCKS = {
    "NASDAQ": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
    ],
    "NYSE": [
        "JPM",
        "KO",
        "WMT",
        "DIS",
        "IBM",
    ],
}


def download_stock(symbol, exchange, market):

    print(f"Downloading {symbol}...")

    try:
        stock = yf.Ticker(symbol)

        history = stock.history(
            period=PERIOD,
            interval=INTERVAL,
            auto_adjust=False,
            actions=False
        )

        if history.empty:
            print(f"No data available: {symbol}")
            return None

        # Handle possible multi-level columns
        if isinstance(history.columns, pd.MultiIndex):
            history.columns = history.columns.get_level_values(0)

        # Remove timezone from index
        if history.index.tz is not None:
            history.index = history.index.tz_localize(None)

        df = history.reset_index()

        # Standardize date column
        date_column = next(
            (col for col in df.columns
             if str(col).lower() in ["date", "datetime"]),
            df.columns[0]
        )

        df = df.rename(columns={date_column: "Date"})

        # Required columns
        required_columns = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        for column in required_columns:
            if column not in df.columns:
                print(f"Missing {column} for {symbol}")
                return None

        # Keep available adjusted closing price
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]

        # Add stock information
        df["Symbol"] = symbol
        df["Exchange"] = exchange
        df["Market"] = market

        # Arrange columns
        df = df[
            [
                "Date",
                "Symbol",
                "Exchange",
                "Market",
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume"
            ]
        ]

        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df = df.sort_values("Date")

        print(f"Downloaded {len(df)} rows: {symbol}")

        return df

    except Exception as error:
        print(f"Error downloading {symbol}: {error}")
        return None


def download_market(stocks, market, output_file):

    all_data = []

    for exchange, symbols in stocks.items():

        for symbol in symbols:

            data = download_stock(
                symbol,
                exchange,
                market
            )

            if data is not None:
                all_data.append(data)

    if not all_data:
        print(f"No data downloaded for {market}")
        return

    final_data = pd.concat(
        all_data,
        ignore_index=True
    )

    final_data = final_data.sort_values(
        ["Symbol", "Date"]
    )

    output_path = DATA_DIR / output_file

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    final_data.to_csv(
        output_path,
        index=False
    )

    print("\nSaved:", output_path)
    print("Total records:", len(final_data))


if __name__ == "__main__":

    print("=" * 50)
    print("AI STOCK PREDICTOR - DATA COLLECTION")
    print("Started:", datetime.now())
    print("=" * 50)

    download_market(
        INDIA_STOCKS,
        "India",
        "india/indian_stocks.csv"
    )

    print("\n" + "=" * 50)

    download_market(
        USA_STOCKS,
        "USA",
        "usa/us_stocks.csv"
    )

    print("\nDATA COLLECTION COMPLETED")
