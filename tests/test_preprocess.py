import unittest
import numpy as np
import pandas as pd
from training.preprocess import calculate_features


class TestPreprocess(unittest.TestCase):
    def setUp(self):
        # Create 100 days of synthetic OHLCV data
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        np.random.seed(42)
        base_price = 100 + np.cumsum(np.random.randn(100) * 0.5)

        self.df = pd.DataFrame({
            "Date": dates,
            "Symbol": "TEST",
            "Open": base_price + np.random.uniform(-0.5, 0.5, 100),
            "High": base_price + np.random.uniform(0.5, 1.5, 100),
            "Low": base_price - np.random.uniform(0.5, 1.5, 100),
            "Close": base_price,
            "Adj Close": base_price,
            "Volume": np.random.randint(100000, 500000, 100),
            "Exchange": "NSE",
            "Market": "India"
        })

    def test_calculate_features_generates_required_columns(self):
        result = calculate_features(self.df)
        self.assertFalse(result.empty)

        required = [
            "Daily_Return", "Return_5D", "Return_10D", "Return_20D",
            "SMA_20", "SMA_50", "Price_vs_SMA20", "Price_vs_SMA50",
            "RSI_14", "MACD", "MACD_Signal", "MACD_Histogram",
            "BB_Width", "BB_Position", "ATR_Percent", "Volatility_20D",
            "Volume_Ratio", "Future_Return_5D", "Target_5D"
        ]
        for col in required:
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_insufficient_data_returns_empty(self):
        short_df = self.df.iloc[:20]
        result = calculate_features(short_df)
        self.assertTrue(result.empty)

    def test_target_values_are_valid(self):
        result = calculate_features(self.df)
        non_nan_targets = result["Target_5D"].dropna().unique()
        for t in non_nan_targets:
            self.assertIn(t, [-1.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
