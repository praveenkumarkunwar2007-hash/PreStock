import unittest
from backend.services.stock_data import stock_data_service
from backend.routes.predict import predict_by_symbol, predict_custom, CustomFeatureInput
from backend.routes.stocks import get_all_stocks, get_stock_history


class TestBackend(unittest.TestCase):
    def test_list_stocks(self):
        stocks = get_all_stocks()
        self.assertIsInstance(stocks, list)
        self.assertGreater(len(stocks), 0)
        symbols = [s["symbol"] for s in stocks]
        self.assertIn("AAPL", symbols)
        self.assertIn("RELIANCE.NS", symbols)

    def test_predict_tracked_symbol(self):
        res = predict_by_symbol("AAPL")
        self.assertEqual(res["symbol"], "AAPL")
        self.assertIn("prediction", res)
        self.assertIn(res["prediction"], ["Loss", "Neutral", "Profit"])

    def test_predict_custom_payload(self):
        payload = CustomFeatureInput(
            Daily_Return=0.01,
            Return_5D=0.03,
            Return_10D=0.05,
            Return_20D=0.08,
            Price_vs_SMA20=0.02,
            Price_vs_SMA50=0.04,
            RSI_14=62.5,
            BB_Width=0.05,
            BB_Position=0.75,
            ATR_Percent=0.015,
            Volatility_20D=0.22,
            Volume_Ratio=1.2,
            MACD_pct=0.005,
            MACD_Signal_pct=0.004,
            MACD_Histogram_pct=0.001
        )
        res = predict_custom(payload)
        self.assertIn("prediction", res)
        self.assertIn(res["prediction"], ["Loss", "Neutral", "Profit"])

    def test_stock_history(self):
        res = get_stock_history("AAPL", limit=10)
        self.assertEqual(res["symbol"], "AAPL")
        self.assertEqual(len(res["history"]), 10)


if __name__ == "__main__":
    unittest.main()
