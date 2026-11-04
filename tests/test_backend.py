import unittest
from backend.services.stock_data import stock_data_service
from backend.routes.predict import predict_by_symbol, predict_custom, CustomFeatureInput
from backend.routes.stocks import get_all_stocks, get_stock_history, get_stock_news, search_stocks


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
        self.assertIn("rationale", res)
        self.assertIn("summary", res["rationale"])
        self.assertIn("bullish_signals", res["rationale"])
        self.assertIn("bearish_signals", res["rationale"])
        self.assertIn("return_5d", res)
        self.assertIn("daily_return", res)
        self.assertIn("rsi_14", res)

    def test_predict_symbol_suffix_resolution(self):
        # Should resolve RELIANCE to RELIANCE.NS automatically
        res = predict_by_symbol("RELIANCE")
        self.assertEqual(res["symbol"], "RELIANCE.NS")
        self.assertIn("prediction", res)

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
        self.assertIn("rationale", res)

    def test_stock_history(self):
        res = get_stock_history("AAPL", limit=10)
        self.assertEqual(res["symbol"], "AAPL")
        self.assertEqual(len(res["history"]), 10)
        self.assertIn("volatility_20d", res["history"][0])
        self.assertIn("sma_20", res["history"][0])
        self.assertIn("return_5d", res["history"][0])

    def test_stock_news(self):
        news = get_stock_news("AAPL", limit=4)
        self.assertEqual(news["symbol"], "AAPL")
        self.assertIn("articles", news)
        self.assertIn("overall_sentiment", news)
        self.assertGreaterEqual(len(news["articles"]), 1)

    def test_stock_search(self):
        res = search_stocks(q="AAPL")
        self.assertIsInstance(res, list)
        self.assertGreaterEqual(len(res), 1)
        symbols = [r["symbol"] for r in res]
        self.assertIn("AAPL", symbols)


if __name__ == "__main__":
    unittest.main()
