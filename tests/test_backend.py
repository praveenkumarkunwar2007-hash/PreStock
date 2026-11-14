import unittest
from backend.services.stock_data import stock_data_service
from backend.routes.predict import predict_by_symbol, predict_custom, CustomFeatureInput
from backend.routes.stocks import get_all_stocks, get_stock_history, get_stock_news, search_stocks, get_recommendations


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
        self.assertEqual(res["target_horizon_days"], 5)
        self.assertIn("optimal_holding", res)
        self.assertIn("optimal_days", res["optimal_holding"])

    def test_predict_flexible_horizon_and_holding(self):
        # Test horizon at 14 days
        res_14d = predict_by_symbol("AAPL", days=14)
        self.assertEqual(res_14d["target_horizon_days"], 14)
        self.assertIn("expected_return_pct", res_14d)
        self.assertIn("target_price", res_14d)
        self.assertIn("optimal_holding", res_14d)
        opt = res_14d["optimal_holding"]
        self.assertGreaterEqual(opt["optimal_days"], 1)
        self.assertLessEqual(opt["optimal_days"], 45)
        self.assertIn("duration_label", opt)
        self.assertIn("peak_profit_pct", opt)
        self.assertIn("trajectory", opt)
        self.assertGreater(len(opt["trajectory"]), 0)

        # Test bounds clipping (e.g. 99 -> 45)
        res_clipped = predict_by_symbol("AAPL", days=99)
        self.assertEqual(res_clipped["target_horizon_days"], 45)

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

    def test_recommendations_route(self):
        res = get_recommendations(category="all", market="all", limit=5)
        self.assertIn("recommendations", res)
        self.assertIn("total_scanned", res)
        self.assertGreater(len(res["recommendations"]), 0)
        top = res["recommendations"][0]
        self.assertIn("reasons_for_profit", top)
        self.assertIn("profit_score", top)


if __name__ == "__main__":
    unittest.main()
