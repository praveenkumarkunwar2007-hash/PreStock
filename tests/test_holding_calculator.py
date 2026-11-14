import unittest
from backend.services.holding_calculator import holding_calculator


class TestHoldingCalculator(unittest.TestCase):
    def setUp(self):
        self.mock_features = {
            "Close": 150.0,
            "Volatility_20D": 0.24,
            "RSI_14": 58.5,
            "Price_vs_SMA20": 0.025,
            "Price_vs_SMA50": 0.040,
            "Volume_Ratio": 1.30,
            "Return_5D": 0.035
        }
        self.mock_prediction = {
            "prediction": "Profit",
            "confidence": 0.78,
            "probabilities": {
                "Loss": 0.12,
                "Neutral": 0.10,
                "Profit": 0.78
            }
        }

    def test_flexible_horizon_range(self):
        # Test 1-day horizon
        res_1d = holding_calculator.analyze_horizon_and_holding(
            features=self.mock_features,
            base_prediction=self.mock_prediction,
            target_horizon=1
        )
        self.assertEqual(res_1d["target_horizon_days"], 1)
        self.assertIn("prediction", res_1d)
        self.assertIn("optimal_holding", res_1d)

        # Test 45-day horizon
        res_45d = holding_calculator.analyze_horizon_and_holding(
            features=self.mock_features,
            base_prediction=self.mock_prediction,
            target_horizon=45
        )
        self.assertEqual(res_45d["target_horizon_days"], 45)
        self.assertIn("probabilities", res_45d)

    def test_optimal_holding_period_calculation(self):
        res = holding_calculator.analyze_horizon_and_holding(
            features=self.mock_features,
            base_prediction=self.mock_prediction,
            target_horizon=10
        )
        opt = res["optimal_holding"]
        self.assertIn("optimal_days", opt)
        self.assertGreaterEqual(opt["optimal_days"], 1)
        self.assertLessEqual(opt["optimal_days"], 45)
        self.assertIn("duration_label", opt)
        self.assertIn("peak_profit_pct", opt)
        self.assertIn("peak_target_price", opt)
        self.assertIn("rationale", opt)
        self.assertIn("trajectory", opt)
        self.assertGreater(len(opt["trajectory"]), 0)

        # Check trajectory contains the optimal day flagged
        has_optimal_flag = any(t["is_optimal"] for t in opt["trajectory"])
        self.assertTrue(has_optimal_flag)


if __name__ == "__main__":
    unittest.main()
