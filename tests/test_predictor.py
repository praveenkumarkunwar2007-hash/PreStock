import unittest
from backend.services.predictor import StockPredictor


class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = StockPredictor()

    def test_model_loaded(self):
        self.assertIsNotNone(self.predictor.model)
        self.assertGreater(len(self.predictor.features), 0)

    def test_predict_features(self):
        # Build valid feature dictionary with 0.0 defaults
        mock_features = {f: 0.0 for f in self.predictor.features}
        mock_features["RSI_14"] = 55.0
        mock_features["Volume_Ratio"] = 1.0

        res = self.predictor.predict_features(mock_features)
        self.assertIn("prediction", res)
        self.assertIn(res["prediction"], ["Loss", "Neutral", "Profit"])
        self.assertIn("confidence", res)
        self.assertIn("probabilities", res)

        # Probabilities must sum to ~1.0
        total_prob = sum(res["probabilities"].values())
        self.assertAlmostEqual(total_prob, 1.0, places=4)

    def test_missing_features_raise_error(self):
        with self.assertRaises(ValueError):
            self.predictor.predict_features({"RSI_14": 50.0})


if __name__ == "__main__":
    unittest.main()
