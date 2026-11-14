import unittest
from backend.services.recommendation_service import recommendation_service


class TestRecommendations(unittest.TestCase):
    def test_scan_and_recommend_returns_results(self):
        result = recommendation_service.scan_and_recommend(limit=6)
        self.assertIn("recommendations", result)
        self.assertIn("total_scanned", result)
        self.assertGreater(len(result["recommendations"]), 0)

        top = result["recommendations"][0]
        self.assertIn("symbol", top)
        self.assertIn("name", top)
        self.assertIn("profit_score", top)
        self.assertIn("reasons_for_profit", top)
        self.assertGreaterEqual(len(top["reasons_for_profit"]), 3)
        self.assertIn("expected_upside", top)
        self.assertIn("risk_level", top)

    def test_filter_by_category_ipo(self):
        result = recommendation_service.scan_and_recommend(category="ipo", limit=5)
        for rec in result["recommendations"]:
            self.assertEqual(rec["category"], "IPO")

    def test_filter_by_market_india(self):
        result = recommendation_service.scan_and_recommend(market="india", limit=5)
        for rec in result["recommendations"]:
            self.assertEqual(rec["market"], "India")

    def test_filter_by_market_us(self):
        result = recommendation_service.scan_and_recommend(market="us", limit=5)
        for rec in result["recommendations"]:
            self.assertEqual(rec["market"], "US")


if __name__ == "__main__":
    unittest.main()
