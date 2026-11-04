import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "backend" / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "xgboost_5d_v2.joblib"
DEFAULT_METADATA_PATH = MODEL_DIR / "xgboost_5d_v2_metadata.json"


class StockPredictor:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None
    ):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.metadata_path = metadata_path or DEFAULT_METADATA_PATH
        self.model = None
        self.metadata = {}
        self.features: List[str] = []
        self.classes: Dict[str, str] = {}
        self._load()

    def _load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.model = joblib.load(self.model_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.features = self.metadata.get("features", [])
        self.classes = self.metadata.get("classes", {"0": "Loss", "1": "Neutral", "2": "Profit"})

    def predict_features(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Run inference on a dictionary containing feature values.
        """
        missing_features = [f for f in self.features if f not in feature_dict]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        input_df = pd.DataFrame([{f: float(feature_dict[f]) for f in self.features}])
        probs = self.model.predict_proba(input_df)[0]
        pred_idx = int(np.argmax(probs))
        predicted_class = self.classes.get(str(pred_idx), "Unknown")

        prob_dict = {
            self.classes.get(str(i), f"Class_{i}"): float(prob)
            for i, prob in enumerate(probs)
        }

        return {
            "prediction": predicted_class,
            "prediction_index": pred_idx,
            "confidence": float(probs[pred_idx]),
            "probabilities": prob_dict,
            "horizon": self.metadata.get("prediction_horizon", "5 trading days"),
            "model_name": self.metadata.get("model_name", "XGBoost Stock Classifier"),
        }


# Global singleton instance
predictor = StockPredictor()
