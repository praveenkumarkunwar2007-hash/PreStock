from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.predictor import predictor
from backend.services.stock_data import stock_data_service

router = APIRouter(prefix="/predict", tags=["Prediction"])


class CustomFeatureInput(BaseModel):
    Daily_Return: float = Field(..., description="1-day return (e.g. 0.012 for 1.2%)")
    Return_5D: float = Field(..., description="5-day percentage return")
    Return_10D: float = Field(..., description="10-day percentage return")
    Return_20D: float = Field(..., description="20-day percentage return")
    Price_vs_SMA20: float = Field(..., description="Price deviation from 20-day SMA")
    Price_vs_SMA50: float = Field(..., description="Price deviation from 50-day SMA")
    RSI_14: float = Field(..., description="14-period RSI (0-100)")
    BB_Width: float = Field(..., description="Bollinger Band Width")
    BB_Position: float = Field(..., description="Bollinger Band %b Position")
    ATR_Percent: float = Field(..., description="ATR normalized by Close price")
    Volatility_20D: float = Field(..., description="20-day annualized volatility")
    Volume_Ratio: float = Field(..., description="Current volume / 20-day SMA volume")
    MACD_pct: float = Field(..., description="MACD value / Close")
    MACD_Signal_pct: float = Field(..., description="MACD Signal / Close")
    MACD_Histogram_pct: float = Field(..., description="MACD Histogram / Close")


@router.get("/symbol/{symbol}")
def predict_by_symbol(symbol: str):
    """
    Generate a 5-day horizon prediction for a tracked stock symbol.
    """
    latest_features = stock_data_service.get_latest_features(symbol)
    if not latest_features:
        raise HTTPException(
            status_code=404,
            detail=f"Stock symbol '{symbol}' not found in local feature datasets."
        )

    try:
        result = predictor.predict_features(latest_features)
        result["symbol"] = symbol.upper()
        result["date"] = str(latest_features.get("Date"))
        result["latest_close"] = latest_features.get("Close")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/custom")
def predict_custom(payload: CustomFeatureInput):
    """
    Generate a prediction from custom raw feature values.
    """
    try:
        feature_dict = payload.model_dump()
        return predictor.predict_features(feature_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
