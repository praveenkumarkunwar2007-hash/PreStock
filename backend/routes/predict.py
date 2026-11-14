from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.services.predictor import predictor
from backend.services.stock_data import stock_data_service
from backend.services.news_service import news_service
from backend.services.rationale import generate_prediction_rationale
from backend.services.holding_calculator import holding_calculator

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
    Close: Optional[float] = Field(100.0, description="Current stock close price")
    days: Optional[int] = Field(5, description="Prediction horizon in trading days (1-45)")


@router.get("/symbol/{symbol}")
def predict_by_symbol(
    symbol: str,
    days: int = Query(5, ge=1, le=45, description="Prediction horizon in trading days")
):
    """
    Generate a flexible horizon prediction (1 to 45 days) for a tracked stock or IPO symbol,
    including optimal holding period calculation, data-backed rationale, and live sentiment.
    """
    # Safe parameter unpacking for direct unit test / internal calls
    resolved_days = int(getattr(days, "default", days)) if not isinstance(days, int) else days
    resolved_days = max(1, min(resolved_days, 45))

    latest_features = stock_data_service.get_latest_features(symbol)
    if not latest_features:
        raise HTTPException(
            status_code=404,
            detail=f"Stock symbol '{symbol}' not found in local feature datasets."
        )

    try:
        base_result = predictor.predict_features(latest_features)
        resolved_symbol = str(latest_features.get("Symbol", symbol)).upper()

        # Fetch open-source live news sentiment to correlate with prediction
        news_data = news_service.fetch_stock_news(resolved_symbol, limit=4)
        market_sentiment = news_data.get("overall_sentiment", "Neutral")

        # Perform multi-horizon evaluation and optimal holding period calculation
        holding_analysis = holding_calculator.analyze_horizon_and_holding(
            features=latest_features,
            base_prediction=base_result,
            target_horizon=resolved_days,
            news_sentiment=market_sentiment
        )

        result = {
            "symbol": resolved_symbol,
            "date": str(latest_features.get("Date", "")).split()[0],
            "latest_close": float(latest_features.get("Close", 0.0)),
            "daily_return": float(latest_features.get("Daily_Return", 0.0) or 0.0),
            "return_5d": float(latest_features.get("Return_5D", 0.0) or 0.0),
            "rsi_14": float(latest_features.get("RSI_14", 50.0) or 50.0),
            "volatility_20d": float(latest_features.get("Volatility_20D", 0.20) or 0.20),
            "volume_ratio": float(latest_features.get("Volume_Ratio", 1.0) or 1.0),
            "target_horizon_days": holding_analysis["target_horizon_days"],
            "target_horizon_label": holding_analysis["target_horizon_label"],
            "prediction": holding_analysis["prediction"],
            "confidence": holding_analysis["confidence"],
            "probabilities": holding_analysis["probabilities"],
            "expected_return_pct": holding_analysis["expected_return_pct"],
            "target_price": holding_analysis["target_price"],
            "optimal_holding": holding_analysis["optimal_holding"],
            "news_sentiment": market_sentiment
        }

        # Generate interpretability rationale calibrated to this horizon
        rationale = generate_prediction_rationale(
            features=latest_features,
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            news_sentiment=market_sentiment
        )
        result["rationale"] = rationale

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/custom")
def predict_custom(payload: CustomFeatureInput):
    """
    Generate a prediction from custom raw feature values across flexible horizons (1-45 days),
    including optimal holding period calculation and technical rationale.
    """
    try:
        feature_dict = payload.model_dump()
        target_days = max(1, min(int(feature_dict.get("days", 5) or 5), 45))
        base_result = predictor.predict_features(feature_dict)

        holding_analysis = holding_calculator.analyze_horizon_and_holding(
            features=feature_dict,
            base_prediction=base_result,
            target_horizon=target_days,
            news_sentiment="Neutral"
        )

        result = {
            "target_horizon_days": holding_analysis["target_horizon_days"],
            "target_horizon_label": holding_analysis["target_horizon_label"],
            "prediction": holding_analysis["prediction"],
            "confidence": holding_analysis["confidence"],
            "probabilities": holding_analysis["probabilities"],
            "expected_return_pct": holding_analysis["expected_return_pct"],
            "target_price": holding_analysis["target_price"],
            "optimal_holding": holding_analysis["optimal_holding"],
            "news_sentiment": "Neutral"
        }

        rationale = generate_prediction_rationale(
            features=feature_dict,
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"]
        )
        result["rationale"] = rationale
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
