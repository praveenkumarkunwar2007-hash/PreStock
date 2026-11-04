from typing import Dict, Any, List, Optional


def generate_prediction_rationale(
    features: Dict[str, Any],
    prediction: str,
    confidence: float,
    probabilities: Dict[str, float],
    news_sentiment: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates transparent, human-interpretable technical reasons and catalysts
    explaining why the machine learning model assigned the predicted probability.
    """
    rsi = float(features.get("RSI_14", 50.0))
    vol_ratio = float(features.get("Volume_Ratio", 1.0))
    volatility = float(features.get("Volatility_20D", 0.20))
    ret_5d = float(features.get("Return_5D", 0.0))
    ret_20d = float(features.get("Return_20D", 0.0))
    price_vs_sma20 = float(features.get("Price_vs_SMA20", 0.0))
    price_vs_sma50 = float(features.get("Price_vs_SMA50", 0.0))
    bb_pos = float(features.get("BB_Position", 0.5))
    macd_hist = float(features.get("MACD_Histogram_pct", features.get("MACD_Histogram", 0.0)))

    bullish_signals: List[Dict[str, str]] = []
    bearish_signals: List[Dict[str, str]] = []

    # 1. RSI Analysis
    if rsi < 35:
        bullish_signals.append({
            "indicator": f"RSI 14 ({rsi:.1f})",
            "type": "Oversold Rebound Zone",
            "detail": "RSI is in deeply oversold territory, indicating extreme selling exhaustion and high probability of mean-reversion bounce."
        })
    elif 50 <= rsi <= 68:
        bullish_signals.append({
            "indicator": f"RSI 14 ({rsi:.1f})",
            "type": "Healthy Bullish Momentum",
            "detail": "RSI maintains positive momentum above neutral 50 without hitting overbought resistance (> 70)."
        })
    elif rsi > 70:
        bearish_signals.append({
            "indicator": f"RSI 14 ({rsi:.1f})",
            "type": "Overbought Caution Zone",
            "detail": "RSI is above 70, reflecting stretched valuation where institutional traders frequently take profits."
        })
    else:
        bearish_signals.append({
            "indicator": f"RSI 14 ({rsi:.1f})",
            "type": "Sub-50 Bearish Bias",
            "detail": "RSI is trending below the centerline 50, indicating sellers have the upper hand."
        })

    # 2. Moving Average Alignment
    if price_vs_sma50 > 0.01:
        bullish_signals.append({
            "indicator": f"SMA 50 (+{price_vs_sma50 * 100:.1f}%)",
            "type": "Medium-Term Trend Support",
            "detail": f"Price is trading {price_vs_sma50 * 100:.1f}% above the 50-day moving average, confirming a solid structural uptrend."
        })
    elif price_vs_sma50 < -0.01:
        bearish_signals.append({
            "indicator": f"SMA 50 ({price_vs_sma50 * 100:.1f}%)",
            "type": "Below Medium-Term Trend",
            "detail": f"Price is trading {abs(price_vs_sma50) * 100:.1f}% below the 50-day moving average, signaling ongoing trend pressure."
        })

    if price_vs_sma20 > 0.0:
        bullish_signals.append({
            "indicator": f"SMA 20 (+{price_vs_sma20 * 100:.1f}%)",
            "type": "Short-Term Momentum",
            "detail": "Price holds above the 20-day SMA, providing short-term dynamic support."
        })
    else:
        bearish_signals.append({
            "indicator": f"SMA 20 ({price_vs_sma20 * 100:.1f}%)",
            "type": "Short-Term Resistance",
            "detail": "Price has slipped below the 20-day SMA, turning moving average into overhead resistance."
        })

    # 3. Volume Confirmation
    if vol_ratio >= 1.15:
        bullish_signals.append({
            "indicator": f"Volume Ratio ({vol_ratio:.2f}x)",
            "type": "Volume Surge Confirmation",
            "detail": f"Recent volume is {(vol_ratio - 1) * 100:.0f}% higher than the 20-day average, signaling institutional accumulation."
        })
    elif vol_ratio < 0.85:
        bearish_signals.append({
            "indicator": f"Volume Ratio ({vol_ratio:.2f}x)",
            "type": "Low Volume Participation",
            "detail": "Trading volume is running significantly below average, warning of weak conviction behind recent price moves."
        })

    # 4. MACD & Volatility
    if macd_hist > 0:
        bullish_signals.append({
            "indicator": "MACD Histogram",
            "type": "Bullish Momentum Expanding",
            "detail": "MACD line is above the signal line, confirming upward acceleration."
        })
    else:
        bearish_signals.append({
            "indicator": "MACD Histogram",
            "type": "Bearish Divergence / Pullback",
            "detail": "MACD histogram is negative, reflecting decelerating upward momentum."
        })

    # Narrative Summary Construction
    prob_loss = probabilities.get("Loss", 0.0) * 100
    prob_neutral = probabilities.get("Neutral", 0.0) * 100
    prob_profit = probabilities.get("Profit", 0.0) * 100

    if prediction == "Profit":
        summary = (
            f"The XGBoost model forecasts a bullish 5-day horizon with a {confidence * 100:.1f}% probability of Profit "
            f"(vs {prob_loss:.1f}% Loss and {prob_neutral:.1f}% Neutral). "
            f"Key driving factors include strong trend alignment (trading {price_vs_sma50 * 100:+.1f}% vs 50-day SMA), "
            f"positive momentum on RSI ({rsi:.1f}), and {len(bullish_signals)} confirming technical indicators."
        )
    elif prediction == "Loss":
        summary = (
            f"The XGBoost model forecasts downside pressure with a {confidence * 100:.1f}% probability of Loss "
            f"(vs {prob_profit:.1f}% Profit and {prob_neutral:.1f}% Neutral). "
            f"The forecast is driven by downward price drift relative to moving averages ({price_vs_sma20 * 100:+.1f}% vs 20-day SMA), "
            f"cautionary RSI positioning ({rsi:.1f}), and {len(bearish_signals)} negative technical headwinds."
        )
    else:
        summary = (
            f"The XGBoost model projects a range-bound consolidation with a {confidence * 100:.1f}% probability of Neutral "
            f"movement (under ±1% 5-day change). "
            f"Opposing technical forces (balance between {len(bullish_signals)} bullish and {len(bearish_signals)} caution signals) "
            f"suggest market equilibrium without a decisive breakout trigger."
        )

    if news_sentiment:
        summary += f" Real-time market news sentiment currently registers as '{news_sentiment}'."

    return {
        "summary": summary,
        "prediction": prediction,
        "confidence_pct": round(confidence * 100, 1),
        "bullish_signals": bullish_signals,
        "bearish_signals": bearish_signals,
        "indicators_snapshot": {
            "rsi_14": round(rsi, 2),
            "volume_ratio": round(vol_ratio, 2),
            "volatility_20d_pct": round(volatility * 100, 1),
            "return_5d_pct": round(ret_5d * 100, 2),
            "return_20d_pct": round(ret_20d * 100, 2),
            "price_vs_sma20_pct": round(price_vs_sma20 * 100, 2),
            "price_vs_sma50_pct": round(price_vs_sma50 * 100, 2),
            "bb_position": round(bb_pos, 2)
        }
    }
