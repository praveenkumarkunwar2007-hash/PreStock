import os
import json
import logging
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
FALLBACK_GROQ_MODEL = "llama-3.1-8b-instant"


class GroqService:
    """
    Groq AI Quantitative Analyst Service.
    Leverages Groq's free ultra-fast LPU inference (LLaMA-3.3-70B / LLaMA-3.1-8B)
    to generate institutional-grade investment research, technical verdicts,
    and catalyst breakdowns.
    """

    def __init__(self):
        self._default_api_key = os.getenv("GROQ_API_KEY", "").strip()

    def generate_ai_analysis(
        self,
        symbol: str,
        features: Dict[str, Any],
        prediction_data: Dict[str, Any],
        news_sentiment: str = "Neutral",
        news_headlines: Optional[List[str]] = None,
        custom_question: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive, actionable AI investment memo for a stock or index.
        Uses Groq API if key is available, or structured quant analysis if key is absent.
        """
        key_candidate = api_key if isinstance(api_key, str) else self._default_api_key
        effective_key = (key_candidate or "").strip()
        sym_clean = symbol.upper().strip()
        price = float(features.get("Close", 0.0) or 0.0)
        cur_sym = "₹" if sym_clean.endswith(".NS") or sym_clean.endswith(".BO") or sym_clean in ["^NSEI", "^NSEBANK", "^BSESN"] else "$"

        horizon_days = prediction_data.get("target_horizon_days", 5)
        pred_label = prediction_data.get("prediction", "Neutral")
        confidence = float(prediction_data.get("confidence", 0.5)) * 100
        probs = prediction_data.get("probabilities", {})
        prob_profit = float(probs.get("Profit", 0.33)) * 100
        prob_loss = float(probs.get("Loss", 0.33)) * 100

        opt = prediction_data.get("optimal_holding", {})
        opt_days = opt.get("optimal_days", horizon_days)
        peak_profit = opt.get("peak_profit_pct", 0.0)
        stop_loss_pct = opt.get("stop_loss_pct", 3.0)
        stop_price = opt.get("stop_loss_price", price * 0.97)

        rsi = float(features.get("RSI_14", 50.0))
        volatility = float(features.get("Volatility_20D", 0.20)) * 100
        vol_ratio = float(features.get("Volume_Ratio", 1.0))
        sma20_pct = float(features.get("Price_vs_SMA20", 0.0)) * 100
        sma50_pct = float(features.get("Price_vs_SMA50", 0.0)) * 100

        # If a Groq API key is present, call Groq Cloud API
        if effective_key:
            groq_result = self._call_groq_api(
                api_key=effective_key,
                symbol=sym_clean,
                cur_sym=cur_sym,
                price=price,
                horizon_days=horizon_days,
                pred_label=pred_label,
                confidence=confidence,
                prob_profit=prob_profit,
                prob_loss=prob_loss,
                opt_days=opt_days,
                peak_profit=peak_profit,
                stop_loss_pct=stop_loss_pct,
                stop_price=stop_price,
                rsi=rsi,
                volatility=volatility,
                vol_ratio=vol_ratio,
                sma20_pct=sma20_pct,
                sma50_pct=sma50_pct,
                news_sentiment=news_sentiment,
                news_headlines=news_headlines or [],
                custom_question=custom_question
            )
            if groq_result:
                return groq_result

        # Resilient Fallback: High-precision quantitative analyst narrative
        return self._generate_quant_narrative(
            symbol=sym_clean,
            cur_sym=cur_sym,
            price=price,
            horizon_days=horizon_days,
            pred_label=pred_label,
            confidence=confidence,
            prob_profit=prob_profit,
            prob_loss=prob_loss,
            opt_days=opt_days,
            peak_profit=peak_profit,
            stop_loss_pct=stop_loss_pct,
            stop_price=stop_price,
            rsi=rsi,
            volatility=volatility,
            vol_ratio=vol_ratio,
            sma20_pct=sma20_pct,
            sma50_pct=sma50_pct,
            news_sentiment=news_sentiment,
            has_groq_key=bool(effective_key)
        )

    def _call_groq_api(
        self,
        api_key: str,
        symbol: str,
        cur_sym: str,
        price: float,
        horizon_days: int,
        pred_label: str,
        confidence: float,
        prob_profit: float,
        prob_loss: float,
        opt_days: int,
        peak_profit: float,
        stop_loss_pct: float,
        stop_price: float,
        rsi: float,
        volatility: float,
        vol_ratio: float,
        sma20_pct: float,
        sma50_pct: float,
        news_sentiment: str,
        news_headlines: List[str],
        custom_question: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Call Groq Cloud free endpoint using LLaMA-3.3-70B."""
        system_prompt = (
            "You are a senior quantitative portfolio manager and algorithmic trader. "
            "You provide concise, highly insightful, objective market analyses combining machine learning signals, "
            "technical market structure, and macro risk management. Use clean GitHub markdown formatting with bullet points."
        )

        headlines_text = "\n".join([f"- {h}" for h in news_headlines[:4]]) if news_headlines else "- No major catalysts reported."
        user_prompt = f"""
Asset: {symbol}
Current Price: {cur_sym}{price:,.2f}
Machine Learning Model: XGBoost Gradient Boosted Classifier
Forecast Horizon: {horizon_days} Trading Days
Model Prediction: {pred_label.upper()} (Confidence: {confidence:.1f}%)
Probabilities: Profit {prob_profit:.1f}%, Loss {prob_loss:.1f}%

Mathematical Optimal Holding Duration:
- Peak Profit Window: {opt_days} Trading Days
- Peak Projected Return: {peak_profit:+.2f}%
- Recommended Stop-Loss: {stop_loss_pct:.2f}% (Stop Price: {cur_sym}{stop_price:,.2f})

Technical Indicators:
- 14-Day RSI: {rsi:.1f}
- 20-Day Realized Volatility: {volatility:.1f}% annualized
- Volume vs 20-Day Average: {vol_ratio:.2f}x
- Price vs 20-Day SMA: {sma20_pct:+.1f}%
- Price vs 50-Day SMA: {sma50_pct:+.1f}%

Live Market News Sentiment: {news_sentiment}
Recent News Catalysts:
{headlines_text}

{f"User Inquiry: {custom_question}" if custom_question else "Please provide an executive investment memo with:"}
1. Executive Verdict & Directional Bias
2. Key Technical Drivers & Market Phase
3. Optimal Holding & Timing Strategy (Holding {opt_days} Days vs {horizon_days} Days)
4. Actionable Risk Management & Trade Plan (Entry, Target, Stop-Loss)
Keep the analysis punchy, professional, and directly actionable in 3 to 4 short sections.
"""

        payload = {
            "model": DEFAULT_GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.4,
            "max_tokens": 750
        }

        try:
            req = urllib.request.Request(
                GROQ_ENDPOINT,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PreStock-AI"
                }
            )
            with urllib.request.urlopen(req, timeout=6.0) as resp:
                data = json.loads(resp.read().decode())
                content = data["choices"][0]["message"]["content"]
                return {
                    "ai_analysis": content,
                    "model": DEFAULT_GROQ_MODEL,
                    "status": "success",
                    "provider": "Groq Cloud LPU (Ultra-Fast)",
                    "latency": "sub-second"
                }
        except Exception as e:
            logger.debug(f"Groq API call failed: {e}")
            return None

    def _generate_quant_narrative(
        self,
        symbol: str,
        cur_sym: str,
        price: float,
        horizon_days: int,
        pred_label: str,
        confidence: float,
        prob_profit: float,
        prob_loss: float,
        opt_days: int,
        peak_profit: float,
        stop_loss_pct: float,
        stop_price: float,
        rsi: float,
        volatility: float,
        vol_ratio: float,
        sma20_pct: float,
        sma50_pct: float,
        news_sentiment: str,
        has_groq_key: bool
    ) -> Dict[str, Any]:
        """Generate structured institutional quant memo if Groq key is not active."""
        trend_status = "Solid Bullish Uptrend" if sma50_pct > 0 else "Bearish / Corrective Trend"
        rsi_zone = "Oversold Bounce Candidate" if rsi < 35 else ("Overbought Caution" if rsi > 70 else "Healthy Equilibrium")
        volume_status = "Institutional Accumulation" if vol_ratio >= 1.1 else ("Light Volume Divergence" if vol_ratio < 0.9 else "Normal Market Turnover")

        memo = f"""### 📊 Quantitative Market Verdict: **{pred_label.upper()}** ({confidence:.1f}% Confidence)
- **Asset / Ticker**: **{symbol}** (Current Market Price: **{cur_sym}{price:,.2f}**)
- **Target Horizon**: **{horizon_days} Trading Days** (Profit Probability: **{prob_profit:.1f}%** vs Loss Probability: **{prob_loss:.1f}%**)

---

### ⏱️ Mathematically Optimal Holding Window
- **Peak Profit Period**: **{opt_days} Trading Days** (Optimal Risk-Adjusted Reward).
- **Projected Peak Return**: **{peak_profit:+.2f}%** based on stochastic drift velocity and momentum persistence.
- **Timing Rationale**: Holding beyond **{opt_days} days** subjects the position to increasing volatility drag and moving average mean-reversion.

---

### 🔍 Technical Indicator Breakdown
- **Momentum (RSI 14 = {rsi:.1f})**: Market is currently in a **{rsi_zone}**.
- **Moving Average Structure**: Trading at **{sma20_pct:+.1f}%** vs 20-Day SMA and **{sma50_pct:+.1f}%** vs 50-Day SMA (**{trend_status}**).
- **Volume & Flow**: Volume ratio at **{vol_ratio:.2f}x** indicating **{volume_status}**.
- **Realized Volatility**: Annualized at **{volatility:.1f}%**, requiring calibrated position sizing.
- **News Sentiment**: Classified as **{news_sentiment}**.

---

### 🛡️ Trade Execution & Risk Management Plan
- **Target Exit Price**: **{cur_sym}{price * (1 + peak_profit / 100):,.2f}**
- **Protective Stop-Loss**: **{stop_loss_pct:.2f}%** at **{cur_sym}{stop_price:,.2f}** (Strict Trailing Stop)
- **Risk / Reward**: Asymmetric setup favoring disciplined scaling."""

        return {
            "ai_analysis": memo,
            "model": "PreStock Quantitative Intelligence",
            "status": "quant_engine",
            "provider": "Built-in Quant Engine (Groq LLaMA Ready)",
            "tip": "Tip: Click '⚙️ Groq AI' in the top header to enter your free Groq API key from console.groq.com to enable live LLaMA 3.3 70B inference!"
        }


groq_service = GroqService()
