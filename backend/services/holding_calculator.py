import math
from typing import Dict, Any, List, Optional


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class HoldingPeriodCalculator:
    """
    Quantitative engine to project predictions across flexible horizons (1 to 45 days)
    and compute the optimal holding duration for maximum expected profit.
    """

    def analyze_horizon_and_holding(
        self,
        features: Dict[str, Any],
        base_prediction: Dict[str, Any],
        target_horizon: int = 5,
        news_sentiment: Optional[str] = "Neutral"
    ) -> Dict[str, Any]:
        horizon = max(1, min(int(target_horizon), 45))

        close_price = float(features.get("Close", 100.0) or 100.0)
        if close_price <= 0:
            close_price = 100.0

        vol_20d = float(features.get("Volatility_20D", 0.22) or 0.22)
        vol_20d = max(0.08, min(0.85, vol_20d))

        rsi = float(features.get("RSI_14", 50.0) or 50.0)
        price_vs_sma20 = float(features.get("Price_vs_SMA20", 0.0) or 0.0)
        price_vs_sma50 = float(features.get("Price_vs_SMA50", 0.0) or 0.0)
        vol_ratio = float(features.get("Volume_Ratio", 1.0) or 1.0)
        ret_5d = float(features.get("Return_5D", 0.0) or 0.0)

        base_probs = base_prediction.get("probabilities", {"Loss": 0.33, "Neutral": 0.34, "Profit": 0.33})
        p_profit_base = float(base_probs.get("Profit", 0.33))
        p_loss_base = float(base_probs.get("Loss", 0.33))
        pred_label_base = base_prediction.get("prediction", "Neutral")

        # 1. Base Expected Daily Drift Rate (μ_daily)
        # Derived from ML probability edge + technical momentum factors + sentiment
        ml_directional_edge = (p_profit_base - p_loss_base)  # -1.0 to +1.0
        tech_momentum = 0.0

        # RSI momentum contribution
        if 50 <= rsi <= 68:
            tech_momentum += 0.0012 * ((rsi - 50) / 18.0)
        elif rsi < 35:
            # Oversold bounce catalyst
            tech_momentum += 0.0018 * ((35 - rsi) / 20.0)
        elif rsi > 70:
            # Overbought exhaustion dampening
            tech_momentum -= 0.0015 * ((rsi - 70) / 20.0)
        else:
            tech_momentum -= 0.0008 * ((50 - rsi) / 15.0)

        # Moving average alignment contribution
        if price_vs_sma20 > 0:
            tech_momentum += 0.0010
        else:
            tech_momentum -= 0.0010

        if price_vs_sma50 > 0:
            tech_momentum += 0.0008
        else:
            tech_momentum -= 0.0008

        # Volume confirmation multiplier
        volume_mult = 1.0
        if vol_ratio >= 1.2:
            volume_mult = 1.25
        elif vol_ratio < 0.8:
            volume_mult = 0.85

        # News sentiment boost
        news_drift = 0.0
        if news_sentiment == "Bullish":
            news_drift = 0.0010
        elif news_sentiment == "Bearish":
            news_drift = -0.0012

        daily_drift = (ml_directional_edge * 0.004 + tech_momentum + news_drift) * volume_mult

        # 2. Multi-Day Curve Evaluation (Day 1 to 45)
        # Evaluates expected return, volatility, risk-adjusted score, and profit probability for each day
        day_results = []
        cumulative_return = 0.0

        for day in range(1, 46):
            # Momentum persistence decays over time as technical forces encounter resistance or mean-reversion
            if day <= 5:
                decay = 1.0
            elif day <= 15:
                decay = math.exp(-0.02 * (day - 5))
            elif day <= 30:
                decay = math.exp(-0.035 * (day - 5))
            else:
                decay = math.exp(-0.05 * (day - 5))

            # Cumulative expected return
            marginal_return = daily_drift * decay
            cumulative_return += marginal_return

            # Cumulative volatility over 'day' trading sessions
            day_vol = vol_20d * math.sqrt(day / 252.0)

            # Profit threshold scales slightly with horizon (e.g. 0.3% at 1D, 1.0% at 5D, 2.5% at 45D)
            threshold = 0.004 * math.sqrt(day / 5.0)

            # Normal distribution probability mapping
            denom = max(0.001, day_vol)
            z_profit = (cumulative_return - threshold) / denom
            z_loss = (-cumulative_return - threshold) / denom

            prob_profit_day = normal_cdf(z_profit)
            prob_loss_day = normal_cdf(z_loss)

            # Smoothly blend with base model at day 5 to ensure perfect calibration
            if day == 5:
                prob_profit_day = 0.85 * p_profit_base + 0.15 * prob_profit_day
                prob_loss_day = 0.85 * p_loss_base + 0.15 * prob_loss_day

            prob_neutral_day = max(0.05, 1.0 - prob_profit_day - prob_loss_day)

            # Normalize to sum exactly to 1.0
            total_prob = prob_profit_day + prob_neutral_day + prob_loss_day
            prob_profit_day = round(prob_profit_day / total_prob, 3)
            prob_neutral_day = round(prob_neutral_day / total_prob, 3)
            prob_loss_day = round(prob_loss_day / total_prob, 3)

            # Determine dominant prediction class for this horizon
            if prob_profit_day >= prob_loss_day and prob_profit_day >= prob_neutral_day:
                day_class = "Profit"
                day_conf = prob_profit_day
            elif prob_loss_day >= prob_profit_day and prob_loss_day >= prob_neutral_day:
                day_class = "Loss"
                day_conf = prob_loss_day
            else:
                day_class = "Neutral"
                day_conf = prob_neutral_day

            projected_price = round(close_price * (1.0 + cumulative_return), 2)

            # Risk-adjusted profit score: Rewards higher expected gain while penalizing excess volatility drag
            if cumulative_return > 0:
                profit_efficiency_score = (cumulative_return / (day_vol ** 0.65)) * prob_profit_day
            else:
                profit_efficiency_score = cumulative_return * (1.0 + day_vol)

            day_results.append({
                "day": day,
                "expected_return": cumulative_return,
                "expected_return_pct": round(cumulative_return * 100.0, 2),
                "projected_price": projected_price,
                "profit_probability": prob_profit_day,
                "neutral_probability": prob_neutral_day,
                "loss_probability": prob_loss_day,
                "prediction": day_class,
                "confidence": day_conf,
                "profit_efficiency_score": profit_efficiency_score
            })

        # 3. Prediction at the Specific User Horizon (target_horizon)
        target_idx = horizon - 1
        target_day_res = day_results[target_idx]

        # 4. Deep Calculation of the Optimal Holding Duration for Maximum Profit
        # We find the day T* where profit potential is maximized
        if pred_label_base == "Loss" or daily_drift < -0.001:
            # Bearish scenario: Holding is not recommended for profit
            best_day = 1
            best_res = day_results[0]
            holding_strategy_type = "Defensive / Immediate Exit"
            optimal_rationale = (
                f"Defensive Assessment: Quantitative indicators and model drift project ongoing downward price pressure. "
                f"Holding for extended duration increases capital drawdown risk. If holding, day 1 to 2 represents the minimum downside risk window."
            )
        elif pred_label_base == "Neutral" and abs(daily_drift) < 0.0008:
            # Range-bound consolidation
            best_day = min(5, horizon)
            best_res = day_results[best_day - 1]
            holding_strategy_type = "Range-Bound Swing"
            optimal_rationale = (
                f"Consolidation Assessment: Stock is oscillating in a balanced channel without aggressive trend expansion. "
                f"Holding for {best_day} trading days captures short-term mean-reversion swings before range-bound chop sets in."
            )
        else:
            # Bullish / Profit Scenario: Find the mathematical peak profit day
            # Search within 1 to 45 days for maximum expected profit efficiency
            positive_days = [d for d in day_results if d["expected_return"] > 0]
            if positive_days:
                # Maximize profit efficiency
                best_res = max(positive_days, key=lambda d: d["profit_efficiency_score"])
                best_day = best_res["day"]
            else:
                best_res = max(day_results, key=lambda d: d["expected_return"])
                best_day = best_res["day"]

            weeks_approx = round(best_day / 5.0, 1)
            holding_strategy_type = "Peak Profit Acceleration"
            optimal_rationale = (
                f"Peak Profit Window Identified: Technical momentum (RSI {rsi:.1f}, Volume {vol_ratio:.1f}x) "
                f"and ML probabilistic expansion indicate that buying pressure accelerates strongest over the first {best_day} trading days "
                f"(~{weeks_approx} calendar weeks), reaching an optimal projected profit of +{best_res['expected_return_pct']}%. "
                f"Beyond day {best_day}, momentum decay and resistance levels reduce the risk-adjusted reward, making day {best_day} the peak profit target."
            )

        # Human-friendly duration label
        if best_day == 1:
            duration_label = "1 Trading Day (Intraday / Next Session)"
        elif best_day <= 5:
            duration_label = f"{best_day} Trading Days (~1 Calendar Week)"
        elif best_day <= 10:
            duration_label = f"{best_day} Trading Days (~2 Calendar Weeks)"
        elif best_day <= 20:
            duration_label = f"{best_day} Trading Days (~3-4 Calendar Weeks)"
        else:
            duration_label = f"{best_day} Trading Days (~{round(best_day / 5.0, 1)} Weeks)"

        # Target Take-Profit and Stop-Loss Levels
        target_upside_pct = max(0.015, best_res["expected_return"])
        take_profit_price = round(close_price * (1.0 + target_upside_pct), 2)
        # Recommended Stop Loss based on ATR/volatility
        stop_loss_pct = max(0.012, min(0.05, vol_20d * math.sqrt(best_day / 252.0) * 0.75))
        stop_loss_price = round(close_price * (1.0 - stop_loss_pct), 2)

        risk_reward_ratio = round(target_upside_pct / stop_loss_pct, 1)

        # 5. Build Trajectory Milestones for Visualization & Charting
        # Milestone days: [1, 3, 5, 7, 10, 14, 21, 30, 45] (filtered up to max(horizon, best_day))
        key_milestones = sorted(list(set([1, 3, 5, 7, 10, 14, 21, 30, 45, horizon, best_day])))
        milestone_trajectory = []

        for m_day in key_milestones:
            res_item = day_results[m_day - 1]
            if m_day == best_day:
                phase = "🎯 Peak Profit Target"
            elif m_day <= 3:
                phase = "Initial Momentum"
            elif m_day < best_day:
                phase = "Growth Acceleration"
            elif m_day <= best_day + 7:
                phase = "Take-Profit Zone"
            else:
                phase = "Consolidation / Decay"

            milestone_trajectory.append({
                "day": m_day,
                "day_label": f"Day {m_day}",
                "expected_return_pct": res_item["expected_return_pct"],
                "projected_price": res_item["projected_price"],
                "profit_probability": res_item["profit_probability"],
                "prediction": res_item["prediction"],
                "is_optimal": (m_day == best_day),
                "is_user_selected": (m_day == horizon),
                "phase": phase
            })

        return {
            "target_horizon_days": horizon,
            "target_horizon_label": f"{horizon} Trading Day{'s' if horizon > 1 else ''}",
            "prediction": target_day_res["prediction"],
            "confidence": target_day_res["confidence"],
            "probabilities": {
                "Loss": target_day_res["loss_probability"],
                "Neutral": target_day_res["neutral_probability"],
                "Profit": target_day_res["profit_probability"]
            },
            "expected_return_pct": target_day_res["expected_return_pct"],
            "target_price": target_day_res["projected_price"],
            "optimal_holding": {
                "optimal_days": best_day,
                "duration_label": duration_label,
                "peak_profit_pct": best_res["expected_return_pct"],
                "peak_target_price": best_res["projected_price"],
                "strategy_type": holding_strategy_type,
                "take_profit_price": take_profit_price,
                "stop_loss_price": stop_loss_price,
                "stop_loss_pct": round(stop_loss_pct * 100.0, 2),
                "risk_reward_ratio": f"{risk_reward_ratio} : 1",
                "rationale": optimal_rationale,
                "trajectory": milestone_trajectory
            }
        }


# Global singleton instance
holding_calculator = HoldingPeriodCalculator()
