"""ai_insights.py – ML predictions and rule-based AI recommendations.

Two capabilities:
1. predict_wfh_growth()  – scikit-learn LinearRegression trained on county
   features (broadband gap, income score) to forecast WFH% in 2026 and 2027.
2. generate_recommendations() – translates each county's scoring-model metrics
   into prioritised, human-readable Action items. No external API required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# ML – WFH Growth Prediction
# ---------------------------------------------------------------------------

def predict_wfh_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Train a simple linear model and project WFH% to 2026 and 2027.

    Features: Broadband_Gap, Median_Income (normalised)
    Target:   WFH_Change (2019→2024 percentage-point growth)

    Returns the input DataFrame with four new columns:
        Predicted_WFH_Change  – model estimate of the 2019-2024 change
        WFH_Pct_2026          – projected WFH% in 2026
        WFH_Pct_2027          – projected WFH% in 2027
        Trend_Label           – qualitative label (Strong / Moderate / Low)
    """
    df = df.copy()

    features = df[["Broadband_Gap", "Median_Income"]].copy()

    # Replace any NaNs with column medians so the model always gets clean data
    features = features.fillna(features.median())
    target = df["WFH_Change"].fillna(df["WFH_Change"].median())

    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    y = target.values

    model = LinearRegression()
    model.fit(X, y)

    predicted_change = model.predict(X)
    df["Predicted_WFH_Change"] = predicted_change

    # Annual growth rate = predicted 5-year change / 5
    annual_rate = predicted_change / 5.0

    df["WFH_Pct_2026"] = df["WFH_Pct_2024"] + annual_rate * 2
    df["WFH_Pct_2027"] = df["WFH_Pct_2024"] + annual_rate * 3

    # Qualitative label based on predicted 5-year change
    def _label(v: float) -> str:
        if v >= 5.0:
            return "Strong"
        if v >= 2.0:
            return "Moderate"
        return "Low"

    df["Trend_Label"] = df["Predicted_WFH_Change"].apply(_label)

    return df


# ---------------------------------------------------------------------------
# Rule-based AI Recommendations
# ---------------------------------------------------------------------------

def _percentile_band(series: pd.Series, value: float) -> str:
    """Return 'high', 'medium', or 'low' relative to the series distribution."""
    p33 = series.quantile(0.33)
    p67 = series.quantile(0.67)
    if value >= p67:
        return "high"
    if value >= p33:
        return "medium"
    return "low"


def _score_to_priority(score: float, max_score: float) -> str:
    ratio = score / max_score if max_score > 0 else 0
    if ratio >= 0.70:
        return "🔴 Top Priority"
    if ratio >= 0.45:
        return "🟡 Medium Priority"
    return "🟢 Lower Priority"


def generate_recommendations(df: pd.DataFrame) -> list[dict]:
    """Return a list of recommendation dicts, one per county in *df*.

    Each dict has keys:
        county          – county name
        priority        – overall investment priority label
        opportunity_score
        headline        – one-sentence summary
        bullets         – list of specific recommendation strings
        strengths       – list of metric strengths
        watch_outs      – list of caution notes
    """
    if df.empty:
        return []

    max_score = df["Opportunity_Score"].max()
    recs = []

    for _, row in df.iterrows():
        county = row["County"]
        score = row["Opportunity_Score"]
        gap = row["Broadband_Gap"]
        wfh24 = row["WFH_Pct_2024"]
        income = row["Median_Income"]
        revenue = row["Potential_Annual_Revenue"]

        gap_band = _percentile_band(df["Broadband_Gap"], gap)
        wfh_band = _percentile_band(df["WFH_Pct_2024"], wfh24)
        income_band = _percentile_band(df["Median_Income"], income)
        priority = _score_to_priority(score, max_score)

        bullets: list[str] = []
        strengths: list[str] = []
        watch_outs: list[str] = []

        # --- Broadband gap logic ---
        if gap_band == "high":
            bullets.append(
                f"Prioritise infrastructure build-out: {gap:.1f}% of households "
                "lack broadband — the largest unmet demand tier."
            )
            strengths.append(f"Very high broadband gap ({gap:.1f}%) → large addressable market")
        elif gap_band == "medium":
            bullets.append(
                f"Target broadband gap ({gap:.1f}%) with a phased rollout. "
                "Focus first on denser census tracts to reduce cost per home passed."
            )
        else:
            bullets.append(
                f"Broadband gap is relatively low ({gap:.1f}%). "
                "Compete on speed and reliability rather than coverage alone."
            )
            watch_outs.append(f"Lower broadband gap ({gap:.1f}%) means higher existing competition")

        # --- WFH demand logic ---
        if wfh_band == "high":
            bullets.append(
                f"Lead marketing with gigabit home-office plans — "
                f"{wfh24:.1f}% WFH rate signals strong demand for premium tiers."
            )
            strengths.append(f"High WFH rate ({wfh24:.1f}%) → premium plan adoption likely")
        elif wfh_band == "medium":
            bullets.append(
                f"Bundle work-from-home packages with mid-tier plans. "
                f"WFH rate ({wfh24:.1f}%) is growing and warrants proactive positioning."
            )
        else:
            bullets.append(
                f"WFH penetration is currently low ({wfh24:.1f}%). "
                "Consider lower-cost entry plans to build subscriber base first."
            )
            watch_outs.append(f"Lower WFH rate ({wfh24:.1f}%) may limit premium plan uptake initially")

        # --- Income logic ---
        if income_band == "high":
            bullets.append(
                f"High median income (${income:,.0f}) supports premium pricing. "
                "Offer 1 Gbps and 2 Gbps tiers at $90–$120/month."
            )
            strengths.append(f"Strong income (${income:,.0f}) → higher ARPU potential")
        elif income_band == "medium":
            bullets.append(
                f"Mid-range income (${income:,.0f}) suggests a balanced tier structure "
                "(100 Mbps at ~$60 and 500 Mbps at ~$80) will maximise take rate."
            )
        else:
            bullets.append(
                f"Lower median income (${income:,.0f}) — consider subsidised entry plans "
                "or apply for ACP / BEAD federal funding to offset build costs."
            )
            watch_outs.append(
                f"Lower income (${income:,.0f}) may require subsidy programs to reach households"
            )

        # --- Revenue projection ---
        bullets.append(
            f"Estimated potential annual revenue: ${revenue:,.0f} "
            f"(at 15% adoption of unserved households × $80/month)."
        )

        # --- Headline ---
        if priority == "🔴 Top Priority":
            headline = (
                f"{county} is a top-tier target — high unmet demand, strong WFH trends, "
                "and solid income combine for the best overall opportunity."
            )
        elif priority == "🟡 Medium Priority":
            headline = (
                f"{county} offers a solid mid-tier opportunity with at least one "
                "strong driver. Selective investment recommended."
            )
        else:
            headline = (
                f"{county} has a lower composite score. Monitor growth trends "
                "before committing heavy capital."
            )

        recs.append(
            {
                "county": county,
                "priority": priority,
                "opportunity_score": score,
                "headline": headline,
                "bullets": bullets,
                "strengths": strengths,
                "watch_outs": watch_outs,
            }
        )

    # Sort by score descending
    recs.sort(key=lambda r: r["opportunity_score"], reverse=True)
    return recs
