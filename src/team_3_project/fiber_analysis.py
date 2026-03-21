from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

plt.style.use("seaborn-v0_8-whitegrid")

COLOR_MAIN = "#4C78A8"
COLOR_ACCENT = "#F58518"
COLOR_HIGHLIGHT = "#54A24B"
FIBER_MONTHLY_PRICE = 80
TARGET_ADOPTION_RATE = 0.15


def min_max_scale(series: pd.Series) -> pd.Series:
    spread = series.max() - series.min()
    if spread == 0:
        return pd.Series(50, index=series.index, dtype=float)
    return (series - series.min()) / spread * 100


def generate_predictive_data(df: pd.DataFrame) -> pd.DataFrame:
    df_pred = df.copy()
    for year in [2025]:
        ratio = 1.0
        df_pred[f"WFH_Pct_{year}"] = (df_pred["WFH_Pct_2024"] + df_pred["WFH_Change"] * 0.6 * ratio).clip(0, 100)
        df_pred[f"Broadband_Gap_{year}"] = (df_pred["Broadband_Gap"] * (1 - 0.15 * ratio)).clip(0, 100)
        df_pred[f"Opportunity_Score_{year}"] = (
            0.40 * min_max_scale(df_pred[f"Broadband_Gap_{year}"])
            + 0.35 * min_max_scale(df_pred[f"WFH_Pct_{year}"])
            + 0.25 * min_max_scale(df_pred["Median_Income"])
        )
    top_5_indices = df_pred.nlargest(5, "Opportunity_Score_2025").index
    df_pred["Flagged"] = False
    df_pred.loc[top_5_indices, "Flagged"] = True
    return df_pred


def fetch_county_data(state_fip: str = "37") -> pd.DataFrame:
    base_url = "https://api.census.gov/data"
    variables = [
        "NAME",
        "B08301_021E",
        "B08301_001E",
        "B19013_001E",
        "B28002_004E",
        "B28002_001E",
    ]
    vars_str = ",".join(variables)

    url_2019 = f"{base_url}/2019/acs/acs5"
    url_2024 = f"{base_url}/2024/acs/acs5"
    params = {"get": vars_str, "for": "county:*", "in": f"state:{state_fip}"}

    resp_2019 = requests.get(url_2019, params=params, timeout=30)
    resp_2024 = requests.get(url_2024, params=params, timeout=30)
    resp_2019.raise_for_status()
    resp_2024.raise_for_status()

    data_2019 = resp_2019.json()
    data_2024 = resp_2024.json()

    df_2019 = pd.DataFrame(data_2019[1:], columns=data_2019[0])
    df_2024 = pd.DataFrame(data_2024[1:], columns=data_2024[0])

    numeric_cols = [
        "B08301_021E",
        "B08301_001E",
        "B19013_001E",
        "B28002_004E",
        "B28002_001E",
    ]
    for col in numeric_cols:
        df_2019[col] = pd.to_numeric(df_2019[col], errors="coerce")
        df_2024[col] = pd.to_numeric(df_2024[col], errors="coerce")

    df_2019["GEOID"] = df_2019["state"] + df_2019["county"]
    df_2024["GEOID"] = df_2024["state"] + df_2024["county"]

    df = pd.merge(
        df_2019[["GEOID", "B08301_021E", "B08301_001E"]],
        df_2024[
            [
                "GEOID",
                "NAME",
                "B08301_021E",
                "B08301_001E",
                "B19013_001E",
                "B28002_004E",
                "B28002_001E",
            ]
        ],
        on="GEOID",
        suffixes=("_2019", "_2024"),
    )

    df = df.rename(
        columns={
            "B08301_021E_2019": "WFH_Workers_2019",
            "B08301_001E_2019": "Total_Workers_2019",
            "B08301_021E_2024": "WFH_Workers_2024",
            "B08301_001E_2024": "Total_Workers_2024",
            "B19013_001E": "Median_Income",
            "B28002_004E": "Broadband_Households",
            "B28002_001E": "Total_Households",
        }
    )

    df["County"] = df["NAME"].str.replace(" County, North Carolina", "", regex=False)
    df["WFH_Pct_2019"] = df["WFH_Workers_2019"] / df["Total_Workers_2019"] * 100
    df["WFH_Pct_2024"] = df["WFH_Workers_2024"] / df["Total_Workers_2024"] * 100
    df["WFH_Change"] = df["WFH_Pct_2024"] - df["WFH_Pct_2019"]
    df["Broadband_Pct"] = df["Broadband_Households"] / df["Total_Households"] * 100
    df["Broadband_Gap"] = 100 - df["Broadband_Pct"]
    df["Potential_Fiber_Households"] = (
        df["Total_Households"] - df["Broadband_Households"]
    ).clip(lower=0)
    df["Potential_Annual_Revenue"] = (
        df["Potential_Fiber_Households"]
        * TARGET_ADOPTION_RATE
        * FIBER_MONTHLY_PRICE
        * 12
    )
    df["Opportunity_Score"] = (
        0.40 * min_max_scale(df["Broadband_Gap"])
        + 0.35 * min_max_scale(df["WFH_Pct_2024"])
        + 0.25 * min_max_scale(df["Median_Income"])
    )

    df = generate_predictive_data(df)

    return df.sort_values("Opportunity_Score", ascending=False).reset_index(drop=True)


def filter_counties(
    df: pd.DataFrame,
    selected_counties: list[str],
    income_range: tuple[int, int],
    gap_range: tuple[float, float],
    wfh_range: tuple[float, float],
) -> pd.DataFrame:
    mask = (
        df["County"].isin(selected_counties)
        & df["Median_Income"].between(income_range[0], income_range[1])
        & df["Broadband_Gap"].between(gap_range[0], gap_range[1])
        & df["WFH_Pct_2024"].between(wfh_range[0], wfh_range[1])
    )
    return df.loc[mask].copy()


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def create_wfh_growth_chart(df: pd.DataFrame, top_n: int = 10) -> Figure:
    top_growth = df.sort_values("WFH_Change", ascending=False).head(top_n).sort_values("WFH_Change")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_growth["County"], top_growth["WFH_Change"], color=COLOR_MAIN)
    ax.set_xlabel("WFH growth (percentage points, 2019 → 2024)")
    ax.set_ylabel("County")
    ax.set_title("Top NC counties by growth in work-from-home")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f} pts"))
    for bar, value in zip(bars, top_growth["WFH_Change"]):
        ax.text(value + 0.05, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center")
    _style_axis(ax)
    fig.tight_layout()
    return fig


def create_wfh_forecast_line_chart(df: pd.DataFrame, top_n: int = 10) -> Figure:
    top_targets = df.sort_values("Opportunity_Score_2025", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in top_targets.iterrows():
        ax.plot(["2019", "2024"], [row["WFH_Pct_2019"], row["WFH_Pct_2024"]], color=COLOR_MAIN, marker="o", linestyle="-")
        ax.plot(["2024", "2025"], 
                [row["WFH_Pct_2024"], row["WFH_Pct_2025"]], 
                color=COLOR_ACCENT, marker="o", linestyle="--")
        ax.text("2025", row["WFH_Pct_2025"], f" {row['County']}", va="center", fontsize=8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Work-from-home households (%)")
    ax.set_title("Predicted work-from-home growth trajectory (Top targets)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    _style_axis(ax)
    fig.tight_layout()
    return fig


def create_broadband_gap_chart(df: pd.DataFrame, top_n: int = 12) -> Figure:
    top_gap = df.sort_values("Broadband_Gap", ascending=False).head(top_n).sort_values("Broadband_Gap")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_gap["County"], top_gap["Broadband_Gap"], color=COLOR_ACCENT)
    ax.set_xlabel("Households without broadband (%)")
    ax.set_ylabel("County")
    ax.set_title("NC counties with the largest broadband gap")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    for bar, value in zip(bars, top_gap["Broadband_Gap"]):
        ax.text(value + 0.15, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center")
    _style_axis(ax)
    fig.tight_layout()
    return fig


def create_revenue_chart(df: pd.DataFrame, top_n: int = 10) -> Figure:
    top_revenue = (
        df.sort_values("Potential_Annual_Revenue", ascending=False)
        .head(top_n)
        .sort_values("Potential_Annual_Revenue")
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_revenue["County"], top_revenue["Potential_Annual_Revenue"], color=COLOR_HIGHLIGHT)
    ax.set_xlabel("Estimated annual fiber revenue")
    ax.set_ylabel("County")
    ax.set_title("Estimated fiber revenue by county")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x / 1_000_000:.1f}M"))
    for bar, value in zip(bars, top_revenue["Potential_Annual_Revenue"]):
        ax.text(value + 25000, bar.get_y() + bar.get_height() / 2, f"${value / 1_000_000:.2f}M", va="center")
    _style_axis(ax)
    fig.tight_layout()
    return fig


def create_income_gap_chart(df: pd.DataFrame, year: int = 2024) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 7))
    if year > 2024 and f"WFH_Pct_{year}" in df.columns:
        wfh_col = f"WFH_Pct_{year}"
        gap_col = f"Broadband_Gap_{year}"
        score_col = f"Opportunity_Score_{year}"
        title_year = f"{year} (Predicted)"
        workers = df["Total_Workers_2024"] * df[wfh_col] / 100
        bubble_sizes = np.clip(workers / 120, 40, 1000)
    else:
        wfh_col = "WFH_Pct_2024"
        gap_col = "Broadband_Gap"
        score_col = "Opportunity_Score"
        title_year = "2024"
        bubble_sizes = np.clip(df["WFH_Workers_2024"] / 120, 40, 1000)

    scatter = ax.scatter(
        df["Median_Income"],
        df[gap_col],
        s=bubble_sizes,
        c=df[score_col],
        cmap="Blues",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.8,
    )

    focus_counties = df.head(min(12, len(df)))
    for _, row in focus_counties.iterrows():
        ax.annotate(
            row["County"],
            (row["Median_Income"], row[gap_col]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Median household income")
    ax.set_ylabel("Broadband gap (%)")
    if year > 2024 and "Flagged" in df.columns:
        flagged = df[df["Flagged"] == True]
        if not flagged.empty:
            ax.scatter(flagged["Median_Income"], flagged[gap_col], marker="*", color=COLOR_HIGHLIGHT, s=150, edgecolor="black", linewidth=1, label="Flagged Targets")
            ax.legend()

    ax.set_title(f"Higher-income counties with unmet broadband demand - {title_year}")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    _style_axis(ax)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Opportunity score")
    fig.tight_layout()
    return fig


def create_target_chart(df: pd.DataFrame, top_n: int = 12) -> Figure:
    top_targets = (
        df[["County", "Opportunity_Score", "WFH_Pct_2024", "Broadband_Gap", "Median_Income"]]
        .head(top_n)
        .sort_values("Opportunity_Score")
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_targets["County"], top_targets["Opportunity_Score"], color=COLOR_MAIN)
    ax.set_xlabel("Composite opportunity score")
    ax.set_ylabel("County")
    ax.set_title("Best NC county targets for fiber-optic expansion")
    for bar, value in zip(bars, top_targets["Opportunity_Score"]):
        ax.text(value + 0.8, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center")
    _style_axis(ax)
    fig.tight_layout()
    return fig


def make_summary_table(df: pd.DataFrame, year: int = 2024) -> pd.DataFrame:
    if year > 2024 and f"WFH_Pct_{year}" in df.columns:
        summary = df[["County", f"Opportunity_Score_{year}", f"WFH_Pct_{year}", f"Broadband_Gap_{year}", "Median_Income", "Potential_Annual_Revenue", "Flagged"]].copy()
        return summary.rename(
            columns={
                f"Opportunity_Score_{year}": f"Score ({year})",
                f"WFH_Pct_{year}": f"WFH % ({year})",
                f"Broadband_Gap_{year}": f"Gap % ({year})",
                "Median_Income": "Median Income",
                "Potential_Annual_Revenue": "Potential Annual Revenue",
                "Flagged": "Flagged"
            }
        )
    else:
        summary = df[["County", "Opportunity_Score", "WFH_Pct_2024", "Broadband_Gap", "Median_Income", "Potential_Annual_Revenue"]].copy()
        return summary.rename(
            columns={
                "Opportunity_Score": "Opportunity Score",
                "WFH_Pct_2024": "WFH % (2024)",
                "Broadband_Gap": "Broadband Gap %",
                "Median_Income": "Median Income",
                "Potential_Annual_Revenue": "Potential Annual Revenue",
            }
        )
