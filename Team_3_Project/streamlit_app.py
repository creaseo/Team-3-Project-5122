from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from team_3_project.fiber_analysis import (  # noqa: E402
    FIBER_MONTHLY_PRICE,
    TARGET_ADOPTION_RATE,
    create_broadband_gap_chart,
    create_income_gap_chart,
    create_revenue_chart,
    create_target_chart,
    create_wfh_growth_chart,
    fetch_county_data,
    filter_counties,
    make_summary_table,
)
from team_3_project.ai_insights import (  # noqa: E402
    generate_recommendations,
    predict_wfh_growth,
)

st.set_page_config(
    page_title="NC Fiber Opportunity Explorer",
    page_icon="📶",
    layout="wide",
)


@st.cache_data(show_spinner="Loading North Carolina county data...")
def load_data():
    return fetch_county_data("37")


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def render_chart_section(title: str, figure, explanation: str) -> None:
    st.subheader(title)
    st.pyplot(figure, use_container_width=True)
    st.markdown(explanation)


def main() -> None:
    st.title("North Carolina Fiber Opportunity Explorer")
    st.caption(
        "Interactive view of county-level broadband need, work-from-home demand, income, and fiber revenue potential."
    )

    df = load_data()

    st.sidebar.header("Filters")
    top_n = st.sidebar.slider("Top counties to show", min_value=5, max_value=25, value=12, step=1)

    county_options = sorted(df["County"].tolist())
    selected_counties = st.sidebar.multiselect(
        "Counties",
        options=county_options,
        default=county_options,
    )

    income_min = int(df["Median_Income"].min())
    income_max = int(df["Median_Income"].max())
    income_range = st.sidebar.slider(
        "Median household income range",
        min_value=income_min,
        max_value=income_max,
        value=(income_min, income_max),
        step=1000,
        format="$%d",
    )

    gap_min = float(df["Broadband_Gap"].min())
    gap_max = float(df["Broadband_Gap"].max())
    gap_range = st.sidebar.slider(
        "Broadband gap range (%)",
        min_value=round(gap_min, 1),
        max_value=round(gap_max, 1),
        value=(round(gap_min, 1), round(gap_max, 1)),
        step=0.5,
    )

    wfh_min = float(df["WFH_Pct_2024"].min())
    wfh_max = float(df["WFH_Pct_2024"].max())
    wfh_range = st.sidebar.slider(
        "2024 work-from-home rate range (%)",
        min_value=round(wfh_min, 1),
        max_value=round(wfh_max, 1),
        value=(round(wfh_min, 1), round(wfh_max, 1)),
        step=0.5,
    )

    filtered_df = filter_counties(
        df,
        selected_counties=selected_counties,
        income_range=income_range,
        gap_range=gap_range,
        wfh_range=wfh_range,
    )

    if filtered_df.empty:
        st.warning("No counties match the current filters. Try widening one or more filter ranges.")
        return

    filtered_df = filtered_df.sort_values("Opportunity_Score", ascending=False).reset_index(drop=True)
    lead_county = filtered_df.iloc[0]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Matching counties", f"{len(filtered_df)}")
    metric_cols[1].metric("Top target", lead_county["County"])
    metric_cols[2].metric("Top opportunity score", f"{lead_county['Opportunity_Score']:.1f}")
    metric_cols[3].metric("Top annual revenue", format_currency(lead_county["Potential_Annual_Revenue"]))

    st.markdown("---")

    chart_tabs = st.tabs(
        [
            "WFH growth",
            "Broadband gap",
            "Revenue",
            "Income vs gap",
            "Best targets",
            "Table",
            "ML Forecast",
            "AI Recommendations",
        ]
    )

    with chart_tabs[0]:
        render_chart_section(
            "Work-from-home growth",
            create_wfh_growth_chart(filtered_df, top_n=min(top_n, len(filtered_df))),
            """
### What this chart means
This chart shows which counties had the biggest increase in working from home between 2019 and 2024.

- A longer bar means the county had a larger increase in remote work.
- Counties near the top may be stronger fiber targets because work-from-home households usually care more about fast and reliable internet.
- This chart only shows remote-work growth. It does not include income or broadband access gaps yet.
            """,
        )

    with chart_tabs[1]:
        render_chart_section(
            "Broadband gap",
            create_broadband_gap_chart(filtered_df, top_n=min(top_n, len(filtered_df))),
            """
### What this chart means
This chart shows the counties with the biggest broadband gap.

- The broadband gap is the share of households that do not currently have broadband.
- A larger percentage means more households may still need better internet access.
- These counties may have strong need, but they are not automatically the best business targets on their own. Income, work-from-home demand, and revenue still matter.
            """,
        )

    with chart_tabs[2]:
        render_chart_section(
            "Estimated fiber revenue",
            create_revenue_chart(filtered_df, top_n=min(top_n, len(filtered_df))),
            f"""
### What this chart means
This chart estimates how much yearly fiber revenue each county could produce.

- A longer bar means a larger estimated revenue opportunity.
- This is an estimate, not a guarantee.

$$
\\text{{Potential Annual Revenue}} = \\text{{Households Without Broadband}} \times {TARGET_ADOPTION_RATE:.2f} \times \\${FIBER_MONTHLY_PRICE} \times 12
$$

In simple terms:
- start with households that do not already have broadband,
- assume {TARGET_ADOPTION_RATE:.0%} of them become fiber customers,
- assume each customer pays ${FIBER_MONTHLY_PRICE} per month,
- multiply by 12 months for yearly revenue.
            """,
        )

    with chart_tabs[3]:
        st.subheader("Income vs. broadband gap")
        st.plotly_chart(
            create_income_gap_chart(filtered_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown("""
### What this chart means
This chart compares three things at the same time:

- horizontal position shows median household income,
- vertical position shows the broadband gap,
- bubble size reflects the number of work-from-home workers in 2024.

Simple way to read it:
- farther right = higher income,
- higher up = more unmet broadband need,
- bigger bubble = more work-from-home households.

**Hover over any bubble** to see the county name, income, broadband gap, WFH rate, and opportunity score.

The color shows the `Opportunity_Score` — darker (yellow) means a stronger opportunity overall.
        """)

    with chart_tabs[4]:
        render_chart_section(
            "Best county targets",
            create_target_chart(filtered_df, top_n=min(top_n, len(filtered_df))),
            """
### What this chart means
This is the final summary ranking of the best county targets for fiber-optic expansion.

- A higher `Opportunity_Score` means the county looks better overall.
- The score combines broadband need, remote-work demand, and income strength.

$$
\\text{Opportunity Score} = 0.40 \\times \\text{Broadband Gap Score} + 0.35 \\times \\text{WFH Score} + 0.25 \\times \\text{Income Score}
$$

In simple terms:
- 40% of the score comes from the broadband gap,
- 35% comes from the 2024 work-from-home rate,
- 25% comes from median household income.

Each part was first scaled to a 0 to 100 range so counties could be compared fairly before combining them into one final score.
            """,
        )

    with chart_tabs[5]:
        st.subheader("Filtered county table")
        st.dataframe(
            make_summary_table(filtered_df.head(top_n)).style.format(
                {
                    "Opportunity_Score": "{:.1f}",
                    "WFH % (2024)": "{:.1f}%",
                    "Broadband Gap %": "{:.1f}%",
                    "Median Income": "${:,.0f}",
                    "Potential Annual Revenue": "${:,.0f}",
                }
            ),
            use_container_width=True,
        )

    # ------------------------------------------------------------------
    # ML Forecast tab
    # ------------------------------------------------------------------
    with chart_tabs[6]:
        st.subheader("ML Work-From-Home Forecast")
        st.markdown(
            """
            A **Linear Regression** model trained on each county's broadband gap and median income
            predicts the expected work-from-home growth trajectory through **2026 and 2027**.
            The model uses the 2019→2024 WFH change as the target variable.
            """
        )

        with st.spinner("Running ML model…"):
            predicted_df = predict_wfh_growth(filtered_df)

        display_n = min(top_n, len(predicted_df))
        forecast_df = (
            predicted_df.sort_values("WFH_Pct_2027", ascending=False)
            .head(display_n)
            .reset_index(drop=True)
        )

        # Show summary table
        forecast_table = forecast_df[[
            "County",
            "WFH_Pct_2024",
            "WFH_Pct_2026",
            "WFH_Pct_2027",
            "Predicted_WFH_Change",
            "Trend_Label",
        ]].rename(columns={
            "WFH_Pct_2024": "WFH % (2024 actual)",
            "WFH_Pct_2026": "WFH % (2026 forecast)",
            "WFH_Pct_2027": "WFH % (2027 forecast)",
            "Predicted_WFH_Change": "Predicted 5-yr change (pts)",
            "Trend_Label": "Trend",
        })
        st.dataframe(
            forecast_table.style.format({
                "WFH % (2024 actual)": "{:.1f}%",
                "WFH % (2026 forecast)": "{:.1f}%",
                "WFH % (2027 forecast)": "{:.1f}%",
                "Predicted 5-yr change (pts)": "{:+.1f}",
            }),
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("### Forecast comparison chart")

        # Bar chart comparing 2024 actual vs 2026/2027 forecasts
        fig, ax = plt.subplots(figsize=(10, max(4, display_n * 0.55)))
        y = range(display_n)
        width = 0.28
        counties = forecast_df["County"].tolist()

        bars1 = ax.barh(
            [i - width for i in y],
            forecast_df["WFH_Pct_2024"],
            width,
            label="2024 (actual)",
            color="#4C78A8",
            alpha=0.9,
        )
        bars2 = ax.barh(
            [i for i in y],
            forecast_df["WFH_Pct_2026"],
            width,
            label="2026 (forecast)",
            color="#F58518",
            alpha=0.9,
        )
        bars3 = ax.barh(
            [i + width for i in y],
            forecast_df["WFH_Pct_2027"],
            width,
            label="2027 (forecast)",
            color="#54A24B",
            alpha=0.9,
        )

        ax.set_yticks(list(y))
        ax.set_yticklabels(counties, fontsize=9)
        ax.set_xlabel("Work-from-home rate (%)")
        ax.set_title("WFH Rate: 2024 Actual vs 2026/2027 ML Forecast")
        ax.legend(loc="lower right", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.markdown("""
### How the model works
The model learns from the 5-year WFH trend (2019→2024) and uses each county's
**broadband gap** and **median income** as predictors.
- Counties with higher broadband gaps tend to have more room for remote-work growth.
- Higher income counties tend to have more knowledge workers with WFH-compatible jobs.
- Projections extend the learned annual rate 2–3 years forward.

> **Note:** These are statistical projections, not guarantees. Treat 2026/2027 values
> as planning estimates, not exact forecasts.
        """)

    # ------------------------------------------------------------------
    # AI Recommendations tab
    # ------------------------------------------------------------------
    with chart_tabs[7]:
        st.subheader("AI Investment Recommendations")
        st.markdown(
            """
            The AI engine translates each county's scoring-model metrics into
            **plain-language, prioritised recommendations**. Expand any county card
            to see the full analysis.
            """
        )

        with st.spinner("Generating recommendations…"):
            recs = generate_recommendations(filtered_df.head(top_n))

        if not recs:
            st.info("No recommendations available for the current filter set.")
        else:
            # Summary metrics row
            top_priority_count = sum(1 for r in recs if r["priority"] == "🔴 Top Priority")
            mid_priority_count = sum(1 for r in recs if r["priority"] == "🟡 Medium Priority")
            low_priority_count = sum(1 for r in recs if r["priority"] == "🟢 Lower Priority")

            col1, col2, col3 = st.columns(3)
            col1.metric("🔴 Top Priority counties", top_priority_count)
            col2.metric("🟡 Medium Priority counties", mid_priority_count)
            col3.metric("🟢 Lower Priority counties", low_priority_count)

            st.markdown("---")

            for rec in recs:
                priority_color = {
                    "🔴 Top Priority": "#ff4b4b",
                    "🟡 Medium Priority": "#ffa500",
                    "🟢 Lower Priority": "#21c354",
                }.get(rec["priority"], "#cccccc")

                label = f"{rec['priority']} — **{rec['county']}**  (Score: {rec['opportunity_score']:.1f})"
                with st.expander(label, expanded=(rec == recs[0])):
                    st.markdown(f"*{rec['headline']}*")
                    st.markdown("**Recommendations:**")
                    for bullet in rec["bullets"]:
                        st.markdown(f"- {bullet}")

                    col_s, col_w = st.columns(2)
                    with col_s:
                        if rec["strengths"]:
                            st.markdown("**✅ Strengths**")
                            for s in rec["strengths"]:
                                st.markdown(f"- {s}")
                    with col_w:
                        if rec["watch_outs"]:
                            st.markdown("**⚠️ Watch outs**")
                            for w in rec["watch_outs"]:
                                st.markdown(f"- {w}")


if __name__ == "__main__":
    main()
