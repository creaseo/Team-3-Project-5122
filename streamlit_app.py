from __future__ import annotations

import sys
from pathlib import Path

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
    create_wfh_forecast_line_chart,
    create_wfh_growth_chart,
    fetch_county_data,
    filter_counties,
    make_summary_table,
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
        ]
    )

    with chart_tabs[0]:
        predictive_mode = st.toggle("Toggle predictions")
        if predictive_mode:
            forecast_year = 2025
        else:
            forecast_year = 2024

        if predictive_mode:
            render_chart_section(
                "Predicted Work-from-home growth (2025)",
                create_wfh_forecast_line_chart(filtered_df, top_n=min(top_n, len(filtered_df))),
                """
### What this chart means
This chart projects WFH growth trajectory from historical points (2019-2024) to a simulated AI prediction in 2025.

- The dashed line represents future predictions.
- Shows how top targets are predicted to evolve based on AI models.
                """,
            )
        else:
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
        render_chart_section(
            "Income vs. broadband gap",
            create_income_gap_chart(filtered_df, year=forecast_year),
            """
### What this chart means
This chart compares three things at the same time:

- horizontal position shows median household income,
- vertical position shows the broadband gap,
- bubble size reflects the number of work-from-home workers in 2024.

Simple way to read it:
- farther right = higher income,
- higher up = more unmet broadband need,
- bigger bubble = more work-from-home households.

The color shows the `Opportunity_Score`. A darker bubble means a county looks stronger overall based on the combined scoring method used in the final ranking.
            """,
        )

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
        if forecast_year > 2024:
            st.dataframe(
                make_summary_table(filtered_df.head(top_n), year=forecast_year).style.format(
                    {
                        f"Score ({forecast_year})": "{:.1f}",
                        f"WFH % ({forecast_year})": "{:.1f}%",
                        f"Gap % ({forecast_year})": "{:.1f}%",
                        "Median Income": "${:,.0f}",
                        "Potential Annual Revenue": "${:,.0f}",
                        "Flagged": "{}"
                    }
                ),
                use_container_width=True,
            )
        else:
            st.dataframe(
                make_summary_table(filtered_df.head(top_n), year=2024).style.format(
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


if __name__ == "__main__":
    main()
