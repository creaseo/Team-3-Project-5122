# NC Fiber Opportunity Explorer

## Description

An interactive Streamlit dashboard that analyzes North Carolina county-level data to identify the best targets for fiber-optic broadband expansion. It pulls live data from the U.S. Census Bureau ACS API and combines broadband gap, work-from-home demand, and median income into an opportunity scoring model. The app also includes an ML-based WFH growth forecast and AI-generated investment recommendations.

## Deployed App

https://team-3-project-5122-aplnsqx23fcxjqtrn8nmxb.streamlit.app/

## Requirements

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

If you do not have uv installed, you can install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Local Setup

1. Clone the repository and move into the project folder:

```bash
git clone https://github.com/creaseo/Team-3-Project-5122
cd Team-3-Project-5122/Team_3_Project
```

2. Install all dependencies:

```bash
uv sync
```

3. Run the app:

```bash
uv run streamlit run streamlit_app.py
```

4. Open your browser and go to `http://localhost:8501`.

The app fetches county data from the Census Bureau API on first load. This may take a few seconds. Results are cached so subsequent loads are fast.

## Project Structure

```
Team_3_Project/
    streamlit_app.py          # Main Streamlit app
    src/
        team_3_project/
            fiber_analysis.py # Data fetching, scoring model, and charts
            ai_insights.py    # ML forecast and AI recommendation engine
    pyproject.toml            # Project metadata and dependencies
```
