from __future__ import annotations

import streamlit as st

try:
    from mm_app.private_data import list_available_output_years
except ImportError:
    # Defensive fallback: if Streamlit Cloud is temporarily running an older
    # revision of mm_app/private_data.py, keep the app bootable.
    def list_available_output_years(*, root=None):  # type: ignore[no-redef]
        return []


st.set_page_config(page_title="Predicting March Madness Overview", layout="wide")
st.title("Predicting March Madness Overview")
st.caption("Methodology overview and page guide")

DEFAULT_PUBLIC_YEAR = 2025

from mm_app.year_ui import render_year_sidebar

year = render_year_sidebar(default_year=DEFAULT_PUBLIC_YEAR)

st.subheader("Modeling methodology (high level)")
st.markdown(
        """
This project’s core workflow is:

1) **Build matchup-level features**
- The training table is a game-level / matchup-level dataset of historical tournament games.
- Features are primarily engineered “difference” features (Team A minus Team B style deltas), plus contextual inputs like:
    - each team’s seed
    - the tournament round
    - head-to-head and shared-opponent summary features (when available)

2) **Normalize within season**
- Many numeric matchup features are z-scored **within each season** to reduce year-to-year scale drift.
- Seeds are kept in their original scale (they are already comparable year-to-year).

3) **Train + tune models**
- Primary model: an **XGBoost classifier** trained to predict whether Team A wins.
- Hyperparameters are tuned using randomized search with cross-validation; selection is typically based on ROC AUC.
- Some years are held out as a more realistic “future tournament” test set (year-based holdout).

4) **Inference for a tournament year**
- For a given tournament year, we generate a matrix of team-vs-team matchups (all combos) and run inference.
- We produce a compact pairwise probability map for the tournament field.

5) **Monte Carlo tournament simulation**
- Using the matchup probabilities, we simulate the full bracket many times.
- This produces **advancement odds by round** per team.

6) **(Optional) Explanations / drivers**
- For matchup “why”/drivers, we can compute SHAP-style per-feature contributions from the XGBoost model.

The Streamlit pages are intentionally thin: they mostly visualize these precomputed results.
        """
)

st.subheader("Page guide (what each page is for)")
st.markdown(
        """
Use the left sidebar to open:

- **Interactive Bracket Builder**: Fill out a bracket round-by-round. Picks auto-advance and the UI shows per-game win %
    (when matchup probs exist) plus advancement odds by round (from Monte Carlo simulation).
- **Game-by-Game Probability Explorer**: Pick any two teams and view P(A beats B), plus a simple confidence proxy.
    If explanation data is available, it also shows the strongest positive/negative feature drivers.
- **Upset Finder**: Ranks Round-of-64 underdogs by a combined “severity” score (seed gap weighted by underdog win prob)
    to quickly scan the upset landscape.
- **Upset Portfolio Builder**: Builds a *set* of Round-of-64 upsets designed to maximize expected hit-rate, with simple
    constraints (min seed gap / min probability / max per region).
- **Simulation-Based Bracket Optimizer**: Generates many candidate brackets and ranks them by expected ESPN-style score
    against simulated tournament outcomes; includes controls for chalk vs contrarian ("leverage") and champion diversity.
- **Team Deep-Dive**: For a selected team, shows advancement odds by round and highlights statistical outliers vs the
    tournament field using raw per-team stats (if present).
- **Historical Tournament Trends**: Uses historical tournament results to summarize seed-vs-seed win rates, upset rates,
    champions, and Round-of-64 win% by seed.
        """
)
