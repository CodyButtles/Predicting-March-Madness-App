from __future__ import annotations

import streamlit as st

from mm_app.paths import get_output_paths


st.set_page_config(page_title="Predicting March Madness Overview", layout="wide")
st.title("Predicting March Madness Overview")
st.caption("Streamlit app backed by precomputed model outputs")

DEFAULT_PUBLIC_YEAR = 2024

# Public deployment is pinned to a single year to ensure all pages work
# (no missing Output artifacts for other years).
year = st.sidebar.number_input(
    "Year",
    min_value=DEFAULT_PUBLIC_YEAR,
    max_value=DEFAULT_PUBLIC_YEAR,
    value=DEFAULT_PUBLIC_YEAR,
    step=1,
)
st.session_state["year"] = DEFAULT_PUBLIC_YEAR

paths = get_output_paths(DEFAULT_PUBLIC_YEAR)

st.subheader("Required Outputs")
st.write(
    "This app is **offline-first**: pages read precomputed artifacts written to `Output/` (typically generated via "
    "the 06 notebook and/or the scripts in `Scripts/`)."
)

rows = [
    ("Bracket field", paths.bracket_field_json),
    ("Advancement odds", paths.advancement_probs_json),
    ("Matchup probs", paths.matchup_probs_json),
]

for label, path in rows:
    st.write(f"- {label}: `{path.as_posix()}`  (exists={path.exists()})")

st.subheader("Modeling methodology (high level)")
st.markdown(
        """
This project’s core workflow is:

1) **Build matchup-level features**
- The training table is a game-level / matchup-level dataset (see `Output/2000-2025_Spreads.csv`).
- Features are primarily the engineered `*_spread` columns (team1 minus team2 style deltas), plus contextual inputs like:
    - `team1_seed`, `team2_seed`
    - `round_ind`
    - head-to-head and shared-opponent summary features (when available)

2) **Normalize within season**
- Many numeric matchup features are z-scored **within each `tournament_year`** to reduce year-to-year scale drift.
- Seeds are kept in their original scale (they are already comparable year-to-year).

3) **Train + tune models**
- Primary model: an **XGBoost classifier** trained to predict `team1_win_loss_ind`.
- Hyperparameters are tuned using randomized search with cross-validation; selection is typically based on ROC AUC.
- Some years are held out as a more realistic “future tournament” test set (year-based holdout).

4) **Inference for a tournament year**
- For a given tournament year, we generate a matrix of team-vs-team matchups (all combos) and run inference.
- We write a compact pairwise probability map `matchup_probabilities_YYYY.json` for the 64 tournament teams.

5) **Monte Carlo tournament simulation**
- Using `matchup_probabilities_YYYY.json`, we simulate the full bracket many times.
- This produces **advancement odds by round** per team (`advancement_probs_YYYY.json`).

6) **(Optional) Explanations / drivers**
- For matchup “why”/drivers, we can compute SHAP-style per-feature contributions from the XGBoost model and export
    `matchup_explanations_YYYY.json` (see `Scripts/generate_matchup_explanations.py`).

The Streamlit pages are intentionally thin: they mostly visualize these precomputed artifacts.
        """
)

st.subheader("Page guide (what each page is for)")
st.markdown(
        """
Use the left sidebar to open:

- **Interactive Bracket Builder**: Fill out a bracket round-by-round. Picks auto-advance and the UI shows per-game win %
    (when matchup probs exist) plus advancement odds by round (from Monte Carlo simulation).
- **Game-by-Game Probability Explorer**: Pick any two teams and view P(A beats B), plus a simple confidence proxy.
    If `matchup_explanations_YYYY.json` exists, it also shows the strongest positive/negative feature drivers.
- **Upset Finder**: Ranks Round-of-64 underdogs by a combined “severity” score (seed gap weighted by underdog win prob)
    to quickly scan the upset landscape.
- **Upset Portfolio Builder**: Builds a *set* of Round-of-64 upsets designed to maximize expected hit-rate, with simple
    constraints (min seed gap / min probability / max per region).
- **Simulation-Based Bracket Optimizer**: Generates many candidate brackets and ranks them by expected ESPN-style score
    against simulated tournament outcomes; includes controls for chalk vs contrarian ("leverage") and champion diversity.
- **Team Deep-Dive**: For a selected team, shows advancement odds by round and highlights statistical outliers vs the
    tournament field using `Data/Raw_Team_Data.csv` (if present).
- **Historical Tournament Trends**: Uses historical tournament results (from `Output/2000-2025_Spreads.csv`) to summarize
    seed-vs-seed win rates, upset rates, and Round-of-64 win% by seed.
        """
)

st.subheader("Where the artifacts come from")
st.markdown(
        """
- `Scripts/05_Model_Development.ipynb`: model training + tuning (XGBoost and experiments)
- `Scripts/06_Streamlit_Interactive_App.ipynb`: produces the JSON artifacts consumed by the app
- `Scripts/generate_optimizer_sims.py` and `Scripts/generate_optimizer_top_brackets.py`: optional precomputes for the optimizer
        """
)
