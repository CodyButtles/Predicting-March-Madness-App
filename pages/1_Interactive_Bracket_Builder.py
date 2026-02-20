from __future__ import annotations

import streamlit as st

from mm_app.component import render_bracket_builder
from mm_app.load import load_advancement_probs, load_bracket_field, load_matchup_probs
from mm_app.paths import get_output_paths


st.set_page_config(page_title="Bracket Builder", layout="wide")
st.title("Interactive Bracket Builder")
st.caption("Click a team to advance; hover for round-by-round odds")

DEFAULT_PUBLIC_YEAR = 2024
st.session_state.setdefault("year", DEFAULT_PUBLIC_YEAR)

with st.expander("What does “conf %” mean?", expanded=False):
        st.markdown(
                r"""
The **conf %** shown next to a matchup is a simple proxy for how far the predicted win probability is from a coin-flip.

- Let $p$ be the model’s win probability for a team in that matchup.
- We compute $\text{confidence} = 2 \cdot |p - 0.5|$, which ranges from 0 to 1.
    - `0%` means $p=0.50$ (pure coin flip)
    - `100%` means $p=1.00$ or $p=0.00$

This is **not** a statistical uncertainty estimate; it’s just “distance from 50/50.”
                """
        )

year = int(st.session_state.get("year", DEFAULT_PUBLIC_YEAR))
paths = get_output_paths(year)

try:
    bracket_field = load_bracket_field(year)
    adv = load_advancement_probs(year)
except FileNotFoundError:
    st.error(
        "Missing required JSON inputs for this page. Generate them via the 06 notebook "
        "or configure private-data fetching for public deployments. "
        f"Expected {paths.bracket_field_json.name} and {paths.advancement_probs_json.name}."
    )
    st.stop()

# matchup probs are optional (page still works but can't show per-match win probs)
try:
    matchup_probs = load_matchup_probs(year)
except FileNotFoundError:
    matchup_probs = {}

render_bracket_builder(
    bracket_field=bracket_field,
    advancement_probs=adv,
    matchup_probs=matchup_probs,
    year=year,
)
