from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    import altair as alt
except Exception:  # pragma: no cover
    alt = None  # type: ignore[assignment]

from mm_app.bracket import ROUND1_PAIRINGS
from mm_app.load import load_bracket_field, load_matchup_probs
from mm_app.paths import get_output_paths
from mm_app.probs import get_matchup_probability
from mm_app.util import slug_to_display_name


st.set_page_config(page_title="Upset Finder", layout="wide")
st.title("Upset Finder")
st.caption(
    "Ranks Round of 64 underdogs by win probability and seed gap. "
    "Definition: `upset_score = seed_gap * P(underdog wins)` (higher = larger seed-gap-weighted upset severity)."
)

DEFAULT_PUBLIC_YEAR = 2024
st.session_state["year"] = DEFAULT_PUBLIC_YEAR

st.session_state.setdefault("year", 2024)
year = int(st.session_state.get("year", 2024))
paths = get_output_paths(year)

try:
    bracket_field = load_bracket_field(year)
    probs = load_matchup_probs(year)
except FileNotFoundError:
    st.error(
        "Missing required JSON inputs. Generate them via the 06 notebook "
        "or configure private-data fetching for public deployments. "
        f"Expected {paths.bracket_field_json.name} and {paths.matchup_probs_json.name}."
    )
    st.stop()

rows: list[dict] = []
for region_key, seeds in bracket_field.items():
    seed_to_team = {int(x["seed"]): x["team"] for x in seeds}
    for seed_a, seed_b in ROUND1_PAIRINGS:
        team_a = seed_to_team[seed_a]
        team_b = seed_to_team[seed_b]

        # underdog is higher seed number
        if seed_a > seed_b:
            under_seed, under_team = seed_a, team_a
            fav_seed, fav_team = seed_b, team_b
        else:
            under_seed, under_team = seed_b, team_b
            fav_seed, fav_team = seed_a, team_a

        p_under = get_matchup_probability(under_team, fav_team, probs)
        seed_gap = under_seed - fav_seed
        upset_score = float(p_under) * float(seed_gap)

        rows.append(
            {
                "region": region_key,
                "underdog_seed": under_seed,
                "underdog": slug_to_display_name(under_team),
                "favorite_seed": fav_seed,
                "favorite": slug_to_display_name(fav_team),
                "p_underdog_win": round(float(p_under), 3),
                "seed_gap": seed_gap,
                "upset_score": round(upset_score, 3),
            }
        )

df = pd.DataFrame(rows).sort_values(["upset_score", "p_underdog_win"], ascending=False)

df_plot = df.copy()
df_plot["matchup"] = (
    df_plot["underdog"]
    + " ("
    + df_plot["underdog_seed"].astype(str)
    + ") vs "
    + df_plot["favorite"]
    + " ("
    + df_plot["favorite_seed"].astype(str)
    + ")"
)

if alt is not None and not df_plot.empty:
    st.subheader("Round of 64 Upset Landscape")
    scatter = (
        alt.Chart(df_plot)
        .mark_circle(opacity=0.85)
        .encode(
            x=alt.X("p_underdog_win:Q", title="P(underdog wins)"),
            y=alt.Y("seed_gap:Q", title="seed_gap (underdog_seed - favorite_seed)"),
            size=alt.Size("upset_score:Q", legend=None),
            color=alt.Color("upset_score:Q", title="upset_score"),
            tooltip=[
                alt.Tooltip("region:N"),
                alt.Tooltip("matchup:N"),
                alt.Tooltip("p_underdog_win:Q", format=".3f"),
                alt.Tooltip("seed_gap:Q"),
                alt.Tooltip("upset_score:Q", format=".3f"),
            ],
        )
        .properties(height=220)
    )
    st.altair_chart(scatter, use_container_width=True)

    st.subheader("Top Upsets by Upset Score")
    topn = df_plot.head(15)
    bars = (
        alt.Chart(topn)
        .mark_bar()
        .encode(
            x=alt.X("upset_score:Q", title="upset_score = seed_gap * P(underdog wins)"),
            y=alt.Y("matchup:N", sort="-x", title=None),
            color=alt.Color("region:N", title="region"),
            tooltip=[
                alt.Tooltip("region:N"),
                alt.Tooltip("matchup:N"),
                alt.Tooltip("p_underdog_win:Q", format=".3f"),
                alt.Tooltip("seed_gap:Q"),
                alt.Tooltip("upset_score:Q", format=".3f"),
            ],
        )
        .properties(height=min(420, 24 * len(topn)))
    )
    st.altair_chart(bars, use_container_width=True)

st.dataframe(df, use_container_width=True, hide_index=True)
