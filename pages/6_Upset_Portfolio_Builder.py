from __future__ import annotations

from collections import Counter

import pandas as pd
import streamlit as st

from mm_app.bracket import ROUND1_PAIRINGS
from mm_app.load import load_bracket_field, load_matchup_probs
from mm_app.paths import get_output_paths
from mm_app.probs import get_matchup_probability
from mm_app.util import slug_to_display_name


st.set_page_config(page_title="Upset Portfolio", layout="wide")
st.title("Upset Portfolio Builder")
st.caption(
    "Selects a *set* of Round of 64 upsets designed to maximize expected upset hit-rate. "
    "Definition: an upset is the higher seed number (e.g., 12) beating the lower seed number (e.g., 5). "
    "Objective: maximize the average P(underdog wins) across the selected upsets."
)

DEFAULT_PUBLIC_YEAR = 2024
st.session_state.setdefault("year", DEFAULT_PUBLIC_YEAR)

year = int(st.session_state.get("year", DEFAULT_PUBLIC_YEAR))
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

st.subheader("Settings")

c1, c2, c3, c4 = st.columns(4)
with c1:
    k = int(st.slider("# of upsets", min_value=1, max_value=16, value=6, step=1))
with c2:
    min_seed_gap = int(st.slider("Min seed gap", min_value=1, max_value=15, value=3, step=1))
with c3:
    min_p = float(st.slider("Min P(underdog wins)", min_value=0.0, max_value=1.0, value=0.35, step=0.01))
with c4:
    max_per_region = int(st.slider("Max per region", min_value=1, max_value=8, value=2, step=1))

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

        p_under = float(get_matchup_probability(under_team, fav_team, probs))
        seed_gap = int(under_seed - fav_seed)

        rows.append(
            {
                "region": region_key,
                "underdog_seed": under_seed,
                "underdog": slug_to_display_name(under_team),
                "favorite_seed": fav_seed,
                "favorite": slug_to_display_name(fav_team),
                "p_underdog_win": p_under,
                "seed_gap": seed_gap,
            }
        )

df_all = pd.DataFrame(rows)

if df_all.empty:
    st.error("No Round of 64 matchups could be constructed from bracket_field.")
    st.stop()

df_pool = df_all[(df_all["seed_gap"] >= min_seed_gap) & (df_all["p_underdog_win"] >= min_p)].copy()
df_pool = df_pool.sort_values(["p_underdog_win", "seed_gap"], ascending=False)

selected: list[dict] = []
region_counts: Counter[str] = Counter()

for _, r in df_pool.iterrows():
    if len(selected) >= k:
        break
    region = str(r["region"])
    if region_counts[region] >= max_per_region:
        continue
    selected.append(r.to_dict())
    region_counts[region] += 1

st.subheader("Portfolio")

if not selected:
    st.warning("No upsets met the current filters. Try lowering min seed gap or min probability.")
else:
    df_sel = pd.DataFrame(selected)

    expected_hit_rate = float(df_sel["p_underdog_win"].mean()) if not df_sel.empty else 0.0
    expected_successes = float(df_sel["p_underdog_win"].sum()) if not df_sel.empty else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Selected upsets", str(len(df_sel)))
    m2.metric("Expected upset hit-rate", f"{expected_hit_rate:.1%}")
    m3.metric("Expected successful upsets", f"{expected_successes:.2f}")

    df_sel["p_underdog_win"] = df_sel["p_underdog_win"].map(lambda x: round(float(x), 3))
    df_sel = df_sel[[
        "region",
        "underdog_seed",
        "underdog",
        "favorite_seed",
        "favorite",
        "p_underdog_win",
        "seed_gap",
    ]]

    st.dataframe(df_sel, use_container_width=True, hide_index=True)

with st.expander("Candidate pool", expanded=False):
    st.caption(
        "This is the filtered pool the builder selects from (sorted by P(underdog wins)). "
        "The portfolio is formed by taking the top candidates while respecting the per-region cap."
    )
    df_show = df_pool.copy()
    if not df_show.empty:
        df_show["p_underdog_win"] = df_show["p_underdog_win"].map(lambda x: round(float(x), 3))
        st.dataframe(df_show, use_container_width=True, hide_index=True)
    else:
        st.info("No candidates meet the current filters.")
