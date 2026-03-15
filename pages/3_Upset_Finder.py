from __future__ import annotations

import io
from pathlib import Path

import numpy as np
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
from mm_app.year_ui import render_year_sidebar


st.set_page_config(page_title="Upset Finder", layout="wide")
st.title("Upset Finder")
st.caption(
    "Compares model predictions against historical seed-matchup win rates. "
    "**Model edge** = model probability minus the historical win rate for that seed pairing — "
    "positive means the model sees *more* upset potential than history alone would suggest."
)

DEFAULT_PUBLIC_YEAR = 2025
year = render_year_sidebar(default_year=DEFAULT_PUBLIC_YEAR)
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


@st.cache_data(show_spinner=False)
def _load_r64_seed_win_rates() -> dict[tuple[int, int], float]:
    """Return {(underdog_seed, fav_seed): historical_win_pct} from the agg CSV (R64 only)."""
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "Output" / "2024" / "historical_seed_results_agg.csv",
        root / "Output" / "historical_seed_results_agg.csv",
    ]
    for p in candidates:
        try:
            if p.exists():
                df = pd.read_csv(p)
            else:
                from mm_app.private_data import read_bytes_maybe_private
                df = pd.read_csv(io.BytesIO(read_bytes_maybe_private(p)))

            needed = {"round", "seed", "opp_seed", "games", "wins"}
            if not needed.issubset(set(df.columns)):
                continue

            r64 = df[df["round"] == 1].copy()
            for c in ["seed", "opp_seed", "games", "wins"]:
                r64[c] = pd.to_numeric(r64[c], errors="coerce")
            r64 = r64.dropna(subset=["seed", "opp_seed", "games", "wins"])
            agg = (
                r64.groupby(["seed", "opp_seed"])[["games", "wins"]]
                .sum()
                .reset_index()
            )
            agg["win_pct"] = agg["wins"] / agg["games"].clip(lower=1)
            return {
                (int(row["seed"]), int(row["opp_seed"])): float(row["win_pct"])
                for _, row in agg.iterrows()
            }
        except Exception:
            continue
    return {}


hist_rates = _load_r64_seed_win_rates()
has_hist = bool(hist_rates)

rows: list[dict] = []
for region_key, seeds in bracket_field.items():
    seed_to_team = {int(x["seed"]): x["team"] for x in seeds}
    for seed_a, seed_b in ROUND1_PAIRINGS:
        team_a = seed_to_team[seed_a]
        team_b = seed_to_team[seed_b]

        # underdog is the higher seed number
        if seed_a > seed_b:
            under_seed, under_team = seed_a, team_a
            fav_seed, fav_team = seed_b, team_b
        else:
            under_seed, under_team = seed_b, team_b
            fav_seed, fav_team = seed_a, team_a

        p_under = float(get_matchup_probability(under_team, fav_team, probs))
        seed_gap = under_seed - fav_seed
        hist_win_pct = hist_rates.get((under_seed, fav_seed), float("nan"))
        model_edge = (p_under - hist_win_pct) if not np.isnan(hist_win_pct) else float("nan")

        rows.append(
            {
                "region": region_key,
                "underdog_seed": under_seed,
                "underdog": slug_to_display_name(under_team),
                "favorite_seed": fav_seed,
                "favorite": slug_to_display_name(fav_team),
                "p_underdog_win": round(p_under, 3),
                "hist_win_pct": round(hist_win_pct, 3) if not np.isnan(hist_win_pct) else None,
                "model_edge": round(model_edge, 3) if not np.isnan(model_edge) else None,
                "seed_gap": seed_gap,
            }
        )

# ── Sort selector ─────────────────────────────────────────────────────────────
SORT_OPTIONS = {
    "Model edge (model − history)": "model_edge",
    "Model probability": "p_underdog_win",
    "Historical win %": "hist_win_pct",
}
if not has_hist:
    SORT_OPTIONS = {"Model probability": "p_underdog_win"}

sort_label = st.selectbox("Sort upsets by", list(SORT_OPTIONS.keys()), index=0)
sort_col = SORT_OPTIONS[sort_label]

df = pd.DataFrame(rows).sort_values(sort_col, ascending=False, na_position="last")

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
    common_tooltip = [
        alt.Tooltip("region:N"),
        alt.Tooltip("matchup:N"),
        alt.Tooltip("p_underdog_win:Q", title="Model P(upset)", format=".3f"),
        alt.Tooltip("hist_win_pct:Q", title="Hist. win %", format=".3f"),
        alt.Tooltip("model_edge:Q", title="Model edge", format="+.3f"),
        alt.Tooltip("seed_gap:Q"),
    ]

    if has_hist:
        st.subheader("Model vs Historical Upset Rate")
        st.caption(
            "Each dot is a R64 matchup. "
            "**Above the line** → model is more bullish on the upset than history suggests. "
            "**Below** → model is more cautious. Size = seed gap."
        )

        # Diagonal reference line spanning the data range
        axis_max = float(df_plot[["p_underdog_win", "hist_win_pct"]].max().max()) + 0.05
        diag = pd.DataFrame({"x": [0, axis_max], "y": [0, axis_max]})
        ref_line = (
            alt.Chart(diag)
            .mark_line(color="gray", strokeDash=[4, 4], opacity=0.6)
            .encode(x="x:Q", y="y:Q")
        )

        scatter = (
            alt.Chart(df_plot.dropna(subset=["hist_win_pct", "model_edge"]))
            .mark_circle(opacity=0.85)
            .encode(
                x=alt.X("hist_win_pct:Q", title="Historical upset win % (seed pair, all years)", scale=alt.Scale(domain=[0, axis_max])),
                y=alt.Y("p_underdog_win:Q", title="Model P(underdog wins)", scale=alt.Scale(domain=[0, axis_max])),
                size=alt.Size("seed_gap:Q", legend=None, scale=alt.Scale(range=[60, 400])),
                color=alt.Color(
                    "model_edge:Q",
                    title="Model edge",
                    scale=alt.Scale(scheme="redyellowgreen", domainMid=0),
                ),
                tooltip=common_tooltip,
            )
            .properties(height=320)
        )
        st.altair_chart(ref_line + scatter, use_container_width=True)

    st.subheader(f"Top Upsets — sorted by {sort_label}")
    topn = df_plot.head(15)

    bar_color = (
        alt.Color("model_edge:Q", title="Model edge", scale=alt.Scale(scheme="redyellowgreen", domainMid=0))
        if has_hist
        else alt.Color("region:N", title="Region")
    )

    bars = (
        alt.Chart(topn)
        .mark_bar()
        .encode(
            x=alt.X(f"{sort_col}:Q", title=sort_label),
            y=alt.Y("matchup:N", sort="-x", title=None),
            color=bar_color,
            tooltip=common_tooltip,
        )
        .properties(height=min(420, 28 * len(topn)))
    )
    st.altair_chart(bars, use_container_width=True)

display_cols = ["region", "underdog_seed", "underdog", "favorite_seed", "favorite",
                "p_underdog_win", "hist_win_pct", "model_edge", "seed_gap"]
if not has_hist:
    display_cols = [c for c in display_cols if c not in ("hist_win_pct", "model_edge")]

st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
