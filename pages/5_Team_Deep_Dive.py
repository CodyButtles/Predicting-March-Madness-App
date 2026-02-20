from __future__ import annotations

from pathlib import Path
from typing import Any
import io

import pandas as pd
import streamlit as st
from typing import cast

from mm_app.bracket import ROUND1_PAIRINGS
from mm_app.load import load_advancement_probs, load_bracket_field, load_matchup_probs
from mm_app.paths import get_output_paths
from mm_app.probs import get_matchup_probability
from mm_app.util import slug_to_display_name


st.set_page_config(page_title="Team Deep-Dive", layout="wide")
st.title("Team Deep-Dive")

DEFAULT_PUBLIC_YEAR = 2024
st.session_state.setdefault("year", DEFAULT_PUBLIC_YEAR)
year = int(st.session_state.get("year", DEFAULT_PUBLIC_YEAR))
paths = get_output_paths(year)

try:
    bracket_field = load_bracket_field(year)
    adv = load_advancement_probs(year)
except FileNotFoundError:
    st.error(
        "Missing required JSON inputs. Generate them via the 06 notebook "
        "or configure private-data fetching for public deployments. "
        f"Expected {paths.bracket_field_json.name} and {paths.advancement_probs_json.name}."
    )
    st.stop()

try:
    probs = load_matchup_probs(year)
except FileNotFoundError:
    probs = {}


@st.cache_data(show_spinner=False)
def load_raw_team_data(root: Path) -> pd.DataFrame:
    p = root / "Data" / "Raw_Team_Data.csv"
    try:
        if p.exists():
            # Low-memory read; the file is large.
            return pd.read_csv(p, low_memory=False)

        # Public deployments may keep sensitive data in a private repo.
        from mm_app.private_data import read_bytes_maybe_private

        raw_bytes = read_bytes_maybe_private(p)
        return pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load Raw_Team_Data.csv: {e}")


def tournament_teams_from_bracket_field(field: dict[str, Any]) -> list[str]:
    teams: list[str] = []
    for seeds in field.values():
        for row in seeds:
            t = row.get("team")
            if isinstance(t, str):
                teams.append(t)
    return sorted(set(teams))


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    # Only numeric columns that aren't identifiers.
    ignore = {
        "seed",
        "tournament_win_count",
        "tourney_year",
        "team_id",
    }
    cols = []
    for c in df.columns:
        if c in ignore:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

teams = sorted(adv.keys())
if not teams:
    st.error("No teams found in advancement odds JSON.")
    st.stop()

team = cast(str, st.selectbox("Team", teams, format_func=slug_to_display_name))

st.subheader("Advancement odds")
row = adv.get(team, {})
order = ["R32", "S16", "E8", "F4", "Final", "Champ"]
df = pd.DataFrame([{"round": r, "pct": row.get(r)} for r in order])
st.dataframe(df, use_container_width=True, hide_index=True)


st.subheader("Raw stat outliers (vs field)")
st.caption(
    "Uses `Data/Raw_Team_Data.csv` (raw per-team stats) and compares the selected team to a peer group "
    "using within-year z-scores and percentiles. This does not depend on model feature importances. "
    "(In this dataset, the peer group is the teams present for that tournament year.)"
)

root = Path(__file__).resolve().parents[1]
try:
    raw = load_raw_team_data(root)
except FileNotFoundError as e:
    st.warning(str(e))
    raw = pd.DataFrame()

if not raw.empty:
    # Map raw schema → expected columns
    # Raw file uses `team_x` and `tourney_year`.
    if "team_x" not in raw.columns or "tourney_year" not in raw.columns:
        st.warning("Raw_Team_Data.csv is missing required columns: team_x and/or tourney_year")
    else:
        df_year = raw.loc[raw["tourney_year"].astype(int) == int(year)].copy()
        df_group = df_year

        df_group = df_group.dropna(subset=["team_x"]).copy()
        df_team_rows = df_group.loc[df_group["team_x"] == team].copy()
        if df_team_rows.empty:
            st.info("Could not find this team in Raw_Team_Data.csv for the selected year.")
        else:
            if len(df_team_rows) > 1:
                st.warning(
                    "Multiple rows found for this team/year in Raw_Team_Data.csv; using the first row."
                )
            df_team = df_team_rows.iloc[0]

            include_opponent = st.checkbox("Include opponent_ metrics", value=False)
            include_last3 = st.checkbox("Include *_last_3 metrics", value=False)

            feat_cols = numeric_feature_columns(df_group)
            if not include_opponent:
                feat_cols = [c for c in feat_cols if not c.lower().startswith("opponent_")]
            if not include_last3:
                feat_cols = [c for c in feat_cols if not c.lower().endswith("_last_3")]

            z_cut = float(st.slider("Outlier threshold (|z|)", min_value=0.5, max_value=3.0, value=1.25, step=0.05))
            top_n = int(st.number_input("Show top N", min_value=5, max_value=30, value=12, step=1))

            rows: list[dict[str, Any]] = []
            for c in feat_cols:
                series = pd.to_numeric(df_group[c], errors="coerce")
                x = float(pd.to_numeric(df_team.get(c), errors="coerce")) if pd.notna(df_team.get(c)) else float("nan")
                mu = float(series.mean(skipna=True))
                sd = float(series.std(skipna=True, ddof=0))
                if not pd.notna(x) or not pd.notna(mu) or not pd.notna(sd) or sd <= 0:
                    continue
                z = (x - mu) / sd
                pct = float(series.rank(pct=True, ascending=True).loc[df_group["team_x"] == team].iloc[0]) if (df_group["team_x"] == team).any() else float("nan")
                rows.append(
                    {
                        "metric": c,
                        "value": x,
                        "field_mean": mu,
                        "field_std": sd,
                        "z": float(z),
                        "percentile": float(pct),
                    }
                )

            stats = pd.DataFrame(rows)
            if stats.empty:
                st.info("No numeric metrics available after filtering.")
            else:
                stats["abs_z"] = stats["z"].abs()
                outliers = stats.loc[stats["abs_z"] >= z_cut].copy()

                highs = outliers.sort_values(["z", "abs_z"], ascending=[False, False]).head(top_n)
                lows = outliers.sort_values(["z", "abs_z"], ascending=[True, False]).head(top_n)
                st.markdown("**High outliers (z-score)**")
                st.dataframe(
                    highs[["metric", "value", "z", "percentile"]],
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown("**Low outliers (z-score)**")
                st.dataframe(
                    lows[["metric", "value", "z", "percentile"]],
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("**Explore a metric**")
                metric = cast(str, st.selectbox("Metric", options=sorted(stats["metric"].unique().tolist())))
                s = pd.to_numeric(df_group[metric], errors="coerce")
                team_val = float(pd.to_numeric(df_team.get(metric), errors="coerce"))
                mu = float(s.mean(skipna=True))
                sd = float(s.std(skipna=True, ddof=0))
                team_z = (team_val - mu) / sd if pd.notna(team_val) and sd > 0 else float("nan")

                st.write(
                    f"{slug_to_display_name(team)}: **{team_val:.3f}** | "
                    f"field mean: {mu:.3f} | field std: {sd:.3f} | "
                    f"z: **{team_z:.2f}**"
                )

                # Histogram of z-scores, with a marker for the selected team.
                z_series = (s - mu) / sd if sd > 0 else pd.Series(dtype=float)
                z_series = z_series.replace([float("inf"), float("-inf")], pd.NA).dropna()
                maxbins = int(st.slider("Histogram bins", min_value=10, max_value=50, value=25, step=1))

                try:
                    import altair as alt

                    df_plot = pd.DataFrame({"z": z_series.astype(float)})
                    base = (
                        alt.Chart(df_plot)
                        .mark_bar()
                        .encode(
                            x=alt.X("z:Q", bin=alt.Bin(maxbins=maxbins), title="Z-score (within year)")
                            ,
                            y=alt.Y("count():Q", title="Number of teams"),
                        )
                    )
                    rule = alt.Chart(pd.DataFrame({"z": [team_z]})).mark_rule(color="#d62728").encode(
                        x="z:Q"
                    )
                    st.altair_chart(base + rule, use_container_width=True)
                except Exception:
                    # Fallback: plain histogram counts via Streamlit bar chart.
                    import numpy as np

                    counts, edges = np.histogram(z_series.to_numpy(dtype=float), bins=maxbins)
                    centers = (edges[:-1] + edges[1:]) / 2.0
                    df_hist = pd.DataFrame({"count": counts}, index=pd.Index(centers, name="z"))
                    st.bar_chart(df_hist, height=200)

                st.divider()
                st.markdown("**Team profile radar (role-based groupings)**")
                st.caption(
                    "Each axis is the average within-year percentile across a small set of raw metrics. "
                    "This is meant to be a quick 'style/profile' view, not a model feature importance view."
                )

                # Representative metric sets (must exist in Raw_Team_Data.csv to be used).
                category_metrics = {
                    "Shooting": [
                        "effective_field_goal_percent",
                        "true_shooting_percent",
                        "three_point_percent",
                        "two_point_percent",
                        "free_throw_percent",
                    ],
                    "Rebounding": [
                        "offensive_rebounding_percent",
                        "defensive_rebounding_percent",
                        "total_rebounding_percent_rebound_rate",
                    ],
                    "Ball security": [
                        "turnovers_per_possession",
                        "assist__turnover_ratio",
                    ],
                    "Pace": [
                        "possessions_per_game",
                    ],
                    "Defense": [
                        "defensive_efficiency",
                        "opponent_points_per_game",
                        "steals_per_game",
                        "blocks_per_game",
                    ],
                }

                available = set(df_group.columns)
                category_metrics = {
                    k: [m for m in ms if m in available]
                    for k, ms in category_metrics.items()
                }

                st.caption(
                    "Note: for the radar only, `defensive_efficiency` and `opponent_points_per_game` are inverted "
                    "(because lower is better), so a higher percentile always means better on those two axes inputs."
                )

                invert_metrics = {"defensive_efficiency", "opponent_points_per_game"}

                # Compute percentiles for the selected team for each metric.
                metric_pct: dict[str, float] = {}
                for cat, ms in category_metrics.items():
                    for m in ms:
                        s_m = pd.to_numeric(df_group[m], errors="coerce")
                        if (df_group["team_x"] == team).any():
                            p_m = float(s_m.rank(pct=True, ascending=True).loc[df_group["team_x"] == team].iloc[0])
                        else:
                            p_m = float("nan")
                        if not pd.notna(p_m):
                            continue
                        pct_val = p_m * 100.0
                        if m.lower() in invert_metrics:
                            pct_val = 100.0 - pct_val
                        metric_pct[m] = pct_val

                # Aggregate to category scores.
                cat_rows: list[dict[str, Any]] = []
                radar_labels: list[str] = []
                radar_values: list[float] = []

                for cat, ms in category_metrics.items():
                    vals = [metric_pct[m] for m in ms if m in metric_pct]
                    if not vals:
                        continue
                    score = float(pd.Series(vals).mean())
                    cat_rows.append(
                        {
                            "category": cat,
                            "score_percentile": round(score, 1),
                            "metrics_used": ", ".join(ms),
                        }
                    )
                    radar_labels.append(cat)
                    radar_values.append(score)

                if not radar_labels:
                    st.info("Radar chart is unavailable because the expected raw metrics were not found for this year/team group.")
                else:
                    st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)

                    import plotly.graph_objects as go

                    labels_closed = radar_labels + [radar_labels[0]]
                    values_closed = radar_values + [radar_values[0]]
                    fig = go.Figure(
                        data=[
                            go.Scatterpolar(
                                r=values_closed,
                                theta=labels_closed,
                                fill="toself",
                                name=slug_to_display_name(team),
                            )
                        ]
                    )
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 100], tickvals=[0, 25, 50, 75, 100])
                        ),
                        showlegend=False,
                        margin=dict(l=30, r=30, t=30, b=30),
                        height=420,
                    )
                    st.plotly_chart(fig, use_container_width=True)

# First-round context (if matchup_probs exist)
if probs:
    st.subheader("Round of 64 matchup")
    found = None
    for region_key, seeds in bracket_field.items():
        seed_to_team = {int(x["seed"]): x["team"] for x in seeds}
        team_to_seed = {v: k for k, v in seed_to_team.items()}
        if team not in team_to_seed:
            continue
        seed = team_to_seed[team]

        for a, b in ROUND1_PAIRINGS:
            if seed not in (a, b):
                continue
            opp_seed = b if seed == a else a
            opp = seed_to_team[opp_seed]
            found = (region_key, seed, opp_seed, opp)
            break

    if found:
        region_key, seed, opp_seed, opp = found
        p = get_matchup_probability(team, opp, probs)
        st.write(
            f"{slug_to_display_name(team)} ({seed}) vs {slug_to_display_name(opp)} ({opp_seed}) in {region_key}"
        )
        st.write(f"P(win) = **{p:.3f}**")
    else:
        st.info("Could not locate Round of 64 opponent from bracket_field.")
