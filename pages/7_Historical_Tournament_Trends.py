from __future__ import annotations

from pathlib import Path
import io

import numpy as np
import pandas as pd
import streamlit as st

try:
    import altair as alt
except Exception:  # pragma: no cover
    alt = None  # type: ignore[assignment]


st.set_page_config(page_title="Historical Trends", layout="wide")
st.title("Historical Tournament Trends")
st.caption(
    "Uses historical tournament game results (with seeds) to compute seed-vs-seed win rates and upset rates."
)


@st.cache_data(show_spinner=False)
def _load_historical_seed_results_agg() -> pd.DataFrame:
    """Optional small aggregated file (preferred over the full spreads CSV).

    Expected columns:
      - year, round, seed, opp_seed, games, wins
    """

    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "Output" / "2024" / "historical_seed_results_agg.csv",
        root / "Output" / "historical_seed_results_agg.csv",
    ]

    for p in candidates:
        try:
            if p.exists():
                buf: io.BytesIO | None = None
            else:
                from mm_app.private_data import read_bytes_maybe_private

                raw_bytes = read_bytes_maybe_private(p)
                buf = io.BytesIO(raw_bytes)

            df = pd.read_csv(buf if buf is not None else p)
            needed = {"year", "round", "seed", "opp_seed", "games", "wins"}
            if not needed.issubset(set(df.columns)):
                continue

            out = df[list(needed)].copy()
            for c in ["year", "round", "seed", "opp_seed", "games", "wins"]:
                out[c] = pd.to_numeric(out[c], errors="coerce")
            out = out.dropna().astype(int)
            out = out[(out["seed"].between(1, 16)) & (out["opp_seed"].between(1, 16))]
            return out
        except Exception:
            continue

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_historical_upsets_agg() -> pd.DataFrame:
    """Optional small aggregated file.

    Expected columns:
      - year, round, games, upsets
    """

    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "Output" / "2024" / "historical_upsets_agg.csv",
        root / "Output" / "historical_upsets_agg.csv",
    ]

    for p in candidates:
        try:
            if p.exists():
                buf: io.BytesIO | None = None
            else:
                from mm_app.private_data import read_bytes_maybe_private

                raw_bytes = read_bytes_maybe_private(p)
                buf = io.BytesIO(raw_bytes)

            df = pd.read_csv(buf if buf is not None else p)
            needed = {"year", "round", "games", "upsets"}
            if not needed.issubset(set(df.columns)):
                continue

            out = df[list(needed)].copy()
            for c in ["year", "round", "games", "upsets"]:
                out[c] = pd.to_numeric(out[c], errors="coerce")
            out = out.dropna().astype(int)
            return out
        except Exception:
            continue

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_historical_champions_agg() -> pd.DataFrame:
    """Optional small file.

    Expected columns:
      - year, champ_seed, champ_quad
    """

    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "Output" / "2024" / "historical_champions_agg.csv",
        root / "Output" / "historical_champions_agg.csv",
    ]

    for p in candidates:
        try:
            if p.exists():
                buf: io.BytesIO | None = None
            else:
                from mm_app.private_data import read_bytes_maybe_private

                raw_bytes = read_bytes_maybe_private(p)
                buf = io.BytesIO(raw_bytes)

            df = pd.read_csv(buf if buf is not None else p)
            needed = {"year", "champ_seed", "champ_quad"}
            if not needed.issubset(set(df.columns)):
                continue

            out = df[list(needed)].copy()
            out["year"] = pd.to_numeric(out["year"], errors="coerce")
            out["champ_seed"] = pd.to_numeric(out["champ_seed"], errors="coerce")
            out = out.dropna(subset=["year", "champ_seed"]).copy()
            out["year"] = out["year"].astype(int)
            out["champ_seed"] = out["champ_seed"].astype(int)
            out["champ_quad"] = out["champ_quad"].fillna("").astype(str)
            out = out[out["champ_seed"].between(1, 16)]
            return out
        except Exception:
            continue

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_historical_games() -> pd.DataFrame:
    root = Path(__file__).resolve().parents[1]
    p = root / "Output" / "2000-2025_Spreads.csv"
    try:
        if p.exists():
            raw_buf: io.BytesIO | None = None
        else:
            from mm_app.private_data import read_bytes_maybe_private

            raw_bytes = read_bytes_maybe_private(p)
            raw_buf = io.BytesIO(raw_bytes)
    except Exception:
        return pd.DataFrame()

    needed = {
        "tournament_year",
        "round_ind",
        "real_game_ind",
        "team_name",
        "opponent_name",
        "team1_seed",
        "team2_seed",
        "team1_bracket_quad",
        "team2_bracket_quad",
        "team1_win_loss_ind",
        "win_loss_indicator",
    }

    df = pd.read_csv(raw_buf if raw_buf is not None else p, low_memory=False, usecols=lambda c: c in needed)

    def _s(col: str, *, default) -> pd.Series:
        if col in df.columns:
            return df[col]
        return pd.Series(default, index=df.index)

    outcome = _s("team1_win_loss_ind", default=np.nan)
    if outcome.isna().all():
        outcome = _s("win_loss_indicator", default=np.nan)

    out = pd.DataFrame(
        {
            "year": pd.to_numeric(_s("tournament_year", default=np.nan), errors="coerce"),
            "round": pd.to_numeric(_s("round_ind", default=np.nan), errors="coerce"),
            "real_game": pd.to_numeric(_s("real_game_ind", default=1), errors="coerce").fillna(1),
            "team_a": _s("team_name", default="").astype(str),
            "team_b": _s("opponent_name", default="").astype(str),
            "seed_a": pd.to_numeric(_s("team1_seed", default=np.nan), errors="coerce"),
            "seed_b": pd.to_numeric(_s("team2_seed", default=np.nan), errors="coerce"),
            "quad_a": _s("team1_bracket_quad", default="").astype(str),
            "quad_b": _s("team2_bracket_quad", default="").astype(str),
            "a_wins": pd.to_numeric(outcome, errors="coerce"),
        }
    )

    out = out.dropna(subset=["year", "round", "seed_a", "seed_b", "a_wins"]).copy()
    out["year"] = out["year"].astype(int)
    out["round"] = out["round"].astype(int)
    out["seed_a"] = out["seed_a"].astype(int)
    out["seed_b"] = out["seed_b"].astype(int)
    out["a_wins"] = out["a_wins"].astype(int)

    out = out[(out["seed_a"].between(1, 16)) & (out["seed_b"].between(1, 16))]
    out = out[out["a_wins"].isin([0, 1])]

    if "real_game" in out.columns:
        out = out[out["real_game"] == 1]

    ta = out["team_a"].astype(str)
    tb = out["team_b"].astype(str)
    min_team = np.where(ta < tb, ta, tb)
    max_team = np.where(ta < tb, tb, ta)
    out["_k"] = (
        out["year"].astype(str)
        + "|"
        + out["round"].astype(str)
        + "|"
        + pd.Series(min_team).astype(str)
        + "|"
        + pd.Series(max_team).astype(str)
    )
    out = out.drop_duplicates(subset=["_k"], keep="first").drop(columns=["_k"])

    return out


ROUND_LABEL = {
    1: "R64",
    2: "R32",
    3: "S16",
    4: "E8",
    5: "F4",
    6: "Final",
}


df = _load_historical_games()
seed_agg_df = _load_historical_seed_results_agg()
upsets_agg_df = _load_historical_upsets_agg()
champs_agg_df = _load_historical_champions_agg()

if df.empty and seed_agg_df.empty:
    st.info(
        "Historical trends data is not available in this deployment. "
        "This page expects `Output/2000-2025_Spreads.csv` with seed + result columns."
    )
    st.stop()

if not seed_agg_df.empty:
    min_year, max_year = int(seed_agg_df["year"].min()), int(seed_agg_df["year"].max())
    rounds_available = sorted(int(x) for x in seed_agg_df["round"].unique())
else:
    min_year, max_year = int(df["year"].min()), int(df["year"].max())
    rounds_available = sorted(int(x) for x in df["round"].unique())

st.subheader("Filters")
f1, f2, f3 = st.columns([2, 3, 2])
with f1:
    year_range = st.slider("Year range", min_value=min_year, max_value=max_year, value=(min_year, max_year))
with f2:
    round_labels = [f"{ROUND_LABEL.get(r, f'R{r}')} ({r})" for r in rounds_available]
    picked_round_labels = st.multiselect("Rounds", options=round_labels, default=round_labels)
    picked_rounds = [
        rounds_available[round_labels.index(lbl)] for lbl in picked_round_labels if lbl in round_labels
    ]
with f3:
    min_games = int(st.slider("Min games (per cell)", min_value=1, max_value=50, value=3, step=1))

if not seed_agg_df.empty:
    f_seed_agg = seed_agg_df[
        (seed_agg_df["year"].between(year_range[0], year_range[1]))
        & (seed_agg_df["round"].isin(picked_rounds))
    ].copy()
    if f_seed_agg.empty:
        st.warning("No games match the current filters.")
        st.stop()
else:
    fdf = df[(df["year"].between(year_range[0], year_range[1])) & (df["round"].isin(picked_rounds))].copy()
    if fdf.empty:
        st.warning("No games match the current filters.")
        st.stop()

st.subheader("Champions")
st.caption("Champion seed distribution and which bracket quadrant the champions came from.")

if not champs_agg_df.empty:
    champs_df = champs_agg_df[champs_agg_df["year"].between(year_range[0], year_range[1])].copy()
else:
    cdf = df[df["year"].between(year_range[0], year_range[1])].copy() if not df.empty else pd.DataFrame()
    if cdf.empty:
        champs_df = pd.DataFrame()
    else:
        max_round_by_year = cdf.groupby("year", as_index=False).agg(final_round=("round", "max"))
        finals = cdf.merge(max_round_by_year, on="year", how="inner")
        finals = finals[finals["round"] == finals["final_round"]].copy()

        finals["champ_seed"] = np.where(finals["a_wins"].astype(int) == 1, finals["seed_a"], finals["seed_b"])
        finals["champ_quad"] = np.where(finals["a_wins"].astype(int) == 1, finals["quad_a"], finals["quad_b"])
        champs_df = finals[["year", "champ_seed", "champ_quad"]].copy()
        champs_df["champ_seed"] = pd.to_numeric(champs_df["champ_seed"], errors="coerce")
        champs_df = champs_df.dropna(subset=["year", "champ_seed"]).copy()
        champs_df["champ_seed"] = champs_df["champ_seed"].astype(int)
        champs_df["champ_quad"] = champs_df["champ_quad"].fillna("").astype(str)

if champs_df.empty:
    st.info(
        "Champion summaries are not available for the selected filters. "
        "This requires either `historical_champions_agg.csv` or quadrant columns in the full spreads CSV."
    )
else:
    c1, c2 = st.columns(2)
    with c1:
        seed_counts = (
            champs_df.groupby("champ_seed", as_index=False)
            .size()
            .rename(columns={"size": "titles"})
            .sort_values("champ_seed")
        )
        st.write("**Champion seed distribution**")
        if alt is not None:
            st.altair_chart(
                alt.Chart(seed_counts)
                .mark_bar()
                .encode(
                    x=alt.X("champ_seed:O", title="Seed"),
                    y=alt.Y("titles:Q", title="Titles"),
                    tooltip=[
                        alt.Tooltip("champ_seed:O", title="seed"),
                        alt.Tooltip("titles:Q", title="titles"),
                    ],
                )
                .properties(height=220),
                use_container_width=True,
            )
        else:
            st.dataframe(seed_counts, use_container_width=True, hide_index=True)

    with c2:
        quad_counts = champs_df.copy()
        quad_counts["champ_quad"] = quad_counts["champ_quad"].replace({"nan": ""})
        quad_counts = (
            quad_counts[quad_counts["champ_quad"].str.len() > 0]
            .groupby("champ_quad", as_index=False)
            .size()
            .rename(columns={"size": "titles"})
        )
        st.write("**Champion quadrant origins**")
        if quad_counts.empty:
            st.info("Quadrant data not present for these years.")
        elif alt is not None:
            quad_order = ["UL", "UR", "LL", "LR"]
            st.altair_chart(
                alt.Chart(quad_counts)
                .mark_bar()
                .encode(
                    x=alt.X("champ_quad:N", sort=quad_order, title="Quadrant"),
                    y=alt.Y("titles:Q", title="Titles"),
                    tooltip=[
                        alt.Tooltip("champ_quad:N", title="quad"),
                        alt.Tooltip("titles:Q", title="titles"),
                    ],
                )
                .properties(height=220),
                use_container_width=True,
            )
        else:
            st.dataframe(quad_counts.sort_values("champ_quad"), use_container_width=True, hide_index=True)

    with st.expander("Champion list (by year)", expanded=False):
        st.dataframe(
            champs_df.sort_values("year"),
            use_container_width=True,
            hide_index=True,
        )

if not seed_agg_df.empty:
    agg = (
        f_seed_agg.groupby(["seed", "opp_seed"], as_index=False)
        .agg(games=("games", "sum"), wins=("wins", "sum"))
        .assign(win_rate=lambda d: d["wins"] / d["games"])
    )
    agg = agg[agg["games"] >= min_games].copy()
else:
    seed_a = fdf["seed_a"].astype(int)
    seed_b = fdf["seed_b"].astype(int)
    a_wins = fdf["a_wins"].astype(int)

    long = pd.concat(
        [
            pd.DataFrame({"seed": seed_a, "opp_seed": seed_b, "win": a_wins}),
            pd.DataFrame({"seed": seed_b, "opp_seed": seed_a, "win": 1 - a_wins}),
        ],
        ignore_index=True,
    )

    agg = (
        long.groupby(["seed", "opp_seed"], as_index=False)
        .agg(games=("win", "size"), wins=("win", "sum"))
        .assign(win_rate=lambda d: d["wins"] / d["games"])
    )

    agg = agg[agg["games"] >= min_games].copy()

st.subheader("Seed-vs-Seed Win Rates")
st.caption(
    "Cell shows P(row seed beats column seed) in actual tournament games. "
    "Use filters above to restrict years/rounds."
)

if alt is not None and not agg.empty:
    heat = (
        alt.Chart(agg)
        .mark_rect()
        .encode(
            x=alt.X("opp_seed:O", title="Opponent seed"),
            y=alt.Y("seed:O", title="Seed"),
            color=alt.Color("win_rate:Q", title="Win %", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("seed:O", title="seed"),
                alt.Tooltip("opp_seed:O", title="opp"),
                alt.Tooltip("games:Q", title="games"),
                alt.Tooltip("wins:Q", title="wins"),
                alt.Tooltip("win_rate:Q", title="win%", format=".1%"),
            ],
        )
        .properties(height=520)
    )
    st.altair_chart(heat, use_container_width=True)
else:
    st.dataframe(agg.sort_values(["seed", "opp_seed"]), use_container_width=True, hide_index=True)

st.subheader("Upset Rates")

st.caption(
    "An upset is defined as the winner having a worse (higher) seed number than the loser. "
    "Use the filters above to restrict years and rounds."
)

if not upsets_agg_df.empty:
    up_f = upsets_agg_df[
        (upsets_agg_df["year"].between(year_range[0], year_range[1]))
        & (upsets_agg_df["round"].isin(picked_rounds))
    ].copy()
    if up_f.empty:
        up_by_round = pd.DataFrame(columns=["round", "round_label", "games", "upset_rate"])
        up_by_year = pd.DataFrame(columns=["year", "games", "upsets", "upset_rate"])
        overall_games = 0
        overall_upset_rate = float("nan")
    else:
        up_by_round = (
            up_f.groupby("round", as_index=False)
            .agg(games=("games", "sum"), upsets=("upsets", "sum"))
            .assign(
                upset_rate=lambda d: d["upsets"] / d["games"],
                round_label=lambda d: d["round"].map(lambda r: ROUND_LABEL.get(int(r), f"R{int(r)}")),
            )
            .sort_values("round")
        )
        up_by_year = (
            up_f.groupby("year", as_index=False)
            .agg(games=("games", "sum"), upsets=("upsets", "sum"))
            .assign(upset_rate=lambda d: d["upsets"] / d["games"])
            .sort_values("year")
        )
        overall_games = int(up_f["games"].sum())
        overall_upset_rate = float(up_f["upsets"].sum() / up_f["games"].sum())
else:
    if df.empty:
        up_by_round = pd.DataFrame(columns=["round", "round_label", "games", "upset_rate"])
        up_by_year = pd.DataFrame(columns=["year", "games", "upsets", "upset_rate"])
        overall_games = 0
        overall_upset_rate = float("nan")
    else:
        upset_src = df[(df["year"].between(year_range[0], year_range[1])) & (df["round"].isin(picked_rounds))].copy()
        if upset_src.empty:
            up_by_round = pd.DataFrame(columns=["round", "round_label", "games", "upset_rate"])
            up_by_year = pd.DataFrame(columns=["year", "games", "upsets", "upset_rate"])
            overall_games = 0
            overall_upset_rate = float("nan")
        else:
            seed_a_u = upset_src["seed_a"].astype(int)
            seed_b_u = upset_src["seed_b"].astype(int)
            a_wins_u = upset_src["a_wins"].astype(int)

            winner_seed = np.where(a_wins_u == 1, seed_a_u, seed_b_u)
            loser_seed = np.where(a_wins_u == 1, seed_b_u, seed_a_u)
            upset = (winner_seed > loser_seed).astype(int)

            up_df = upset_src[["year", "round"]].copy()
            up_df["upset"] = upset
            up_df["round_label"] = up_df["round"].map(lambda r: ROUND_LABEL.get(int(r), f"R{int(r)}"))

            up_by_round = (
                up_df.groupby(["round", "round_label"], as_index=False)
                .agg(games=("upset", "size"), upset_rate=("upset", "mean"))
                .sort_values("round")
            )
            up_by_year = (
                up_df.groupby("year", as_index=False)
                .agg(games=("upset", "size"), upsets=("upset", "sum"))
                .assign(upset_rate=lambda d: d["upsets"] / d["games"])
                .sort_values("year")
            )
            overall_games = int(len(up_df))
            overall_upset_rate = float(up_df["upset"].mean())

c1, c2 = st.columns(2)
with c1:
    st.metric("Games in filter", str(overall_games))
with c2:
    if np.isnan(overall_upset_rate):
        st.metric("Overall upset rate", "N/A")
    else:
        st.metric("Overall upset rate", f"{overall_upset_rate:.1%}")

if alt is not None and not up_by_round.empty:
    bar = (
        alt.Chart(up_by_round)
        .mark_bar()
        .encode(
            x=alt.X("round_label:N", sort=None, title="Round"),
            y=alt.Y("upset_rate:Q", title="Upset rate", axis=alt.Axis(format="%")),
            tooltip=[
                alt.Tooltip("round_label:N", title="round"),
                alt.Tooltip("games:Q", title="games"),
                alt.Tooltip("upset_rate:Q", title="upset%", format=".1%"),
            ],
        )
        .properties(height=220)
    )
    st.altair_chart(bar, use_container_width=True)
else:
    st.dataframe(up_by_round, use_container_width=True, hide_index=True)


st.subheader("Upsets by Year")
st.caption("Counts upsets within the selected year range and rounds.")

if up_by_year.empty:
    st.info("No upsets-by-year data available for the current filters.")
else:
    if alt is not None:
        year_bar = (
            alt.Chart(up_by_year)
            .mark_bar()
            .encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("upsets:Q", title="# Upsets"),
                tooltip=[
                    alt.Tooltip("year:O", title="year"),
                    alt.Tooltip("games:Q", title="games"),
                    alt.Tooltip("upsets:Q", title="upsets"),
                    alt.Tooltip("upset_rate:Q", title="upset%", format=".1%"),
                ],
            )
            .properties(height=240)
        )
        st.altair_chart(year_bar, use_container_width=True)
    else:
        st.dataframe(up_by_year, use_container_width=True, hide_index=True)

st.subheader("Round of 64: Win % by Seed")
st.caption(
    "For the selected year range, this shows how often each seed wins in Round of 64 games. "
    "(A game contributes once to the winning seed and once to the losing seed.)"
)

if not seed_agg_df.empty:
    r64 = seed_agg_df[(seed_agg_df["year"].between(year_range[0], year_range[1])) & (seed_agg_df["round"] == 1)].copy()
else:
    r64 = df[(df["year"].between(year_range[0], year_range[1])) & (df["round"] == 1)].copy()

if r64.empty:
    st.info("No Round of 64 games found in the selected year range.")
else:
    if not seed_agg_df.empty:
        by_seed = (
            r64.groupby("seed", as_index=False)
            .agg(games=("games", "sum"), wins=("wins", "sum"))
            .assign(win_rate=lambda d: d["wins"] / d["games"])
            .sort_values("seed")
        )
        by_seed = by_seed[["seed", "games", "win_rate"]].copy()
    else:
        seed_a = r64["seed_a"].astype(int)
        seed_b = r64["seed_b"].astype(int)
        a_wins = r64["a_wins"].astype(int)

        r64_long = pd.concat(
            [
                pd.DataFrame({"seed": seed_a, "win": a_wins}),
                pd.DataFrame({"seed": seed_b, "win": 1 - a_wins}),
            ],
            ignore_index=True,
        )

        by_seed = (
            r64_long.groupby("seed", as_index=False)
            .agg(games=("win", "size"), win_rate=("win", "mean"))
            .sort_values("seed")
        )

    by_seed = by_seed[by_seed["games"] >= min_games].copy()

    if alt is not None and not by_seed.empty:
        chart = (
            alt.Chart(by_seed)
            .mark_bar()
            .encode(
                x=alt.X("seed:O", title="Seed"),
                y=alt.Y("win_rate:Q", title="Win %", axis=alt.Axis(format="%")),
                tooltip=[
                    alt.Tooltip("seed:O", title="seed"),
                    alt.Tooltip("games:Q", title="games"),
                    alt.Tooltip("win_rate:Q", title="win%", format=".1%"),
                ],
            )
            .properties(height=240)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.dataframe(by_seed, use_container_width=True, hide_index=True)
