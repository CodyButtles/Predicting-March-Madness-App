from __future__ import annotations

from typing import cast

import pandas as pd
import streamlit as st

try:
    import altair as alt
except Exception:  # pragma: no cover
    alt = None  # type: ignore[assignment]

from mm_app.load import load_matchup_explanations, load_matchup_probs
from mm_app.paths import get_output_paths
from mm_app.probs import confidence_from_probability, get_matchup_probability
from mm_app.util import slug_to_display_name


st.set_page_config(page_title="Game Explorer", layout="wide")
st.title("Game-by-Game Probability Explorer")

DEFAULT_PUBLIC_YEAR = 2024
st.session_state.setdefault("year", DEFAULT_PUBLIC_YEAR)


def _get_query_params() -> dict[str, str]:
    # Streamlit has two APIs depending on version.
    if hasattr(st, "query_params"):
        qp = st.query_params  # type: ignore[attr-defined]
        out: dict[str, str] = {}
        for k in ["team_a", "team_b", "year"]:
            v = qp.get(k)
            if v is None:
                continue
            if isinstance(v, list):
                if v:
                    out[k] = str(v[0])
            else:
                out[k] = str(v)
        return out

    qp2 = st.experimental_get_query_params()  # type: ignore[attr-defined]
    return {k: str(v[0]) for k, v in qp2.items() if v}


def _set_query_params(**kwargs: str) -> None:
    if hasattr(st, "query_params"):
        # type: ignore[attr-defined]
        st.query_params.update(kwargs)
        return
    st.experimental_set_query_params(**kwargs)  # type: ignore[attr-defined]


def _flip_teams_callback(*, year: int) -> None:
    a = st.session_state.get("gbg_team_a")
    b = st.session_state.get("gbg_team_b")
    if not a or not b or a == b:
        return
    st.session_state["gbg_team_a"] = b
    st.session_state["gbg_team_b"] = a
    _set_query_params(team_a=str(b), team_b=str(a), year=str(year))

qp = _get_query_params()
if qp.get("year") and str(qp.get("year")).isdigit():
    # Public deployment is pinned to a single year.
    # Ignore arbitrary querystring overrides to avoid missing artifact errors.
    if int(qp["year"]) == DEFAULT_PUBLIC_YEAR:
        st.session_state["year"] = DEFAULT_PUBLIC_YEAR

year = int(st.session_state.get("year", DEFAULT_PUBLIC_YEAR))
paths = get_output_paths(year)

try:
    probs = load_matchup_probs(year)
except FileNotFoundError:
    st.error(
        "Missing matchup probability JSON for this page. Generate it via the 06 notebook "
        "or configure private-data fetching for public deployments. "
        f"Expected {paths.matchup_probs_json.name}."
    )
    st.stop()
teams = sorted(probs.keys())

if not teams:
    st.error("No teams found in matchup probabilities JSON.")
    st.stop()

# If navigated from the bracket builder, preselect Team A / Team B from query params.
qa = qp.get("team_a")
qb = qp.get("team_b")

if qa in teams:
    st.session_state["gbg_team_a"] = qa

# Only set Team B after Team A is known.
if qb in teams:
    st.session_state["gbg_team_b"] = qb

col1, col2 = st.columns(2)
with col1:
    team_a = cast(
        str,
        st.selectbox(
            "Team A",
            teams,
            index=teams.index(st.session_state["gbg_team_a"]) if st.session_state.get("gbg_team_a") in teams else 0,
            key="gbg_team_a",
            format_func=slug_to_display_name,
        ),
    )
with col2:
    team_b_options = [t for t in teams if t != team_a]
    if st.session_state.get("gbg_team_b") not in team_b_options:
        st.session_state["gbg_team_b"] = team_b_options[0]
    team_b = cast(
        str,
        st.selectbox(
        "Team B",
        team_b_options,
        index=team_b_options.index(st.session_state["gbg_team_b"]) if st.session_state.get("gbg_team_b") in team_b_options else 0,
        key="gbg_team_b",
        format_func=slug_to_display_name,
        ),
    )

btn_cols = st.columns([1, 5])
with btn_cols[0]:
    st.button(
        "Flip teams",
        use_container_width=True,
        on_click=_flip_teams_callback,
        kwargs={"year": year},
    )

# Keep URL in sync for shareable links.
try:
    _set_query_params(team_a=str(team_a), team_b=str(team_b), year=str(year))
except Exception:
    pass

p_a = get_matchup_probability(team_a, team_b, probs)
conf = confidence_from_probability(p_a)

st.subheader("Matchup")
st.write(f"P({slug_to_display_name(team_a)} beats {slug_to_display_name(team_b)}) = **{p_a:.3f}**")
st.write(
    "Confidence (0-1 proxy) = "
    f"**{conf:.3f}**  "
    "(computed as `2 * abs(p - 0.5)`; this is *not* a statistical uncertainty interval)"
)
st.caption(
    "How to read Confidence: it is just the distance-from-coinflip rescaled to 0–1. "
    "If $p=0.50$ then confidence = 0.00; if $p=0.60$ then confidence = 0.20; "
    "if $p=0.90$ then confidence = 0.80. It does *not* measure variance across models."
)

st.subheader("Context")
st.caption(
    "These charts use the precomputed matchup probabilities JSON (the same source as the headline probability above)."
)

opp_list = [t for t in teams if t != team_a]
team_a_vs_field = pd.DataFrame(
    {
        "opponent": opp_list,
        "p_team_a_wins": [get_matchup_probability(team_a, opp, probs) for opp in opp_list],
    }
)

if alt is not None and not team_a_vs_field.empty:
    hist = (
        alt.Chart(team_a_vs_field)
        .mark_bar()
        .encode(
            x=alt.X("p_team_a_wins:Q", bin=alt.Bin(maxbins=20), title="P(Team A beats opponent)"),
            y=alt.Y("count()", title="# opponents"),
            tooltip=[alt.Tooltip("count()", title="# opponents")],
        )
        .properties(height=140)
    )
    rule = alt.Chart(pd.DataFrame({"p_team_a_wins": [p_a]})).mark_rule(color="red").encode(
        x="p_team_a_wins:Q"
    )
    st.altair_chart(hist + rule, use_container_width=True)
else:
    st.dataframe(
        team_a_vs_field.sort_values("p_team_a_wins", ascending=False).head(15),
        use_container_width=True,
        hide_index=True,
    )

# Optional: show precomputed feature drivers from the explainability model
try:
    expl = load_matchup_explanations(year)
except FileNotFoundError:
    expl = {}

if expl:
    row = (expl.get(team_a, {}) or {}).get(team_b)

    if row:
        st.subheader("Drivers (precomputed)")
        st.caption(
            "The tables below come from `best_xgb_model.pkl` using XGBoost’s `pred_contribs` (SHAP-style) values. "
            "The `contrib` numbers are **additive contributions to the model’s log-odds** for Team A winning."
        )
        st.markdown(
            "- **Sign**: `contrib > 0` pushes Team A toward winning; `contrib < 0` pushes toward losing.\n"
            "- **Magnitude**: contributions are in *log-odds* units (not probability points). A change of $+x$ "
            "multiplies the odds by $e^x$. Example: `contrib = +0.69` roughly doubles the odds ($e^{0.69}\approx2$).\n"
            "- **Additivity**: $\\text{logit}(p) = \\text{base\\_value} + \\sum_i \\text{contrib}_i$ (over all features).\n"
            "- **value column**: the feature value the explainability model saw *after* the same preprocessing used for inference "
            "(z-scoring + imputation), so it may not match the raw stat scale."
        )

        st.write(f"Explainability model P(Team A wins) = **{float(row.get('proba', 0.0)):.3f}**")

        # Optional chart: contribution bars for the selected matchup
        pos = list(row.get("top_positive", []) or [])
        neg = list(row.get("top_negative", []) or [])
        contrib_df = pd.DataFrame(pos + neg)
        if not contrib_df.empty:
            contrib_df["sign"] = contrib_df["contrib"].apply(lambda x: "helps" if float(x) >= 0 else "hurts")
            contrib_df = contrib_df.sort_values("contrib")

            if alt is not None:
                chart = (
                    alt.Chart(contrib_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("contrib:Q", title="contrib (log-odds units)"),
                        y=alt.Y("feature:N", sort=None, title="feature"),
                        color=alt.Color(
                            "sign:N",
                            scale=alt.Scale(domain=["hurts", "helps"], range=["#d62728", "#2ca02c"]),
                            legend=alt.Legend(title="effect"),
                        ),
                        tooltip=[
                            alt.Tooltip("feature:N"),
                            alt.Tooltip("value:Q", format=".3f"),
                            alt.Tooltip("contrib:Q", format=".3f"),
                        ],
                    )
                    .properties(height=min(360, 22 * len(contrib_df)))
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.dataframe(contrib_df, use_container_width=True, hide_index=True)

        colp, coln = st.columns(2)
        with colp:
            st.markdown("**Top positive (helps Team A)**")
            st.dataframe(
                row.get("top_positive", []),
                use_container_width=True,
                hide_index=True,
            )
        with coln:
            st.markdown("**Top negative (hurts Team A)**")
            st.dataframe(
                row.get("top_negative", []),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("Explanations JSON is present, but no entry exists for this matchup.")
else:
    st.info(
        "No precomputed explanations found for this year. "
        "Generate them with: `python Scripts/generate_matchup_explanations.py --year <YEAR>`"
    )
