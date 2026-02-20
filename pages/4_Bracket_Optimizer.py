from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from mm_app.bracket import ROUND1_PAIRINGS
from mm_app.load import load_bracket_field, load_matchup_probs, load_optimizer_sims, load_optimizer_top25
from mm_app.paths import get_output_paths
from mm_app.optimizer import (
    decode_winners,
    make_plan,
    random_search,
    select_diverse_topk,
    simulate_many,
)
from mm_app.util import slug_to_display_name


st.set_page_config(page_title="Bracket Optimizer", layout="wide")
st.title("Simulation-Based Bracket Optimizer")

DEFAULT_PUBLIC_YEAR = 2024
st.session_state.setdefault("year", DEFAULT_PUBLIC_YEAR)
year = int(st.session_state.get("year", DEFAULT_PUBLIC_YEAR))
paths = get_output_paths(year)

st.caption(
    "Generates candidate brackets and ranks them by expected ESPN-style score against Monte Carlo tournament outcomes. "
    "Uses `bracket_field_YYYY.json` + `matchup_probabilities_YYYY.json`. "
    "Optional: `optimizer_sims_YYYY.json` for faster iteration and reproducibility."
)

with st.expander("How the Monte Carlo optimizer works", expanded=False):
    st.markdown(
        """
This page uses **Monte Carlo tournament simulation** to approximate the *expected ESPN-style bracket score* for a candidate bracket.

**Step 1 — Simulate many tournament outcomes**
- Using `matchup_probabilities_YYYY.json`, we simulate the entire 63-game tournament many times.
- Each simulation produces one complete set of winners (Round of 64 → Champion).

**Step 2 — Generate candidate brackets**
- We generate many candidate brackets by simulating picks using the same matchup probabilities.
- The **Leverage** slider only affects *how candidates are generated*:
  - `0.0` is more “chalky” (leans toward favorites).
  - higher values produce more contrarian brackets (more underdogs).

**Step 3 — Score each candidate against the simulated outcomes**
- For a candidate bracket, we compute an ESPN-style score vs **each** simulated tournament outcome.
- This gives a *distribution of scores* for that candidate.

**Step 4 — Rank brackets**
- `expected_score` = mean score across simulations (proxy for “how good on average”).
- `std_score` = standard deviation of score (proxy for “high-variance / boom-bust”).

Important note: “optimized” here means **highest expected ESPN points under the model**, not “most likely to be perfect”.
        """
    )

with st.expander("What each setting does (and what changes champion variety)", expanded=False):
        st.markdown(
                """
**Tournament sims**
- Controls how many full tournament outcomes are simulated for *scoring*.
- More sims → more stable `expected_score` estimates (less Monte Carlo noise), but slower.
- This does **not** directly force more/different champions; it just makes the ranking more reliable.

**Candidate brackets**
- Controls how many candidate brackets we generate and evaluate.
- More candidates → better chance of finding high-scoring brackets *and* naturally discovering more distinct champion picks.

**Leverage (chalk → contrarian)**
- Only affects how candidate brackets are generated.
- Higher leverage makes underdogs more likely to be picked in candidates.
- This often increases champion variety (more long-shot champions appear), but can also reduce `expected_score`.

**Random seed**
- Controls the RNG for both candidate generation (and sims if generated on the fly).
- Changing the seed can change which specific brackets (and champions) show up, especially when `Candidate brackets` is small.

---

**Champion diversity (display selector)**
This controls **how we choose the displayed Top 25** from the sorted list of evaluated brackets.

- **None (pure top-K)**: shows the top 25 highest `expected_score` brackets. This often results in many repeats of the same champion.
- **Limit per champion**: enforces a cap (e.g., max 5 brackets per champion) and then fills the list in score order.
    - The *maximum* distinct champions you can see is up to `ceil(25 / max_per_champ)` if enough good champions exist.
- **Top champions × brackets**: first picks the top `Top champions` champions (by their best bracket), then includes up to `Brackets per champion` for each.
    - This is the most direct way to “ensure” multiple different champions show up.

**Max per champion**
- Only used when **Limit per champion** is selected.
- Smaller value → forces more champion variety, but may skip some high-scoring near-duplicates.

**Top champions**
- Only used when **Top champions × brackets** is selected.
- Larger value → allows more distinct champions (if enough exist in the candidate pool).

**Brackets per champion**
- Only used when **Top champions × brackets** is selected.
- Larger value → fewer distinct champions (because each champion gets more slots).

---

**Precomputed vs live**
- If you’re viewing **precomputed** top-25 (`optimizer_top25_YYYY.json`), the file contains a fixed set of brackets generated earlier.
    - The diversity controls can only *filter/select* from what’s already in that file — they can’t create new champions that weren’t generated.
- If you click **Run optimizer**, the diversity controls apply to the freshly-generated candidate results.
    - If you want *more* champion options in the live view: increase `Candidate brackets` and/or increase `Leverage`.
    - If you want *more* champion options in the precomputed view: regenerate the precompute with a bigger search (and optionally diversity enabled).
                """
        )

st.write("Input file status:")
st.write(f"- {paths.bracket_field_json.name}: local_exists={paths.bracket_field_json.exists()}")
st.write(f"- {paths.matchup_probs_json.name}: local_exists={paths.matchup_probs_json.exists()}")
st.write(f"- {paths.optimizer_sims_json.name} (optional): local_exists={paths.optimizer_sims_json.exists()}")
st.write(f"- {paths.optimizer_top25_json.name} (optional): local_exists={paths.optimizer_top25_json.exists()}")

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
plan = make_plan(bracket_field)

st.subheader("Settings")
col1, col2, col3 = st.columns(3)
with col1:
    n_sims = int(st.number_input("Tournament sims", min_value=200, max_value=50000, value=5000, step=500))
with col2:
    n_candidates = int(st.number_input("Candidate brackets", min_value=50, max_value=20000, value=2000, step=250))
with col3:
    leverage = float(
        st.slider(
            "Leverage (chalk → contrarian)",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
        )
    )

seed = int(st.number_input("Random seed", min_value=0, max_value=10_000_000, value=7, step=1))

st.subheader("Top-25 display")
dc1, dc2, dc3 = st.columns(3)
with dc1:
    diversity_mode = st.selectbox(
        "Champion diversity",
        options=["None (pure top-K)", "Limit per champion", "Top champions × brackets"],
        index=1,
    )
with dc2:
    max_per_champ = int(st.number_input("Max per champion (if enabled)", min_value=1, max_value=25, value=5, step=1))
with dc3:
    top_champs = int(st.number_input("Top champions (if enabled)", min_value=1, max_value=25, value=10, step=1))
per_champ = int(st.number_input("Brackets per champion (Top champions mode)", min_value=1, max_value=25, value=3, step=1))

mode_map = {
    "None (pure top-K)": "none",
    "Limit per champion": "cap",
    "Top champions × brackets": "top_champs",
}
mode_key = mode_map.get(diversity_mode, "none")

try:
    _ = load_optimizer_sims(year)
    use_precomputed = True
except FileNotFoundError:
    use_precomputed = False

if use_precomputed:
    st.info(
        f"Using precomputed simulations from {paths.optimizer_sims_json.name}. "
        "(To regenerate: run Scripts/generate_optimizer_sims.py)"
    )

run_clicked = st.button("Run optimizer", type="primary")


def _normalize_topk(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in records:
        picks_idx = r.get("picks_idx", [])
        picks_idx_list = [int(x) for x in np.asarray(picks_idx, dtype=np.int16).tolist()]
        out.append(
            {
                **r,
                "expected_score": float(r.get("expected_score", 0.0)),
                "std_score": float(r.get("std_score", 0.0)),
                "champ": str(r.get("champ", "")),
                "picks_idx": picks_idx_list,
            }
        )
    return out


def _live_key(year: int) -> str:
    return f"live_optimizer_topk_{int(year)}"


live_state_key = _live_key(year)
live_topk: list[dict[str, Any]] = list(st.session_state.get(live_state_key, []) or [])

if live_topk:
    st.info("Showing results from your most recent optimizer run (live).")
    if st.button("Clear live results"):
        st.session_state.pop(live_state_key, None)
        live_topk = []

if (not run_clicked) and (not live_topk):
    try:
        payload = load_optimizer_top25(year)
    except FileNotFoundError:
        payload = {}

if payload and (not run_clicked) and (not live_topk):
    st.subheader("Top brackets (precomputed)")
    pre = list(payload.get("brackets", []) or [])
    pre_settings = payload.get("settings", {}) or {}
    if pre_settings:
        st.caption(
            "Precompute settings: "
            f"n_sims={pre_settings.get('n_sims')}, "
            f"n_candidates={pre_settings.get('n_candidates')}, "
            f"leverage={pre_settings.get('leverage')}, "
            f"seed={pre_settings.get('seed')}"
        )

    pre_teams = list(payload.get("teams", []) or [])
    if pre_teams and pre_teams != plan.teams:
        st.warning(
            "Precomputed top25 file was generated for a different field/team list. "
            "Run the optimizer live (or regenerate the precompute file) for this year."
        )
    if not pre:
        st.info("Precomputed file exists but contains no brackets.")
    else:
        pre = select_diverse_topk(
            pre,
            top_k=min(25, len(pre)),
            mode=mode_key,
            max_per_champ=max_per_champ,
            top_champs=top_champs,
            per_champ=per_champ,
        )
        df_pre = pd.DataFrame(
            [
                {
                    "rank": int(b.get("rank", i + 1)),
                    "expected_score": round(float(b.get("expected_score", 0.0)), 2),
                    "std_score": round(float(b.get("std_score", 0.0)), 2),
                    "champion_pick": slug_to_display_name(str(b.get("champ", ""))),
                }
                for i, b in enumerate(pre)
            ]
        )
        st.dataframe(df_pre, use_container_width=True, hide_index=True)

        st.subheader("Bracket details (precomputed)")
        pick_rank = int(
            st.number_input(
                "Show bracket rank",
                min_value=1,
                max_value=len(pre),
                value=1,
                step=1,
                key="pre_rank",
            )
        )
        chosen = pre[pick_rank - 1]
        picks_idx = chosen.get("picks_idx", [])
        picks = decode_winners(plan, picks_idx)

        sizes = [32, 16, 8, 4, 2, 1]
        labels = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Champion"]
        start = 0
        for size, label in zip(sizes, labels, strict=True):
            end = start + size
            round_teams = [slug_to_display_name(t) for t in picks[start:end]]
            st.markdown(f"**{label} winners (your picks)**")
            st.write(", ".join(round_teams))
            start = end

if run_clicked:
    with st.spinner("Preparing simulations..."):
        if use_precomputed:
            sims_payload = load_optimizer_sims(year)
            teams = list(sims_payload.get("teams", []))
            winners = np.asarray(sims_payload.get("winners", []), dtype=np.int16)
            plan = make_plan(bracket_field)
            if teams and teams != plan.teams:
                st.warning(
                    "Precomputed sims team list does not match the current bracket_field team list. "
                    "Falling back to fresh simulations."
                )
                plan, winners = simulate_many(
                    bracket_field=bracket_field,
                    probs=probs,
                    n_sims=n_sims,
                    leverage=0.0,
                    seed=seed,
                    round1_pairings=ROUND1_PAIRINGS,
                )
        else:
            plan, winners = simulate_many(
                bracket_field=bracket_field,
                probs=probs,
                n_sims=n_sims,
                leverage=0.0,
                seed=seed,
                round1_pairings=ROUND1_PAIRINGS,
            )

    with st.spinner("Searching candidate brackets..."):
        results = random_search(
            bracket_field=bracket_field,
            probs=probs,
            sims=winners,
            plan=plan,
            n_candidates=n_candidates,
            leverage=leverage,
            seed=seed,
            round1_pairings=ROUND1_PAIRINGS,
        )

    st.subheader("Top brackets")
    topk = select_diverse_topk(
        results,
        top_k=min(25, len(results)),
        mode=mode_key,
        max_per_champ=max_per_champ,
        top_champs=top_champs,
        per_champ=per_champ,
    )
    topk = _normalize_topk(topk)
    st.session_state[live_state_key] = topk
    live_topk = topk


if live_topk:
    st.subheader("Top brackets (live)")
    topk = live_topk
    df = pd.DataFrame(
        [
            {
                "rank": i + 1,
                "expected_score": round(r["expected_score"], 2),
                "std_score": round(r["std_score"], 2),
                "champion_pick": slug_to_display_name(str(r["champ"])),
            }
            for i, r in enumerate(topk)
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    if topk:
        st.subheader("Bracket details (live)")
        pick_rank = int(
            st.number_input(
                "Show bracket rank",
                min_value=1,
                max_value=len(topk),
                value=1,
                step=1,
                key="live_rank",
            )
        )
        chosen = topk[pick_rank - 1]
        picks_idx = chosen["picks_idx"]
        picks = decode_winners(plan, picks_idx)

        sizes = [32, 16, 8, 4, 2, 1]
        labels = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Champion"]
        start = 0
        for size, label in zip(sizes, labels, strict=True):
            end = start + size
            round_teams = [slug_to_display_name(t) for t in picks[start:end]]
            st.markdown(f"**{label} winners (your picks)**")
            st.write(", ".join(round_teams))
            start = end
