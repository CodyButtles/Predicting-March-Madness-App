from __future__ import annotations

import pandas as pd
import streamlit as st

from mm_app.load import load_advancement_probs, load_bracket_field
from mm_app.paths import get_output_paths
from mm_app.util import slug_to_display_name
from mm_app.year_ui import render_year_sidebar

st.set_page_config(page_title="Advancement Odds Table", layout="wide")
st.title("Team Advancement Odds")
st.caption(
    "Round-by-round advancement probabilities for every team in the field, "
    "derived from thousands of bracket simulations."
)

DEFAULT_PUBLIC_YEAR = 2026
year = render_year_sidebar(default_year=DEFAULT_PUBLIC_YEAR)

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

# ── Region display-name mapping ─────────────────────────────────────────────
REGION_LABELS: dict[str, str] = {
    "UL": "Upper Left",
    "UR": "Upper Right",
    "LL": "Lower Left",
    "LR": "Lower Right",
}

# ── Build a flat list of {team_slug, seed, region_key} ──────────────────────
team_meta: list[dict] = []
for region_key, entries in bracket_field.items():
    label = REGION_LABELS.get(region_key, region_key)
    for entry in entries:
        team_meta.append(
            {
                "team_slug": entry["team"],
                "seed": int(entry["seed"]),
                "region_key": region_key,
                "region": label,
            }
        )

# ── Detect which round columns are present in the data ──────────────────────
# Round keys expected (ordered), with display labels.
ALL_ROUND_KEYS = [
    ("FF",    "First Four %"),
    ("R32",   "Round of 32 %"),
    ("S16",   "Sweet 16 %"),
    ("E8",    "Elite 8 %"),
    ("F4",    "Final Four %"),
    ("Final", "Champ. Game %"),
    ("Champ", "Champion %"),
]
# Only include rounds that actually appear in any team's adv entry.
all_adv_keys: set[str] = set()
for v in adv.values():
    all_adv_keys.update(v.keys())
present_rounds = [(k, lbl) for k, lbl in ALL_ROUND_KEYS if k in all_adv_keys]

# ── Assemble DataFrame ───────────────────────────────────────────────────────
rows: list[dict] = []
for meta in team_meta:
    slug = meta["team_slug"]
    team_adv = adv.get(slug, {})
    row: dict = {
        "Team": slug_to_display_name(slug),
        "Seed": meta["seed"],
        "Region": meta["region"],
    }
    for key, col_label in present_rounds:
        row[col_label] = team_adv.get(key, 0.0)
    rows.append(row)

df = pd.DataFrame(rows)
# Sort default: seed asc, then team name
df = df.sort_values(["Seed", "Team"]).reset_index(drop=True)

# ── Sidebar filters ──────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

available_regions = sorted(df["Region"].unique())
selected_regions = st.sidebar.multiselect(
    "Region",
    options=available_regions,
    default=available_regions,
)

seed_min, seed_max = int(df["Seed"].min()), int(df["Seed"].max())
selected_seeds = st.sidebar.slider(
    "Seed range",
    min_value=seed_min,
    max_value=seed_max,
    value=(seed_min, seed_max),
)

# Minimum probability threshold applied to a chosen round
prob_cols = [lbl for _, lbl in present_rounds]
filter_round = st.sidebar.selectbox(
    "Min % filter — round",
    options=["(none)"] + prob_cols,
)
filter_min_pct = st.sidebar.slider(
    "Min % threshold",
    min_value=0,
    max_value=100,
    value=0,
    step=1,
    disabled=(filter_round == "(none)"),
)

team_search = st.sidebar.text_input("Search team name", "")

# ── Apply filters ────────────────────────────────────────────────────────────
mask = (
    df["Region"].isin(selected_regions)
    & df["Seed"].between(selected_seeds[0], selected_seeds[1])
)
if filter_round != "(none)" and filter_min_pct > 0:
    mask &= df[filter_round] >= filter_min_pct
if team_search.strip():
    mask &= df["Team"].str.contains(team_search.strip(), case=False, na=False)

filtered = df[mask].copy()

# ── Sort control ─────────────────────────────────────────────────────────────
sort_options = ["Seed", "Team", "Region"] + prob_cols
sort_col = st.selectbox(
    "Sort by",
    options=sort_options,
    index=sort_options.index("Champion %") if "Champion %" in sort_options else 0,
    horizontal=True,
    label_visibility="collapsed",
    key="adv_sort_col",
)
sort_asc = st.checkbox("Sort ascending", value=False, key="adv_sort_asc")
filtered = filtered.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

# ── Summary row ──────────────────────────────────────────────────────────────
num_teams = len(filtered)
st.markdown(
    f"Showing **{num_teams}** team{'s' if num_teams != 1 else ''} "
    f"across **{filtered['Region'].nunique()}** region(s)."
)

# ── Styled table ─────────────────────────────────────────────────────────────
def _fmt_pct(val: float) -> str:
    """Format as '12.3%' or '—' for zero/missing."""
    if val is None or (isinstance(val, float) and val == 0.0):
        return "—"
    return f"{val:.1f}%"


pct_col_labels = [lbl for _, lbl in present_rounds]

# Format display copy
display_df = filtered.copy()
for col in pct_col_labels:
    if col in display_df.columns:
        display_df[col] = display_df[col].apply(_fmt_pct)

# Seed as string so it doesn't render with commas
display_df["Seed"] = display_df["Seed"].astype(str)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Team": st.column_config.TextColumn("Team", width="medium"),
        "Seed": st.column_config.TextColumn("Seed", width="small"),
        "Region": st.column_config.TextColumn("Region", width="medium"),
        **{
            col: st.column_config.TextColumn(col, width="small")
            for col in pct_col_labels
            if col in display_df.columns
        },
    },
)

# ── Bar-chart quick view ─────────────────────────────────────────────────────
st.markdown("---")
with st.expander("Championship odds — bar chart", expanded=False):
    champ_col = "Champion %"
    if champ_col in filtered.columns:
        chart_df = (
            filtered[["Team", "Seed", champ_col]]
            .rename(columns={champ_col: "Champ %"})
            .sort_values("Champ %", ascending=False)
            .head(30)
        )
        chart_df["Label"] = chart_df.apply(
            lambda r: f"({r['Seed']}) {r['Team']}", axis=1
        )
        st.bar_chart(
            chart_df.set_index("Label")["Champ %"],
            use_container_width=True,
        )
        st.caption("Top 30 teams by championship probability (filtered view).")

# ── Per-region champion probability ─────────────────────────────────────────
with st.expander("Region champion probability breakdown", expanded=False):
    f4_col = "Final Four %"
    if f4_col in filtered.columns:
        region_summary = (
            filtered.groupby("Region")[f4_col]
            .sum()
            .reset_index()
            .rename(columns={f4_col: "Total Final Four %"})
            .sort_values("Total Final Four %", ascending=False)
        )
        st.markdown(
            "Sum of Final Four % across teams in each region "
            "(filtered view — should total ~200 % for the full field)."
        )
        st.dataframe(region_summary, use_container_width=True, hide_index=True)

# ── Upset value spotlight ────────────────────────────────────────────────────
with st.expander("Upset value spotlight (high odds for low seeds)", expanded=False):
    champ_col = "Champion %"
    if champ_col in df.columns:
        upset_df = df[df["Seed"] >= 9].copy()
        upset_df = upset_df.sort_values(champ_col, ascending=False).head(10)
        upset_df["Label"] = upset_df.apply(
            lambda r: f"({r['Seed']}) {r['Team']} — {r['Region']}", axis=1
        )
        upset_df_display = upset_df[["Label"] + prob_cols].copy()
        for col in prob_cols:
            if col in upset_df_display.columns:
                upset_df_display[col] = upset_df_display[col].apply(_fmt_pct)
        st.markdown(
            "Top 10 **double-digit or lower seeds** by championship probability — "
            "teams where the model sees meaningful upset potential."
        )
        st.dataframe(
            upset_df_display.rename(columns={"Label": "Team (Seed — Region)"}),
            use_container_width=True,
            hide_index=True,
        )
