from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mm_app.load import load_cluster_data

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(page_title="Team Clusters", layout="wide")
st.title("Team Clusters — Tournament Team Archetypes")
st.markdown(
    "Explore how NCAA tournament teams cluster based on season-long stats. "
    "Teams are z-score normalized **within each year** so profiles represent "
    "relative strengths, not era-level differences."
)

# ── Load data ────────────────────────────────────────────────────────
try:
    raw = load_cluster_data()
except FileNotFoundError:
    st.error(
        "Missing `cluster_all_teams.json`.  Run notebook **04_Cluster_Modeling** "
        "to generate this artifact, then commit it to the private repo."
    )
    st.stop()

meta = raw.get("metadata", {})
cluster_info = raw.get("cluster_info", {})
teams_list = raw.get("teams", [])

if not teams_list:
    st.warning("Clustering data is empty.")
    st.stop()

df = pd.DataFrame(teams_list)
df["cluster"] = df["cluster"].astype(str)

# Friendly labels
RESULT_ORDER = meta.get("result_labels", sorted(df["result"].unique()))
RESULT_COLORS = meta.get("result_colors", {})
LATEST_YEAR = meta.get("latest_year", int(df["year"].max()))

# ── Sidebar filters ──────────────────────────────────────────────────
st.sidebar.header("Filters")

all_years = sorted(df["year"].unique())
sel_years = st.sidebar.multiselect(
    "Years",
    options=all_years,
    default=all_years,
    help="Select tournament years to display",
)

all_seeds = sorted(df["seed"].unique())
seed_range = st.sidebar.slider(
    "Seed range",
    min_value=int(min(all_seeds)),
    max_value=int(max(all_seeds)),
    value=(int(min(all_seeds)), int(max(all_seeds))),
)

all_clusters = sorted(df["cluster"].unique(), key=int)
sel_clusters = st.sidebar.multiselect(
    "Clusters",
    options=all_clusters,
    default=all_clusters,
    format_func=lambda c: f"Cluster {c}",
)

all_results = [r for r in RESULT_ORDER if r in df["result"].values]
sel_results = st.sidebar.multiselect(
    "Tournament Result",
    options=all_results,
    default=all_results,
)

# ── Apply filters ────────────────────────────────────────────────────
mask = (
    df["year"].isin(sel_years)
    & df["seed"].between(seed_range[0], seed_range[1])
    & df["cluster"].isin(sel_clusters)
    & df["result"].isin(sel_results)
)
dff = df[mask].copy()

if dff.empty:
    st.warning("No teams match the current filters. Adjust sidebar filters.")
    st.stop()

st.sidebar.markdown(f"**{len(dff):,}** teams shown")

# ── Color-by selector ────────────────────────────────────────────────
COLOR_OPTIONS = {
    "Cluster": "cluster",
    "Seed": "seed",
    "Tournament Result": "result",
    "Year": "year",
}
color_choice = st.radio(
    "Color dots by",
    list(COLOR_OPTIONS.keys()),
    index=0,
    horizontal=True,
)
color_col = COLOR_OPTIONS[color_choice]

# Build hover data available in all charts
hover_cols = {
    "team": True,
    "year": True,
    "seed": True,
    "cluster": True,
    "result": True,
    "wins": True,
    "off_z": ":.2f",
    "def_z": ":.2f",
}


def _color_kwargs(col: str, dff: pd.DataFrame) -> dict:
    """Return plotly express color kwargs depending on the chosen column type."""
    if col == "cluster":
        return dict(color="cluster", color_discrete_sequence=px.colors.qualitative.T10)
    if col == "result":
        cats = [r for r in RESULT_ORDER if r in dff["result"].values]
        colors = [RESULT_COLORS.get(r, "#999") for r in cats]
        return dict(
            color="result",
            category_orders={"result": cats},
            color_discrete_sequence=colors,
        )
    if col == "seed":
        return dict(color="seed", color_continuous_scale="RdYlGn_r")
    if col == "year":
        return dict(color="year", color_continuous_scale="Viridis")
    return dict(color=col)


# =====================================================================
# TAB LAYOUT
# =====================================================================
tab_quad, tab_umap, tab_profile = st.tabs(
    ["Offense / Defense Quadrant", "UMAP Scatter", "Cluster Profiles"]
)

# ── TAB 1: Quadrant chart ────────────────────────────────────────────
with tab_quad:
    st.subheader("Offense vs Defense Quadrant")
    st.markdown(
        "**X-axis** = composite offensive z-score &nbsp;|&nbsp; "
        "**Y-axis** = composite defensive z-score (higher = better).  "
        "Crosshairs at (0, 0) = year-average."
    )

    fig_quad = px.scatter(
        dff,
        x="off_z",
        y="def_z",
        hover_name="team",
        hover_data=hover_cols,
        opacity=0.6,
        **_color_kwargs(color_col, dff),
    )
    fig_quad.update_traces(marker_size=6)

    # Quadrant lines
    fig_quad.add_hline(y=0, line_dash="dash", line_color="grey", line_width=0.8)
    fig_quad.add_vline(x=0, line_dash="dash", line_color="grey", line_width=0.8)

    # Quadrant labels
    x_range = [dff["off_z"].min() - 0.1, dff["off_z"].max() + 0.1]
    y_range = [dff["def_z"].min() - 0.1, dff["def_z"].max() + 0.1]
    annotations = [
        dict(x=x_range[1], y=y_range[1], text="ELITE", showarrow=False,
             font=dict(size=13, color="green"), opacity=0.4, xanchor="right", yanchor="top"),
        dict(x=x_range[0], y=y_range[1], text="DEFENSIVE", showarrow=False,
             font=dict(size=13, color="blue"), opacity=0.4, xanchor="left", yanchor="top"),
        dict(x=x_range[1], y=y_range[0], text="OFFENSIVE", showarrow=False,
             font=dict(size=13, color="orange"), opacity=0.4, xanchor="right", yanchor="bottom"),
        dict(x=x_range[0], y=y_range[0], text="BELOW AVG", showarrow=False,
             font=dict(size=13, color="red"), opacity=0.4, xanchor="left", yanchor="bottom"),
    ]
    fig_quad.update_layout(
        annotations=annotations,
        xaxis_title="Offensive Quality (composite z-score) →",
        yaxis_title="Defensive Quality (composite z-score) →",
        height=650,
        template="plotly_white",
    )
    st.plotly_chart(fig_quad, use_container_width=True)

    # ── Current-year spotlight ───────────────────────────────────────
    curr = dff[dff["year"] == LATEST_YEAR]
    if not curr.empty:
        with st.expander(f"Highlight {LATEST_YEAR} teams on the quadrant", expanded=False):
            fig_curr = go.Figure()
            hist = dff[dff["year"] != LATEST_YEAR]
            if not hist.empty:
                fig_curr.add_trace(go.Scatter(
                    x=hist["off_z"], y=hist["def_z"],
                    mode="markers",
                    marker=dict(size=4, color="#CCCCCC", opacity=0.3),
                    name="Historical",
                    hoverinfo="skip",
                ))
            fig_curr.add_trace(go.Scatter(
                x=curr["off_z"], y=curr["def_z"],
                mode="markers+text",
                marker=dict(size=9, color="crimson", symbol="diamond",
                            line=dict(width=0.5, color="black")),
                text=curr["team"],
                textposition="top center",
                textfont=dict(size=8),
                name=str(LATEST_YEAR),
                customdata=curr[["team", "seed", "cluster", "result"]].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Seed: %{customdata[1]}<br>"
                    "Cluster: %{customdata[2]}<br>"
                    "Off z: %{x:.2f}  |  Def z: %{y:.2f}"
                    "<extra></extra>"
                ),
            ))
            fig_curr.add_hline(y=0, line_dash="dash", line_color="grey", line_width=0.8)
            fig_curr.add_vline(x=0, line_dash="dash", line_color="grey", line_width=0.8)
            fig_curr.update_layout(
                xaxis_title="Offensive Quality →",
                yaxis_title="Defensive Quality →",
                height=600,
                template="plotly_white",
                title=f"{LATEST_YEAR} Teams vs Historical Field",
            )
            st.plotly_chart(fig_curr, use_container_width=True)

# ── TAB 2: UMAP scatter ─────────────────────────────────────────────
with tab_umap:
    st.subheader("UMAP 2-D Projection")
    st.markdown(
        "UMAP reduces ~180 features into 2 abstract dimensions that preserve "
        "neighborhood structure.  Distance between points is only meaningful "
        "locally — nearby teams are genuinely similar."
    )

    fig_umap = px.scatter(
        dff,
        x="x",
        y="y",
        hover_name="team",
        hover_data=hover_cols,
        opacity=0.6,
        **_color_kwargs(color_col, dff),
    )
    fig_umap.update_traces(marker_size=6)
    fig_umap.update_layout(
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        height=650,
        template="plotly_white",
    )
    st.plotly_chart(fig_umap, use_container_width=True)

# ── TAB 3: Cluster profiles ─────────────────────────────────────────
with tab_profile:
    st.subheader("Cluster Profiles")
    st.markdown("Top distinguishing features per cluster (by mean z-score deviation from the field).")

    n_clusters = meta.get("n_clusters", len(cluster_info))

    # Summary table
    summary_rows = []
    for cid in sorted(cluster_info, key=int):
        info = cluster_info[cid]
        summary_rows.append({
            "Cluster": int(cid),
            "Teams": info.get("size", ""),
            "Avg Seed": info.get("avg_seed", ""),
            "Avg Wins (historical)": info.get("avg_wins_historical", ""),
        })
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows).set_index("Cluster"), use_container_width=True)

    # Per-cluster trait expanders
    for cid in sorted(cluster_info, key=int):
        info = cluster_info[cid]
        traits = info.get("top_traits", [])
        with st.expander(f"Cluster {cid}  ({info.get('size', '?')} teams, avg seed {info.get('avg_seed', '?')})"):
            if traits:
                trait_df = pd.DataFrame(traits)
                trait_df.columns = ["Feature", "Mean Z-Score", "Direction"]
                trait_df["Mean Z-Score"] = trait_df["Mean Z-Score"].apply(lambda v: f"{v:+.3f}")
                st.table(trait_df)
            else:
                st.write("No trait data available.")

            # Show example teams in this cluster
            cl_teams = dff[dff["cluster"] == cid].sort_values("seed")
            if not cl_teams.empty:
                st.markdown(f"**Sample teams** ({len(cl_teams)} match filters):")
                st.dataframe(
                    cl_teams[["team", "year", "seed", "result"]].head(20).reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                )

# ── Metadata footer ──────────────────────────────────────────────────
with st.expander("About this analysis"):
    st.markdown(f"""
- **Clustering method:** {meta.get('clustering_method', 'KMeans')} on PCA-reduced features
- **PCA components:** {meta.get('pca_components', '?')} (explains {meta.get('pca_variance_explained', '?'):.1%} variance)
- **Silhouette score:** {meta.get('silhouette_score', '?')}
- **Features used:** {meta.get('features_used', '?')} season-long stats (z-scored by year)
- **Years covered:** {min(meta.get('years', [0]))}–{max(meta.get('years', [0]))} ({len(meta.get('years', []))} seasons)
- **Offense composite:** {', '.join(meta.get('offense_features', []))}
- **Defense composite (negated):** {', '.join(meta.get('defense_features_negated', []))}
- **Defense composite (positive):** {', '.join(meta.get('defense_features_positive', []))}
""")
