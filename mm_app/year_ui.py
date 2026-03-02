from __future__ import annotations

from typing import Optional

import streamlit as st


def render_year_sidebar(*, default_year: int, label: str = "Year") -> int:
    """Render a consistent year selector in the sidebar and return the selected year.

    Why this exists:
    - In Streamlit multi-page apps, widget-backed session state can be cleared
      when a widget key doesn't exist on the active page.
    - To keep the selected year persistent across page navigation, every page
      should render the same sidebar selectbox with key='year'.
    """

    try:
        from .private_data import list_available_output_years

        years = list_available_output_years()
    except Exception:
        years = []

    if not years:
        years = [int(default_year)]

    default_year = int(default_year)
    if default_year not in years:
        default_year = max(years)

    st.session_state.setdefault("year", default_year)

    # If the current selection isn't valid for this deployment, snap to default.
    try:
        current = int(st.session_state.get("year", default_year))
    except Exception:
        current = default_year

    if current not in years:
        current = default_year
        st.session_state["year"] = default_year

    selected = st.sidebar.selectbox(
        label,
        options=years,
        index=years.index(int(current)),
        key="year",
    )

    # IMPORTANT: Do not write to st.session_state["year"] after the widget
    # is instantiated. The widget owns that key for this run.
    try:
        return int(selected)
    except Exception:
        return int(default_year)
