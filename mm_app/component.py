from __future__ import annotations

import json
from pathlib import Path

import streamlit.components.v1 as components

from .bracket import build_round1_games


def render_bracket_builder(*, bracket_field: dict, advancement_probs: dict, matchup_probs: dict | None, year: int) -> None:
    """Render the interactive JS bracket builder.

    All interactions (click-to-advance, tooltips) are handled client-side in JS.
    """
    payload = {
        "year": int(year),
        "regions": [
            {
                "key": key,
                "label": key,
                "round1": build_round1_games(bracket_field[key]),
                "seeds": bracket_field[key],
            }
            for key in ["UL", "UR", "LL", "LR"]
            if key in bracket_field
        ],
        "adv": advancement_probs,
        "matchup": matchup_probs or {},
    }

    root = Path(__file__).resolve().parents[1]
    css_path = root / "frontend" / "bracket_builder.css"
    js_path = root / "frontend" / "bracket_builder.js"

    css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    js = js_path.read_text(encoding="utf-8") if js_path.exists() else ""

    html = f"""
<div id=\"mm-root\"></div>
<style>{css}</style>
<script>
window.MM_PAYLOAD = {json.dumps(payload)};
</script>
<script>{js}</script>
"""

    components.html(html, height=680, scrolling=True)
