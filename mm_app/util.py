from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .private_data import read_text_maybe_private


def slug_to_display_name(team_slug: str) -> str:
    parts = [p for p in team_slug.split("/") if p]
    raw = parts[-1] if parts else team_slug
    raw = raw.replace("-", " ")
    return " ".join(w.capitalize() for w in raw.split())


def read_json(path: Path) -> Any:
    return json.loads(read_text_maybe_private(path, encoding="utf-8"))


def write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent), encoding="utf-8")
