from __future__ import annotations

from pathlib import Path

import streamlit as st

from .paths import get_output_paths
from .util import read_json


@st.cache_data(show_spinner=False)
def _read_json_cached(path_str: str, mtime: float) -> dict:
    # mtime is an intentional cache-buster: if the file changes on disk,
    # Streamlit will recompute.
    return read_json(Path(path_str))


def _load_json_with_mtime(path: Path) -> dict:
    mtime = path.stat().st_mtime if path.exists() else 0.0
    return _read_json_cached(str(path), mtime)


def _load_json_candidates(paths: list[Path]) -> dict:
    last_err: Exception | None = None
    for p in paths:
        try:
            return _load_json_with_mtime(p)
        except FileNotFoundError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    raise FileNotFoundError("No candidate paths provided")


def load_bracket_field(year: int) -> dict:
    paths = get_output_paths(year)
    return _load_json_candidates(paths.bracket_field_json_candidates)


def load_advancement_probs(year: int) -> dict:
    paths = get_output_paths(year)
    return _load_json_candidates(paths.advancement_probs_json_candidates)


def load_matchup_probs(year: int) -> dict:
    paths = get_output_paths(year)
    return _load_json_candidates(paths.matchup_probs_json_candidates)


def load_matchup_explanations(year: int) -> dict:
    paths = get_output_paths(year)
    return _load_json_candidates(paths.matchup_explanations_json_candidates)


def load_optimizer_sims(year: int) -> dict:
    paths = get_output_paths(year)
    return _load_json_candidates(paths.optimizer_sims_json_candidates)


def load_optimizer_top25(year: int) -> dict:
    paths = get_output_paths(year)
    return _load_json_candidates(paths.optimizer_top25_json_candidates)
