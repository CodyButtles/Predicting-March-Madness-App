from __future__ import annotations

from typing import Mapping


def get_matchup_probability(team_a: str, team_b: str, probs: Mapping) -> float:
    """Return P(team_a beats team_b) with symmetry + default fallback."""
    try:
        if team_a in probs and team_b in probs[team_a]:
            return float(probs[team_a][team_b])
        if team_b in probs and team_a in probs[team_b]:
            return 1.0 - float(probs[team_b][team_a])
    except Exception:
        pass
    return 0.5


def confidence_from_probability(p: float) -> float:
    """Simple 0..1 confidence proxy based on distance from 0.5."""
    try:
        p = float(p)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, abs(p - 0.5) * 2.0))
