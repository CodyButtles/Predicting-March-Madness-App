from __future__ import annotations

from typing import Any


ROUND1_PAIRINGS: list[tuple[int, int]] = [
    (1, 16),
    (8, 9),
    (5, 12),
    (4, 13),
    (6, 11),
    (3, 14),
    (7, 10),
    (2, 15),
]


def build_round1_games(region_seed_list: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Return Round-of-64 matchups for a region using standard seed pairings."""
    seed_to_team = {int(x["seed"]): x["team"] for x in region_seed_list}
    games: list[list[dict[str, Any]]] = []
    for a, b in ROUND1_PAIRINGS:
        games.append([
            {"seed": a, "team": seed_to_team[a]},
            {"seed": b, "team": seed_to_team[b]},
        ])
    return games


def iter_all_teams(bracket_field: dict[str, list[dict[str, Any]]]) -> list[str]:
    teams: list[str] = []
    for quad in bracket_field.values():
        for x in quad:
            teams.append(x["team"])
    return sorted(set(teams))
