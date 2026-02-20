from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .probs import get_matchup_probability


ROUND_LABELS: list[str] = ["R64", "R32", "S16", "E8", "F4", "Champ"]
ESPN_POINTS_BY_ROUND: list[int] = [10, 20, 40, 80, 160, 320]


@dataclass(frozen=True)
class SimPlan:
    teams: list[str]
    team_to_index: dict[str, int]
    seed_by_team: dict[str, int]


def build_seed_map(bracket_field: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for region_seeds in bracket_field.values():
        for x in region_seeds:
            team = str(x["team"])
            seed = int(x["seed"])
            out[team] = seed
    return out


def list_teams(bracket_field: dict[str, list[dict[str, Any]]]) -> list[str]:
    teams: list[str] = []
    for region_seeds in bracket_field.values():
        for x in region_seeds:
            teams.append(str(x["team"]))
    return sorted(set(teams))


def make_plan(bracket_field: dict[str, list[dict[str, Any]]]) -> SimPlan:
    teams = list_teams(bracket_field)
    return SimPlan(
        teams=teams,
        team_to_index={t: i for i, t in enumerate(teams)},
        seed_by_team=build_seed_map(bracket_field),
    )


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, float(p)))
    return math.log(p / (1.0 - p))


def adjust_probability(*, p_team_a: float, seed_a: int, seed_b: int, leverage: float) -> float:
    """Adjust P(A wins) using a simple chalk-vs-contrarian lever.

    - leverage=0: more chalky (more extreme probabilities)
    - leverage=1: more contrarian/noisy + mild underdog tilt by seed gap

    This is intentionally a simple heuristic, not a calibrated decision policy.
    """
    leverage = max(0.0, min(1.0, float(leverage)))

    gap = (float(seed_a) - float(seed_b)) / 15.0  # >0 means A is underdog
    gamma = 1.25 * leverage
    temperature = 0.70 + 0.80 * leverage

    logit = _logit(p_team_a)
    logit = (logit + gamma * gap) / temperature
    return _sigmoid(logit)


def choose_winner(
    *,
    team_a: str,
    team_b: str,
    probs: dict,
    seed_by_team: dict[str, int],
    rng: random.Random,
    leverage: float,
) -> str:
    p_a = get_matchup_probability(team_a, team_b, probs)
    seed_a = int(seed_by_team.get(team_a, 16))
    seed_b = int(seed_by_team.get(team_b, 16))
    p_adj = adjust_probability(p_team_a=p_a, seed_a=seed_a, seed_b=seed_b, leverage=leverage)
    return team_a if rng.random() < p_adj else team_b


def simulate_region(
    *,
    seed_to_team: dict[int, str],
    probs: dict,
    seed_by_team: dict[str, int],
    rng: random.Random,
    leverage: float,
    round1_pairings: list[tuple[int, int]],
) -> tuple[str, list[str], list[str], list[str], list[str]]:
    """Return (champion, winners_r64, winners_r32, winners_s16, winners_e8) for a region."""
    r64_winners: list[str] = []
    for s1, s2 in round1_pairings:
        team_a = seed_to_team[int(s1)]
        team_b = seed_to_team[int(s2)]
        r64_winners.append(
            choose_winner(
                team_a=team_a,
                team_b=team_b,
                probs=probs,
                seed_by_team=seed_by_team,
                rng=rng,
                leverage=leverage,
            )
        )

    r32_winners: list[str] = []
    for i in range(0, len(r64_winners), 2):
        r32_winners.append(
            choose_winner(
                team_a=r64_winners[i],
                team_b=r64_winners[i + 1],
                probs=probs,
                seed_by_team=seed_by_team,
                rng=rng,
                leverage=leverage,
            )
        )

    s16_winners: list[str] = []
    for i in range(0, len(r32_winners), 2):
        s16_winners.append(
            choose_winner(
                team_a=r32_winners[i],
                team_b=r32_winners[i + 1],
                probs=probs,
                seed_by_team=seed_by_team,
                rng=rng,
                leverage=leverage,
            )
        )

    e8_winner = choose_winner(
        team_a=s16_winners[0],
        team_b=s16_winners[1],
        probs=probs,
        seed_by_team=seed_by_team,
        rng=rng,
        leverage=leverage,
    )

    return e8_winner, r64_winners, r32_winners, s16_winners, [e8_winner]


def simulate_tournament_winners(
    *,
    bracket_field: dict[str, list[dict[str, Any]]],
    probs: dict,
    leverage: float,
    rng: random.Random,
    round1_pairings: list[tuple[int, int]],
) -> list[str]:
    """Return winners for all 63 games in a fixed order (by round)."""
    seed_by_team = build_seed_map(bracket_field)

    region_order = ["UL", "UR", "LL", "LR"]

    r64: list[str] = []
    r32: list[str] = []
    s16: list[str] = []
    e8: list[str] = []

    region_champs: dict[str, str] = {}

    for rk in region_order:
        seeds = bracket_field[rk]
        seed_to_team = {int(x["seed"]): str(x["team"]) for x in seeds}
        champ, w64, w32, w16, w8 = simulate_region(
            seed_to_team=seed_to_team,
            probs=probs,
            seed_by_team=seed_by_team,
            rng=rng,
            leverage=leverage,
            round1_pairings=round1_pairings,
        )
        region_champs[rk] = champ
        r64.extend(w64)
        r32.extend(w32)
        s16.extend(w16)
        e8.extend(w8)

    # Final Four + Championship
    # Quadrants are layout positions:
    # - UL/LL are the left side of the printed bracket
    # - UR/LR are the right side of the printed bracket
    # The national semifinals are therefore left-vs-left and right-vs-right.
    f4_1 = choose_winner(
        team_a=region_champs["UL"],
        team_b=region_champs["LL"],
        probs=probs,
        seed_by_team=seed_by_team,
        rng=rng,
        leverage=leverage,
    )
    f4_2 = choose_winner(
        team_a=region_champs["UR"],
        team_b=region_champs["LR"],
        probs=probs,
        seed_by_team=seed_by_team,
        rng=rng,
        leverage=leverage,
    )
    champ = choose_winner(
        team_a=f4_1,
        team_b=f4_2,
        probs=probs,
        seed_by_team=seed_by_team,
        rng=rng,
        leverage=leverage,
    )

    return [*r64, *r32, *s16, *e8, f4_1, f4_2, champ]


def encode_winners(plan: SimPlan, winners: list[str]) -> list[int]:
    return [int(plan.team_to_index[w]) for w in winners]


def decode_winners(plan: SimPlan, winners_idx: Iterable[int]) -> list[str]:
    teams = plan.teams
    return [teams[int(i)] for i in winners_idx]


def simulate_many(
    *,
    bracket_field: dict[str, list[dict[str, Any]]],
    probs: dict,
    n_sims: int,
    leverage: float,
    seed: int,
    round1_pairings: list[tuple[int, int]],
) -> tuple[SimPlan, np.ndarray]:
    plan = make_plan(bracket_field)
    rng = random.Random(int(seed))

    sims: list[list[int]] = []
    for _ in range(int(n_sims)):
        winners = simulate_tournament_winners(
            bracket_field=bracket_field,
            probs=probs,
            leverage=leverage,
            rng=rng,
            round1_pairings=round1_pairings,
        )
        sims.append(encode_winners(plan, winners))

    return plan, np.asarray(sims, dtype=np.int16)


def score_bracket_vs_sims(*, picks: np.ndarray, sims: np.ndarray) -> np.ndarray:
    """Return total score for each simulation (shape: n_sims,)."""
    # picks: (63,)
    # sims: (n_sims, 63)
    correct = sims == picks.reshape(1, -1)

    # Round boundaries
    sizes = [32, 16, 8, 4, 2, 1]
    weights = ESPN_POINTS_BY_ROUND

    start = 0
    scores = np.zeros((sims.shape[0],), dtype=np.float32)
    for size, w in zip(sizes, weights, strict=True):
        end = start + size
        scores += correct[:, start:end].sum(axis=1).astype(np.float32) * float(w)
        start = end
    return scores


def random_search(
    *,
    bracket_field: dict[str, list[dict[str, Any]]],
    probs: dict,
    sims: np.ndarray,
    plan: SimPlan,
    n_candidates: int,
    leverage: float,
    seed: int,
    round1_pairings: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    rng = random.Random(int(seed))
    results: list[dict[str, Any]] = []

    for _ in range(int(n_candidates)):
        winners = simulate_tournament_winners(
            bracket_field=bracket_field,
            probs=probs,
            leverage=leverage,
            rng=rng,
            round1_pairings=round1_pairings,
        )
        picks_idx = np.asarray(encode_winners(plan, winners), dtype=np.int16)
        scores = score_bracket_vs_sims(picks=picks_idx, sims=sims)

        results.append(
            {
                "expected_score": float(scores.mean()),
                "std_score": float(scores.std(ddof=0)),
                "champ": winners[-1],
                "picks_idx": picks_idx,
            }
        )

    results.sort(key=lambda x: (x["expected_score"], -x["std_score"]), reverse=True)
    return results


def select_diverse_topk(
    results: list[dict[str, Any]],
    *,
    top_k: int,
    mode: str = "none",
    max_per_champ: int = 5,
    top_champs: int = 10,
    per_champ: int = 3,
) -> list[dict[str, Any]]:
    """Select a top-K list, optionally diversified by champion pick.

    This operates on a list already sorted best→worst (expected_score primary).
    Modes:
    - none: return first K
    - cap: at most max_per_champ per champion
    - top_champs: choose top_champs champions (by best bracket), then up to per_champ each
    """

    k = int(max(1, top_k))
    mode_norm = (mode or "none").strip().lower()
    if mode_norm in {"none", "off", "false", "0"}:
        return results[:k]

    def champ_of(r: dict[str, Any]) -> str:
        return str(r.get("champ", ""))

    if mode_norm in {"cap", "max_per_champ", "limit"}:
        cap = int(max(1, max_per_champ))
        counts: dict[str, int] = {}
        picked: list[dict[str, Any]] = []
        for r in results:
            c = champ_of(r)
            if counts.get(c, 0) >= cap:
                continue
            picked.append(r)
            counts[c] = counts.get(c, 0) + 1
            if len(picked) >= k:
                break
        return picked

    if mode_norm in {"top_champs", "champions", "per_champ"}:
        n_champs = int(max(1, top_champs))
        per = int(max(1, per_champ))

        # Identify top champions by the best bracket for that champion.
        champ_best: dict[str, dict[str, Any]] = {}
        for r in results:
            c = champ_of(r)
            if c not in champ_best:
                champ_best[c] = r
        champ_ranked = list(champ_best.keys())[:n_champs]

        picked: list[dict[str, Any]] = []
        counts: dict[str, int] = {c: 0 for c in champ_ranked}
        for r in results:
            c = champ_of(r)
            if c not in counts:
                continue
            if counts[c] >= per:
                continue
            picked.append(r)
            counts[c] += 1
            if len(picked) >= k:
                break

        return picked

    # Unknown mode → fall back to non-diversified.
    return results[:k]
