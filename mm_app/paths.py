from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Find repo root by looking for Data/ and Output/ directories."""
    start = (start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p / "Data").exists() and (p / "Output").exists():
            return p
    return start


@dataclass(frozen=True)
class OutputPaths:
    year: int
    root: Path

    @property
    def output_dir(self) -> Path:
        return self.root / "Output"

    @property
    def year_output_dir(self) -> Path:
        return self.output_dir / str(self.year)

    def _candidates(self, *, canonical_name: str, legacy_name: str) -> list[Path]:
        return [self.year_output_dir / canonical_name, self.output_dir / legacy_name]

    @property
    def bracket_field_json_candidates(self) -> list[Path]:
        return self._candidates(
            canonical_name="bracket_field.json",
            legacy_name=f"bracket_field_{self.year}.json",
        )

    @property
    def bracket_field_json(self) -> Path:
        return self.bracket_field_json_candidates[0]

    @property
    def advancement_probs_json_candidates(self) -> list[Path]:
        return self._candidates(
            canonical_name="advancement_probs.json",
            legacy_name=f"advancement_probs_{self.year}.json",
        )

    @property
    def advancement_probs_json(self) -> Path:
        return self.advancement_probs_json_candidates[0]

    @property
    def matchup_probs_json_candidates(self) -> list[Path]:
        return self._candidates(
            canonical_name="matchup_probabilities.json",
            legacy_name=f"matchup_probabilities_{self.year}.json",
        )

    @property
    def matchup_probs_json(self) -> Path:
        return self.matchup_probs_json_candidates[0]

    @property
    def matchup_explanations_json_candidates(self) -> list[Path]:
        return self._candidates(
            canonical_name="matchup_explanations.json",
            legacy_name=f"matchup_explanations_{self.year}.json",
        )

    @property
    def matchup_explanations_json(self) -> Path:
        return self.matchup_explanations_json_candidates[0]

    @property
    def optimizer_sims_json_candidates(self) -> list[Path]:
        return self._candidates(
            canonical_name="optimizer_sims.json",
            legacy_name=f"optimizer_sims_{self.year}.json",
        )

    @property
    def optimizer_sims_json(self) -> Path:
        return self.optimizer_sims_json_candidates[0]

    @property
    def optimizer_top25_json_candidates(self) -> list[Path]:
        return self._candidates(
            canonical_name="optimizer_top25.json",
            legacy_name=f"optimizer_top25_{self.year}.json",
        )

    @property
    def optimizer_top25_json(self) -> Path:
        return self.optimizer_top25_json_candidates[0]


def get_output_paths(year: int, root: Path | None = None) -> OutputPaths:
    return OutputPaths(year=year, root=(root or find_repo_root()))
