from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import os

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PrivateRepoConfig:
    repo: str
    ref: str
    token: str


def _get_private_repo_config() -> Optional[PrivateRepoConfig]:
    """Return private repo config from Streamlit Secrets or env vars.

    Secrets format (recommended):

    [private_data]
    repo = "owner/private-data-repo"
    ref = "main"
    token = "<fine-grained token>"

    Env var fallback:
    - PRIVATE_DATA_REPO
    - PRIVATE_DATA_REF (default: main)
    - PRIVATE_DATA_TOKEN
    """

    if st is not None:
        try:
            if "private_data" in st.secrets:  # type: ignore[operator]
                sec = st.secrets["private_data"]  # type: ignore[index]
                repo = str(sec.get("repo", "")).strip()
                token = str(sec.get("token", "")).strip()
                ref = str(sec.get("ref", "main")).strip() or "main"
                if repo and token:
                    return PrivateRepoConfig(repo=repo, ref=ref, token=token)
        except Exception:
            # If secrets are misconfigured, fail closed by returning None.
            return None

    repo = str(os.getenv("PRIVATE_DATA_REPO", "")).strip()
    token = str(os.getenv("PRIVATE_DATA_TOKEN", "")).strip()
    ref = str(os.getenv("PRIVATE_DATA_REF", "main")).strip() or "main"
    if repo and token:
        return PrivateRepoConfig(repo=repo, ref=ref, token=token)
    return None


def _to_repo_rel_path(local_path: Path) -> str:
    """Map an absolute local path to a repo-relative path.

    We only allow pulling from the same relative paths you use locally,
    e.g. Output/foo.json or Data/bar.csv.
    """

    parts = [p for p in local_path.as_posix().split("/") if p]

    # Prefer anchoring at Output/ or Data/.
    for anchor in ("Output", "Data"):
        if anchor in parts:
            i = parts.index(anchor)
            rel = "/".join(parts[i:])
            return rel

    # Fallback: just filename.
    return local_path.name


def _fetch_raw_bytes(*, cfg: PrivateRepoConfig, rel_path: str) -> bytes:
    import requests

    url = f"https://raw.githubusercontent.com/{cfg.repo}/{cfg.ref}/{rel_path}"
    headers = {
        "Authorization": f"Bearer {cfg.token}",
        "Accept": "application/vnd.github.raw",
    }
    resp = requests.get(url, headers=headers, timeout=60)
    if resp.status_code != 200:
        raise FileNotFoundError(f"Failed to fetch {rel_path} from {cfg.repo}@{cfg.ref} (HTTP {resp.status_code})")
    return resp.content


def _fetch_raw_bytes_cached(repo: str, ref: str, token: str, rel_path: str) -> bytes:
    return _fetch_raw_bytes(cfg=PrivateRepoConfig(repo=repo, ref=ref, token=token), rel_path=rel_path)


if st is not None:
    _fetch_raw_bytes_cached = st.cache_data(show_spinner=False)(_fetch_raw_bytes_cached)  # type: ignore[misc]


def read_bytes_maybe_private(local_path: Path) -> bytes:
    """Read bytes from local disk, or fall back to a private GitHub repo.

    This lets you keep sensitive data out of the public app repo while still
    running a public Streamlit deployment.
    """

    if local_path.exists():
        return local_path.read_bytes()

    cfg = _get_private_repo_config()
    if cfg is None:
        raise FileNotFoundError(f"Missing file and no private data config: {local_path.as_posix()}")

    rel = _to_repo_rel_path(local_path)
    return _fetch_raw_bytes_cached(cfg.repo, cfg.ref, cfg.token, rel)


def read_text_maybe_private(local_path: Path, *, encoding: str = "utf-8") -> str:
    return read_bytes_maybe_private(local_path).decode(encoding)
