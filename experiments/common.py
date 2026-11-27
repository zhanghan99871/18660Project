"""Shared run bookkeeping: every run records its config and code version."""

from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from swarm.config import PROJECT_ROOT, config_hash

RESULTS_ROOT = PROJECT_ROOT / "results"
DATA_ROOT = PROJECT_ROOT / "data"


def git_info(root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    """Current commit hash and whether the working tree has uncommitted changes.

    Returns `{"commit": None, "dirty": None}` outside a git repository.
    """

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status)}


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import jax

        info["jax"] = jax.__version__
        info["jax_backend"] = jax.default_backend()
        info["jax_x64"] = bool(jax.config.read("jax_enable_x64"))
    except Exception:  # pragma: no cover - jax missing or misconfigured
        pass
    try:
        import numpy

        info["numpy"] = numpy.__version__
    except ImportError:  # pragma: no cover
        pass
    return info


def make_run_dir(
    experiment: str,
    config: dict[str, Any],
    results_root: str | Path = RESULTS_ROOT,
    tag: str | None = None,
) -> Path:
    """Create `results/<experiment>/<timestamp>[_tag]/` with config + metadata.

    Writes `config.yaml` (the exact config used) and `meta.json` (git commit,
    dirty flag, config hash, command line, library versions).
    """
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    name = f"{stamp}_{tag}" if tag else stamp
    run_dir = Path(results_root) / experiment / name
    run_dir.mkdir(parents=True, exist_ok=False)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    meta = {
        "experiment": experiment,
        "tag": tag,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "config_hash": config_hash(config),
        "argv": sys.argv,
        "git": git_info(),
        "environment": environment_info(),
    }
    with open(run_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return run_dir
