"""Generate and cache every dataset split (common random numbers for all optimizers).

    python -m experiments.build_datasets                 # all splits, skip up-to-date ones
    python -m experiments.build_datasets --splits exp1_train,val_id
    python -m experiments.build_datasets --force         # rebuild even if current

Output: data/<split>.npz (arrays) + data/<split>.json (metadata, hash, git info).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.common import git_info
from swarm.config import DATA_ROOT, load_sim_params
from swarm.datasets import build_split, is_current, load_data_config, save_split, split_paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default="all", help="comma-separated split names, or 'all'")
    ap.add_argument("--data-dir", type=Path, default=DATA_ROOT)
    ap.add_argument("--data-config", type=Path, default=None, help="defaults to configs/data.yaml")
    ap.add_argument("--base-config", type=Path, default=None, help="defaults to configs/base.yaml")
    ap.add_argument("--force", action="store_true", help="rebuild splits that are already current")
    args = ap.parse_args(argv)

    sim = load_sim_params(args.base_config)
    cfg = load_data_config(args.data_config)
    names = list(cfg.splits) if args.splits == "all" else [s.strip() for s in args.splits.split(",")]
    for name in names:
        cfg.split(name)   # fail early on typos

    git = git_info()
    header = f"{'split':32s} {'maps':>5s} {'eps':>5s} {'N':>4s} {'K':>4s} {'T':>4s} {'rejected':>8s} {'sec':>6s}"
    print(header)
    print("-" * len(header))
    for name in names:
        if not args.force and is_current(name, sim, cfg, args.data_dir):
            print(f"{name:32s} up to date")
            continue
        batch, meta = build_split(cfg.split(name), sim, cfg)
        meta["git"] = git
        save_split(batch, meta, args.data_dir)
        print(f"{name:32s} {meta['n_maps']:5d} {meta['n_episodes']:5d} {meta['n_agents']:4d} "
              f"{meta['k_max']:4d} {meta['horizon']:4d} {sum(meta['rejections']):8d} "
              f"{meta['build_seconds']:6.2f}")
    print(f"\nsaved to {args.data_dir}")

    # sanity: everything requested is now loadable and current
    stale = [n for n in names if not is_current(n, sim, cfg, args.data_dir)]
    if stale:
        raise SystemExit(f"splits not current after build: {stale}")


if __name__ == "__main__":
    main()
