"""Hand-tuned baseline and frozen pre-optimization quantities (doc §5.3, §4.2, §8.2).

    python -m experiments.hand_tuned            # run every trial in configs/hand_tuned.yaml
    python -m experiments.hand_tuned --no-write # evaluate only; leave configs/frozen.yaml alone

Steps:
  1. evaluate every logged trial (fixed controller, hard simulator) on the
     validation split; at most `max_trials` trials are allowed;
  2. select the trial with the highest success-and-collision-free rate
     (ties: lower task loss);
  3. lambda balance check on the selected rule: mean weighted loss terms, the
     ratio between the largest and smallest non-zero term, and whether safety
     terms dominate in episodes with contacts;
  4. feature centering mu_z = mean local features of the selected rule's
     rollouts on the FI training split and on the PI training split (p = 0.2,
     masked features);
  5. write configs/frozen.yaml (selected weights, logits, mu_z, provenance).

Every trial's metrics go to results/hand_tuned/<timestamp>/trials.csv.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
from pathlib import Path

from swarm.config import CONFIG_DIR, FROZEN_PATH, PROJECT_ROOT, configure_jax, load_sim_params, load_yaml

configure_jax("float64")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from experiments.common import git_info, make_run_dir  # noqa: E402
from swarm.controllers import ControllerSpec, logits_from_weights  # noqa: E402
from swarm.datasets import load_split  # noqa: E402
from swarm.losses import LossTerms  # noqa: E402
from swarm.metrics import EpisodeMetrics, evaluate, summarize  # noqa: E402
from swarm.model import SimSpec, episodes_from_batch  # noqa: E402

SAFETY_TERMS = ("collision", "obstacle", "boundary")


def lambda_balance(m: EpisodeMetrics, sim) -> dict:
    """Weighted loss terms of the selected rule (doc §5.3 sanity check)."""
    w = sim.loss.weights._asdict() if hasattr(sim.loss.weights, "_asdict") else vars(sim.loss.weights)
    names = LossTerms._fields[:-1]
    per_ep = {n: np.asarray(getattr(m.loss, n)) * w[n] for n in names}
    means = {n: float(v.mean()) for n, v in per_ep.items()}
    nonzero = {n: v for n, v in means.items() if v > 0}
    ratio = max(nonzero.values()) / min(nonzero.values()) if len(nonzero) > 1 else 1.0
    safety = sum(per_ep[n] for n in SAFETY_TERMS)
    other = sum(per_ep[n] for n in names if n not in SAFETY_TERMS)
    contact = np.asarray(m.contact_steps) > 0
    dominated = float(np.mean(safety[contact] > other[contact])) if contact.any() else float("nan")
    return {
        "weighted_term_means": means,
        "max_over_min_nonzero": float(ratio),
        "ratio_ok": bool(ratio <= 100.0),
        "episodes_with_contacts": int(contact.sum()),
        "safety_dominates_in_contact_episodes": dominated,
    }


def feature_mean(theta, split: str, p: float, sim) -> np.ndarray:
    batch, meta = load_split(split, sim)
    spec = SimSpec(sim, ControllerSpec("fixed"), horizon=meta["horizon"])
    m = evaluate(theta, episodes_from_batch(batch, p=p), spec)
    return np.asarray(m.feature_mean).mean(axis=0)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG_DIR / "hand_tuned.yaml")
    ap.add_argument("--out", type=Path, default=FROZEN_PATH)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    trials = cfg["trials"]
    if len(trials) > cfg["max_trials"]:
        raise SystemExit(f"{len(trials)} trials exceed the protocol limit of {cfg['max_trials']}")
    sim = load_sim_params()
    lo, hi = sim.controller.search_box

    batch, meta = load_split(cfg["split"], sim)
    eps = episodes_from_batch(batch)
    spec = SimSpec(sim, ControllerSpec("fixed"), horizon=meta["horizon"])
    run_dir = make_run_dir("hand_tuned", cfg)

    rows, results = [], []
    for k, trial in enumerate(trials):
        b = logits_from_weights(trial["weights"], sim.controller.w_max)
        if np.any(b < lo) or np.any(b > hi):
            raise SystemExit(f"trial {trial['name']}: logits {b.round(2)} leave the search box [{lo}, {hi}]")
        m = evaluate(jnp.asarray(b), eps, spec)
        s = summarize(m, spec)
        results.append((trial, b, m, s))
        rows.append({"trial": k + 1, "name": trial["name"], "weights": " ".join(map(str, trial["weights"])),
                     **{key: round(val, 5) if isinstance(val, float) else val for key, val in s.items()},
                     "note": trial.get("note", "")})
        print(f"[{k + 1:2d}] {trial['name']:16s} w={trial['weights']}  "
              f"succ&cf={s['success_and_collision_free']:.3f}  success={s['success_rate']:.3f}  "
              f"cf={s['collision_free_rate']:.3f}  arrival={s['arrival']:.3f}  "
              f"min_clear={s['min_clearance']:+.3f}  loss={s['task_loss']:.4f}")

    with open(run_dir / "trials.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best = max(results, key=lambda r: (r[3]["success_and_collision_free"], -r[3]["task_loss"]))
    trial, b, m, s = best
    balance = lambda_balance(m, sim)
    print(f"\nselected: {trial['name']} {trial['weights']}")
    print(f"lambda check: {balance}")

    mu = cfg["mu_z"]
    mu_fi = feature_mean(jnp.asarray(b), mu["fully_informed"], 1.0, sim)
    mu_pi = feature_mean(jnp.asarray(b), mu["partially_informed"], mu["pi_fraction"], sim)
    print(f"mu_z (FI, {mu['fully_informed']}): {mu_fi.round(4)}")
    print(f"mu_z (PI, {mu['partially_informed']}, p={mu['pi_fraction']}): {mu_pi.round(4)}")

    frozen = {
        "hand_weights": [float(v) for v in trial["weights"]],
        "b_hand": [float(v) for v in b],
        "mu_z_fi": [float(v) for v in mu_fi],
        "mu_z_pi": [float(v) for v in mu_pi],
        "selected_trial": trial["name"],
        "selection": {"split": cfg["split"], "metric": "success_and_collision_free",
                      "value": s["success_and_collision_free"], "task_loss": s["task_loss"],
                      "n_trials": len(trials)},
        "mu_z_source": {"fi": mu["fully_informed"], "pi": mu["partially_informed"], "pi_fraction": mu["pi_fraction"]},
        "lambda_check": balance,
        "git_commit": git_info()["commit"],
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "trials_log": str((run_dir / "trials.csv").relative_to(PROJECT_ROOT)),
    }
    with open(run_dir / "frozen.yaml", "w") as fh:
        yaml.safe_dump(frozen, fh, sort_keys=False)
    if not args.no_write:
        header = ("# Generated by `python -m experiments.hand_tuned`; frozen before any optimization.\n"
                  "# Do not edit by hand: change configs/hand_tuned.yaml and re-run instead.\n")
        with open(args.out, "w") as fh:
            fh.write(header)
            yaml.safe_dump(frozen, fh, sort_keys=False)
        print(f"wrote {args.out}")
    print(f"trial log: {run_dir / 'trials.csv'}")


if __name__ == "__main__":
    main()
