"""Rollout benchmark: sets the per-run budget B_run and c_grad (doc §8.2, §8.5, §9.5).

    python -m experiments.benchmark                       # float64 (default)
    python -m experiments.benchmark --precision float32   # derivative-free paths only
    python -m experiments.benchmark --budget-minutes 15

Measures on a training split (default exp2_train: M = 32, N = 30, T = 400):
  * compile time and steady-state run time of every objective path;
  * c_grad = time(value_and_grad) / time(value) on the soft adaptive model;
  * rollouts per second for CMA-ES populations (8 for 5-D, 13 for 25-D);
  * XLA's temporary-memory estimate with (default) and without jax.checkpoint;
  * what fits in --budget-minutes: Exp1 run time, CMA-ES generations,
    L-BFGS-B gradient calls, and the Exp3 offline cost.

Writes results/benchmark/<timestamp>/benchmark.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time


def _parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precision", default="float64", choices=["float32", "float64"])
    ap.add_argument("--split", default="exp2_train")
    ap.add_argument("--eval-split", default="test_id")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--budget-minutes", type=float, default=15.0)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse(argv)
    from swarm.config import configure_jax

    configure_jax(args.precision)            # before any array exists

    import jax
    import jax.numpy as jnp
    import numpy as np

    from experiments.common import environment_info, make_run_dir
    from swarm import metrics, rollout
    from swarm.config import load_frozen, load_sim_params
    from swarm.controllers import ControllerSpec, warm_start
    from swarm.datasets import load_split
    from swarm.model import SimSpec, episodes_from_batch

    sim = load_sim_params()
    frozen = load_frozen()
    batch, meta = load_split(args.split, sim)
    eps = episodes_from_batch(batch)
    dtype = eps.x0.dtype
    M, N, T = batch.n_episodes, batch.n_agents, meta["horizon"]
    b = jnp.asarray(frozen["b_hand"], dtype)
    mu_z = jnp.asarray(frozen["mu_z_fi"], dtype)
    fixed_hard = SimSpec(sim, ControllerSpec("fixed"), hard=True, horizon=T)
    adapt_hard = SimSpec(sim, ControllerSpec("adaptive"), hard=True, horizon=T)
    adapt_soft = adapt_hard.replace(hard=False)
    theta_a = warm_start(b, adapt_hard.controller) + 0.1 * jax.random.normal(jax.random.key(0), (25,), dtype)
    eta = sim.smoothing.fixed_soft_eta

    def pop(p, base):
        return base[None] + 0.1 * jax.random.normal(jax.random.key(p), (p, base.shape[0]), dtype)

    def timed(fn, *a, **kw):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*a, **kw))
        compile_s = time.perf_counter() - t0
        runs = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*a, **kw))
            runs.append(time.perf_counter() - t0)
        return compile_s, statistics.median(runs)

    def temp_mb(jitted, *a, **kw):
        try:
            stats = jitted.lower(*a, **kw).compile().memory_analysis()
            return round(stats.temp_size_in_bytes / 2**20, 1)
        except Exception:  # pragma: no cover - backend without memory analysis
            return None

    cases = {
        "loss_fixed_hard": (rollout.loss, (b, eps, fixed_hard, 0.0, None), M),
        "population8_fixed_hard": (rollout.population_loss, (pop(8, b), eps, fixed_hard, 0.0, None), 8 * M),
        "population13_adaptive_hard": (rollout.population_loss, (pop(13, theta_a), eps, adapt_hard, 0.0, mu_z), 13 * M),
        "loss_adaptive_soft": (rollout.loss, (theta_a, eps, adapt_soft, eta, mu_z), M),
        "grad_adaptive_soft": (rollout.loss_and_grad, (theta_a, eps, adapt_soft, eta, mu_z), M),
        "grad_adaptive_soft_no_remat": (rollout.loss_and_grad, (theta_a, eps, adapt_soft.replace(remat=False), eta, mu_z), M),
    }
    results = {}
    print(f"split={args.split} M={M} N={N} T={T} precision={args.precision} backend={jax.default_backend()}")
    print(f"{'case':30s} {'compile s':>10s} {'run s':>8s} {'rollouts/s':>11s} {'temp MB':>8s}")
    for name, (fn, a, n_roll) in cases.items():
        c, r = timed(fn, *a)
        mem = temp_mb(fn, *a)
        results[name] = {"compile_s": round(c, 3), "run_s": round(r, 4), "rollouts": n_roll,
                         "rollouts_per_s": round(n_roll / r, 1), "temp_mb": mem}
        print(f"{name:30s} {c:10.2f} {r:8.3f} {n_roll / r:11.1f} {str(mem):>8s}")

    ebatch, emeta = load_split(args.eval_split, sim)
    eeps = episodes_from_batch(ebatch)
    espec = fixed_hard.replace(horizon=emeta["horizon"])
    c, r = timed(lambda: metrics.evaluate(b, eeps, espec))
    results["evaluate_fixed_hard"] = {"compile_s": round(c, 3), "run_s": round(r, 4), "episodes": ebatch.n_episodes}
    print(f"{'evaluate (' + args.eval_split + ')':30s} {c:10.2f} {r:8.3f} {ebatch.n_episodes / r:11.1f}")

    c_grad = results["grad_adaptive_soft"]["run_s"] / results["loss_adaptive_soft"]["run_s"]
    budget_s = 60 * args.budget_minutes
    gen13 = results["population13_adaptive_hard"]["run_s"]
    gen8 = results["population8_fixed_hard"]["run_s"]
    grad = results["grad_adaptive_soft"]["run_s"]
    rps = results["population13_adaptive_hard"]["rollouts_per_s"]
    plan = {
        "c_grad": round(c_grad, 2),
        "budget_minutes": args.budget_minutes,
        "exp1_run_minutes_300_evals_pop8": round(300 / 8 * gen8 / 60, 2),
        "cmaes_generations_per_budget_25d": int(budget_s / gen13),
        "lbfgsb_grad_calls_per_budget": int(budget_s / grad),
        "exp3_offline_hours_3e5_rollouts": round(3e5 / rps / 3600, 2),
        "exp2_matrix_hours_15_to_33_runs": [round(15 * budget_s / 3600, 2), round(33 * budget_s / 3600, 2)],
    }
    print("\nplanning numbers:")
    for k, v in plan.items():
        print(f"  {k}: {v}")

    run_dir = make_run_dir("benchmark", vars(args) | {"precision": args.precision})
    out = {"setup": {"split": args.split, "M": M, "N": N, "T": T, "precision": args.precision,
                     "environment": environment_info()},
           "cases": results, "plan": plan}
    with open(run_dir / "benchmark.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {run_dir / 'benchmark.json'}")


if __name__ == "__main__":
    main()
