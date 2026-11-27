"""Animate one rollout.

    python -m visualization.animate --split val_id --index 0 --weights 2 1 1 3 3
    python -m visualization.animate --split pi_test_id --index 3 --p 0.2 --frozen
    python -m visualization.animate --split val_id --index 0 --theta-file theta.npy --controller adaptive --frozen

Agents are colored by their dominant behavior, argmax_j |w_j f_j / nu_j|.
Informed agents get a black outline when p < 1. The heads-up line shows time,
arrival fraction, contacts so far and the current minimum clearance. Writes a
GIF (Pillow, no extra dependency) or an MP4 when the path ends in .mp4 and
ffmpeg is installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from swarm.config import BEHAVIORS, SimParams  # noqa: E402
from swarm.metrics import swept_contacts, swept_min_clearance  # noqa: E402
from swarm.model import Episode  # noqa: E402
from swarm.rollout import Trajectory  # noqa: E402

BEHAVIOR_COLORS = ("#d1495b", "#2a78d6", "#8e6bbf", "#2e9b5f", "#e8a33d")


def contact_timeline(traj: Trajectory, ep: Episode, sim: SimParams) -> tuple[np.ndarray, np.ndarray]:
    """Per-step number of contacts and minimum clearance (T,)."""
    r_a, tol = sim.agent.radius, sim.evaluation.contact_tol

    def one(x0, x1):
        hits = swept_contacts(x0, x1, ep, r_a, tol)
        return sum(h.sum() for h in hits), swept_min_clearance(x0, x1, ep, r_a)

    counts, clear = jax.vmap(one)(traj.x[:-1], traj.x[1:])
    return np.asarray(counts), np.asarray(clear)


def animate(
    traj: Trajectory,
    ep: Episode,
    sim: SimParams,
    out: str | Path,
    stride: int = 2,
    fps: int = 20,
    title: str | None = None,
) -> Path:
    x = np.asarray(traj.x)
    dom = np.asarray(traj.dominant)
    informed = np.asarray(ep.informed, dtype=bool)
    goal, rg = np.asarray(ep.goal), float(ep.goal_radius)
    lo, hi = np.asarray(ep.lower), np.asarray(ep.upper)
    counts, clear = contact_timeline(traj, ep, sim)
    arrival = np.mean(np.linalg.norm(x - goal, axis=-1) <= rg, axis=1)
    dt = sim.simulation.dt

    fig, ax = plt.subplots(figsize=(6, 6.4))
    ax.add_patch(Rectangle(tuple(lo), *(hi - lo), fill=False, lw=1.2, ec="black"))
    ax.add_patch(Circle(tuple(goal), rg, fc="#2e9b5f", alpha=0.15, ec="#2e9b5f"))
    for c, r, valid in zip(np.asarray(ep.centers), np.asarray(ep.radii), np.asarray(ep.valid)):
        if valid > 0:
            ax.add_patch(Circle(tuple(c), r, fc="#6b6b6b", ec="none", alpha=0.85))
    ax.set_xlim(lo[0] - 0.3, hi[0] + 0.3)
    ax.set_ylim(lo[1] - 0.3, hi[1] + 0.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # marker size in points^2 so agents are drawn at their true radius
    px_per_unit = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]
    size = (2 * sim.agent.radius * px_per_unit * 72 / fig.dpi) ** 2
    edge = np.where(informed & ~informed.all(), "black", "none")
    scat = ax.scatter(x[0, :, 0], x[0, :, 1], s=size, c=[BEHAVIOR_COLORS[0]] * x.shape[1],
                      edgecolors=edge, linewidths=0.8, zorder=3)
    hud = ax.text(0.01, 1.01, "", transform=ax.transAxes, fontsize=9, va="bottom", family="monospace")
    ax.legend(handles=[Line2D([], [], ls="", marker="o", color=c, label=b)
                       for b, c in zip(BEHAVIORS, BEHAVIOR_COLORS)],
              loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=5, fontsize=7, frameon=False)
    if title:
        fig.suptitle(title, fontsize=10)

    frames = list(range(0, x.shape[0], stride))
    if frames[-1] != x.shape[0] - 1:
        frames.append(x.shape[0] - 1)

    def draw(t):
        scat.set_offsets(x[t])
        colors = [BEHAVIOR_COLORS[k] for k in dom[min(t, dom.shape[0] - 1)]]
        scat.set_facecolors(colors)
        seen = counts[:t].sum() if t else 0
        cl = clear[min(t, len(clear) - 1)]
        hud.set_text(f"t={t * dt:5.2f}s  arrival={arrival[t]:.2f}  contacts={int(seen):4d}  clearance={cl:+.3f}")
        return scat, hud

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps) if out.suffix == ".mp4" else PillowWriter(fps=fps)
    anim.save(out, writer=writer)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    from swarm.config import configure_jax, load_frozen, load_sim_params

    configure_jax("float64")
    import jax.numpy as jnp

    from swarm.controllers import ControllerSpec, logits_from_weights
    from swarm.datasets import load_split
    from swarm.model import SimSpec, episodes_from_batch, take
    from swarm.rollout import simulate

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val_id")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--p", type=float, default=1.0, help="informed fraction")
    ap.add_argument("--controller", default="fixed", choices=["fixed", "adaptive", "time"])
    ap.add_argument("--weights", type=float, nargs=5, help="fixed-rule behavior weights (sep ali coh goal obs)")
    ap.add_argument("--theta-file", type=Path, help=".npy parameter vector for --controller")
    ap.add_argument("--frozen", action="store_true",
                    help="use configs/frozen.yaml: hand-tuned weights and mu_z")
    ap.add_argument("--soft-eta", type=float, help="simulate with soft neighbors of this eta")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args(argv)

    sim = load_sim_params()
    batch, meta = load_split(args.split, sim)
    ep = take(episodes_from_batch(batch, p=args.p), args.index)
    spec = SimSpec(sim, ControllerSpec(args.controller), hard=args.soft_eta is None, horizon=meta["horizon"])
    frozen = load_frozen() if args.frozen else None
    mu_z = None
    if frozen is not None and args.controller == "adaptive":
        mu_z = jnp.asarray(frozen["mu_z_pi"] if args.p < 1 else frozen["mu_z_fi"])
    if args.theta_file:
        theta = jnp.asarray(np.load(args.theta_file))
    elif args.weights:
        theta = jnp.asarray(logits_from_weights(args.weights, sim.controller.w_max))
    elif frozen is not None:
        theta = jnp.asarray(frozen["b_hand"])
    else:
        raise SystemExit("give --weights, --theta-file or --frozen")
    if args.controller != "fixed" and theta.shape[0] == 5:
        from swarm.controllers import warm_start

        theta = warm_start(theta, spec.controller)
    traj = simulate(theta, ep, spec, args.soft_eta or 0.0, mu_z)
    scenario = meta["map_scenarios"][int(batch.map_index[args.index])]
    out = args.out or Path("results/animations") / f"{args.split}_{args.index}_{args.controller}.gif"
    path = animate(traj, ep, sim, out, stride=args.stride,
                   title=f"{args.split} #{args.index} ({scenario}), {args.controller}, p={args.p}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
