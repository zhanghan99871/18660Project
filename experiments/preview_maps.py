"""Render sample maps of every scenario type for a visual sanity check.

    python -m experiments.preview_maps [--out results/figures/map_preview.png]

Rows: N=30 fixed area (every scenario), then N=100 fixed area and fixed density.
Red agents are the informed ones at p = 0.2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from swarm.config import RESULTS_ROOT, load_sim_params  # noqa: E402
from swarm.datasets import SplitSpec, generate_feasible_map, load_data_config  # noqa: E402
from swarm.descriptors import map_descriptor  # noqa: E402
from swarm.episodes import informed_mask, sample_initial_state  # noqa: E402
from visualization.plots import draw_map  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=RESULTS_ROOT / "figures" / "map_preview.png")
    ap.add_argument("--p", type=float, default=0.2, help="informed fraction to highlight")
    args = ap.parse_args()

    sim, cfg = load_sim_params(), load_data_config()
    panels = [(name, 30, "fixed_area") for name in cfg.scenarios]
    panels += [(name, 100, setting) for setting in ("fixed_area", "fixed_density")
               for name in ("random_obstacles", "bottleneck")]

    cols = 4
    rows = -(-len(panels) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for k, (ax, (scenario, n, setting)) in enumerate(zip(axes.flat, panels)):
        spec = SplitSpec(name=f"preview_{scenario}_{n}_{setting}", mixture="train_id", n_maps=1,
                         episodes_per_map=1, n_agents=n, size_setting=setting)
        m, rejected = generate_feasible_map(spec, 0, scenario, sim, cfg)
        st = sample_initial_state(m, cfg.seed, spec.name, 0, 0, cfg.layout.init_spacing,
                                  sim.agent.radius, cfg.velocity)
        psi = map_descriptor(m, sim, cfg.descriptor_resolution, cfg.gap_cap)
        title = (f"{scenario}\nN={n} {setting.replace('_', ' ')}  K={m.n_obstacles}  "
                 f"gap/r={psi[2]:.2f}")
        draw_map(ax, m, st.x0, informed_mask(st.order, args.p), sim.agent.radius, title)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
