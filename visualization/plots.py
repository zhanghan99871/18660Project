"""Static plots. Phase 1: map and initial-state previews."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Rectangle

from swarm.maps import MapSpec

OBSTACLE_COLOR = "#6b6b6b"
START_COLOR = "#2a78d6"
GOAL_COLOR = "#2e9b5f"
AGENT_COLOR = "#1d3557"
INFORMED_COLOR = "#d1495b"


def draw_map(
    ax: Axes,
    m: MapSpec,
    x0: np.ndarray | None = None,
    informed: np.ndarray | None = None,
    agent_radius: float = 0.05,
    title: str | None = None,
) -> Axes:
    """Draw domain, obstacles, start box, goal and (optionally) the initial swarm.

    Informed agents (if a mask is given) are drawn in a contrasting color.
    """
    lx, ly = m.lower
    ux, uy = m.upper
    ax.add_patch(Rectangle((lx, ly), ux - lx, uy - ly, fill=False, lw=1.2, ec="black"))
    w, h = m.start_hi - m.start_lo
    ax.add_patch(Rectangle(tuple(m.start_lo), w, h, fc=START_COLOR, alpha=0.15, ec=START_COLOR, lw=1))
    ax.add_patch(Circle(tuple(m.goal), m.goal_radius, fc=GOAL_COLOR, alpha=0.15, ec=GOAL_COLOR, lw=1))
    for c, r in zip(m.centers, m.radii):
        ax.add_patch(Circle(tuple(c), r, fc=OBSTACLE_COLOR, ec="none", alpha=0.85))
    if x0 is not None:
        colors = np.full(len(x0), AGENT_COLOR, dtype=object)
        if informed is not None:
            colors[np.asarray(informed, dtype=bool)] = INFORMED_COLOR
        for p, col in zip(x0, colors):
            ax.add_patch(Circle(tuple(p), agent_radius, fc=col, ec="none"))
    pad = 0.3
    ax.set_xlim(lx - pad, ux + pad)
    ax.set_ylim(ly - pad, uy + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():   # the domain rectangle is the only border
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9)
    return ax
