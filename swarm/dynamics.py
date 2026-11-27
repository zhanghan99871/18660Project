"""Damped second-order dynamics with semi-implicit Euler (doc §3.1).

    u      = clipnorm_{a_max}(u_raw)
    v'     = clipnorm_{v_max}(v + dt (u - beta v))
    x'     = x + dt v'
"""

from __future__ import annotations

from swarm.config import SimParams
from swarm.geometry import clipnorm


def integrate(x, v, u_raw, sim: SimParams, dt: float | None = None):
    """One step. Returns (x_next, v_next, u) where u is the applied (clipped) control."""
    a, eps = sim.agent, sim.numerics.eps
    dt = sim.simulation.dt if dt is None else dt
    u = clipnorm(u_raw, a.a_max, eps)
    v_next = clipnorm(v + dt * (u - a.damping * v), a.v_max, eps)
    return x + dt * v_next, v_next, u
