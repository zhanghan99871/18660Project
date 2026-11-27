"""Evaluation metrics on recorded trajectories (doc §8.3).

Contacts use exact swept geometry with *raw* distances and no safety margin:
a segment pair is in contact when its closest distance is <= the contact
distance + `contact_tol` (tol only absorbs floating-point rounding). A contact
that continues over consecutive steps counts as one event; its duration is
reported separately (`contact_steps`).

    evaluate(theta, episodes, spec, eta, mu_z)  -> EpisodeMetrics with a leading batch axis
    trajectory_metrics(traj, ep, spec)          -> EpisodeMetrics for one recorded trajectory
    summarize(metrics, spec)                    -> dict of aggregate numbers

Evaluate on the hard simulator (spec.hard=True) for reported results.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from swarm.config import SimParams
from swarm.geometry import swept_geometry
from swarm.losses import LossTerms, StepTerms, combine, initial_scale, step_terms
from swarm.model import Episode, SimSpec, take
from swarm.rollout import Trajectory, simulate


class EpisodeMetrics(NamedTuple):
    arrival: jnp.ndarray                # fraction of agents inside the goal at T
    arrival_informed: jnp.ndarray       # same, informed agents only
    arrival_uninformed: jnp.ndarray     # same, uninformed agents only (nan if none)
    success: jnp.ndarray                # arrival >= success_threshold
    collision_free: jnp.ndarray         # no contact of any kind on any segment
    pair_events: jnp.ndarray            # new agent-agent contacts
    obstacle_events: jnp.ndarray        # new agent-obstacle contacts
    wall_events: jnp.ndarray            # new agent-wall contacts
    contact_steps: jnp.ndarray          # (pair, obstacle, wall) contacts summed over steps
    min_clearance: jnp.ndarray          # smallest signed clearance over the episode
    deadlock: jnp.ndarray               # fraction of agents stalled outside the goal at the end
    completion_step: jnp.ndarray        # first step arrival >= threshold and held for the dwell; T if never
    completed: jnp.ndarray
    energy: jnp.ndarray                 # mean |u|^2 over agents and steps
    loss: LossTerms                     # task loss (swept, with margin; no regularization)
    arrival_curve: jnp.ndarray          # (T+1,) arrival fraction over time
    arrival_curve_uninformed: jnp.ndarray   # (T+1,), nan if no uninformed agents
    feature_mean: jnp.ndarray           # (4,) mean local features
    weight_mean: jnp.ndarray            # (5,) mean behavior weights


# --------------------------------------------------------------------------- #
# Contacts
# --------------------------------------------------------------------------- #
def swept_contacts(x0, x1, ep: Episode, r_a, tol):
    """Boolean contact masks for one step: pair (N, N) upper-triangular, obstacle (N, K), wall (N,)."""
    pair, obs, wall = swept_geometry(x0, x1, ep.centers, ep.lower, ep.upper, r_a)
    n = x0.shape[0]
    upper_tri = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    pair_hit = upper_tri & (jnp.sum(pair**2, axis=-1) <= (2 * r_a + tol) ** 2)
    obs_hit = (ep.valid[None, :] > 0) & (jnp.sum(obs**2, axis=-1) <= (ep.radii[None, :] + r_a + tol) ** 2)
    wall_hit = wall <= tol
    return pair_hit, obs_hit, wall_hit


def swept_min_clearance(x0, x1, ep: Episode, r_a):
    """Smallest signed clearance (pairs, valid obstacles, walls) along one step."""
    pair, obs, wall = swept_geometry(x0, x1, ep.centers, ep.lower, ep.upper, r_a)
    n = x0.shape[0]
    upper_tri = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    d_pair = jnp.where(upper_tri, jnp.sqrt(jnp.sum(pair**2, axis=-1)) - 2 * r_a, jnp.inf)
    d_obs = jnp.where(ep.valid[None, :] > 0,
                      jnp.sqrt(jnp.sum(obs**2, axis=-1)) - ep.radii[None, :] - r_a, jnp.inf)
    return jnp.minimum(jnp.minimum(jnp.min(d_pair), jnp.min(d_obs)), jnp.min(wall))


# --------------------------------------------------------------------------- #
# Metrics from a trajectory
# --------------------------------------------------------------------------- #
def _masked_mean(values, mask):
    count = jnp.sum(mask)
    return jnp.where(count > 0, jnp.sum(values * mask) / jnp.maximum(count, 1), jnp.nan)


def _completion(curve, threshold, dwell):
    """First index where curve >= threshold holds for `dwell` consecutive entries."""
    ok = curve >= threshold

    def body(run_next, ok_t):
        run = jnp.where(ok_t, run_next + 1, 0)
        return run, run

    _, runs = jax.lax.scan(body, jnp.zeros((), jnp.int32), ok, reverse=True)
    hit = runs >= dwell
    completed = jnp.any(hit)
    n_steps = curve.shape[0] - 1
    return jnp.where(completed, jnp.argmax(hit), n_steps), completed


def trajectory_metrics(traj: Trajectory, ep: Episode, spec: SimSpec) -> EpisodeMetrics:
    sim: SimParams = spec.sim
    e, r_a = sim.evaluation, sim.agent.radius
    T, n = traj.u.shape[0], traj.x.shape[1]
    L0 = initial_scale(ep.x0, ep.goal, sim.loss.L0_floor)
    k = ep.centers.shape[0]

    class Carry(NamedTuple):
        prev: tuple
        events: jnp.ndarray     # (3,) pair, obstacle, wall
        steps: jnp.ndarray
        min_clear: jnp.ndarray
        sums: StepTerms

    def body(c: Carry, inp):
        x0, x1, u = inp
        hits = swept_contacts(x0, x1, ep, r_a, e.contact_tol)
        new = jnp.stack([jnp.sum(h & ~p) for h, p in zip(hits, c.prev)]).astype(c.events.dtype)
        steps = c.steps + sum(jnp.sum(h) for h in hits).astype(c.steps.dtype)
        min_clear = jnp.minimum(c.min_clear, swept_min_clearance(x0, x1, ep, r_a))
        st = step_terms(x0, x1, u, ep, L0, sim)
        sums = StepTerms(*(a + b for a, b in zip(c.sums, st)))
        return Carry(hits, c.events + new, steps, min_clear, sums), None

    dtype = traj.x.dtype
    zero = jnp.zeros((), dtype)
    init = Carry(
        prev=(jnp.zeros((n, n), bool), jnp.zeros((n, k), bool), jnp.zeros(n, bool)),
        events=jnp.zeros(3, jnp.int32),
        steps=jnp.zeros((), jnp.int32),
        min_clear=jnp.asarray(jnp.inf, dtype),
        sums=StepTerms(zero, zero, zero, zero, zero),
    )
    c, _ = jax.lax.scan(body, init, (traj.x[:-1], traj.x[1:], traj.u))
    loss = combine(c.sums, traj.x[-1], ep, L0, n, T, sim)

    inside = jnp.linalg.norm(traj.x - ep.goal, axis=-1) <= ep.goal_radius        # (T+1, N), raw distance
    informed = ep.informed
    curve = jnp.mean(inside, axis=1)
    curve_un = jax.vmap(lambda row: _masked_mean(row, ~informed))(inside)
    arrived = inside[-1]

    speed = jnp.linalg.norm(traj.v[1:], axis=-1)                                 # (T, N)
    window = min(e.deadlock_window, T)
    stalled = jnp.all(speed[-window:] < e.deadlock_speed_fraction * sim.agent.v_max, axis=0)
    completion, completed = _completion(curve, e.success_threshold, e.completion_dwell)

    return EpisodeMetrics(
        arrival=jnp.mean(arrived),
        arrival_informed=_masked_mean(arrived, informed),
        arrival_uninformed=_masked_mean(arrived, ~informed),
        success=jnp.mean(arrived) >= e.success_threshold,
        collision_free=jnp.sum(c.events) == 0,
        pair_events=c.events[0],
        obstacle_events=c.events[1],
        wall_events=c.events[2],
        contact_steps=c.steps,
        min_clearance=c.min_clear,
        deadlock=jnp.mean(stalled & ~arrived),
        completion_step=completion,
        completed=completed,
        energy=jnp.mean(jnp.sum(traj.u**2, axis=-1)),
        loss=loss,
        arrival_curve=curve,
        arrival_curve_uninformed=curve_un,
        feature_mean=jnp.mean(traj.features, axis=(0, 1)),
        weight_mean=jnp.mean(traj.weights, axis=(0, 1)),
    )


@partial(jax.jit, static_argnames="spec")
def _evaluate_batch(theta, episodes: Episode, spec: SimSpec, eta, mu_z) -> EpisodeMetrics:
    def one(ep):
        return trajectory_metrics(simulate(theta, ep, spec, eta, mu_z), ep, spec)

    return jax.vmap(one)(episodes)


def evaluate(theta, episodes: Episode, spec: SimSpec, eta=0.0, mu_z=None, chunk: int = 64) -> EpisodeMetrics:
    """Metrics for every episode of a batched Episode, `chunk` episodes per call."""
    n_ep = episodes.x0.shape[0]
    parts = [
        _evaluate_batch(theta, take(episodes, slice(i, i + chunk)), spec, eta, mu_z)
        for i in range(0, n_ep, chunk)
    ]
    if len(parts) == 1:
        return parts[0]
    return jax.tree.map(lambda *xs: jnp.concatenate(xs), *parts)


def summarize(m: EpisodeMetrics, spec: SimSpec) -> dict[str, float]:
    """Aggregate over episodes (means; nan-aware for partially informed fields)."""
    dt = spec.sim.simulation.dt
    a = lambda v: np.asarray(v, dtype=np.float64)
    completed = a(m.completed).astype(bool)
    out = {
        "episodes": int(a(m.arrival).size),
        "success_and_collision_free": float(np.mean(a(m.success) * a(m.collision_free))),
        "success_rate": float(np.mean(a(m.success))),
        "collision_free_rate": float(np.mean(a(m.collision_free))),
        "arrival": float(np.mean(a(m.arrival))),
        "arrival_informed": float(np.nanmean(a(m.arrival_informed))),
        "arrival_uninformed": float(np.nanmean(a(m.arrival_uninformed)))
        if np.any(np.isfinite(a(m.arrival_uninformed))) else float("nan"),
        "pair_events": float(np.mean(a(m.pair_events))),
        "obstacle_events": float(np.mean(a(m.obstacle_events))),
        "wall_events": float(np.mean(a(m.wall_events))),
        "contact_steps": float(np.mean(a(m.contact_steps))),
        "min_clearance": float(np.min(a(m.min_clearance))),
        "deadlock": float(np.mean(a(m.deadlock))),
        "completed_rate": float(np.mean(completed)),
        "completion_time_s": float(np.mean(a(m.completion_step) * dt)),   # censored at T
        "completion_time_s_completed": float(np.mean(a(m.completion_step)[completed] * dt))
        if completed.any() else float("nan"),
        "energy": float(np.mean(a(m.energy))),
        "task_loss": float(np.mean(a(m.loss.total))),
    }
    for name in LossTerms._fields[:-1]:
        out[f"loss_{name}"] = float(np.mean(a(getattr(m.loss, name))))
    return out
