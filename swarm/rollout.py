"""Rollouts, batch objectives and gradients (doc §6.4, §9.3).

    episode_terms     one episode -> LossTerms (lax.scan over T steps)
    batch_loss        mean total loss over a batch of episodes + l2_reg ||A||^2
    loss              jitted batch_loss
    loss_and_grad     jitted value_and_grad(batch_loss) (the L-BFGS-B path)
    population_loss   jitted batch_loss for a whole population of thetas (CMA-ES/BO/RS)
    simulate          one episode -> Trajectory (for plots and animation)

`spec` (SimSpec) is a static argument; theta, episodes, eta and mu_z are
dynamic, so changing eta between continuation stages does not recompile.
Use float64 (swarm.config.configure_jax) for the gradient path.

mu_z is the frozen feature centering; it is required by the adaptive
controller and ignored by the others (pass None).
"""

from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from swarm.controllers import regularizer
from swarm.dynamics import integrate
from swarm.features import N_FEATURES
from swarm.losses import LossTerms, StepTerms, combine, initial_scale, step_terms
from swarm.model import Episode, SimSpec, compute_control


def _mu(mu_z, spec: SimSpec, dtype):
    if mu_z is None:
        if spec.controller.kind == "adaptive":
            raise ValueError("the adaptive controller needs mu_z (frozen feature centering)")
        return jnp.zeros(N_FEATURES, dtype)
    return jnp.asarray(mu_z, dtype)


def _step_fn(theta, ep: Episode, spec: SimSpec, eta, mu_z, L0):
    def step(carry, t):
        x, v = carry
        ctl = compute_control(theta, x, v, t, ep, spec, eta, mu_z)
        x1, v1, u = integrate(x, v, ctl.u_raw, spec.sim)
        return (x1, v1), step_terms(x, x1, u, ep, L0, spec.sim)

    return jax.checkpoint(step) if spec.remat else step


def episode_terms(theta, ep: Episode, spec: SimSpec, eta=0.0, mu_z=None) -> LossTerms:
    """Task loss terms of one episode (no controller regularization)."""
    mu_z = _mu(mu_z, spec, ep.x0.dtype)
    L0 = initial_scale(ep.x0, ep.goal, spec.sim.loss.L0_floor)
    step = _step_fn(theta, ep, spec, eta, mu_z, L0)
    (x_final, _), per_step = jax.lax.scan(step, (ep.x0, ep.v0), jnp.arange(spec.T))
    sums = StepTerms(*(jnp.sum(s) for s in per_step))
    return combine(sums, x_final, ep, L0, ep.x0.shape[0], spec.T, spec.sim)


def batch_terms(theta, episodes: Episode, spec: SimSpec, eta=0.0, mu_z=None) -> LossTerms:
    """Episode-averaged loss terms over a batched Episode."""
    terms = jax.vmap(lambda ep: episode_terms(theta, ep, spec, eta, mu_z))(episodes)
    return LossTerms(*(jnp.mean(t) for t in terms))


def batch_loss(theta, episodes: Episode, spec: SimSpec, eta=0.0, mu_z=None):
    """SAA objective: mean task loss + l2_reg * ||A||_F^2 (adaptive / time only)."""
    task = batch_terms(theta, episodes, spec, eta, mu_z).total
    return task + spec.sim.controller.l2_reg * regularizer(theta, spec.controller)


loss = jax.jit(batch_loss, static_argnames="spec")
loss_and_grad = jax.jit(jax.value_and_grad(batch_loss), static_argnames="spec")
terms = jax.jit(batch_terms, static_argnames="spec")


@partial(jax.jit, static_argnames="spec")
def population_loss(thetas, episodes: Episode, spec: SimSpec, eta=0.0, mu_z=None):
    """batch_loss for every row of thetas (P, d) -> (P,), in one compiled call."""
    return jax.vmap(lambda th: batch_loss(th, episodes, spec, eta, mu_z))(thetas)


# --------------------------------------------------------------------------- #
# Trajectories (no loss; for inspection and animation)
# --------------------------------------------------------------------------- #
class Trajectory(NamedTuple):
    x: jnp.ndarray          # (T+1, N, 2)
    v: jnp.ndarray          # (T+1, N, 2)
    u: jnp.ndarray          # (T, N, 2) applied control
    weights: jnp.ndarray    # (T, N, 5) behavior weights
    dominant: jnp.ndarray   # (T, N) index of the largest |w_j f_j / nu_j|
    features: jnp.ndarray   # (T, N, 4)


@partial(jax.jit, static_argnames="spec")
def simulate(theta, ep: Episode, spec: SimSpec, eta=0.0, mu_z=None) -> Trajectory:
    mu_z = _mu(mu_z, spec, ep.x0.dtype)

    def step(carry, t):
        x, v = carry
        ctl = compute_control(theta, x, v, t, ep, spec, eta, mu_z)
        x1, v1, u = integrate(x, v, ctl.u_raw, spec.sim)
        contrib = jnp.linalg.norm(ctl.weights[..., None] * ctl.forces, axis=-1)
        return (x1, v1), (x1, v1, u, ctl.weights, jnp.argmax(contrib, axis=-1), ctl.features)

    _, (xs, vs, us, ws, dom, zs) = jax.lax.scan(step, (ep.x0, ep.v0), jnp.arange(spec.T))
    return Trajectory(
        x=jnp.concatenate([ep.x0[None], xs]),
        v=jnp.concatenate([ep.v0[None], vs]),
        u=us,
        weights=ws,
        dominant=dom,
        features=zs,
    )


# --------------------------------------------------------------------------- #
# NumPy-facing closures for optimizers
# --------------------------------------------------------------------------- #
class BoundObjective(NamedTuple):
    f: Callable[[np.ndarray], float]
    f_pop: Callable[[np.ndarray], np.ndarray]
    f_and_grad: Callable[[np.ndarray], tuple[float, np.ndarray]]


def bind(spec: SimSpec, episodes: Episode, eta=0.0, mu_z=None) -> BoundObjective:
    """Fix everything but theta; inputs and outputs are NumPy (float64)."""
    dtype = episodes.x0.dtype

    def f(theta):
        return float(loss(jnp.asarray(theta, dtype), episodes, spec, eta, mu_z))

    def f_pop(thetas):
        return np.asarray(population_loss(jnp.asarray(thetas, dtype), episodes, spec, eta, mu_z))

    def f_and_grad(theta):
        val, grad = loss_and_grad(jnp.asarray(theta, dtype), episodes, spec, eta, mu_z)
        return float(val), np.asarray(grad, dtype=np.float64)

    return BoundObjective(f, f_pop, f_and_grad)
