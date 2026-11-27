import jax
import jax.numpy as jnp
import numpy as np
import pytest

from swarm import rollout
from swarm.controllers import ControllerSpec, logits_from_weights, warm_start
from swarm.model import Episode, compute_control, take

MU_Z = jnp.array([0.4, 0.8, 0.5, 0.3])
B_HAND = jnp.asarray(logits_from_weights([2, 1, 1, 3, 3], (4.0,) * 5))
OBSTACLE = dict(centers=[[5.0, 5.0]], radii=[0.8])


def _theta(spec, seed=0, scale=0.3):
    th = warm_start(B_HAND, spec.controller)
    return th + scale * jax.random.normal(jax.random.key(seed), th.shape)


def _batch(*eps) -> Episode:
    return Episode(*(jnp.stack(f) for f in zip(*eps)))


# --------------------------------------------------------------------------- #
# Gradients (doc §9.4 P4)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("hard", [False, True])
def test_autodiff_matches_central_differences(make_spec, make_episode, hard):
    """60-step rollout, partially informed, adaptive controller, swept loss."""
    spec = make_spec("adaptive", hard=hard, horizon=60)
    ep = make_episode(n_informed=6, n_agents=24, **OBSTACLE)
    eps = _batch(ep, make_episode(n_informed=6, n_agents=24, seed=1, **OBSTACLE))
    theta = _theta(spec)
    direction = jax.random.normal(jax.random.key(9), theta.shape)
    direction = direction / jnp.linalg.norm(direction)

    f = lambda th: rollout.batch_loss(th, eps, spec, 0.4, MU_Z)
    _, g = rollout.loss_and_grad(theta, eps, spec, 0.4, MU_Z)
    h = 1e-6
    fd = (f(theta + h * direction) - f(theta - h * direction)) / (2 * h)
    assert float(g @ direction) == pytest.approx(float(fd), rel=1e-5, abs=1e-9)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_remat_gives_identical_value_and_gradient(make_spec, make_episode):
    spec = make_spec("adaptive", hard=False, horizon=30)
    eps = _batch(make_episode(**OBSTACLE))
    theta = _theta(spec)
    assert spec.remat                                    # default on
    v1, g1 = rollout.loss_and_grad(theta, eps, spec, 0.2, MU_Z)
    v2, g2 = rollout.loss_and_grad(theta, eps, spec.replace(remat=False), 0.2, MU_Z)
    np.testing.assert_allclose(v1, v2, rtol=1e-12)
    np.testing.assert_allclose(g1, g2, rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------- #
# Consistency between paths and models
# --------------------------------------------------------------------------- #
def test_population_equals_loop(make_spec, make_episode):
    spec = make_spec("adaptive", horizon=40)
    eps = _batch(make_episode(**OBSTACLE), make_episode(seed=3))
    thetas = jnp.stack([_theta(spec, seed=s) for s in range(4)])
    pop = rollout.population_loss(thetas, eps, spec, 0.0, MU_Z)
    loop = [rollout.loss(th, eps, spec, 0.0, MU_Z) for th in thetas]
    np.testing.assert_allclose(pop, loop, rtol=1e-12)


def test_adaptive_with_zero_matrix_equals_fixed_rollout(make_spec, make_episode):
    eps = _batch(make_episode(**OBSTACLE))
    fixed, adaptive = make_spec("fixed"), make_spec("adaptive")
    lf = rollout.terms(B_HAND, eps, fixed, 0.0, None)
    la = rollout.terms(warm_start(B_HAND, adaptive.controller), eps, adaptive, 0.0, MU_Z)
    np.testing.assert_allclose(lf.total, la.total, rtol=1e-12)


def test_soft_with_tiny_eta_matches_hard(make_spec, make_episode):
    eps = _batch(make_episode(**OBSTACLE))
    hard, soft = make_spec("fixed", hard=True), make_spec("fixed", hard=False)
    np.testing.assert_allclose(rollout.loss(B_HAND, eps, hard, 0.0, None),
                               rollout.loss(B_HAND, eps, soft, 1e-12, None), rtol=1e-10)


def test_loss_is_invariant_to_agent_order(make_spec, make_episode):
    spec = make_spec("adaptive")
    ep = make_episode(n_informed=5, **OBSTACLE)
    perm = np.random.default_rng(0).permutation(ep.x0.shape[0])
    ep_p = ep._replace(x0=ep.x0[perm], v0=ep.v0[perm], informed=ep.informed[perm])
    theta = _theta(spec)
    np.testing.assert_allclose(rollout.loss(theta, _batch(ep), spec, 0.0, MU_Z),
                               rollout.loss(theta, _batch(ep_p), spec, 0.0, MU_Z), rtol=1e-10)


def test_regularizer_only_for_matrix_controllers(make_spec, make_episode, sim):
    eps = _batch(make_episode())
    spec = make_spec("adaptive")
    theta = _theta(spec)
    task = rollout.terms(theta, eps, spec, 0.0, MU_Z).total
    A_sq = float(jnp.sum(theta[:20] ** 2))
    np.testing.assert_allclose(rollout.loss(theta, eps, spec, 0.0, MU_Z),
                               task + sim.controller.l2_reg * A_sq, rtol=1e-12)
    fixed = make_spec("fixed")
    np.testing.assert_allclose(rollout.loss(B_HAND, eps, fixed, 0.0, None),
                               rollout.terms(B_HAND, eps, fixed, 0.0, None).total)


def test_adaptive_requires_feature_centering(make_spec, make_episode):
    with pytest.raises(ValueError, match="mu_z"):
        rollout.loss(_theta(make_spec("adaptive")), _batch(make_episode()), make_spec("adaptive"), 0.0, None)


# --------------------------------------------------------------------------- #
# Leakage at the level of the full control law (doc §8.2)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("hard", [True, False])
def test_uninformed_control_does_not_depend_on_goal(make_spec, make_episode, hard):
    spec = make_spec("adaptive", hard=hard)
    ep = make_episode(n_informed=3, **OBSTACLE)
    moved = ep._replace(goal=jnp.array([6.0, 9.0]), goal_radius=jnp.asarray(0.9))
    x, v = ep.x0, 0.3 * jax.random.normal(jax.random.key(1), ep.x0.shape)
    theta = _theta(spec)
    c1 = compute_control(theta, x, v, 5, ep, spec, 0.4, MU_Z)
    c2 = compute_control(theta, x, v, 5, moved, spec, 0.4, MU_Z)
    un = ~np.asarray(ep.informed)
    np.testing.assert_array_equal(np.asarray(c1.u_raw)[un], np.asarray(c2.u_raw)[un])
    assert np.abs(np.asarray(c1.u_raw)[~un] - np.asarray(c2.u_raw)[~un]).max() > 1e-3


def test_all_informed_equals_fully_informed_default(make_spec, make_episode):
    spec = make_spec("adaptive")
    ep = make_episode(**OBSTACLE)                      # informed defaults to all True
    pi = make_episode(n_informed=ep.x0.shape[0], **OBSTACLE)
    theta = _theta(spec)
    np.testing.assert_array_equal(rollout.loss(theta, _batch(ep), spec, 0.0, MU_Z),
                                  rollout.loss(theta, _batch(pi), spec, 0.0, MU_Z))


# --------------------------------------------------------------------------- #
# Trajectories
# --------------------------------------------------------------------------- #
def test_simulate_shapes_limits_and_consistency(make_spec, make_episode, sim):
    spec = make_spec("time", horizon=80)
    ep = make_episode(**OBSTACLE)
    theta = _theta(spec)
    tr = rollout.simulate(theta, ep, spec, 0.0, None)
    n = ep.x0.shape[0]
    assert tr.x.shape == (81, n, 2) and tr.u.shape == (80, n, 2) and tr.weights.shape == (80, n, 5)
    assert tr.dominant.shape == (80, n) and tr.features.shape == (80, n, 4)
    np.testing.assert_array_equal(tr.x[0], ep.x0)
    for arr in tr:
        assert bool(jnp.all(jnp.isfinite(arr)))
    assert float(jnp.linalg.norm(tr.v, axis=-1).max()) <= sim.agent.v_max + 1e-12
    assert float(jnp.linalg.norm(tr.u, axis=-1).max()) <= sim.agent.a_max + 1e-12
    assert bool(jnp.all((tr.features >= 0) & (tr.features <= 1)))
    # the recorded trajectory is the one the loss integrates
    np.testing.assert_allclose(tr.x[1:] - tr.x[:-1], sim.simulation.dt * tr.v[1:], atol=1e-12)


def test_time_controller_uses_clock_horizon(make_spec, make_episode):
    ep = make_episode()
    spec = make_spec("time", horizon=40)
    theta = _theta(spec, scale=1.0)
    a = rollout.simulate(theta, ep, spec, 0.0, None).weights
    b = rollout.simulate(theta, ep, spec.replace(clock_horizon=400), 0.0, None).weights
    assert not np.allclose(a, b)


def test_bind_returns_numpy(make_spec, make_episode):
    spec = make_spec("fixed", horizon=20)
    obj = rollout.bind(spec, _batch(make_episode()), 0.0, None)
    theta = np.asarray(B_HAND)
    val = obj.f(theta)
    pop = obj.f_pop(np.stack([theta, theta + 0.1]))
    v2, g = obj.f_and_grad(theta)
    assert isinstance(val, float) and isinstance(pop, np.ndarray) and pop.shape == (2,)
    assert v2 == pytest.approx(val) and g.shape == (5,) and g.dtype == np.float64
