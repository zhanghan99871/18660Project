import jax
import jax.numpy as jnp
import numpy as np
import pytest

from swarm.dynamics import integrate
from swarm.losses import StepTerms, combine, goal_excess, initial_scale, step_terms, swept_safety

R_A, MARGIN, EPS = 0.05, 0.025, 1e-6
LO, HI = jnp.zeros(2), jnp.array([10.0, 10.0])


# --------------------------------------------------------------------------- #
# Dynamics
# --------------------------------------------------------------------------- #
def test_limits_are_respected(sim):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (50, 2), maxval=10.0)
    v = 5.0 * jax.random.normal(jax.random.key(1), (50, 2))
    u_raw = 50.0 * jax.random.normal(jax.random.key(2), (50, 2))
    x1, v1, u = integrate(x, v, u_raw, sim)
    assert bool(jnp.all(jnp.linalg.norm(u, axis=-1) <= sim.agent.a_max + 1e-12))
    assert bool(jnp.all(jnp.linalg.norm(v1, axis=-1) <= sim.agent.v_max + 1e-12))
    np.testing.assert_allclose(x1, x + sim.simulation.dt * v1)     # semi-implicit: uses v'


def test_damping_without_control(sim):
    v = jnp.array([[1.0, 0.0]])
    _, v1, u = integrate(jnp.zeros((1, 2)), v, jnp.zeros((1, 2)), sim)
    np.testing.assert_array_equal(u, 0.0)
    np.testing.assert_allclose(v1, v * (1 - sim.simulation.dt * sim.agent.damping))


def test_integrate_gradient_finite_at_rest(sim):
    g = jax.grad(lambda u: jnp.sum(integrate(jnp.zeros((2, 2)), jnp.zeros((2, 2)), u, sim)[0]))(
        jnp.zeros((2, 2)))
    assert bool(jnp.all(jnp.isfinite(g)))


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def _safety(x0, x1, centers=jnp.zeros((1, 2)), radii=jnp.zeros(1), valid=jnp.zeros(1)):
    return swept_safety(x0, x1, centers, radii, valid, LO, HI, R_A, MARGIN, EPS)


def test_no_penalty_when_everything_is_far():
    x0 = jnp.array([[2.0, 2.0], [4.0, 4.0]])
    pair, obs, wall = _safety(x0, x0 + 0.05, jnp.array([[7.0, 7.0]]), jnp.array([1.0]), jnp.ones(1))
    assert float(pair) == float(obs) == float(wall) == 0.0


def test_pair_penalty_hand_computed():
    # agents end 0.08 apart: (2 r_a + delta - 0.08) / r_a = 0.9 -> 0.81
    x0 = jnp.array([[2.0, 2.0], [2.5, 2.0]])
    x1 = jnp.array([[2.0, 2.0], [2.08, 2.0]])
    pair, _, _ = _safety(x0, x1)
    assert float(pair) == pytest.approx(0.81, rel=1e-4)


def test_swept_pair_penalty_catches_a_crossing():
    # agents swap sides within one step: endpoints are far, the segment passes through contact
    x0 = jnp.array([[2.0, 2.0], [2.2, 2.0]])
    x1 = jnp.array([[2.2, 2.0], [2.0, 2.0]])
    pair, _, _ = _safety(x0, x1)
    assert float(pair) == pytest.approx(((2 * R_A + MARGIN) / R_A) ** 2, rel=1e-3)


def test_obstacle_and_wall_penalties():
    x = jnp.array([[5.0, 5.0 + 1.0 + R_A + 0.0], [0.04, 3.0]])     # touching obstacle; past the wall
    pair, obs, wall = _safety(x, x, jnp.array([[5.0, 5.0]]), jnp.array([1.0]), jnp.ones(1))
    assert float(obs) == pytest.approx((MARGIN / R_A) ** 2, rel=1e-3)
    assert float(wall) == pytest.approx(((MARGIN - (0.04 - R_A)) / R_A) ** 2, rel=1e-6)
    # padded obstacles never penalize
    _, obs_pad, _ = _safety(x, x, jnp.array([[5.0, 5.0]]), jnp.array([1.0]), jnp.zeros(1))
    assert float(obs_pad) == 0.0


@pytest.mark.parametrize(
    "x0,x1",
    [
        ([[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]),        # coincident and static
        ([[1.0, 1.0], [1.3, 1.0]], [[1.0, 1.0], [1.3, 1.0]]),        # static apart
        ([[5.0, 5.0], [3.0, 3.0]], [[5.1, 5.0], [3.0, 3.0]]),        # inside the obstacle
    ],
)
def test_safety_gradients_finite(x0, x1):
    centers, radii, valid = jnp.array([[5.0, 5.0]]), jnp.array([1.0]), jnp.ones(1)
    f = lambda a, b: sum(_safety(a, b, centers, radii, valid))
    ga, gb = jax.grad(f, argnums=(0, 1))(jnp.array(x0), jnp.array(x1))
    assert bool(jnp.all(jnp.isfinite(ga))) and bool(jnp.all(jnp.isfinite(gb)))


def test_goal_terms_and_scale():
    g = jnp.array([8.0, 5.0])
    x = jnp.array([[8.0, 5.0], [8.0 + 2.0, 5.0]])
    np.testing.assert_allclose(goal_excess(x, g, 1.0, EPS), [0.0, 1.0], atol=1e-6)
    assert float(initial_scale(jnp.array([[2.0, 5.0], [4.0, 5.0]]), g, 1.0)) == pytest.approx(5.0)
    assert float(initial_scale(jnp.array([[8.1, 5.0]]), g, 1.0)) == 1.0     # floored


def test_combine_normalizes_and_weights(sim, make_episode):
    ep = make_episode(goal=(8.5, 5.0))
    sums = StepTerms(*(jnp.array(v) for v in (60.0, 6.0, 12.0, 18.0, 30.0)))
    x_final = jnp.tile(ep.goal, (30, 1))                       # everyone at the goal center
    terms = combine(sums, x_final, ep, 2.0, 30, 10, sim)
    np.testing.assert_allclose([terms.goal, terms.time, terms.collision, terms.obstacle,
                                terms.boundary, terms.energy],
                               [0.0, 0.2, 0.02, 0.04, 0.06, 0.1], atol=1e-9)
    w = sim.loss.weights
    expected = w.time * 0.2 + w.collision * 0.02 + w.obstacle * 0.04 + w.boundary * 0.06 + w.energy * 0.1
    assert float(terms.total) == pytest.approx(expected)


def test_step_terms_progress_and_energy(sim, make_episode):
    ep = make_episode(goal=(8.5, 5.0))
    x = jnp.array([[8.5 + ep.goal_radius + 1.0, 5.0], [8.5, 5.0]])
    u = jnp.array([[3.0, 0.0], [0.0, 0.0]])
    t = step_terms(x, x, u, ep, 2.0, sim)
    assert float(t.progress) == pytest.approx(0.5, rel=1e-5)  # 1.0 outside / L0 = 2
    assert float(t.energy) == pytest.approx(1.0)              # |u|^2 / a_max^2
