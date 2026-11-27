import jax
import jax.numpy as jnp
import numpy as np
import pytest

from swarm.forces import boundary_force, force_scales, goal_force, obstacle_force

R_A, D_S, K_T, C_T, B_T = 0.05, 0.5, 1.0, 5.0, 0.1
LO, HI = jnp.zeros(2), jnp.array([10.0, 10.0])


def _obs(x, heading, centers, radii, valid, k_t=K_T):
    return obstacle_force(x, heading, centers, radii, valid, R_A, D_S, k_t, C_T, B_T)


def test_force_scales(sim):
    np.testing.assert_allclose(force_scales(sim), [1.0, 1.5, 1.0, 1.5, 0.5])


def test_goal_force():
    g = jnp.array([5.0, 5.0])
    # at the goal center at rest: no force
    np.testing.assert_allclose(goal_force(g[None], jnp.zeros((1, 2)), g, 1.5, 1.3), 0.0, atol=1e-6)
    # far away: desired velocity ~ v_max towards the goal
    f = goal_force(jnp.array([[-20.0, 5.0]]), jnp.zeros((1, 2)), g, 1.5, 1.3)
    np.testing.assert_allclose(f, [[1.5, 0.0]], atol=1e-9)
    # velocity feedback
    f = goal_force(jnp.array([[-20.0, 5.0]]), jnp.array([[1.5, 0.0]]), g, 1.5, 1.3)
    np.testing.assert_allclose(f, 0.0, atol=1e-9)
    # decays towards the center: inside the goal the pull is weaker but inward
    inner = goal_force(jnp.array([[5.5, 5.0]]), jnp.zeros((1, 2)), g, 1.5, 1.3)
    assert -1.5 < float(inner[0, 0]) < 0.0


def test_head_on_obstacle_matches_doc_value():
    """Doc §9.4: approaching an obstacle center head-on gives force (-0.15, -0.069)."""
    R = 0.5
    x = jnp.array([[-(R + R_A + 0.35), 0.0]])            # clearance 0.35 -> push 0.15
    f, d_near = _obs(x, jnp.array([[1.0, 0.0]]), jnp.zeros((1, 2)), jnp.array([R]), jnp.ones(1))
    np.testing.assert_allclose(f, [[-0.15, -0.15 * np.tanh(C_T * B_T)]], atol=1e-6)
    np.testing.assert_allclose(d_near, [0.35], atol=1e-6)


def test_no_push_beyond_influence_distance():
    x = jnp.array([[-(1.0 + R_A + D_S + 0.01), 0.0]])
    f, _ = _obs(x, jnp.array([[1.0, 0.0]]), jnp.zeros((1, 2)), jnp.array([1.0]), jnp.ones(1))
    np.testing.assert_array_equal(f, 0.0)


def test_padded_obstacles_are_ignored_and_safe():
    x = jnp.array([[0.0, 0.0], [3.0, 3.0]])                  # first agent sits on the padding center
    heading = jnp.array([[1.0, 0.0], [0.0, 0.0]])
    centers, radii, valid = jnp.zeros((3, 2)), jnp.zeros(3), jnp.zeros(3)
    f, d_near = _obs(x, heading, centers, radii, valid)
    np.testing.assert_array_equal(f, 0.0)
    assert bool(jnp.all(jnp.isinf(d_near)))
    g = jax.grad(lambda x: jnp.sum(_obs(x, heading, centers, radii, valid)[0]))(x)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_normal_part_is_negative_gradient_of_potential():
    centers = jnp.array([[0.0, 0.0], [1.2, 0.4]])
    radii, valid = jnp.array([0.5, 0.3]), jnp.ones(2)

    def potential(x):
        diff = x[None, :] - centers
        clear = jnp.sqrt(jnp.sum(diff**2, -1) + 1e-12) - radii - R_A
        return 0.5 * jnp.sum(jnp.maximum(D_S - clear, 0.0) ** 2)

    x = jnp.array([0.6, 0.6])
    f, _ = _obs(x[None], jnp.zeros((1, 2)), centers, radii, valid, k_t=0.0)
    np.testing.assert_allclose(f[0], -jax.grad(potential)(x), atol=1e-6)


def test_tangential_term_steers_toward_the_heading_side():
    # obstacle between agent and goal, agent slightly above the center line
    x = jnp.array([[-0.8, 0.05]])
    f, _ = _obs(x, jnp.array([[1.0, 0.0]]), jnp.zeros((1, 2)), jnp.array([0.4]), jnp.ones(1))
    assert float(f[0, 1]) != 0.0 and bool(jnp.all(jnp.isfinite(f)))


def test_isolated_obstacle_matches_per_obstacle_doc_formula():
    """For a single obstacle, the combined-normal tangent equals the doc's per-obstacle form."""
    c, R = jnp.array([[2.0, 3.0]]), jnp.array([0.7])
    key = jax.random.key(5)
    x = c + jax.random.uniform(key, (40, 2), minval=-1.2, maxval=1.2)
    heading = jax.random.normal(jax.random.key(6), (40, 2))
    heading = heading / jnp.linalg.norm(heading, axis=1, keepdims=True)
    f, _ = _obs(x, heading, c, R, jnp.ones(1))
    diff = x - c
    n = diff / jnp.linalg.norm(diff, axis=1, keepdims=True)
    push = jnp.maximum(D_S - (jnp.linalg.norm(diff, axis=1) - R - R_A), 0.0)
    t = jnp.stack([-n[:, 1], n[:, 0]], -1)
    side = jnp.tanh(C_T * (jnp.sum(t * heading, -1) + B_T))
    np.testing.assert_allclose(f, push[:, None] * (n + K_T * side[:, None] * t), atol=1e-5)


def test_chained_wall_never_pushes_agents_through():
    """Regression: per-circle tangents used to push agents through overlapping-circle walls."""
    from swarm.maps import wall_circle_ys

    ys = wall_circle_ys(8.0, 1.0, 0.3, 0.3, 0.0, 10.0)
    centers = jnp.stack([jnp.full(len(ys), 5.0), jnp.asarray(ys)], -1)
    radii, valid = jnp.full(len(ys), 0.3), jnp.ones(len(ys))
    gx, gy = jnp.meshgrid(jnp.linspace(4.3, 4.95, 14), jnp.linspace(2.0, 6.5, 31))
    x = jnp.stack([gx.ravel(), gy.ravel()], -1)                   # approach side, away from the gap
    heading = jnp.tile(jnp.array([1.0, 0.0]), (x.shape[0], 1))    # goal straight through the wall
    f, _ = _obs(x, heading, centers, radii, valid)
    in_range = jnp.linalg.norm(f, axis=1) > 1e-9
    assert bool(jnp.all(f[in_range, 0] < 0.0))


def test_boundary_force():
    x = jnp.array([[5.0, 5.0], [0.25, 5.0], [5.0, 9.6], [-0.5, 5.0]])
    f, d_wall = boundary_force(x, LO, HI, R_A, D_S)
    np.testing.assert_allclose(d_wall, [4.95, 0.2, 0.35, -0.55])
    np.testing.assert_allclose(f[0], [0.0, 0.0])
    np.testing.assert_allclose(f[1], [0.3, 0.0])          # left wall pushes right
    np.testing.assert_allclose(f[2], [0.0, -0.15])        # top wall pushes down
    assert float(f[3, 0]) == pytest.approx(1.05)          # outside: strong inward push
