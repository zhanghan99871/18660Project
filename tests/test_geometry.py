import jax
import jax.numpy as jnp
import numpy as np
import pytest

from swarm.geometry import (
    clipnorm,
    safe_norm,
    smoothstep,
    swept_closest,
    swept_geometry,
    unit_direction,
    wall_clearance,
    zero_norm,
)

EPS = 1e-6
ZERO = jnp.zeros(2)


def _finite(tree):
    return all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(tree))


# --------------------------------------------------------------------------- #
# Regularized norms
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fn",
    [
        lambda z: safe_norm(z),
        lambda z: zero_norm(z),
        lambda z: unit_direction(z),
        lambda z: clipnorm(z, 1.5),
    ],
    ids=["safe_norm", "zero_norm", "unit_direction", "clipnorm"],
)
def test_jacobians_finite_at_zero_vector(fn):
    assert _finite(jax.jacfwd(fn)(ZERO))
    assert _finite(jax.jacrev(fn)(ZERO))


def test_values_at_zero():
    assert float(safe_norm(ZERO)) == pytest.approx(EPS)
    assert float(zero_norm(ZERO)) == 0.0
    np.testing.assert_array_equal(unit_direction(ZERO), ZERO)
    np.testing.assert_array_equal(clipnorm(ZERO, 1.5), ZERO)


def test_zero_norm_underestimates_by_at_most_eps():
    z = jax.random.normal(jax.random.key(0), (1000, 2)) * jnp.logspace(-8, 2, 1000)[:, None]
    true = jnp.sqrt(jnp.sum(z * z, axis=-1))
    gap = true - zero_norm(z)
    assert bool(jnp.all(gap >= -1e-15))
    assert bool(jnp.all(gap <= EPS + 1e-15))


def test_unit_direction_norm_below_one_and_close_for_long_vectors():
    z = jax.random.normal(jax.random.key(1), (500, 2))
    n = jnp.sqrt(jnp.sum(unit_direction(z) ** 2, axis=-1))
    assert bool(jnp.all(n < 1.0))
    np.testing.assert_allclose(unit_direction(jnp.array([3.0, 4.0])), [0.6, 0.8], atol=1e-12)


def test_clipnorm_bounds_norm_and_preserves_short_vectors():
    z = jax.random.normal(jax.random.key(2), (500, 2)) * 5.0
    out = clipnorm(z, 1.5)
    assert bool(jnp.all(jnp.sqrt(jnp.sum(out**2, -1)) <= 1.5 + 1e-12))
    short = jnp.array([[0.3, -0.4], [1e-3, 0.0]])
    np.testing.assert_array_equal(clipnorm(short, 1.5), short)
    # direction is preserved
    long = jnp.array([10.0, 0.0])
    np.testing.assert_allclose(clipnorm(long, 1.5), [1.5, 0.0], rtol=1e-9)


# --------------------------------------------------------------------------- #
# smoothstep
# --------------------------------------------------------------------------- #
def test_smoothstep_endpoints_clipping_and_monotonicity():
    assert float(smoothstep(0.0)) == 0.0
    assert float(smoothstep(1.0)) == 1.0
    assert float(smoothstep(0.5)) == pytest.approx(0.5)
    assert float(smoothstep(-3.0)) == 0.0 and float(smoothstep(7.0)) == 1.0
    q = jnp.linspace(-0.5, 1.5, 401)
    assert bool(jnp.all(jnp.diff(smoothstep(q)) >= 0))
    d = jax.grad(smoothstep)
    assert float(d(0.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(d(1.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(d(0.5)) == pytest.approx(1.5)


# --------------------------------------------------------------------------- #
# Swept closest point
# --------------------------------------------------------------------------- #
def _brute_force_closest(p0, p1, n=20001):
    s = np.linspace(0.0, 1.0, n)[:, None]
    pts = p0[None] + s * (p1 - p0)[None]
    return pts[np.argmin(np.sum(pts**2, axis=-1))]


def test_swept_closest_matches_dense_sampling():
    rng = np.random.default_rng(0)
    p0 = rng.normal(size=(200, 2))
    p1 = rng.normal(size=(200, 2))
    got = np.asarray(swept_closest(jnp.asarray(p0), jnp.asarray(p1)))
    for i in range(len(p0)):
        ref = _brute_force_closest(p0[i], p1[i])
        assert np.linalg.norm(got[i]) <= np.linalg.norm(ref) + 1e-12
        np.testing.assert_allclose(np.linalg.norm(got[i]), np.linalg.norm(ref), atol=1e-4)


def test_swept_closest_branches():
    p = jnp.array([0.3, -0.2])
    # zero relative motion: closest point is the (only) endpoint
    np.testing.assert_array_equal(swept_closest(p, p), p)
    # moving away: closest at the start
    np.testing.assert_allclose(swept_closest(jnp.array([1.0, 0.0]), jnp.array([2.0, 0.0])), [1.0, 0.0])
    # stops before the closest approach: closest at the end
    np.testing.assert_allclose(swept_closest(jnp.array([-2.0, 1.0]), jnp.array([-1.0, 1.0])), [-1.0, 1.0])
    # passes through the closest approach
    np.testing.assert_allclose(swept_closest(jnp.array([-1.0, 0.5]), jnp.array([1.0, 0.5])), [0.0, 0.5])


def test_grazing_contact_missed_by_endpoints_is_caught():
    # §9.4: lateral offset 0.09, relative displacement 0.15, contact distance 2 r_a = 0.1
    p0, p1 = jnp.array([-0.075, 0.09]), jnp.array([0.075, 0.09])
    endpoint_min = min(float(jnp.linalg.norm(p0)), float(jnp.linalg.norm(p1)))
    assert endpoint_min == pytest.approx(0.117, abs=1e-3) and endpoint_min > 0.1
    assert float(jnp.linalg.norm(swept_closest(p0, p1))) == pytest.approx(0.09)


@pytest.mark.parametrize(
    "p0,p1",
    [
        ([0.0, 0.0], [0.0, 0.0]),      # coincident and static
        ([0.4, 0.1], [0.4, 0.1]),      # static, apart
        ([-1.0, 0.0], [1.0, 0.0]),     # passes exactly through the origin
        ([-1.0, 0.3], [0.5, -0.2]),    # generic
    ],
)
def test_swept_distance_gradient_finite(p0, p1):
    f = lambda a, b: zero_norm(swept_closest(a, b))
    g = jax.grad(f, argnums=(0, 1))(jnp.array(p0), jnp.array(p1))
    assert _finite(g)


def test_swept_distance_gradient_matches_finite_differences():
    f = lambda a, b: zero_norm(swept_closest(a, b))
    a, b = jnp.array([-1.0, 0.3]), jnp.array([0.5, -0.2])
    ga, gb = jax.grad(f, argnums=(0, 1))(a, b)
    h = 1e-6
    for k in range(2):
        e = jnp.zeros(2).at[k].set(h)
        fd_a = (f(a + e, b) - f(a - e, b)) / (2 * h)
        fd_b = (f(a, b + e) - f(a, b - e)) / (2 * h)
        assert float(ga[k]) == pytest.approx(float(fd_a), rel=1e-6, abs=1e-9)
        assert float(gb[k]) == pytest.approx(float(fd_b), rel=1e-6, abs=1e-9)


# --------------------------------------------------------------------------- #
# Walls and swept geometry
# --------------------------------------------------------------------------- #
def test_wall_clearance():
    lo, hi, r_a = jnp.zeros(2), jnp.array([10.0, 10.0]), 0.05
    x = jnp.array([[5.0, 5.0], [0.05, 3.0], [9.0, 9.99], [-0.2, 5.0]])
    np.testing.assert_allclose(wall_clearance(x, lo, hi, r_a), [4.95, 0.0, -0.04, -0.25], atol=1e-12)


def test_swept_geometry_shapes_and_consistency():
    key = jax.random.key(3)
    k0, k1 = jax.random.split(key)
    x0 = jax.random.uniform(k0, (6, 2), minval=1.0, maxval=9.0)
    x1 = x0 + 0.1 * jax.random.normal(k1, (6, 2))
    centers = jnp.array([[5.0, 5.0], [2.0, 8.0], [0.0, 0.0]])
    lo, hi = jnp.zeros(2), jnp.array([10.0, 10.0])
    pair, obs, wall = swept_geometry(x0, x1, centers, lo, hi, 0.05)
    assert pair.shape == (6, 6, 2) and obs.shape == (6, 3, 2) and wall.shape == (6,)
    # antisymmetric pair geometry, zero on the diagonal
    np.testing.assert_allclose(pair, -jnp.swapaxes(pair, 0, 1), atol=1e-12)
    np.testing.assert_array_equal(pair[jnp.arange(6), jnp.arange(6)], 0.0)
    # swept distance never exceeds either endpoint distance
    d_end = jnp.minimum(
        jnp.linalg.norm(x0[:, None] - centers[None], axis=-1),
        jnp.linalg.norm(x1[:, None] - centers[None], axis=-1),
    )
    assert bool(jnp.all(jnp.linalg.norm(obs, axis=-1) <= d_end + 1e-12))
    np.testing.assert_allclose(
        wall, jnp.minimum(wall_clearance(x0, lo, hi, 0.05), wall_clearance(x1, lo, hi, 0.05))
    )
