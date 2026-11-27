import jax
import jax.numpy as jnp
import numpy as np
import pytest

from swarm.perception import alignment, cohesion, local_interactions, neighbor_weights, neighborhood

R, R_SEP, ELL_S = 1.0, 0.3, 0.05


@pytest.mark.parametrize("hard", [True, False])
def test_isolated_agents_feel_nothing(hard):
    x = jnp.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    v = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    sep, ali, coh, nb = local_interactions(x, v, R, R_SEP, 0.4, ELL_S, hard)
    for f in (sep, ali, coh, nb.count):
        np.testing.assert_array_equal(f, 0.0)


@pytest.mark.parametrize("eta", [0.05, 0.4])
def test_soft_weights_have_compact_support(eta):
    d = jnp.linspace(0.0, 2.0, 2001)
    w = neighbor_weights(d, R, eta, hard=False)
    np.testing.assert_array_equal(w[d >= R], 0.0)
    np.testing.assert_array_equal(w[d <= (1 - eta) * R], 1.0)
    assert bool(jnp.all(jnp.diff(w) <= 0))


def test_hard_weights_and_zero_diagonal(random_state):
    x, _ = random_state(n=10)
    nb = neighborhood(x, R, R_SEP, 0.4, hard=True)
    dist = np.linalg.norm(np.asarray(x)[:, None] - np.asarray(x)[None], axis=-1)
    expected = (dist < R) & ~np.eye(10, dtype=bool)
    np.testing.assert_array_equal(np.asarray(nb.a) > 0, expected)
    np.testing.assert_array_equal(np.diag(nb.a), 0.0)
    np.testing.assert_array_equal(np.diag(nb.s), 0.0)


def test_soft_weights_approach_hard_as_eta_shrinks(random_state):
    x, _ = random_state(n=15, seed=3)
    hard = neighborhood(x, R, R_SEP, 0.4, hard=True).a
    soft = neighborhood(x, R, R_SEP, 1e-9, hard=False).a
    np.testing.assert_allclose(soft, hard, atol=1e-12)


@pytest.mark.parametrize("hard", [True, False])
def test_permutation_equivariance(random_state, hard):
    x, v = random_state(n=12, seed=1)
    perm = np.random.default_rng(0).permutation(12)
    out = local_interactions(x, v, R, R_SEP, 0.4, ELL_S, hard)
    out_p = local_interactions(x[perm], v[perm], R, R_SEP, 0.4, ELL_S, hard)
    for f, fp in zip(out[:3], out_p[:3]):
        np.testing.assert_allclose(fp, f[perm], atol=1e-12)


def test_alignment_and_cohesion_formulas():
    # agent 0 has two hard neighbors; m = n + 1 = 3
    x = jnp.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]])
    v = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    nb = neighborhood(x, R, R_SEP, 0.4, hard=True)
    np.testing.assert_allclose(alignment(nb, v)[0], [1.0 / 3, 2.0 / 3])
    np.testing.assert_allclose(cohesion(nb)[0], [0.5 / 3, 0.5 / 3])
    # |f_c| < r for random swarms
    x = jax.random.uniform(jax.random.key(2), (40, 2), maxval=3.0)
    for hard in (True, False):
        f_c = cohesion(neighborhood(x, R, R_SEP, 0.4, hard))
        assert bool(jnp.all(jnp.linalg.norm(f_c, axis=-1) < R))


def test_soft_transition_is_smooth_with_self_inclusive_average():
    """Doc §9.4: one neighbor crossing r at eta = 0.4 -> 10-90% band ~0.24, max slope ~2.1."""

    def f_align(d):
        x = jnp.stack([jnp.zeros(2), jnp.array([d, 0.0])])
        v = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        return alignment(neighborhood(x, R, R_SEP, 0.4, hard=False), v)[0, 0]

    d = jnp.linspace(0.5, 1.05, 5501)
    f = jax.vmap(f_align)(d)
    lo, hi = 0.1 * f.max(), 0.9 * f.max()
    band = float(d[jnp.argmax(f <= lo)] - d[jnp.argmax(f <= hi)])
    slope = float(jnp.max(jnp.abs(jax.vmap(jax.grad(f_align))(d))))
    assert band == pytest.approx(0.24, abs=0.015)
    assert slope == pytest.approx(2.1, abs=0.1)


def test_separation_is_bounded_and_smooth_at_coincidence():
    def f_sep(x0, x1):
        x = jnp.stack([x0, x1])
        sep, *_ = local_interactions(x, jnp.zeros((2, 2)), R, R_SEP, 0.4, ELL_S, hard=True)
        return sep[0]

    same = jnp.array([1.0, 1.0])
    np.testing.assert_array_equal(f_sep(same, same), 0.0)
    jac = jax.jacfwd(f_sep)(same, same)
    np.testing.assert_allclose(jac, (R_SEP / ELL_S**2) * np.eye(2))        # 120 I
    d = jnp.linspace(1e-4, R_SEP * 0.999, 3000)
    mags = jax.vmap(lambda s: jnp.linalg.norm(f_sep(jnp.zeros(2), jnp.array([s, 0.0]))))(d)
    assert float(mags.max()) == pytest.approx(R_SEP / (2 * ELL_S), rel=1e-4)  # 3.0


def test_separation_is_summed_not_averaged():
    one = jnp.array([[0.0, 0.0], [0.1, 0.0]])
    two = jnp.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.0]])
    s1, *_ = local_interactions(one, jnp.zeros((2, 2)), R, R_SEP, 0.4, ELL_S, hard=True)
    s2, *_ = local_interactions(two, jnp.zeros((3, 2)), R, R_SEP, 0.4, ELL_S, hard=True)
    np.testing.assert_allclose(s2[0], 2 * s1[0])
