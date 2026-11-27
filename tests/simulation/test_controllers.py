import jax
import jax.numpy as jnp
import numpy as np
import pytest

from swarm.controllers import (
    MASKS,
    ControllerSpec,
    behavior_weights,
    bounds,
    logits_from_weights,
    pack,
    preactivation,
    regularizer,
    time_basis,
    time_basis_mean,
    unpack,
    warm_start,
    weights_from_logits,
)

W_MAX = (4.0,) * 5
MU_Z = jnp.array([0.3, 0.7, 0.5, 0.2])


@pytest.mark.parametrize(
    "kind,mask,dim",
    [("fixed", "full", 5), ("fixed", "no_alignment", 4), ("fixed", "no_alignment_cohesion", 3),
     ("adaptive", "full", 25), ("adaptive", "no_cohesion", 20), ("adaptive", "no_alignment_cohesion", 15),
     ("time", "full", 25)],
)
def test_parameter_dimensions(kind, mask, dim, sim):
    spec = ControllerSpec(kind, MASKS[mask])
    assert spec.n_params == dim
    lo, hi = bounds(spec, sim)
    assert lo.shape == hi.shape == (dim,) and np.all(lo == -4) and np.all(hi == 4)


@pytest.mark.parametrize("kind", ["fixed", "adaptive", "time"])
@pytest.mark.parametrize("mask", ["full", "no_alignment", "no_alignment_cohesion"])
def test_pack_unpack_roundtrip(kind, mask):
    spec = ControllerSpec(kind, MASKS[mask])
    theta = jax.random.normal(jax.random.key(0), (spec.n_params,))
    A, b = unpack(theta, spec)
    assert A.shape == (5, spec.n_inputs) and b.shape == (5,)
    inactive = np.array([i for i in range(5) if not spec.mask[i]], dtype=int)
    np.testing.assert_array_equal(np.asarray(b)[inactive], 0.0)
    if spec.has_matrix:
        np.testing.assert_array_equal(np.asarray(A)[inactive], 0.0)
    np.testing.assert_array_equal(pack(A, b, spec), theta)


def test_unpack_rejects_wrong_length():
    with pytest.raises(ValueError):
        unpack(jnp.zeros(24), ControllerSpec("adaptive"))


@pytest.mark.parametrize("bad", [dict(kind="mlp"), dict(mask=(1, 1, 1)), dict(mask=(0,) * 5),
                                 dict(mask=(1, 2, 1, 1, 1))])
def test_spec_validation(bad):
    with pytest.raises(ValueError):
        ControllerSpec(**bad)


def test_adaptive_with_zero_matrix_equals_fixed():
    b = jnp.array([0.5, -1.0, 0.2, 1.5, 0.8])
    z = jax.random.uniform(jax.random.key(1), (7, 4))
    fixed, adaptive = ControllerSpec("fixed"), ControllerSpec("adaptive")
    s_f = preactivation(b, fixed, z, 3, MU_Z, 400)
    s_a = preactivation(warm_start(b, adaptive), adaptive, z, 3, MU_Z, 400)
    np.testing.assert_array_equal(s_f, s_a)
    s_t = preactivation(warm_start(b, ControllerSpec("time")), ControllerSpec("time"), z, 3, MU_Z, 400)
    np.testing.assert_allclose(s_f, s_t)


def test_adaptive_preactivation_is_affine_in_centered_features():
    spec = ControllerSpec("adaptive")
    theta = jax.random.normal(jax.random.key(2), (25,))
    A, b = unpack(theta, spec)
    z = jax.random.uniform(jax.random.key(3), (6, 4))
    np.testing.assert_allclose(preactivation(theta, spec, z, 0, MU_Z, 400), (z - MU_Z) @ A.T + b)
    # at the centering point the weights are the "typical environment" weights b
    np.testing.assert_allclose(preactivation(theta, spec, MU_Z[None], 0, MU_Z, 400)[0], b)


def test_time_controller_ignores_features_and_varies_in_time():
    spec = ControllerSpec("time")
    theta = jax.random.normal(jax.random.key(4), (25,))
    z1, z2 = jnp.zeros((3, 4)), jnp.ones((3, 4))
    np.testing.assert_array_equal(preactivation(theta, spec, z1, 10, MU_Z, 400),
                                  preactivation(theta, spec, z2, 10, MU_Z, 400))
    assert not np.allclose(preactivation(theta, spec, z1, 10, MU_Z, 400),
                           preactivation(theta, spec, z1, 300, MU_Z, 400))


def test_time_basis():
    np.testing.assert_allclose(time_basis(0, 400), [0, 0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(time_basis(100, 400), [0.25, 0.0625, 1.0, 0.0], atol=1e-12)
    mean = time_basis_mean(400)
    np.testing.assert_allclose(mean[2:], 0.0, atol=1e-12)          # full periods average out
    assert mean[0] == pytest.approx(399 / 800)


def test_masked_behaviors_have_exactly_zero_weight():
    s = jnp.full((4, 5), 10.0)                                    # sigmoid ~ 1 everywhere
    w = behavior_weights(s, W_MAX, MASKS["no_alignment"])
    np.testing.assert_array_equal(w[:, 1], 0.0)
    np.testing.assert_allclose(w[:, [0, 2, 3, 4]], 4.0 * jax.nn.sigmoid(10.0))


def test_regularizer():
    spec = ControllerSpec("adaptive")
    theta = jnp.concatenate([jnp.full(20, 0.5), jnp.full(5, 3.0)])
    assert float(regularizer(theta, spec)) == pytest.approx(20 * 0.25)
    assert float(regularizer(jnp.ones(5), ControllerSpec("fixed"))) == 0.0


def test_logit_weight_conversion():
    w = np.array([2.0, 1.0, 1.0, 3.0, 3.0])
    b = logits_from_weights(w, W_MAX)
    np.testing.assert_allclose(b[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(weights_from_logits(b, W_MAX), w)
    with pytest.raises(ValueError):
        logits_from_weights([4.0, 1, 1, 1, 1], W_MAX)
