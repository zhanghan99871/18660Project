import jax
import jax.numpy as jnp
import numpy as np
import pytest

from swarm.episodes import (
    EpisodeBatch,
    InitialState,
    VelocityParams,
    informed_mask,
    map_from_batch,
    n_informed,
    sample_initial_state,
    sample_positions,
    stack_episodes,
    to_jax,
)

ZERO_V = VelocityParams("zero", 0.0)
ISO_V = VelocityParams("isotropic", 0.5)
SPACING, R_A = 0.25, 0.05


def _state(m, episode=0, velocity=ZERO_V, split="test_episodes", seed=7, map_index=0):
    return sample_initial_state(m, seed, split, map_index, episode, SPACING, R_A, velocity)


# --------------------------------------------------------------------------- #
# Initial positions and velocities
# --------------------------------------------------------------------------- #
def test_positions_inside_start_box_and_spaced(make_map):
    m = make_map(start_lo=(1.0, 3.0), start_hi=(3.5, 7.0))
    x = _state(m).x0
    assert x.shape == (30, 2)
    assert np.all(x >= m.start_lo + R_A) and np.all(x <= m.start_hi - R_A)
    d = np.linalg.norm(x[:, None] - x[None], axis=-1) + np.eye(30) * 1e9
    assert d.min() >= SPACING


def test_overfull_start_box_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="placed only"):
        sample_positions(rng, np.zeros(2), np.ones(2), 100, SPACING, R_A, max_attempts_per_agent=50)
    with pytest.raises(ValueError, match="too small"):
        sample_positions(rng, np.zeros(2), np.full(2, 0.08), 1, SPACING, R_A)


def test_velocity_modes(make_map):
    m = make_map()
    np.testing.assert_array_equal(_state(m).v0, 0.0)
    v = _state(m, velocity=ISO_V).v0
    speed = np.linalg.norm(v, axis=1)
    assert np.all(speed <= 0.5) and speed.max() > 0
    with pytest.raises(ValueError):
        VelocityParams("towards_goal", 1.0)


def test_order_is_a_permutation(make_map):
    order = _state(make_map()).order
    assert sorted(order.tolist()) == list(range(30))


# --------------------------------------------------------------------------- #
# Leakage and stream independence (doc §4.5, §8.2)
# --------------------------------------------------------------------------- #
def test_changing_only_the_goal_leaves_initial_state_unchanged(make_map):
    a = make_map(goal=(8.5, 5.0))
    b = a.replace(goal=np.array([7.0, 2.0]))
    c = a.replace(centers=np.array([[5.0, 5.0]]), radii=np.array([1.0]))   # obstacles too
    for other in (b, c):
        for velocity in (ZERO_V, ISO_V):
            s1, s2 = _state(a, velocity=velocity), _state(other, velocity=velocity)
            np.testing.assert_array_equal(s1.x0, s2.x0)
            np.testing.assert_array_equal(s1.v0, s2.v0)
            np.testing.assert_array_equal(s1.order, s2.order)


def test_streams_are_independent(make_map):
    m = make_map()
    zero, iso = _state(m, velocity=ZERO_V), _state(m, velocity=ISO_V)
    np.testing.assert_array_equal(zero.x0, iso.x0)        # velocity draw doesn't shift x0
    np.testing.assert_array_equal(zero.order, iso.order)
    other_ep = _state(m, episode=1)
    assert not np.array_equal(zero.x0, other_ep.x0)
    assert not np.array_equal(zero.order, other_ep.order)
    other_split = _state(m, split="another_split")
    assert not np.array_equal(zero.x0, other_split.x0)


def test_initial_state_is_deterministic(make_map):
    m = make_map()
    s1, s2 = _state(m, velocity=ISO_V), _state(m, velocity=ISO_V)
    np.testing.assert_array_equal(s1.x0, s2.x0)
    np.testing.assert_array_equal(s1.v0, s2.v0)


# --------------------------------------------------------------------------- #
# Informed masks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "p,n,expected",
    [(1.0, 30, 30), (0.5, 30, 15), (0.2, 30, 6), (0.1, 30, 3), (0.05, 30, 2),
     (0.01, 30, 1), (0.25, 10, 3), (0.2, 20, 4), (0.1, 100, 10)],
)
def test_n_informed(p, n, expected):
    assert n_informed(p, n) == expected


@pytest.mark.parametrize("p", [0.0, -0.1, 1.5])
def test_n_informed_rejects_invalid_fraction(p):
    with pytest.raises(ValueError):
        n_informed(p, 30)


def test_informed_masks_are_nested_and_sized(make_map):
    order = _state(make_map()).order
    ps = [0.05, 0.1, 0.2, 0.5, 1.0]
    masks = [informed_mask(order, p) for p in ps]
    for p, mask in zip(ps, masks):
        assert mask.sum() == n_informed(p, 30)
    for small, large in zip(masks, masks[1:]):
        assert np.all(small <= large)                       # nested
    assert masks[-1].all()
    # the informed agents are exactly the prefix of the order
    assert set(np.flatnonzero(masks[2])) == set(order[:6].tolist())


def test_informed_mask_batched():
    rng = np.random.default_rng(0)
    orders = np.stack([rng.permutation(12) for _ in range(5)])
    masks = informed_mask(orders, 0.25)
    assert masks.shape == (5, 12)
    assert np.all(masks.sum(axis=1) == 3)
    for o, mk in zip(orders, masks):
        np.testing.assert_array_equal(mk, informed_mask(o, 0.25))


# --------------------------------------------------------------------------- #
# Batches
# --------------------------------------------------------------------------- #
def _batch(make_map, k_max=None):
    maps = [make_map(), make_map(centers=[[5, 5], [5, 8]], radii=[1.0, 0.5]),
            make_map(centers=[[5, 2]], radii=[0.7])]
    states = [_state(m, episode=i) for i, m in enumerate(maps)]
    return maps, stack_episodes(maps, states, [0, 1, 2], [0, 0, 0], role=[0, 1, 0], k_max=k_max)


def test_stack_pads_obstacles_with_finite_masked_entries(make_map):
    maps, batch = _batch(make_map)
    assert batch.n_episodes == 3 and batch.n_agents == 30 and batch.k_max == 2
    np.testing.assert_array_equal(batch.valid, [[0, 0], [1, 1], [1, 0]])
    assert np.all(np.isfinite(batch.centers)) and np.all(np.isfinite(batch.radii))
    np.testing.assert_array_equal(batch.radii[batch.valid == 0], 0.0)
    _, padded = _batch(make_map, k_max=8)
    assert padded.k_max == 8
    with pytest.raises(ValueError, match="k_max"):
        _batch(make_map, k_max=1)


def test_repadding_to_a_common_k(make_map):
    maps, batch = _batch(make_map)                    # K = 2
    wide = batch.with_k_max(5)
    assert wide.k_max == 5 and wide.valid.sum() == batch.valid.sum()
    np.testing.assert_array_equal(wide.centers[:, :2], batch.centers)
    np.testing.assert_array_equal(wide.radii[:, 2:], 0.0)
    back = wide.with_k_max(2)
    for a, b in zip(back, batch):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="smaller than"):
        batch.with_k_max(1)


def test_open_only_batch_keeps_one_padded_slot(make_map):
    m = make_map()
    batch = stack_episodes([m], [_state(m)], [0], [0])
    assert batch.k_max == 1 and batch.valid.sum() == 0


def test_stack_rejects_mixed_agent_counts(make_map):
    a, b = make_map(n_agents=30), make_map(n_agents=20)
    with pytest.raises(ValueError, match="single agent count"):
        stack_episodes([a, b], [_state(a), _state(b)], [0, 1], [0, 0])


def test_map_roundtrip_through_batch(make_map):
    maps, batch = _batch(make_map)
    for b, m in enumerate(maps):
        r = map_from_batch(batch, b)
        for field in ("lower", "upper", "centers", "radii", "start_lo", "start_hi", "goal"):
            np.testing.assert_array_equal(getattr(r, field), getattr(m, field))
        assert r.goal_radius == m.goal_radius and r.scenario == m.scenario


def test_batch_select_and_informed(make_map):
    _, batch = _batch(make_map)
    sub = batch.select(np.array([2, 0]))
    np.testing.assert_array_equal(sub.map_index, [2, 0])
    np.testing.assert_array_equal(sub.x0[0], batch.x0[2])
    assert batch.informed(0.2).shape == (3, 30)


def test_batch_is_a_vmappable_pytree(make_map):
    _, batch = _batch(make_map)
    jb = to_jax(batch)
    assert isinstance(jb, EpisodeBatch)
    assert len(jax.tree.leaves(jb)) == len(EpisodeBatch._fields)
    assert jb.x0.dtype == jnp.float64
    out = jax.vmap(lambda e: jnp.sum(e.valid) + jnp.sum(e.x0))(jb)
    assert out.shape == (3,)
    np.testing.assert_allclose(out, batch.valid.sum(1) + batch.x0.sum((1, 2)))
