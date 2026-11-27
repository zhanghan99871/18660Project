"""Fixtures for the simulator tests. The root tests/conftest.py (float64, `sim`,
`data_cfg`, `make_map`) applies here as well."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from swarm.controllers import ControllerSpec
from swarm.episodes import sample_initial_state
from swarm.model import Episode, SimSpec, episode_from_map


@pytest.fixture
def random_state():
    """Factory: (x, v) for n agents spread over a box, deterministic per seed."""

    def _make(n=12, seed=0, lo=(3.0, 3.0), hi=(5.0, 5.0), speed=0.5):
        kx, kv = jax.random.split(jax.random.key(seed))
        x = jax.random.uniform(kx, (n, 2), minval=jnp.array(lo), maxval=jnp.array(hi))
        v = speed * jax.random.normal(kv, (n, 2))
        return x, v

    return _make


@pytest.fixture
def make_episode(make_map, data_cfg):
    """Factory: a JAX Episode built from a hand-made map and a sampled initial state."""

    def _make(n_informed=None, k_max=None, seed=0, **map_kwargs) -> Episode:
        m = make_map(**map_kwargs)
        st = sample_initial_state(m, seed, "sim_tests", 0, 0, data_cfg.layout.init_spacing, 0.05,
                                  data_cfg.velocity)
        informed = np.ones(m.n_agents, dtype=bool)
        if n_informed is not None:
            informed[:] = False
            informed[st.order[:n_informed]] = True
        return episode_from_map(m, st.x0, st.v0, informed, k_max=k_max)

    return _make


@pytest.fixture
def make_spec(sim):
    """Factory: SimSpec with short horizons for fast tests."""

    def _make(kind="fixed", mask=(1, 1, 1, 1, 1), hard=True, horizon=60, **kw) -> SimSpec:
        return SimSpec(sim=sim, controller=ControllerSpec(kind, tuple(mask)), hard=hard,
                       horizon=horizon, **kw)

    return _make
