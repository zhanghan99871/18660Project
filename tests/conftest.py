"""Test-wide setup. float64 must be enabled before any JAX array exists."""

import numpy as np
import pytest

from swarm.config import configure_jax, load_sim_params

configure_jax("float64")

from swarm.datasets import load_data_config  # noqa: E402
from swarm.maps import MapSpec  # noqa: E402


@pytest.fixture(scope="session")
def sim():
    """The project's base SimParams (configs/base.yaml)."""
    return load_sim_params()


@pytest.fixture(scope="session")
def data_cfg():
    """The project's data config (configs/data.yaml)."""
    return load_data_config()


@pytest.fixture
def make_map(sim):
    """Factory for hand-built maps: 10x10 domain, start box left, goal right.

    The default start box (2.2 x 4) holds 30 agents at the project spacing.
    """

    def _make(
        centers=(),
        radii=(),
        size=(10.0, 10.0),
        start_lo=(0.8, 3.0),
        start_hi=(3.0, 7.0),
        goal=(8.5, 5.0),
        goal_radius=None,
        n_agents=30,
        scenario="open",
    ) -> MapSpec:
        return MapSpec(
            scenario=scenario,
            n_agents=n_agents,
            lower=np.zeros(2),
            upper=np.asarray(size, dtype=float),
            centers=np.asarray(centers, dtype=float).reshape(-1, 2),
            radii=np.asarray(radii, dtype=float).reshape(-1),
            start_lo=start_lo,
            start_hi=start_hi,
            goal=goal,
            goal_radius=sim.goal_radius(n_agents) if goal_radius is None else goal_radius,
        )

    return _make
