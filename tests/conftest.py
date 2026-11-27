"""Test-wide setup. float64 must be enabled before any JAX array exists."""

import pytest

from swarm.config import configure_jax, load_sim_params

configure_jax("float64")


@pytest.fixture(scope="session")
def sim():
    """The project's base SimParams (configs/base.yaml)."""
    return load_sim_params()
