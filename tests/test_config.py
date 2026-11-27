import math

import jax.numpy as jnp
import pytest

from swarm.config import (
    BEHAVIORS,
    SimParams,
    config_hash,
    deep_update,
    from_dict,
    load_sim_params,
    load_yaml,
    CONFIG_DIR,
)


def test_base_config_matches_project_doc(sim):
    # §10 initial configuration
    assert sim.world.size == (10.0, 10.0)
    assert sim.world.n_agents == 30
    assert sim.agent.radius == 0.05
    assert sim.perception.sensor_radius == 1.0
    assert sim.perception.sep_radius == 0.3
    assert sim.perception.sep_softening == sim.agent.radius
    assert sim.forces.obstacle_influence == 0.5
    assert (sim.agent.v_max, sim.agent.a_max, sim.agent.damping) == (1.5, 3.0, 0.1)
    assert (sim.simulation.dt, sim.simulation.horizon) == (0.05, 400)
    assert sim.loss.safety_margin == 0.025
    assert sim.evaluation.contact_tol == 1e-9
    assert sim.controller.w_max == (4.0,) * len(BEHAVIORS)
    assert sim.smoothing.eta_schedule == (0.4, 0.2, 0.1, 0.05)
    # §5.3 loss weights
    w = sim.loss.weights
    assert (w.goal, w.time, w.collision, w.obstacle, w.boundary, w.energy) == (
        1.0, 1.0, 10.0, 10.0, 10.0, 0.05)


def test_derived_quantities(sim):
    assert sim.braking_distance == pytest.approx(0.375)
    assert sim.goal_radius(30) == pytest.approx(0.8 * 0.3 * math.sqrt(30))  # ~1.31
    assert sim.goal_radius(100) == pytest.approx(2.4)
    assert sim.goal_length_scale(30) == pytest.approx(sim.goal_radius(30))


def test_sim_params_is_hashable_and_frozen(sim):
    # Needed to pass SimParams as a static jit argument.
    assert hash(sim) == hash(load_sim_params())
    with pytest.raises(AttributeError):
        sim.agent.radius = 1.0  # type: ignore[misc]


def test_validation_rejects_bad_constants():
    with pytest.raises(ValueError, match="braking distance"):
        load_sim_params(overrides={"forces": {"obstacle_influence": 0.3}})
    with pytest.raises(ValueError, match="sep_radius"):
        load_sim_params(overrides={"perception": {"sep_radius": 2.0}})
    with pytest.raises(ValueError, match="eta_schedule"):
        load_sim_params(overrides={"smoothing": {"eta_schedule": [0.05, 0.4]}})


def test_strict_loading_rejects_typos_and_missing_keys():
    data = load_yaml(CONFIG_DIR / "base.yaml")
    bad = deep_update(data, {"agent": {"radiuss": 0.05}})
    with pytest.raises(KeyError, match="unknown keys"):
        from_dict(SimParams, bad)
    missing = dict(data)
    del missing["goal"]
    with pytest.raises(KeyError, match="missing keys"):
        from_dict(SimParams, missing)
    wrong_type = deep_update(data, {"simulation": {"horizon": 400.5}})
    with pytest.raises(TypeError):
        from_dict(SimParams, wrong_type)


def test_overrides_and_roundtrip(sim):
    p = load_sim_params(overrides={"simulation": {"horizon": 100}})
    assert p.simulation.horizon == 100
    assert p.agent == sim.agent
    assert from_dict(SimParams, sim.to_dict()) == sim


def test_config_hash_is_order_independent():
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_float64_enabled_in_tests():
    assert jnp.ones(1).dtype == jnp.float64
