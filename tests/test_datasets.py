import dataclasses
from collections import Counter

import numpy as np
import pytest

from swarm.config import deep_update, load_sim_params
from swarm.datasets import (
    SplitSpec,
    StaleDatasetError,
    assign_scenarios,
    build_split,
    is_current,
    load_split,
    parse_data_config,
    read_split,
    save_split,
    split_geometry,
    split_hash,
)
from swarm.episodes import EpisodeBatch, n_informed
from swarm.feasibility import check_map
from swarm.episodes import map_from_batch

TINY_SPLITS = {
    "tiny": {"mixture": "train_id", "n_maps": 4, "episodes_per_map": 3, "n_agents": 12,
             "roles": {"adapt": 1, "eval": 2}},
    "tiny_density": {"mixture": "train_id", "n_maps": 2, "episodes_per_map": 1, "n_agents": 120,
                     "size_setting": "fixed_density"},
    "tiny_twin": {"mixture": "train_id", "n_maps": 4, "episodes_per_map": 3, "n_agents": 12,
                  "roles": {"adapt": 1, "eval": 2}},
}


@pytest.fixture(scope="module")
def tiny_cfg(data_cfg):
    raw = dict(data_cfg.raw)
    raw["splits"] = TINY_SPLITS
    return parse_data_config(raw)


@pytest.fixture(scope="module")
def tiny(sim, tiny_cfg):
    return build_split(tiny_cfg.split("tiny"), sim, tiny_cfg)


# --------------------------------------------------------------------------- #
# Config parsing
# --------------------------------------------------------------------------- #
def test_project_splits_match_the_doc(data_cfg):
    s = data_cfg.splits
    assert s["exp1_train"].n_maps == 16
    assert s["exp2_train"].n_maps == 32
    assert (s["exp2_train_single_bottleneck"].n_maps, s["exp2_train_single_bottleneck"].episodes_per_map) == (1, 32)
    assert s["test_id"].n_maps == 100 and s["test_ood"].n_maps == 100
    assert s["test_ood"].mixture == "ood"
    scale = {(v.n_agents, v.size_setting) for k, v in s.items() if k.startswith("test_scale")}
    assert scale == {(n, st) for n in (20, 50, 100) for st in ("fixed_area", "fixed_density")}
    assert all(v.n_maps == 50 for k, v in s.items() if k.startswith("test_scale"))
    assert (s["exp3_train"].n_maps, s["exp3_val"].n_maps) == (96, 16)
    assert s["exp3_test_interp"].n_maps + s["exp3_test_extrap"].n_maps == 32
    for k in ("exp3_train", "exp3_val", "exp3_test_interp", "exp3_test_extrap"):
        assert s[k].roles == (("adapt", 8), ("eval", 16))


@pytest.mark.parametrize(
    "patch,error",
    [
        ({"splits": {"bad": {"mixture": "nope", "n_maps": 1, "episodes_per_map": 1, "n_agents": 5}}}, ValueError),
        ({"splits": {"bad": {"mixture": "train_id", "n_maps": 1, "episodes_per_map": 2, "n_agents": 5,
                             "roles": {"a": 1}}}}, ValueError),
        ({"splits": {"bad": {"mixture": "train_id", "n_maps": 1, "episodes_per_map": 1, "n_agents": 5,
                             "typo": 1}}}, KeyError),
        ({"mixtures": {"train_id": ["open", "maze"]}}, ValueError),
    ],
)
def test_invalid_data_configs_are_rejected(data_cfg, patch, error):
    with pytest.raises(error):
        parse_data_config(deep_update(data_cfg.raw, patch))


def test_split_geometry(sim):
    area = SplitSpec("a", "train_id", 1, 1, 100)
    dens = SplitSpec("d", "train_id", 1, 1, 120, size_setting="fixed_density")
    g = split_geometry(area, sim)
    assert g.world_size == (10.0, 10.0) and g.horizon == 400 and g.count_scale == 1.0
    g = split_geometry(dens, sim)
    assert g.world_size == pytest.approx((20.0, 20.0)) and g.horizon == 800
    assert g.count_scale == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #
def test_scenario_assignment_is_balanced(data_cfg):
    spec = SplitSpec("balance", "train_id", 18, 1, 30)
    names = assign_scenarios(spec, data_cfg)
    counts = Counter(names)
    assert set(counts) == set(data_cfg.mixtures["train_id"])
    assert max(counts.values()) - min(counts.values()) <= 1
    assert names == assign_scenarios(spec, data_cfg)


def test_built_split_structure(tiny, sim, tiny_cfg):
    batch, meta = tiny
    assert batch.n_episodes == 12 and batch.n_agents == 12
    np.testing.assert_array_equal(batch.map_index, np.repeat(np.arange(4), 3))
    np.testing.assert_array_equal(batch.episode_index, np.tile(np.arange(3), 4))
    np.testing.assert_array_equal(batch.role, np.tile([0, 1, 1], 4))
    assert meta["role_names"] == ["adapt", "eval"]
    assert meta["hash"] == split_hash(tiny_cfg.split("tiny"), sim, tiny_cfg)
    assert len(meta["rejections"]) == len(meta["map_scenarios"]) == 4
    # episodes of the same map share the map but not the initial state
    np.testing.assert_array_equal(batch.centers[0], batch.centers[1])
    assert not np.array_equal(batch.x0[0], batch.x0[1])
    # every stored map is feasible and every state valid
    for b in range(batch.n_episodes):
        m = map_from_batch(batch, b)
        assert m.scenario == meta["map_scenarios"][batch.map_index[b]]
        assert check_map(m, sim, tiny_cfg.feasibility_resolution).ok
        assert np.all(batch.x0[b] >= batch.start_lo[b]) and np.all(batch.x0[b] <= batch.start_hi[b])
    assert batch.informed(0.2).sum(axis=1).tolist() == [n_informed(0.2, 12)] * 12


def test_fixed_density_split_metadata(sim, tiny_cfg):
    batch, meta = build_split(tiny_cfg.split("tiny_density"), sim, tiny_cfg)
    assert meta["world_size"] == pytest.approx([20.0, 20.0])
    assert meta["horizon"] == 800
    np.testing.assert_allclose(batch.upper, 20.0)


def test_build_is_deterministic_and_splits_are_independent(tiny, sim, tiny_cfg):
    batch, _ = tiny
    again, _ = build_split(tiny_cfg.split("tiny"), sim, tiny_cfg)
    for a, b in zip(batch, again):
        np.testing.assert_array_equal(a, b)
    twin, _ = build_split(tiny_cfg.split("tiny_twin"), sim, tiny_cfg)   # same spec, other name
    assert not np.array_equal(batch.goal, twin.goal)
    assert not np.array_equal(batch.x0, twin.x0)


@pytest.mark.parametrize("name", [
    "exp1_train", "exp2_train_single_bottleneck", "test_ood", "test_scale_n100_fixed_area",
    "test_scale_n100_fixed_density", "exp3_test_extrap",
])
def test_every_kind_of_project_split_can_be_generated(sim, data_cfg, name):
    spec = data_cfg.split(name)
    small = dataclasses.replace(spec, n_maps=min(spec.n_maps, 3), episodes_per_map=1, roles=())
    batch, meta = build_split(small, sim, data_cfg)
    assert batch.n_agents == spec.n_agents
    assert batch.n_episodes == small.n_maps
    assert set(meta["map_scenarios"]) <= set(data_cfg.mixtures[spec.mixture])


# --------------------------------------------------------------------------- #
# Cache I/O and staleness
# --------------------------------------------------------------------------- #
def test_save_and_load_roundtrip(tiny, sim, tiny_cfg, tmp_path):
    batch, meta = tiny
    save_split(batch, meta, tmp_path)
    loaded, loaded_meta = load_split("tiny", sim, tiny_cfg, tmp_path)
    assert isinstance(loaded, EpisodeBatch)
    for a, b in zip(batch, loaded):
        np.testing.assert_array_equal(a, b)
        assert a.dtype == b.dtype
    assert loaded_meta == meta
    assert is_current("tiny", sim, tiny_cfg, tmp_path)


def test_stale_cache_is_refused(tiny, sim, tiny_cfg, tmp_path):
    batch, meta = tiny
    save_split(batch, meta, tmp_path)
    new_seed = parse_data_config(deep_update(tiny_cfg.raw, {"seed": 1}))
    with pytest.raises(StaleDatasetError):
        load_split("tiny", sim, new_seed, tmp_path)
    bigger_agents = load_sim_params(overrides={"agent": {"radius": 0.06}})
    with pytest.raises(StaleDatasetError):
        load_split("tiny", bigger_agents, tiny_cfg, tmp_path)
    assert not is_current("tiny", sim, new_seed, tmp_path)
    # reading without the check still works
    read_split("tiny", tmp_path)


def test_settings_that_do_not_affect_data_keep_the_cache_valid(tiny, sim, tiny_cfg, tmp_path):
    batch, meta = tiny
    save_split(batch, meta, tmp_path)
    reweighted = load_sim_params(overrides={"loss": {"weights": {"energy": 0.2}},
                                            "controller": {"search_box": [-3.0, 3.0]}})
    load_split("tiny", reweighted, tiny_cfg, tmp_path)   # no error
    other_split_changed = parse_data_config(
        deep_update(tiny_cfg.raw, {"splits": {"tiny_density": {"n_maps": 5}}}))
    load_split("tiny", sim, other_split_changed, tmp_path)


def test_missing_split_raises(sim, tiny_cfg, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split("tiny", sim, tiny_cfg, tmp_path)
