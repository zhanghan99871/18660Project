"""Dataset splits: generation, caching and loading (doc §5.4 CRN, §8.1, §8.2).

Every split is generated once from `configs/data.yaml` and saved as
`data/<split>.npz` + `data/<split>.json`. All optimizers read the same files
(common random numbers). Each file carries a hash of everything that affects
its contents; `load_split` refuses a file whose hash no longer matches the
current configs, so stale data can't silently mix with new code.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from swarm.config import CONFIG_DIR, DATA_ROOT, SimParams, config_hash, load_yaml
from swarm.episodes import EpisodeBatch, VelocityParams, sample_initial_state, stack_episodes
from swarm.feasibility import check_map
from swarm.maps import SCENARIOS, LayoutParams, MapSpec, sample_map
from swarm.rng import Stream, stream_rng

# Bump whenever generator code changes what a given config produces.
GENERATOR_VERSION = 1

SIZE_SETTINGS = ("fixed_area", "fixed_density")


class StaleDatasetError(RuntimeError):
    """A cached split was built from different configs or generator code."""


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class SplitSpec:
    name: str
    mixture: str
    n_maps: int
    episodes_per_map: int
    n_agents: int
    size_setting: str = "fixed_area"
    roles: tuple[tuple[str, int], ...] = ()

    def __post_init__(self):
        if self.size_setting not in SIZE_SETTINGS:
            raise ValueError(f"{self.name}: size_setting must be one of {SIZE_SETTINGS}")
        if min(self.n_maps, self.episodes_per_map, self.n_agents) < 1:
            raise ValueError(f"{self.name}: n_maps, episodes_per_map and n_agents must be >= 1")
        if self.roles and sum(c for _, c in self.roles) != self.episodes_per_map:
            raise ValueError(f"{self.name}: role counts must sum to episodes_per_map")

    @property
    def role_names(self) -> tuple[str, ...]:
        return tuple(r for r, _ in self.roles) or ("default",)

    def role_of(self, episode: int) -> int:
        """Role index of the episode-th episode within a map (roles are consecutive)."""
        end = 0
        for i, (_, count) in enumerate(self.roles):
            end += count
            if episode < end:
                return i
        return 0

    @classmethod
    def from_config(cls, name: str, data: dict) -> "SplitSpec":
        allowed = {"mixture", "n_maps", "episodes_per_map", "n_agents", "size_setting", "roles"}
        unknown = set(data) - allowed
        if unknown:
            raise KeyError(f"split {name}: unknown keys {sorted(unknown)}")
        roles = tuple((str(k), int(v)) for k, v in (data.get("roles") or {}).items())
        return cls(
            name=name,
            mixture=data["mixture"],
            n_maps=int(data["n_maps"]),
            episodes_per_map=int(data["episodes_per_map"]),
            n_agents=int(data["n_agents"]),
            size_setting=data.get("size_setting", "fixed_area"),
            roles=roles,
        )


@dataclasses.dataclass(frozen=True, eq=False)
class DataConfig:
    raw: dict[str, Any]
    seed: int
    layout: LayoutParams
    velocity: VelocityParams
    feasibility_resolution: float
    descriptor_resolution: float
    gap_cap: float
    scenarios: dict[str, dict]
    mixtures: dict[str, list[str]]
    splits: dict[str, SplitSpec]

    def split(self, name: str) -> SplitSpec:
        if name not in self.splits:
            raise KeyError(f"unknown split {name!r}; known: {sorted(self.splits)}")
        return self.splits[name]


def parse_data_config(raw: dict[str, Any]) -> DataConfig:
    scenarios = raw["scenarios"]
    for name in scenarios:
        if name not in SCENARIOS:
            raise ValueError(f"scenario {name!r} is not registered in maps.SCENARIOS")
    mixtures = {k: list(v) for k, v in raw["mixtures"].items()}
    for mix, names in mixtures.items():
        missing = [n for n in names if n not in scenarios]
        if missing:
            raise ValueError(f"mixture {mix!r} references unknown scenarios {missing}")
    splits = {name: SplitSpec.from_config(name, spec) for name, spec in raw["splits"].items()}
    for spec in splits.values():
        if spec.mixture not in mixtures:
            raise ValueError(f"split {spec.name!r}: unknown mixture {spec.mixture!r}")
    vel = raw["initial_velocity"]
    return DataConfig(
        raw=raw,
        seed=int(raw["seed"]),
        layout=LayoutParams.from_config(raw["layout"]),
        velocity=VelocityParams(mode=vel["mode"], max_speed=float(vel["max_speed"])),
        feasibility_resolution=float(raw["feasibility"]["resolution"]),
        descriptor_resolution=float(raw["descriptor"]["resolution"]),
        gap_cap=float(raw["descriptor"]["gap_cap"]),
        scenarios=scenarios,
        mixtures=mixtures,
        splits=splits,
    )


def load_data_config(path: str | Path | None = None) -> DataConfig:
    return parse_data_config(load_yaml(path or CONFIG_DIR / "data.yaml"))


# --------------------------------------------------------------------------- #
# Split geometry and hashing
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class SplitGeometry:
    world_size: tuple[float, float]
    horizon: int
    count_scale: float      # multiplies random-obstacle counts (area ratio)


def split_geometry(spec: SplitSpec, sim: SimParams) -> SplitGeometry:
    """fixed_area keeps the base world; fixed_density scales the side by sqrt(N/N_ref)."""
    if spec.size_setting == "fixed_area":
        return SplitGeometry(tuple(sim.world.size), sim.simulation.horizon, 1.0)
    s = math.sqrt(spec.n_agents / sim.world.n_agents)
    size = tuple(float(v * s) for v in sim.world.size)
    return SplitGeometry(size, int(round(sim.simulation.horizon * s)), s * s)


def _generation_inputs(spec: SplitSpec, sim: SimParams, cfg: DataConfig) -> dict[str, Any]:
    """Everything that determines a split's contents, and nothing else."""
    used = cfg.mixtures[spec.mixture]
    return {
        "generator_version": GENERATOR_VERSION,
        "split": dataclasses.asdict(spec),
        "seed": cfg.seed,
        "layout": cfg.raw["layout"],
        "initial_velocity": cfg.raw["initial_velocity"],
        "feasibility": cfg.raw["feasibility"],
        "scenarios": {name: cfg.scenarios[name] for name in used},
        "mixture": used,
        "sim": {
            "world": dataclasses.asdict(sim.world),
            "agent_radius": sim.agent.radius,
            "sep_radius": sim.perception.sep_radius,
            "goal_capacity_factor": sim.goal.capacity_factor,
            "safety_margin": sim.loss.safety_margin,
            "horizon": sim.simulation.horizon,
        },
    }


def split_hash(spec: SplitSpec, sim: SimParams, cfg: DataConfig) -> str:
    return config_hash(_generation_inputs(spec, sim, cfg))


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def assign_scenarios(spec: SplitSpec, cfg: DataConfig) -> list[str]:
    """Balanced scenario assignment (counts differ by at most one), random order.

    Assigned once per map, before any rejection sampling, so rejections can't
    shift the mixture towards easier scenarios.
    """
    mixture = cfg.mixtures[spec.mixture]
    base = [mixture[i % len(mixture)] for i in range(spec.n_maps)]
    perm = stream_rng(cfg.seed, spec.name, 0, Stream.SCENARIO).permutation(spec.n_maps)
    return [base[i] for i in perm]


def generate_feasible_map(
    spec: SplitSpec, map_index: int, scenario: str, sim: SimParams, cfg: DataConfig
) -> tuple[MapSpec, int]:
    """Rejection-sample a feasible map. Returns (map, number of rejected attempts)."""
    geom = split_geometry(spec, sim)
    rejected = 0
    for attempt in range(cfg.layout.max_map_tries):
        rng = lambda s: stream_rng(cfg.seed, spec.name, map_index, s, attempt)
        m = sample_map(
            scenario,
            cfg.scenarios[scenario],
            spec.n_agents,
            geom.world_size,
            sim,
            cfg.layout,
            rng(Stream.START_BOX),
            rng(Stream.GOAL),
            rng(Stream.OBSTACLES),
            count_scale=geom.count_scale,
        )
        if m is not None and check_map(m, sim, cfg.feasibility_resolution).ok:
            return m, rejected
        rejected += 1
    raise RuntimeError(
        f"split {spec.name!r}, map {map_index} ({scenario}): no feasible map in "
        f"{cfg.layout.max_map_tries} attempts"
    )


def build_split(spec: SplitSpec, sim: SimParams, cfg: DataConfig) -> tuple[EpisodeBatch, dict[str, Any]]:
    t0 = time.perf_counter()
    scenarios = assign_scenarios(spec, cfg)
    maps, states, map_idx, ep_idx, roles, rejections = [], [], [], [], [], []
    for i, scenario in enumerate(scenarios):
        m, rejected = generate_feasible_map(spec, i, scenario, sim, cfg)
        rejections.append(rejected)
        for e in range(spec.episodes_per_map):
            states.append(
                sample_initial_state(
                    m, cfg.seed, spec.name, i, e, cfg.layout.init_spacing, sim.agent.radius, cfg.velocity
                )
            )
            maps.append(m)
            map_idx.append(i)
            ep_idx.append(e)
            roles.append(spec.role_of(e))
    batch = stack_episodes(maps, states, map_idx, ep_idx, roles)
    geom = split_geometry(spec, sim)
    meta = {
        "name": spec.name,
        "hash": split_hash(spec, sim, cfg),
        "generator_version": GENERATOR_VERSION,
        "spec": dataclasses.asdict(spec),
        "n_maps": spec.n_maps,
        "n_episodes": batch.n_episodes,
        "n_agents": spec.n_agents,
        "k_max": batch.k_max,
        "world_size": list(geom.world_size),
        "horizon": geom.horizon,
        "count_scale": geom.count_scale,
        "role_names": list(spec.role_names),
        "scenario_names": list(SCENARIOS),
        "map_scenarios": scenarios,
        "rejections": rejections,
        "build_seconds": round(time.perf_counter() - t0, 3),
    }
    # normalize to what JSON stores (tuples -> lists) so a loaded meta compares equal
    return batch, json.loads(json.dumps(meta))


# --------------------------------------------------------------------------- #
# Cache I/O
# --------------------------------------------------------------------------- #
def split_paths(name: str, directory: str | Path = DATA_ROOT) -> tuple[Path, Path]:
    d = Path(directory)
    return d / f"{name}.npz", d / f"{name}.json"


def save_split(batch: EpisodeBatch, meta: dict[str, Any], directory: str | Path = DATA_ROOT) -> Path:
    npz, js = split_paths(meta["name"], directory)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **batch._asdict())
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return npz


def read_split(name: str, directory: str | Path = DATA_ROOT) -> tuple[EpisodeBatch, dict[str, Any]]:
    """Load a cached split without checking it against the current configs."""
    npz, js = split_paths(name, directory)
    if not npz.exists() or not js.exists():
        raise FileNotFoundError(f"split {name!r} not found in {directory}; run experiments.build_datasets")
    with np.load(npz) as f:
        batch = EpisodeBatch(**{k: f[k] for k in EpisodeBatch._fields})
    with open(js, encoding="utf-8") as fh:
        meta = json.load(fh)
    return batch, meta


def load_split(
    name: str,
    sim: SimParams,
    cfg: DataConfig | None = None,
    directory: str | Path = DATA_ROOT,
) -> tuple[EpisodeBatch, dict[str, Any]]:
    """Load a cached split, refusing it if it no longer matches the configs."""
    cfg = cfg or load_data_config()
    batch, meta = read_split(name, directory)
    expected = split_hash(cfg.split(name), sim, cfg)
    if meta.get("hash") != expected:
        raise StaleDatasetError(
            f"split {name!r} was built from different configs (cached {meta.get('hash')}, "
            f"expected {expected}); rebuild with `python -m experiments.build_datasets --force`"
        )
    return batch, meta


def is_current(name: str, sim: SimParams, cfg: DataConfig, directory: str | Path = DATA_ROOT) -> bool:
    try:
        load_split(name, sim, cfg, directory)
    except (FileNotFoundError, StaleDatasetError):
        return False
    return True
