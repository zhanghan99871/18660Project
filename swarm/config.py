"""Configuration loading and the precision policy.

`SimParams` holds every *static* constant of the forward model (project doc §10).
It is a tree of frozen dataclasses, so it is hashable and can be passed to
`jax.jit` as a static argument. Changing any value triggers a recompile.

Dynamic, per-episode quantities (obstacles, goal, initial state, informed order)
are not part of the config: they are generated offline by `swarm.episodes` and
cached in `data/`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import typing
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"

BEHAVIORS = ("separation", "alignment", "cohesion", "goal", "obstacle")


# --------------------------------------------------------------------------- #
# Config sections
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class WorldParams:
    size: tuple[float, float]
    n_agents: int


@dataclasses.dataclass(frozen=True)
class AgentParams:
    radius: float
    v_max: float
    a_max: float
    damping: float


@dataclasses.dataclass(frozen=True)
class PerceptionParams:
    sensor_radius: float
    sep_radius: float
    sep_softening: float


@dataclasses.dataclass(frozen=True)
class ForceParams:
    obstacle_influence: float
    tangential_gain: float
    tangential_sharpness: float
    tangential_bias: float


@dataclasses.dataclass(frozen=True)
class GoalParams:
    capacity_factor: float
    length_scale_factor: float


@dataclasses.dataclass(frozen=True)
class FeatureParams:
    n_ref: float
    goal_distance_scale: float


@dataclasses.dataclass(frozen=True)
class ControllerParams:
    w_max: tuple[float, ...]
    search_box: tuple[float, float]


@dataclasses.dataclass(frozen=True)
class SimulationParams:
    dt: float
    horizon: int


@dataclasses.dataclass(frozen=True)
class SmoothingParams:
    eta_schedule: tuple[float, ...]
    fixed_soft_eta: float


@dataclasses.dataclass(frozen=True)
class LossWeights:
    goal: float
    time: float
    collision: float
    obstacle: float
    boundary: float
    energy: float


@dataclasses.dataclass(frozen=True)
class LossParams:
    safety_margin: float
    weights: LossWeights


@dataclasses.dataclass(frozen=True)
class EvaluationParams:
    contact_tol: float


@dataclasses.dataclass(frozen=True)
class NumericsParams:
    eps: float
    precision: str


@dataclasses.dataclass(frozen=True)
class SimParams:
    world: WorldParams
    agent: AgentParams
    perception: PerceptionParams
    forces: ForceParams
    goal: GoalParams
    features: FeatureParams
    controller: ControllerParams
    simulation: SimulationParams
    smoothing: SmoothingParams
    loss: LossParams
    evaluation: EvaluationParams
    numerics: NumericsParams

    # ---- derived quantities ------------------------------------------------ #
    def goal_radius(self, n_agents: int) -> float:
        """Goal capacity rule (§3.3): R_g = c * r_sep * sqrt(N)."""
        return self.goal.capacity_factor * self.perception.sep_radius * math.sqrt(n_agents)

    def goal_length_scale(self, n_agents: int) -> float:
        """l_g = factor * R_g (§3.3)."""
        return self.goal.length_scale_factor * self.goal_radius(n_agents)

    @property
    def braking_distance(self) -> float:
        """Full-speed braking distance v_max^2 / (2 a_max) (§3.3)."""
        return self.agent.v_max**2 / (2.0 * self.agent.a_max)

    def validate(self) -> None:
        """Raise ValueError if the constants violate a modelling requirement."""
        p, a, f = self.perception, self.agent, self.forces
        problems = []
        if not 0 < p.sep_radius < p.sensor_radius:
            problems.append("need 0 < sep_radius < sensor_radius")
        if not self.braking_distance < f.obstacle_influence <= p.sensor_radius:
            problems.append(
                f"need braking distance ({self.braking_distance:.3f}) < obstacle_influence "
                f"({f.obstacle_influence}) <= sensor_radius ({p.sensor_radius})"
            )
        if len(self.controller.w_max) != len(BEHAVIORS):
            problems.append(f"w_max must have {len(BEHAVIORS)} entries")
        if any(w <= 0 for w in self.controller.w_max):
            problems.append("w_max entries must be positive")
        lo, hi = self.controller.search_box
        if not lo < hi:
            problems.append("search_box must satisfy low < high")
        if list(self.smoothing.eta_schedule) != sorted(self.smoothing.eta_schedule, reverse=True):
            problems.append("eta_schedule must be decreasing")
        if self.numerics.precision not in ("float32", "float64"):
            problems.append("precision must be float32 or float64")
        if min(a.radius, a.v_max, a.a_max, self.simulation.dt, self.numerics.eps) <= 0:
            problems.append("radius, v_max, a_max, dt and eps must be positive")
        if problems:
            raise ValueError("invalid SimParams: " + "; ".join(problems))

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(dataclasses.asdict(self))


# --------------------------------------------------------------------------- #
# Strict construction from nested dicts
# --------------------------------------------------------------------------- #
def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _coerce(tp: Any, value: Any, where: str) -> Any:
    if dataclasses.is_dataclass(tp):
        return from_dict(tp, value, where)
    origin = typing.get_origin(tp)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{where}: expected a list, got {value!r}")
        args = typing.get_args(tp)
        elem = args[0]
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(elem, v, where) for v in value)
        if len(args) != len(value):
            raise ValueError(f"{where}: expected {len(args)} entries, got {len(value)}")
        return tuple(_coerce(t, v, where) for t, v in zip(args, value))
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{where}: expected an integer, got {value!r}")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise TypeError(f"{where}: expected a string, got {value!r}")
        return value
    return value


def from_dict(cls: type, data: dict[str, Any], where: str = "config") -> Any:
    """Build a (nested) dataclass, rejecting missing and unknown keys."""
    if not isinstance(data, dict):
        raise TypeError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    unknown = sorted(set(data) - set(names))
    missing = [n for n in names if n not in data]
    if unknown:
        raise KeyError(f"{where}: unknown keys {unknown}")
    if missing:
        raise KeyError(f"{where}: missing keys {missing}")
    return cls(**{n: _coerce(hints[n], data[n], f"{where}.{n}") for n in names})


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if data is not None else {}


def load_sim_params(path: str | Path | None = None, overrides: dict | None = None) -> SimParams:
    """Load `configs/base.yaml` (or `path`), apply nested `overrides`, validate."""
    data = load_yaml(path or CONFIG_DIR / "base.yaml")
    if overrides:
        data = deep_update(data, overrides)
    params = from_dict(SimParams, data, "base")
    params.validate()
    return params


def deep_update(base: dict, updates: dict) -> dict:
    """Return a copy of `base` with `updates` merged in recursively."""
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def config_hash(obj: Any, length: int = 12) -> str:
    """Stable short hash of a JSON-serializable config (key order independent)."""
    blob = json.dumps(_to_plain(obj), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


# --------------------------------------------------------------------------- #
# Precision policy
# --------------------------------------------------------------------------- #
def configure_jax(precision: str = "float64") -> None:
    """Set JAX's default float width.

    Must run before any JAX array is created (call it at the top of every entry
    point). float64 is the default for the whole project; the L-BFGS-B path
    requires it (§6.4). If a derivative-free run uses float32 for speed, all
    validation and test scoring must still happen in float64.
    """
    if precision not in ("float32", "float64"):
        raise ValueError(f"precision must be float32 or float64, got {precision!r}")
    import jax

    jax.config.update("jax_enable_x64", precision == "float64")
