"""Independent, reproducible random streams for data generation.

Each random quantity draws from its own stream, keyed by
(root seed, split, map index, stream, sub-index). Two consequences:

* Changing how one quantity is drawn never shifts another. In particular the
  goal stream is separate from the initial-state and informed-order streams, so
  moving the goal cannot change x0, v0 or the informed mask (the data-side
  leakage check in doc §8.2).
* Splits are independent of each other (train / validation / test seeds never
  overlap), and results don't depend on generation order.
"""

from __future__ import annotations

import enum
import zlib

import numpy as np


class Stream(enum.IntEnum):
    SCENARIO = 0      # per-split scenario assignment
    START_BOX = 1
    GOAL = 2
    OBSTACLES = 3
    INIT_POS = 4
    INIT_VEL = 5
    INFORMED = 6


def _split_key(split: str) -> int:
    # Python's hash() is salted per process; crc32 is stable.
    return zlib.crc32(split.encode("utf-8"))


def stream_rng(seed: int, split: str, index: int, stream: Stream, sub: int = 0) -> np.random.Generator:
    """Generator for one (split, map index, stream, sub-index) combination.

    `sub` is the attempt number for map-geometry streams and the episode number
    within a map for initial-state streams.
    """
    ss = np.random.SeedSequence(entropy=seed, spawn_key=(_split_key(split), index, int(stream), sub))
    return np.random.default_rng(ss)
