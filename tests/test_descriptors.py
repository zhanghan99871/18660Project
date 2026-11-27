import math

import numpy as np
import pytest

from swarm.datasets import SplitSpec, generate_feasible_map
from swarm.descriptors import (
    DESCRIPTOR_NAMES,
    DescriptorScaler,
    map_descriptor,
    obstacle_area_fraction,
    passage_width,
)
from swarm.maps import wall_circle_ys

RES, CAP = 0.05, 4.0


def _wall_map(make_map, gap, center=5.0):
    ys = wall_circle_ys(center, gap, 0.3, 0.3, 0.0, 10.0)
    return make_map(centers=np.stack([np.full_like(ys, 5.0), ys], -1), radii=np.full(len(ys), 0.3))


def test_open_map_descriptor(make_map, sim):
    m = make_map(start_lo=(1.0, 4.0), start_hi=(2.0, 6.0), goal=(8.5, 5.0))
    psi = map_descriptor(m, sim, RES, CAP)
    assert psi.shape == (len(DESCRIPTOR_NAMES),)
    expected = [0.0, 0.0, CAP / sim.perception.sensor_radius, 7.0 / 10.0, math.log(30), 30 / 2.0]
    np.testing.assert_allclose(psi, expected)


def test_area_fraction_of_single_disk(make_map):
    m = make_map(centers=[[5.0, 5.0]], radii=[1.5])
    assert obstacle_area_fraction(m, RES) == pytest.approx(math.pi * 1.5**2 / 100, rel=0.02)


def test_area_fraction_counts_overlap_once_and_clips_to_domain(make_map):
    overlap = make_map(centers=[[5.0, 5.0], [5.0, 5.0]], radii=[1.0, 1.0])
    assert obstacle_area_fraction(overlap, RES) == pytest.approx(math.pi / 100, rel=0.02)
    corner = make_map(centers=[[0.0, 0.0]], radii=[2.0])     # only a quarter is inside
    assert obstacle_area_fraction(corner, RES) == pytest.approx(math.pi / 100, rel=0.03)


@pytest.mark.parametrize("gap", [0.35, 0.6, 1.0, 1.5])
def test_passage_width_measures_the_bottleneck_gap(make_map, gap):
    w = passage_width(_wall_map(make_map, gap), RES, CAP)
    assert w == pytest.approx(gap, abs=2.5 * RES)


def test_passage_width_extremes(make_map):
    assert passage_width(make_map(), RES, CAP) == CAP
    ys = np.arange(0.0, 10.0 + 1e-9, 0.3)
    solid = make_map(centers=np.stack([np.full_like(ys, 5.0), ys], -1), radii=np.full(len(ys), 0.3))
    assert passage_width(solid, RES, CAP) == 0.0


def test_passage_width_prefers_the_widest_route(make_map):
    # one wide (1.4) and one narrow (0.5) opening: the best route uses the wide one
    ys = np.arange(0.0, 10.0 + 1e-9, 0.3)
    keep = ~(((ys > 2.0) & (ys < 3.0 + 0.4)) | ((ys > 6.5) & (ys < 7.2)))
    m = make_map(centers=np.stack([np.full(keep.sum(), 5.0), ys[keep]], -1), radii=np.full(keep.sum(), 0.3))
    narrow_only = passage_width(_wall_map(make_map, 0.5, center=7.0), RES, CAP)
    assert passage_width(m, RES, CAP) > narrow_only + 0.3


def test_descriptors_of_generated_maps_are_finite_and_ordered(sim, data_cfg):
    spec = SplitSpec("desc", "train_id", 1, 1, 30)
    psis = {}
    for scenario in ("open", "bottleneck", "narrow_bottleneck", "dense_random"):
        m, _ = generate_feasible_map(spec, 0, scenario, sim, data_cfg)
        psis[scenario] = map_descriptor(m, sim, data_cfg.descriptor_resolution, data_cfg.gap_cap)
        assert np.all(np.isfinite(psis[scenario]))
    gap = DESCRIPTOR_NAMES.index("passage_width")
    assert psis["narrow_bottleneck"][gap] < psis["bottleneck"][gap] < psis["open"][gap]
    k = DESCRIPTOR_NAMES.index("n_obstacles")
    assert psis["open"][k] == 0 and psis["dense_random"][k] >= 10


def test_scaler_standardizes_and_handles_constant_features():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(3.0, 2.0, 50), np.full(50, math.log(30)), rng.uniform(size=50)])
    sc = DescriptorScaler.fit(X)
    Z = sc.transform(X)
    np.testing.assert_allclose(Z[:, [0, 2]].mean(0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Z[:, [0, 2]].std(0), 1.0)
    np.testing.assert_allclose(Z[:, 1], 0.0, atol=1e-12)
    assert sc.scale[1] == 1.0
    # an unseen N keeps its raw deviation rather than blowing up
    assert sc.transform(np.array([[3.0, math.log(100), 0.5]]))[0, 1] == pytest.approx(math.log(100 / 30))
    np.testing.assert_allclose(sc.inverse_transform(Z), X)
    restored = DescriptorScaler.from_dict(sc.to_dict())
    np.testing.assert_allclose(restored.transform(X), Z)
