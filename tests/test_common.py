import json

import yaml

from experiments.common import git_info, make_run_dir
from swarm.config import config_hash


def test_make_run_dir_records_config_and_metadata(tmp_path):
    cfg = {"optimizer": "cma", "budget": 300, "nested": {"seeds": [0, 1]}}
    run_dir = make_run_dir("exp_test", cfg, results_root=tmp_path, tag="seed0")

    assert run_dir.parent == tmp_path / "exp_test"
    assert run_dir.name.endswith("_seed0")
    with open(run_dir / "config.yaml") as fh:
        assert yaml.safe_load(fh) == cfg
    with open(run_dir / "meta.json") as fh:
        meta = json.load(fh)
    assert meta["config_hash"] == config_hash(cfg)
    assert meta["experiment"] == "exp_test"
    assert {"commit", "dirty"} <= set(meta["git"])
    assert "jax" in meta["environment"]


def test_run_dirs_are_unique(tmp_path):
    a = make_run_dir("exp", {}, results_root=tmp_path)
    b = make_run_dir("exp", {}, results_root=tmp_path)
    assert a != b


def test_git_info_inside_and_outside_repo(tmp_path):
    info = git_info()
    assert info["commit"] is not None and len(info["commit"]) == 40
    assert git_info(tmp_path) == {"commit": None, "dirty": None}
