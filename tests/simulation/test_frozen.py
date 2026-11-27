import numpy as np
import pytest

from swarm.config import load_frozen
from swarm.controllers import weights_from_logits


def test_frozen_quantities_are_consistent(sim):
    frozen = load_frozen()
    b, w = frozen["b_hand"], frozen["hand_weights"]
    assert b.shape == (5,) and w.shape == (5,)
    lo, hi = sim.controller.search_box
    assert np.all((b >= lo) & (b <= hi))                       # inside every optimizer's box
    np.testing.assert_allclose(weights_from_logits(b, sim.controller.w_max), w, rtol=1e-9)
    for key in ("mu_z_fi", "mu_z_pi"):
        mu = frozen[key]
        assert mu.shape == (4,) and np.all((mu >= 0) & (mu <= 1))
    # uninformed agents carry gamma = 0, so the PI goal-distance mean is lower
    assert frozen["mu_z_pi"][2] < frozen["mu_z_fi"][2]
    assert frozen["selection"]["n_trials"] <= 20


def test_missing_frozen_file_has_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="hand_tuned"):
        load_frozen(tmp_path / "frozen.yaml")
