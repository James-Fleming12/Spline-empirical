import numpy as np
import pytest

from splinebench import motions, samplers
from splinebench.experiment import NoiseModel, ObservationOracle


def _oracle(seed=0):
    motion = motions.build("staccato")
    return motion, ObservationOracle(motion, NoiseModel(std=0.01), seed)


@pytest.mark.parametrize("name", samplers.list_samplers())
def test_sampler_budget_and_domain(name):
    motion, oracle = _oracle()
    budget = 12
    sampler = samplers.build(name)
    rng = np.random.default_rng(0)
    t = sampler.sample(motion, budget, rng, oracle)
    assert len(t) == budget
    assert np.all(np.diff(t) > 0)
    assert t[0] >= 0.0 and t[-1] <= 1.0


def test_oracle_samplers_are_flagged():
    assert samplers.build("jerk_importance", oracle=True).uses_ground_truth
    assert not samplers.build("random").uses_ground_truth
    assert samplers.build("active_gp").adaptive


def test_peak_sampler_hits_derivative_extrema():
    motion = motions.build("double_step")
    oracle = ObservationOracle(motion, NoiseModel(std=0.005), 1)
    t = samplers.build("derivative_peaks", oracle=True).sample(motion, 24, np.random.default_rng(0), oracle)
    near = np.min(np.abs(t[:, None] - np.array([0.30, 0.36, 0.42, 0.48])[None, :]), axis=1)
    assert np.mean(near < 0.06) > 0.3


def test_unknown_sampler_raises():
    with pytest.raises(KeyError):
        samplers.build("does_not_exist")
