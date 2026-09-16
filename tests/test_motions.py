import numpy as np
import pytest

from splinebench import motions


def _fd_median_error(motion, order=3, n=4001):
    t = np.linspace(0.0, 1.0, n)
    state = motion.eval_state(t, order=order)
    errors = []
    for k in range(1, order + 1):
        fd = state[..., 0]
        for _ in range(k):
            fd = np.gradient(fd, t, axis=0)
        err = np.abs(fd - state[..., k])
        errors.append(np.median(err) / (np.max(np.abs(fd)) + 1e-9))
    return errors


@pytest.mark.parametrize("name", motions.list_motions())
def test_motion_derivatives(name):
    motion = motions.build(name)
    errors = _fd_median_error(motion, order=3)
    assert max(errors) < 5e-3, f"{name} derivative mismatch: {errors}"
    state = motion.eval_state(np.linspace(0, 1, 51), order=4)
    assert state.shape == (51, motion.dim, 5)
    assert np.isfinite(state).all()


def test_motion_normalized():
    for name in motions.list_motions():
        motion = motions.build(name)
        assert abs(motion.max_abs(order=0) - 1.0) < 1e-6


def test_sine_analytic():
    prim = motions.Sine(freq=2.0, amp=0.5, phase=0.1)
    t = np.linspace(0, 1, 101)
    state = prim.derivs(t, order=4)
    assert np.allclose(state[:, 0], 0.5 * np.sin(2 * np.pi * 2 * t + 0.1))
    assert np.allclose(state[:, 1], 0.5 * 2 * np.pi * 2 * np.cos(2 * np.pi * 2 * t + 0.1))


def test_minjerk_step_boundaries():
    prim = motions.MinJerkStep(0.2, 0.4, 1.0)
    assert np.allclose(prim.derivs(np.array([0.1, 0.5]))[0], 0.0)
    inside = prim.derivs(np.array([0.3]))[0]
    assert abs(inside[0] - 0.5) < 1e-9


def test_snappy_groups_present():
    for name in motions.SNAPPY_MOTIONS + motions.HIGH_FREQUENCY_MOTIONS + motions.SMOOTH_MOTIONS:
        assert name in motions._BUILDERS
