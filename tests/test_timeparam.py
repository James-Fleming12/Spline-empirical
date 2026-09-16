import numpy as np

from splinebench import timeparam


def test_linear_identity():
    tp = timeparam.build("linear")
    u = np.linspace(0.0, 1.0, 50)
    state = tp.to_u(u)
    assert np.allclose(state[:, 0], u)
    assert np.allclose(state[:, 1], 1.0)
    assert np.allclose(state[:, 2:], 0.0)


def test_chord_monotone_endpoints():
    t = np.linspace(0.0, 1.0, 400)
    y = np.stack([t, np.sin(2.0 * np.pi * t), np.zeros_like(t)], axis=1)
    tp = timeparam.build("chord")
    tp.fit(t, y)
    state = tp.to_u(t)
    u = state[:, 0]
    assert np.all(np.diff(u) >= -1e-9)
    assert u[0] == 0.0 and abs(u[-1] - 1.0) < 1e-9
    fd = np.gradient(u, t)
    assert np.max(np.abs(fd[2:-2] - state[2:-2, 1])) < 5e-3


def test_feature_oracle_and_estimated_agree():
    from splinebench import motions

    motion = motions.build("staccato")
    t = np.linspace(0.0, 1.0, 120)
    y = motion.eval(t)
    est = timeparam.build("accel", oracle=False)
    est.fit(t, y)
    oracle = timeparam.build("accel", oracle=True)
    oracle.fit(t, y, motion=motion)
    u_est = est.to_u(t)[:, 0]
    u_oracle = oracle.to_u(t)[:, 0]
    assert np.max(np.abs(u_est - u_oracle)) < 0.35


def test_short_series_falls_back_to_linear():
    t = np.array([0.0, 0.5, 1.0])
    y = np.stack([t, t, t], axis=1)
    tp = timeparam.build("jerk")
    tp.fit(t, y)
    assert np.allclose(tp.to_u(t)[:, 0], t)
