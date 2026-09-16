import numpy as np
import pytest

from splinebench.fitters import fit_linear


def _problem(seed=0, n=60, k=6):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, k))
    A[:, 0] = 1.0
    c_true = rng.normal(size=(k, 2))
    y = A @ c_true
    return A, y, c_true


def test_least_squares_recovers_exact():
    A, y, c_true = _problem()
    c = fit_linear(A, y, fitter="least_squares")
    assert np.allclose(c, c_true, atol=1e-6)


def test_ridge_shrinks():
    rng = np.random.default_rng(1)
    A, y, _ = _problem()
    noisy = y + 0.5 * rng.normal(size=y.shape)
    c_ls = fit_linear(A, noisy, fitter="least_squares")
    c_r = fit_linear(A, noisy, fitter="ridge", reg={"kind": "ridge", "lam": 1.0})
    assert np.linalg.norm(c_r) < np.linalg.norm(c_ls)


def test_huber_beats_ls_under_outliers():
    rng = np.random.default_rng(0)
    A, y, c_true = _problem(seed=3)
    y = y + rng.normal(0, 0.01, size=y.shape)
    y[:8] += 5.0
    c_ls = fit_linear(A, y, fitter="least_squares")
    c_h = fit_linear(A, y, fitter="huber")
    pred = np.linalg.norm(A @ c_h - A @ c_true)
    assert pred < np.linalg.norm(A @ c_ls - A @ c_true)


def test_ransac_beats_ls_under_outliers():
    rng = np.random.default_rng(0)
    A, y, c_true = _problem(seed=5)
    y = y + rng.normal(0, 0.01, size=y.shape)
    y[:15] += 4.0
    c_ls = fit_linear(A, y, fitter="least_squares")
    c_r = fit_linear(A, y, fitter="ransac", seed=0)
    assert np.linalg.norm(A @ c_r - A @ c_true) < 0.5 * np.linalg.norm(A @ c_ls - A @ c_true)


def test_difference_penalty_smooths():
    rng = np.random.default_rng(0)
    A, y, _ = _problem(seed=7)
    y = y + rng.normal(0, 0.2, size=y.shape)
    c_plain = fit_linear(A, y, fitter="least_squares")
    c_smooth = fit_linear(A, y, fitter="least_squares", reg={"kind": "diff", "order": 2, "lam": 1e-2})
    assert np.linalg.norm(np.diff(c_smooth, n=2, axis=0)) < np.linalg.norm(np.diff(c_plain, n=2, axis=0))


def test_unknown_fitter_raises():
    A, y, _ = _problem()
    with pytest.raises(KeyError):
        fit_linear(A, y, fitter="does_not_exist")
