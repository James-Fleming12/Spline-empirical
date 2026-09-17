"""Tests for the Iteration 2 stress-test additions.

Covers structured noise, extrapolation splits, conditioning telemetry,
auto-tuned knot/penalty selection, adversarial knots and the extra metrics.
"""

import numpy as np
import pytest

from splinebench import knots, metrics, motions, representations
from splinebench.experiment import (
    Condition,
    NoiseModel,
    ObservationOracle,
    remap_to_windows,
    run_condition,
    split_windows,
)


# ---------------------------------------------------------------------------
# Structured observation models
# ---------------------------------------------------------------------------

def _residuals(noise, seed=0, motion_name="bounce", n=200):
    motion = motions.build(motion_name)
    oracle = ObservationOracle(motion, noise, seed)
    t = np.linspace(0.01, 0.99, n)
    y = oracle.query(t)
    return motion, t, y - motion.eval(t)


def test_structured_noise_is_deterministic():
    noise = NoiseModel(std=0.02, ar1_rho=0.95, bias_drift=0.03)
    motion, t, r1 = _residuals(noise)
    oracle = ObservationOracle(motion, noise, 0)
    r2 = oracle.query(t) - motion.eval(t)
    assert np.allclose(r1, r2, equal_nan=True)


def test_ar1_noise_is_temporally_correlated():
    _, _, iid = _residuals(NoiseModel(std=0.05))
    _, _, ar1 = _residuals(NoiseModel(std=0.05, ar1_rho=0.98))

    def acf1(x):
        x = x[:, 0] - x[:, 0].mean()
        return float(np.sum(x[:-1] * x[1:]) / (np.sum(x**2) + 1e-12))

    assert acf1(ar1) > 0.5 > acf1(iid)


def test_axis_correlated_noise():
    motion, _, resid = _residuals(NoiseModel(std=0.05, axis_corr=0.9), n=400)
    corr = float(np.corrcoef(resid[:, 0], resid[:, 1])[0, 1])
    assert corr > 0.5


def test_missing_bursts_and_quantization():
    motion = motions.build("bounce")
    t = np.linspace(0.01, 0.99, 200)
    y = ObservationOracle(motion, NoiseModel(std=0.02, quantize=0.05), 0).query(t)
    finite = y[np.isfinite(y)]
    assert np.allclose(finite / 0.05, np.round(finite / 0.05), atol=1e-8)

    noise = NoiseModel(std=0.01, missing_bursts=2, burst_width=0.1)
    motion = motions.build("bounce")
    oracle = ObservationOracle(motion, noise, 0)
    y = oracle.query(np.linspace(0.01, 0.99, 300))
    assert np.isnan(y).any()
    assert 0.02 < np.isnan(y).mean() < 0.5


def test_heteroscedastic_noise_scales_with_speed():
    motion, t, resid = _residuals(NoiseModel(std=0.01, hetero=3.0), n=600)
    speed = np.linalg.norm(motion.eval_state(t, order=1)[..., 1], axis=1)
    mag = np.linalg.norm(resid, axis=1)
    hi = mag[speed > np.quantile(speed, 0.8)].mean()
    lo = mag[speed < np.quantile(speed, 0.2)].mean()
    assert hi > lo


# ---------------------------------------------------------------------------
# Extrapolation splits
# ---------------------------------------------------------------------------

def test_split_windows_and_remap():
    train, test = split_windows("interp")
    assert train == [(0.0, 0.45), (0.55, 1.0)]
    assert test == [(0.45, 0.55)]
    mapped = remap_to_windows(np.linspace(0, 1, 50), train)
    assert mapped.min() >= 0.0 and mapped.max() <= 1.0
    assert np.all(np.diff(mapped) > 0)


@pytest.mark.parametrize("split", ["early", "late", "middle", "interp"])
def test_extrapolation_records_holdout_metrics(split):
    cond = Condition(
        motion="staccato",
        representation="gp_matern52",
        knots="uniform",
        budget=20,
        seed=0,
        split=split,
        noise_std=0.01,
    )
    rec = run_condition(cond)
    assert np.isfinite(rec["holdout_pos_rmse"])
    assert np.isfinite(rec["trainwin_pos_rmse"])
    assert "holdout_dtw" in rec and "holdout_phase_err" in rec


# ---------------------------------------------------------------------------
# Conditioning telemetry
# ---------------------------------------------------------------------------

def test_linear_basis_diagnostics():
    motion = motions.build("staccato")
    t = np.linspace(0, 1, 60)
    y = motion.eval(t)
    rep = representations.build("bspline3", positions=np.linspace(0, 1, 8), dim=motion.dim)
    rep.fit(t, y, reg={"kind": "diff", "order": 2, "lam": 1e-3})
    diag = rep.diagnostics()
    for key in ("design_cond", "design_rank", "eff_dof", "penalty_rank", "penalty_nullity"):
        assert key in diag
    assert 0 < diag["eff_dof"] <= rep.n_params
    assert diag["penalty_rank"] > 0


def test_gp_diagnostics():
    motion = motions.build("orbit")
    t = np.linspace(0, 1, 40)
    rep = representations.build("gp_matern52", dim=motion.dim, optimize=False)
    rep.fit(t, motion.eval(t))
    diag = rep.diagnostics()
    assert np.isfinite(diag["gp_kernel_cond"]) and diag["gp_kernel_cond"] > 0
    assert diag["gp_lengthscale"] > 0
    assert diag["gp_n_support"] == 40


def test_runner_records_telemetry_and_knot_stats():
    cond = Condition(motion="staccato", representation="bspline3", knots="split_merge",
                     budget=40, seed=0, n_sites=8, noise_std=0.01)
    rec = run_condition(cond)
    for key in ("design_cond", "eff_dof", "knot_min_gap", "knot_max_gap", "resid_acf1"):
        assert key in rec
    assert rec["n_selected_sites"] <= rec["n_sites"]
    assert np.isfinite(rec["knot_median_gap"])


# ---------------------------------------------------------------------------
# Auto-tuned knot count / penalty
# ---------------------------------------------------------------------------

def test_pspline_gcv_selects_penalty():
    motion = motions.build("staccato")
    t = np.linspace(0, 1, 40)
    y = motion.eval(t) + 0.02 * np.random.default_rng(0).normal(size=(40, motion.dim))
    rep = representations.build("pspline_gcv", positions=np.linspace(0, 1, 8), dim=motion.dim)
    rep.fit(t, y)
    assert rep.gcv_lam > 0
    assert np.isfinite(rep.eval(t)).all()
    assert "gcv_lam" in rep.diagnostics()


def test_cv_knots_selects_bounded_count():
    motion = motions.build("wobble")
    t = np.linspace(0, 1, 60)
    y = motion.eval(t)
    ctx = knots.KnotContext(u=t, y=y, n=10, rng=np.random.default_rng(0), motion=motion)
    pos = knots.build("cv", max_sites=8, folds=4).place(ctx)
    assert pos.ndim == 1 and len(pos) >= 3
    assert pos.min() >= 0.0 and pos.max() <= 1.0
    assert np.all(np.diff(pos) > 0)


def test_clustered_knots_condition_the_design():
    motion = motions.build("wobble")
    t = np.linspace(0, 1, 80)
    y = motion.eval(t)
    ctx = knots.KnotContext(u=t, y=y, n=8, rng=np.random.default_rng(0), motion=motion)
    pos = knots.build("clustered", mode="near_duplicate").place(ctx)
    assert len(pos) == 8
    rep = representations.build("bspline3", positions=pos, dim=motion.dim)
    rep.fit(t, y)
    cond = rep.diagnostics()["design_cond"]
    assert not np.isfinite(cond) or cond > 1e4


# ---------------------------------------------------------------------------
# Metrics beyond RMSE
# ---------------------------------------------------------------------------

def test_phase_error_detects_shift():
    x = np.sin(2 * np.pi * np.linspace(0, 1, 200))
    shifted = np.roll(x, 20)
    err = metrics.phase_error(shifted, x)
    assert abs(err - 20 / 199) < 0.02
    assert metrics.phase_error(x, x) == 0.0


def test_dtw_zero_for_identical_paths():
    x = np.stack([np.sin(np.linspace(0, 3, 50)), np.cos(np.linspace(0, 3, 50))], axis=1)
    assert metrics.dtw_distance(x, x) == 0.0
    assert metrics.dtw_distance(x, x + 1.0) > 0.0


def test_iteration2_suites_build():
    from splinebench import suites

    for name in ("i2_core", "i2_noise", "i2_timeparam", "i2_extrap", "i2_adversarial",
                 "i2_knotcount", "iteration2"):
        conds = suites.build(name, quick=True)
        assert len(conds) > 0
