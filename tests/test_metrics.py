import numpy as np

from splinebench import metrics, motions, timeparam


class ExactRep:
    name = "exact"
    deriv_method = "analytic"
    uses_ground_truth = True
    n_params = 0
    n_data_params = 0
    n_params_total = 0

    def __init__(self, motion):
        self.motion = motion

    def eval_derivs(self, u, order=4):
        return self.motion.eval_state(np.asarray(u, dtype=float).ravel(), order=order)

    def eval(self, u):
        return self.eval_derivs(u, 0)[..., 0]


def test_exact_rep_scores_zero_error():
    motion = motions.build("staccato")
    rep = ExactRep(motion)
    tp = timeparam.build("linear")
    tp.fit(np.linspace(0, 1, 10))
    out = metrics.evaluate(motion, rep, tp)
    assert out["pos_rmse"] < 1e-10
    assert out["vel_rmse"] < 1e-9
    assert out["acc_rmse"] < 1e-8
    assert abs(out["isj_ratio"] - 1.0) < 1e-6
    assert out["peak_time_err"] < 1e-3
    assert out["overshoot"] < 1e-9


def test_metrics_finite_on_real_fit():
    from splinebench.representations import BSpline

    motion = motions.build("bounce")
    t = np.linspace(0.0, 1.0, 40)
    y = motion.eval(t)
    rep = BSpline(np.linspace(0, 1, 6), dim=motion.dim, degree=3)
    rep.fit(t, y)
    tp = timeparam.build("linear")
    tp.fit(t, y)
    out = metrics.evaluate(motion, rep, tp, u_train=t, y_train=y, fit_time_s=0.01)
    for key in ("pos_rmse", "vel_rmse", "jerk_rmse", "snap_rmse", "vel_nrmse", "acc_nrmse",
                "jerk_nrmse", "snap_nrmse", "overshoot", "settle_time_err",
                "isj_ratio", "fit_time_s", "train_rmse", "n_params_total"):
        assert key in out
        assert np.isfinite(out[key]), key


def test_pareto_front_drops_dominated():
    import pandas as pd

    df = pd.DataFrame({"n_params_total": [10, 20, 15, 30], "pos_rmse": [1.0, 0.9, 0.8, 0.7]})
    front = metrics.pareto_front(df, x="n_params_total", y="pos_rmse")
    assert 20 not in front["n_params_total"].values


def test_aulc_monotone_error_curve():
    import pandas as pd

    df = pd.DataFrame(
        {
            "motion": ["m"] * 3,
            "representation": ["r"] * 3,
            "budget": [5, 10, 20],
            "pos_rmse": [0.5, 0.2, 0.1],
        }
    )
    out = metrics.aulc(df, group_cols=["motion", "representation"])
    assert out["aulc"].iloc[0] > 0
    assert out["final"].iloc[0] == 0.1
