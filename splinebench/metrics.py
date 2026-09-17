import numpy as np

from .utils import MAX_ORDER, chain_rule_derivs, trapz

ORDER_NAMES = {0: "pos", 1: "vel", 2: "acc", 3: "jerk", 4: "snap"}


def predict(motion, rep, time_param, t):
    t = np.asarray(t, dtype=float).ravel()
    u_state = time_param.to_u(t)
    rep_state = rep.eval_derivs(u_state[:, 0], order=MAX_ORDER)
    return chain_rule_derivs(rep_state, u_state[:, None, :])


def evaluate(motion, rep, time_param, grid=None, limits=None, fit_time_s=None,
             eval_time_s=None, u_train=None, y_train=None):
    if grid is None:
        grid = np.linspace(motion.window[0], motion.window[1], 2001)
    grid = np.asarray(grid, dtype=float).ravel()
    state_fit = predict(motion, rep, time_param, grid)
    state_gt = motion.eval_state(grid, order=MAX_ORDER)
    window = motion.window[1] - motion.window[0]
    out = {}

    for k, name in ORDER_NAMES.items():
        err = state_fit[..., k] - state_gt[..., k]
        out[f"{name}_rmse"] = float(np.sqrt(np.mean(err**2)))
        if k > 0:
            gt_rms = float(np.sqrt(np.mean(state_gt[..., k] ** 2)))
            out[f"{name}_nrmse"] = float(out[f"{name}_rmse"] / (gt_rms + 1e-12))
        if k == 0:
            out["pos_mae"] = float(np.mean(np.abs(err)))
            out["pos_max"] = float(np.max(np.linalg.norm(err, axis=1)))

    out["jerk_max_fit"] = float(np.max(np.abs(state_fit[..., 3])))
    out["snap_max_fit"] = float(np.max(np.abs(state_fit[..., 4])))

    gt_peak_axis = int(np.argmax(np.ptp(state_gt[..., 0], axis=0)))
    gt_peak_t = grid[int(np.argmax(np.abs(state_gt[:, gt_peak_axis, 0])))]
    fit_peak_t = grid[int(np.argmax(np.abs(state_fit[:, gt_peak_axis, 0])))]
    out["peak_time_err"] = float(abs(fit_peak_t - gt_peak_t) / window)

    fit_env = np.max(state_fit[..., 0], axis=0)
    gt_env = np.max(state_gt[..., 0], axis=0)
    fit_lo = np.min(state_fit[..., 0], axis=0)
    gt_lo = np.min(state_gt[..., 0], axis=0)
    out["overshoot"] = float(np.maximum(fit_env - gt_env, 0).max() + np.maximum(gt_lo - fit_lo, 0).max())

    err_norm = np.linalg.norm(state_fit[..., 0] - state_gt[..., 0], axis=1)
    peak = float(err_norm.max()) + 1e-12
    tol = 0.05 * peak
    idx = np.flatnonzero(err_norm > tol)
    out["settle_time_err"] = float(grid[idx[-1]] - grid[0]) / window if len(idx) else 0.0

    isj_fit = trapz(np.mean(state_fit[..., 3] ** 2, axis=1), grid)
    isj_gt = trapz(np.mean(state_gt[..., 3] ** 2, axis=1), grid)
    out["isj_fit"] = isj_fit
    out["isj_ratio"] = float(isj_fit / (isj_gt + 1e-12))

    if limits:
        for key, order in (("v", 1), ("a", 2), ("j", 3), ("s", 4)):
            if key in limits:
                excess = np.maximum(np.abs(state_fit[..., order]) - float(limits[key]), 0.0)
                out[f"cstr_{key}_mean"] = float(excess.mean())
                out[f"cstr_{key}_max"] = float(excess.max())

    if u_train is not None and y_train is not None:
        train_pred = rep.eval(np.asarray(u_train, dtype=float).ravel())
        out["train_rmse"] = float(np.sqrt(np.mean((train_pred - np.asarray(y_train)) ** 2)))

    out["n_params"] = int(rep.n_params)
    out["n_data_params"] = int(rep.n_data_params)
    out["n_params_total"] = int(rep.n_params_total)
    out["deriv_method"] = rep.deriv_method
    if fit_time_s is not None:
        out["fit_time_s"] = float(fit_time_s)
    if eval_time_s is not None:
        out["eval_time_s"] = float(eval_time_s)
    return out


def aggregate(df, group_cols=None, metric_cols=None):
    import pandas as pd

    if group_cols is None:
        group_cols = [
            c
            for c in [
                "motion",
                "representation",
                "knots",
                "n_sites",
                "fitter",
                "reg_kind",
                "reg_lam",
                "time_param",
                "sampling",
                "noise_std",
                "outlier_frac",
                "missing_frac",
            ]
            if c in df.columns
        ]
    if metric_cols is None:
        metric_cols = [
            c
            for c in df.columns
            if c.endswith("_rmse") or c in ("pos_mae", "pos_max", "overshoot", "peak_time_err",
                                            "settle_time_err", "isj_ratio")
        ]
    numeric = [c for c in metric_cols if pd.api.types.is_numeric_dtype(df[c])]
    grouped = df.groupby(group_cols, dropna=False)[numeric].agg(["mean", "std", "count"])
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
    return grouped.reset_index()


def aulc(df, metric="pos_rmse", group_cols=None, budget_col="budget"):
    import pandas as pd

    if group_cols is None:
        group_cols = [
            c
            for c in ["motion", "representation", "knots", "fitter", "time_param", "sampling",
                      "noise_std", "outlier_frac"]
            if c in df.columns
        ]
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        g = g.groupby(budget_col)[metric].mean().sort_index()
        if len(g) >= 2:
            area = trapz(g.values, np.log10(g.index.values.astype(float)))
        elif len(g) == 1:
            area = float(g.values[0])
        else:
            area = np.nan
        row = dict(zip(group_cols, key))
        row["aulc"] = area
        row["n_budgets"] = len(g)
        row["final"] = float(g.values[-1]) if len(g) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pareto_front(df, x="n_params_total", y="pos_rmse"):
    pts = df[[x, y]].dropna().to_numpy(dtype=float)
    if len(pts) == 0:
        return df.iloc[0:0]
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i == j or not keep[i]:
                continue
            dominated = (
                pts[j, 0] <= pts[i, 0]
                and pts[j, 1] <= pts[i, 1]
                and (pts[j, 0] < pts[i, 0] or pts[j, 1] < pts[i, 1])
            )
            if dominated:
                keep[i] = False
    return df.loc[df.index[keep]]
