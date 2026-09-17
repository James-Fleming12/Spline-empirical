"""Reproducible README table generation from results/*.jsonl.

Usage:
    python -m splinebench.report --results results --out results/ANALYSIS.md
    python -m splinebench.report --results results --readme README.md --facts

Tables are emitted between AUTO markers in README.md so re-running the full
suites later refreshes the numbers without hand-editing.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from . import experiment, plots, suites
from .utils import trapz

STARTER_BUDGETS = [5, 10, 20, 50, 100, 200]
STARTER_MOTIONS = ["staccato", "bounce", "wobble"]
DERIV_REPS = ["hermite", "bspline5", "pspline", "gp_rbf", "gp_matern52", "dmp"]
DERIV_MOTIONS = ["bounce", "double_step", "wobble", "chirp"]
PLACEMENT_REPS = ["bspline3", "bspline5", "hermite", "pspline"]
PLACEMENT_KNOTS = ["uniform", "chord", "curvature", "feature_peaks",
                   "split_merge", "greedy", "bayesopt", "active_residual"]
PLACEMENT_MOTIONS = ["staccato", "bounce", "wobble", "chirp"]
SAMPLING_REPS = ["bspline5", "gp_rbf", "dmp", "mlp"]
SAMPLING_ORDER = ["random", "lhs", "sobol", "jerk_importance", "derivative_peaks",
                  "active_gp", "active_residual", "query_by_committee"]
ROBUST_ROWS = [
    ("hermite + LS", "hermite", "feature_peaks", "least_squares", "none"),
    ("hermite + huber", "hermite", "feature_peaks", "huber", "none"),
    ("hermite + ransac", "hermite", "feature_peaks", "ransac", "none"),
    ("pspline (diff)", "pspline", "split_merge", "least_squares", "diff"),
    ("gp_matern52", "gp_matern52", "uniform", "least_squares", "none"),
]
NOISE_COLS = [
    ("clean", 0.0, 0.0, 0.0, 0.0),
    ("σ=0.01", 0.01, 0.0, 0.0, 0.0),
    ("σ=0.03", 0.03, 0.0, 0.0, 0.0),
    ("5% outliers", 0.01, 0.05, 0.0, 0.0),
    ("15% outliers", 0.02, 0.15, 0.0, 0.0),
    ("10% missing", 0.01, 0.0, 0.1, 0.0),
    ("bias 0.05", 0.01, 0.0, 0.0, 0.05),
]


def _starter_config_order():
    order = []
    for cond in suites.starter_conditions(quick=True):
        if cond.representation not in order:
            order.append(cond.representation)
    return order


def load_results(paths):
    if isinstance(paths, str):
        if os.path.isdir(paths):
            paths = sorted(glob.glob(os.path.join(paths, "*.jsonl")))
        else:
            paths = [paths]
    frames = []
    for path in paths:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            frames.append(experiment.load_results(path))
    if not frames:
        raise FileNotFoundError("no non-empty result files found")
    df = pd.concat(frames, ignore_index=True)
    if "_key" in df.columns:
        df = df.drop_duplicates(subset=["_key"]).reset_index(drop=True)
    return df


def _fmt(value, digits=3):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}g}"


def _ms(values, digits=3):
    values = pd.Series(values).dropna()
    if len(values) == 0:
        return "—"
    if len(values) == 1:
        return _fmt(float(values.iloc[0]), digits)
    mean, sd = float(values.mean()), float(values.std(ddof=0))
    return f"{_fmt(mean, digits)} ± {_fmt(sd, digits)}"


def _bold_if(values, value):
    return f"**{value}**" if value == min(values) else value


def _cell(a, b, digits=3):
    return f"{_fmt(a, digits)} / {_fmt(b, digits)}"


def table_starter(df):
    order = _starter_config_order()
    data = df[df["motion"].isin(STARTER_MOTIONS)]
    rows = []
    for motion in STARTER_MOTIONS:
        sub = data[data["motion"] == motion]
        entries = []
        for rep in order:
            g = sub[sub["representation"] == rep]
            if g.empty:
                continue
            knots = str(g["knots"].mode().iloc[0])
            means = []
            for budget in STARTER_BUDGETS:
                gb = g[g["budget"] == budget]
                means.append((budget, float(gb["pos_rmse"].mean()), len(gb)))
            available = [(b, m) for b, m, n in means if n > 0]
            if len(available) >= 2:
                aulc = trapz(
                    np.array([m for _, m in available]),
                    np.log10(np.array([b for b, _ in available], dtype=float)),
                )
            elif available:
                aulc = float(available[0][1])
            else:
                aulc = np.nan
            cells = [(_ms(g[g["budget"] == b]["pos_rmse"]) if len(g[g["budget"] == b]) else "—") for b in STARTER_BUDGETS]
            entries.append((aulc, [motion, rep, knots, "random", *cells, _fmt(aulc)]))
        entries.sort(key=lambda e: e[0] if np.isfinite(e[0]) else np.inf)
        rows.extend(entry[1] for entry in entries)
    header = ["motion", "representation", "knots", "sampler"] + [str(b) for b in STARTER_BUDGETS] + ["AULC ↓"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    lines.append("")
    lines.append("Mean ± sd over seeds (`n=3`); AULC is the area under the "
                 "error-vs-log-budget curve over the budgets with data.")
    return "\n".join(lines)


def table_derivative(df):
    data = df[(df["motion"].isin(DERIV_MOTIONS)) & (df["budget"] == 50)
              & (df["representation"].isin(DERIV_REPS))]
    header = ["motion", "method", "`vel_nrmse`", "`acc_nrmse`", "`jerk_nrmse`",
              "`peak_time_err`", "`overshoot`"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for motion in DERIV_MOTIONS:
        sub = data[data["motion"] == motion]
        rows = []
        for rep in DERIV_REPS:
            g = sub[sub["representation"] == rep]
            if g.empty:
                continue
            rows.append(
                [
                    motion,
                    rep,
                    f"{float(g['vel_nrmse'].mean()):.3g}" if "vel_nrmse" in g else "—",
                    f"{float(g['acc_nrmse'].mean()):.3g}" if "acc_nrmse" in g else "—",
                    f"{float(g['jerk_nrmse'].mean()):.3g}" if "jerk_nrmse" in g else "—",
                    f"{float(g['peak_time_err'].mean()):.3g}",
                    f"{float(g['overshoot'].mean()):.3g}",
                ]
            )
        if not rows:
            continue
        for col in range(2, len(header)):
            vals = [float(row[col]) for row in rows if row[col] != "—"]
            if not vals:
                continue
            for row in rows:
                if row[col] != "—" and float(row[col]) == min(vals):
                    row[col] = f"**{row[col]}**"
        lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    lines.append("")
    lines.append("Budget 50, mean over 3 seeds; derivatives normalized by the ground-truth "
                 "derivative RMS (`*_nrmse`, lower is better); best per motion/column in bold.")
    return "\n".join(lines)


def table_placement(df):
    data = df[(df["n_sites"] == 8) & (df["budget"] == 50) & (df["motion"].isin(PLACEMENT_MOTIONS))]
    blocks = []
    for rep in PLACEMENT_REPS:
        sub = data[data["representation"] == rep]
        if sub.empty:
            continue
        header = ["knots", *PLACEMENT_MOTIONS]
        lines = [f"**{rep}** (n_sites=8, budget=50, `pos_rmse` / `jerk_rmse`)", ""]
        lines += ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
        for knot in PLACEMENT_KNOTS:
            g = sub[sub["knots"] == knot]
            row = [knot]
            for motion in PLACEMENT_MOTIONS:
                gm = g[g["motion"] == motion]
                if gm.empty:
                    row.append("—")
                else:
                    row.append((float(gm["pos_rmse"].mean()), float(gm["jerk_rmse"].mean())))
            blocks.append(row)
        for motion_idx in range(len(PLACEMENT_MOTIONS)):
            vals = [row[1 + motion_idx][0] for row in blocks if isinstance(row[1 + motion_idx], tuple)]
            if not vals:
                continue
            for row in blocks:
                cell = row[1 + motion_idx]
                if isinstance(cell, tuple):
                    text = _cell(*cell)
                    if cell[0] == min(vals):
                        text = f"**{text}**"
                    row[1 + motion_idx] = text
        lines += ["| " + " | ".join(str(c) if not isinstance(c, str) else c for c in row) + " |" for row in blocks]
        blocks = []
        lines.append("")
        lines.append("Mean over 3 seeds; best position RMSE per motion in bold.")
        yield "\n".join(lines)


def _noise_key(row):
    return (round(float(row.get("noise_std", 0.0)), 3), round(float(row.get("outlier_frac", 0.0)), 3),
            round(float(row.get("missing_frac", 0.0)), 3), round(float(row.get("bias", 0.0)), 3))


def table_robustness(df):
    keys = {name: (std, out, miss, bias) for name, std, out, miss, bias in NOISE_COLS}
    df = df.copy()
    df["_noise_key"] = df.apply(_noise_key, axis=1)
    header = ["method \\ condition", *[name for name, *_ in NOISE_COLS]]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    rows = []
    for label, rep, fits, fitter, reg in ROBUST_ROWS:
        row = [label]
        for name, std, out, miss, bias in NOISE_COLS:
            g = df[(df["representation"] == rep) & (df["knots"] == fits)
                   & (df["fitter"] == fitter) & (df["reg_kind"] == reg)
                   & (df["budget"] == 50) & (df["sampling"] == "random")
                   & (df["time_param"] == "linear") & (df["_noise_key"] == keys[name])]
            row.append(_ms(g["pos_rmse"]))
        rows.append(row)
    for col in range(1, len(header)):
        numeric = []
        for row in rows:
            try:
                numeric.append(float(str(row[col]).split("±")[0]))
            except ValueError:
                numeric.append(np.inf)
        for row in rows:
            try:
                value = float(str(row[col]).split("±")[0])
            except ValueError:
                continue
            if value == min(numeric):
                row[col] = f"**{row[col]}**"
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    lines.append("")
    lines.append("Budget 50, mean ± sd over 3 seeds × {staccato, bounce}; best per column in bold.")
    return "\n".join(lines)


def table_sampling(df):
    data = df[(df["motion"].isin(["staccato", "wobble"])) & (df["representation"].isin(SAMPLING_REPS))
              & (df["knots"] == "curvature") & (df["n_sites"] == 8)]
    header = ["sampler", *SAMPLING_REPS]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    rows = []
    for sampler in SAMPLING_ORDER:
        row = [sampler]
        for rep in SAMPLING_REPS:
            g = data[(data["sampling"] == sampler) & (data["representation"] == rep)]
            scores = []
            for _, group in g.groupby(["motion", "seed"]):
                curve = group.groupby("budget")["pos_rmse"].mean().sort_index()
                if len(curve) >= 2:
                    scores.append(trapz(curve.values, np.log10(curve.index.values.astype(float))))
            row.append(_fmt(float(np.median(scores))) if scores else "—")
        rows.append(row)
    for col in range(1, len(header)):
        numeric = []
        for row in rows:
            try:
                numeric.append(float(row[col]))
            except ValueError:
                numeric.append(np.inf)
        for row in rows:
            try:
                if float(row[col]) == min(numeric):
                    row[col] = f"**{row[col]}**"
            except ValueError:
                pass
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    lines.append("")
    lines.append("Median AULC over (motion, seed) curves for budgets {10, 20, 50} at fixed "
                 "`n_sites=8`; lower is better, best per column in bold.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Iteration 2 tables
# ---------------------------------------------------------------------------

def _json_dict(value):
    if isinstance(value, dict):
        return value
    if not value or (isinstance(value, float) and not np.isfinite(value)):
        return {}
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return {}


def _noise_label(value):
    d = _json_dict(value)
    key = json.dumps(d, sort_keys=True)
    for name, kw in suites.I2_NOISE_MODELS:
        if json.dumps(kw, sort_keys=True) == key:
            return name
    return "other"


def _mode_label(value):
    d = _json_dict(value)
    return d.get("mode", "default")


def _method_label(rep, knot, fitter, reg_kind, reg_lam):
    label = f"{rep} + {knot} + {fitter}"
    if reg_kind and reg_kind != "none":
        label += f" + {reg_kind}({_fmt(reg_lam, 2)})"
    return label


def paired_bootstrap(a, b, n_boot=5000, seed=0):
    """Paired bootstrap CI and sign p-value for mean(a - b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return (np.nan, np.nan, np.nan, np.nan, 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(int(n_boot), len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2.0 * min(float((boot <= 0).mean()), float((boot >= 0).mean()))
    return (float(d.mean()), float(lo), float(hi), float(min(p, 1.0)), int(len(d)))


def _paired_series(df, mask_a, mask_b, metric, keys=("motion", "budget", "seed")):
    a = df[mask_a].groupby(list(keys))[metric].mean()
    b = df[mask_b].groupby(list(keys))[metric].mean()
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return joined["a"].values, joined["b"].values


def table_i2_conditioning(df):
    data = df.copy()
    if "noise_kwargs" in data.columns:
        data["_noise"] = data["noise_kwargs"].apply(_noise_label)
        data = data[data["_noise"] == "iid"]
    has_rank = "design_rank" in data.columns
    rows = []
    for rep, g in data.groupby("representation"):
        linear = g[g["design_rank"].notna()] if has_rank else g.iloc[0:0]
        if len(linear):
            finite = np.isfinite(linear["design_cond"])
            pct = f"{100 * float((~finite).mean()):.0f}"
            med_cond = _fmt(linear.loc[finite, "design_cond"].median()) if finite.any() else "inf"
        else:
            pct, med_cond = "—", "—"
        rows.append(
            [
                rep,
                pct,
                med_cond,
                _fmt(g["eff_dof"].median() if "eff_dof" in g else np.nan),
                _fmt(g["eff_dof_ratio"].median() if "eff_dof_ratio" in g else np.nan),
                _fmt(g["gp_kernel_cond"].median() if "gp_kernel_cond" in g else np.nan),
                _fmt(g["resid_acf1"].median() if "resid_acf1" in g else np.nan),
            ]
        )
    header = ["representation", "% rank-def", "median `design_cond`", "median `eff_dof`",
              "median `eff_dof_ratio`", "median `gp_kernel_cond`", "median `resid_acf1`"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    lines.append("")
    lines.append("`i2_core`, iid noise, budgets 10–50; `design_cond` infinite/NaN counts as rank-deficient; "
                 "medians over all motions/seeds.")
    return "\n".join(lines)


def table_i2_noise(df):
    header = ["method"] + [name for name, _ in suites.I2_NOISE_MODELS]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    rows = []
    for rep, knot, fitter, reg_kind, reg_lam in suites.I2_NOISE_METHODS:
        label = _method_label(rep, knot, fitter, reg_kind, reg_lam)
        mask = (
            (df["representation"] == rep) & (df["knots"] == knot)
            & (df["fitter"] == fitter) & (df["reg_kind"] == reg_kind)
            & (df["reg_lam"] == reg_lam)
        )
        g = df[mask]
        if g.empty:
            continue
        row = [label]
        for name, _ in suites.I2_NOISE_MODELS:
            sub = g[g["noise_kwargs"].apply(_noise_label) == name]["pos_rmse"]
            row.append(_fmt(sub.mean(), 3) if len(sub) else "—")
        rows.append(row)
    for col in range(1, len(header)):
        vals = []
        for row in rows:
            try:
                vals.append(float(row[col]))
            except ValueError:
                vals.append(np.inf)
        best = min(vals)
        for row in rows:
            try:
                if float(row[col]) == best:
                    row[col] = f"**{row[col]}**"
            except ValueError:
                pass
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    lines.append("")
    lines.append("`i2_noise`: mean hold-out `pos_rmse` (lower is better), over {staccato, bounce, wobble} × "
                 "budgets {20, 50} × 5 seeds; best *absolute* error per column in bold. Compare each cell with "
                 "its own `iid` column to read off the degradation ratio (e.g. RANSAC is >400× worse under "
                 "per-axis correlation and 3× worse under missing bursts).")
    return "\n".join(lines)


def table_i2_timeparam(df):
    reps = ["gp_matern52", "pspline", "bspline3"]
    tps = ["linear", "chord", "centripetal", "accel", "jerk"]
    header = ["representation", "time param", "realistic `pos_rmse`", "oracle `pos_rmse`",
              "realistic `peak_time_err`", "oracle `peak_time_err`"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for rep in reps:
        for tp in tps:
            g = df[(df["representation"] == rep) & (df["time_param"] == tp)]
            if g.empty:
                continue
            real = g[~g["time_param_oracle"]]
            orac = g[g["time_param_oracle"]]
            lines.append(
                "| " + " | ".join(
                    [rep, tp, _fmt(real["pos_rmse"].mean()), _fmt(orac["pos_rmse"].mean()),
                     _fmt(real["peak_time_err"].mean()), _fmt(orac["peak_time_err"].mean())]
                ) + " |"
            )
    lines.append("")
    lines.append("`i2_timeparam`: mean over {staccato, double_step, wobble} × budgets {20, 50} × 5 seeds; "
                 "the spline is fit in the warped `u(t)` and metrics compose derivatives through it.")
    return "\n".join(lines)


def table_i2_extrap(df):
    reps = ["gp_matern52", "gp_rbf", "pspline", "bspline3", "catmull_rom", "hermite"]
    splits = ["none", "early", "late", "middle", "interp"]
    header = ["representation"] + splits + ["holdout/trainwin (mean)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for rep in reps:
        row = [rep]
        ratios = []
        for split in splits:
            g = df[(df["representation"] == rep) & (df["split"] == split)]
            if g.empty:
                row.append("—")
                continue
            metric = "holdout_pos_rmse" if split != "none" else "pos_rmse"
            row.append(_fmt(g[metric].mean()))
            if split != "none":
                ratios.append(float(g["holdout_pos_rmse"].mean() / (g["trainwin_pos_rmse"].mean() + 1e-12)))
        row.append(_fmt(float(np.mean(ratios))) if ratios else "—")
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines.append("")
    lines.append("`i2_extrap`: hold-out `pos_rmse`; `early`/`late`/`middle` extrapolate beyond the training "
                 "window, `interp` holds out a disjoint middle interval (train on 0–0.45 ∪ 0.55–1). "
                 "Mean over {staccato, bounce, chirp} × budgets {20, 50} × 5 seeds, linear time.")
    return "\n".join(lines)


def table_i2_adversarial(df):
    header = ["representation", "reg", "mode", "selected sites", "median `design_cond`",
              "`pos_rmse`", "`jerk_rmse`"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for rep, g_rep in df.groupby("representation"):
        for (reg_kind, mode), g in g_rep.groupby([g_rep["reg_kind"], g_rep["knot_kwargs"].apply(_mode_label)]):
            cond = g["design_cond"].replace([np.inf, -np.inf], np.nan)
            lines.append(
                "| " + " | ".join(
                    [rep, reg_kind, mode, _fmt(g["n_selected_sites"].mean()),
                     "inf" if np.isinf(g["design_cond"]).any() else _fmt(cond.median()),
                     _fmt(g["pos_rmse"].mean()), _fmt(g["jerk_rmse"].mean())]
                ) + " |"
            )
    lines.append("")
    lines.append("`i2_adversarial`, budget 50, mean over {staccato, bounce, wobble} × 5 seeds; `uniform` is the "
                 "reference row. `near_duplicate` sites are 1.1e-4 apart, below any useful resolution.")
    return "\n".join(lines)


def table_i2_knotcount(df):
    header = ["method", "knots", "reg", "mean sites", "B20 `pos_rmse`", "B50 `pos_rmse`", "`jerk_rmse`"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for (rep, knot, reg_kind, reg_lam), g in df.groupby(
        ["representation", "knots", "reg_kind", "reg_lam"]
    ):
        lines.append(
            "| " + " | ".join(
                [rep, knot, reg_kind if reg_kind != "none" else "—", _fmt(g["n_selected_sites"].mean()),
                 _fmt(g[g["budget"] == 20]["pos_rmse"].mean()), _fmt(g[g["budget"] == 50]["pos_rmse"].mean()),
                 _fmt(g["jerk_rmse"].mean())]
            ) + " |"
        )
    lines.append("")
    lines.append("`i2_knotcount`, mean over {staccato, double_step, bounce, wobble} × 5 seeds; `n_sites` is the "
                 "budget/3 default upper bound, `mean sites` is what the placer actually used.")
    return "\n".join(lines)


def table_i2_stats(core, adversarial, knotcount):
    def cond(df, rep, knot, fitter, reg_kind, reg_lam, budget=None):
        m = (
            (df["representation"] == rep) & (df["knots"] == knot) & (df["fitter"] == fitter)
            & (df["reg_kind"] == reg_kind) & (df["reg_lam"] == reg_lam)
        )
        if budget is not None:
            m &= df["budget"] == budget
        return m

    comparisons = []
    a = cond(core, "gp_matern52", "uniform", "least_squares", "none", 0.0)
    b = cond(core, "pspline", "split_merge", "least_squares", "diff", 1e-3)
    comparisons.append(("`i2_core`", "GP-Matérn − pspline (B20)", "pos_rmse", a & (core["budget"] == 20), b & (core["budget"] == 20), core))
    comparisons.append(("`i2_core`", "GP-Matérn − pspline (B50)", "pos_rmse", a & (core["budget"] == 50), b & (core["budget"] == 50), core))
    c = cond(core, "catmull_rom", "uniform", "least_squares", "none", 0.0)
    comparisons.append(("`i2_core`", "GP-Matérn − Catmull-Rom", "pos_rmse", a, c, core))
    adv_pen = (adversarial["knots"] == "clustered") & (adversarial["reg_kind"] == "diff")
    adv_plain = (adversarial["knots"] == "clustered") & (adversarial["reg_kind"] == "none")
    comparisons.append(("`i2_adversarial`", "clustered: diff − none", "pos_rmse", adv_pen, adv_plain, adversarial))
    kc_cv = (knotcount["knots"] == "cv") & (knotcount["representation"] == "bspline3")
    kc_fix = (knotcount["knots"] == "split_merge") & (knotcount["representation"] == "bspline3") & (knotcount["reg_kind"] == "none")
    comparisons.append(("`i2_knotcount`", "bspline3: CV − fixed knots (B20)", "pos_rmse", kc_cv & (knotcount["budget"] == 20), kc_fix & (knotcount["budget"] == 20), knotcount))
    comparisons.append(("`i2_knotcount`", "bspline3: CV − fixed knots (B50)", "pos_rmse", kc_cv & (knotcount["budget"] == 50), kc_fix & (knotcount["budget"] == 50), knotcount))

    header = ["suite", "paired comparison", "metric", "mean Δ [95% CI]", "p", "n"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for suite, label, metric, mask_a, mask_b, df in comparisons:
        va, vb = _paired_series(df, mask_a, mask_b, metric)
        mean, lo, hi, p, n = paired_bootstrap(va, vb)
        if not np.isfinite(mean):
            lines.append(f"| {suite} | {label} | `{metric}` | — | — | 0 |")
            continue
        ci = f"{_fmt(mean)} [{_fmt(lo)}, {_fmt(hi)}]"
        lines.append(f"| {suite} | {label} | `{metric}` | {ci} | {_fmt(p)} | {n} |")
    lines.append("")
    lines.append("Paired bootstrap (5000 resamples) over matched (motion, budget, seed) cells; "
                 "Δ = first − second, negative favours the first method. p is the two-sided bootstrap sign p-value.")
    return "\n".join(lines)


def table_figures(df=None, results_dir="results"):
    figures = [
        ("starter_sample_efficiency.png", "`--plots results/starter`"),
        ("starter_pareto.png", "`--plots results/starter`"),
        ("robustness.png", "`plots.plot_robustness`"),
    ]
    lines = ["| figure | command | status |", "|---|---|---|"]
    for name, command in figures:
        path = os.path.join(results_dir, name)
        status = "generated" if os.path.exists(path) else "pending"
        lines.append(f"| `{path}` | {command} | {status} |")
    return "\n".join(lines)


def facts(df):
    out = []
    starter = df[df["motion"].isin(STARTER_MOTIONS)]
    out.append("== 6.1 best per motion by AULC ==")
    for motion in STARTER_MOTIONS:
        sub = starter[starter["motion"] == motion]
        scores = []
        for rep, g in sub.groupby("representation"):
            rows = g.groupby("budget")["pos_rmse"].mean().sort_index()
            if len(rows) >= 2:
                scores.append((trapz(rows.values, np.log10(rows.index.values.astype(float))), rep))
        scores.sort()
        out.append(f"{motion}: " + ", ".join(f"{rep}={score:.4g}" for score, rep in scores[:5]))

    out.append("")
    out.append("== 6.1 budget-20 leader ==")
    for motion in STARTER_MOTIONS:
        sub = starter[(starter["motion"] == motion) & (starter["budget"] == 20)]
        means = sub.groupby("representation")["pos_rmse"].mean().sort_values()
        out.append(f"{motion}: " + ", ".join(f"{rep}={v:.4g}" for rep, v in means.head(4).items()))

    out.append("")
    out.append("== 6.3 placement at n_sites=8, budget=50 ==")
    place = df[(df["n_sites"] == 8) & (df["budget"] == 50)]
    for rep in PLACEMENT_REPS:
        for motion in PLACEMENT_MOTIONS:
            sub = place[(place["representation"] == rep) & (place["motion"] == motion)]
            if sub.empty:
                continue
            means = sub.groupby("knots")["pos_rmse"].mean().sort_values()
            uniform = means.get("uniform", np.nan)
            best = means.index[0]
            out.append(f"{rep}/{motion}: uniform={uniform:.4g} best={best}={means.iloc[0]:.4g}")

    out.append("")
    out.append("== H2 global vs piecewise (knots=uniform, budget 50) ==")
    for motion in ["bang_bang", "pulses", "wobble"]:
        sub = df[(df["motion"] == motion) & (df["budget"] == 50) & (df["knots"] == "uniform")]
        means = sub.groupby("representation")["pos_rmse"].mean().sort_values()
        out.append(f"{motion}: " + ", ".join(f"{rep}={v:.4g}" for rep, v in means.items()))

    out.append("")
    out.append("== H3 difference-penalty sweep (bspline3, uniform, n_sites=8, sigma=0.03) ==")
    sub = df[(df["representation"] == "bspline3") & (df["knots"] == "uniform") & (df["n_sites"] == 8)]
    for motion in ["staccato", "bang_bang"]:
        for budget in (5, 10, 20):
            g = sub[(sub["motion"] == motion) & (sub["budget"] == budget)]
            entries = [f"lam={lam:g}:{gg['pos_rmse'].mean():.3g}" for lam, gg in g.groupby("reg_lam")]
            if entries:
                out.append(f"{motion}/B{budget}: " + ", ".join(entries))

    out.append("")
    out.append("== 6.4 robust rows (budget 50, mean) ==")
    robust = df[(df["budget"] == 50) & (df["motion"].isin(["staccato", "bounce"]))]
    for noise in NOISE_COLS:
        name, std, out_, miss, bias = noise
        mask = (robust["noise_std"] == std) & (robust["outlier_frac"] == out_) \
            & (robust["missing_frac"] == miss) & (robust["bias"] == bias)
        sub = robust[mask & (robust["sampling"] == "random") & (robust["time_param"] == "linear")]
        if sub.empty:
            continue
        entries = []
        for label, rep, fits, fitter, reg in ROBUST_ROWS:
            g = sub[(sub["representation"] == rep) & (sub["knots"] == fits)
                    & (sub["fitter"] == fitter) & (sub["reg_kind"] == reg)]
            if len(g):
                entries.append(f"{label}={g['pos_rmse'].mean():.4g}")
        out.append(f"{name}: " + ", ".join(entries))

    out.append("")
    out.append("== 6.5 sampling AULC (median over motion x seed, n_sites=8) ==")
    sampling = df[(df["motion"].isin(["staccato", "wobble"])) & (df["knots"] == "curvature")
                  & (df["representation"].isin(SAMPLING_REPS)) & (df["n_sites"] == 8)]
    for rep in SAMPLING_REPS:
        rows = []
        for sampler, g in sampling[sampling["representation"] == rep].groupby("sampling"):
            scores = []
            for _, group in g.groupby(["motion", "seed"]):
                curve = group.groupby("budget")["pos_rmse"].mean().sort_index()
                if len(curve) >= 2:
                    scores.append(trapz(curve.values, np.log10(curve.index.values.astype(float))))
            if scores:
                rows.append((float(np.median(scores)), sampler))
        rows.sort()
        if rows:
            out.append(f"{rep}: " + ", ".join(f"{s}={a:.4g}" for a, s in rows))
    out.append("")
    out.append("== fitters quick: LS vs huber vs ransac under 10% outliers ==")
    fit = df[(df["reg_kind"] == "none") & (df["outlier_frac"] == 0.1)]
    for rep in ["hermite", "bspline5"]:
        for fitter in ["least_squares", "huber", "ransac", "adam"]:
            g = fit[(fit["representation"] == rep) & (fit["fitter"] == fitter)]
            if len(g):
                out.append(f"{rep}/{fitter}: budget10={g[g['budget']==10]['pos_rmse'].mean():.4g} "
                           f"budget50={g[g['budget']==50]['pos_rmse'].mean():.4g}")
    return "\n".join(out)


def patch_readme(path, blocks):
    with open(path, "r") as fh:
        text = fh.read()
    for name, body in blocks.items():
        start = f"<!-- AUTO:{name} START -->"
        end = f"<!-- AUTO:{name} END -->"
        if start not in text or end not in text:
            continue
        head = text[: text.index(start) + len(start)]
        tail = text[text.index(end):]
        text = head + "\n" + body.strip() + "\n" + tail
    with open(path, "w") as fh:
        fh.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate README tables from results")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/ANALYSIS.md")
    parser.add_argument("--readme", default=None, help="patch AUTO blocks in this README")
    parser.add_argument("--facts", action="store_true", help="print decision-relevant comparisons")
    parser.add_argument("--plots", action="store_true", help="regenerate figures into results/")
    parser.add_argument("--i2-dir", default="results/iteration2", help="directory with i2_*.jsonl")
    args = parser.parse_args(argv)

    df = load_results(args.results)
    print(f"loaded {len(df)} records")

    if args.plots:
        sample = df[(df["sampling"] == "random") & (df["motion"].isin(["staccato", "bounce", "wobble"]))]
        plots.plot_sample_efficiency(sample, "results/starter_sample_efficiency.png", motions=["staccato"])
        plots.plot_pareto(df, "results/starter_pareto.png")
        try:
            plots.plot_robustness(df[df["fitter"].isin(["least_squares", "huber", "ransac"])],
                                  "results/robustness.png")
        except Exception as exc:
            print(f"robustness plot skipped: {exc}")

    blocks = {
        "6.1": table_starter(df),
        "6.2": table_derivative(df),
        "6.3": "\n\n".join(table_placement(df)),
        "6.4": table_robustness(df),
        "6.5": table_sampling(df),
        "6.6": table_figures(df, args.results if os.path.isdir(args.results) else "results"),
    }

    i2_blocks = {}
    i2_dir = args.i2_dir
    needed = ["i2_core", "i2_noise", "i2_timeparam", "i2_extrap", "i2_adversarial", "i2_knotcount"]
    i2_paths = {name: os.path.join(i2_dir, f"{name}.jsonl") for name in needed}
    if all(os.path.exists(p) for p in i2_paths.values()):
        i2 = {name: load_results(p) for name, p in i2_paths.items()}
        i2_blocks = {
            "I2.1": table_i2_conditioning(i2["i2_core"]),
            "I2.2": table_i2_noise(i2["i2_noise"]),
            "I2.3": table_i2_timeparam(i2["i2_timeparam"]),
            "I2.4": table_i2_extrap(i2["i2_extrap"]),
            "I2.5": table_i2_adversarial(i2["i2_adversarial"]),
            "I2.6": table_i2_knotcount(i2["i2_knotcount"]),
            "I2.7": table_i2_stats(i2["i2_core"], i2["i2_adversarial"], i2["i2_knotcount"]),
        }
        print(f"loaded iteration-2 records: " + ", ".join(f"{n}={len(v)}" for n, v in i2.items()))
    else:
        print(f"iteration-2 results not found under {i2_dir}; skipping I2 tables")

    all_blocks = {**blocks, **i2_blocks}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("# Auto-generated analysis tables\n\n")
        for name, body in all_blocks.items():
            fh.write(f"## {name}\n\n{body}\n\n")
    print(f"wrote {args.out}")

    if args.facts:
        print(facts(df))

    if args.readme:
        patch_readme(args.readme, all_blocks)
        print(f"patched {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
