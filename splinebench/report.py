"""Reproducible README table generation from results/*.jsonl.

Usage:
    python -m splinebench.report --results results --out results/ANALYSIS.md
    python -m splinebench.report --results results --readme README.md --facts

Tables are emitted between AUTO markers in README.md so re-running the full
suites later refreshes the numbers without hand-editing.
"""

import argparse
import glob
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
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("# Auto-generated analysis tables\n\n")
        for name, body in blocks.items():
            fh.write(f"## {name}\n\n{body}\n\n")
    print(f"wrote {args.out}")

    if args.facts:
        print(facts(df))

    if args.readme:
        patch_readme(args.readme, blocks)
        print(f"patched {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
