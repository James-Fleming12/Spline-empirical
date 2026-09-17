"""Supplementary runs that fill the gaps in the README quick-grid tables.

Quick suites cover budgets <= 50 and a subset of motions/noise configs.
This script adds:

  A. starter configs x {staccato, bounce, wobble} x budgets 5..200      (6.1)
  B. derivative-focused reps x {bounce, double_step, wobble, chirp}     (6.2)
  C. placement ablation at fixed n_sites=8, budget=50                   (6.3)
  D. robustness grid: 5 method rows x 7 noise configs                   (6.4)
  E. active samplers for the four sampling reps                         (6.5)

Usage:  python scripts/run_extra.py
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splinebench import suites
from splinebench.experiment import Condition, run_suite

OUT = "results/extra.jsonl"
EXTRA_MOTIONS = ["staccato", "bounce", "wobble"]
DERIV_REPS = ["hermite", "bspline5", "pspline", "gp_rbf", "dmp"]
DERIV_MOTIONS = ["bounce", "double_step", "wobble", "chirp"]
PLACEMENT_REPS = ["bspline3", "bspline5", "hermite", "pspline"]
PLACEMENT_KNOTS = [
    "uniform", "chord", "curvature", "feature_peaks",
    "split_merge", "greedy", "bayesopt", "active_residual",
]
PLACEMENT_MOTIONS = ["staccato", "bounce", "wobble", "chirp"]
ROBUST_METHODS = [
    {"representation": "hermite", "knots": "feature_peaks", "fitter": "least_squares"},
    {"representation": "hermite", "knots": "feature_peaks", "fitter": "huber"},
    {"representation": "hermite", "knots": "feature_peaks", "fitter": "ransac"},
    {"representation": "pspline", "knots": "split_merge", "fitter": "least_squares",
     "reg_kind": "diff", "reg_lam": 1e-3, "reg_order": 2},
    {"representation": "gp_matern52", "knots": "uniform", "fitter": "least_squares"},
]
NOISE_CONFIGS = [
    {"noise_std": 0.0},
    {"noise_std": 0.01},
    {"noise_std": 0.03},
    {"noise_std": 0.01, "outlier_frac": 0.05, "outlier_scale": 8.0},
    {"noise_std": 0.02, "outlier_frac": 0.15, "outlier_scale": 10.0},
    {"noise_std": 0.01, "missing_frac": 0.1},
    {"noise_std": 0.01, "bias": 0.05},
]
SAMPLING_REPS = ["bspline5", "gp_rbf", "dmp", "mlp"]
ACTIVE_SAMPLERS = ["active_gp", "active_residual", "query_by_committee"]
TABLE_SAMPLERS = ["random", "lhs", "sobol", "jerk_importance", "derivative_peaks",
                  "active_gp", "active_residual", "query_by_committee"]


def starter_configs():
    by_rep = {}
    for cond in suites.starter_conditions(quick=True):
        by_rep.setdefault(cond.representation, cond)
    return by_rep


def conditions():
    configs = starter_configs()
    conds = []

    for base in configs.values():
        for motion in EXTRA_MOTIONS:
            for budget in suites.QUICK_BUDGETS + [100, 200]:
                for seed in suites.QUICK_SEEDS:
                    conds.append(replace(base, motion=motion, budget=budget, seed=seed))

    for rep in DERIV_REPS:
        base = configs[rep] if rep in configs else Condition(representation=rep, knots="uniform")
        for motion in DERIV_MOTIONS:
            for budget in (20, 50):
                for seed in suites.QUICK_SEEDS:
                    conds.append(replace(base, motion=motion, budget=budget, seed=seed))
    for motion in DERIV_MOTIONS:
        for budget in (20, 50):
            for seed in suites.QUICK_SEEDS:
                conds.append(
                    Condition(
                        representation="gp_matern52",
                        knots="uniform",
                        fitter="least_squares",
                        motion=motion,
                        budget=budget,
                        seed=seed,
                        noise_std=0.01,
                    )
                )

    for rep in PLACEMENT_REPS:
        base = configs[rep]
        for knot in PLACEMENT_KNOTS:
            for motion in PLACEMENT_MOTIONS:
                for seed in suites.QUICK_SEEDS:
                    conds.append(
                        replace(base, knots=knot, motion=motion, budget=50, seed=seed, n_sites=8)
                    )

    for method in ROBUST_METHODS:
        for noise in NOISE_CONFIGS:
            for motion in ("staccato", "bounce"):
                for seed in suites.QUICK_SEEDS:
                    conds.append(Condition(motion=motion, budget=50, seed=seed, **method, **noise))

    for rep in SAMPLING_REPS:
        base = replace(configs[rep], knots="curvature", noise_std=0.02)
        for sampler in ACTIVE_SAMPLERS:
            for motion in ("staccato", "wobble"):
                for budget in (10, 20, 50):
                    for seed in (0, 1):
                        conds.append(
                            replace(base, sampling=sampler, motion=motion, budget=budget, seed=seed)
                        )

    for rep in SAMPLING_REPS:
        base = replace(configs[rep], knots="curvature", noise_std=0.02, n_sites=8)
        for sampler in TABLE_SAMPLERS:
            for motion in ("staccato", "wobble"):
                for budget in (10, 20, 50):
                    for seed in (0, 1):
                        conds.append(
                            replace(base, sampling=sampler, motion=motion, budget=budget, seed=seed)
                        )

    for rep in ["chebyshev", "fourier", "bspline3", "hermite"]:
        for motion in ["bang_bang", "pulses", "wobble"]:
            for budget in (20, 50):
                for seed in suites.QUICK_SEEDS:
                    conds.append(
                        Condition(representation=rep, knots="uniform", fitter="least_squares",
                                  motion=motion, budget=budget, seed=seed, noise_std=0.01)
                    )
    for rep in ["gp_rbf", "gp_matern52"]:
        for motion in ["bang_bang", "pulses"]:
            for budget in (20, 50):
                for seed in suites.QUICK_SEEDS:
                    conds.append(
                        Condition(representation=rep, knots="uniform", fitter="least_squares",
                                  motion=motion, budget=budget, seed=seed, noise_std=0.01)
                    )

    for reg_lam in (0.0, 1e-4, 1e-3, 1e-2):
        for motion in ["staccato", "bang_bang"]:
            for budget in (5, 10, 20):
                for seed in suites.QUICK_SEEDS:
                    conds.append(
                        Condition(representation="bspline3", knots="uniform", fitter="least_squares",
                                  reg_kind="diff" if reg_lam > 0 else "none", reg_lam=reg_lam,
                                  reg_order=2, n_sites=8, motion=motion, budget=budget,
                                  seed=seed, noise_std=0.03)
                    )

    return conds


def main():
    conds = conditions()
    print(f"supplementary conditions: {len(conds)}")
    os.makedirs("results", exist_ok=True)
    run_suite(conds, out_path=OUT, resume=True)


if __name__ == "__main__":
    main()
