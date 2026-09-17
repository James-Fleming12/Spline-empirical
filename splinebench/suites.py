import itertools

from .experiment import Condition

QUICK_MOTIONS = ["staccato", "bang_bang", "wobble"]
FULL_MOTIONS = ["staccato", "double_step", "bang_bang", "bounce", "wobble", "chirp", "combo"]
QUICK_BUDGETS = [5, 10, 20, 50]
FULL_BUDGETS = [5, 10, 20, 50, 100, 200]
QUICK_SEEDS = [0, 1, 2]
FULL_SEEDS = [0, 1, 2, 3, 4]


def expand(rows):
    out = []
    for row in rows:
        keys = list(row)
        for values in itertools.product(*[row[k] for k in keys]):
            out.append(Condition(**dict(zip(keys, values))))
    return out


def smoke_conditions():
    rows = [
        {
            "motion": ["staccato"],
            "representation": ["hermite", "bspline3", "gp_rbf"],
            "knots": ["uniform"],
            "fitter": ["least_squares"],
            "sampling": ["random"],
            "budget": [8, 16],
            "seed": [0],
            "noise_std": [0.0],
        },
    ]
    return expand(rows)


def starter_conditions(quick=True):
    motions = QUICK_MOTIONS if quick else FULL_MOTIONS
    budgets = QUICK_BUDGETS if quick else FULL_BUDGETS
    seeds = QUICK_SEEDS if quick else FULL_SEEDS
    reps = [
        ("hermite", "feature_peaks", "least_squares", "none", 0.0, 2, "random", {}),
        ("bspline3", "chord", "least_squares", "none", 0.0, 2, "random", {}),
        ("bspline5", "curvature", "least_squares", "none", 0.0, 2, "random", {}),
        ("pspline", "split_merge", "least_squares", "diff", 1e-3, 2, "random", {}),
        ("catmull_rom", "uniform", "least_squares", "none", 0.0, 2, "random", {}),
        ("gp_rbf", "uniform", "least_squares", "none", 0.0, 2, "random", {}),
        ("dmp", "uniform", "least_squares", "none", 0.0, 2, "random", {}),
        ("mlp", "uniform", "least_squares", "none", 0.0, 2, "random", {"iters": 600}),
        ("bspline_mlp", "uniform", "least_squares", "diff", 1e-3, 2, "random", {"iters": 400}),
    ]
    rows = []
    for rep, knot, fitter, reg_kind, reg_lam, reg_order, sampling, rep_kwargs in reps:
        rows.append(
            {
                "representation": [rep],
                "knots": [knot],
                "fitter": [fitter],
                "reg_kind": [reg_kind],
                "reg_lam": [reg_lam],
                "reg_order": [reg_order],
                "sampling": [sampling],
                "time_param": ["linear"] if quick else ["linear", "chord"],
                "motion": motions,
                "budget": budgets,
                "seed": seeds,
                "noise_std": [0.01],
                "rep_kwargs": [rep_kwargs],
            }
        )
    return expand(rows)


def placement_conditions(quick=True):
    knots = (
        ["uniform", "quantile", "chord", "curvature", "feature_peaks", "split_merge", "rdp"]
        if quick
        else [
            "uniform", "quantile", "chord", "centripetal", "curvature", "feature_peaks", "rdp",
            "split_merge", "greedy", "kmeans", "farthest_point", "bayesopt", "active_residual",
        ]
    )
    reps = ["bspline3", "bspline5", "hermite", "pspline"] if quick else [
        "bspline2", "bspline3", "bspline5", "bspline7", "hermite", "catmull_rom", "pspline", "nurbs"
    ]
    budgets = [10, 20, 50] if quick else FULL_BUDGETS
    seeds = [0, 1] if quick else FULL_SEEDS
    return expand(
        [
            {
                "representation": reps,
                "knots": knots,
                "fitter": ["least_squares"],
                "reg_kind": ["none"],
                "reg_lam": [0.0],
                "sampling": ["random"],
                "time_param": ["linear"],
                "motion": QUICK_MOTIONS if quick else FULL_MOTIONS,
                "budget": budgets,
                "seed": seeds,
                "noise_std": [0.01],
                "knot_kwargs": [{}],
            }
        ]
    )


def fitter_conditions(quick=True):
    fitters = ["least_squares", "ridge", "huber", "ransac", "adam"] if quick else [
        "least_squares", "weighted", "ridge", "huber", "ransac", "lasso", "elastic_net", "tv", "adam"
    ]
    seeds = [0, 1] if quick else FULL_SEEDS
    return expand(
        [
            {
                "representation": ["bspline5", "hermite"],
                "knots": ["split_merge"],
                "fitter": fitters,
                "reg_kind": ["none"],
                "reg_lam": [0.0],
                "sampling": ["random"],
                "time_param": ["linear"],
                "motion": QUICK_MOTIONS if quick else FULL_MOTIONS,
                "budget": [10, 50] if quick else FULL_BUDGETS,
                "seed": seeds,
                "noise_std": [0.02],
                "outlier_frac": [0.05, 0.15] if not quick else [0.1],
                "outlier_scale": [8.0],
            }
        ]
    )


def sampling_conditions(quick=True):
    sampling = ["random", "lhs", "sobol", "jerk_importance", "derivative_peaks"] if quick else [
        "random", "grid", "stratified", "lhs", "sobol", "halton", "farthest_point",
        "jerk_importance", "derivative_peaks", "hybrid", "active_gp", "active_residual",
        "query_by_committee",
    ]
    reps = ["bspline5", "gp_rbf", "dmp", "mlp"]
    seeds = [0, 1] if quick else FULL_SEEDS
    return expand(
        [
            {
                "representation": reps,
                "knots": ["curvature"],
                "fitter": ["least_squares"],
                "reg_kind": ["none"],
                "reg_lam": [0.0],
                "sampling": sampling,
                "time_param": ["linear"],
                "motion": ["staccato", "wobble"] if quick else FULL_MOTIONS,
                "budget": [10, 20, 50] if quick else FULL_BUDGETS,
                "seed": seeds,
                "noise_std": [0.02],
            }
        ]
    )


def robustness_conditions(quick=True):
    noise = [
        {"noise_std": 0.0},
        {"noise_std": 0.01},
        {"noise_std": 0.03},
        {"noise_std": 0.01, "outlier_frac": 0.05, "outlier_scale": 8.0},
        {"noise_std": 0.02, "outlier_frac": 0.15, "outlier_scale": 10.0},
        {"noise_std": 0.01, "missing_frac": 0.1},
        {"noise_std": 0.01, "bias": 0.05},
    ]
    if quick:
        noise = noise[:3]
    rows = []
    for n in noise:
        rows.append(
            {
                "representation": ["hermite", "bspline5", "pspline", "gp_rbf", "bspline_mlp"],
                "knots": ["split_merge"],
                "fitter": ["least_squares"] if quick else ["least_squares", "huber", "ransac"],
                "reg_kind": ["none"],
                "reg_lam": [0.0],
                "sampling": ["random"],
                "time_param": ["linear"],
                "motion": ["staccato", "bounce"] if quick else ["staccato", "bang_bang", "bounce", "wobble"],
                "budget": [10, 50] if quick else FULL_BUDGETS,
                "seed": [0, 1] if quick else FULL_SEEDS,
                **{k: [v] for k, v in n.items()},
            }
        )
    return expand(rows)


def full_conditions():
    return (
        starter_conditions(quick=False)
        + placement_conditions(quick=False)
        + fitter_conditions(quick=False)
        + sampling_conditions(quick=False)
        + robustness_conditions(quick=False)
    )


# ---------------------------------------------------------------------------
# Iteration 2: stress the Iteration 1 winners (GP-Matern, adaptive knots,
# penalized cubic B-splines, Sobol) under realistic noise, time
# parameterization, extrapolation, adversarial conditioning and auto-tuned
# knot/penalty counts.
# ---------------------------------------------------------------------------

I2_SEEDS = [0, 1, 2, 3, 4]
I2_QUICK_SEEDS = [0, 1, 2]
I2_BUDGETS = [10, 20, 50]
I2_ALL_MOTIONS = [
    "orbit", "staccato", "double_step", "bang_bang", "bounce",
    "pulses", "wobble", "chirp", "combo", "contact_drop",
]

# (representation, knots, fitter, reg_kind, reg_lam)
I2_NOISE_METHODS = [
    ("gp_matern52", "uniform", "least_squares", "none", 0.0),
    ("pspline", "split_merge", "least_squares", "diff", 1e-3),
    ("hermite", "feature_peaks", "huber", "none", 0.0),
    ("catmull_rom", "uniform", "least_squares", "none", 0.0),
    ("bspline3", "split_merge", "ransac", "none", 0.0),
]
I2_NOISE_MODELS = [
    ("iid", {}),
    ("ar1_0.90", {"ar1_rho": 0.90}),
    ("ar1_0.98", {"ar1_rho": 0.98}),
    ("hetero_speed", {"hetero": 2.0}),
    ("speed_outliers", {"speed_outlier": 0.15}),
    ("missing_bursts", {"missing_bursts": 2}),
    ("quantized", {"quantize": 0.02}),
    ("axis_corr", {"axis_corr": 0.85}),
    ("bias_drift", {"bias_drift": 0.05}),
]


def _method_row(rep, knot, fitter, reg_kind, reg_lam, motions, budgets, seeds, **extra):
    return {
        "representation": [rep],
        "knots": [knot],
        "fitter": [fitter],
        "reg_kind": [reg_kind],
        "reg_lam": [reg_lam],
        "reg_order": [2],
        "sampling": ["random"],
        "time_param": ["linear"],
        "motion": motions,
        "budget": budgets,
        "seed": seeds,
        **extra,
    }


def iteration2_core_conditions(quick=True):
    seeds = I2_QUICK_SEEDS if quick else I2_SEEDS
    reps = [
        ("gp_matern52", "uniform", "least_squares", "none", 0.0),
        ("gp_rbf", "uniform", "least_squares", "none", 0.0),
        ("pspline", "split_merge", "least_squares", "diff", 1e-3),
        ("bspline3", "split_merge", "least_squares", "none", 0.0),
        ("catmull_rom", "uniform", "least_squares", "none", 0.0),
        ("hermite", "feature_peaks", "least_squares", "none", 0.0),
    ]
    rows = [
        _method_row(rep, kn, ft, rk, rl, I2_ALL_MOTIONS, I2_BUDGETS, seeds, noise_std=[0.01])
        for rep, kn, ft, rk, rl in reps
    ]
    return expand(rows)


def iteration2_noise_conditions(quick=True):
    seeds = I2_QUICK_SEEDS if quick else I2_SEEDS
    rows = []
    for name, nkw in I2_NOISE_MODELS:
        for rep, kn, ft, rk, rl in I2_NOISE_METHODS:
            rows.append(
                _method_row(
                    rep, kn, ft, rk, rl,
                    ["staccato", "bounce", "wobble"], [20, 50], seeds,
                    noise_std=[0.01], noise_kwargs=[dict(nkw)],
                )
            )
    return expand(rows)


def iteration2_timeparam_conditions(quick=True):
    seeds = I2_QUICK_SEEDS if quick else I2_SEEDS
    reps = [
        ("gp_matern52", "uniform", "least_squares", "none", 0.0),
        ("pspline", "split_merge", "least_squares", "diff", 1e-3),
        ("bspline3", "split_merge", "least_squares", "none", 0.0),
    ]
    rows = []
    for rep, kn, ft, rk, rl in reps:
        for tp, oracle in itertools.product(["linear", "chord", "centripetal", "accel", "jerk"], [False, True]):
            rows.append(
                _method_row(
                    rep, kn, ft, rk, rl,
                    ["staccato", "double_step", "wobble"], [20, 50], seeds,
                    noise_std=[0.01], time_param=[tp], time_param_oracle=[oracle],
                )
            )
    return expand(rows)


def iteration2_extrapolation_conditions(quick=True):
    seeds = I2_QUICK_SEEDS if quick else I2_SEEDS
    reps = [
        ("gp_matern52", "uniform", "least_squares", "none", 0.0),
        ("gp_rbf", "uniform", "least_squares", "none", 0.0),
        ("pspline", "split_merge", "least_squares", "diff", 1e-3),
        ("bspline3", "split_merge", "least_squares", "none", 0.0),
        ("catmull_rom", "uniform", "least_squares", "none", 0.0),
        ("hermite", "feature_peaks", "least_squares", "none", 0.0),
    ]
    rows = []
    for rep, kn, ft, rk, rl in reps:
        for split in ["none", "early", "late", "middle", "interp"]:
            rows.append(
                _method_row(
                    rep, kn, ft, rk, rl,
                    ["staccato", "bounce", "chirp"], [20, 50], seeds,
                    noise_std=[0.01], split=[split],
                )
            )
    return expand(rows)


def iteration2_adversarial_conditions(quick=True):
    seeds = I2_QUICK_SEEDS if quick else I2_SEEDS
    rows = [
        _method_row("bspline3", "uniform", "least_squares", "none", 0.0,
                    ["staccato", "bounce", "wobble"], [50], seeds, noise_std=[0.01]),
        _method_row("bspline3", "clustered", "least_squares", "none", 0.0,
                    ["staccato", "bounce", "wobble"], [50], seeds, noise_std=[0.01],
                    knot_kwargs=[{"mode": "cluster"}]),
        _method_row("bspline3", "clustered", "least_squares", "diff", 1e-3,
                    ["staccato", "bounce", "wobble"], [50], seeds, noise_std=[0.01],
                    knot_kwargs=[{"mode": "cluster"}]),
        _method_row("bspline5", "clustered", "least_squares", "none", 0.0,
                    ["staccato", "bounce", "wobble"], [50], seeds, noise_std=[0.01],
                    knot_kwargs=[{"mode": "cluster"}]),
        _method_row("bspline5", "clustered", "least_squares", "diff", 1e-3,
                    ["staccato", "bounce", "wobble"], [50], seeds, noise_std=[0.01],
                    knot_kwargs=[{"mode": "cluster"}]),
        _method_row("bspline7", "clustered", "least_squares", "none", 0.0,
                    ["staccato", "bounce", "wobble"], [50], seeds, noise_std=[0.01],
                    knot_kwargs=[{"mode": "cluster"}]),
        _method_row("bspline7", "clustered", "least_squares", "diff", 1e-3,
                    ["staccato", "bounce", "wobble"], [50], seeds, noise_std=[0.01],
                    knot_kwargs=[{"mode": "cluster"}]),
        _method_row("bspline3", "clustered", "least_squares", "none", 0.0,
                    ["staccato", "bounce", "wobble"], [50], seeds, noise_std=[0.01],
                    knot_kwargs=[{"mode": "near_duplicate"}]),
        _method_row("bspline3", "clustered", "least_squares", "none", 0.0,
                    ["staccato", "bounce", "wobble"], [50], seeds, noise_std=[0.01],
                    knot_kwargs=[{"mode": "one_gap"}]),
        _method_row("hermite", "clustered", "least_squares", "none", 0.0,
                    ["staccato", "bounce", "wobble"], [50], seeds, noise_std=[0.01],
                    knot_kwargs=[{"mode": "cluster"}]),
        _method_row("pspline", "clustered", "least_squares", "diff", 1e-3,
                    ["staccato", "bounce", "wobble"], [50], seeds, noise_std=[0.01],
                    knot_kwargs=[{"mode": "cluster"}]),
    ]
    return expand(rows)


def iteration2_knotcount_conditions(quick=True):
    seeds = I2_QUICK_SEEDS if quick else I2_SEEDS
    methods = [
        ("pspline", "split_merge", "none", 0.0),
        ("pspline", "cv", "none", 0.0),
        ("pspline_gcv", "split_merge", "none", 0.0),
        ("bspline3", "split_merge", "none", 0.0),
        ("bspline3", "cv", "none", 0.0),
        ("bspline3", "split_merge", "diff", 1e-3),
    ]
    rows = [
        _method_row(rep, kn, "least_squares", rk, rl,
                    ["staccato", "double_step", "bounce", "wobble"], [20, 50], seeds,
                    noise_std=[0.01])
        for rep, kn, rk, rl in methods
    ]
    return expand(rows)


def iteration2_conditions(quick=True):
    return (
        iteration2_core_conditions(quick=quick)
        + iteration2_noise_conditions(quick=quick)
        + iteration2_timeparam_conditions(quick=quick)
        + iteration2_extrapolation_conditions(quick=quick)
        + iteration2_adversarial_conditions(quick=quick)
        + iteration2_knotcount_conditions(quick=quick)
    )


SUITES = {
    "smoke": smoke_conditions,
    "starter": starter_conditions,
    "placement": placement_conditions,
    "fitters": fitter_conditions,
    "sampling": sampling_conditions,
    "robustness": robustness_conditions,
    "full": full_conditions,
    "i2_core": iteration2_core_conditions,
    "i2_noise": iteration2_noise_conditions,
    "i2_timeparam": iteration2_timeparam_conditions,
    "i2_extrap": iteration2_extrapolation_conditions,
    "i2_adversarial": iteration2_adversarial_conditions,
    "i2_knotcount": iteration2_knotcount_conditions,
    "iteration2": iteration2_conditions,
}


def estimate(conditions):
    return len(conditions)


def build(name, quick=True):
    if name not in SUITES:
        raise KeyError(f"unknown suite '{name}', options: {sorted(SUITES)}")
    if name in ("smoke", "full"):
        return SUITES[name]()
    return SUITES[name](quick=quick)
