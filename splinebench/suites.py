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


SUITES = {
    "smoke": smoke_conditions,
    "starter": starter_conditions,
    "placement": placement_conditions,
    "fitters": fitter_conditions,
    "sampling": sampling_conditions,
    "robustness": robustness_conditions,
    "full": full_conditions,
}


def estimate(conditions):
    return len(conditions)


def build(name, quick=True):
    if name not in SUITES:
        raise KeyError(f"unknown suite '{name}', options: {sorted(SUITES)}")
    if name in ("smoke", "full"):
        return SUITES[name]()
    return SUITES[name](quick=quick)
