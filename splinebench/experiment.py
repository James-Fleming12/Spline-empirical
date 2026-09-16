import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import knots, metrics as metrics_mod, motions, representations, samplers, timeparam


@dataclass
class NoiseModel:
    std: float = 0.0
    outlier_frac: float = 0.0
    outlier_scale: float = 6.0
    missing_frac: float = 0.0
    bias: float = 0.0

    def observe(self, clean, rng):
        y = np.asarray(clean, dtype=float).copy()
        y = y + self.bias
        if self.std > 0:
            y = y + rng.normal(0.0, self.std, size=y.shape)
        base = self.std if self.std > 0 else 0.05
        if self.outlier_frac > 0 and len(y) > 0:
            mask = rng.random(len(y)) < self.outlier_frac
            if mask.any():
                y[mask] = y[mask] + rng.normal(0.0, base * self.outlier_scale, size=(int(mask.sum()), y.shape[1]))
        if self.missing_frac > 0 and len(y) > 0:
            mask = rng.random(len(y)) < self.missing_frac
            y[mask] = np.nan
        return y


class ObservationOracle:
    def __init__(self, motion, noise, seed=0):
        self.motion = motion
        self.noise = noise
        self.seed = int(seed)
        self._cache = {}

    def query(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float).ravel())
        out = np.empty((len(t), self.motion.dim), dtype=float)
        for i, ti in enumerate(t):
            key = round(float(np.clip(ti, 0.0, 1.0)), 9)
            if key not in self._cache:
                rng = np.random.default_rng([self.seed, int(key * 1e9) % (2**31), 7])
                clean = self.motion.eval(np.array([key]))
                self._cache[key] = self.noise.observe(clean, rng)[0]
            out[i] = self._cache[key]
        return out


@dataclass
class Condition:
    motion: str = "staccato"
    representation: str = "bspline3"
    knots: str = "uniform"
    fitter: str = "least_squares"
    time_param: str = "linear"
    sampling: str = "random"
    budget: int = 20
    seed: int = 0
    n_sites: int | None = None
    reg_kind: str = "none"
    reg_lam: float = 0.0
    reg_order: int = 2
    noise_std: float = 0.01
    outlier_frac: float = 0.0
    outlier_scale: float = 6.0
    missing_frac: float = 0.0
    bias: float = 0.0
    knot_oracle: bool = False
    sampling_oracle: bool = False
    time_param_oracle: bool = False
    rep_kwargs: dict = field(default_factory=dict)
    fitter_kwargs: dict = field(default_factory=dict)
    knot_kwargs: dict = field(default_factory=dict)
    sampler_kwargs: dict = field(default_factory=dict)
    eval_grid: int = 2001
    tags: tuple = ()

    def to_record(self):
        rec = asdict(self)
        rec["rep_kwargs"] = json.dumps(self.rep_kwargs, sort_keys=True)
        rec["fitter_kwargs"] = json.dumps(self.fitter_kwargs, sort_keys=True)
        rec["knot_kwargs"] = json.dumps(self.knot_kwargs, sort_keys=True)
        rec["sampler_kwargs"] = json.dumps(self.sampler_kwargs, sort_keys=True)
        rec["tags"] = "|".join(self.tags)
        return rec

    @property
    def key(self):
        rec = self.to_record()
        rec.pop("tags", None)
        return json.dumps(rec, sort_keys=True)


def default_n_sites(budget):
    return int(max(2, min(16, round(budget / 3))))


def reg_dict(cond):
    if cond.reg_kind in (None, "none") or cond.reg_lam <= 0:
        return None
    return {"kind": cond.reg_kind, "lam": float(cond.reg_lam), "order": int(cond.reg_order)}


def run_condition(cond):
    t_start = time.perf_counter()
    motion = motions.build(cond.motion)
    noise = NoiseModel(
        std=cond.noise_std,
        outlier_frac=cond.outlier_frac,
        outlier_scale=cond.outlier_scale,
        missing_frac=cond.missing_frac,
        bias=cond.bias,
    )
    oracle = ObservationOracle(motion, noise, cond.seed)
    sampler = samplers.build(cond.sampling, oracle=cond.sampling_oracle, **cond.sampler_kwargs)
    t_obs = sampler.sample(motion, cond.budget, np.random.default_rng(cond.seed), oracle)
    t_obs = np.unique(np.round(np.clip(t_obs, 0.0, 1.0), 9))
    y_obs = oracle.query(t_obs)
    n_sites = int(cond.n_sites or default_n_sites(cond.budget))

    tp = timeparam.build(cond.time_param, oracle=cond.time_param_oracle)
    tp.fit(t_obs, y_obs, motion=motion if cond.time_param_oracle else None)
    u_obs = tp.to_u(t_obs)[:, 0]
    reg = reg_dict(cond)
    weight = np.isfinite(y_obs).all(axis=1).astype(float)

    def fit_rep(positions):
        rep = representations.build(
            cond.representation, positions=positions, dim=motion.dim, seed=cond.seed, **cond.rep_kwargs
        )
        rep.fit(u_obs, y_obs, weight=weight, fitter=cond.fitter, reg=reg, **cond.fitter_kwargs)
        return rep

    placer = knots.build(cond.knots, oracle=cond.knot_oracle, **cond.knot_kwargs)
    ctx = knots.KnotContext(
        u=u_obs,
        y=np.nan_to_num(y_obs, nan=0.0),
        n=n_sites,
        rng=np.random.default_rng(cond.seed + 101),
        motion=motion,
        oracle=cond.knot_oracle,
        rep_factory=fit_rep,
    )
    positions = placer.place(ctx)

    t_fit = time.perf_counter()
    rep = fit_rep(positions)
    fit_time = time.perf_counter() - t_fit

    t_eval = time.perf_counter()
    grid = np.linspace(motion.window[0], motion.window[1], cond.eval_grid)
    rec = metrics_mod.evaluate(
        motion,
        rep,
        tp,
        grid=grid,
        fit_time_s=fit_time,
        eval_time_s=None,
        u_train=u_obs,
        y_train=y_obs,
    )
    rec["eval_time_s"] = time.perf_counter() - t_eval
    rec["wall_time_s"] = time.perf_counter() - t_start
    rec.update(cond.to_record())
    rec.update(
        {
            "n_sites": n_sites,
            "n_queries": int(len(t_obs)),
            "n_missing": int(np.isnan(y_obs).any(axis=1).sum()),
            "sampling_uses_gt": bool(getattr(sampler, "uses_ground_truth", False)),
            "knots_uses_gt": bool(getattr(placer, "uses_ground_truth", False)),
            "rep_uses_gt": bool(getattr(rep, "uses_ground_truth", False)),
            "time_param_uses_gt": bool(getattr(tp, "uses_ground_truth", False)),
            "knot_positions": json.dumps(np.round(np.asarray(positions, dtype=float), 6).tolist()),
        }
    )
    return rec


def run_suite(conditions, out_path=None, resume=True, limit=None, verbose=True):
    existing = set()
    if out_path is not None and resume:
        try:
            with open(out_path, "r") as fh:
                for line in fh:
                    try:
                        existing.add(json.loads(line)["_key"])
                    except (KeyError, json.JSONDecodeError):
                        continue
        except FileNotFoundError:
            pass
    records = []
    handle = open(out_path, "a") if out_path is not None else None
    try:
        for i, cond in enumerate(conditions):
            if limit is not None and len(records) >= limit:
                break
            if cond.key in existing:
                continue
            rec = run_condition(cond)
            rec["_key"] = cond.key
            records.append(rec)
            if handle is not None:
                handle.write(json.dumps(rec) + "\n")
                handle.flush()
            if verbose:
                print(
                    f"[{i + 1}/{len(conditions)}] {cond.motion:>12} {cond.representation:>12} "
                    f"{cond.knots:>14} B={cond.budget:>4} seed={cond.seed} "
                    f"pos_rmse={rec.get('pos_rmse', float('nan')):.5f} "
                    f"wall={rec['wall_time_s']:.2f}s"
                )
    finally:
        if handle is not None:
            handle.close()
    return records


def load_results(path):
    import pandas as pd

    rows = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def summarize(df, out_csv=None):
    agg = metrics_mod.aggregate(df)
    curves = metrics_mod.aulc(df)
    merged = agg.merge(curves, on=[c for c in curves.columns if c in agg.columns], how="left")
    if out_csv is not None:
        merged.to_csv(out_csv, index=False)
    return merged
