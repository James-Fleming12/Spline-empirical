import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import knots, metrics as metrics_mod, motions, representations, samplers, timeparam


@dataclass
class NoiseModel:
    """Observation noise model.

    The base model is i.i.d. Gaussian (``std``), plus uniform outliers,
    independent missing values and a constant per-axis bias.  The extra fields
    add the *structured* observation processes that real GS/mocap tracks have:
    temporally correlated (AR(1)) noise, speed-dependent heteroscedasticity,
    outliers proportional to speed, missing *bursts*, timestamp jitter,
    quantization and per-axis correlation.  When every extra field is zero the
    behaviour is bit-identical to the original i.i.d. model.
    """

    std: float = 0.0
    outlier_frac: float = 0.0
    outlier_scale: float = 6.0
    missing_frac: float = 0.0
    bias: float = 0.0
    ar1_rho: float = 0.0
    hetero: float = 0.0
    speed_outlier: float = 0.0
    missing_bursts: int = 0
    burst_width: float = 0.06
    jitter: float = 0.0
    quantize: float = 0.0
    axis_corr: float = 0.0
    bias_drift: float = 0.0
    grid_n: int = 2001

    @property
    def structured(self):
        return any(
            [
                self.ar1_rho > 0,
                self.hetero > 0,
                self.speed_outlier > 0,
                self.missing_bursts > 0,
                self.jitter > 0,
                self.quantize > 0,
                self.axis_corr > 0,
                self.bias_drift > 0,
            ]
        )

    # -- legacy i.i.d. path -------------------------------------------------
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

    # -- structured path ----------------------------------------------------
    def prepare(self, motion, seed):
        """Precompute deterministic continuous noise fields over [0, 1]."""
        if not self.structured:
            return None
        rng = np.random.default_rng([int(seed), 99173, 11])
        g = np.linspace(0.0, 1.0, int(self.grid_n))
        dim = int(motion.dim)
        fields = {"grid": g, "dim": dim}
        if self.axis_corr > 0 and dim > 1:
            r = float(np.clip(self.axis_corr, 0.0, 0.999))
            R = np.full((dim, dim), r)
            np.fill_diagonal(R, 1.0)
            R = R + 1e-8 * np.eye(dim)
            fields["L"] = np.linalg.cholesky(R)
        else:
            fields["L"] = np.eye(dim)
        if self.ar1_rho > 0:
            rho = float(np.clip(self.ar1_rho, 1e-6, 0.999999))
            tau = -1.0 / np.log(rho)
            d = float(g[1] - g[0])
            a = float(np.exp(-d / max(tau, 1e-12)))
            sr = float(np.sqrt(max(1.0 - a * a, 0.0)))
            w = rng.normal(size=(len(g), dim)) @ fields["L"].T
            eps = np.zeros((len(g), dim))
            for i in range(1, len(g)):
                eps[i] = a * eps[i - 1] + sr * w[i]
            eps = eps / (eps.std(axis=0, keepdims=True) + 1e-9)
            fields["ar1"] = eps
        if self.bias_drift > 0:
            from scipy.ndimage import uniform_filter1d

            raw = rng.normal(size=(len(g), dim))
            dr = uniform_filter1d(raw, size=max(3, len(g) // 20), axis=0, mode="nearest")
            dr = (dr - dr.mean(axis=0, keepdims=True)) / (dr.std(axis=0, keepdims=True) + 1e-9)
            fields["drift"] = dr
        if self.jitter > 0:
            from scipy.ndimage import uniform_filter1d

            j = uniform_filter1d(rng.normal(size=len(g)), size=max(3, len(g) // 50), mode="nearest")
            fields["jitter"] = j / (np.abs(j).max() + 1e-9)
        if self.missing_bursts > 0:
            bursts = []
            for _ in range(int(self.missing_bursts)):
                c = float(rng.uniform(0.05, 0.95))
                wdt = float(self.burst_width)
                bursts.append((c - 0.5 * wdt, c + 0.5 * wdt))
            fields["bursts"] = bursts
        if self.hetero > 0 or self.speed_outlier > 0:
            speed = np.linalg.norm(motion.eval_state(g, order=1)[..., 1], axis=1)
            fields["speed"] = speed
            fields["speed_rms"] = float(np.sqrt(np.mean(speed**2))) + 1e-9
            fields["speed_max"] = float(speed.max()) + 1e-9
        return fields

    def _field_at(self, fields, name, t, dim):
        return np.array([np.interp(t, fields["grid"], fields[name][:, d]) for d in range(dim)])

    def observe_structured(self, clean, t, fields, seed):
        dim = int(fields.get("dim", 3))
        y = np.asarray(clean, dtype=float).reshape(-1).copy()
        y = y + self.bias
        key = float(np.clip(t, 0.0, 1.0))
        rng = np.random.default_rng([int(seed), int(np.round(key * 1e9)) % (2**31), 7])
        if "ar1" in fields:
            base = self.std * self._field_at(fields, "ar1", key, dim)
        else:
            base = (fields["L"] @ rng.normal(size=dim)) * self.std
        if self.hetero > 0 and self.std > 0:
            sp = float(np.interp(key, fields["grid"], fields["speed"]))
            base = base * (1.0 + self.hetero * sp / fields["speed_rms"])
        if "drift" in fields:
            y = y + self.bias_drift * self._field_at(fields, "drift", key, dim)
        y = y + base
        if self.speed_outlier > 0:
            sp = float(np.interp(key, fields["grid"], fields["speed"]))
            p = float(np.clip(self.speed_outlier * sp / fields["speed_max"], 0.0, 1.0))
            if rng.random() < p:
                y = y + rng.normal(0.0, self.outlier_scale * (self.std or 0.05), size=dim)
        if self.outlier_frac > 0 and rng.random() < self.outlier_frac:
            y = y + rng.normal(0.0, self.outlier_scale * (self.std or 0.05), size=dim)
        if self.quantize > 0:
            y = np.round(y / self.quantize) * self.quantize
        if any(lo <= key <= hi for lo, hi in fields.get("bursts", [])):
            y = np.full(dim, np.nan)
        if self.missing_frac > 0 and rng.random() < self.missing_frac:
            y = np.full(dim, np.nan)
        return y


class ObservationOracle:
    def __init__(self, motion, noise, seed=0):
        self.motion = motion
        self.noise = noise
        self.seed = int(seed)
        self._cache = {}
        self._fields = noise.prepare(motion, seed) if hasattr(noise, "prepare") else None

    def _query_time(self, key):
        if self._fields is None or self.noise.jitter <= 0:
            return key
        shift = float(np.interp(key, self._fields["grid"], self._fields["jitter"])) * self.noise.jitter
        return float(np.clip(key + shift, 0.0, 1.0))

    def query(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float).ravel())
        out = np.empty((len(t), self.motion.dim), dtype=float)
        for i, ti in enumerate(t):
            key = round(float(np.clip(ti, 0.0, 1.0)), 9)
            if key not in self._cache:
                if self._fields is None:
                    rng = np.random.default_rng([self.seed, int(key * 1e9) % (2**31), 7])
                    clean = self.motion.eval(np.array([key]))
                    self._cache[key] = self.noise.observe(clean, rng)[0]
                else:
                    query_t = self._query_time(key)
                    clean = self.motion.eval(np.array([query_t]))[0]
                    self._cache[key] = self.noise.observe_structured(clean, key, self._fields, self.seed)
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
    split: str = "none"
    rep_kwargs: dict = field(default_factory=dict)
    fitter_kwargs: dict = field(default_factory=dict)
    knot_kwargs: dict = field(default_factory=dict)
    sampler_kwargs: dict = field(default_factory=dict)
    noise_kwargs: dict = field(default_factory=dict)
    eval_grid: int = 2001
    tags: tuple = ()

    def to_record(self):
        rec = asdict(self)
        rec["rep_kwargs"] = json.dumps(self.rep_kwargs, sort_keys=True)
        rec["fitter_kwargs"] = json.dumps(self.fitter_kwargs, sort_keys=True)
        rec["knot_kwargs"] = json.dumps(self.knot_kwargs, sort_keys=True)
        rec["sampler_kwargs"] = json.dumps(self.sampler_kwargs, sort_keys=True)
        rec["noise_kwargs"] = json.dumps(self.noise_kwargs, sort_keys=True)
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


def split_windows(split):
    """Return (train_windows, test_windows) for an extrapolation protocol.

    Both are lists of (lo, hi) intervals; ``test_windows`` is ``None`` for the
    standard interpolation protocol.  Training observations are remapped into
    the union of the training windows so samplers still work unchanged.
    """
    if split in (None, "none"):
        return [(0.0, 1.0)], None
    if split == "early":  # train early, extrapolate late
        return [(0.0, 0.5)], [(0.5, 1.0)]
    if split == "late":  # train late, extrapolate early
        return [(0.5, 1.0)], [(0.0, 0.5)]
    if split == "middle":  # train middle, extrapolate both ends
        return [(0.3, 0.7)], [(0.0, 0.3), (0.7, 1.0)]
    if split == "interp":  # disjoint train window, hold out a middle interval
        return [(0.0, 0.45), (0.55, 1.0)], [(0.45, 0.55)]
    raise ValueError(f"unknown split '{split}'")


def remap_to_windows(t, windows):
    """Map points from [0, 1] into the union of ``windows`` preserving order."""
    if windows == [(0.0, 1.0)]:
        return np.asarray(t, dtype=float)
    lengths = np.array([hi - lo for lo, hi in windows], dtype=float)
    total = lengths.sum()
    if total <= 0:
        return np.asarray(t, dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(lengths)]) / total
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    j = np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(windows) - 1)
    local = (t - cum[j]) / np.maximum(cum[j + 1] - cum[j], 1e-9)
    return np.array([windows[jj][0] + ll * lengths[jj] for jj, ll in zip(j, local)])


def residual_acf1(resid):
    r = np.asarray(resid, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    if len(r) < 3:
        return float("nan")
    vals = []
    for d in range(r.shape[1]):
        x = r[:, d] - np.mean(r[:, d])
        denom = float(np.sum(x**2))
        if denom <= 0:
            continue
        vals.append(float(np.sum(x[:-1] * x[1:]) / denom))
    return float(np.mean(vals)) if vals else float("nan")


def knot_spacing_stats(knots):
    k = np.sort(np.unique(np.asarray(knots, dtype=float).ravel()))
    if len(k) < 2:
        return {"knot_min_gap": 0.0, "knot_median_gap": 0.0, "knot_max_gap": 0.0,
                "knot_near_dupes": 0, "knot_endpoint_span": 0.0}
    d = np.diff(k)
    return {
        "knot_min_gap": float(d.min()),
        "knot_median_gap": float(np.median(d)),
        "knot_max_gap": float(d.max()),
        "knot_near_dupes": int((d < 1e-3).sum()),
        "knot_endpoint_span": float(k[-1] - k[0]),
    }


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
    for key, value in (cond.noise_kwargs or {}).items():
        if hasattr(noise, key):
            setattr(noise, key, value)
    oracle = ObservationOracle(motion, noise, cond.seed)
    sampler = samplers.build(cond.sampling, oracle=cond.sampling_oracle, **cond.sampler_kwargs)
    t_obs = sampler.sample(motion, cond.budget, np.random.default_rng(cond.seed), oracle)
    t_obs = np.unique(np.round(np.clip(t_obs, 0.0, 1.0), 9))
    train_windows, test_windows = split_windows(cond.split)
    if train_windows != [(0.0, 1.0)]:
        t_obs = np.unique(np.round(remap_to_windows(t_obs, train_windows), 9))
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
            "n_selected_sites": int(len(np.unique(np.asarray(positions, dtype=float).ravel()))),
            "n_queries": int(len(t_obs)),
            "n_missing": int(np.isnan(y_obs).any(axis=1).sum()),
            "sampling_uses_gt": bool(getattr(sampler, "uses_ground_truth", False)),
            "knots_uses_gt": bool(getattr(placer, "uses_ground_truth", False)),
            "rep_uses_gt": bool(getattr(rep, "uses_ground_truth", False)),
            "time_param_uses_gt": bool(getattr(tp, "uses_ground_truth", False)),
            "knot_positions": json.dumps(np.round(np.asarray(positions, dtype=float), 6).tolist()),
        }
    )
    # -- per-fit conditioning telemetry ------------------------------------
    try:
        rec.update(rep.diagnostics())
    except Exception:
        pass
    rec.update(knot_spacing_stats(positions))
    try:
        resid = y_obs - rep.eval(u_obs)
        rec["resid_acf1"] = residual_acf1(resid)
    except Exception:
        rec["resid_acf1"] = float("nan")
    # -- extrapolation / interpolation hold-out metrics ---------------------
    if test_windows is not None:
        try:
            rec.update(metrics_mod.window_metrics(motion, rep, tp, test_windows, prefix="holdout_"))
            rec.update(metrics_mod.window_metrics(motion, rep, tp, train_windows, prefix="trainwin_"))
        except Exception:
            pass
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
