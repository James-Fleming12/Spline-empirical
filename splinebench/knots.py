from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .fitters import bayes_opt_minimize
from .utils import savgol_deriv, weighted_quantile_positions


@dataclass
class KnotContext:
    u: np.ndarray
    y: np.ndarray
    n: int
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    motion: object = None
    oracle: bool = False
    rep_factory: Callable | None = None


def _derivs(ctx, order):
    if ctx.oracle and ctx.motion is not None:
        return ctx.motion.eval_state(ctx.u, order=order)[..., order]
    return savgol_deriv(ctx.u, ctx.y, order=order)


def _dedupe_pad(pos, n, lo=0.0, hi=1.0):
    pos = np.unique(np.round(np.clip(np.asarray(pos, dtype=float).ravel(), lo, hi), 9))
    if len(pos) == 0:
        pos = np.array([lo, hi])
    if len(pos) > n:
        idx = np.linspace(0, len(pos) - 1, n).round().astype(int)
        pos = pos[np.unique(idx)]
    while len(pos) < n:
        gaps = np.diff(pos)
        if len(gaps) == 0:
            pos = np.array([lo, hi])
            continue
        i = int(np.argmax(gaps))
        if gaps[i] <= 1e-9:
            break
        pos = np.insert(pos, i + 1, 0.5 * (pos[i] + pos[i + 1]))
    return pos


def _linear_sse(positions, ctx):
    from .representations import PiecewiseLinear

    rep = PiecewiseLinear(positions, dim=ctx.y.shape[1])
    rep.fit(ctx.u, ctx.y)
    return float(np.sum((rep.eval(ctx.u) - ctx.y) ** 2))


def _fit_probe(ctx, positions=None):
    if ctx.rep_factory is not None:
        return ctx.rep_factory(positions)
    from .representations import PiecewiseLinear

    if positions is None:
        positions = np.linspace(0.0, 1.0, max(4, ctx.n // 2))
    rep = PiecewiseLinear(positions, dim=ctx.y.shape[1])
    rep.fit(ctx.u, ctx.y)
    return rep


def _score(ctx, positions):
    if ctx.rep_factory is not None:
        rep = ctx.rep_factory(positions)
        return float(np.sum((rep.eval(ctx.u) - ctx.y) ** 2))
    return _linear_sse(positions, ctx)


class KnotPlacer:
    name = "uniform"
    uses_ground_truth = False
    citation = ""

    def place(self, ctx):
        raise NotImplementedError


class UniformKnots(KnotPlacer):
    name = "uniform"
    citation = "deboor1978"

    def place(self, ctx):
        return np.linspace(0.0, 1.0, max(ctx.n, 2))


class QuantileKnots(KnotPlacer):
    name = "quantile"
    citation = ""

    def place(self, ctx):
        n = max(ctx.n, 2)
        if len(ctx.u) >= n:
            pos = np.quantile(ctx.u, np.linspace(0.0, 1.0, n))
        else:
            pos = np.linspace(0.0, 1.0, n)
        return _dedupe_pad(pos, n)


class ChordKnots(KnotPlacer):
    name = "chord"
    citation = ""

    def _place_feature(self, ctx, feature):
        order = np.argsort(ctx.u)
        u = ctx.u[order]
        y = ctx.y[order]
        step = np.linalg.norm(np.diff(y, axis=0), axis=1)
        if feature == "centripetal":
            step = np.sqrt(np.maximum(step, 0.0))
        cum = np.concatenate([[0.0], np.cumsum(step)])
        if cum[-1] <= 0:
            return np.linspace(0.0, 1.0, max(ctx.n, 2))
        cn = (cum - cum[0]) / (cum[-1] - cum[0])
        targets = np.linspace(0.0, 1.0, max(ctx.n, 2))
        pos = np.interp(targets, cn, u)
        return _dedupe_pad(pos, max(ctx.n, 2))


class ChordLengthKnots(ChordKnots):
    name = "chord"

    def place(self, ctx):
        return self._place_feature(ctx, "chord")


class CentripetalKnots(ChordKnots):
    name = "centripetal"

    def place(self, ctx):
        return self._place_feature(ctx, "centripetal")


class CurvatureKnots(KnotPlacer):
    name = "curvature"
    uses_ground_truth = False
    citation = ""

    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)

    def place(self, ctx):
        acc = np.abs(_derivs(ctx, 2)).sum(axis=1)
        w = (acc + 1e-9) ** self.alpha
        pos = weighted_quantile_positions(ctx.u, w, max(ctx.n, 2))
        return _dedupe_pad(pos, max(ctx.n, 2))


class FeaturePeakKnots(KnotPlacer):
    name = "feature_peaks"
    citation = ""

    def __init__(self, oracle=False):
        self.oracle = bool(oracle)
        self.uses_ground_truth = self.oracle

    def place(self, ctx):
        n = max(ctx.n, 2)
        jerk = np.abs(_derivs(ctx, 3)).sum(axis=1)
        acc = np.abs(_derivs(ctx, 2)).sum(axis=1)
        peaks = []
        for signal in (jerk, acc):
            if len(signal) >= 3:
                local = (signal[1:-1] >= signal[:-2]) & (signal[1:-1] > signal[2:])
                idx = np.flatnonzero(local) + 1
                peaks.extend((float(signal[i]), float(ctx.u[i])) for i in idx)
        reversals = np.flatnonzero(np.diff(np.sign(_derivs(ctx, 3).sum(axis=1))) != 0)
        peaks.extend((0.0, float(ctx.u[i])) for i in reversals)
        peaks.sort(key=lambda p: -p[0])
        chosen = [0.0, 1.0]
        for _, t in peaks:
            if len(chosen) >= n:
                break
            if all(abs(t - c) > 1e-3 for c in chosen):
                chosen.append(t)
        if len(chosen) < n:
            fill = weighted_quantile_positions(
                ctx.u, (jerk + 1e-9), n - len(chosen), lo=0.0, hi=1.0
            )
            chosen.extend(fill.tolist())
        return _dedupe_pad(chosen, n)


class RDPKnots(KnotPlacer):
    name = "rdp"
    citation = "ramer1972"

    def _simplify(self, u, y, tol):
        keep = np.zeros(len(u), dtype=bool)
        keep[0] = keep[-1] = True
        stack = [(0, len(u) - 1)]
        while stack:
            i0, i1 = stack.pop()
            if i1 <= i0 + 1:
                continue
            span = u[i1] - u[i0]
            if span <= 0:
                dist = np.linalg.norm(y[i0 + 1 : i1] - y[i0], axis=1)
            else:
                w = (u[i0 + 1 : i1] - u[i0]) / span
                proj = y[i0][None, :] + w[:, None] * (y[i1] - y[i0])[None, :]
                dist = np.linalg.norm(y[i0 + 1 : i1] - proj, axis=1)
            idx = int(np.argmax(dist)) + i0 + 1
            if dist.max() > tol:
                keep[idx] = True
                stack.extend([(i0, idx), (idx, i1)])
        return keep

    def place(self, ctx):
        n = max(ctx.n, 2)
        order = np.argsort(ctx.u)
        u = ctx.u[order]
        y = ctx.y[order]
        lo, hi = 0.0, float(np.linalg.norm(y[-1] - y[0]) + 1e-6)
        for _ in range(40):
            tol = 0.5 * (lo + hi)
            keep = self._simplify(u, y, tol)
            if keep.sum() > n:
                lo = tol
            else:
                hi = tol
        pos = u[self._simplify(u, y, hi)]
        if len(pos) < n:
            pos = np.linspace(0.0, 1.0, n)
        return _dedupe_pad(pos, n)


class SplitMergeKnots(KnotPlacer):
    name = "split_merge"
    citation = ""

    def place(self, ctx):
        n = max(ctx.n, 2)
        order = np.argsort(ctx.u)
        u = ctx.u[order]
        y = ctx.y[order]
        segments = [(0, len(u) - 1)]
        while len(segments) + 1 < n:
            best = None
            for i0, i1 in segments:
                if i1 <= i0 + 1:
                    continue
                span = max(u[i1] - u[i0], 1e-12)
                w = (u[i0 + 1 : i1] - u[i0]) / span
                proj = y[i0][None, :] + w[:, None] * (y[i1] - y[i0])[None, :]
                dist = np.linalg.norm(y[i0 + 1 : i1] - proj, axis=1)
                idx = int(np.argmax(dist)) + i0 + 1
                score = float(np.sum(dist**2))
                if best is None or score > best[0]:
                    best = (score, idx, (i0, i1))
            if best is None or best[0] <= 1e-12:
                break
            _, idx, (i0, i1) = best
            segments.remove((i0, i1))
            segments.extend([(i0, idx), (idx, i1)])
        chosen = sorted({0.0, 1.0} | {float(u[i]) for i, _ in segments} | {float(u[j]) for _, j in segments})
        return _dedupe_pad(chosen, n)


class GreedyKnots(KnotPlacer):
    name = "greedy"
    citation = "tropp2007"

    def __init__(self, candidates=None):
        self.candidates = candidates

    def place(self, ctx):
        n = max(ctx.n, 2)
        m = int(self.candidates or max(4 * n, 60))
        cand = np.unique(np.concatenate([np.linspace(0.0, 1.0, m), ctx.u]))
        cand = np.clip(cand, 0.0, 1.0)
        pos = [0.0, 1.0]
        while len(pos) < n:
            best = (np.inf, None)
            for c in cand:
                if min(abs(c - p) for p in pos) < 1e-4:
                    continue
                score = _score(ctx, np.sort(np.append(pos, c)))
                if score < best[0]:
                    best = (score, float(c))
            if best[1] is None:
                break
            pos = sorted(pos + [best[1]])
        return _dedupe_pad(pos, n)


class KMeansKnots(KnotPlacer):
    name = "kmeans"
    citation = ""

    def __init__(self, alpha=1.0, iters=25):
        self.alpha = float(alpha)
        self.iters = int(iters)

    def place(self, ctx):
        n = max(ctx.n, 2)
        acc = np.abs(_derivs(ctx, 2)).sum(axis=1)
        w = (acc + 1e-9) ** self.alpha
        centers = weighted_quantile_positions(ctx.u, w, n)
        for _ in range(self.iters):
            d = np.abs(ctx.u[:, None] - centers[None, :])
            assign = np.argmin(d, axis=1)
            new = centers.copy()
            for k in range(len(centers)):
                m = assign == k
                if m.any() and w[m].sum() > 0:
                    new[k] = np.average(ctx.u[m], weights=w[m])
            new = np.sort(new)
            if np.linalg.norm(new - centers) < 1e-9:
                centers = new
                break
            centers = new
        return _dedupe_pad(centers, n)


class FarthestPointKnots(KnotPlacer):
    name = "farthest_point"
    citation = "elden1997"

    def __init__(self, beta=0.25, oracle=False):
        self.beta = float(beta)
        self.oracle = bool(oracle)
        self.uses_ground_truth = self.oracle

    def place(self, ctx):
        n = max(ctx.n, 2)
        acc = np.abs(_derivs(ctx, 2)).sum(axis=1)
        jerk = np.abs(_derivs(ctx, 3)).sum(axis=1)
        a_max = max(acc.max(), 1e-9)
        j_max = max(jerk.max(), 1e-9)
        z = np.stack([ctx.u, self.beta * acc / a_max, self.beta * jerk / j_max], axis=1)
        chosen = [int(np.argmin(ctx.u))]
        d = np.linalg.norm(z - z[chosen[0]], axis=1)
        while len(chosen) < n:
            i = int(np.argmax(d))
            if d[i] <= 1e-12:
                break
            chosen.append(i)
            d = np.minimum(d, np.linalg.norm(z - z[i], axis=1))
        pos = np.sort(ctx.u[chosen])
        if len(pos) < n:
            pos = _dedupe_pad(pos, n)
        return _dedupe_pad(pos, n)


class BayesOptKnots(KnotPlacer):
    name = "bayesopt"
    citation = "snoek2012"

    def __init__(self, n_init=10, n_iter=25, oracle=False):
        self.n_init = int(n_init)
        self.n_iter = int(n_iter)
        self.oracle = bool(oracle)
        self.uses_ground_truth = self.oracle

    def place(self, ctx):
        n = max(ctx.n, 2)
        if n <= 2:
            return np.array([0.0, 1.0])
        free = n - 2

        def objective(x):
            pos = np.sort(np.concatenate([[0.0], np.clip(x, 0.0, 1.0), [1.0]]))
            return _score(ctx, pos)

        result = bayes_opt_minimize(
            objective,
            bounds=[(0.0, 1.0)] * free,
            n_init=self.n_init,
            n_iter=self.n_iter,
            seed=int(ctx.rng.integers(0, 2**31 - 1)),
            candidates=512,
        )
        pos = np.sort(np.concatenate([[0.0], np.clip(result["x"], 0.0, 1.0), [1.0]]))
        return _dedupe_pad(pos, n)


class ActiveResidualKnots(KnotPlacer):
    name = "active_residual"
    citation = "cohn1996"

    def __init__(self, growth=4, n_candidates=256):
        self.growth = int(growth)
        self.n_candidates = int(n_candidates)

    def place(self, ctx):
        n = max(ctx.n, 2)
        pos = list(np.linspace(0.0, 1.0, min(self.growth, n)))
        if pos[-1] < 1.0:
            pos.append(1.0)
        grid = np.linspace(0.0, 1.0, self.n_candidates)
        while len(pos) < n:
            rep = _fit_probe(ctx, np.sort(np.array(pos)))
            pred = rep.eval(grid)
            resid = np.linalg.norm(pred - _interp_target(ctx, grid), axis=1)
            for p in pos:
                resid[np.abs(grid - p) < 1e-3] = -np.inf
            idx = int(np.argmax(resid))
            if not np.isfinite(resid[idx]):
                break
            pos.append(float(grid[idx]))
        return _dedupe_pad(pos, n)


def _interp_target(ctx, grid):
    order = np.argsort(ctx.u)
    u = ctx.u[order]
    y = ctx.y[order]
    return np.stack([np.interp(grid, u, y[:, d]) for d in range(y.shape[1])], axis=1)


KNOT_PLACERS = {
    "uniform": UniformKnots,
    "quantile": QuantileKnots,
    "chord": ChordLengthKnots,
    "centripetal": CentripetalKnots,
    "curvature": CurvatureKnots,
    "feature_peaks": FeaturePeakKnots,
    "rdp": RDPKnots,
    "split_merge": SplitMergeKnots,
    "greedy": GreedyKnots,
    "kmeans": KMeansKnots,
    "farthest_point": FarthestPointKnots,
    "bayesopt": BayesOptKnots,
    "active_residual": ActiveResidualKnots,
}

PLANNED_KNOTS = [
    "foley", "universal", "corner_detection", "zero_crossings", "velocity_peaks",
    "acceleration_peaks", "jerk_peaks", "impact_events", "douglas_peucker",
    "orthogonal_matching_pursuit", "lars", "genetic", "pso", "simulated_annealing",
    "cmaes_knots", "gmm_placement", "dynamic_programming", "dtw", "pca",
    "autoencoder", "incremental_insertion", "multiresolution", "cross_validation",
    "aic_bic_mdl", "expected_error_reduction", "d_optimal", "sobol_knots",
    "poisson_disk", "adversarial",
]


def list_knot_placers():
    return sorted(KNOT_PLACERS)


def build(name, **kw):
    if name not in KNOT_PLACERS:
        raise KeyError(f"unknown knot placer '{name}', options: {list_knot_placers()}")
    from .utils import filter_kwargs

    cls = KNOT_PLACERS[name]
    return cls(**filter_kwargs(cls.__init__, kw))
