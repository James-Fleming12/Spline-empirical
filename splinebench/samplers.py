import numpy as np

from .representations import GaussianProcess, PiecewiseLinear
from .utils import savgol_deriv


def _finalize(t, budget, rng):
    t = np.unique(np.round(np.clip(np.asarray(t, dtype=float).ravel(), 0.0, 1.0), 9))
    if len(t) > budget:
        idx = np.linspace(0, len(t) - 1, budget).round().astype(int)
        t = t[np.unique(idx)]
    while len(t) < budget:
        if len(t) == 0:
            t = np.array([0.0])
            continue
        gaps = np.diff(t)
        if len(gaps) == 0:
            t = np.union1d(t, [0.0, 1.0])
            if len(t) == 1:
                return t
            continue
        i = int(np.argmax(gaps))
        if gaps[i] <= 1e-9:
            break
        t = np.insert(t, i + 1, 0.5 * (t[i] + t[i + 1]))
    return t[:budget]


def _grid(n):
    return np.linspace(0.0, 1.0, n)


def _density_from_derivs(motion, order, alpha, grid_n=2001):
    grid = np.linspace(0.0, 1.0, grid_n)
    d = np.abs(motion.eval_state(grid, order=order)[..., order]).sum(axis=1)
    return grid, (d + 1e-9) ** alpha


def _pilot_density(t_pilot, y_pilot, order, alpha, grid_n=2001):
    grid = np.linspace(0.0, 1.0, grid_n)
    if len(t_pilot) < 4:
        return grid, np.ones(grid_n)
    d = np.abs(savgol_deriv(t_pilot, y_pilot, order=order)).sum(axis=1)
    dens = np.interp(grid, t_pilot, d)
    return grid, (dens + 1e-9) ** alpha


def _sample_density(grid, density, n, rng):
    cdf = np.cumsum(density)
    cdf = cdf / cdf[-1]
    return np.interp(rng.uniform(0.0, 1.0, n), cdf, grid)


class Sampler:
    name = "random"
    adaptive = False
    uses_ground_truth = False
    citation = ""

    def sample(self, motion, budget, rng, oracle, **kw):
        raise NotImplementedError


class RandomSampler(Sampler):
    name = "random"

    def sample(self, motion, budget, rng, oracle, **kw):
        return _finalize(rng.uniform(0.0, 1.0, budget), budget, rng)


class GridSampler(Sampler):
    name = "grid"

    def sample(self, motion, budget, rng, oracle, **kw):
        return _finalize(_grid(budget), budget, rng)


class StratifiedSampler(Sampler):
    name = "stratified"
    citation = "mckay1979"

    def sample(self, motion, budget, rng, oracle, **kw):
        edges = np.linspace(0.0, 1.0, budget + 1)
        t = rng.uniform(edges[:-1], edges[1:])
        return _finalize(t, budget, rng)


class LatinHypercubeSampler(Sampler):
    name = "lhs"
    citation = "mckay1979"

    def sample(self, motion, budget, rng, oracle, **kw):
        try:
            from scipy.stats.qmc import LatinHypercube

            t = LatinHypercube(d=1, seed=rng).random(budget).ravel()
        except ImportError:
            t = (np.arange(budget) + rng.uniform(size=budget)) / budget
        return _finalize(t, budget, rng)


class SobolSampler(Sampler):
    name = "sobol"
    citation = "sobol1967"

    def sample(self, motion, budget, rng, oracle, **kw):
        from scipy.stats.qmc import Sobol

        m = int(np.ceil(np.log2(max(budget, 2))))
        pts = Sobol(d=1, seed=int(rng.integers(0, 2**31 - 1))).random_base2(m).ravel()
        return _finalize(pts, budget, rng)


class HaltonSampler(Sampler):
    name = "halton"

    def sample(self, motion, budget, rng, oracle, **kw):
        from scipy.stats.qmc import Halton

        pts = Halton(d=1, seed=int(rng.integers(0, 2**31 - 1))).random(budget).ravel()
        return _finalize(pts, budget, rng)


class FarthestPointSampler(Sampler):
    name = "farthest_point"
    citation = "elden1997"

    def __init__(self, oracle=False, beta=0.25, pilot_frac=0.2):
        self.oracle = bool(oracle)
        self.uses_ground_truth = self.oracle
        self.beta = float(beta)
        self.pilot_frac = float(pilot_frac)

    def sample(self, motion, budget, rng, oracle, **kw):
        n_pilot = max(3, int(round(self.pilot_frac * budget))) if not self.oracle else 0
        t = list(rng.uniform(0.0, 1.0, n_pilot))
        if self.oracle:
            grid, dens = _density_from_derivs(motion, 3, 1.0)
            z = np.stack([grid, self.beta * dens / max(dens.max(), 1e-9)], axis=1)
        else:
            t_pilot = np.sort(np.array(t))
            y_pilot = oracle.query(t_pilot)
            d = np.abs(savgol_deriv(t_pilot, y_pilot, order=3)).sum(axis=1) if len(t_pilot) >= 4 else np.ones(len(t_pilot))
            grid = np.linspace(0.0, 1.0, 2001)
            dd = np.interp(grid, t_pilot, d) if len(t_pilot) >= 4 else np.ones_like(grid)
            z = np.stack([grid, self.beta * dd / max(dd.max(), 1e-9)], axis=1)
            t = list(t_pilot)
        chosen = []
        first = int(np.argmin(grid))
        chosen.append(first)
        t.append(float(grid[first]))
        d = np.linalg.norm(z - z[chosen[0]], axis=1)
        while len(t) < budget:
            i = int(np.argmax(d))
            if d[i] <= 1e-12:
                break
            chosen.append(i)
            t.append(float(grid[i]))
            d = np.minimum(d, np.linalg.norm(z - z[i], axis=1))
        return _finalize(t, budget, rng)


class JerkImportanceSampler(Sampler):
    name = "jerk_importance"
    citation = "cohn1996"

    def __init__(self, oracle=False, alpha=1.0, order=3, pilot_frac=0.2):
        self.oracle = bool(oracle)
        self.uses_ground_truth = self.oracle
        self.alpha = float(alpha)
        self.order = int(order)
        self.pilot_frac = float(pilot_frac)

    def sample(self, motion, budget, rng, oracle, **kw):
        if self.oracle:
            grid, dens = _density_from_derivs(motion, self.order, self.alpha)
            return _finalize(_sample_density(grid, dens, budget, rng), budget, rng)
        n_pilot = max(4, int(round(self.pilot_frac * budget)))
        t_pilot = np.sort(rng.uniform(0.0, 1.0, n_pilot))
        y_pilot = oracle.query(t_pilot)
        grid, dens = _pilot_density(t_pilot, y_pilot, self.order, self.alpha)
        extra = _sample_density(grid, dens, budget - n_pilot, rng)
        return _finalize(np.concatenate([t_pilot, extra]), budget, rng)


class DerivativePeakSampler(Sampler):
    name = "derivative_peaks"
    citation = ""

    def __init__(self, oracle=False, order=3, jitter=0.01, pilot_frac=0.2):
        self.oracle = bool(oracle)
        self.uses_ground_truth = self.oracle
        self.order = int(order)
        self.jitter = float(jitter)
        self.pilot_frac = float(pilot_frac)

    def sample(self, motion, budget, rng, oracle, **kw):
        if self.oracle:
            grid, dens = _density_from_derivs(motion, self.order, 1.0)
            signal = dens
            pilot = np.array([])
        else:
            n_pilot = max(4, int(round(self.pilot_frac * budget)))
            pilot = np.sort(rng.uniform(0.0, 1.0, n_pilot))
            y_pilot = oracle.query(pilot)
            grid, dens = _pilot_density(pilot, y_pilot, self.order, 1.0)
            signal = dens
        local = (signal[1:-1] >= signal[:-2]) & (signal[1:-1] > signal[2:])
        idx = np.flatnonzero(local) + 1
        order = idx[np.argsort(-signal[idx])]
        peaks = []
        for i in order:
            if len(peaks) >= budget:
                break
            peaks.append(float(grid[i]) + rng.uniform(-self.jitter, self.jitter))
        fill = _sample_density(grid, signal, budget - len(peaks), rng)
        return _finalize(np.concatenate([pilot, np.array(peaks), fill]), budget, rng)


class HybridSampler(Sampler):
    name = "hybrid"
    citation = ""

    def __init__(self, oracle=False, ratio=0.5, order=3):
        self.oracle = bool(oracle)
        self.uses_ground_truth = self.oracle
        self.ratio = float(ratio)
        self.order = int(order)

    def sample(self, motion, budget, rng, oracle, **kw):
        n_uniform = int(round(self.ratio * budget))
        t_uniform = _grid(max(n_uniform, 2))
        if self.oracle:
            grid, dens = _density_from_derivs(motion, self.order, 1.0)
        else:
            n_pilot = max(4, budget // 5)
            pilot = np.sort(rng.uniform(0.0, 1.0, n_pilot))
            y_pilot = oracle.query(pilot)
            grid, dens = _pilot_density(pilot, y_pilot, self.order, 1.0)
            t_uniform = np.concatenate([pilot, t_uniform])
        extra = _sample_density(grid, dens, budget - len(t_uniform), rng)
        return _finalize(np.concatenate([t_uniform, extra]), budget, rng)


class ActiveGPSampler(Sampler):
    name = "active_gp"
    adaptive = True
    citation = "cohn1996"

    def __init__(self, init_frac=0.2, n_candidates=512):
        self.init_frac = float(init_frac)
        self.n_candidates = int(n_candidates)

    def sample(self, motion, budget, rng, oracle, **kw):
        n0 = max(4, int(round(self.init_frac * budget)))
        t = list(rng.uniform(0.0, 1.0, min(n0, budget)))
        grid = np.linspace(0.0, 1.0, self.n_candidates)
        while len(t) < budget:
            tt = np.sort(np.array(t))
            yy = oracle.query(tt)
            gp = GaussianProcess(dim=motion.dim, optimize=len(tt) >= 4)
            gp.fit(tt, yy)
            sigma = gp.posterior_std(grid)
            for q in t:
                sigma[np.abs(grid - q) < 1e-4] = -np.inf
            idx = int(np.argmax(sigma))
            if not np.isfinite(sigma[idx]):
                break
            t.append(float(grid[idx]))
        return _finalize(t, budget, rng)


class ActiveResidualSampler(Sampler):
    name = "active_residual"
    adaptive = True
    citation = ""

    def __init__(self, init_frac=0.25, n_candidates=512):
        self.init_frac = float(init_frac)
        self.n_candidates = int(n_candidates)

    def sample(self, motion, budget, rng, oracle, **kw):
        n0 = max(4, int(round(self.init_frac * budget)))
        t = list(rng.uniform(0.0, 1.0, min(n0, budget)))
        grid = np.linspace(0.0, 1.0, self.n_candidates)
        while len(t) < budget:
            tt = np.sort(np.array(t))
            yy = oracle.query(tt)
            knots = np.quantile(tt, np.linspace(0.0, 1.0, max(3, len(tt) // 3)))
            probe = PiecewiseLinear(knots, dim=motion.dim)
            probe.fit(tt, yy)
            resid_points = np.linalg.norm(probe.eval(tt) - yy, axis=1)
            resid = np.interp(grid, tt, resid_points)
            for q in t:
                resid[np.abs(grid - q) < 1e-4] = -np.inf
            idx = int(np.argmax(resid))
            if not np.isfinite(resid[idx]):
                break
            t.append(float(grid[idx]))
        return _finalize(t, budget, rng)


class QueryByCommitteeSampler(Sampler):
    name = "query_by_committee"
    adaptive = True
    citation = "seung1992"

    def __init__(self, committee=5, init_frac=0.25, n_candidates=256):
        self.committee = int(committee)
        self.init_frac = float(init_frac)
        self.n_candidates = int(n_candidates)

    def sample(self, motion, budget, rng, oracle, **kw):
        n0 = max(4, int(round(self.init_frac * budget)))
        t = list(rng.uniform(0.0, 1.0, min(n0, budget)))
        grid = np.linspace(0.0, 1.0, self.n_candidates)
        while len(t) < budget:
            tt = np.sort(np.array(t))
            yy = oracle.query(tt)
            preds = []
            for _ in range(self.committee):
                jitter = rng.uniform(0.0, 0.05, size=len(tt))
                knots = np.clip(np.sort(tt + jitter), 0.0, 1.0)
                knots = np.unique(np.concatenate([[0.0, 1.0], knots]))
                probe = PiecewiseLinear(knots, dim=motion.dim)
                probe.fit(tt, yy)
                preds.append(probe.eval(grid))
            var = np.var(np.stack(preds, axis=0), axis=0).sum(axis=1)
            for q in t:
                var[np.abs(grid - q) < 1e-4] = -np.inf
            idx = int(np.argmax(var))
            if not np.isfinite(var[idx]):
                break
            t.append(float(grid[idx]))
        return _finalize(t, budget, rng)


SAMPLERS = {
    "random": RandomSampler,
    "grid": GridSampler,
    "stratified": StratifiedSampler,
    "lhs": LatinHypercubeSampler,
    "sobol": SobolSampler,
    "halton": HaltonSampler,
    "farthest_point": FarthestPointSampler,
    "jerk_importance": JerkImportanceSampler,
    "derivative_peaks": DerivativePeakSampler,
    "hybrid": HybridSampler,
    "active_gp": ActiveGPSampler,
    "active_residual": ActiveResidualSampler,
    "query_by_committee": QueryByCommitteeSampler,
}

PLANNED_SAMPLERS = [
    "poisson_disk", "expected_error_reduction", "expected_model_change",
    "bayesian_experimental_design", "d_optimal", "a_optimal", "e_optimal",
    "recursive_least_squares", "kalman", "particle_filter",
    "incremental_knot_insertion", "compressed_sensing", "adversarial",
    "transfer_meta_learning",
]


def list_samplers():
    return sorted(SAMPLERS)


def build(name, **kw):
    if name not in SAMPLERS:
        raise KeyError(f"unknown sampler '{name}', options: {list_samplers()}")
    from .utils import filter_kwargs

    cls = SAMPLERS[name]
    return cls(**filter_kwargs(cls.__init__, kw))
