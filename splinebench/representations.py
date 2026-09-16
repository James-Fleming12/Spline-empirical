import numpy as np

from .fitters import fit_linear, fit_nonlinear
from .utils import MAX_ORDER, savgol_deriv

DEFAULT_DIM = 3


class Representation:
    name = "abstract"
    kind = "nonlinear"
    status = "implemented"
    deriv_method = "analytic"
    citation = ""
    uses_ground_truth = False

    def __init__(self, positions=None, dim=DEFAULT_DIM, seed=0, **kw):
        self.positions = None if positions is None else np.asarray(positions, dtype=float).ravel()
        self.dim = int(dim)
        self.seed = int(seed)
        self.fitted = False

    def fit(self, u, y, weight=None, fitter="least_squares", reg=None, **kw):
        raise NotImplementedError

    def eval_derivs(self, u, order=MAX_ORDER):
        raise NotImplementedError

    def eval(self, u):
        return self.eval_derivs(np.asarray(u, dtype=float).ravel(), order=0)[..., 0]

    def _prepare(self, u, y, weight=None):
        u = np.asarray(u, dtype=float).ravel()
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        w = np.ones(len(u)) if weight is None else np.asarray(weight, dtype=float).ravel()
        keep = np.isfinite(u) & np.isfinite(y).all(axis=1) & np.isfinite(w) & (w > 0)
        return u[keep], y[keep], w[keep]

    @property
    def n_params(self):
        raise NotImplementedError

    @property
    def n_data_params(self):
        return 0

    @property
    def n_params_total(self):
        return self.n_params + self.n_data_params


def _design(u, knots, degree):
    from scipy.interpolate import BSpline

    knots = np.asarray(knots, dtype=float)
    u = np.clip(np.asarray(u, dtype=float).ravel(), knots[degree], knots[-degree - 1])
    return np.asarray(BSpline.design_matrix(u, knots, degree, extrapolate=False).todense(), dtype=float)


def _bspline_derivative_matrices(knots, degree, order):
    mats = [np.eye(len(knots) - degree - 1)]
    for r in range(1, order + 1):
        k = degree - (r - 1)
        t = knots[r - 1 : len(knots) - (r - 1)]
        n = len(t) - k - 1
        if n <= 1:
            break
        T = np.zeros((n - 1, n))
        for i in range(n - 1):
            den = t[i + k + 1] - t[i + 1]
            if abs(den) < 1e-12:
                continue
            T[i, i] = -k / den
            T[i, i + 1] = k / den
        mats.append(T @ mats[-1])
    return mats


def _hermite_value_derivs(d, s):
    if d == 0:
        return 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    if d == 1:
        return 6 * s**2 - 6 * s, 3 * s**2 - 4 * s + 1, -6 * s**2 + 6 * s, 3 * s**2 - 2 * s
    if d == 2:
        return 12 * s - 6, 6 * s - 4, -12 * s + 6, 6 * s - 2
    if d == 3:
        return 12 + 0 * s, 6 + 0 * s, -12 + 0 * s, 6 + 0 * s
    z = 0 * s
    return z, z, z, z


def _hermite_derivative_matrices(knots, u, order=MAX_ORDER):
    knots = np.asarray(knots, dtype=float)
    n_knots = len(knots)
    if n_knots < 2:
        raise ValueError("Hermite interpolation needs at least two knots")
    u = np.clip(np.asarray(u, dtype=float).ravel(), knots[0], knots[-1])
    seg = np.clip(np.searchsorted(knots, u, side="right") - 1, 0, n_knots - 2)
    h = knots[seg + 1] - knots[seg]
    s = (u - knots[seg]) / h
    rows = np.arange(len(u))
    out = []
    for d in range(order + 1):
        A = np.zeros((len(u), n_knots))
        h00, h10, h01, h11 = _hermite_value_derivs(d, s)
        p_scale = h ** (-d)
        np.add.at(A, (rows, seg), h00 * p_scale)
        np.add.at(A, (rows, seg + 1), h01 * p_scale)
        m_scale = h ** (1 - d)
        for tangent, weight in ((seg, h10 * m_scale), (seg + 1, h11 * m_scale)):
            first = tangent == 0
            last = tangent == n_knots - 1
            mid = ~(first | last)
            if mid.any():
                i = tangent[mid]
                den = knots[i + 1] - knots[i - 1]
                np.add.at(A, (rows[mid], i + 1), weight[mid] / den)
                np.add.at(A, (rows[mid], i - 1), -weight[mid] / den)
            if first.any():
                den = knots[1] - knots[0]
                np.add.at(A, (rows[first], 1), weight[first] / den)
                np.add.at(A, (rows[first], 0), -weight[first] / den)
            if last.any():
                den = knots[-1] - knots[-2]
                np.add.at(A, (rows[last], n_knots - 1), weight[last] / den)
                np.add.at(A, (rows[last], n_knots - 2), -weight[last] / den)
        out.append(A)
    return out


class LinearBasis(Representation):
    kind = "linear"

    def basis_derivs(self, u, order=MAX_ORDER):
        raise NotImplementedError

    def fit(self, u, y, weight=None, fitter="least_squares", reg=None, **kw):
        uu, yy, ww = self._prepare(u, y, weight)
        A = self.basis_derivs(uu, 0)[0]
        self.coeffs = fit_linear(A, yy, ww, fitter=fitter, reg=reg, seed=self.seed, **kw)
        self.fitted = True
        return self

    def eval_derivs(self, u, order=MAX_ORDER):
        uu = np.asarray(u, dtype=float).ravel()
        mats = self.basis_derivs(uu, order)
        return np.stack([np.asarray(B @ self.coeffs, dtype=float) for B in mats], axis=2)

    @property
    def n_params(self):
        return int(self.coeffs.size)


class PiecewiseLinear(LinearBasis):
    name = "linear"
    citation = "deboor1978"

    def __init__(self, positions=None, dim=DEFAULT_DIM, seed=0, **kw):
        pos = np.linspace(0.0, 1.0, 8) if positions is None else positions
        pos = np.unique(np.clip(np.asarray(pos, dtype=float).ravel(), 0.0, 1.0))
        if len(pos) < 2:
            pos = np.linspace(0.0, 1.0, 2)
        super().__init__(pos, dim, seed, **kw)
        self.knots = pos

    def basis_derivs(self, u, order=MAX_ORDER):
        t = self.knots
        k = len(t)
        u = np.clip(np.asarray(u, dtype=float).ravel(), 0.0, 1.0)
        seg = np.clip(np.searchsorted(t, u, side="right") - 1, 0, k - 2)
        h = t[seg + 1] - t[seg]
        rows = np.arange(len(u))
        out = [np.zeros((len(u), k)) for _ in range(order + 1)]
        A0 = out[0]
        np.add.at(A0, (rows, seg), (t[seg + 1] - u) / h)
        np.add.at(A0, (rows, seg + 1), (u - t[seg]) / h)
        if order >= 1:
            A1 = out[1]
            np.add.at(A1, (rows, seg), -1.0 / h)
            np.add.at(A1, (rows, seg + 1), 1.0 / h)
        return out


class BSpline(LinearBasis):
    name = "bspline"
    citation = "deboor1978"

    def __init__(self, positions=None, dim=DEFAULT_DIM, degree=3, seed=0, **kw):
        super().__init__(positions, dim, seed, **kw)
        self.degree = int(degree)
        interior = np.linspace(0.0, 1.0, 8)[1:-1] if positions is None else np.asarray(positions, dtype=float).ravel()
        interior = np.unique(np.clip(interior, 1e-9, 1.0 - 1e-9))
        if len(interior) > 1:
            interior = interior[np.r_[True, np.diff(interior) > 1e-9]]
        self.knots = np.concatenate([[0.0] * (self.degree + 1), interior, [1.0] * (self.degree + 1)])
        self.n_basis = len(self.knots) - self.degree - 1
        self._mats = _bspline_derivative_matrices(self.knots, self.degree, MAX_ORDER)

    def basis_derivs(self, u, order=MAX_ORDER):
        u = np.clip(np.asarray(u, dtype=float).ravel(), 0.0, 1.0)
        out = []
        for r in range(order + 1):
            if r > self.degree or r >= len(self._mats):
                out.append(np.zeros((len(u), self.n_basis)))
                continue
            knots_r = self.knots[r : len(self.knots) - r] if r > 0 else self.knots
            B = _design(u, knots_r, self.degree - r)
            out.append(B @ self._mats[r])
        return out


class PSpline(BSpline):
    name = "pspline"
    citation = "eilers1996"

    def __init__(self, positions=None, dim=DEFAULT_DIM, degree=3, lam=1e-4, seed=0, **kw):
        super().__init__(positions, dim=dim, degree=degree, seed=seed, **kw)
        self.default_reg = {"kind": "diff", "order": 2, "lam": float(lam)}

    def fit(self, u, y, weight=None, fitter="least_squares", reg=None, **kw):
        return super().fit(u, y, weight=weight, fitter=fitter, reg=reg or self.default_reg, **kw)


class Hermite(LinearBasis):
    name = "hermite"
    citation = "park2024splinegs"

    def __init__(self, positions=None, dim=DEFAULT_DIM, seed=0, **kw):
        pos = np.linspace(0.0, 1.0, 8) if positions is None else positions
        pos = np.unique(np.clip(np.asarray(pos, dtype=float).ravel(), 0.0, 1.0))
        if len(pos) < 2:
            pos = np.linspace(0.0, 1.0, 2)
        super().__init__(pos, dim, seed, **kw)
        self.knots = pos

    def basis_derivs(self, u, order=MAX_ORDER):
        return _hermite_derivative_matrices(self.knots, u, order)


class CatmullRom(Hermite):
    name = "catmull_rom"
    citation = "catmull1974"

    def __init__(self, positions=None, dim=DEFAULT_DIM, seed=0, **kw):
        n = 8 if positions is None else max(len(np.asarray(positions).ravel()), 2)
        super().__init__(np.linspace(0.0, 1.0, n), dim, seed, **kw)


class Chebyshev(LinearBasis):
    name = "chebyshev"
    citation = "trefethen2013"

    def __init__(self, positions=None, dim=DEFAULT_DIM, degree=None, seed=0, **kw):
        super().__init__(positions, dim, seed, **kw)
        self.degree = int(degree) if degree is not None else max(len(positions) - 1, 2)

    def basis_derivs(self, u, order=MAX_ORDER):
        from numpy.polynomial.chebyshev import chebder, chebvander

        x = 2.0 * np.clip(np.asarray(u, dtype=float).ravel(), 0.0, 1.0) - 1.0
        out = []
        M = np.eye(self.degree + 1)
        for r in range(order + 1):
            if r == 0:
                out.append(chebvander(x, self.degree))
                continue
            M = chebder(M, axis=0)
            deg = self.degree - r
            if deg < 0 or M.shape[0] != deg + 1:
                out.append(np.zeros((len(x), self.degree + 1)))
                continue
            out.append(chebvander(x, deg) @ M)
        return out


class Fourier(LinearBasis):
    name = "fourier"
    citation = ""

    def __init__(self, positions=None, dim=DEFAULT_DIM, n_harmonics=None, seed=0, **kw):
        super().__init__(positions, dim, seed, **kw)
        if n_harmonics is None:
            n_harmonics = max((len(positions) - 1) // 2, 1) if positions is not None else 4
        self.n_harmonics = int(n_harmonics)

    def basis_derivs(self, u, order=MAX_ORDER):
        u = np.asarray(u, dtype=float).ravel()
        ks = np.arange(1, self.n_harmonics + 1)
        w = 2.0 * np.pi * ks
        out = []
        for r in range(order + 1):
            const = np.ones((len(u), 1)) if r == 0 else np.zeros((len(u), 1))
            arg = np.outer(u, w)
            cos = np.cos(arg + 0.5 * r * np.pi) * (w**r)[None, :]
            sin = np.sin(arg + 0.5 * r * np.pi) * (w**r)[None, :]
            out.append(np.concatenate([const, cos, sin], axis=1))
        return out

    @property
    def n_params(self):
        return int(self.coeffs.size)


class NURBS(Representation):
    name = "nurbs"
    citation = "piegl1995"

    def __init__(self, positions=None, dim=DEFAULT_DIM, degree=3, optimize_weights=False, seed=0, **kw):
        super().__init__(positions, dim, seed, **kw)
        self.degree = int(degree)
        interior = np.linspace(0.0, 1.0, 8)[1:-1] if positions is None else np.asarray(positions, dtype=float).ravel()
        interior = np.unique(np.clip(interior, 1e-9, 1.0 - 1e-9))
        if len(interior) > 1:
            interior = interior[np.r_[True, np.diff(interior) > 1e-9]]
        self.knots = np.concatenate([[0.0] * (self.degree + 1), interior, [1.0] * (self.degree + 1)])
        self.n_basis = len(self.knots) - self.degree - 1
        self.optimize_weights = bool(optimize_weights)
        self._mats = _bspline_derivative_matrices(self.knots, self.degree, MAX_ORDER)
        self.weights = np.ones(self.n_basis)

    def _raw_derivs(self, u, order):
        u = np.clip(np.asarray(u, dtype=float).ravel(), 0.0, 1.0)
        out = []
        for r in range(order + 1):
            if r > self.degree or r >= len(self._mats):
                out.append(np.zeros((len(u), self.n_basis)))
                continue
            knots_r = self.knots[r : len(self.knots) - r] if r > 0 else self.knots
            out.append(_design(u, knots_r, self.degree - r) @ self._mats[r])
        return out

    def _rational(self, u, order, coeffs):
        raw = self._raw_derivs(u, order)
        w = self.weights
        numer = [B * w[None, :] for B in raw]
        W = numer[0].sum(axis=1)
        out = []
        for k in range(order + 1):
            acc = numer[k].copy()
            for j in range(1, k + 1):
                from math import comb

                acc -= comb(k, j) * out[k - j] * np.outer(raw[j].sum(axis=1), np.ones(self.n_basis))
            out.append(acc / W[:, None])
        return np.stack([B @ coeffs for B in out], axis=2)

    def fit(self, u, y, weight=None, fitter="least_squares", reg=None, **kw):
        uu, yy, ww = self._prepare(u, y, weight)
        A = self._raw_derivs(uu, 0)[0]
        W = A @ self.weights
        R = A * self.weights[None, :] / W[:, None]
        coeffs = fit_linear(R, yy, ww, fitter=fitter, reg=reg, seed=self.seed, **kw)
        if self.optimize_weights:
            def residual(theta):
                logw = theta[: self.n_basis]
                P = theta[self.n_basis :].reshape(self.n_basis, self.dim)
                w = np.exp(np.clip(logw, -8.0, 8.0))
                W = A @ w
                pred = A @ (w[:, None] * P) / W[:, None]
                return ((pred - yy) * ww[:, None]).ravel()

            theta0 = np.concatenate([np.zeros(self.n_basis), coeffs.ravel()])
            theta = fit_nonlinear(residual, theta0, method="lm", max_iter=4000)
            self.weights = np.exp(np.clip(theta[: self.n_basis], -8.0, 8.0))
            coeffs = theta[self.n_basis :].reshape(self.n_basis, self.dim)
        self.coeffs = coeffs
        self.fitted = True
        return self

    def eval_derivs(self, u, order=MAX_ORDER):
        return self._rational(np.asarray(u, dtype=float).ravel(), order, self.coeffs)

    @property
    def n_params(self):
        base = int(self.coeffs.size)
        return base + (self.n_basis if self.optimize_weights else 0)


class InterpolatingSpline(Representation):
    name = "pchip"
    citation = "fritsch1980"

    def __init__(self, positions=None, dim=DEFAULT_DIM, scheme="pchip", seed=0, **kw):
        super().__init__(positions, dim, seed, **kw)
        self.scheme = scheme
        pos = np.linspace(0.0, 1.0, 8) if positions is None else np.asarray(positions, dtype=float).ravel()
        self.knots = np.unique(np.clip(pos, 0.0, 1.0))
        if len(self.knots) < 2:
            self.knots = np.linspace(0.0, 1.0, 2)
        self.values = None

    def _interpolator(self, values):
        if self.scheme == "akima":
            from scipy.interpolate import Akima1DInterpolator

            return Akima1DInterpolator(self.knots, values, axis=0, extrapolate=True)
        from scipy.interpolate import PchipInterpolator

        return PchipInterpolator(self.knots, values, axis=0, extrapolate=True)

    def fit(self, u, y, weight=None, fitter="least_squares", reg=None, **kw):
        uu, yy, ww = self._prepare(u, y, weight)
        v0 = np.stack([np.interp(self.knots, uu, yy[:, d]) for d in range(yy.shape[1])], axis=1)

        def residual(theta):
            values = theta.reshape(len(self.knots), yy.shape[1])
            pred = self._interpolator(values)(uu)
            return ((pred - yy) * ww[:, None]).ravel()

        theta = fit_nonlinear(residual, v0.ravel(), method="lm", max_iter=kw.get("iters", 2000))
        self.values = theta.reshape(len(self.knots), yy.shape[1])
        self.fitted = True
        return self

    def eval_derivs(self, u, order=MAX_ORDER):
        uu = np.asarray(u, dtype=float).ravel()
        interp = self._interpolator(self.values)
        out = []
        for k in range(order + 1):
            if k <= 3:
                out.append(interp.derivative(k)(uu))
            else:
                out.append(np.zeros((len(uu), self.dim)))
        return np.stack(out, axis=2)

    @property
    def n_params(self):
        return 0 if self.values is None else int(self.values.size)


class Akima(InterpolatingSpline):
    name = "akima"
    citation = "akima1970"

    def __init__(self, positions=None, dim=DEFAULT_DIM, seed=0, **kw):
        super().__init__(positions, dim, scheme="akima", seed=seed, **kw)


class PCHIP(InterpolatingSpline):
    name = "pchip"
    citation = "fritsch1980"


class GaussianProcess(Representation):
    name = "gp_rbf"
    kind = "kernel"
    deriv_method = "analytic"
    citation = "rasmussen2006"

    def __init__(self, positions=None, dim=DEFAULT_DIM, kernel="rbf", optimize=True, seed=0, **kw):
        super().__init__(positions, dim, seed, **kw)
        self.kernel = kernel
        self.optimize = bool(optimize)
        self.log_params = np.zeros(3)
        self.inputs = None
        self.targets = None

    def _kernel(self, a, b, order=0):
        a = np.asarray(a, dtype=float).ravel()[:, None]
        b = np.asarray(b, dtype=float).ravel()[None, :]
        l = float(np.exp(self.log_params[0]))
        sf2 = float(np.exp(self.log_params[1]))
        tau = a - b
        if self.kernel == "matern52":
            from numpy.polynomial.polynomial import polyder, polyval

            aa = np.sqrt(5.0) / l
            coef = np.array([1.0, aa, aa**2 / 3.0]) * sf2
            q = coef
            for _ in range(order):
                dq = polyder(q)
                if len(dq) < len(q):
                    dq = np.concatenate([dq, np.zeros(len(q) - len(dq))])
                q = dq - aa * q
            if len(q) == 0 or not q.any():
                return np.zeros_like(tau)
            return polyval(tau, q) * np.exp(-aa * np.abs(tau))
        x = tau / l
        if order == 0:
            return sf2 * np.exp(-0.5 * x**2)
        from numpy.polynomial.hermite_e import hermeval

        c = np.zeros(order + 1)
        c[order] = 1.0
        return sf2 * ((-1.0) ** order) * (l ** (-order)) * hermeval(x, c) * np.exp(-0.5 * x**2)

    def _neg_log_marginal(self, logs):
        self.log_params = np.asarray(logs, dtype=float)
        sn2 = np.exp(self.log_params[2])
        K = self._kernel(self.inputs, self.inputs, 0) + (sn2 + 1e-10) * np.eye(len(self.inputs))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e12
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.targets))
        n = len(self.inputs)
        nll = 0.0
        for d in range(self.targets.shape[1]):
            nll += 0.5 * float(self.targets[:, d] @ alpha[:, d])
        nll += self.targets.shape[1] * np.sum(np.log(np.diag(L)))
        nll += self.targets.shape[1] * 0.5 * n * np.log(2 * np.pi)
        return nll

    def fit(self, u, y, weight=None, fitter="least_squares", reg=None, **kw):
        from scipy.optimize import minimize

        uu, yy, _ = self._prepare(u, y, weight)
        order = np.argsort(uu)
        self.inputs, self.targets = uu[order], yy[order]
        span = max(self.inputs[-1] - self.inputs[0], 1e-6)
        y_var = float(np.var(self.targets)) + 1e-12
        self.log_params = np.array([np.log(span / 5.0), np.log(y_var), np.log(max(1e-6, 1e-4 * y_var))])
        if self.optimize and len(self.inputs) >= 4:
            res = minimize(
                self._neg_log_marginal,
                self.log_params,
                method="L-BFGS-B",
                bounds=[(np.log(1e-3), np.log(1e3)), (np.log(1e-8), np.log(1e4)), (np.log(1e-10), np.log(1e0))],
                options={"maxiter": 200},
            )
            self.log_params = res.x
        self.fitted = True
        return self

    def eval_derivs(self, u, order=MAX_ORDER):
        uu = np.asarray(u, dtype=float).ravel()
        sn2 = np.exp(self.log_params[2])
        K = self._kernel(self.inputs, self.inputs, 0) + (sn2 + 1e-10) * np.eye(len(self.inputs))
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.targets))
        return np.stack([self._kernel(uu, self.inputs, k) @ alpha for k in range(order + 1)], axis=2)

    def posterior_std(self, u):
        from scipy.linalg import solve_triangular

        uu = np.asarray(u, dtype=float).ravel()
        sn2 = np.exp(self.log_params[2])
        K = self._kernel(self.inputs, self.inputs, 0) + (sn2 + 1e-10) * np.eye(len(self.inputs))
        L = np.linalg.cholesky(K)
        ks = self._kernel(uu, self.inputs, 0)
        v = solve_triangular(L, ks.T, lower=True)
        sf2 = float(np.exp(self.log_params[1]))
        return np.sqrt(np.maximum(sf2 - np.sum(v**2, axis=0), 0.0))

    @property
    def n_params(self):
        return 3

    @property
    def n_data_params(self):
        return 0 if self.inputs is None else len(self.inputs)


class DMP(Representation):
    name = "dmp"
    kind = "nonlinear"
    deriv_method = "numerical"
    citation = "ijspeert2013"

    def __init__(self, positions=None, dim=DEFAULT_DIM, n_basis=None, alpha_y=25.0, alpha_s=4.0,
                 seed=0, grid=401, **kw):
        super().__init__(positions, dim, seed, **kw)
        self.n_basis = int(n_basis or (len(positions) if positions is not None else 20))
        self.alpha_y = float(alpha_y)
        self.beta_y = self.alpha_y / 4.0
        self.alpha_s = float(alpha_s)
        self.grid_n = int(grid)

    def _basis_at(self, s):
        centers = np.exp(-self.alpha_s * np.linspace(0.0, 1.0, self.n_basis))
        if self.n_basis > 1:
            widths = np.maximum(np.abs(np.gradient(centers)), 1e-4)
        else:
            widths = np.array([1.0 / (self.alpha_s + 1e-9)])
        return np.exp(-((s[:, None] - centers[None, :]) / widths[None, :]) ** 2)

    def fit(self, u, y, weight=None, fitter="least_squares", reg=None, **kw):
        uu, yy, ww = self._prepare(u, y, weight)
        order = np.argsort(uu)
        uu, yy, ww = uu[order], yy[order], ww[order]
        if len(uu) < 5:
            self.x0 = yy[0].copy()
            self.goal = yy[-1].copy()
            self.weights = np.zeros((self.n_basis, self.dim))
            self.fitted = True
            return self
        self.x0 = yy[0].copy()
        self.goal = yy[-1].copy()
        v = savgol_deriv(uu, yy, order=1)
        a = savgol_deriv(uu, yy, order=2)
        s = np.exp(-self.alpha_s * uu)
        psi = self._basis_at(s)
        f_target = a - self.alpha_y * (self.beta_y * (self.goal[None, :] - yy) - v)
        A = s[:, None] * psi / np.maximum(psi.sum(axis=1, keepdims=True), 1e-12)
        self.weights = fit_linear(A, f_target, ww, fitter="least_squares",
                                  reg={"kind": "ridge", "lam": 1e-8})
        self.fitted = True
        return self

    def _integrate(self):
        grid = np.linspace(0.0, 1.0, self.grid_n)
        state = np.concatenate([self.x0, np.zeros(self.dim)])
        out = np.zeros((self.grid_n, 2 * self.dim))
        dt = grid[1] - grid[0]

        def deriv(t, x):
            pos, vel = x[: self.dim], x[self.dim :]
            s = np.exp(-self.alpha_s * t)
            psi = self._basis_at(np.array([s]))[0]
            forcing = (psi @ self.weights) / max(psi.sum(), 1e-12) * s
            acc = self.alpha_y * (self.beta_y * (self.goal - pos) - vel) + forcing
            return np.concatenate([vel, acc])

        for i in range(self.grid_n - 1):
            k1 = deriv(grid[i], state)
            k2 = deriv(grid[i] + 0.5 * dt, state + 0.5 * dt * k1)
            k3 = deriv(grid[i] + 0.5 * dt, state + 0.5 * dt * k2)
            k4 = deriv(grid[i] + dt, state + dt * k3)
            state = state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            out[i + 1] = state
        return grid, out[:, : self.dim]

    def eval_derivs(self, u, order=MAX_ORDER):
        from scipy.interpolate import CubicSpline

        uu = np.clip(np.asarray(u, dtype=float).ravel(), 0.0, 1.0)
        grid, pos = self._integrate()
        spline = CubicSpline(grid, pos, axis=0)
        out = []
        for k in range(order + 1):
            out.append(spline.derivative(k)(uu) if k <= 3 else np.zeros((len(uu), self.dim)))
        return np.stack(out, axis=2)

    @property
    def n_params(self):
        return 0 if self.weights is None else int(self.weights.size + 2 * self.dim)


class TorchMLP(Representation):
    name = "mlp"
    kind = "nonlinear"
    deriv_method = "autograd"
    citation = "sitzmann2020"

    def __init__(self, positions=None, dim=DEFAULT_DIM, hidden=(64, 64), n_features=8, iters=1500,
                 lr=5e-3, weight_decay=1e-5, seed=0, **kw):
        super().__init__(positions, dim, seed, **kw)
        self.hidden = tuple(hidden)
        self.n_features = int(n_features)
        self.iters = int(iters)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.net = None

    def _features(self, t):
        import torch

        t = t.reshape(-1)
        feats = [t[:, None]]
        for k in range(1, self.n_features + 1):
            feats.append(torch.sin(2.0 * np.pi * k * t)[:, None])
            feats.append(torch.cos(2.0 * np.pi * k * t)[:, None])
        return torch.cat(feats, dim=1)

    def _build(self):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        layers = []
        in_dim = 1 + 2 * self.n_features
        for h in self.hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers.append(nn.Linear(in_dim, self.dim))
        net = nn.Sequential(*layers).double()
        return net

    def fit(self, u, y, weight=None, fitter=None, reg=None, **kw):
        import torch

        uu, yy, ww = self._prepare(u, y, weight)
        self.net = self._build()
        iters = int(kw.get("iters", self.iters))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        t = torch.tensor(uu, dtype=torch.float64)[:, None]
        target = torch.tensor(yy, dtype=torch.float64)
        w = torch.tensor(ww, dtype=torch.float64)[:, None]
        loss_fn = torch.nn.MSELoss(reduction="none")
        for _ in range(iters):
            opt.zero_grad()
            pred = self.net(self._features(t))
            loss = (loss_fn(pred, target) * w).mean()
            loss.backward()
            opt.step()
        self.fitted = True
        return self

    def eval_derivs(self, u, order=MAX_ORDER):
        import torch

        uu = np.asarray(u, dtype=float).ravel()
        t = torch.tensor(uu, dtype=torch.float64, requires_grad=True)
        pred = self.net(self._features(t))
        out = np.zeros((len(uu), self.dim, order + 1))
        for d in range(self.dim):
            cur = pred[:, d]
            out[:, d, 0] = cur.detach().numpy()
            for k in range(1, order + 1):
                cur = torch.autograd.grad(cur.sum(), t, create_graph=True, retain_graph=True)[0]
                out[:, d, k] = cur.detach().numpy()
            pred = self.net(self._features(t))
        return out

    @property
    def n_params(self):
        if self.net is None:
            return 0
        return int(sum(p.numel() for p in self.net.parameters()))


class BSplineMLP(Representation):
    name = "bspline_mlp"
    kind = "nonlinear"
    deriv_method = "analytic"
    citation = ""

    def __init__(self, positions=None, dim=DEFAULT_DIM, degree=3, hidden=(64, 64), iters=800, seed=0, **kw):
        super().__init__(positions, dim, seed, **kw)
        self.degree = int(degree)
        self.hidden = tuple(hidden)
        self.iters = int(iters)
        self.backbone = None
        self.head = None

    def fit(self, u, y, weight=None, fitter="least_squares", reg=None, **kw):
        uu, yy, ww = self._prepare(u, y, weight)
        self.backbone = BSpline(self.positions, dim=self.dim, degree=self.degree, seed=self.seed)
        self.backbone.fit(uu, yy, weight=ww, fitter=fitter, reg=reg)
        residual = yy - self.backbone.eval(uu)
        self.head = TorchMLP(dim=self.dim, hidden=self.hidden, iters=self.iters, seed=self.seed + 1)
        self.head.fit(uu, residual, weight=ww)
        self.fitted = True
        return self

    def eval_derivs(self, u, order=MAX_ORDER):
        base = np.zeros((len(np.asarray(u).ravel()), self.dim, order + 1))
        for part in (self.backbone, self.head):
            if part is not None:
                base += part.eval_derivs(u, order)
        return base

    @property
    def n_params(self):
        return (0 if self.backbone is None else self.backbone.n_params) + (
            0 if self.head is None else self.head.n_params
        )


REPRESENTATIONS = {
    "linear": PiecewiseLinear,
    "hermite": Hermite,
    "catmull_rom": CatmullRom,
    "bspline2": lambda positions=None, **kw: BSpline(positions, degree=2, **kw),
    "bspline3": lambda positions=None, **kw: BSpline(positions, degree=3, **kw),
    "bspline5": lambda positions=None, **kw: BSpline(positions, degree=5, **kw),
    "bspline7": lambda positions=None, **kw: BSpline(positions, degree=7, **kw),
    "pspline": PSpline,
    "chebyshev": Chebyshev,
    "fourier": Fourier,
    "nurbs": NURBS,
    "akima": Akima,
    "pchip": PCHIP,
    "gp_rbf": lambda positions=None, **kw: GaussianProcess(positions, kernel="rbf", **kw),
    "gp_matern52": lambda positions=None, **kw: GaussianProcess(positions, kernel="matern52", **kw),
    "dmp": DMP,
    "mlp": TorchMLP,
    "bspline_mlp": BSplineMLP,
}

PLANNED_REPRESENTATIONS = [
    "quadratic", "quintic", "septic", "quintic_hermite", "bezier", "rational_bezier",
    "composite_bezier", "uniform_bspline", "nonuniform_bspline", "periodic_bspline",
    "tspline", "lr_bspline", "hierarchical_bspline", "mspline", "ispline",
    "kochanek_bartels", "steffen", "thin_plate", "polyharmonic", "wavelet",
    "legendre", "bernstein", "lagrange", "newton_divided",
    "exponential_spline", "trigonometric_spline", "l_spline", "box_spline",
    "chaikin", "catmull_clark", "loop_subdivision", "doo_sabin",
    "slerp", "squad", "so3_bezier", "so3_bspline", "se3_screw", "dual_quaternion_spline",
    "riemannian_spline", "promp", "prodmp", "bayesian_interaction_primitive",
    "gmm_gmr", "hmm", "neural_ode", "siren", "implicit_neural", "neural_spline_flow",
    "diffusion_policy", "transformer_policy", "minimum_jerk_poly", "minimum_snap_poly",
    "bang_bang", "trapezoidal", "s_curve", "asymmetric_s_curve", "topp",
    "spline_gp_residual", "spline_mpc",
]

REP_GROUPS = {
    "interpolating": ["linear", "hermite", "catmull_rom", "akima", "pchip"],
    "bspline_family": ["bspline2", "bspline3", "bspline5", "bspline7", "pspline", "nurbs"],
    "global_basis": ["chebyshev", "fourier"],
    "kernel": ["gp_rbf", "gp_matern52"],
    "motion_primitive": ["dmp"],
    "neural": ["mlp", "bspline_mlp"],
}


def list_representations():
    return sorted(REPRESENTATIONS)


def build(name, positions=None, dim=DEFAULT_DIM, **kw):
    if name not in REPRESENTATIONS:
        raise KeyError(f"unknown representation '{name}', options: {list_representations()}")
    return REPRESENTATIONS[name](positions=positions, dim=dim, **kw)
