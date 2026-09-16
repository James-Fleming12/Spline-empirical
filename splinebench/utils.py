import numpy as np
from scipy.interpolate import PchipInterpolator

MAX_ORDER = 4


def as_rng(seed):
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def trapz(y, x):
    fn = getattr(np, "trapezoid", None)
    if fn is None:
        fn = np.trapz
    return float(fn(y, x))


def chain_rule_derivs(f, u):
    """Compose f(u(t)) derivatives up to order 4.

    f: (..., 5) derivatives of f with respect to u, orders 0..4.
    u: (..., 5) derivatives of u with respect to t, orders 0..4.
    Returns (..., 5) derivatives with respect to t.
    """
    f = np.asarray(f, dtype=float)
    u = np.asarray(u, dtype=float)
    f0, f1, f2, f3, f4 = (f[..., k] for k in range(5))
    u1, u2, u3, u4 = (u[..., k] for k in range(1, 5))
    shape = f.shape[: f.ndim - 1] + (5,)
    out = np.empty(shape, dtype=float)
    out[..., 0] = f0
    out[..., 1] = f1 * u1
    out[..., 2] = f2 * u1**2 + f1 * u2
    out[..., 3] = f3 * u1**3 + 3.0 * f2 * u1 * u2 + f1 * u3
    out[..., 4] = (
        f4 * u1**4
        + 6.0 * f3 * u1**2 * u2
        + 3.0 * f2 * u2**2
        + 4.0 * f2 * u1 * u3
        + f1 * u4
    )
    return out


def fd_derivs(t, values, order=4):
    """Finite-difference derivatives of values sampled on grid t (axis 0)."""
    d = np.asarray(values, dtype=float)
    t = np.asarray(t, dtype=float)
    out = [d]
    for _ in range(order):
        d = np.gradient(d, t, axis=0)
        out.append(d)
    return np.stack(out, axis=-1)


def savgol_deriv(t, y, order=0, window=None, polyorder=3):
    """Savitzky-Golay derivative estimate on possibly non-uniform samples."""
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    n = len(t)
    if order == 0 or n < 4:
        out = y
        for _ in range(order):
            out = np.gradient(out, t, axis=0)
        return out if order else y
    from scipy.signal import savgol_filter

    grid = np.linspace(t.min(), t.max(), max(64, 8 * n))
    vals = np.stack([PchipInterpolator(t, y[:, d])(grid) for d in range(y.shape[1])], axis=1)
    m = window or max(5, 2 * (n // 4) + 1)
    if m % 2 == 0:
        m += 1
    m = min(m, len(grid) if len(grid) % 2 else len(grid) - 1)
    m = max(m, order + 2 + (order % 2 == 0))
    if m % 2 == 0:
        m += 1
    po = min(max(polyorder, order + 1), m - 1)
    dv = np.stack(
        [savgol_filter(vals[:, d], m, po, deriv=order, mode="interp") for d in range(y.shape[1])],
        axis=1,
    )
    out = np.stack([PchipInterpolator(grid, dv[:, d])(t) for d in range(y.shape[1])], axis=1)
    return out[:, 0] if out.shape[1] == 1 else out


def monotone_map(x, cum, eps=1e-9):
    """PCHIP map from strictly increasing x to cumulative feature in [0, 1]."""
    x = np.asarray(x, dtype=float)
    cum = np.asarray(cum, dtype=float)
    if len(x) < 3 or not np.isfinite(cum).all():
        return None
    cum = np.maximum.accumulate(cum)
    span = cum[-1] - cum[0]
    if not np.isfinite(span) or span <= 0:
        return None
    c = (cum - cum[0]) / span
    c = c + eps * np.linspace(0.0, 1.0, len(c))
    c = (c - c[0]) / (c[-1] - c[0])
    return PchipInterpolator(x, c)


def weighted_quantile_positions(values, weights, n, lo=0.0, hi=1.0):
    """Positions in [lo, hi] at equal-weight quantiles of weighted values."""
    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    order = np.argsort(values)
    v = values[order]
    w = np.maximum(weights[order], 0.0)
    if w.sum() <= 0:
        return np.linspace(lo, hi, n)
    cw = np.cumsum(w)
    cw = (cw - cw[0]) / max(cw[-1] - cw[0], 1e-12)
    targets = np.linspace(0.0, 1.0, n)
    pos = np.interp(targets, cw, v)
    pos[0], pos[-1] = min(pos[0], lo), max(pos[-1], hi)
    return np.clip(np.unique(pos), lo, hi)


def first_nonzero(mask, default=0):
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else default


def filter_kwargs(fn, kwargs):
    import inspect

    if fn is object.__init__:
        return {}
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}
