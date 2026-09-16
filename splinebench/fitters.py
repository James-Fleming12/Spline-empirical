import numpy as np

LINEAR_FITTERS = [
    "least_squares",
    "weighted",
    "ridge",
    "huber",
    "ransac",
    "lasso",
    "elastic_net",
    "tv",
    "adam",
]

NONLINEAR_METHODS = [
    "lm",
    "trf",
    "dogbox",
    "nelder_mead",
    "powell",
    "differential_evolution",
    "bayes_opt",
]

REGULARIZERS = ["none", "ridge", "diff", "tv"]


def diff_matrix(k, order):
    pts = max(k - order, 1)
    if order <= 0:
        return np.eye(k)
    return np.diff(np.eye(k), n=order, axis=0)


def _as_columns(y):
    y = np.asarray(y, dtype=float)
    return y[:, None] if y.ndim == 1 else y


def _mask(A, y, w):
    A = np.asarray(A, dtype=float)
    y = _as_columns(y)
    if w is None:
        w = np.ones(len(A))
    w = np.asarray(w, dtype=float).ravel()
    keep = np.isfinite(A).all(axis=1) & np.isfinite(y).all(axis=1) & np.isfinite(w) & (w > 0)
    return A[keep], y[keep], w[keep]


def _reg_kind(reg):
    if not reg:
        return "none", 0.0, 2
    return str(reg.get("kind", "none")), float(reg.get("lam", 0.0) or 0.0), int(reg.get("order", 2))


def _augment(A, y, w, reg):
    kind, lam, order = _reg_kind(reg)
    blocks_a = [A * w[:, None]]
    blocks_y = [y * w[:, None]]
    if lam > 0 and kind == "ridge":
        k = A.shape[1]
        blocks_a.append(np.sqrt(lam) * np.eye(k))
        blocks_y.append(np.zeros((k, y.shape[1])))
    if lam > 0 and kind == "diff":
        d = diff_matrix(A.shape[1], order)
        blocks_a.append(np.sqrt(lam) * d)
        blocks_y.append(np.zeros((d.shape[0], y.shape[1])))
    return np.vstack(blocks_a), np.vstack(blocks_y)


def _solve(A, y, w, reg):
    Aa, ya = _augment(A, y, w, reg)
    return np.linalg.lstsq(Aa, ya, rcond=None)[0]


def _huber(A, y, w, reg, delta=None, iters=30):
    coeffs = _solve(A, y, w, reg)
    for _ in range(iters):
        r = y - A @ coeffs
        scale = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
        d = delta if delta is not None else 1.345 * scale
        rw = np.minimum(1.0, d / np.maximum(np.abs(r), 1e-12)).min(axis=1)
        ww = w * np.maximum(rw, 1e-3)
        new = _solve(A, y, ww, reg)
        if np.linalg.norm(new - coeffs) <= 1e-10 * (1.0 + np.linalg.norm(coeffs)):
            coeffs = new
            break
        coeffs = new
    return coeffs


def _ransac(A, y, w, reg, seed, n_trials=200, thresh_scale=2.5):
    rng = np.random.default_rng(seed)
    n, k = A.shape
    if n <= k:
        return _solve(A, y, w, reg)
    base = _solve(A, y, w, None)
    r0 = np.linalg.norm((A @ base - y) * w[:, None], axis=1)
    scale = 1.4826 * np.median(np.abs(r0 - np.median(r0))) + 1e-12
    thresh = max(thresh_scale * scale, 1e-9)
    best_mask = None
    best_score = -1
    trials = int(min(n_trials, max(50, 8 * n)))
    for _ in range(trials):
        idx = rng.choice(n, size=min(k, n), replace=False)
        try:
            c = np.linalg.lstsq(A[idx] * w[idx, None], y[idx] * w[idx, None], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        r = np.linalg.norm((A @ c - y) * w[:, None], axis=1)
        inl = r < thresh
        score = int(inl.sum())
        if score > best_score:
            best_score, best_mask = score, inl
    if best_mask is None or best_mask.sum() < k:
        return _solve(A, y, w, reg)
    return _solve(A[best_mask], y[best_mask], w[best_mask], reg)


def _sklearn_linear(A, y, fitter, reg, seed, **kw):
    try:
        from sklearn.linear_model import ElasticNet, Lasso
    except ImportError as exc:
        raise ImportError(f"fitter '{fitter}' requires scikit-learn") from exc
    kind, lam, _ = _reg_kind(reg)
    alpha = float(kw.get("alpha", lam if lam > 0 else 1e-4))
    model = Lasso(alpha=alpha, max_iter=20000, random_state=seed) if fitter == "lasso" else ElasticNet(
        alpha=alpha, l1_ratio=float(kw.get("l1_ratio", 0.5)), max_iter=20000, random_state=seed
    )
    coeffs = np.zeros((A.shape[1], y.shape[1]))
    for d in range(y.shape[1]):
        model.fit(A, y[:, d], sample_weight=kw.get("sample_weight"))
        coeffs[:, d] = model.coef_
    return coeffs


def _tv(A, y, w, reg, iters=200):
    from scipy.optimize import minimize

    kind, lam, order = _reg_kind(reg)
    lam = lam if lam > 0 else 1e-3
    d = diff_matrix(A.shape[1], order)
    x0, _ = np.linalg.lstsq(A * w[:, None], y * w[:, None], rcond=None)[:2]

    def objective(vec):
        c = vec.reshape(A.shape[1], y.shape[1])
        data = np.sum(((A @ c - y) * w[:, None]) ** 2)
        smooth = np.sqrt((d @ c) ** 2 + 1e-8).sum()
        return data + lam * smooth

    res = minimize(objective, x0.ravel(), method="L-BFGS-B", options={"maxiter": iters})
    return res.x.reshape(A.shape[1], y.shape[1])


def _adam(A, y, w, seed, iters=1500, lr=0.02, weight_decay=0.0):
    import torch

    torch.manual_seed(seed)
    at = torch.tensor(A, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    wt = torch.tensor(w, dtype=torch.float64)[:, None]
    c = torch.zeros(A.shape[1], y.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([c], lr=lr, weight_decay=weight_decay)
    for _ in range(int(iters)):
        opt.zero_grad()
        loss = (((at @ c - yt) * wt) ** 2).mean()
        loss.backward()
        opt.step()
    return c.detach().numpy()


def fit_linear(A, y, weight=None, fitter="least_squares", reg=None, seed=0, **kw):
    A, y, w = _mask(A, y, weight)
    if len(A) == 0:
        raise ValueError("no finite observations to fit")
    if fitter in ("least_squares", "weighted", "ridge"):
        return _solve(A, y, w, reg)
    if fitter == "huber":
        return _huber(A, y, w, reg, delta=kw.get("delta"), iters=kw.get("iters", 30))
    if fitter == "ransac":
        return _ransac(A, y, w, reg, seed, n_trials=kw.get("n_trials", 200))
    if fitter in ("lasso", "elastic_net"):
        return _sklearn_linear(A, y, fitter, reg, seed, **kw)
    if fitter == "tv":
        return _tv(A, y, w, reg, iters=kw.get("iters", 200))
    if fitter == "adam":
        return _adam(A, y, w, seed, iters=kw.get("iters", 1500), lr=kw.get("lr", 0.02))
    raise KeyError(f"unknown linear fitter '{fitter}', options: {LINEAR_FITTERS}")


def fit_nonlinear(residual_fn, x0, method="lm", bounds=None, max_iter=4000, seed=0, **kw):
    from scipy.optimize import differential_evolution, least_squares, minimize

    x0 = np.asarray(x0, dtype=float).ravel()
    if method in ("lm", "trf", "dogbox"):
        kwargs = {"method": method, "max_nfev": int(max_iter), "xtol": 1e-12, "ftol": 1e-12}
        if method != "lm" and bounds is not None:
            kwargs["bounds"] = bounds
        else:
            kwargs["method"] = "lm"
        res = least_squares(residual_fn, x0, **kwargs)
        return res.x
    objective = lambda x: float(np.sum(residual_fn(x) ** 2))
    if method == "nelder_mead":
        res = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": int(max_iter)})
        return res.x
    if method == "powell":
        res = minimize(objective, x0, method="Powell", options={"maxiter": int(max_iter)})
        return res.x
    if method == "differential_evolution":
        res = differential_evolution(
            objective, bounds or [(-1.0, 1.0)] * len(x0), maxiter=int(kw.get("de_iters", 120)), seed=seed
        )
        return res.x
    if method == "bayes_opt":
        out = bayes_opt_minimize(objective, bounds or [(-1.0, 1.0)] * len(x0), seed=seed, **kw)
        return out["x"]
    raise KeyError(f"unknown nonlinear method '{method}', options: {NONLINEAR_METHODS}")


def bayes_opt_minimize(fn, bounds, n_init=12, n_iter=30, seed=0, candidates=2048):
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    dim = bounds.shape[0]
    try:
        from scipy.stats.qmc import LatinHypercube

        sampler = LatinHypercube(d=dim, seed=rng)
        init = bounds[:, 0] + sampler.random(n_init) * (bounds[:, 1] - bounds[:, 0])
    except ImportError:
        init = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_init, dim))
    xs = [np.asarray(x, dtype=float) for x in init]
    ys = [float(fn(x)) for x in xs]
    best = int(np.argmin(ys))
    for _ in range(n_iter):
        X = np.array(xs)
        y = np.array(ys)
        lo, hi = bounds[:, 0], bounds[:, 1]
        xn = (X - lo) / np.maximum(hi - lo, 1e-12)
        span = np.maximum(xn.max(0) - xn.min(0), 1e-6)
        dist2 = ((xn[:, None, :] - xn[None, :, :]) / span) ** 2
        d2 = dist2.sum(-1)
        med = np.median(d2[np.triu_indices(len(xn), 1)]) if len(xn) > 1 else 1.0
        length = max(np.sqrt(max(med, 1e-8)), 1e-3)
        var = max(np.var(y), 1e-12)
        K = var * np.exp(-0.5 * d2 / length**2) + 1e-8 * np.eye(len(xn))
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y - y.mean()))
        except np.linalg.LinAlgError:
            alpha = np.zeros(len(xn))
        cand = rng.uniform(lo, hi, size=(candidates, dim))
        cn = (cand - lo) / np.maximum(hi - lo, 1e-12)
        cd2 = ((cn[:, None, :] - xn[None, :, :]) / span) ** 2
        k = var * np.exp(-0.5 * cd2.sum(-1) / length**2)
        mu = y.mean() + k @ alpha
        try:
            v = np.linalg.solve(L, k.T)
            sigma = np.sqrt(np.maximum(var - np.sum(v**2, axis=0), 1e-12))
        except np.linalg.LinAlgError:
            sigma = np.sqrt(var) * np.ones(len(cand))
        z = (y.min() - mu) / sigma
        from scipy.stats import norm

        ei = (y.min() - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        pick = int(np.argmax(ei))
        x_new = cand[pick]
        xs.append(x_new)
        ys.append(float(fn(x_new)))
        if ys[-1] < ys[best]:
            best = len(ys) - 1
    return {"x": xs[best], "fun": ys[best], "xs": np.array(xs), "ys": np.array(ys)}
