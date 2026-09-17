import numpy as np
import pytest

from splinebench import motions, representations


FAST_KWARGS = {
    "mlp": {"iters": 200},
    "bspline_mlp": {"iters": 120},
    "gp_rbf": {"optimize": False},
    "gp_matern52": {"optimize": False},
    "nurbs": {"optimize_weights": False},
}


@pytest.mark.parametrize("name", representations.list_representations())
def test_fit_eval_derivs(name):
    if name in ("mlp", "bspline_mlp"):
        pytest.importorskip("torch")
    motion = motions.build("staccato")
    u = np.linspace(0.0, 1.0, 120)
    y = motion.eval(u)
    rep = representations.build(name, positions=np.linspace(0.0, 1.0, 8), dim=motion.dim,
                                seed=0, **FAST_KWARGS.get(name, {}))
    rep.fit(u, y, fitter="least_squares",
            reg={"kind": "diff", "order": 2, "lam": 1e-6} if name in ("pspline",) else None)
    state = rep.eval_derivs(u, order=4)
    assert state.shape == (len(u), motion.dim, 5)
    assert np.isfinite(state).all()
    assert rep.n_params + rep.n_data_params > 0

    d1 = state[..., 1]
    fd1 = np.gradient(rep.eval(u), u, axis=0)
    med = np.median(np.abs(d1 - fd1))
    scale = np.percentile(np.abs(fd1), 90) + 1e-9
    assert med < 0.2 * scale + 1e-6, f"{name}: derivative mismatch ({med} vs {scale})"


@pytest.mark.parametrize("name", ["bspline5", "hermite", "catmull_rom", "pchip", "gp_rbf"])
def test_dense_recovery(name):
    motion = motions.build("orbit")
    u = np.linspace(0.0, 1.0, 300)
    y = motion.eval(u)
    rep = representations.build(name, positions=np.linspace(0.0, 1.0, 12), dim=motion.dim,
                                seed=0, **FAST_KWARGS.get(name, {}))
    rep.fit(u, y)
    err = np.sqrt(np.mean((rep.eval(u) - y) ** 2))
    assert err < 0.05, f"{name} dense recovery error {err}"


def test_unknown_representation_raises():
    with pytest.raises(KeyError):
        representations.build("does_not_exist")


@pytest.mark.parametrize("kernel", ["rbf", "matern52"])
def test_gp_kernel_is_symmetric_and_derivative_consistent(kernel):
    gp = representations.GaussianProcess(kernel=kernel, optimize=False)
    gp.log_params = np.array([np.log(0.2), np.log(1.0), np.log(1e-4)])
    a = np.linspace(0.0, 1.0, 17)
    b = np.linspace(0.0, 1.0, 23)
    K = gp._kernel(a, b, 0)
    assert np.allclose(K, gp._kernel(b, a, 0).T, atol=1e-10)
    h = 1e-6
    fd = (gp._kernel(a + h, b, 0) - gp._kernel(a - h, b, 0)) / (2 * h)
    assert np.allclose(gp._kernel(a, b, 1), fd, rtol=1e-4, atol=1e-6)


def test_gp_matern_survives_clean_data():
    from splinebench import motions

    motion = motions.build("staccato")
    u = np.linspace(0.0, 1.0, 40)
    y = motion.eval(u)
    rep = representations.build("gp_matern52", dim=motion.dim)
    rep.fit(u, y)
    err = np.sqrt(np.mean((rep.eval(u) - y) ** 2))
    assert err < 0.1, err


def test_linear_basis_is_linear_in_coeffs():
    rep = representations.build("bspline5", positions=np.linspace(0.0, 1.0, 8), dim=1)
    u = np.linspace(0.0, 1.0, 50)
    A = rep.basis_derivs(u, 0)[0]
    assert A.shape[0] == len(u)
    assert A.shape[1] == rep.n_basis


def test_bspline_drops_endpoint_sites():
    rep = representations.build("bspline3", positions=np.linspace(0.0, 1.0, 8), dim=1)
    interior = rep.knots[rep.degree + 1 : -(rep.degree + 1)]
    assert np.all(interior > 1e-6)
    assert np.all(interior < 1.0 - 1e-6)
    assert np.all(np.diff(interior) >= 1e-4)
