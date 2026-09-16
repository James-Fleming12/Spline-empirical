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


def test_linear_basis_is_linear_in_coeffs():
    rep = representations.build("bspline5", positions=np.linspace(0.0, 1.0, 8), dim=1)
    u = np.linspace(0.0, 1.0, 50)
    A = rep.basis_derivs(u, 0)[0]
    assert A.shape[0] == len(u)
    assert A.shape[1] == rep.n_basis
