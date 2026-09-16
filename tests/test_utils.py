import numpy as np

from splinebench.utils import chain_rule_derivs


def test_chain_rule_matches_finite_differences():
    t = np.linspace(0.0, 1.0, 4001)
    u = t + 0.1 * np.sin(2.0 * np.pi * t)
    u_state = np.stack(
        [
            u,
            1.0 + 0.2 * np.pi * np.cos(2.0 * np.pi * t),
            -0.4 * np.pi**2 * np.sin(2.0 * np.pi * t),
            -0.8 * np.pi**3 * np.cos(2.0 * np.pi * t),
            1.6 * np.pi**4 * np.sin(2.0 * np.pi * t),
        ],
        axis=-1,
    )
    f = np.stack(
        [
            np.sin(3.0 * u) + 0.5 * u**2,
            3.0 * np.cos(3.0 * u) + u,
            -9.0 * np.sin(3.0 * u) + 1.0,
            -27.0 * np.cos(3.0 * u),
            81.0 * np.sin(3.0 * u),
        ],
        axis=-1,
    )
    composed = chain_rule_derivs(f, u_state)
    fd = [composed[..., 0]]
    cur = f[..., 0]
    for _ in range(4):
        cur = np.gradient(cur, t)
        fd.append(cur)
    for order in (1, 2, 3):
        a = composed[5:-5, order]
        b = fd[order][5:-5]
        scale = np.max(np.abs(b)) + 1e-9
        assert np.max(np.abs(a - b)) < 1e-3 * scale
