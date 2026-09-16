import numpy as np
import pytest

from splinebench import knots, motions


def _context(name):
    motion = motions.build("staccato")
    t = np.linspace(0.0, 1.0, 80)
    y = motion.eval(t)
    rng = np.random.default_rng(0)
    y = y + rng.normal(0, 0.005, size=y.shape)

    def factory(pos):
        from splinebench.representations import BSpline

        rep = BSpline(pos, dim=motion.dim, degree=3)
        rep.fit(t, y)
        return rep

    return knots.KnotContext(u=t, y=y, n=9, rng=rng, motion=motion, oracle=False, rep_factory=factory)


@pytest.mark.parametrize("name", knots.list_knot_placers())
def test_placer_returns_sorted_positions(name):
    ctx = _context(name)
    placer = knots.build(name)
    pos = placer.place(ctx)
    assert len(pos) >= 2
    assert np.all(np.diff(pos) > 0)
    assert pos[0] >= -1e-9 and pos[-1] <= 1.0 + 1e-9
    assert set(np.round(np.unique(pos), 9)) == set(np.round(pos, 9))


def test_uniform_exact():
    ctx = _context("uniform")
    pos = knots.build("uniform").place(ctx)
    assert np.allclose(pos, np.linspace(0.0, 1.0, ctx.n))


def test_curvature_concentrates_near_snap():
    motion = motions.build("staccato")
    t = np.linspace(0.0, 1.0, 400)
    y = motion.eval(t)
    ctx = knots.KnotContext(u=t, y=y, n=12, rng=np.random.default_rng(0), motion=motion, oracle=True)
    pos = knots.build("curvature", oracle=True).place(ctx)
    near_snap = np.sum((pos > 0.2) & (pos < 0.75))
    uniform_near = np.sum((np.linspace(0, 1, 12) > 0.2) & (np.linspace(0, 1, 12) < 0.75))
    assert near_snap >= uniform_near


def test_unknown_placer_raises():
    with pytest.raises(KeyError):
        knots.build("does_not_exist")
