import numpy as np

from .utils import MAX_ORDER

DEFAULT_DIM = 3


class Primitive:
    name = "primitive"
    citation = ""

    def derivs(self, t, order=MAX_ORDER):
        raise NotImplementedError


class Sine(Primitive):
    name = "sine"
    citation = ""

    def __init__(self, freq=1.0, amp=1.0, phase=0.0):
        self.freq = float(freq)
        self.amp = float(amp)
        self.phase = float(phase)

    def derivs(self, t, order=MAX_ORDER):
        w = 2.0 * np.pi * self.freq
        p = w * np.asarray(t, dtype=float) + self.phase
        s, c = np.sin(p), np.cos(p)
        vals = [s, w * c, -(w**2) * s, -(w**3) * c, (w**4) * s]
        return self.amp * np.stack(vals[: order + 1], axis=-1)


class Chirp(Primitive):
    name = "chirp"
    citation = ""

    def __init__(self, f0=1.0, f1=10.0, amp=1.0, phase=0.0):
        self.f0 = float(f0)
        self.f1 = float(f1)
        self.amp = float(amp)
        self.phase = float(phase)

    def derivs(self, t, order=MAX_ORDER):
        t = np.asarray(t, dtype=float)
        two = 2.0 * np.pi
        d = self.f1 - self.f0
        phi = two * (self.f0 * t + 0.5 * d * t * t) + self.phase
        d1 = two * (self.f0 + d * t)
        d2 = two * d * np.ones_like(t)
        s, c = np.sin(phi), np.cos(phi)
        vals = [
            s,
            c * d1,
            -s * d1**2 + c * d2,
            -c * d1**3 - 3.0 * s * d1 * d2,
            s * d1**4 - 6.0 * c * d1**2 * d2 - 3.0 * s * d2**2,
        ]
        return self.amp * np.stack(vals[: order + 1], axis=-1)


class MinJerkStep(Primitive):
    name = "minjerk_step"
    citation = "hogan1984"

    def __init__(self, t0=0.0, t1=1.0, amp=1.0):
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.amp = float(amp)

    def derivs(self, t, order=MAX_ORDER):
        t = np.asarray(t, dtype=float)
        span = max(self.t1 - self.t0, 1e-12)
        u = np.clip((t - self.t0) / span, 0.0, 1.0)
        vals = [
            10 * u**3 - 15 * u**4 + 6 * u**5,
            30 * u**2 - 60 * u**3 + 30 * u**4,
            60 * u - 180 * u**2 + 120 * u**3,
            60 - 360 * u + 360 * u**2,
            -360 + 720 * u,
        ]
        scale = span ** (-np.arange(order + 1))
        f = np.stack(vals[: order + 1], axis=-1) * scale
        mask = ((t >= self.t0) & (t <= self.t1))[..., None]
        return self.amp * np.where(mask, f, 0.0)


class BangBang(Primitive):
    name = "bang_bang"
    citation = ""

    def __init__(self, t0=0.0, t1=1.0, amp=1.0, accel_frac=0.25):
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.amp = float(amp)
        self.accel_frac = float(accel_frac)

    def derivs(self, t, order=MAX_ORDER):
        t = np.asarray(t, dtype=float)
        T = max(self.t1 - self.t0, 1e-12)
        af = min(max(self.accel_frac, 1e-3), 0.49)
        Ta = af * T
        Td = Ta
        Tc = max(T - Ta - Td, 0.0)
        vpk = self.amp / max(0.5 * Ta + Tc + 0.5 * Td, 1e-12)
        a1 = vpk / Ta
        a2 = -vpk / Td
        s = t - self.t0
        y = np.zeros_like(t)
        v = np.zeros_like(t)
        a = np.zeros_like(t)
        m1 = (s > 0) & (s <= Ta)
        y[m1] = 0.5 * a1 * s[m1] ** 2
        v[m1] = a1 * s[m1]
        a[m1] = a1
        y1 = 0.5 * a1 * Ta**2
        m2 = (s > Ta) & (s <= Ta + Tc)
        y[m2] = y1 + vpk * (s[m2] - Ta)
        v[m2] = vpk
        y2 = y1 + vpk * Tc
        m3 = (s > Ta + Tc) & (s <= T)
        sm = s[m3] - (Ta + Tc)
        y[m3] = y2 + vpk * sm + 0.5 * a2 * sm**2
        v[m3] = vpk + a2 * sm
        a[m3] = a2
        z = np.zeros_like(t)
        vals = [y, v, a, z, z]
        return np.stack(vals[: order + 1], axis=-1)


class Impulse(Primitive):
    name = "impulse"
    citation = ""

    def __init__(self, t0=0.0, amp=1.0, decay=20.0):
        self.t0 = float(t0)
        self.amp = float(amp)
        self.decay = float(decay)

    def derivs(self, t, order=MAX_ORDER):
        t = np.asarray(t, dtype=float)
        tau = t - self.t0
        d = self.decay
        e = np.exp(-d * np.maximum(tau, 0.0))
        vals = [
            tau * e,
            e * (1.0 - d * tau),
            e * (d * d * tau - 2.0 * d),
            e * (-(d**3) * tau + 3.0 * d * d),
            e * (d**4 * tau - 4.0 * d**3),
        ]
        out = self.amp * np.stack(vals[: order + 1], axis=-1)
        return np.where((tau >= 0)[..., None], out, 0.0)


class DampedStep(Primitive):
    name = "damped_step"
    citation = ""

    def __init__(self, t0=0.0, amp=1.0, omega=30.0, zeta=0.3):
        self.t0 = float(t0)
        self.amp = float(amp)
        self.omega = float(omega)
        self.zeta = float(zeta)
        self.sigma = self.zeta * self.omega
        self.wd = self.omega * np.sqrt(max(1.0 - self.zeta**2, 1e-6))
        self.B = self.sigma / self.wd

    def derivs(self, t, order=MAX_ORDER):
        t = np.asarray(t, dtype=float)
        tau = t - self.t0
        lam = complex(-self.sigma, self.wd)
        C = complex(1.0, -self.B)
        z = np.exp(lam * np.maximum(tau, 0.0))
        out = np.empty((len(t), order + 1), dtype=float)
        out[:, 0] = (1.0 - C * z).real
        for k in range(1, order + 1):
            out[:, k] = -(C * (lam**k) * z).real
        out = self.amp * out
        return np.where((tau >= 0)[:, None], out, 0.0)


class GaussianBump(Primitive):
    name = "gaussian_bump"
    citation = ""

    def __init__(self, t0=0.5, width=0.05, amp=1.0):
        self.t0 = float(t0)
        self.width = float(width)
        self.amp = float(amp)

    def derivs(self, t, order=MAX_ORDER):
        from numpy.polynomial.hermite import hermval

        t = np.asarray(t, dtype=float)
        w = self.width
        x = (t - self.t0) / w
        e = np.exp(-(x**2))
        out = []
        for k in range(order + 1):
            c = np.zeros(k + 1)
            c[k] = 1.0
            out.append(((-1.0) ** k) * (w ** (-k)) * hermval(x, c) * e)
        return self.amp * np.stack(out, axis=-1)


class Component:
    def __init__(self, axis, prim, scale=1.0):
        self.axis = int(axis)
        self.prim = prim
        self.scale = float(scale)


class Motion:
    def __init__(self, name, components, dim=DEFAULT_DIM, window=(0.0, 1.0)):
        self.name = name
        self.components = list(components)
        self.dim = int(dim)
        self.window = (float(window[0]), float(window[1]))

    def eval_state(self, t, order=MAX_ORDER):
        t = np.asarray(t, dtype=float).ravel()
        out = np.zeros((len(t), self.dim, order + 1))
        for comp in self.components:
            out[:, comp.axis, :] += comp.scale * comp.prim.derivs(t, order)
        return out

    def eval(self, t, order=0):
        return self.eval_state(t, order=order)[..., 0]

    def max_abs(self, order=0, grid=2001):
        t = np.linspace(self.window[0], self.window[1], grid)
        return float(np.abs(self.eval_state(t, order)).max())

    def scale(self, factor):
        for comp in self.components:
            comp.scale /= factor
        return self


def _normalize(motion):
    peak = motion.max_abs(order=0)
    if peak > 0:
        motion.scale(peak)
    return motion


def _orbit(**kw):
    comps = [
        Component(0, Sine(freq=1.0, amp=0.8)),
        Component(1, Sine(freq=4.0, amp=0.15)),
        Component(2, Sine(freq=1.0, amp=0.8, phase=-np.pi / 2)),
    ]
    return _normalize(Motion("orbit", comps))


def _staccato(**kw):
    comps = [
        Component(0, MinJerkStep(0.25, 0.35, 0.85)),
        Component(0, MinJerkStep(0.60, 0.70, -0.85)),
        Component(1, Sine(freq=0.5, amp=0.10)),
    ]
    return _normalize(Motion("staccato", comps))


def _double_step(**kw):
    comps = [
        Component(0, MinJerkStep(0.30, 0.36, 0.9)),
        Component(0, MinJerkStep(0.42, 0.48, -0.9)),
        Component(2, MinJerkStep(0.36, 0.42, 0.4)),
    ]
    return _normalize(Motion("double_step", comps))


def _bang_bang(**kw):
    comps = [
        Component(0, BangBang(0.10, 0.40, 1.0, accel_frac=0.2)),
        Component(0, BangBang(0.55, 0.90, -1.0, accel_frac=0.2)),
        Component(2, Sine(freq=0.5, amp=0.1)),
    ]
    return _normalize(Motion("bang_bang", comps))


def _bounce(**kw):
    times = [0.15, 0.30, 0.45, 0.60, 0.75]
    amps = [1.0, 0.8, 0.62, 0.48, 0.36]
    comps = [Component(0, Sine(freq=0.5, amp=0.15))]
    for i, (t0, a) in enumerate(zip(times, amps)):
        sign = 1.0 if i % 2 == 0 else -1.0
        comps.append(Component(1, DampedStep(t0=t0, amp=sign * a, omega=55.0, zeta=0.18)))
    return _normalize(Motion("bounce", comps))


def _wobble(**kw):
    f = 18.0
    comps = [
        Component(0, Sine(freq=f, amp=0.5)),
        Component(1, Sine(freq=f, amp=0.5, phase=0.7)),
        Component(2, Sine(freq=f / 2.0, amp=0.3)),
    ]
    return _normalize(Motion("wobble", comps))


def _chirp(**kw):
    comps = [
        Component(0, Chirp(f0=1.0, f1=20.0, amp=0.7)),
        Component(1, Chirp(f0=20.0, f1=2.0, amp=0.5)),
        Component(2, Chirp(f0=5.0, f1=15.0, amp=0.3, phase=0.5)),
    ]
    return _normalize(Motion("chirp", comps))


def _pulses(**kw):
    comps = [
        Component(0, GaussianBump(0.20, 0.020, 1.0)),
        Component(0, GaussianBump(0.50, 0.008, -0.9)),
        Component(1, GaussianBump(0.80, 0.030, 0.7)),
        Component(2, GaussianBump(0.35, 0.015, 0.5)),
    ]
    return _normalize(Motion("pulses", comps))


def _combo(**kw):
    comps = [
        Component(0, MinJerkStep(0.15, 0.22, 0.8)),
        Component(0, MinJerkStep(0.65, 0.72, -0.8)),
        Component(0, Sine(freq=6.0, amp=0.15)),
        Component(1, Chirp(f0=2.0, f1=16.0, amp=0.5)),
        Component(2, DampedStep(t0=0.40, amp=0.8, omega=60.0, zeta=0.12)),
        Component(2, DampedStep(t0=0.55, amp=-0.5, omega=80.0, zeta=0.15)),
    ]
    return _normalize(Motion("combo", comps))


def _contact_drop(**kw):
    comps = [
        Component(1, DampedStep(t0=0.10, amp=1.0, omega=25.0, zeta=0.35)),
        Component(1, Impulse(t0=0.55, amp=0.4, decay=30.0)),
        Component(0, Sine(freq=0.5, amp=0.1)),
    ]
    return _normalize(Motion("contact_drop", comps))


_BUILDERS = {
    "orbit": _orbit,
    "staccato": _staccato,
    "double_step": _double_step,
    "bang_bang": _bang_bang,
    "bounce": _bounce,
    "wobble": _wobble,
    "chirp": _chirp,
    "pulses": _pulses,
    "combo": _combo,
    "contact_drop": _contact_drop,
}

SMOOTH_MOTIONS = ["orbit"]
SNAPPY_MOTIONS = ["staccato", "double_step", "bang_bang", "bounce", "contact_drop", "pulses"]
HIGH_FREQUENCY_MOTIONS = ["wobble", "chirp", "combo"]

CITATIONS = {
    "hogan1984": "Hogan, N. (1984). An organizing principle for a class of voluntary movements. J. Neuroscience.",
}


def list_motions():
    return sorted(_BUILDERS)


def build(name, **kw):
    if name not in _BUILDERS:
        raise KeyError(f"unknown motion '{name}', options: {list_motions()}")
    return _BUILDERS[name](**kw)
