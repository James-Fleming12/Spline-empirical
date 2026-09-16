import numpy as np

from .utils import MAX_ORDER, monotone_map, savgol_deriv


class TimeParameterization:
    name = "linear"
    uses_ground_truth = False
    citation = ""

    def __init__(self, oracle=False, feature=None):
        self.oracle = bool(oracle)
        self.feature = feature or self.name
        self.uses_ground_truth = self.oracle
        self._map = None
        self.t_obs = None
        self.y_obs = None

    def fit(self, t, y=None, motion=None):
        t = np.asarray(t, dtype=float).ravel()
        order = np.argsort(t)
        self.t_obs = t[order]
        self.y_obs = None if y is None else np.asarray(y, dtype=float)[order]
        self._map = None
        if self.feature == "linear":
            return self
        feature = self._feature_values(motion)
        if feature is None or len(self.t_obs) < 4:
            return self
        cum = np.concatenate([[0.0], np.cumsum(feature)])
        if len(cum) == len(self.t_obs) + 1:
            cum = cum[:-1]
        self._map = monotone_map(self.t_obs, cum)
        return self

    def _feature_values(self, motion):
        t = self.t_obs
        if self.feature in ("chord", "centripetal"):
            if motion is not None and self.oracle:
                y = motion.eval(t)
            elif self.y_obs is not None:
                y = self.y_obs
            else:
                return None
            step = np.linalg.norm(np.diff(y, axis=0), axis=1)
            if self.feature == "centripetal":
                step = np.sqrt(np.maximum(step, 0.0))
            return step
        order = 2 if self.feature == "accel" else 3
        if motion is not None and self.oracle:
            derivs = np.abs(motion.eval_state(t, order=order)[..., order]).sum(axis=1)
        elif self.y_obs is not None:
            derivs = np.abs(savgol_deriv(t, self.y_obs, order=order)).sum(axis=1)
        else:
            return None
        return derivs + 1e-9

    def to_u(self, t):
        t = np.asarray(t, dtype=float).ravel()
        out = np.zeros((len(t), MAX_ORDER + 1))
        if self._map is None:
            out[:, 0] = np.clip(t, 0.0, 1.0)
            out[:, 1] = 1.0
            return out
        out[:, 0] = np.clip(self._map(np.clip(t, self.t_obs[0], self.t_obs[-1])), 0.0, 1.0)
        for k in range(1, MAX_ORDER + 1):
            out[:, k] = self._map.derivative(k)(np.clip(t, self.t_obs[0], self.t_obs[-1]))
        return out


class LinearTime(TimeParameterization):
    name = "linear"

    def __init__(self, oracle=False):
        super().__init__(oracle=False, feature="linear")


class ChordLengthTime(TimeParameterization):
    name = "chord"

    def __init__(self, oracle=False):
        super().__init__(oracle=oracle, feature="chord")


class CentripetalTime(TimeParameterization):
    name = "centripetal"

    def __init__(self, oracle=False):
        super().__init__(oracle=oracle, feature="centripetal")


class AccelerationTime(TimeParameterization):
    name = "accel"

    def __init__(self, oracle=False):
        super().__init__(oracle=oracle, feature="accel")


class JerkTime(TimeParameterization):
    name = "jerk"

    def __init__(self, oracle=False):
        super().__init__(oracle=oracle, feature="jerk")


TIME_PARAMS = {
    "linear": LinearTime,
    "chord": ChordLengthTime,
    "centripetal": CentripetalTime,
    "accel": AccelerationTime,
    "jerk": JerkTime,
}


def list_time_params():
    return sorted(TIME_PARAMS)


def build(name, **kw):
    if name not in TIME_PARAMS:
        raise KeyError(f"unknown time parameterization '{name}', options: {list_time_params()}")
    return TIME_PARAMS[name](**kw)
