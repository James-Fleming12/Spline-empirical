"""Spline-empirical: sample-efficiency and robustness benchmark for
trajectory spline parameterizations on synthetic snappy motion.

Experimental unit is the tuple

    (representation, knot placement, fitting, time parameterization,
     regularization, sampling strategy)

See README.md for the study design and references.bib for citations.
"""

__version__ = "0.1.0"

from . import catalog, experiment, fitters, knots, metrics, motions, representations, samplers, timeparam

__all__ = [
    "catalog",
    "experiment",
    "fitters",
    "knots",
    "metrics",
    "motions",
    "representations",
    "samplers",
    "timeparam",
    "__version__",
]
