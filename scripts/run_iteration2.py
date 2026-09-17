"""Iteration 2 runner: stress-test the Iteration 1 winners.

Grids (see README Iteration 2 and ``splinebench.suites``):

  i2_core        all 10 motions x 6 methods x budgets {10,20,50}, 5 seeds
  i2_noise       9 structured observation models x 5 methods, 5 seeds
  i2_timeparam   linear/chord/centripetal/accel/jerk x oracle/realistic, 5 seeds
  i2_extrap      none/early/late/middle/interp hold-out protocols, 5 seeds
  i2_adversarial adversarial clustered / near-duplicate / one-gap knots
  i2_knotcount   fixed n_sites vs CV-selected knot count vs GCV penalty

Usage:
    python scripts/run_iteration2.py            # full (5 seeds)
    python scripts/run_iteration2.py --quick    # 3 seeds
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splinebench import suites
from splinebench.experiment import run_suite

SUITES = [
    "i2_core",
    "i2_noise",
    "i2_timeparam",
    "i2_extrap",
    "i2_adversarial",
    "i2_knotcount",
]
OUT_DIR = "results/iteration2"


def main():
    quick = "--quick" in sys.argv
    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    for name in SUITES:
        conds = suites.build(name, quick=quick)
        total += len(conds)
        out = f"{OUT_DIR}/{name}.jsonl"
        print(f"=== {name}: {len(conds)} conditions -> {out} ===", flush=True)
        start = time.perf_counter()
        run_suite(conds, out_path=out, resume=True, verbose=False)
        print(f"    done in {time.perf_counter() - start:.1f}s", flush=True)
    print(f"all iteration-2 suites complete ({total} conditions)")


if __name__ == "__main__":
    main()
