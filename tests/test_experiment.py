import json

import numpy as np
import pytest

from splinebench import experiment, metrics, suites
from splinebench.experiment import Condition


def test_smoke_suite_runs(tmp_path):
    out = tmp_path / "smoke.jsonl"
    conditions = suites.smoke_conditions()[:3]
    records = experiment.run_suite(conditions, out_path=str(out), verbose=False)
    assert len(records) == 3
    lines = [json.loads(line) for line in out.read_text().strip().splitlines()]
    assert len(lines) == 3
    for rec in lines:
        assert np.isfinite(rec["pos_rmse"])
        assert rec["n_queries"] >= 2
        assert rec["wall_time_s"] > 0


def test_resume_skips_completed(tmp_path):
    out = tmp_path / "smoke.jsonl"
    conditions = suites.smoke_conditions()[:2]
    experiment.run_suite(conditions, out_path=str(out), verbose=False)
    records = experiment.run_suite(conditions, out_path=str(out), verbose=False)
    assert records == []


def test_adaptive_sampler_condition_runs():
    cond = Condition(
        motion="staccato",
        representation="bspline3",
        knots="split_merge",
        sampling="active_gp",
        budget=10,
        seed=0,
        n_sites=5,
        noise_std=0.005,
    )
    rec = experiment.run_condition(cond)
    assert np.isfinite(rec["pos_rmse"])
    assert rec["sampling_uses_gt"] is False


def test_summary_and_aulc(tmp_path):
    import pandas as pd

    out = tmp_path / "smoke.jsonl"
    conditions = suites.smoke_conditions()[:6]
    experiment.run_suite(conditions, out_path=str(out), verbose=False)
    df = experiment.load_results(str(out))
    assert not df.empty
    grouped = metrics.aggregate(df)
    assert len(grouped) >= 1
    curves = metrics.aulc(df)
    assert "aulc" in curves.columns
    summary = experiment.summarize(df, out_csv=str(tmp_path / "summary.csv"))
    assert (tmp_path / "summary.csv").exists()
