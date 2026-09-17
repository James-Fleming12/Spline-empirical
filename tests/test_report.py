import numpy as np
import pandas as pd
import pytest

from splinebench import report
from splinebench.experiment import Condition


def test_fmt_and_ms():
    assert report._fmt(float("nan")) == "—"
    assert report._ms([1.0]) == "1"
    assert "±" in report._ms([1.0, 3.0])


def test_sampling_table_marks_best():
    rows = []
    for sampler, base in (("random", 0.5), ("lhs", 0.2), ("sobol", 0.3)):
        for budget, scale in ((10, 1.0), (20, 0.5), (50, 0.25)):
            for seed in (0, 1):
                rows.append(
                    {
                        "motion": "staccato",
                        "representation": "bspline5",
                        "knots": "curvature",
                        "budget": budget,
                        "seed": seed,
                        "sampling": sampler,
                        "n_sites": 8,
                        "pos_rmse": base * scale,
                    }
                )
    df = pd.DataFrame(rows)
    table = report.table_sampling(df)
    assert "lhs" in table
    assert "**" in table


def test_patch_readme_roundtrip(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# T\n\n<!-- AUTO:6.1 START -->\nOLD\n<!-- AUTO:6.1 END -->\n")
    report.patch_readme(str(readme), {"6.1": "| a |\n|---|\n| 1 |"})
    text = readme.read_text()
    assert "| a |" in text and "OLD" not in text


def test_facts_runs_on_smoke_records(tmp_path):
    from splinebench import experiment, suites

    out = tmp_path / "r.jsonl"
    experiment.run_suite(suites.smoke_conditions()[:2], out_path=str(out), verbose=False)
    df = report.load_results(str(out))
    text = report.facts(df)
    assert "6.1" in text
