import pathlib
import re

import pytest

from splinebench import catalog


@pytest.mark.parametrize("axis", ["representations", "knots", "fitters", "sampling", "time_param"])
def test_implemented_catalog_entries_resolve(axis):
    registry = catalog.registry_methods(axis)
    missing = set(catalog.implemented(axis)) - registry
    assert not missing, f"{axis} implemented but not in registry: {sorted(missing)}"


@pytest.mark.parametrize("axis", ["representations", "knots", "sampling", "time_param"])
def test_planned_catalog_entries_are_not_registered(axis):
    registry = catalog.registry_methods(axis)
    overlap = set(catalog.planned(axis)) & registry
    assert not overlap, f"{axis} planned but registered: {sorted(overlap)}"


def test_citations_well_formed():
    for axis, entries in catalog.CATALOG.items():
        for name, (status, cite) in entries.items():
            assert status in ("implemented", "planned"), (axis, name, status)
            assert isinstance(cite, str)


def test_citation_keys_resolve_in_bib():
    bib = pathlib.Path(__file__).resolve().parents[1] / "references.bib"
    keys = set(re.findall(r"@\w+\{([^,]+),", bib.read_text()))
    used = {cite for entries in catalog.CATALOG.values() for _, cite in entries.values() if cite}
    assert not (used - keys), f"missing bib entries: {sorted(used - keys)}"
