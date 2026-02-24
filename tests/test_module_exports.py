"""
tests/test_module_exports.py — Smoke-test that public symbols can be imported.

Any time a symbol is renamed or removed, this test fails at collection time
with a clear ImportError or AttributeError rather than a cryptic runtime error.
"""

from __future__ import annotations

import importlib

import pytest

_EXPECTED_EXPORTS: dict[str, list[str]] = {
    # ── public API ──────────────────────────────────────────────────────────
    "pyhvdfa": [
        "compute_hv",
        "Model",
        "Layer",
        "HVResult",
    ],
    # ── _types ──────────────────────────────────────────────────────────────
    "pyhvdfa._types": [
        "FLOAT",
        "CMPLX",
        "I_UNIT",
        "PI",
        "TWO_PI",
        "Layer",
        "Model",
        "HVResult",
    ],
}


@pytest.mark.parametrize(
    "module_name,symbols",
    [pytest.param(mod, syms, id=mod) for mod, syms in _EXPECTED_EXPORTS.items()],
)
def test_module_importable(module_name: str, symbols: list[str]) -> None:
    """The module itself must import without errors."""
    importlib.import_module(module_name)


@pytest.mark.parametrize(
    "module_name,symbol",
    [
        pytest.param(mod, sym, id=f"{mod}.{sym}")
        for mod, syms in _EXPECTED_EXPORTS.items()
        for sym in syms
    ],
)
def test_symbol_accessible(module_name: str, symbol: str) -> None:
    """Every listed symbol must exist in its module after import."""
    mod = importlib.import_module(module_name)
    assert hasattr(mod, symbol), (
        f"'{symbol}' not found in '{module_name}'. "
        "If it was intentionally removed, update _EXPECTED_EXPORTS in "
        "tests/test_module_exports.py."
    )


def test_hv_core_extension_importable() -> None:
    """The compiled Fortran extension _hv_core must be importable."""
    import pyhvdfa._hv_core as core  # noqa: F401

    assert hasattr(core, "compute_hv"), (
        "_hv_core.compute_hv not found — was the Fortran extension compiled? "
        "Run: uv pip install -e . --no-build-isolation"
    )
