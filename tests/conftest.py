"""
conftest.py — shared pytest fixtures for pyhvdfa tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyhvdfa._types import Model, Layer


# ──────────────────────────────────────────────────────────────────────────────
# Canonical test models
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def simple_2layer():
    """Minimal 1-layer + halfspace model (shallow soft layer)."""
    return Model.from_layers([
        Layer(thickness=20.0,  vp=500.0,  vs=200.0,  density=1800.0),
        Layer(thickness=0.0,   vp=1800.0, vs=800.0,  density=2200.0),
    ])


@pytest.fixture(scope="session")
def model_3layer():
    """Three-layer model with a velocity inversion (classic basin model)."""
    return Model.from_layers([
        Layer(thickness=10.0,  vp=400.0,  vs=180.0,  density=1700.0),
        Layer(thickness=30.0,  vp=800.0,  vs=350.0,  density=1900.0),
        Layer(thickness=0.0,   vp=2000.0, vs=1000.0, density=2300.0),
    ])


@pytest.fixture(scope="session")
def model_kirk():
    """Kirchheimer (2004) test model from Fortran test suite.
    Layers: h, vp, vs, rho (SI)
    """
    return Model.from_layers([
        Layer(thickness=5.0,   vp=200.0,  vs=100.0,  density=1700.0),
        Layer(thickness=10.0,  vp=600.0,  vs=300.0,  density=1900.0),
        Layer(thickness=15.0,  vp=1200.0, vs=600.0,  density=2100.0),
        Layer(thickness=0.0,   vp=2500.0, vs=1200.0, density=2400.0),
    ])


@pytest.fixture(scope="session")
def freq_vector():
    """Standard test frequency vector."""
    return np.geomspace(0.2, 10.0, 50)
