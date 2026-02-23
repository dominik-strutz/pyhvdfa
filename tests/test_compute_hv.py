"""
tests/test_compute_hv.py — Integration tests for the full compute_hv() pipeline.

These are smoke/sanity tests that verify the H/V ratio has physically reasonable
behaviour.  Numerical comparisons against compiled Fortran reference output are
in tests/test_reference.py.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyhvdfa import compute_hv
from pyhvdfa._types import Model, Layer, HVResult


@pytest.fixture(scope="module")
def soft_sediment_model():
    """Shallow soft layer over stiff halfspace — should produce a clear peak."""
    return Model.from_layers([
        Layer(thickness=30.0, vp=600.0, vs=150.0, density=1800.0),
        Layer(thickness=0.0,  vp=2000.0, vs=800.0, density=2300.0),
    ])


class TestComputeHV:
    def test_returns_hvresult(self, simple_2layer):
        result = compute_hv(simple_2layer, freq_min=0.5, freq_max=10.0, n_freq=20,
                            n_modes_rayleigh=1, n_modes_love=1,
                            include_body_waves=False)
        assert isinstance(result, HVResult)

    def test_output_shapes(self, simple_2layer):
        n = 25
        result = compute_hv(simple_2layer, freq_min=0.5, freq_max=10.0, n_freq=n,
                            n_modes_rayleigh=1, n_modes_love=1,
                            include_body_waves=False)
        assert result.freq.shape == (n,)
        assert result.hv.shape   == (n,)

    def test_hv_has_finite_values(self, simple_2layer):
        """At least some H/V values must be finite — guards against all-NaN output."""
        result = compute_hv(simple_2layer, freq_min=0.5, freq_max=10.0, n_freq=20,
                            n_modes_rayleigh=1, n_modes_love=1,
                            include_body_waves=False)
        assert np.sum(np.isfinite(result.hv)) > 0, "All H/V values are NaN/Inf"

    def test_hv_non_negative(self, simple_2layer):
        result = compute_hv(simple_2layer, freq_min=0.5, freq_max=10.0, n_freq=20,
                            n_modes_rayleigh=1, n_modes_love=1,
                            include_body_waves=False)
        hv = result.hv[np.isfinite(result.hv)]
        assert len(hv) > 0, "All H/V values are NaN/Inf"
        assert np.all(hv >= 0.0), "H/V values must be non-negative"

    def test_has_peak_near_resonance(self, soft_sediment_model):
        """H/V peak should occur near the S-wave resonance f0 ≈ Vs/(4H) = 1.25 Hz."""
        result = compute_hv(soft_sediment_model,
                            freq_min=0.2, freq_max=5.0, n_freq=50,
                            n_modes_rayleigh=2, n_modes_love=2,
                            include_body_waves=False)
        hv   = result.hv
        freq = result.freq
        finite_mask = np.isfinite(hv)
        assert np.sum(finite_mask) >= 5, (
            f"Expected ≥5 finite H/V values, got {np.sum(finite_mask)}"
        )
        peak_freq = freq[np.nanargmax(hv)]
        expected  = 150.0 / (4.0 * 30.0)   # 1.25 Hz
        assert expected / 3.0 < peak_freq < expected * 3.0, (
            f"Peak at {peak_freq:.2f} Hz, expected near {expected:.2f} Hz"
        )

    def test_freq_vector_is_geomspace(self, simple_2layer):
        n = 20
        f_min, f_max = 0.5, 10.0
        result = compute_hv(simple_2layer, freq_min=f_min, freq_max=f_max, n_freq=n,
                            n_modes_rayleigh=1, n_modes_love=1,
                            include_body_waves=False)
        np.testing.assert_allclose(result.freq, np.geomspace(f_min, f_max, n),
                                   rtol=1e-12)

    def test_3layer_model(self, model_3layer):
        result = compute_hv(model_3layer, freq_min=0.5, freq_max=8.0, n_freq=15,
                            n_modes_rayleigh=1, n_modes_love=1,
                            include_body_waves=False)
        assert result.hv is not None

    def test_bad_freq_range_raises(self, simple_2layer):
        with pytest.raises(ValueError):
            compute_hv(simple_2layer, freq_min=10.0, freq_max=1.0)

    def test_bad_freq_min_raises(self, simple_2layer):
        with pytest.raises(ValueError):
            compute_hv(simple_2layer, freq_min=-1.0, freq_max=10.0)
