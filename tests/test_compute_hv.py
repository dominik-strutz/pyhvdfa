"""
tests/test_compute_hv.py — Integration tests for the full compute_hv() pipeline.

These are smoke/sanity tests that verify the H/V ratio has physically reasonable
behaviour.  Numerical comparisons against compiled Fortran reference output are
in tests/test_reference.py.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import pytest

from pyhvdfa import compute_hv
from pyhvdfa import compute_hv_batch
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


class TestComputeHVBatch:
    """Tests for the thread-parallel compute_hv_batch function."""

    @pytest.fixture()
    def four_models(self):
        """Four models with different layer thicknesses."""
        return [
            Model.from_layers([
                Layer(thickness=h, vp=600.0, vs=200.0, density=1800.0),
                Layer(thickness=0.0, vp=2000.0, vs=800.0, density=2300.0),
            ])
            for h in [10.0, 20.0, 30.0, 40.0]
        ]

    def test_batch_returns_list(self, four_models):
        results = compute_hv_batch(
            four_models, n_workers=2,
            freq_min=0.5, freq_max=10.0, n_freq=20,
            n_modes_rayleigh=1, n_modes_love=1,
            include_body_waves=False,
        )
        assert isinstance(results, list)
        assert len(results) == 4
        for r in results:
            assert isinstance(r, HVResult)

    def test_batch_matches_sequential(self, four_models):
        """Each batch result must be identical to the corresponding single call."""
        kwargs = dict(
            freq_min=0.5, freq_max=10.0, n_freq=30,
            n_modes_rayleigh=2, n_modes_love=2,
            include_body_waves=False,
        )
        sequential = [compute_hv(m, **kwargs) for m in four_models]
        batch = compute_hv_batch(four_models, n_workers=4, **kwargs)
        for seq, bat in zip(sequential, batch):
            np.testing.assert_allclose(bat.freq, seq.freq, rtol=1e-12)
            np.testing.assert_allclose(bat.hv, seq.hv, rtol=1e-12)

    def test_batch_preserves_order(self, four_models):
        """Results must correspond to input order, not completion order."""
        results = compute_hv_batch(
            four_models, n_workers=4,
            freq_min=0.5, freq_max=10.0, n_freq=20,
            n_modes_rayleigh=1, n_modes_love=1,
            include_body_waves=False,
        )
        # Thicker layers → lower peak frequency
        peaks = [r.freq[np.nanargmax(r.hv)] for r in results]
        assert peaks == sorted(peaks, reverse=True), (
            f"Peak frequencies should decrease with increasing thickness: {peaks}"
        )

    def test_batch_empty_input(self):
        assert compute_hv_batch([]) == []

    def test_batch_single_model(self, simple_2layer):
        results = compute_hv_batch(
            [simple_2layer], n_workers=1,
            freq_min=0.5, freq_max=10.0, n_freq=20,
            n_modes_rayleigh=1, n_modes_love=1,
            include_body_waves=False,
        )
        assert len(results) == 1
        assert isinstance(results[0], HVResult)

    def test_batch_with_body_waves(self, four_models):
        """Verify batch works with body-wave integrals enabled."""
        results = compute_hv_batch(
            four_models, n_workers=2,
            freq_min=0.5, freq_max=10.0, n_freq=15,
            n_modes_rayleigh=2, n_modes_love=2,
            include_body_waves=True, nks=50,
        )
        assert len(results) == 4
        for r in results:
            assert np.sum(np.isfinite(r.hv)) > 0

    def test_batch_omp_env_restored(self, four_models):
        """OMP_NUM_THREADS must be restored to its original value after batch."""
        original = os.environ.get("OMP_NUM_THREADS")
        sentinel = "42"
        os.environ["OMP_NUM_THREADS"] = sentinel
        try:
            compute_hv_batch(
                four_models, n_workers=2,
                freq_min=0.5, freq_max=10.0, n_freq=10,
                n_modes_rayleigh=1, n_modes_love=0,
                include_body_waves=False,
            )
            assert os.environ.get("OMP_NUM_THREADS") == sentinel
        finally:
            if original is None:
                os.environ.pop("OMP_NUM_THREADS", None)
            else:
                os.environ["OMP_NUM_THREADS"] = original

    def test_batch_omp_env_restored_when_unset(self, simple_2layer):
        """OMP_NUM_THREADS removed after batch if it was not set before."""
        original = os.environ.pop("OMP_NUM_THREADS", None)
        try:
            compute_hv_batch(
                [simple_2layer], n_workers=1,
                freq_min=0.5, freq_max=10.0, n_freq=10,
                n_modes_rayleigh=1, n_modes_love=0,
                include_body_waves=False,
            )
            assert "OMP_NUM_THREADS" not in os.environ
        finally:
            if original is not None:
                os.environ["OMP_NUM_THREADS"] = original

    def test_batch_many_models(self):
        """Batch of 16 models — tests higher concurrency."""
        models = [
            Model.from_layers([
                Layer(thickness=float(h), vp=600.0, vs=200.0, density=1800.0),
                Layer(thickness=0.0, vp=2000.0, vs=800.0, density=2300.0),
            ])
            for h in range(5, 85, 5)  # 16 models
        ]
        results = compute_hv_batch(
            models, n_workers=4,
            freq_min=0.5, freq_max=10.0, n_freq=20,
            n_modes_rayleigh=2, n_modes_love=2,
            include_body_waves=False,
        )
        assert len(results) == 16
        for r in results:
            assert isinstance(r, HVResult)
            assert r.freq.shape == (20,)
            assert np.sum(np.isfinite(r.hv)) > 0

    def test_batch_mixed_layer_counts(self):
        """Batch with models of different numbers of layers."""
        m2 = Model.from_layers([
            Layer(thickness=20.0, vp=500.0, vs=200.0, density=1800.0),
            Layer(thickness=0.0,  vp=1800.0, vs=800.0, density=2200.0),
        ])
        m3 = Model.from_layers([
            Layer(thickness=10.0, vp=400.0, vs=180.0, density=1700.0),
            Layer(thickness=30.0, vp=800.0, vs=350.0, density=1900.0),
            Layer(thickness=0.0,  vp=2000.0, vs=1000.0, density=2300.0),
        ])
        m4 = Model.from_layers([
            Layer(thickness=5.0,  vp=200.0, vs=100.0, density=1700.0),
            Layer(thickness=10.0, vp=600.0, vs=300.0, density=1900.0),
            Layer(thickness=15.0, vp=1200.0, vs=600.0, density=2100.0),
            Layer(thickness=0.0,  vp=2500.0, vs=1200.0, density=2400.0),
        ])
        results = compute_hv_batch(
            [m2, m3, m4], n_workers=3,
            freq_min=0.5, freq_max=10.0, n_freq=25,
            n_modes_rayleigh=2, n_modes_love=2,
            include_body_waves=False,
        )
        assert len(results) == 3
        for r in results:
            assert r.hv.shape == (25,)

    def test_batch_repeated_calls_stable(self, four_models):
        """Calling compute_hv_batch multiple times gives identical results."""
        kwargs = dict(
            n_workers=2,
            freq_min=0.5, freq_max=10.0, n_freq=20,
            n_modes_rayleigh=1, n_modes_love=1,
            include_body_waves=False,
        )
        results_a = compute_hv_batch(four_models, **kwargs)
        results_b = compute_hv_batch(four_models, **kwargs)
        for a, b in zip(results_a, results_b):
            np.testing.assert_array_equal(a.hv, b.hv)
            np.testing.assert_array_equal(a.freq, b.freq)

    def test_batch_n_workers_1_matches_sequential(self, four_models):
        """n_workers=1 must produce identical results to sequential calls."""
        kwargs = dict(
            freq_min=0.5, freq_max=10.0, n_freq=30,
            n_modes_rayleigh=2, n_modes_love=2,
            include_body_waves=False,
        )
        sequential = [compute_hv(m, **kwargs) for m in four_models]
        batch = compute_hv_batch(four_models, n_workers=1, **kwargs)
        for seq, bat in zip(sequential, batch):
            np.testing.assert_allclose(bat.hv, seq.hv, rtol=1e-12)
