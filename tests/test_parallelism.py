"""
tests/test_parallelism.py — tests for Fortran-native OpenMP parallelism.

Covers:
  - compute_hv(n_workers=N): body-wave integral loop parallelism
  - compute_hv_batch(n_workers=N): model-level parallelism via COMPUTE_HV_BATCH

All numerical comparisons use rtol=1e-12 (bit-identical output expected because
the only change between n_workers=1 and N>1 is the loop scheduling, not
the arithmetic).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyhvdfa import compute_hv, compute_hv_batch
from pyhvdfa._types import HVResult, Layer, Model

RTOL = 1e-12

# ── shared kwargs ──────────────────────────────────────────────────────────────
KW_SW = dict(
    freq_min=0.5,
    freq_max=10.0,
    n_freq=40,
    n_modes_rayleigh=3,
    n_modes_love=3,
    include_body_waves=False,
)

KW_BW = dict(
    freq_min=0.5,
    freq_max=10.0,
    n_freq=40,
    n_modes_rayleigh=3,
    n_modes_love=3,
    include_body_waves=True,
    nks=64,
)


# ── shared models ──────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def m2():
    return Model.from_layers([
        Layer(thickness=30.0, vp=600.0, vs=200.0, density=1800.0),
        Layer(thickness=0.0,  vp=2000.0, vs=800.0, density=2300.0),
    ])


@pytest.fixture(scope="module")
def m3():
    return Model.from_layers([
        Layer(thickness=10.0, vp=400.0, vs=180.0, density=1700.0),
        Layer(thickness=30.0, vp=800.0, vs=350.0, density=1900.0),
        Layer(thickness=0.0,  vp=2000.0, vs=1000.0, density=2300.0),
    ])


@pytest.fixture(scope="module")
def m4():
    return Model.from_layers([
        Layer(thickness=5.0,  vp=200.0, vs=100.0, density=1700.0),
        Layer(thickness=10.0, vp=600.0, vs=300.0, density=1900.0),
        Layer(thickness=15.0, vp=1200.0, vs=600.0, density=2100.0),
        Layer(thickness=0.0,  vp=2500.0, vs=1200.0, density=2400.0),
    ])


# ==============================================================================
# compute_hv — single-model BWR loop parallelism
# ==============================================================================
class TestComputeHVWorkers:
    """compute_hv(n_workers=N) must produce identical output to n_workers=1."""

    @pytest.mark.parametrize("n_workers", [2, 4])
    def test_bw_matches_serial(self, m2, n_workers):
        ref = compute_hv(m2, n_workers=1, **KW_BW)
        par = compute_hv(m2, n_workers=n_workers, **KW_BW)
        assert not np.any(np.isnan(par.hv)), f"NaN in output for n_workers={n_workers}"
        np.testing.assert_allclose(par.hv, ref.hv, rtol=RTOL,
                                   err_msg=f"n_workers={n_workers} differs from serial")

    @pytest.mark.parametrize("n_workers", [2, 4])
    def test_bw_freq_unchanged(self, m2, n_workers):
        """Frequency axis must not depend on n_workers."""
        ref = compute_hv(m2, n_workers=1, **KW_BW)
        par = compute_hv(m2, n_workers=n_workers, **KW_BW)
        np.testing.assert_array_equal(par.freq, ref.freq)

    @pytest.mark.parametrize("n_workers", [2, 4])
    def test_no_bw_nworkers_is_noop(self, m2, n_workers):
        """When body waves are disabled, n_workers has no effect on output."""
        ref = compute_hv(m2, n_workers=1, **KW_SW)
        par = compute_hv(m2, n_workers=n_workers, **KW_SW)
        np.testing.assert_array_equal(par.hv, ref.hv)

    @pytest.mark.parametrize("model_fix", ["m2", "m3", "m4"])
    def test_bw_all_finite_serial(self, request, model_fix):
        """Baseline: serial BW results are finite and positive."""
        m = request.getfixturevalue(model_fix)
        r = compute_hv(m, n_workers=1, **KW_BW)
        assert np.all(np.isfinite(r.hv)), "serial result contains NaN/Inf"
        assert np.all(r.hv > 0), "serial H/V contains non-positive values"

    @pytest.mark.parametrize("model_fix", ["m2", "m3", "m4"])
    def test_bw_parallel_all_finite(self, request, model_fix):
        """Parallel BW results must be finite and positive for all model sizes."""
        m = request.getfixturevalue(model_fix)
        r = compute_hv(m, n_workers=4, **KW_BW)
        assert np.all(np.isfinite(r.hv)), "parallel result contains NaN/Inf"
        assert np.all(r.hv > 0), "parallel H/V contains non-positive values"

    def test_repeated_calls_stable(self, m2):
        """Repeated parallel calls must return bit-identical results."""
        r_a = compute_hv(m2, n_workers=4, **KW_BW)
        r_b = compute_hv(m2, n_workers=4, **KW_BW)
        np.testing.assert_array_equal(r_b.hv, r_a.hv)

    def test_n_workers_exceeds_n_freq(self, m2):
        """n_workers > n_freq must not crash or corrupt results."""
        kw = dict(KW_BW, n_freq=4)
        ref = compute_hv(m2, n_workers=1, **kw)
        par = compute_hv(m2, n_workers=16, **kw)
        np.testing.assert_allclose(par.hv, ref.hv, rtol=RTOL)

    def test_omp_env_not_mutated(self, m2):
        """compute_hv must not modify OMP_NUM_THREADS in the calling environment."""
        import os
        before = os.environ.get("OMP_NUM_THREADS")
        compute_hv(m2, n_workers=4, **KW_BW)
        assert os.environ.get("OMP_NUM_THREADS") == before


# ==============================================================================
# compute_hv_batch — model-level Fortran OMP parallelism
# ==============================================================================
class TestComputeHVBatchWorkers:
    """compute_hv_batch with n_workers>1 must match sequential compute_hv calls."""

    @pytest.fixture(scope="class")
    def four_models(self):
        return [
            Model.from_layers([
                Layer(thickness=float(h), vp=600.0, vs=200.0, density=1800.0),
                Layer(thickness=0.0,      vp=2000.0, vs=800.0, density=2300.0),
            ])
            for h in [10.0, 20.0, 30.0, 40.0]
        ]

    @pytest.mark.parametrize("n_workers", [1, 2, 4])
    def test_bw_matches_sequential(self, four_models, n_workers):
        sequential = [compute_hv(m, n_workers=1, **KW_BW) for m in four_models]
        batch = compute_hv_batch(four_models, n_workers=n_workers, **KW_BW)
        for i, (s, b) in enumerate(zip(sequential, batch)):
            np.testing.assert_allclose(
                b.hv, s.hv, rtol=RTOL,
                err_msg=f"model {i} mismatch with n_workers={n_workers}",
            )

    @pytest.mark.parametrize("n_workers", [1, 2, 4])
    def test_sw_matches_sequential(self, four_models, n_workers):
        sequential = [compute_hv(m, **KW_SW) for m in four_models]
        batch = compute_hv_batch(four_models, n_workers=n_workers, **KW_SW)
        for i, (s, b) in enumerate(zip(sequential, batch)):
            np.testing.assert_allclose(
                b.hv, s.hv, rtol=RTOL,
                err_msg=f"model {i} mismatch with n_workers={n_workers}",
            )

    def test_mixed_layer_counts_bw(self, m2, m3, m4):
        """Mixed layer counts with body waves enabled must match sequential."""
        mixed = [m2, m3, m4, m3, m2]
        sequential = [compute_hv(m, n_workers=1, **KW_BW) for m in mixed]
        batch = compute_hv_batch(mixed, n_workers=4, **KW_BW)
        for i, (s, b) in enumerate(zip(sequential, batch)):
            np.testing.assert_allclose(b.hv, s.hv, rtol=RTOL,
                                       err_msg=f"mixed model {i} mismatch")

    def test_n_workers_1_vs_N_identical(self, four_models):
        """n_workers=1 and n_workers=4 must give bit-identical batch results."""
        bat1 = compute_hv_batch(four_models, n_workers=1, **KW_BW)
        bat4 = compute_hv_batch(four_models, n_workers=4, **KW_BW)
        for i, (a, b) in enumerate(zip(bat1, bat4)):
            np.testing.assert_array_equal(b.hv, a.hv,
                                          err_msg=f"model {i}: n_workers=1 vs 4")

    def test_n_workers_exceeds_n_models(self, m2, m3):
        """n_workers > n_models must not crash or corrupt results."""
        seq = [compute_hv(m, n_workers=1, **KW_BW) for m in [m2, m3]]
        bat = compute_hv_batch([m2, m3], n_workers=16, **KW_BW)
        for s, b in zip(seq, bat):
            np.testing.assert_allclose(b.hv, s.hv, rtol=RTOL)

    def test_repeated_batch_calls_stable(self, four_models):
        """Repeated batch calls must return bit-identical results."""
        kw = dict(KW_BW, n_workers=4)
        a = compute_hv_batch(four_models, **kw)
        b = compute_hv_batch(four_models, **kw)
        for ra, rb in zip(a, b):
            np.testing.assert_array_equal(rb.hv, ra.hv)

    def test_batch_all_finite_parallel(self, four_models):
        """All batch results must be finite and positive when running in parallel."""
        results = compute_hv_batch(four_models, n_workers=4, **KW_BW)
        for i, r in enumerate(results):
            assert np.all(np.isfinite(r.hv)), f"model {i}: NaN/Inf in batch output"
            assert np.all(r.hv > 0), f"model {i}: non-positive H/V in batch output"

    def test_omp_env_not_mutated(self, four_models):
        """compute_hv_batch must not modify OMP_NUM_THREADS in the calling env."""
        import os
        before = os.environ.get("OMP_NUM_THREADS")
        compute_hv_batch(four_models, n_workers=4, **KW_BW)
        assert os.environ.get("OMP_NUM_THREADS") == before
