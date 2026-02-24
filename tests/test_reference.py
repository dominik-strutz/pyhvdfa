"""
tests/test_reference.py — Comparison against Fortran reference output.

Run generate_reference.sh first to populate tests/reference_data/hv_*.csv.
Tests in this module are automatically skipped if reference files are missing.

The tolerance RTOL=5e-3 (0.5%) accounts for:
  - Float32 frequency input discretization in hv_core.pyf (x_in: real(4))
  - Mode-transition frequency artifacts where both float32 and float64 see jumps
  - The limit of the Fortran iterative root solver (precision = 1e-4 fractional)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyhvdfa import compute_hv
from pyhvdfa._types import Model

REF_DIR = Path(__file__).parent / "reference_data"
MODEL_DIR = REF_DIR / "models"

RTOL = 5e-3  # Relative tolerance for H/V comparison (float32/float64 precision loss)
PEAK_FREQ_TOLERANCE = 0.05  # Peak frequency tolerance (5%)


def _load_reference(tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Load Fortran reference (freq, hv) from CSV.

    Parameters
    ----------
    tag : str
        Model tag (e.g., 'simple_2layer')

    Returns
    -------
    freq : np.ndarray
        Frequency array (Hz)
    hv : np.ndarray
        H/V spectral ratio
    """
    ref_file = REF_DIR / f"hv_{tag}.csv"
    if not ref_file.exists():
        pytest.skip(
            f"Reference file missing: {ref_file}. Run generate_reference.sh first."
        )
    data = np.loadtxt(ref_file, delimiter=",")
    return data[:, 0], data[:, 1]


def _load_model(tag: str) -> Model:
    """Load a Model from a Fortran-format text file.

    Parameters
    ----------
    tag : str
        Model tag (filename prefix)

    Returns
    -------
    Model
    """
    model_file = MODEL_DIR / f"{tag}.txt"
    if not model_file.exists():
        pytest.skip(f"Model file missing: {model_file}")
    return Model.from_file(model_file)


def _run_pyhvdfa(model: Model, freq: np.ndarray) -> np.ndarray:
    """Compute H/V using pyhvdfa with reference parameters.

    Parameters
    ----------
    model : Model
        Layered Earth model
    freq : np.ndarray
        Frequency array (Hz)

    Returns
    -------
    hv : np.ndarray
        H/V spectral ratio
    """
    result = compute_hv(
        model,
        freq_min=float(freq[0]),
        freq_max=float(freq[-1]),
        n_freq=len(freq),
        # Use same parameters as generate_reference.sh:
        # nmr=20, nml=20, nks=256, prec=1e-4% (→ G_PRECISION=1e-6)
        n_modes_rayleigh=20,
        n_modes_love=20,
        include_body_waves=True,
        # sh_damp and psv_damp use compute_hv defaults (1e-3),
        # matching generate_reference.sh (-ash 1e-3 -apsv 1e-3)
    )
    return result.hv


@pytest.mark.parametrize(
    "tag",
    [
        "simple_2layer",
        "model_3layer",
        "model_kirk",
        "soft_sediment",
    ],
)
def test_hv_matches_fortran(tag: str):
    """H/V ratio must match Fortran reference output within rtol=5e-3 (0.5%).

    This test validates the core numerical accuracy of the Python wrapper
    against the original Fortran code. It filters for H/V > 1e-3 to avoid
    noise in regions where the curve is nearly flat.
    """
    ref_freq, ref_hv = _load_reference(tag)
    model = _load_model(tag)
    py_hv = _run_pyhvdfa(model, ref_freq)

    # Filter for valid, significant H/V values
    mask = np.isfinite(ref_hv) & np.isfinite(py_hv) & (ref_hv > 1e-3)
    if mask.sum() < 5:
        pytest.skip(f"Too few valid comparison points for {tag}")

    np.testing.assert_allclose(
        py_hv[mask],
        ref_hv[mask],
        rtol=RTOL,
        err_msg=f"H/V mismatch for model {tag!r}",
    )


@pytest.mark.parametrize(
    "tag",
    [
        "simple_2layer",
        "model_3layer",
        "model_kirk",
        "soft_sediment",
    ],
)
def test_peak_frequency_matches_fortran(tag: str):
    """Peak H/V frequency must match Fortran reference within 5%.

    The fundamental-mode resonance frequency is a key quantity in seismic
    hazard and site characterization. This test ensures the Python wrapper
    correctly identifies the peak position.
    """
    ref_freq, ref_hv = _load_reference(tag)
    model = _load_model(tag)
    py_hv = _run_pyhvdfa(model, ref_freq)

    # Find peaks (handle NaN in Python result)
    ref_peak_idx = np.nanargmax(ref_hv)
    py_peak_idx = np.nanargmax(np.where(np.isfinite(py_hv), py_hv, 0.0))

    ref_peak = ref_freq[ref_peak_idx]
    py_peak = ref_freq[py_peak_idx]

    rel_error = abs(py_peak - ref_peak) / ref_peak
    assert rel_error < PEAK_FREQ_TOLERANCE, (
        f"Peak frequencies differ for {tag}: Python={py_peak:.3f} Hz, "
        f"Fortran={ref_peak:.3f} Hz (rel. error={rel_error*100:.1f}%)"
    )


@pytest.mark.parametrize(
    "tag",
    [
        "simple_2layer",
        "model_3layer",
        "model_kirk",
        "soft_sediment",
    ],
)
def test_hv_result_shape_and_finitude(tag: str):
    """H/V output must have the correct shape and contain finite values.

    This is a basic sanity check: the Python wrapper must return an array
    of the correct length with no NaN or Inf values (except possibly at
    very low frequencies where numerical precision may degrade).
    """
    ref_freq, ref_hv = _load_reference(tag)
    model = _load_model(tag)
    py_hv = _run_pyhvdfa(model, ref_freq)

    # Check shape
    assert (
        py_hv.shape == ref_hv.shape
    ), f"Shape mismatch for {tag}: Python {py_hv.shape} vs Fortran {ref_hv.shape}"

    # Check that most values are finite (exclude extreme ends if needed)
    finite_count = np.isfinite(py_hv).sum()
    assert finite_count > 0.9 * len(
        py_hv
    ), f"Too many non-finite values in {tag}: {finite_count}/{len(py_hv)}"

    # Check reasonable magnitude (H/V should be positive and typically range 0.5–5)
    py_valid = py_hv[np.isfinite(py_hv)]
    assert np.all(py_valid > 0), f"Found non-positive H/V values in {tag}"
    assert np.all(py_valid < 100), f"Found unreasonably large H/V values in {tag}"


@pytest.mark.parametrize(
    "tag",
    [
        "simple_2layer",
        "model_3layer",
        "model_kirk",
        "soft_sediment",
    ],
)
def test_correlation_with_fortran(tag: str):
    """Python and Fortran H/V curves must be strongly correlated (r > 0.95).

    Even if absolute values differ slightly, the overall shape and behavior
    of the H/V curve should track closely.
    """
    ref_freq, ref_hv = _load_reference(tag)
    model = _load_model(tag)
    py_hv = _run_pyhvdfa(model, ref_freq)

    mask = np.isfinite(ref_hv) & np.isfinite(py_hv) & (ref_hv > 1e-3)
    if mask.sum() < 5:
        pytest.skip(f"Too few valid comparison points for {tag}")

    correlation = np.corrcoef(ref_hv[mask], py_hv[mask])[0, 1]
    assert correlation > 0.95, f"Low correlation for {tag}: r={correlation:.4f}"
