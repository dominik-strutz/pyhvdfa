"""
Type stubs for the compiled Fortran extension _hv_core.

These stubs describe the interface f2py generates from hv_core.pyf so that
IDEs (VS Code / Pylance, PyCharm, etc.) provide correct type hints without
needing the compiled .so.

The hidden arguments ncapas_in and nx_in are not part of the Python signature
because they are declared intent(hide) in the .pyf file.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

def compute_hv(
    alfa_in: NDArray[np.float64],
    bta_in: NDArray[np.float64],
    h_in: NDArray[np.float64],
    rho_in: NDArray[np.float64],
    x_in: NDArray[np.float32],
    nmr: int,
    nml: int,
    nks_in: int,
    shdamp_in: np.float32,
    psvdamp_in: np.float32,
    prec_in: np.float32,
    nthreads_in: int,
) -> NDArray[np.float64]:
    """
    Low-level f2py wrapper for SUBROUTINE COMPUTE_HV.

    Prefer calling ``pyhvdfa.compute_hv`` instead; it validates inputs,
    converts units (Hz → rad/s), and provides a friendlier interface.

    Parameters
    ----------
    alfa_in : ndarray, shape (n_layers,), float64
        P-wave velocities [m/s].
    bta_in : ndarray, shape (n_layers,), float64
        S-wave velocities [m/s].
    h_in : ndarray, shape (n_layers,), float64
        Layer thicknesses [m] (last element ignored).
    rho_in : ndarray, shape (n_layers,), float64
        Layer densities [kg/m³].
    x_in : ndarray, shape (n_freq,), float32
        Angular frequencies ω = 2π·f [rad/s].
    nmr : int
        Maximum number of Rayleigh modes.
    nml : int
        Maximum number of Love modes (0 = disable).
    nks_in : int
        Body-wave wavenumber integration points (0 = skip).
    shdamp_in : float32
        SH body-wave imaginary-frequency fraction.
    psvdamp_in : float32
        P-SV body-wave imaginary-frequency fraction.
    prec_in : float32
        Root-search precision [%].
    nthreads_in : int
        OMP thread count for the body-wave integral loop.
        Use 1 (default) for serial execution or when called from
        ``compute_hv_batch`` (outer model-level OMP handles concurrency).

    Returns
    -------
    hv_out : ndarray, shape (n_freq,), float64
        H/V spectral ratio.
    """
    ...


def compute_hv_batch(
    alfa_in2: NDArray[np.float64],
    bta_in2: NDArray[np.float64],
    h_in2: NDArray[np.float64],
    rho_in2: NDArray[np.float64],
    ncapas_arr: NDArray[np.int32],
    x_in: NDArray[np.float32],
    nmr: int,
    nml: int,
    nks_in: int,
    shdamp_in: np.float32,
    psvdamp_in: np.float32,
    prec_in: np.float32,
    nthreads_in: int,
) -> NDArray[np.float64]:
    """
    Low-level f2py wrapper for SUBROUTINE COMPUTE_HV_BATCH.

    Prefer calling ``pyhvdfa.compute_hv_batch`` instead.

    Parameters
    ----------
    alfa_in2 : ndarray, shape (max_layers, n_models), float64
        P-wave velocities [m/s], zero-padded to max_layers rows.
    bta_in2 : ndarray, shape (max_layers, n_models), float64
        S-wave velocities [m/s].
    h_in2 : ndarray, shape (max_layers, n_models), float64
        Layer thicknesses [m] (last row of each column ignored).
    rho_in2 : ndarray, shape (max_layers, n_models), float64
        Layer densities [kg/m³].
    ncapas_arr : ndarray, shape (n_models,), int32
        Actual number of layers (including halfspace) for each model.
    x_in : ndarray, shape (n_freq,), float32
        Angular frequencies ω = 2π·f [rad/s], shared across all models.
    nmr : int
        Maximum number of Rayleigh modes.
    nml : int
        Maximum number of Love modes (0 = disable).
    nks_in : int
        Body-wave wavenumber integration points (0 = skip).
    shdamp_in : float32
        SH body-wave imaginary-frequency fraction.
    psvdamp_in : float32
        P-SV body-wave imaginary-frequency fraction.
    prec_in : float32
        Root-search precision [%].
    nthreads_in : int
        Number of OMP threads for the outer model-parallel loop.

    Returns
    -------
    hv_out2 : ndarray, shape (n_freq, n_models), float64
        H/V spectral ratio for each model.
    """
    ...
