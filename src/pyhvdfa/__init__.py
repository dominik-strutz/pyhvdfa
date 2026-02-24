"""
pyhvdfa — H/V spectral ratio via Diffuse Field Assumption.

Thin Python wrapper around the compiled Fortran extension ``_hv_core``
(built from ``fortran_src/hv_wrapper.f90`` via f2py + meson).

Public API
----------
compute_hv(model, freq_min, freq_max, n_freq, ...)  ->  HVResult
Model.from_layers(layers)
Model.from_arrays(vp, vs, density, thickness)
Model.from_file(path)
Layer(thickness, vp, vs, density)
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pyhvdfa import _hv_core
from pyhvdfa._types import HVResult, Layer, Model

__version__ = "0.1.0"
__all__ = ["compute_hv", "compute_hv_batch", "Layer", "Model", "HVResult"]


def compute_hv(
    model: Model,
    freq_min: float = 0.1,
    freq_max: float = 20.0,
    n_freq: int = 100,
    n_modes_rayleigh: int = 20,
    n_modes_love: int = 20,
    include_body_waves: bool = True,
    nks: int = 256,
    sh_damp: float = 1e-3,
    psv_damp: float = 1e-3,
    precision: float = 1e-6,
) -> HVResult:
    """Compute the H/V spectral ratio using the Diffuse Field Assumption.

    Parameters
    ----------
    model : Model
        1-D layered earth model (layers + halfspace).
    freq_min : float
        Minimum frequency in Hz.
    freq_max : float
        Maximum frequency in Hz.
    n_freq : int
        Number of logarithmically-spaced frequency samples.
    n_modes_rayleigh : int
        Maximum number of Rayleigh modes to search (≥1 recommended).
        Default 20 matches the reference HV-DFA example (``-nmr 20``).
    n_modes_love : int
        Maximum number of Love modes to search (0 = disable Love contribution).
        Default 20 matches the reference HV-DFA example (``-nml 20``).
    include_body_waves : bool
        Add body-wave integral contribution.  Set to ``False`` to use only
        surface waves (faster, less accurate at the lowest frequencies).
    nks : int
        Number of wavenumber integration steps for the body-wave part (ignored
        when *include_body_waves* is ``False``).  Typical: 256.
    sh_damp : float
        Imaginary-frequency damping fraction for the SH body-wave integration
        (default: 1e-3).  HV-INV uses 1e-3 to 1e-2; lower values (e.g. 1e-5)
        reduce regularisation and may produce spurious spikes.
    psv_damp : float
        Imaginary-frequency damping fraction for the P-SV body-wave integration
        (default: 1e-3).  Same guidance as *sh_damp*.
    precision : float
        Relative slowness root-search tolerance (fractional, default: 1e-6).
        Passed to Fortran as a percentage: ``prec_in = precision * 100``.
        The Fortran standalone defaults to ``PREC = 1e-4`` (percent), which
        corresponds to ``G_PRECISION = 1e-6`` internally — so the Python
        default of ``1e-6`` reproduces that same effective tolerance.

    Returns
    -------
    HVResult
        Object with ``freq`` (Hz) and ``hv`` arrays of shape *(n_freq,)*.

    Notes
    -----
    The Fortran subroutine ``COMPUTE_HV`` (``fortran_src/hv_wrapper.f90``)
    implements the full DFA algorithm of Sánchez-Sesma et al. (2011).

    References
    ----------
    Sánchez-Sesma et al. (2011), *Bull. Seismol. Soc. Am.*, 101(3), 1332–1342.
    García-Jerez et al. (2016), *Comput. Geosci.*, 97, 67–78.
    """
    if freq_min <= 0.0:
        raise ValueError(f"freq_min must be positive, got {freq_min}")
    if freq_max <= freq_min:
        raise ValueError(
            f"freq_max ({freq_max}) must be greater than freq_min ({freq_min})"
        )
    if n_freq < 2:
        raise ValueError(f"n_freq must be ≥ 2, got {n_freq}")
    if n_modes_rayleigh < 1:
        raise ValueError(f"n_modes_rayleigh must be ≥ 1, got {n_modes_rayleigh}")

    # Frequency vector and angular frequencies (float32 matches Fortran REAL)
    freq = np.geomspace(freq_min, freq_max, n_freq)
    x_in = (2.0 * math.pi * freq).astype(np.float32)

    # Pad thickness array: model.h has shape (n-1,); Fortran COMPUTE_HV expects
    # H_IN(NCAPAS_IN) i.e. shape (n,) — last element is the halfspace (ignored).
    h_full = np.append(model.h, 0.0).astype(np.float64)

    # Body-wave integration points (0 tells Fortran to skip the BW integrals)
    nks_in = int(nks) if include_body_waves else 0

    # Fortran uses PREC_IN in percent and then applies G_PRECISION = PREC_IN * 1e-2.
    # So we pass precision * 100 here to recover the user's fractional tolerance.
    prec_in = np.float32(precision * 100.0)

    hv_out = _hv_core.compute_hv(
        alfa_in=model.alfa,
        bta_in=model.bta,
        h_in=h_full,
        rho_in=model.rho,
        x_in=x_in,
        nmr=int(n_modes_rayleigh),
        nml=int(n_modes_love),
        nks_in=nks_in,
        shdamp_in=np.float32(sh_damp),
        psvdamp_in=np.float32(psv_damp),
        prec_in=prec_in,
    )

    return HVResult(freq=freq, hv=hv_out)


def compute_hv_batch(
    models: list[Model],
    *,
    n_workers: int | None = None,
    freq_min: float = 0.1,
    freq_max: float = 20.0,
    n_freq: int = 100,
    n_modes_rayleigh: int = 20,
    n_modes_love: int = 20,
    include_body_waves: bool = True,
    nks: int = 256,
    sh_damp: float = 1e-3,
    psv_damp: float = 1e-3,
    precision: float = 1e-6,
) -> list[HVResult]:
    """Compute H/V for multiple models in parallel using threads.

    Each model is evaluated in its own OS thread.  The Fortran extension
    is compiled with ``threadsafe`` (GIL released) and all module-level
    state is ``!$OMP THREADPRIVATE``, so threads have fully isolated
    Fortran state with zero IPC or pickling overhead.

    Parameters
    ----------
    models : list[Model]
        Earth models to evaluate.
    n_workers : int | None
        Maximum number of threads.  Defaults to ``os.cpu_count()``.
    freq_min, freq_max, n_freq, n_modes_rayleigh, n_modes_love,
    include_body_waves, nks, sh_damp, psv_damp, precision
        Forwarded to :func:`compute_hv` — see its docstring for details.
        All models are evaluated with the same computational parameters.

    Returns
    -------
    list[HVResult]
        One result per model, in the same order as *models*.

    Notes
    -----
    ``OMP_NUM_THREADS`` is temporarily set to ``"1"`` to suppress the
    residual inner OpenMP pragma inside the read-only submodule file
    ``GL.f90`` (Love-wave Green's functions).  This prevents thread
    over-subscription when many models run concurrently.  The original
    value is restored after the batch completes.

    For inversion loops, pass ``n_workers`` once and reuse the same call
    site — the ``ThreadPoolExecutor`` is lightweight and creation cost
    is negligible (~µs).

    Examples
    --------
    >>> from pyhvdfa import compute_hv_batch, Model, Layer
    >>> models = [
    ...     Model.from_layers([
    ...         Layer(thickness=h, vp=600.0, vs=200.0, density=1800.0),
    ...         Layer(thickness=0.0, vp=2000.0, vs=800.0, density=2300.0),
    ...     ])
    ...     for h in [10.0, 20.0, 30.0, 40.0]
    ... ]
    >>> results = compute_hv_batch(models, n_workers=4)
    >>> len(results)
    4
    """
    if not models:
        return []

    n = n_workers or os.cpu_count() or 1
    kwargs = dict(
        freq_min=freq_min,
        freq_max=freq_max,
        n_freq=n_freq,
        n_modes_rayleigh=n_modes_rayleigh,
        n_modes_love=n_modes_love,
        include_body_waves=include_body_waves,
        nks=nks,
        sh_damp=sh_damp,
        psv_damp=psv_damp,
        precision=precision,
    )

    # Suppress inner OpenMP (GL.f90) to avoid oversubscription.
    old_omp = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        with ThreadPoolExecutor(max_workers=n) as executor:
            futures = [executor.submit(compute_hv, model, **kwargs) for model in models]
            return [f.result() for f in futures]
    finally:
        if old_omp is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = old_omp
