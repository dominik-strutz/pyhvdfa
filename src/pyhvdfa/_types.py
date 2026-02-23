"""
_types.py — dtype aliases and data structures for pyhvdfa.

This module is intentionally free of Numba imports so that it can be used
in both JIT and non-JIT contexts.  All numerical kernels receive plain
numpy arrays — not Model objects.

References
----------
Fortran equivalent: modules.f90 (module TYPES, module globales)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

# ── dtype aliases ──────────────────────────────────────────────────────────────
FLOAT = np.float64    # kind(0.0d0) in Fortran
CMPLX = np.complex128 # kind((0.0d0,0.0d0)) in Fortran

# Imaginary unit — matches "i = (0,1)" in globales
I_UNIT = np.complex128(1j)


# ── physical constants ─────────────────────────────────────────────────────────
PI: float = np.pi
TWO_PI: float = 2.0 * np.pi


# ── Layer dataclass ────────────────────────────────────────────────────────────
@dataclass
class Layer:
    """One layer in the earth model (SI units throughout).

    Parameters
    ----------
    thickness : float
        Layer thickness in metres.  Use 0.0 for the halfspace (bottom layer).
    vp : float
        P-wave velocity in m/s.
    vs : float
        S-wave velocity in m/s.
    density : float
        Density in kg/m³.
    """
    thickness: float
    vp: float
    vs: float
    density: float


# ── Model dataclass ────────────────────────────────────────────────────────────
@dataclass
class Model:
    """1-D layered earth model (SI units throughout).

    The last layer is the halfspace (thickness is ignored).

    Attributes
    ----------
    alfa : np.ndarray, shape (n,)   P-wave velocities (m/s)
    bta  : np.ndarray, shape (n,)   S-wave velocities (m/s)
    rho  : np.ndarray, shape (n,)   Densities (kg/m³)
    h    : np.ndarray, shape (n-1,) Layer thicknesses (m)  — halfspace excluded
    mu   : np.ndarray, shape (n,)   Shear moduli (Pa) = rho * bta²
    n    : int                      Total number of layers including halfspace
    """
    alfa: np.ndarray  # shape (n,)
    bta:  np.ndarray  # shape (n,)
    rho:  np.ndarray  # shape (n,)
    h:    np.ndarray  # shape (n-1,)  thicknesses, halfspace excluded
    mu:   np.ndarray = field(init=False)  # shape (n,)
    n:    int = field(init=False)

    def __post_init__(self) -> None:
        self.alfa = np.asarray(self.alfa, dtype=FLOAT)
        self.bta  = np.asarray(self.bta,  dtype=FLOAT)
        self.rho  = np.asarray(self.rho,  dtype=FLOAT)
        self.h    = np.asarray(self.h,    dtype=FLOAT)
        self.n    = len(self.alfa)
        if self.n < 2:
            raise ValueError("Model must have at least 2 layers (one layer + halfspace).")
        if len(self.h) != self.n - 1:
            raise ValueError(
                f"h must have n-1={self.n-1} elements, got {len(self.h)}."
            )
        self.mu = self.rho * self.bta**2

    # ── factory methods ────────────────────────────────────────────────────────
    @classmethod
    def from_layers(cls, layers: Sequence[Layer]) -> "Model":
        """Build a Model from a list of Layer objects.

        The last element must be the halfspace (thickness is ignored).

        Example
        -------
        >>> m = Model.from_layers([
        ...     Layer(thickness=20.0, vp=400.0, vs=200.0, density=1800.0),
        ...     Layer(thickness=0.0,  vp=1600.0, vs=800.0, density=2200.0),
        ... ])
        """
        if len(layers) < 2:
            raise ValueError("Need at least one layer plus a halfspace.")
        alfa    = np.array([l.vp       for l in layers], dtype=FLOAT)
        bta     = np.array([l.vs       for l in layers], dtype=FLOAT)
        rho     = np.array([l.density  for l in layers], dtype=FLOAT)
        # thicknesses: all except the halfspace
        h       = np.array([l.thickness for l in layers[:-1]], dtype=FLOAT)
        return cls(alfa=alfa, bta=bta, rho=rho, h=h)

    @classmethod
    def from_file(cls, path: str | Path) -> "Model":
        """Read a model from the Fortran-format text file.

        File format (same as HV-DFA ``-f`` option)::

            N_layers
            thickness  Vp  Vs  density   (layers 1..N-1)
            0          Vp  Vs  density   (halfspace, thickness ignored)

        Units: metres, m/s, kg/m³.
        """
        path = Path(path)
        with open(path) as fh:
            lines = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
        n = int(lines[0])
        if len(lines) - 1 < n:
            raise ValueError(f"File declares {n} layers but only {len(lines)-1} data lines found.")
        rows = [list(map(float, lines[i + 1].split())) for i in range(n)]
        h    = np.array([row[0] for row in rows[:-1]], dtype=FLOAT)
        vp   = np.array([row[1] for row in rows], dtype=FLOAT)
        vs   = np.array([row[2] for row in rows], dtype=FLOAT)
        dens = np.array([row[3] for row in rows], dtype=FLOAT)
        return cls(alfa=vp, bta=vs, rho=dens, h=h)

    @classmethod
    def from_arrays(
        cls,
        vp:       Sequence[float],
        vs:       Sequence[float],
        density:  Sequence[float],
        thickness: Sequence[float],
    ) -> "Model":
        """Build from plain lists/arrays.

        ``thickness`` must have ``len(vp) - 1`` elements (halfspace excluded).
        """
        return cls(
            alfa=np.asarray(vp,        dtype=FLOAT),
            bta =np.asarray(vs,        dtype=FLOAT),
            rho =np.asarray(density,   dtype=FLOAT),
            h   =np.asarray(thickness, dtype=FLOAT),
        )


# ── HVResult dataclass ─────────────────────────────────────────────────────────
@dataclass
class HVResult:
    """Output of :func:`pyhvdfa.compute_hv`.

    The Fortran extension returns only the assembled H/V ratio; intermediate
    quantities (Green's function components, dispersion curves) are not exposed.

    Attributes
    ----------
    freq : np.ndarray
        Frequency vector in Hz, shape *(n_freq,)*.
    hv : np.ndarray
        H/V spectral ratio, shape *(n_freq,)*.  Values may be NaN at frequencies
        below the fundamental-mode cutoff or where no valid slowness was found.
    """
    freq: np.ndarray
    hv:   np.ndarray
