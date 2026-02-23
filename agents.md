# agents.md — pyhvdfa Developer & Agent Guidelines

This file defines conventions for humans and AI coding agents working on **pyhvdfa**.
Read it before making any changes.

---

## Project Purpose

`pyhvdfa` is a **thin Python wrapper** around [HV-DFA](https://github.com/agarcia-jerez/HV-DFA)
(García-Jerez et al., 2016), which computes the theoretical H/V spectral ratio for a
1-D layered viscoelastic medium using the Diffuse Field Assumption (DFA) of Sánchez-Sesma et al. (2011).

The Fortran source lives in `fortran_src/`. It is the authoritative numerical implementation.
**Do not reimplement the physics in Python.** The Python layer only handles:
- input validation and unit conversion
- building NumPy arrays to pass to the Fortran extension
- wrapping the output in typed dataclasses

> **`fortran_src/HV-DFA/` is a git submodule pointing to the upstream
> [HV-DFA](https://github.com/agarcia-jerez/HV-DFA) repository. The files it
> contains are read-only. Never modify, delete, or reformat any file under that
> directory. Do not detach or replace the submodule without updating `.gitmodules`.**

---

## Package Layout

```
pyhvdfa/
  pyproject.toml          # meson-python build config; no CLI entrypoint
  agents.md               # this file
  README.md
  meson.build             # root meson build: f2py custom_target + subdir
  fortran_src/
    hv_wrapper.f90        # single-unit entry point: COMPUTE_HV subroutine
    HV-DFA/               # git submodule → github.com/agarcia-jerez/HV-DFA (read-only)
  src/
    pyhvdfa/
      __init__.py         # public API: compute_hv(), Layer, Model, HVResult
      _types.py           # Layer, Model, HVResult dataclasses (no Numba)
      hv_core.pyf         # f2py signature file for COMPUTE_HV
      _hv_core.pyi        # type stubs for the compiled extension (IDE support)
      meson.build         # builds _hv_core extension + installs Python files
  tests/
    conftest.py           # shared fixtures (simple_2layer, model_3layer, model_kirk)
    reference_data/
      models/             # model definition .txt files
      hv_*.csv            # Fortran-generated reference H/V outputs
    generate_reference.sh # (re-)generate reference data from Fortran HV.exe
    test_types.py         # tests for Layer, Model, HVResult dataclasses
    test_module_exports.py # import smoke tests
    test_compute_hv.py    # integration tests (sanity checks)
    test_reference.py     # numerical comparison against Fortran reference data
  examples/
    01_quickstart.ipynb
    02_model_comparison.ipynb
    03_parameter_sensitivity.ipynb
```

---

## How the Extension is Built

```
hv_core.pyf   →  (f2py --lower)  →  _hv_coremodule.c   (Python/C bridge)
hv_wrapper.f90 + HV-DFA/*.f90    →  compiled Fortran objects
_hv_coremodule.c + fortranobject.c + Fortran objects  →  _hv_core.<tag>.so
```

`hv_wrapper.f90` uses `INCLUDE './HV-DFA/modules.f90'` etc., so the
`fortran_src/HV-DFA/` subdirectory must exist. It is populated automatically
when cloning with `git clone --recurse-submodules` or by running
`git submodule update --init` after a plain clone.

### Build commands

```bash
# Clone (submodule must be initialised)
git clone --recurse-submodules https://github.com/dominik-strutz/pyhvdfa
# or, after a plain clone:
git submodule update --init

# Editable install (development — rebuilds extension in place)
uv sync --no-build-isolation

# Production wheel
uvx cibuildwheel --platform linux   # or macos / windows
```

---

## Public API

```python
from pyhvdfa import compute_hv, Model, Layer, HVResult

model = Model.from_layers([
    Layer(thickness=30.0, vp=600.0, vs=200.0, density=1800.0),
    Layer(thickness=0.0,  vp=2000.0, vs=800.0, density=2300.0),
])

result = compute_hv(model, freq_min=0.1, freq_max=20.0, n_freq=100)
# result.freq  — np.ndarray (Hz)
# result.hv    — np.ndarray (H/V ratio)
```

### `compute_hv` parameters

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `model` | `Model` | — | Layer stack including halfspace |
| `freq_min` | float | 0.1 | Minimum frequency (Hz) |
| `freq_max` | float | 20.0 | Maximum frequency (Hz) |
| `n_freq` | int | 100 | Number of log-spaced samples |
| `n_modes_rayleigh` | int | 20 | Maximum Rayleigh modes |
| `n_modes_love` | int | 20 | Maximum Love modes (0 = disable) |
| `include_body_waves` | bool | True | Include body-wave integrals |
| `nks` | int | 256 | Body-wave wavenumber integration steps |
| `sh_damp` | float | 1e-5 | SH imaginary-frequency damping fraction |
| `psv_damp` | float | 1e-5 | P-SV imaginary-frequency damping fraction |
| `precision` | float | 1e-6 | Slowness root-search tolerance (fractional) |

The `precision` value is converted internally: `prec_in = precision * 100` because
Fortran's `COMPUTE_HV` takes it as a percentage and then applies `G_PRECISION = PREC_IN * 1e-2`.

`model.h` has shape `(n-1,)` (halfspace excluded); `__init__.py` pads it with 0.0
to shape `(n,)` before passing to Fortran.

### `HVResult`

Only two fields — the Fortran extension returns the assembled H/V array; intermediate
quantities (Green's functions, dispersion curves) are not exposed.

```python
@dataclass
class HVResult:
    freq: np.ndarray   # Hz,  shape (n_freq,)
    hv:   np.ndarray   # H/V, shape (n_freq,)
```

---

## `_types.py` Contract

- **Never import Numba** here. This module must be importable in both regular Python
  and any future JIT context without side effects.
- `Layer` and `Model` are plain dataclasses with NumPy arrays (float64).
- `Model` validates that it has at least 2 layers (one layer + halfspace) and that
  `len(h) == n - 1`.
- `Model.h` excludes the halfspace thickness; `__init__.py` pads it to length `n`.

---

## `hv_core.pyf` Contract

The `.pyf` signature file is the source of truth for the f2py-generated C bridge.
Key decisions encoded there:

- `ncapas_in` and `nx_in` are `intent(hide)` — Python callers never pass them;
  f2py derives them from the array lengths of `alfa_in` and `x_in` respectively.
- `x_in` is `real(4)` (float32) — matches Fortran `REAL` (single precision).
- `alfa_in`, `bta_in`, `h_in`, `rho_in`, `hv_out` are `real(8)` (float64) — matches
  Fortran `REAL(LONG_FLOAT)` from `modules.f90`.

Do not change these types without also updating `_hv_core.pyi` and `__init__.py`.

---

## Contribution Rules

- **Never modify `fortran_src/HV-DFA/`**. It is a git submodule — treat all files
  inside as read-only. Do not edit, delete, rename, reformat, or bump the submodule
  pointer without an explicit upstream release decision.
- **Do not add Python reimplementations of the Fortran algorithms.** The only valid
  Python code is: input validation, array construction, unit conversion, and dataclass
  definitions.
- **Do not rename public API symbols** (`compute_hv`, `Layer`, `Model`, `HVResult`)
  without updating `__init__.py`, `_types.py`, and the notebooks.
- **Do not add runtime dependencies** beyond `numpy`. The build requires `numpy` and
  `meson-python`, but the installed package needs only `numpy`.
- **Coordinate type changes**: any change to the dtypes in `hv_core.pyf` must be
  reflected in `_hv_core.pyi` (IDE stubs) and the array casts in `__init__.py`.
- **Fortran index convention**: Fortran arrays are 1-based; Python is 0-based.
  Layer 0 in Python = Layer 1 in Fortran. The halfspace is the last layer in both.

---

## Testing Strategy

1. **Type tests** (`test_types.py`): `Layer`, `Model`, `HVResult` construction and
   validation; no Fortran dependency.
2. **Export smoke tests** (`test_module_exports.py`): every public symbol importable;
   compiled extension loadable.
3. **Integration tests** (`test_compute_hv.py`): sanity checks on shape, finiteness,
   sign, and approximate peak frequency. These run against the live extension.
4. **Reference tests** (`test_reference.py`): numerical comparison of `compute_hv()`
   output against Fortran `HV.exe` reference CSV files (`rtol=1e-4`). Skipped if
   reference files are absent.

### Generating reference data

```bash
# 1. Compile the standalone Fortran program (once)
cd /path/to/HV-DFA && gfortran HV.f90 -o HV.exe -O2

# 2. Generate reference CSVs for each test model
cd tests && bash generate_reference.sh
```

### Adding a new model test

1. Add a `.txt` file to `tests/reference_data/models/` in the format:
   ```
   N_layers
   thickness_m  Vp_m_s  Vs_m_s  density_kg_m3
   ...
   0  Vp  Vs  density     (halfspace, thickness ignored)
   ```
2. Add the model tag to `MODELS` in `generate_reference.sh` and re-run it.
3. Add a parametrized case to `test_reference.py`.

---

## uv Workflow

```bash
# Clone with submodule
git clone --recurse-submodules https://github.com/dominik-strutz/pyhvdfa

# Install (compiles Fortran extension)
uv sync --no-build-isolation

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=pyhvdfa --cov-report=term-missing

# Build wheel
uvx cibuildwheel --platform linux   # or macos / windows
```
