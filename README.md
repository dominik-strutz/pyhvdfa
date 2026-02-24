# pyhvdfa

[![Tests](https://github.com/dominik-strutz/pyhvdfa/actions/workflows/tests.yml/badge.svg)](https://github.com/dominik-strutz/pyhvdfa/actions/workflows/tests.yml)

Python wrapper around [HV-DFA](https://github.com/agarcia-jerez/HV-DFA) (García-Jerez et al., 2016) — theoretical H/V spectral ratio for a 1-D layered viscoelastic medium using the Diffuse Field Assumption (DFA) of Sánchez-Sesma et al. (2011).

The computationally-intensive work is done entirely by the original Fortran code compiled as a Python extension via f2py + meson. The Python layer handles input validation, array construction, and typed result dataclasses.

> **Disclaimer**: I relied heavily on LLM assistance to write the Python wrapper and build system, so I can not guarantee the correctness of either. However, the underlying Fortran code is unchanged and the wrapper is tested against the original code on a variety of models, so I am reasonably confident in the results.

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.23
- A Fortran compiler (gfortran) — only required when building from source

## Installation

### Pre-built wheel (recommended)

Wheels are provided for Python 3.9–3.13 on Linux (x86-64), macOS (x86-64 / arm64),
and Windows (x86-64) — no Fortran compiler needed.

1. Go to the [Releases page](https://github.com/dominik-strutz/pyhvdfa/releases) and
   find the latest release.
2. Under **Assets**, right-click the `.whl` that matches your platform and Python
   version and copy the link.
3. Install directly from that URL — no manual download required:

```bash
# uv
uv add https://github.com/dominik-strutz/pyhvdfa/releases/download/vX.Y.Z/pyhvdfa-X.Y.Z-cpXXX-...-linux_x86_64.whl

# pip
pip install https://github.com/dominik-strutz/pyhvdfa/releases/download/vX.Y.Z/pyhvdfa-X.Y.Z-cpXXX-...-linux_x86_64.whl
```

Replace the URL with the one you copied in step 2.

---

### Building from source (developers)

A Fortran compiler (gfortran) is required for all source-based installs.

```bash
# Linux
sudo apt-get install gfortran
# macOS
brew install gcc
# Windows — install MinGW-w64 and ensure gfortran is on PATH
```

#### Install directly from GitHub

```bash
uv add git+https://github.com/dominik-strutz/pyhvdfa
```

#### Editable install (for hacking on the code)

```bash
git clone --recurse-submodules https://github.com/dominik-strutz/pyhvdfa
cd pyhvdfa
uv sync --extra test --no-build-isolation
```

`--no-build-isolation` is required because the Fortran extension needs the system
gfortran and a pre-installed numpy to compile correctly.

## Quick start

```python
from pyhvdfa import compute_hv, Layer, Model

model = Model.from_layers([
    Layer(thickness=30.0, vp=600.0,  vs=200.0, density=1800.0),
    Layer(thickness=0.0,  vp=2000.0, vs=800.0, density=2300.0),  # halfspace
])

result = compute_hv(model, freq_min=0.1, freq_max=20.0, n_freq=100)
print(result.freq[:5])   # Hz
print(result.hv[:5])     # H/V ratio
```

## API reference

### `compute_hv`

```python
compute_hv(
    model,
    freq_min=0.1,
    freq_max=20.0,
    n_freq=100,
    n_modes_rayleigh=20,
    n_modes_love=20,
    include_body_waves=True,
    nks=256,
    sh_damp=1e-5,
    psv_damp=1e-5,
    precision=1e-6,
) -> HVResult
```

| Parameter | Default | Description |
|---|---|---|
| `model` | — | `Model` instance (layers + halfspace) |
| `freq_min` | `0.1` | Minimum frequency (Hz) |
| `freq_max` | `20.0` | Maximum frequency (Hz) |
| `n_freq` | `100` | Number of log-spaced frequency samples |
| `n_modes_rayleigh` | `20` | Max Rayleigh modes |
| `n_modes_love` | `20` | Max Love modes (0 = disable) |
| `include_body_waves` | `True` | Include body-wave integral |
| `nks` | `256` | Body-wave wavenumber integration steps |
| `sh_damp` | `1e-5` | SH imaginary-frequency damping fraction |
| `psv_damp` | `1e-5` | P-SV imaginary-frequency damping fraction |
| `precision` | `1e-6` | Slowness root-search tolerance (fractional) |

### `HVResult`

```python
result.freq   # np.ndarray, shape (n_freq,), Hz
result.hv     # np.ndarray, shape (n_freq,), H/V ratio
```

### `Layer`

```python
Layer(thickness, vp, vs, density)
# thickness : float, metres (0.0 for halfspace)
# vp        : float, P-wave velocity (m/s)
# vs        : float, S-wave velocity (m/s)
# density   : float, kg/m3
```

### `Model`

```python
Model.from_layers(layers)                      # list of Layer objects
Model.from_arrays(vp, vs, density, thickness)  # numpy arrays
Model.from_file(path)                          # plain-text model file
```

## Examples

| Notebook | Description |
|---|---|
| [examples/01_quickstart.ipynb](examples/01_quickstart.ipynb) | Basic two-layer model and H/V curve |
| [examples/02_model_comparison.ipynb](examples/02_model_comparison.ipynb) | Comparing multiple velocity models |
| [examples/03_parameter_sensitivity.ipynb](examples/03_parameter_sensitivity.ipynb) | Sensitivity of the H/V peak to model parameters |

## Running tests

```bash
uv run pytest
uv run pytest --cov=pyhvdfa --cov-report=term-missing
```

## Building wheels

Wheels for Python 3.9–3.13 on Linux, macOS, and Windows are built automatically
by GitHub Actions on every `v*.*.*` tag and attached to the corresponding
[GitHub Release](https://github.com/dominik-strutz/pyhvdfa/releases).

To build locally:

```bash
uvx cibuildwheel --platform linux    # or macos / windows
```

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0
International License** (CC BY-NC 4.0) — see [LICENSE](LICENSE) for the full text.

The original HV-DFA Fortran code is
© Antonio García-Jerez and José Piña-Flores (UNAM / University of Almería)
and is distributed under the same CC BY-NC 4.0 license.

**Non-commercial use only.** If you need a commercial license, please contact
the authors of HV-DFA directly.

## References

- Sánchez-Sesma, F.J. et al. (2011). A theory for microtremor H/V spectral ratio:
  application for a layered medium. *Geophysical Journal International*, 186(1), 221–225.
  <https://doi.org/10.1111/j.1365-246X.2011.05064.x\>
- García-Jerez, A. et al. (2016). A computer code for forward calculation and
  inversion of the H/V spectral ratio under the diffuse field assumption.
  *Computers & Geosciences*, 97, 67–78.
  <https://doi.org/10.1016/j.cageo.2016.06.016\>
