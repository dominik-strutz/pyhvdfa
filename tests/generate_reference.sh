#!/usr/bin/env bash
# generate_reference.sh — compile HV.f90 and regenerate reference H/V CSVs.
#
# Run this whenever fortran_src/ is updated to refresh the reference data
# that test_reference.py compares against.
#
# Usage (from the project root):
#   bash tests/generate_reference.sh
#
# Requirements:
#   gfortran  — available on PATH
#   python3   — available on PATH (only used to parse Fortran stdout)
#
# Outputs:
#   tests/reference_data/hv_<tag>.csv  — one file per model in MODELS array
#
# Parameters passed to HV.exe are chosen to match pyhvdfa.compute_hv defaults:
#   -fmin 0.2  -fmax 20.0  -nf 100  -logsam   (log-spaced frequency grid)
#   -nmr 20    -nml 20                          (Rayleigh + Love modes — matches reference example)
#   -nks 256                                    (body-wave integration steps)
#   -prec 1e-4                                  (0.0001% → G_PRECISION=1e-6, matches precision default)
#   -apsv 1e-3 -ash 1e-3                        (damping fractions, match sh_damp/psv_damp defaults)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
FORTRAN_DIR="$ROOT_DIR/fortran_src"
REF_DIR="$SCRIPT_DIR/reference_data"
BUILD_DIR="$(mktemp -d -t hv_build_XXXXXX)"
BIN="$BUILD_DIR/HV.exe"

# Models to generate (must have a matching .txt file in reference_data/models/)
MODELS=(
  simple_2layer
  model_3layer
  model_kirk
  soft_sediment
)

echo "[generate_reference] Fortran source : $FORTRAN_DIR"
echo "[generate_reference] Reference dir  : $REF_DIR"
echo "[generate_reference] Build dir      : $BUILD_DIR"
echo ""

# ── Compile ──────────────────────────────────────────────────────────────────
# HV.f90 uses bare INCLUDE 'modules.f90' etc., so we pass -I to point gfortran
# at the HV-DFA/ subdirectory that contains those files.
# Use -J to put all .mod files in the temporary build directory (not in repo root).
echo "[generate_reference] Compiling HV.f90 …"
gfortran -O2 -I "$FORTRAN_DIR/HV-DFA" -J "$BUILD_DIR" "$FORTRAN_DIR/HV-DFA/HV.f90" -o "$BIN"
echo "[generate_reference] Compiled OK → $BIN"
echo ""

# ── Generate CSVs ─────────────────────────────────────────────────────────────
mkdir -p "$REF_DIR/models"

for tag in "${MODELS[@]}"; do
  modelfile="$REF_DIR/models/${tag}.txt"
  outfile="$REF_DIR/hv_${tag}.csv"

  if [[ ! -f "$modelfile" ]]; then
    echo "[generate_reference] WARNING: model file not found, skipping: $modelfile"
    continue
  fi

  # HV.exe writes (freq hv) pairs space-separated on a single stdout line with -hv.
  # We pipe through python3 to convert to two-column CSV (no header — matches
  # the np.loadtxt call in test_reference.py).
  "$BIN" \
    -f "$modelfile" \
    -fmin 0.2 -fmax 20.0 -nf 100 -logsam \
    -nmr 20 -nml 20 \
    -nks 256 \
    -prec 1e-4 \
    -apsv 1e-3 -ash 1e-3 \
    -hv \
  | python3 -c "
import sys
vals = sys.stdin.read().split()
rows = [(vals[i], vals[i+1]) for i in range(0, len(vals)-1, 2)]
print('\n'.join(f'{f},{h}' for f, h in rows))
" > "$outfile"

  nlines=$(wc -l < "$outfile")
  echo "[generate_reference] $tag → $outfile  ($nlines lines)"
done

# ── Cleanup ───────────────────────────────────────────────────────────────────
rm -rf "$BUILD_DIR"
echo ""
echo "[generate_reference] Done. Run 'uv run pytest tests/test_reference.py' to validate."
