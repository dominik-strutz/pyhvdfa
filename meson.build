project(
  'pyhvdfa',
  ['c', 'fortran'],
  version : '0.1.0',
  license : 'GPL-3.0-or-later',
  meson_version : '>=1.1.0',
  default_options : [
    'buildtype=release',
    'fortran_std=f2003',
    'warning_level=1',
    'b_lto=false',   # LTO can break gfortran + OpenMP on some distros
  ],
)

# ---- Python installation ---------------------------------------------------
py = import('python').find_installation(pure: false)
py_dep = py.dependency()

# ---- NumPy include dirs ----------------------------------------------------
# incdir_numpy  — provides numpy/arrayobject.h etc.
# incdir_f2py   — provides fortranobject.c / fortranobject.h
incdir_numpy = run_command(
  py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check : true,
).stdout().strip()

incdir_f2py = run_command(
  py,
  ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check : true,
).stdout().strip()

inc_np = declare_dependency(
  compile_args : ['-I' + incdir_numpy, '-I' + incdir_f2py],
)

# ---- Compiler flags --------------------------------------------------------
# OpenMP: enable on gfortran / ifort; gracefully absent on MSVC
fc = meson.get_compiler('fortran')
omp_dep = dependency('openmp', required : false)

# ---- Step 1: generate C wrapper from .pyf ---------------------------------
# IMPORTANT: the custom_target must live at the top level so that its CWD
# (= build root) matches where f2py writes './_hv_coremodule.c'.
_hv_core_c = custom_target(
  '_hv_coremodule.c',
  input   : ['src/pyhvdfa/hv_core.pyf'],
  output  : ['_hv_coremodule.c'],
  command : [
    py, '-m', 'numpy.f2py',
    '@INPUT@',
    '--lower',              # normalise UPPER-CASE Fortran names
    '--skip-empty-wrappers',
  ],
)

# ---- Extension module and pure-Python install ------------------------------
subdir('src/pyhvdfa')
