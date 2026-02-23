# src/pyhvdfa/meson.build
#
# Builds the _hv_core Fortran/C extension module and installs the pure-Python
# sources alongside it.
#
# Build steps
# -----------
# 1. f2py reads hv_core.pyf → generates _hv_coremodule.c  (C-side Python bridge)
# 2. gfortran compiles hv_wrapper.f90 (which INCLUDEs the HV-DFA source tree)
# 3. gcc compiles _hv_coremodule.c + fortranobject.c
# 4. Linker produces _hv_core.<platform_tag>.so

# _hv_core_c (the f2py-generated C file) is defined in the parent meson.build
# and is therefore in scope here.

# ---- Path helpers ----------------------------------------------------------
fortran_wrapper = meson.project_source_root() / 'fortran_src' / 'hv_wrapper.f90'

# ---- Compile and link the extension module --------------------------------
_hv_core_sources = [
  fortran_wrapper,                       # all HV-DFA code via INCLUDE + COMPUTE_HV
  _hv_core_c,                            # f2py-generated C glue
  incdir_f2py / 'fortranobject.c',       # f2py runtime support
]

# Fortran extra args.  -fopenmp must appear for both compile and link.
_f_extra_args = []
_link_args    = []
if omp_dep.found()
  _f_extra_args += fc.get_supported_arguments('-fopenmp')
  _link_args    += ['-lgomp']   # auditwheel will bundle libgomp into the wheel
endif

py.extension_module(
  '_hv_core',
  sources      : _hv_core_sources,
  dependencies : [py_dep, omp_dep, inc_np],
  fortran_args        : _f_extra_args,
  link_args           : _link_args,
  install             : true,
  subdir              : 'pyhvdfa',
)

# ---- Install the pure-Python package files --------------------------------
py.install_sources(
  [
    '__init__.py',
    '_types.py',
    '_hv_core.pyi',
  ],
  subdir : 'pyhvdfa',
)
