# Sparse Linear Algebra (API)

```@meta
DocTestSetup = :(using LinearAlgebra, SparseArrays)
```

## [Sparse Linear Algebra](@id stdlib-sparse-linalg)

Sparse matrix solvers call functions from [SuiteSparse](http://suitesparse.com).

The following factorizations are available:

1. [`cholesky`](@ref SparseArrays.CHOLMOD.cholesky)
2. [`ldlt`](@ref SparseArrays.CHOLMOD.ldlt)
3. [`lu`](@ref SparseArrays.UMFPACK.lu)
4. [`qr`](@ref SparseArrays.SPQR.qr)

| Type                  | Description                                   |
|:----------------------|:--------------------------------------------- |
| `CHOLMOD.Factor`      | Cholesky and LDLt factorizations              |
| `UMFPACK.UmfpackLU`   | LU factorization                              |
| `SPQR.QRSparse`       | QR factorization                              |


```@docs; canonical=false
SparseArrays.CHOLMOD.cholesky
SparseArrays.CHOLMOD.cholesky!
SparseArrays.CHOLMOD.lowrankdowndate
SparseArrays.CHOLMOD.lowrankdowndate!
SparseArrays.CHOLMOD.lowrankupdowndate!
SparseArrays.CHOLMOD.ldlt
SparseArrays.CHOLMOD.rcond
SparseArrays.SPQR.qr
SparseArrays.UMFPACK.lu
SparseArrays.UMFPACK.rcond
```

## Using a different SuiteSparse build

The SuiteSparse libraries are loaded on first use from the copies bundled with Julia.
To use another build instead, for example one with GPU support or a development build,
point SparseArrays at a directory holding the whole set of libraries
(`libsuitesparseconfig`, `libamd`, `libcamd`, `libcolamd`, `libccolamd`, `libcholmod`,
`libspqr` and `libumfpack`) under the same file names as the bundled ones and built
from the same major SuiteSparse version. In order of precedence:

1. Call `SparseArrays.LibSuiteSparse.set_libdir!(dir)` before the first solver call.
2. Set the `JULIA_SUITESPARSE_LIBDIR` environment variable before starting Julia.
3. Set the `suitesparse_libdir` preference of SparseArrays, for example with
   [Preferences.jl](https://github.com/JuliaPackaging/Preferences.jl):
   `set_preferences!(SparseArrays, "suitesparse_libdir" => dir)`. SparseArrays must be a
   dependency of the active project. The preference is read at runtime, so no
   recompilation is needed.

The directory applies to the whole set at once, so that every library binds to the same
`libsuitesparseconfig` and the memory management functions SparseArrays installs there.
Packages that call SuiteSparse through `SuiteSparse_jll` directly are not affected.

```@docs
SparseArrays.LibSuiteSparse.set_libdir!
SparseArrays.LibSuiteSparse.libdir
```

```@meta
DocTestSetup = nothing
```
