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
SparseArrays.SPQR.qr
SparseArrays.UMFPACK.lu
```

```@meta
DocTestSetup = nothing
```
