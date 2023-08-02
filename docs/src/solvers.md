# Sparse Linear Algebra

```@meta
DocTestSetup = :(using LinearAlgebra, SparseArrays)
```

Sparse matrix solvers call functions from [SuiteSparse](http://suitesparse.com). The following factorizations are available:

| Type                              | Description                                   |
|:--------------------------------- |:--------------------------------------------- |
| `CHOLMOD.Factor`      | Cholesky factorization                        |
| `UMFPACK.UmfpackLU`   | LU factorization                              |
| `SPQR.QRSparse`       | QR factorization                              |

Other solvers such as [Pardiso.jl](https://github.com/JuliaSparse/Pardiso.jl/) are available as external packages. [Arpack.jl](https://julialinearalgebra.github.io/Arpack.jl/stable/) provides `eigs` and `svds` for iterative solution of eigensystems and singular value decompositions.

These factorizations are described in more detail in [`Linear Algebra`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/) section of the manual:
1. [`cholesky`](@ref SparseArrays.CHOLMOD.cholesky)
2. [`ldlt`](@ref SparseArrays.CHOLMOD.ldlt)
3. [`lu`](@ref SparseArrays.UMFPACK.lu)
4. [`qr`](@ref SparseArrays.SPQR.qr)

```@docs
SparseArrays.CHOLMOD.cholesky
SparseArrays.CHOLMOD.cholesky!
SparseArrays.CHOLMOD.ldlt
SparseArrays.SPQR.qr
SparseArrays.UMFPACK.lu
```

```@docs
SparseArrays.CHOLMOD.lowrankupdate
SparseArrays.CHOLMOD.lowrankupdate!
SparseArrays.CHOLMOD.lowrankdowndate
SparseArrays.CHOLMOD.lowrankdowndate!
SparseArrays.CHOLMOD.lowrankupdowndate!
```

```@meta
DocTestSetup = nothing
```
