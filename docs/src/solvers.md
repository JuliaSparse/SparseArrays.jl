```@meta
EditURL = "https://github.com/JuliaSparse/SparseArrays.jl/blob/main/docs/src/solvers.md"
```

# [Sparse Linear Solvers](@id stdlib-sparse-solvers)

```@meta
DocTestSetup = :(using SparseArrays, LinearAlgebra)
```

The factorizations use the CHOLMOD, UMFPACK and SPQR libraries of SuiteSparse, which are
licensed under the GPL. Julia built with `USE_GPL_LIBS=0` leaves them out, and with them
everything on this page.

## [Solving linear systems](@id man-sparse-solving)

Sparse factorizations call SuiteSparse:

| Function | Returns | Library |
|:---------|:--------|:--------|
| [`cholesky`](@ref SparseArrays.CHOLMOD.cholesky), [`ldlt`](@ref SparseArrays.CHOLMOD.ldlt) | [`CHOLMOD.Factor`](@ref SparseArrays.CHOLMOD.Factor) | CHOLMOD |
| [`lu`](@ref SparseArrays.UMFPACK.lu) | [`UMFPACK.UmfpackLU`](@ref SparseArrays.UMFPACK.UmfpackLU) | UMFPACK |
| [`qr`](@ref SparseArrays.SPQR.qr) | [`SPQR.QRSparse`](@ref SparseArrays.SPQR.QRSparse) | SPQR |
| [`lq`](@ref SparseArrays.SPQR.lq) | `SPQR.AdjointQRSparse`, the adjoint of `qr(A')` | SPQR |

For a sparse `A` and a dense `b`, `A \ b` picks a method from the structure of `A` and
returns a dense result. `factorize(A)` makes the same choice and returns the
factorization. A sparse `b` gives a sparse result, by the methods under
[Sparse right-hand sides](@ref man-sparse-rhs).

* Diagonal or triangular: substitution, no factorization.
* Hermitian (symmetric, if real): [`cholesky`](@ref SparseArrays.CHOLMOD.cholesky). If
  that fails, `\` uses [`lu`](@ref SparseArrays.UMFPACK.lu) and `factorize` uses
  [`ldlt`](@ref SparseArrays.CHOLMOD.ldlt).
* Other square: [`lu`](@ref SparseArrays.UMFPACK.lu).
* Tall: [`qr`](@ref SparseArrays.SPQR.qr), giving the least squares solution.
* Wide: [`lq`](@ref SparseArrays.SPQR.lq), giving the minimum-norm solution, as dense `\`
  does. `qr(A) \ b` instead returns a basic solution, with the free variables zero.

The structure is read from the stored values, so a symmetric matrix does not need a
`Symmetric` wrapper. `A' \ b` and `transpose(A) \ b` make the same choices.

```jldoctest
julia> factorize(sparse([4.0 1 0; 1 4 1; 0 1 4])) isa SparseArrays.CHOLMOD.Factor
true

julia> A = sparse([1.0 0 1 0; 0 1 0 1]); b = [1.0, 2.0];

julia> A \ b ≈ [0.5, 1.0, 0.5, 1.0]
true

julia> qr(A) \ b ≈ [1.0, 2.0, 0.0, 0.0]
true
```

### [Sparse right-hand sides](@id man-sparse-rhs)

For a sparse `A` and a sparse `B`, a vector or a matrix, `A \ B` returns a sparse
solution, and so do `B / A`, `A' \ B` and a solve or a right division with a triangular
wrapper of `A`.
`A` is examined and factored as a whole, whatever `B` is. The solves that follow do
work proportional to the arithmetic that the nonzeros of `B` cause, not to the size of
`A`, so a factorization kept from [`SparseArrays.sparselu`](@ref) solves a sparse
right-hand side in less time than it takes to read a dense one:

* Diagonal or triangular: substitution over the entries that the nonzeros of each
  column of `B` reach in the graph of `A`.
* Hermitian (symmetric, if real): [`cholesky`](@ref SparseArrays.CHOLMOD.cholesky), when
  the GPL-licensed solvers are available and `A` is positive definite. Otherwise it is
  solved like any other square matrix.
* Other square: [`SparseArrays.sparselu`](@ref), an LU
  factorization written in Julia that does not need the GPL-licensed solvers. A
  permutation of a triangular matrix is recognized and solved by substitution.
  Otherwise it permutes `A` to block upper triangular form with [`dmperm`](@ref),
  orders each diagonal block with AMD or COLAMD to reduce fill, factors the blocks, and
  solves only the blocks that the nonzeros of each column of `B` reach.
* Rectangular: the dense solve above, converted to sparse.

The solution is as sparse as `A` is reducible. When `A` has one irreducible block, as
the matrix of a connected mesh does, every entry of the solution is stored, and
`A \ Matrix(B)` is the better choice.

```jldoctest
julia> A = sparse([2.0 1 0 0; 1 2 0 0; 0 0 4 0; 0 0 1 5]);

julia> A \ sparsevec([3], [8.0], 4)
4-element SparseVector{Float64, Int64} with 2 stored entries:
  [3]  =  2.0
  [4]  =  -0.4

julia> length(dmperm(A).colblocks) - 1
3
```

```@docs
SparseArrays.sparselu
SparseArrays.SparseLU
```

### [Reusing a factorization](@id man-sparse-refactorize)

To solve several systems with one matrix, factorize once. `F \ B` takes a vector or a
matrix of right-hand sides, and `ldiv!(x, F, b)` writes into `x`.

`ldiv!` allocates scratch space on each call. To avoid that in a loop, create a workspace
once and pass it with the `workspace` keyword:
[`UMFPACK.UmfpackWS(F)`](@ref SparseArrays.UMFPACK.UmfpackWS) for `lu`,
[`CHOLMOD.CholmodWS(F)`](@ref SparseArrays.CHOLMOD.CholmodWS) for `cholesky` and `ldlt`,
and [`SPQR.SpqrWS(F)`](@ref SparseArrays.SPQR.SpqrWS) for `qr`. A workspace can be reused,
but not by two calls at once; `similar(ws)` makes another one.

For a new matrix with the same sparsity pattern, `lu!(F, A2)`,
[`cholesky!`](@ref SparseArrays.CHOLMOD.cholesky!)`(F, A2)` and `ldlt!(F, A2)` redo only
the numerical factorization, reusing the symbolic analysis in `F`.

The symbolic analysis is valid only for the pattern it was computed from, and an in-place
update of a `SparseMatrixCSC` can change that pattern: `A .= M .- c .* B` drops an entry
wherever the difference is zero. A [fixed-pattern matrix](@ref man-sparse-fixed) cannot
lose or gain an entry, so it can be refilled and refactorized any number of times:

```jldoctest
julia> A = SparseArrays.fixed(sparse([2.0 1 0; 0 3 1; 1 0 4])); b = [1.0, 2.0, 3.0];

julia> F = lu(A); x = similar(b); ldiv!(x, F, b);

julia> A .= 2 .* A; nnz(A)
6

julia> lu!(F, A);

julia> F \ b ≈ x / 2
true
```

### Extracting the factors

The factorizations permute rows and columns to reduce fill-in, so the factors reproduce a
permuted `A`. Using `F.L` as if it were a factor of `A` gives wrong answers. Solve with
`F \ b` where you can.

| Factorization | Factors | Relation |
|:--------------|:--------|:---------|
| `lu` | `L`, `U`, `p`, `q`, `Rs` (row scaling) | `F.L * F.U == (F.Rs .* A)[F.p, F.q]` |
| `cholesky` | `L`, `p` | `L * L' == A[F.p, F.p]` with `L = sparse(F.L)` |
| `ldlt` | `LD`, `p` | `L * D * L' == A[F.p, F.p]`, with `D` on the diagonal of `sparse(F.LD)` and the unit triangular `L` below it |
| `qr` | `Q`, `R`, `prow`, `pcol` | `F.Q * F.R == A[F.prow, F.pcol]` |
| `lq` | `L`, `Q`, `prow`, `pcol` | `F.L * F.Q == A[F.prow, F.pcol]` |

The CHOLMOD factors are lazy: they support solves, and `sparse(F.L)` for `cholesky` or
`sparse(F.LD)` for `ldlt` materializes them. `F.PtL` (`P' * L`) and `F.UP` (`L' * P`)
include the permutation. The `Q` of `qr` is square and is never formed: products with it
return dense arrays.

```jldoctest
julia> S = sparse([4.0 1 0; 1 4 1; 0 1 4]); b = [1.0, 2.0, 3.0];

julia> C = cholesky(S);

julia> sparse(C.L) * sparse(C.L)' ≈ S[C.p, C.p]
true

julia> C.UP \ (C.PtL \ b) ≈ S \ b
true
```

### Failures

`cholesky` and `ldlt` take a `Symmetric` or `Hermitian` view, which reads one triangle,
or a matrix that is itself symmetric or Hermitian. Any other matrix throws an
`ArgumentError`.

A failed factorization throws a `PosDefException` (`cholesky`), a `ZeroPivotException`
(`ldlt`) or a `SingularException` (`lu`). With `check = false` it returns anyway, and
`issuccess(F)` tells whether it can be used.

```jldoctest
julia> N = sparse([1.0 2; 2 1]);

julia> issuccess(cholesky(N; check = false)), issuccess(ldlt(N; check = false))
(false, true)
```

## Multithreading and thread safety

A solve (`\` or `ldiv!`, with `F`, `F'` or `transpose(F)`) only reads the factorization,
and so do `det`, `logdet` and the factor properties. Any number of tasks can therefore use
one factorization at the same time. Give each task its own workspace, made with
`similar(ws)`, to avoid allocating in `ldiv!`.

```julia
F = lu(A)                       # or cholesky(A), or qr(A)
X = similar(B)
Threads.@threads for j in axes(B, 2)
    ldiv!(view(X, :, j), F, view(B, :, j))
end
```

A factorization must not be refactorized (`lu!`, `cholesky!`, `ldlt!`) or updated while
another task solves with it or reads its factors. A task that does so needs its own
`copy(F)`, which is independent of `F` and duplicates the memory of the factors.
Factorizing different matrices from different tasks needs no coordination.

Solves with a supernodal `cholesky` factor, which CHOLMOD chooses for larger matrices, do
not currently speed up when run from several threads, and can slow down
([SparseArrays.jl#1044](https://github.com/JuliaSparse/SparseArrays.jl/issues/1044)).

## Tuning the factorizations

The defaults suit most problems. These keywords change them; the docstring of each
function has the detail.

- `cholesky` and `ldlt` take `perm`, a permutation to use instead of CHOLMOD's
  fill-reducing ordering, and `shift`, to factorize `A + shift*I` without forming it.
- `lu` takes `q`, an initial column ordering, and `control`, UMFPACK's vector of settings.
  Unlike UMFPACK itself, SparseArrays turns iterative refinement off by default.
- `qr` takes `ordering`, the fill-reducing column ordering, and `tol`: columns whose norm
  drops to `tol` or below are treated as zero, which is how rank deficiency is detected.
  `rank(F)` is the numerical rank for that `tol`.

```jldoctest
julia> A = sparse([2.0 1 1; 1 2 0; 1 0 2]);

julia> nnz(cholesky(A)), nnz(cholesky(A; perm = 1:3))
(5, 6)

julia> B = sparse([1.0 1; 1 1; 1 1 + 1e-10]);

julia> rank(qr(B)), rank(qr(B; tol = 1e-8))
(2, 1)
```

## Using a different SuiteSparse build

By default the solvers use the SuiteSparse libraries bundled with Julia. To use another
build, such as one with GPU support, call
[`SparseArrays.LibSuiteSparse.set_libdir!(dir)`](@ref SparseArrays.LibSuiteSparse.set_libdir!)
before the first solver call, or set `JULIA_SUITESPARSE_LIBDIR` before starting Julia.
Packages that call `SuiteSparse_jll` directly are not affected.

```@meta
DocTestSetup = nothing
```

## [Sparse Linear Solvers API](@id stdlib-sparse-solvers-api)

### CHOLMOD

```@docs
SparseArrays.CHOLMOD.Factor
SparseArrays.CHOLMOD.Sparse
SparseArrays.CHOLMOD.Dense
SparseArrays.CHOLMOD.cholesky(::Union{SparseArrays.SparseMatrixCSC{T}, SparseArrays.SparseMatrixCSC{Complex{T}}, SparseArrays.LinearAlgebra.Symmetric{T,<:SparseArrays.SparseMatrixCSC}, SparseArrays.LinearAlgebra.Hermitian{Complex{T},<:SparseArrays.SparseMatrixCSC}, SparseArrays.LinearAlgebra.Hermitian{T,<:SparseArrays.SparseMatrixCSC}}) where {T<:Real}
SparseArrays.CHOLMOD.cholesky!(::SparseArrays.CHOLMOD.Factor, ::SparseArrays.SparseMatrixCSC{Float64})
SparseArrays.CHOLMOD.ldlt(::Union{SparseArrays.SparseMatrixCSC{T}, SparseArrays.SparseMatrixCSC{Complex{T}}, SparseArrays.LinearAlgebra.Symmetric{T,<:SparseArrays.SparseMatrixCSC{T}}, SparseArrays.LinearAlgebra.Hermitian{Complex{T},<:SparseArrays.SparseMatrixCSC{Complex{T}}}, SparseArrays.LinearAlgebra.Hermitian{T,<:SparseArrays.SparseMatrixCSC{T}}}) where {T<:Real}
SparseArrays.CHOLMOD.ldlt!(::SparseArrays.CHOLMOD.Factor, ::SparseArrays.SparseMatrixCSC{Float64})
SparseArrays.CHOLMOD.lowrankupdate(::SparseArrays.CHOLMOD.Factor, ::AbstractArray)
SparseArrays.CHOLMOD.lowrankupdate!(::SparseArrays.CHOLMOD.Factor{Float64}, ::AbstractArray)
SparseArrays.CHOLMOD.lowrankdowndate(::SparseArrays.CHOLMOD.Factor, ::AbstractArray)
SparseArrays.CHOLMOD.lowrankdowndate!(::SparseArrays.CHOLMOD.Factor{Float64}, ::AbstractArray)
SparseArrays.CHOLMOD.rcond
SparseArrays.CHOLMOD.CholmodWS
```

### UMFPACK

```@docs
SparseArrays.UMFPACK.UmfpackLU
SparseArrays.UMFPACK.lu(::SparseArrays.AbstractSparseMatrixCSC{Tv,Ti}) where {Tv<:SparseArrays.UMFPACK.UMFVTypes, Ti<:SparseArrays.UMFPACK.UMFITypes}
SparseArrays.UMFPACK.lu!(::SparseArrays.UMFPACK.UmfpackLU, ::SparseArrays.AbstractSparseMatrixCSC)
SparseArrays.UMFPACK.rcond
SparseArrays.UMFPACK.UmfpackWS
Base.copy(::SparseArrays.UMFPACK.UmfpackLU)
```

### SPQR

```@docs
SparseArrays.SPQR.QRSparse
SparseArrays.SPQR.qr(::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Ti<:SparseArrays.CHOLMOD.ITypes, Tv<:Union{Float64,ComplexF64}}
SparseArrays.SPQR.lq(::SparseArrays.SparseMatrixCSC)
SparseArrays.SPQR.rank(::SparseArrays.SPQR.QRSparse)
SparseArrays.SPQR.rank(::SparseArrays.SparseMatrixCSC)
SparseArrays.SPQR.SpqrWS
Base.:\(::SparseArrays.SPQR.QRSparse, ::AbstractVecOrMat)
Base.:\(::SparseArrays.SPQR.AdjointQRSparse, ::AbstractVecOrMat)
Base.copy(::SparseArrays.SPQR.QRSparse)
```

### SuiteSparse libraries

```@docs
SparseArrays.LibSuiteSparse.set_libdir!
SparseArrays.LibSuiteSparse.libdir
SparseArrays.LibSuiteSparse.init_suitesparse
```
