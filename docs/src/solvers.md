```@meta
EditURL = "https://github.com/JuliaSparse/SparseArrays.jl/blob/main/docs/src/solvers.md"
```

# [Sparse Linear Algebra](@id stdlib-sparse-linalg)

```@meta
DocTestSetup = :(using LinearAlgebra, SparseArrays)
```

Sparse factorizations call [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse):

| Function | Returns | Library |
|:---------|:--------|:--------|
| [`cholesky`](@ref SparseArrays.CHOLMOD.cholesky), [`ldlt`](@ref SparseArrays.CHOLMOD.ldlt) | [`CHOLMOD.Factor`](@ref SparseArrays.CHOLMOD.Factor) | CHOLMOD |
| [`lu`](@ref SparseArrays.UMFPACK.lu) | [`UMFPACK.UmfpackLU`](@ref SparseArrays.UMFPACK.UmfpackLU) | UMFPACK |
| [`qr`](@ref SparseArrays.SPQR.qr) | [`SPQR.QRSparse`](@ref SparseArrays.SPQR.QRSparse) | SPQR |
| [`lq`](@ref SparseArrays.SPQR.lq) | [`SPQR.AdjointQRSparse`](@ref SparseArrays.SPQR.AdjointQRSparse), the adjoint of `qr(A')` | SPQR |

## [Solving linear systems](@id man-sparse-solving)

For a sparse `A` and a dense `b`, `A \ b` picks a method from the structure of `A` and
returns a dense result. `factorize(A)` makes the same choice and returns the
factorization.

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

### Reusing a factorization

To solve several systems with one matrix, factorize once. `F \ B` takes a vector or a
matrix of right-hand sides, and `ldiv!(x, F, b)` writes into `x`.

`ldiv!` allocates scratch space on each call. To avoid that in a loop, create a workspace
once and pass it with the `workspace` keyword:
[`UMFPACK.UmfpackWS(F)`](@ref SparseArrays.UMFPACK.UmfpackWS) for `lu`,
[`CHOLMOD.CholmodWS(F)`](@ref SparseArrays.CHOLMOD.CholmodWS) for `cholesky` and `ldlt`,
and [`SPQR.SpqrWS(F)`](@ref SparseArrays.SPQR.SpqrWS) for `qr`. This works with `F`, `F'`
and `transpose(F)`, and in `ldiv!(F, b)`. A workspace grows as needed and can be reused,
but not by two calls at once.

For a new matrix with the same sparsity pattern, `lu!(F, A2)`,
[`cholesky!`](@ref SparseArrays.CHOLMOD.cholesky!)`(F, A2)` and `ldlt!(F, A2)` redo only
the numerical factorization, reusing the symbolic analysis in `F`.

```jldoctest
julia> A = sparse([2.0 1 0; 0 3 1; 1 0 4]); b = [1.0, 2.0, 3.0];

julia> F = lu(A);

julia> x = similar(b); ldiv!(x, F, b);

julia> lu!(F, 2A);

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

`F.:(:)` returns all five `lu` factors at once. The CHOLMOD factors are lazy: they support
solves, and `sparse(F.L)` for `cholesky` or `sparse(F.LD)` for `ldlt` materializes them. `F.PtL` (`P' * L`) and `F.UP`
(`L' * P`) include the permutation, and `ldlt` adds `F.D`, `F.DU`, `F.PtLD` and `F.DUP`.
The `Q` of `qr` is square and is never formed: products with it return dense arrays.

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

```@docs
SparseArrays.CHOLMOD.Factor
SparseArrays.CHOLMOD.Sparse
SparseArrays.CHOLMOD.Dense
SparseArrays.UMFPACK.UmfpackLU
SparseArrays.SPQR.QRSparse
SparseArrays.SPQR.AdjointQRSparse
SparseArrays.CHOLMOD.cholesky
SparseArrays.CHOLMOD.cholesky!
SparseArrays.CHOLMOD.lowrankupdate
SparseArrays.CHOLMOD.lowrankupdate!
SparseArrays.CHOLMOD.lowrankdowndate
SparseArrays.CHOLMOD.lowrankdowndate!
SparseArrays.CHOLMOD.lowrankupdowndate!
SparseArrays.CHOLMOD.ldlt
SparseArrays.CHOLMOD.ldlt!
SparseArrays.CHOLMOD.rcond
SparseArrays.SPQR.qr
SparseArrays.SPQR.lq
Base.:\(::SparseArrays.SPQR.QRSparse, ::StridedVecOrMat)
Base.:\(::SparseArrays.SPQR.AdjointQRSparse, ::StridedVecOrMat)
SparseArrays.UMFPACK.lu
SparseArrays.UMFPACK.lu!
SparseArrays.UMFPACK.rcond
SparseArrays.UMFPACK.UmfpackWS
SparseArrays.CHOLMOD.CholmodWS
SparseArrays.SPQR.SpqrWS
```

## Multithreading and thread safety

Solving with a factorization does not modify it, so any number of tasks can solve with one
factorization at the same time, with `\` or `ldiv!`, and the solves run in parallel. The
same holds for the other calls that only read it, such as `det`, `F.L` or `copy`, and for
factorizing different matrices from different tasks. A `workspace` passed to `ldiv!` can
only be used by one call at a time, so give each task its own.

With the default OpenBLAS, parallel solves with a supernodal `cholesky` or `ldlt` factor may
not speed up, because OpenBLAS serializes the many small BLAS calls they make
([OpenBLAS#5589](https://github.com/OpenMathLib/OpenBLAS/issues/5589)). A vendor-provided
BLAS, such as MKL through [MKL.jl](https://github.com/JuliaLinearAlgebra/MKL.jl) or Apple's
Accelerate through [AppleAccelerate.jl](https://github.com/JuliaLinearAlgebra/AppleAccelerate.jl),
can potentially give better performance.

Changing a factorization while another task uses it is not safe. This covers `lu!`,
`cholesky!`, `ldlt!`, the low-rank updates and `CHOLMOD.free!`. Synchronize those calls
yourself, or give each task its own `copy(F)`. A copy is independent: nothing done to one
affects the other. `deepcopy(F)` does the same.

```julia
using LinearAlgebra, SparseArrays

F = lu(A)
X = similar(B)
Threads.@threads for j in axes(B, 2)
    ldiv!(view(X, :, j), F, view(B, :, j))
end
```

## Tuning the factorizations

The defaults suit most problems. These keywords change them.

### `cholesky` and `ldlt`

- `perm`: a permutation of `1:size(A, 1)` to use instead of CHOLMOD's AMD ordering.
  `perm = 1:size(A, 1)` disables reordering, which usually increases fill-in.
- `shift`: factorize `A + shift*I` without forming it, for example to regularize a
  semidefinite matrix.
- `check`: see [Failures](@ref).

```jldoctest
julia> A = sparse([2.0 1 1; 1 2 0; 1 0 2]);

julia> nnz(cholesky(A)), nnz(cholesky(A; perm = 1:3))
(5, 6)

julia> B = sparse([1.0 -1; -1 1]);  # singular

julia> issuccess(cholesky(B; check = false)), issuccess(cholesky(B; shift = 1.0))
(false, true)
```

### `lu`

- `q`: an initial column ordering, which UMFPACK may still refine. `F.q` is the final one.
- `control`: UMFPACK's `Control` array. Get the defaults with
  `SparseArrays.UMFPACK.get_umfpack_control(Tv, Ti)` for the element and index types of
  `A`, and index it with the one-based constants `SparseArrays.UMFPACK.JL_UMFPACK_*`,
  such as `JL_UMFPACK_PIVOT_TOLERANCE` or `JL_UMFPACK_ORDERING`. The UMFPACK user guide
  describes each entry. The factorization keeps a copy, which later solves and
  [`lu!`](@ref) use.

Unlike UMFPACK itself, SparseArrays turns iterative refinement off by default. To turn it
back on, for example for an ill-conditioned matrix:

```julia
control = SparseArrays.UMFPACK.get_umfpack_control(Float64, Int)
control[SparseArrays.UMFPACK.JL_UMFPACK_IRSTEP] = 2  # UMFPACK's default
F = lu(A; control)
```

`SparseArrays.UMFPACK.show_umf_ctrl(F)` prints the settings of `F`, and
`SparseArrays.UMFPACK.show_umf_info(F)` prints UMFPACK's statistics for it, such as
fill-in, flop count and a condition estimate.

### `qr`

- `ordering`: the fill-reducing column ordering, one of the `SparseArrays.SPQR.ORDERING_*`
  constants: `DEFAULT` (SPQR's choice), `FIXED` and `NATURAL` (no fill-reducing
  ordering), `COLAMD`, `AMD` (on `A'A`), `METIS`, `CHOLMOD`, `BEST` (best of COLAMD, AMD
  and METIS) and `BESTAMD` (best of COLAMD and AMD).
- `tol`: columns whose norm drops to `tol` or below are treated as zero, which is how
  rank deficiency is detected. The default is
  `20 * (m + n) * eps() * maximum(norm, eachcol(A))`. Raise it for noisy data.

`rank(F)` is the numerical rank for that `tol`, and `rank(A; tol)` computes
`rank(qr(A; tol))`.

```jldoctest
julia> A = sparse([1.0 1; 1 1; 1 1 + 1e-10]);

julia> rank(qr(A)), rank(qr(A; tol = 1e-8))
(2, 1)
```

## Using a different SuiteSparse build

By default the solvers use the SuiteSparse libraries bundled with Julia. To use another
build, such as one with GPU support, point SparseArrays at a directory holding all of
`libsuitesparseconfig`, `libamd`, `libcamd`, `libcolamd`, `libccolamd`, `libcholmod`,
`libspqr` and `libumfpack`, with the bundled file names and the same major SuiteSparse
version. Either call `SparseArrays.LibSuiteSparse.set_libdir!(dir)` before the first
solver call, or set `JULIA_SUITESPARSE_LIBDIR` before starting Julia; `set_libdir!` wins.
Packages that call `SuiteSparse_jll` directly are not affected.

```@docs
SparseArrays.LibSuiteSparse.set_libdir!
SparseArrays.LibSuiteSparse.libdir
```

```@meta
DocTestSetup = nothing
```
