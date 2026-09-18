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
5. [`lq`](@ref SparseArrays.SPQR.lq)

| Type                  | Description                                   |
|:----------------------|:--------------------------------------------- |
| `CHOLMOD.Factor`      | Cholesky and LDLt factorizations              |
| `UMFPACK.UmfpackLU`   | LU factorization                              |
| `SPQR.QRSparse`       | QR factorization                              |
| `SPQR.AdjointQRSparse` | LQ factorization, the adjoint of a QR factorization |


```@docs; canonical=false
SparseArrays.CHOLMOD.cholesky
SparseArrays.CHOLMOD.cholesky!
SparseArrays.CHOLMOD.lowrankdowndate
SparseArrays.CHOLMOD.lowrankdowndate!
SparseArrays.CHOLMOD.lowrankupdowndate!
SparseArrays.CHOLMOD.ldlt
SparseArrays.CHOLMOD.rcond
SparseArrays.SPQR.qr
SparseArrays.SPQR.lq
Base.:\(::SparseArrays.SPQR.QRSparse, ::StridedVecOrMat)
Base.:\(::SparseArrays.SPQR.AdjointQRSparse, ::StridedVecOrMat)
SparseArrays.UMFPACK.lu
SparseArrays.UMFPACK.rcond
```

## Multithreading and thread safety

Each factorization object carries scratch space for its in-place solves, guarded by an
internal lock. Calls that take the lock are therefore safe from several tasks but run one
at a time. To solve in parallel, give every task its own `copy` of the factorization:

| Type | `copy(F)` | Calls serialized by the lock of one `F` |
|:-----|:----------|:-----------------------------------------|
| `UMFPACK.UmfpackLU` | shares the matrix and the symbolic and numeric factors; new workspace, `control`, `info` and lock | `\`, `ldiv!`, `det`, `lu!` |
| `SPQR.QRSparse` | shares the factors and permutations; new workspace and lock | `\`, `ldiv!` |
| `CHOLMOD.Factor` | independent deep copy of the whole factor | `ldiv!`, `cholesky!`, `ldlt!` |

The copies of an `UmfpackLU` or a `QRSparse` are cheap, since only the workspace is
duplicated:

```julia
using LinearAlgebra, SparseArrays

F = lu(A)                       # or qr(A)
X = similar(B)
Threads.@threads for j in axes(B, 2)
    Fj = copy(F)                # own workspace, shared factors
    ldiv!(view(X, :, j), Fj, view(B, :, j))
end
```

A loop that performs many solves per task should make the copy once per task rather than
once per right-hand side. Because the copies of an `UmfpackLU` share its factors, do not
call [`lu!`](@ref) on the original or on any copy while another task is solving with one
of them: refactorization frees the numeric object they all point to.

For CHOLMOD, only `ldiv!` uses the buffers stored in the `Factor`, and only `ldiv!`,
[`cholesky!`](@ref SparseArrays.CHOLMOD.cholesky!) and `ldlt!` take its lock. `F \ b` and
the low-rank updates do not, so a `Factor` is not safe to share between tasks when any of
them may refactorize or update it. Use a separate `copy(F)` per task in that case, and for
parallel `ldiv!`; note that this duplicates the factor's memory.

CHOLMOD and SPQR keep their parameters, statistics and error state in a `cholmod_common`
structure. SparseArrays creates one lazily for each Julia task (and index type) and keeps
it in task-local storage, so factorizing different matrices from different tasks needs no
coordination.

## Tuning the factorizations

The defaults suit most problems. The keywords below are the supported ways to change them.

### `cholesky` and `ldlt`

- `perm`: a permutation of `1:size(A, 1)` to use instead of the fill-reducing AMD ordering
  that CHOLMOD computes by default. `perm = 1:size(A, 1)` disables reordering, which
  usually increases fill-in. The ordering in use is available as `F.p`.
- `shift`: factorize `A + shift*I` without forming it, for example to regularize a
  semidefinite matrix or for shifted solves with [`cholesky!`](@ref SparseArrays.CHOLMOD.cholesky!),
  which reuses the symbolic analysis.
- `check`: with `check = false` a failed factorization does not throw; test it with
  [`issuccess`](@ref).

```jldoctest
julia> A = sparse([2.0 1 1; 1 2 0; 1 0 2]);

julia> nnz(cholesky(A)), nnz(cholesky(A; perm = 1:3))
(5, 6)

julia> B = sparse([1.0 -1; -1 1]);  # singular

julia> issuccess(cholesky(B; check = false)), issuccess(cholesky(B; shift = 1.0))
(false, true)
```

### `lu`

- `check`: as above; a singular matrix otherwise throws a `SingularException`.
- `q`: an initial column ordering that replaces UMFPACK's fill-reducing one. UMFPACK may
  still refine it during the numerical factorization; `F.q` holds the final permutation.
- `control`: the UMFPACK `Control` array as a `Vector{Float64}`. Obtain the defaults with
  `SparseArrays.UMFPACK.get_umfpack_control(Tv, Ti)`, where `Tv` and `Ti` are the element
  and index types of the matrix, and set entries through the one-based index constants
  `SparseArrays.UMFPACK.JL_UMFPACK_*` (such as `JL_UMFPACK_PIVOT_TOLERANCE`,
  `JL_UMFPACK_ORDERING`, `JL_UMFPACK_SCALE` and `JL_UMFPACK_IRSTEP`). Their meaning is
  described in the UMFPACK user guide. `lu` copies the vector into `F.control`, where
  later solves and [`lu!`](@ref) read it.

SparseArrays changes one UMFPACK default: iterative refinement is off
(`JL_UMFPACK_IRSTEP` is 0), which also lets solves use a smaller workspace. To turn it
back on, for example for ill-conditioned systems:

```jldoctest
julia> A = sparse([4.0 1 0; 1 4 1; 0 1 4]);

julia> control = SparseArrays.UMFPACK.get_umfpack_control(Float64, Int);

julia> control[SparseArrays.UMFPACK.JL_UMFPACK_IRSTEP]
0.0

julia> control[SparseArrays.UMFPACK.JL_UMFPACK_IRSTEP] = 2;  # UMFPACK's own default

julia> F = lu(A; control);

julia> F \ [5.0, 6.0, 5.0] ≈ ones(3)
true
```

`SparseArrays.UMFPACK.show_umf_ctrl(F)` prints the control settings of a factorization (or
of a control vector) and `SparseArrays.UMFPACK.show_umf_info(F)` prints the statistics
UMFPACK recorded for it, such as the fill-in, the flop count and a condition estimate.

### `qr`

- `ordering`: the fill-reducing column ordering, one of the constants in `SparseArrays.SPQR`:
  `ORDERING_DEFAULT` (SuiteSparseQR's own choice, the default), `ORDERING_FIXED` and
  `ORDERING_NATURAL` (keep the given column order), `ORDERING_COLAMD`, `ORDERING_AMD` (AMD on
  `A'A`), `ORDERING_METIS`, `ORDERING_CHOLMOD` (CHOLMOD's strategy), `ORDERING_BEST` (try
  COLAMD, AMD and METIS and keep the best) and `ORDERING_BESTAMD` (try COLAMD and AMD).
- `tol`: columns whose norm falls to `tol` or below during the factorization are treated as
  zero, which is how SPQR detects rank deficiency. The default is
  `20 * (m + n) * eps() * maximum(norm, eachcol(A))`. Raise it when the data carry noise
  well above rounding error.

`rank(F)` returns the numerical rank found for the chosen `tol`, and `rank(A; tol)` is a
shorthand for `rank(qr(A; tol))`.

```jldoctest
julia> A = sparse([1.0 1; 1 1; 1 1 + 1e-10]);

julia> rank(qr(A)), rank(qr(A; tol = 1e-8))
(2, 1)
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
