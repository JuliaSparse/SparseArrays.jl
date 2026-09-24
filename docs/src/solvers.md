```@meta
EditURL = "https://github.com/JuliaSparse/SparseArrays.jl/blob/main/docs/src/solvers.md"
```

# [Sparse Linear Algebra](@id stdlib-sparse-linalg)

```@meta
DocTestSetup = :(using LinearAlgebra, SparseArrays)
```

Sparse matrix solvers call functions from [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse).

The following factorizations are available:

1. [`cholesky`](@ref SparseArrays.CHOLMOD.cholesky)
2. [`ldlt`](@ref SparseArrays.CHOLMOD.ldlt)
3. [`lu`](@ref SparseArrays.UMFPACK.lu)
4. [`qr`](@ref SparseArrays.SPQR.qr)
5. [`lq`](@ref SparseArrays.SPQR.lq)

| Type                  | Description                                   |
|:----------------------|:--------------------------------------------- |
| [`CHOLMOD.Factor`](@ref SparseArrays.CHOLMOD.Factor) | Cholesky and LDLt factorizations |
| [`UMFPACK.UmfpackLU`](@ref SparseArrays.UMFPACK.UmfpackLU) | LU factorization |
| [`SPQR.QRSparse`](@ref SparseArrays.SPQR.QRSparse) | QR factorization |
| [`SPQR.AdjointQRSparse`](@ref SparseArrays.SPQR.AdjointQRSparse) | LQ factorization, the adjoint of a QR factorization |

## [Solving linear systems](@id man-sparse-solving)

### What `A \ b` does

For a sparse `A`, `A \ b` inspects the matrix and picks a method, and
`factorize` makes the same choice and returns the factorization instead of
the solution. The right-hand side is a dense vector or matrix and so is the result.

* A square `A` that is diagonal or triangular is solved by substitution, with no
  factorization.
* A square `A` that is Hermitian (symmetric, if real) is tried with
  [`cholesky`](@ref SparseArrays.CHOLMOD.cholesky). If it is not positive definite, `\`
  falls back to [`lu`](@ref SparseArrays.UMFPACK.lu), while `factorize` falls back to
  [`ldlt`](@ref SparseArrays.CHOLMOD.ldlt).
* Any other square `A` is factorized with [`lu`](@ref SparseArrays.UMFPACK.lu).
* A tall `A` is factorized with [`qr`](@ref SparseArrays.SPQR.qr), and `\` returns the
  least squares solution.
* A wide `A` is factorized with [`lq`](@ref SparseArrays.SPQR.lq), and `\` returns the
  minimum-norm solution, as dense `\` does.

The structure tests look at the stored values, so a symmetric positive definite matrix
gets a Cholesky factorization without being wrapped in `Symmetric`. `A' \ b` and
`transpose(A) \ b` make the same choices without copying `A` where the factorization of
`A` can be reused.

```jldoctest
julia> A = sparse([4.0 1 0; 1 4 1; 0 1 4]); b = [1.0, 2.0, 3.0];

julia> A \ b ≈ Matrix(A) \ b
true

julia> factorize(A) isa SparseArrays.CHOLMOD.Factor
true

julia> factorize(sparse([2.0 1 0; 0 3 1; 1 0 4])) isa SparseArrays.UMFPACK.UmfpackLU
true

julia> factorize(sparse([1.0 0; 2 1; 0 3])) isa SparseArrays.SPQR.QRSparse
true
```

An underdetermined system has many solutions. `A \ b` and `lq(A) \ b` return the one of
smallest norm, whereas `qr(A) \ b` returns a basic solution, in which the free variables
are zero.

```jldoctest
julia> A = sparse([1.0 0 1 0; 0 1 0 1]); b = [1.0, 2.0];

julia> A \ b ≈ [0.5, 1.0, 0.5, 1.0]
true

julia> qr(A) \ b ≈ [1.0, 2.0, 0.0, 0.0]
true
```

`lq(A)` is computed as the adjoint of `qr(A')`, so it costs one sparse QR factorization
and `lq(A')` reuses `qr(A)` without a copy. The `Q` of a sparse QR factorization is kept
as a product of Householder reflectors and is never formed: products of `F.Q` or `F.Q'`
with a dense or a sparse operand apply the reflectors and return a dense array.

### Reusing a factorization

When several systems share a matrix, factorize once and solve with the factorization.
`F \ B` accepts a vector or a matrix whose columns are the right-hand sides, and
`ldiv!` writes the solution into a preallocated array.

```jldoctest reuse
julia> A = sparse([2.0 1 0; 0 3 1; 1 0 4]); b = [1.0, 2.0, 3.0];

julia> F = lu(A);

julia> B = [1.0 2; 3 4; 5 6];

julia> A * (F \ B) ≈ B
true

julia> x = similar(b);

julia> ldiv!(x, F, b);

julia> A * x ≈ b
true
```

A sequence of matrices with the same sparsity pattern, as in a time-stepping or Newton
iteration, can share the symbolic analysis, which is the part that depends only on the
pattern. `lu!`, [`cholesky!`](@ref SparseArrays.CHOLMOD.cholesky!) and `ldlt!` recompute
the numerical factorization of a new matrix in an existing `F`.

```jldoctest reuse
julia> A2 = copy(A); nonzeros(A2) .*= 2;

julia> lu!(F, A2);

julia> F \ b ≈ x / 2
true

julia> S = sparse([4.0 1 0; 1 4 1; 0 1 4]);

julia> C = cholesky(S);

julia> cholesky!(C, 2S);

julia> C \ b ≈ (S \ b) / 2
true
```

### Extracting the factors

All the factorizations permute rows and columns to reduce fill-in, and `lu` also scales
the rows. The factors therefore reproduce a permuted `A`, not `A` itself, and using
`F.L` alone as if it were the factor of `A` gives wrong answers. Solve with `F \ b`
where possible, and include the permutations when the factors themselves are needed.

An `lu` factorization has sparse factors `F.L` and `F.U`, a row permutation `F.p`, a column
permutation `F.q` and a vector of row scaling factors `F.Rs`, with
`F.L * F.U == (F.Rs .* A)[F.p, F.q]`. `F.:(:)` returns all five.

```jldoctest factors
julia> A = sparse([2.0 1 0; 0 3 1; 1 0 4]);

julia> F = lu(A);

julia> F.L * F.U ≈ (F.Rs .* A)[F.p, F.q]
true
```

A `cholesky` factorization has the permutation `F.p` and the factor `F.L`, with
`L * L' == A[F.p, F.p]`. `F.L` is a lazy component that can be used in solves and
products, and `sparse(F.L)` materializes it. The combined components `F.PtL` and `F.UP`
stand for `P' * L` and `L' * P`, and can be used in solves without handling `F.p`.
An `ldlt` factorization also has `F.D` and the combinations `F.LD`, `F.DU`, `F.PtLD` and
`F.DUP`.

```jldoctest factors
julia> S = sparse([4.0 1 0; 1 4 1; 0 1 4]); b = [1.0, 2.0, 3.0];

julia> C = cholesky(S);

julia> L = sparse(C.L);

julia> L * L' ≈ S[C.p, C.p]
true

julia> C.UP \ (C.PtL \ b) ≈ S \ b
true
```

A `qr` factorization has the row and column permutations `F.prow` and `F.pcol`, the sparse
upper triangular `F.R` and the orthogonal `F.Q`, with `F.Q * F.R == A[F.prow, F.pcol]`.
`F.R` has `size(A, 2)` columns and `F.Q` is square, and the product pads `F.R` with zero
rows as needed. An `lq` factorization has `F.L`, `F.Q`, `F.prow` and `F.pcol` with
`F.L * F.Q == A[F.prow, F.pcol]`.

```jldoctest factors
julia> T = sparse([1.0 0; 2 1; 0 3]);

julia> Q = qr(T);

julia> Q.Q * Q.R ≈ T[Q.prow, Q.pcol]
true

julia> G = lq(sparse([1.0 0 1 0; 0 1 0 1]));

julia> G.L * G.Q ≈ sparse([1.0 0 1 0; 0 1 0 1])[G.prow, G.pcol]
true
```

### Practical notes

`cholesky` and `ldlt` accept a `Symmetric` or `Hermitian` view of a sparse
matrix, which reads one triangle only, or a plain sparse matrix that is itself symmetric
or Hermitian. Any other matrix throws an `ArgumentError` instead of being symmetrized.

```jldoctest notes
julia> A = sparse([4.0 1 0; 9 4 1; 9 9 4]); b = [1.0, 2.0, 3.0];

julia> cholesky(A)
ERROR: ArgumentError: sparse matrix is not symmetric/Hermitian
[...]

julia> cholesky(Symmetric(A)) \ b ≈ Matrix(Symmetric(A)) \ b
true
```

A failed factorization throws: a `PosDefException` from `cholesky`, a
`ZeroPivotException` from `ldlt` and a `SingularException` from `lu`. With
`check = false` the factorization is returned regardless, and `issuccess` tells
whether it can be used. This is how `\` falls back from `cholesky` to `lu`, and it avoids
a `try` block when trying a cheaper factorization first.

```jldoctest notes
julia> N = sparse([1.0 2; 2 1]);

julia> issuccess(cholesky(N; check = false))
false

julia> issuccess(ldlt(N; check = false))
true

julia> issuccess(lu(sparse([1.0 2; 2 4]); check = false))
false
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
```

## Multithreading and thread safety

Each factorization object carries scratch space for its solves, and some can be
refactorized in place. The rule the solver wrappers follow is that every call that reads
or changes the mutable state of a factorization holds that factorization's internal lock
for the whole call. Such calls are therefore safe on one factorization shared by several
tasks, but the ones that need the lock exclusively run one at a time. `QRSparse` and
`UmfpackLU` follow this rule; the exceptions for `Factor` are described below. The lock is
internal: there is no public interface to it. To solve in parallel, give every task its
own `copy` of the factorization:

| Type | `copy(F)` | Calls serialized by the lock of one `F` |
|:-----|:----------|:-----------------------------------------|
| `UMFPACK.UmfpackLU` | shares the matrix and the symbolic and numeric factors until either object is refactorized (copy-on-write); new workspace, `control`, `info` and lock | every call |
| `SPQR.QRSparse` | shares the factors and permutations, which never change; new, empty workspace and lock | `\`, `ldiv!`; every other call only reads the factors and needs no lock |
| `CHOLMOD.Factor` | independent deep copy of the whole factor | `ldiv!`, `cholesky!`, `ldlt!`, `lowrankupdate!`, `lowrankdowndate!` |

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
once per right-hand side.

The copies of an `UmfpackLU` are copy-on-write: calling [`lu!`](@ref) on the original or
on any copy gives that object its own matrix and factors, and the others keep solving with
the old ones, so refactorizing one object while tasks solve with its copies is safe.
Every call on an `UmfpackLU` takes its lock exclusively, including the queries and the
properties `F.L`, `F.U`, `F.p`, `F.q` and `F.Rs`, since any of them may compute missing
factors. `lu!` frees the old factors eagerly only when the object has no copies;
otherwise they are freed by the garbage collector once no copy uses them. A copy counts
until it has been finalized, and the collector does not see the memory UMFPACK allocates,
so call `GC.gc()` if a loop that copies and refactorizes large factorizations needs that
memory back sooner.

For CHOLMOD, only `ldiv!` uses the buffers stored in the `Factor`, and only `ldiv!`,
[`cholesky!`](@ref SparseArrays.CHOLMOD.cholesky!), `ldlt!`, `lowrankupdate!` and
`lowrankdowndate!` take its lock. `F \ b`, the queries on `F` and `lowrankupdowndate!` do
not, so a `Factor` is not safe to share between tasks when any of them may refactorize or
update it. Use a separate `copy(F)` per task in that case, and for
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
