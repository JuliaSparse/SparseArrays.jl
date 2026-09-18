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
with a dense or a sparse operand apply the reflectors and return a dense array. SPQR
computes in double precision. `Float32`, `Float16` and the corresponding complex inputs
are factorized as a double-precision copy and the factors are converted back to the
element type of `A`, while integer and other element types give a `Float64` factorization.

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

For an `lu` factorization, `ldiv!(F, b)` overwrites `b` with the solution. The `cholesky`,
`ldlt`, `qr` and `lq` factorizations only have the three-argument form. Pass a separate
output array: `lu`, `cholesky` and `ldlt` throw an `ArgumentError` when `x` aliases `b`.

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
