# SparseArrays.jl release history

SparseArrays is a Julia standard library. Each Julia release ships one fixed version of it,
numbered like the release: SparseArrays v1.14 ships with Julia v1.14. Its code lived in the
Julia repository until Julia v1.8, and the entries for those releases link to Julia pull
requests.

Each change is listed under the first Julia v1.x.0 release that shipped it. Fixes backported
to patch releases are not listed again, and a bug fix is listed only when it changes a
result that code could have relied on. Internal and experimental features are not listed.

Each change is marked **New**, **Changed**, **Faster**, **Deprecated** or **Removed**, and
the larger releases group their changes by topic. v1.14 starts with its highlights and the
changes that can require updating code written for v1.13.

## Unreleased

Changes on `main` that no release branch has yet.

* **Changed:** `sprand` and `sprandn` accept a matrix of 2^63 or more entries again, drawing
  it column by column, instead of throwing an `OverflowError` ([#1143]).
* **Faster:** The package image holds native code for `show`, `\`, products, broadcast,
  concatenation and `sprand` of sparse `Float64` and `ComplexF64` arrays, so their first
  call compiles less ([#1103]).
* **Faster:** `transpose(A) * x`, `A' * x` and `mul!(y, A', x)` with a sparse matrix `A` and
  a sparse vector `x` no longer merge `x` with every column of `A`, which is much faster
  when `A` has many columns ([#1139]).

## Julia v1.14

Ships with Julia v1.14.0, not yet released. SuiteSparse 7.14.0; SparseArrays requires
SuiteSparse 7.0 or later ([#806]).

### Highlights

* A sparse result that would store every entry is dense: sparse `+` and `-` with a dense
  array, broadcasts with a scalar or a dense array, and broadcasts of a function that is not
  zero at zero, such as `cos.(A)`.
* Products and solves follow the dense factor: sparse times dense is dense, sparse times a
  banded matrix stays sparse, and solves return dense arrays.
* One factorization can be used for solves from several tasks at once.
* Views, adjoints and transposes of sparse arrays, and `Symmetric`, `Hermitian` and
  triangular wrappers, use the sparse kernels instead of fallbacks that read every element.
* In-place functions check their arguments before writing, and leave the destination
  unchanged when they throw.
* `getcolptr`, `getrowval` and `getnzval` are exported, and are the interface for custom
  `AbstractSparseMatrixCSC` types.
* New: `lq`, `opnorm`, `rcond`, sorting along a dimension, sparse results from reductions
  with `sparse=true`, and `ldiv!` without allocation through a `workspace` keyword.
* The package compiles with `juliac --trim=safe`.

### Upgrading from v1.13

```julia
A = sparse([1.0 0; 0 2])
A .+ 1           # v1.13: SparseMatrixCSC with 4 stored entries; v1.14: Matrix{Float64}
cos.(A)          # v1.13: SparseMatrixCSC with 4 stored entries; v1.14: Matrix{Float64}
A + ones(2, 2)'  # v1.13: SparseMatrixCSC with 4 stored entries; v1.14: Matrix{Float64}
A[true, 1]       # v1.13: 1.0; v1.14: ArgumentError
```

* Sparse `+` and `-` with any strided dense array, including its views, adjoints and
  transposes, broadcast `+` and `-` with a dense array or a scalar, and broadcasts of a
  function that is not zero at zero (`cos.(A)`, `A ./ A`) return a dense array instead of a
  sparse one that stores every entry ([#949], [#1004], [#1086]).
* `+` and `-` of a sparse vector and a dense vector return a dense `Vector` ([#887]).
* A `Bool` is not accepted as a scalar index in `getindex`, `setindex!` and `dropstored!`
  ([#1067]).
* `findall` of a `SparseVector` returns a `Vector{Int}`, whatever the index type of the
  vector ([#923]).
* `any` and `all` of a sparse array with non-Boolean values throw a `TypeError`, as for
  dense arrays ([#1056]).
* `sparsevec(A)` of a sparse matrix returns a `SparseVector` instead of a reshaped array
  ([#860]).
* `rowvals`, `nonzeros` and `nzrange` of an `UpperTriangular` or `LowerTriangular` wrapper
  of a sparse matrix throw a `MethodError` instead of returning the parent's storage,
  entries outside the triangle included ([#893], [#1050]).
* `sprand` and `sprandn` of a matrix with 2^63 or more entries throw an `OverflowError`
  ([#1137]).
* A subtype of `AbstractSparseMatrixCSC` that implements `rowvals` and `nonzeros` instead of
  `getrowval` and `getnzval` gets a deprecation warning ([#1011]).

### Construction and conversion

* **New:** `sparse` accepts a vector of `CartesianIndex` positions in place of the row and
  column index vectors ([#600]).
* **New:** `sparse!` and `spzeros!` are public ([#668]).
* **New:** `SparseMatrixCSC{Tv}(M::AbstractMatrix)` converts the element type to `Tv`
  ([#927]).
* **New:** `promote_type` and `promote` of `SparseMatrixCSC` and `SparseVector` types with
  different element or index types return a concrete sparse type ([#807]).
* **Changed:** `repeat` of a sparse matrix, and column slices such as `A[:, j]`, keep the
  index type of the matrix ([#919], [#945]).
* **Changed:** `SparseMatrixCSC` of a view of a sparse vector or column returns an `n×1`
  matrix ([#1113]).
* **Changed:** `one(A)` puts the multiplicative identity of the element type on the diagonal
  when it is not `1` ([#732]).
* **Changed:** `sprand` with a value function honours the element type `T` and the
  documented forms of `rfn` ([#1109]).
* **Faster:** `stack` of sparse vectors builds the sparse result directly ([#836]).

### Indexing and assignment

* **New:** `getindex` accepts `StepRangeLen` row ranges, a vector of `CartesianIndex{1}` as
  a row or column index, and a `Bool` unit range as a mask ([#1057], [#1058], [#1062]).
* **New:** `getindex` lowers its indices with `to_indices`, as `setindex!` does, so it
  accepts custom index types such as `InvertedIndices.Not` ([#857]).
* **Changed:** In-place functions (`copyto!`, `setindex!`, `map!`, `broadcast!`, `mul!`,
  `ldiv!`, `circshift!`, `sparse!`, `fkeep!`, `dropstored!`, `lu!` and others) check sizes,
  aliasing, value conversion and index-type overflow before writing, and throw with the
  destination unchanged ([#1019], [#1060], [#1110]).
* **Changed:** Scalar and nonscalar `setindex!` both store `-0.0` and `missing`, and both
  drop a heap-allocated zero such as `big(0)` ([#891], [#896], [#898]).
* **Changed:** Scalar `setindex!` at an index that does not fit the index type throws a
  `BoundsError` ([#1145]).
* **Changed:** `getindex` with an index array of zero or three or more dimensions returns a
  dense array ([#1059]).
* **Changed:** Reductions and linear indexing work on sparse matrices of 2^63 or more
  entries; linear indexing of one that `Int` cannot index throws an `OverflowError`
  ([#1128]).
* **Faster:** `A[I, J]` with a sorted `I` that selects few columns, or on a hypersparse
  matrix, does not allocate a cache of length `size(A, 1)` ([#870]).
* **Faster:** `copyto!` of a sparse source into a view of a sparse array, and ranged `fill!`
  on a sparse view, merge in place in one pass ([#799], [#897]).
* **Faster:** Row indexing of the adjoint or transpose of a sparse matrix ([#734]).

### Views, wrappers and storage

* **New:** `AbstractSparseMatrixCSC`, `getcolptr`, `getrowval`, `getnzval` and `indtype` are
  exported and documented. `getcolptr`, `getrowval` and `getnzval` are the storage
  interface: a subtype that defines them works with every function of the package, and
  `getrowval` and `getnzval` also accept sparse vectors and their views ([#767], [#805],
  [#1008]).
* **New:** `nonzeros`, `rowvals` and `nzrange` of a view of some columns of a sparse matrix
  describe only the entries inside the view, and `nnz` of a triangular, `Symmetric` or
  `Hermitian` wrapper of a sparse matrix counts the entries of its triangle ([#893]).
* **Faster:** Views of sparse matrices and vectors, and lazy adjoints and transposes, use
  the sparse kernels in reductions, products, `norm`, `diag`, `istriu`, `istril`,
  `issymmetric`, `ishermitian`, `count`, `==`, `reverse`, `circshift`, `permutedims`,
  `sort`, `findnext`, `findprev`, `spdiagm` and conversion, instead of reading every element
  ([#969], [#1024], [#1119]).
* **Faster:** `Matrix` and `Vector` of views, adjoints and transposes of sparse arrays fill
  the result and then visit only the stored entries ([#828]).

### Arithmetic and broadcasting

* **New:** `map` and `broadcast` over sparse arrays accept functions that return values of a
  type without `zero`, such as `String`, `Tuple`, `Pair` or `nothing` ([#1055]).
* **Changed:** `map` and broadcast over a single sparse array keep its stored entries,
  including stored zeros and entries where `f` returns zero, as `float(A)`, `-A` and `2A` do
  ([#892]).
* **Changed:** Broadcasting a sparse array against a dense array does not store zeros such
  as `0 / x` where the sparse argument has no entry ([#886]).
* **Changed:** Broadcasting a sparse array against a view of a dense array gives the same
  kind of result as against the dense array itself, and broadcasting over sparse views of
  whole columns or of a sparse vector returns a sparse result ([#772], [#917]).
* **Changed:** Scaling a sparse array by `NaN` or `Inf` with `rmul!`, `lmul!`, `mul!` or `x
  * NaN`, the outer product `x .* y'`, and `prod` propagate the non-finite value to the
  unstored entries, as for dense arrays ([#1065], [#1077], [#1101]).
* **Changed:** `kron` with a view of a sparse matrix, and `diff` of a sparse view, transpose
  or adjoint, return sparse results ([#953], [#1078], [#1118]).
* **Faster:** Broadcasts of a sparse array with a scalar or a dense array, such as `A .+ 1`,
  skip the dense copy of the sparse operand ([#982], [#983], [#1105]).
* **Faster:** Sparse broadcast grows its result storage on demand instead of allocating the
  upper bound on the number of stored entries ([#803]).
* **Faster:** `+` and `-` of sparse vectors and their views are one merge of the stored
  entries ([#954]).
* **Faster:** `reverse`, `circshift` and rotations of sparse arrays, and broadcasts that
  scale the rows of a sparse matrix by a vector, visit only the stored entries ([#888],
  [#955]).
* **Faster:** `map!` and `broadcast!` into a sparse array do not allocate to unalias their
  inputs ([#1013]).

### Reductions, sorting and comparison

* **New:** Reductions along a dimension (`sum`, `prod`, `maximum`, `minimum`, `count`,
  `any`, `all` and `mapreduce`) take a `sparse` keyword that returns a sparse result instead
  of a dense one ([#798]).
* **New:** `sort` and `sort!` of a `SparseMatrixCSC` along a dimension ([#784], [#786]).
* **New:** `findnz` and `isequal` accept adjoints and transposes of sparse matrices and
  vectors ([#765], [#768]).
* **Changed:** `mapreduce` over sparse arrays folds the entries in element order unless `op`
  is known to commute, so a non-commutative `op` gives the dense result ([#1006], [#1063]).
* **Changed:** `sum`, `prod` and `mapreduce` with `+` or `*` over sparse arrays of small
  integers widen the unstored entries as for dense arrays, so their result types match
  ([#932]).
* **Changed:** `==` and `isequal` of two sparse arrays compare the implicit zeros of their
  element types ([#985]).
* **Changed:** `maximum`, `minimum` and `extrema` along a dimension of an empty sparse
  array, and reductions along a dimension of the adjoint or transpose of a sparse vector,
  return a dense `Array` ([#928], [#935]).
* **Changed:** `findmin` and `findmax` of a `SparseVector` always return an `Int` index, and
  apply `f` to a vector with no stored entries ([#922]).
* **Faster:** `isequal`, `findnext`, `findprev` and `hash` of sparse arrays visit only the
  stored entries ([#766], [#769]).
* **Faster:** `mapreduce(f, op, A, B)` over two sparse arrays reduces without building
  `map(f, A, B)` ([#1002]).

### Concatenation

* **New:** `sparse_hcat`, `sparse_vcat` and `sparse_hvcat` accept `UniformScaling` arguments
  ([#920]).
* **Changed:** Loading SparseArrays no longer changes the result or the speed of dense
  `hcat`, `vcat`, `hvcat` and `cat`, nor makes those methods of other packages' array types
  ambiguous ([#890], [#1157], [#1160]).
* **Changed:** A leading number in a sparse concatenation fills its block as in dense
  concatenation ([#890]).
* **Changed:** Concatenations of sparse arrays with dense arrays, numbers, `I` or a
  `Diagonal` keep the stored zeros of the sparse arguments ([#1102]).
* **Changed:** The index type of a mixed sparse and dense concatenation comes from its
  sparse arguments only ([#1127]).
* **Changed:** A concatenation of a leading sparse array with an array of non-`Number`
  element type returns a dense array instead of throwing ([#975]).

### Products and linear algebra

* **New:** `opnorm(A)`, the 2-norm, of a sparse matrix returns an estimate of the largest
  singular value instead of throwing ([#859]).
* **New:** `kron!` accepts a dense vector operand next to a sparse one ([#988]).
* **Changed:** Products of a sparse matrix with a structured or dense matrix take their
  result type from the other factor: sparse times dense is dense, sparse times a banded
  structured matrix stays sparse. This holds on either side and through adjoint, transpose,
  triangular, `Symmetric` and `Hermitian` wrappers and row views ([#693], [#1075]).
* **Changed:** Solves of structured matrices with sparse matrices return dense arrays, for
  left and right division ([#708]).
* **Changed:** `cond` of a singular sparse matrix returns `Inf` instead of throwing, and
  `cond` of an empty one returns zero ([#989], [#1126], [#1131]).
* **Changed:** `D \ A` and `A / D` with a sparse `A` and a singular `Diagonal` `D` throw
  `SingularException` ([#944]).
* **Faster:** Sparse `A*B` sizes its result from a sample of its columns and releases spare
  capacity, which cuts the memory it holds several times over where rows collide, as for a
  stencil ([#1142]).
* **Faster:** `mul!` of sparse matrices into dense or sparse destinations, and products with
  `Symmetric`, `Hermitian` and triangular sparse matrices, use the sparse kernels instead of
  generic fallbacks ([#861], [#864], [#865]).
* **Faster:** Products of a sparse matrix with a `Bidiagonal`, `Tridiagonal` or
  `SymTridiagonal` matrix visit only the stored entries ([#1084]).
* **Faster:** Products with `Diagonal`, `dot` with adjoints and transposes of sparse
  matrices, and products of mixed adjoint and transpose operands and of column views
  ([#770], [#858], [#866]).

### Factorizations and solvers

* **New:** One `UmfpackLU`, `QRSparse` or `CHOLMOD.Factor` can be used for solves from
  several tasks at once, and reading an `UmfpackLU` (`det`, `logabsdet`, its factors) takes
  no lock. A factorization must not be refactorized while another task uses it; such a task
  works on its own `copy(F)` ([#906], [#1041], [#1045]).
* **New:** `ldiv!` with a CHOLMOD, UMFPACK or SPQR factorization takes a `workspace`
  keyword, so that repeated solves do not allocate ([#727], [#914]).
* **New:** `lq` of a sparse matrix, computed from the sparse QR of its adjoint ([#833]).
* **New:** `qr(A)' \ b`, and products of the `Q` of a sparse QR factorization, its adjoint
  and its transpose with sparse operands, which apply the Householder reflectors instead of
  forming `Q` ([#804], [#830]).
* **New:** `ldlt`, `lu` and `inv` accept `Adjoint`, `Transpose`, `Symmetric`, `Hermitian`
  and view wrappers of sparse matrices, and `lu` of a wrapped sparse matrix returns a sparse
  factorization ([#855]).
* **New:** `rcond` of a `CHOLMOD.Factor` or an `UmfpackLU` returns the solver's estimate of
  the reciprocal condition number ([#776]).
* **Changed:** `A \ b` with a wide sparse `A` returns the minimum-norm solution, as dense
  `\` does ([#831]).
* **Changed:** Sparse `\`, its adjoint and transpose forms, and `qr(A) \ b` return the
  element type of dense `\`, whatever the shape of `A` ([#950], [#1124]).
* **Changed:** `\` with a sparse right-hand side, and `sparse(F.L)` of a factorization,
  return a `SparseMatrixCSC` ([#933]).
* **Changed:** Sparse `\` and `factorize` treat a Hermitian matrix as Hermitian only for
  floating-point element types; integer matrices take the general sparse path ([#921]).
* **Changed:** `factorize` of a sparse indefinite Hermitian matrix whose LDLt fails falls
  back to `lu` instead of throwing `ZeroPivotException` ([#1116]).
* **Changed:** The sparse triangular solve throws `SingularException` for a stored zero on
  the diagonal, as for a missing diagonal entry ([#1066]).
* **Changed:** `copy` of an `UmfpackLU` or a `QRSparse` and `deepcopy` of a `CHOLMOD.Factor`
  return independent factorizations ([#1030], [#1031], [#1032]).
* **Faster:** The sparse triangular solve finds the diagonal of a stored triangle without a
  search ([#1081]).

### Display

* **Changed:** `show` of a sparse matrix uses Base's display pipeline: the spy plot uses the
  full display width, and stored zeros print centred at a fixed width ([#736]).
* **Changed:** Sparse views and `Hermitian`, `Symmetric`, triangular and `Diagonal` wrappers
  of sparse arrays print with the sparse display, and the `Q` of a sparse `qr` or `lq`
  prints like any other `AbstractQ` ([#978], [#1025], [#1036]).

### Compilation

* **New:** Dense and sparse `cat` and `hvcat`, sparse block literals, the SuiteSparse
  library paths and the main sparse operations compile with `juliac --trim=safe` ([#972],
  [#973], [#1028]).

## Julia v1.13

Released with Julia v1.13.0 on 2026-09-09. SuiteSparse 7.10.1.

* **New:** `ldiv!` solves in place with a sparse `qr` factorization (`SPQR.QRSparse`)
  ([#679]).
* **New:** 3-argument `dot(x, A, y)` with a sparse `Symmetric` or `Hermitian` `A` accepts
  element types other than real and complex numbers, such as quaternions ([#643]).
* **New:** `propertynames` of a CHOLMOD factorization lists its components, and the
  `cholesky` and `ldlt` docstrings describe them ([#645]).
* **New:** `sparse`, `spdiagm`, `sparsevec`, `mapreduce` and the SPQR factorization can be
  compiled with `juliac --trim` ([#751], [#753], [#754]).
* **Changed:** `A \ b` with a sparse `Float32` or `Float16` matrix returns a result of that
  precision, instead of `Float64` ([#684]).
* **Changed:** `zero` of a sparse array keeps its index type ([#710]).
* **Changed:** `findmin` and `findmax` of sparse arrays treat `NaN` as dense arrays do
  ([#722]).
* **Faster:** Sparse-times-dense and dense-times-sparse matrix multiplication ([#666]), and
  multiplication of a sparse `Symmetric` or `Hermitian` matrix by an adjoint or transposed
  dense matrix ([#692]).
* **Faster:** `isdiag` of a sparse matrix visits only the stored entries ([#672]).
* **Faster:** 5-argument `mul!` with a `Diagonal` and a sparse matrix, and 3-argument
  `dot(x, A, y)` with sparse vectors and a dense matrix, visit only the stored entries
  ([#603], [#640]).
* **Faster:** `ldiv!` with a `QRSparse` does not allocate a square copy of `R` ([#697]).
* **Faster:** CHOLMOD and the other SuiteSparse libraries are initialized on first use
  instead of at load time ([#626], [#677]).
* **Faster:** Loading SparseArrays causes fewer invalidations ([#694]).

## Julia v1.12

Released with Julia v1.12.0 on 2025-10-07. SuiteSparse 7.8.3.

* **New:** `ldiv!` solves in place with a CHOLMOD `cholesky` or `ldlt` factorization
  ([#547]).
* **New:** `qr` of a sparse `Float32` or `ComplexF32` matrix factorizes in single precision;
  `Float16` and `ComplexF16` matrices are converted to single precision ([#586], [#597]).
* **New:** `CHOLMOD.symbolic` takes a `nested_dissection` keyword to enable nested
  dissection ordering ([#550]).
* **New:** `SparseMatrixCSC{Tv,Ti}(undef, (m, n))` and `SparseVector{Tv,Ti}(undef, (n,))`
  accept a tuple of dimensions, so `similar(SparseMatrixCSC{Tv,Ti}, dims)` works ([#523]).
* **New:** `ldiv!` and `rdiv!` with a `Diagonal` and a sparse matrix accept differing
  element types when the result can be stored in the destination ([#616]).
* **Changed:** CHOLMOD does not try nested dissection ordering by default, which could be
  slow for large matrices ([#550]).
* **Changed:** The default `tol` of sparse `qr` uses `eps()` of `Float64` for every element
  type, as SPQR itself does ([#557]).
* **Faster:** `repeat` of sparse matrices and vectors builds the result column by column
  ([#532]).
* **Faster:** `copytrito!` from a sparse source visits only the stored entries ([#533]).
* **Faster:** `Diagonal * S * Diagonal` with a sparse `S`, `Diagonal` times a sparse vector,
  3-argument `dot(x, D::Diagonal, y)` with a sparse vector, and `ldiv!(D::Diagonal, x)` with
  a sparse vector visit only the stored entries ([#562], [#564], [#565]).
* **Faster:** `istriu(S, k)` and `istril(S, k)` with a band index use the sparse kernel
  ([#590]).
* **Faster:** 5-argument `mul!` of a sparse matrix and a dense vector or matrix does not
  allocate ([#567]).
* **Faster:** Loading SparseArrays causes fewer invalidations of the system image ([#649],
  [#650], [#652]).

## Julia v1.11

Released with Julia v1.11.0 on 2024-10-07. SuiteSparse 7.7.0.

* **New:** CHOLMOD factorizations (`cholesky`, `ldlt`) accept `Float32` and `ComplexF32`
  matrices and factorize them in single precision ([#487]).
* **New:** `lu`, `qr` and `cholesky` of wrapped sparse matrices, such as `Symmetric` and
  `Hermitian` wrappers, reach the sparse solvers through LinearAlgebra's factorization
  dispatch ([#437]).
* **New:** `+`, `-` and other algebraic operators on column views of a `SparseMatrixCSC`
  (`view(S, :, cols)`) return sparse results ([#458]).
* **New:** `kron` of a sparse matrix with an adjoint or transposed dense matrix returns a
  sparse matrix ([#474]).
* **Changed:** `show` of a `SparseVector` in compact or one-line contexts, such as inside a
  container, prints a round-trippable `sparsevec(indices, values, n)` ([#427]).
* **Changed:** Views of a contiguous range of a sparse vector behave as sparse vectors:
  `nnz`, `nonzeros`, `nonzeroinds` and the sparse kernels apply to them ([#416]).
* **Changed:** `reverse` of a sparse vector or matrix returns a sparse array with the same
  number of stored entries, instead of storing explicit zeros ([#450]).
* **Faster:** Adding or subtracting a sparse matrix and a dense `Array` does not materialize
  the sparse operand as a dense matrix first ([#456]).
* **Faster:** Multiplying by a column view of a sparse matrix with a non-contiguous column
  selection uses the sparse kernels instead of a generic fallback ([#476]).
* **Faster:** `findmin` and `findmax` of sparse matrices, with or without `dims`, do not
  allocate for every comparison ([#511]).
* **Faster:** Concatenation of sparse vectors infers its result type ([#422]).

## Julia v1.10

Released with Julia v1.10.0 on 2023-12-25. SuiteSparse 7.2.1.

### Loading

* **Changed:** SparseArrays and the SuiteSparse libraries are no longer in the default
  system image, so the core language no longer contains GPL libraries; they still ship with
  the standard binary distribution ([JuliaLang/julia#44247], [JuliaLang/julia#48979],
  [JuliaLang/julia#49266]).

### Construction and conversion

* **New:** `spzeros([T,] I, J, m, n)` creates a sparse matrix with structural zeros at the
  positions `(I[k], J[k])`, and `SparseArrays.spzeros!` does so into preallocated buffers
  ([#284], [#310], [#315]).
* **New:** `SparseArrays.sparse!(I, J, V)` and `SparseArrays.spzeros!(I, J)` build a sparse
  matrix reusing the memory of the input vectors ([#397]).
* **New:** `sparse`, `SparseMatrixCSC` and `SparseMatrixCSC{Tv,Ti}` construct a sparse
  matrix from an `AbstractQ` ([#203], [#346]).
* **Changed:** `float(S)` and `complex(S)` return `S` itself when its eltype is already
  floating point or complex, like for dense arrays ([#340]).
* **Faster:** `collect`, `Vector` and `Matrix` of sparse arrays, and 3-argument `dot` with a
  sparse `Hermitian` or `Symmetric` matrix ([#312], [#328], [#398]).

### Arithmetic and broadcasting

* **Changed:** `map` keeps the index type of its sparse arguments ([#351]).

### Reductions, sorting and comparison

* **Changed:** `sort!` of a sparse vector keeps its zeros implicit instead of storing them
  ([#303]).

### Concatenation

* **Faster:** Concatenation of dense arrays and numbers with SparseArrays loaded, and
  concatenation of many arguments infers better ([#384], [#407], [#432]).

### Products and linear algebra

* **Changed:** `kron` with one sparse and one dense factor returns a sparse matrix ([#331]).
* **Faster:** Products of an `AbstractQ` with a sparse array no longer fall back to
  elementwise generic multiplication ([#317]).

### Factorizations and solvers

* **New:** CHOLMOD and SPQR accept `Int32` index types as well as `Int64` ([#380]).
* **New:** Solving with the adjoint of a real CHOLMOD `Factor` accepts a complex right-hand
  side ([#327]).
* **Changed:** Solving with a CHOLMOD factorization and a vector right-hand side returns a
  vector ([#320]).
* **Changed:** `adjoint` and `transpose` of an `UmfpackLU` or a CHOLMOD `Factor` return
  LinearAlgebra's `AdjointFactorization` and `TransposeFactorization`, and the `Q` of a
  sparse QR factorization is an `AbstractQ`, following LinearAlgebra ([#203], [#269],
  [#347]).

### Display

* **Changed:** Sparse matrices nested inside other containers print according to the
  `IOContext`, for example compactly ([#423]).

## Julia v1.9

Released with Julia v1.9.0 on 2023-05-07. SuiteSparse 5.10.1.

### Construction and conversion

* **New:** `SparseMatrixCSC{Tv,Ti}(undef, m, n)` and `SparseVector{Tv,Ti}(undef, n)` create
  empty sparse arrays ([#263]).
* **New:** `reinterpret` works on sparse arrays ([#296]).
* **Changed:** `length(::SparseVector{Tv,Ti})` returns a `Ti` ([#176]).
* **Changed:** `sparse!` resizes its output buffers to the lengths the result needs
  ([#314]).

### Indexing and assignment

* **Changed:** Assigning `-0.0` (or any value that is `==` but not `===` to zero) to a
  sparse array stores it ([#296]).
* **Faster:** `transpose!` and `adjoint!` no longer allocate, `cat` of sparse arrays is type
  stable, and `getindex` elides bounds checks it has already made ([#92], [#157], [#273]).
* **Deprecated:** `fkeep!(A, f)` is deprecated in favor of `fkeep!(f, A)` ([#240]).

### Arithmetic and broadcasting

* **Changed:** `map` on sparse arrays determines the result eltype from the values `f`
  returns, and calls `f` on zero only when the result needs it ([#177], [#222]).
* **Changed:** `circshift` shifts in the same direction as for dense arrays ([#260],
  [#272]).
* **Changed:** Adding a `UniformScaling` to a sparse matrix keeps its index type ([#158]).

### Reductions, sorting and comparison

* **New:** `findnz` and reductions work for any `AbstractSparseVector` ([#175]).
* **Changed:** `all` and `any` of an empty sparse array, `findmin`, `findmax`, and
  reductions with an `init` value return the same results as for dense arrays ([#93],
  [#271], [#366]).
* **Faster:** `==` between sparse arrays and reductions such as `sum` and `maximum` visit
  only the stored entries ([#227], [#244], [#248]).

### Products and linear algebra

* **New:** `rank` of a sparse matrix and of a sparse QR factorization takes a tolerance
  `tol` ([#195], [#228]).
* **Faster:** `kron` with `Bidiagonal`, `Tridiagonal` and `SymTridiagonal` matrices, and
  `SparseMatrixCSC(::AbstractQ)` and products with `AbstractQ`, use the structure of their
  arguments ([#196], [#236]).

### Factorizations and solvers

* **New:** The SuiteSparse solver wrappers moved from the SuiteSparse standard library into
  SparseArrays and are available as its submodules `SparseArrays.CHOLMOD`,
  `SparseArrays.UMFPACK` and `SparseArrays.SPQR`; SuiteSparse.jl re-exports them ([#95],
  [#144]).
* **New:** `lu(A; q)` and `lu!(F, A; q)` take a user-supplied column permutation `q` for the
  UMFPACK symbolic factorization ([#165]).
* **New:** `lu!(F::UmfpackLU)` refactorizes with the existing symbolic factorization, and
  `UmfpackLU` gains a constructor that does not factorize ([#224], [#276]).
* **New:** `copy(F::UmfpackLU; copynumeric, copysymbolic)` controls whether the numeric and
  symbolic factorizations are copied ([#259]).
* **New:** UMFPACK `lu` and `lu!` accept any `AbstractSparseMatrixCSC`, not only
  `SparseMatrixCSC` ([#202]).
* **Changed:** UMFPACK and CHOLMOD are safer to use from several threads: they no longer
  keep their workspaces and settings in globals, and multithreaded `ldiv!` with an
  `UmfpackLU` is safe ([#98], [#179], [#206]).
* **Changed:** Serializing an `UmfpackLU` no longer serializes pointers or the workspace
  ([#224]).
* **Changed:** Solving `Hermitian` or `Symmetric` sparse systems with a sparse right-hand
  side throws an `ArgumentError` suggesting a dense right-hand side, and the polyalgorithm
  for sparse self-adjoint `\` no longer tries `ldlt!` ([#322], [#325]).
* **Faster:** `lu!` on an `UmfpackLU` reuses its working buffers, and complex UMFPACK
  `ldiv!` allocates less ([#156], [#179]).

### Display

* **Changed:** Sparse matrices shown as a Braille spy plot have a border, and printing of
  large sparse vectors and of adjoints and transposes of sparse matrices is fixed ([#79],
  [#102], [#211]).

## Julia v1.8

Released with Julia v1.8.0 on 2022-08-17. SuiteSparse 5.10.1.

* **New:** `sparse_hcat`, `sparse_vcat` and `sparse_hvcat` return a `SparseMatrixCSC`
  whatever the types of their arguments. They provide the concatenation behavior in which a
  special "sparse" matrix argument, such as a `Diagonal`, made the result sparse, which is
  no longer possible now that LinearAlgebra is independent of SparseArrays
  ([JuliaLang/julia#43127]).
* **Changed:** LinearAlgebra is independent of SparseArrays, so its methods no longer return
  sparse arrays for Base or LinearAlgebra arguments: concatenations involving `Diagonal`,
  `Bidiagonal`, `Tridiagonal` and `SymTridiagonal` return dense matrices, and so does
  three-argument `similar` of these types ([JuliaLang/julia#43127]).
* **Changed:** `sparse` of a `Diagonal` stores only its nonzero entries
  ([JuliaLang/julia#42577]).
* **Changed:** `kron` of the adjoint or transpose of a sparse matrix returns a sparse matrix
  ([JuliaLang/julia#42181]).
* **Changed:** The adjoint and transpose of a large sparse matrix display as a Braille spy
  plot ([JuliaLang/julia#36177]), and two-argument `show` of a small sparse matrix prints it
  like any other `AbstractArray` ([JuliaLang/julia#41150]).
* **Faster:** `findmax` and `findmin` of a sparse vector visit only the stored entries
  ([JuliaLang/julia#42825]).

## Julia v1.7

Released with Julia v1.7.0 on 2021-11-30. SuiteSparse 5.10.1.

* **New:** `sizehint!(::SparseMatrixCSC, ::Integer)` ([JuliaLang/julia#40523]).
* **New:** `spzeros` accepts its dimensions as a tuple ([JuliaLang/julia#39886],
  [JuliaLang/julia#40515]).
* **New:** `logabsdet` of an `UmfpackLU` ([JuliaLang/julia#40716]).
* **New:** Conversion of a sparse matrix to `Diagonal`, `Tridiagonal`, `SymTridiagonal`,
  `LowerTriangular` and `UpperTriangular` ([JuliaLang/julia#40988]).
* **New:** `sum`, `minimum` and `maximum` of a sparse vector use the sparse method for any
  function `f` ([JuliaLang/julia#29884]).
* **Changed:** `cholesky` preserves the user-specified permutation
  ([JuliaLang/julia#40560]).
* **Changed:** `issparse` applies consistently to all wrapper arrays, including nested ones,
  by checking `issparse` on the wrapped parent array ([JuliaLang/julia#34308]).
* **Changed:** `SparseMatrixCSC` keeps `length(nonzeros(A)) == nnz(A)` for the arrays
  returned by exported functions ([JuliaLang/julia#40523]).
* **Changed:** The length of a `SparseVector` has its index type ([JuliaLang/julia#39645]).
* **Changed:** `sparse(x')` of a sparse vector `x` keeps the index type of `x`
  ([JuliaLang/julia#40836]).
* **Faster:** Products of dense and sparse matrices ([JuliaLang/julia#38876]).
* **Faster:** `dot` of a dense vector, a sparse matrix and a dense vector
  ([JuliaLang/julia#39889]).
* **Faster:** Products of sparse arrays with `BitMatrix` and `BitVector` use sparse
  multiplication ([JuliaLang/julia#39557]).

## Julia v1.6

Released with Julia v1.6.0 on 2021-03-24. SuiteSparse 5.4.0.

* **New:** `spdiagm([m, n,] v::AbstractVector)` methods, equivalent to `spdiagm([m, n,] 0 =>
  v)` like their dense `diagm` counterparts ([JuliaLang/julia#37684]).
* **New:** `kron!`, an in-place Kronecker product, with sparse methods
  ([JuliaLang/julia#31069]).
* **New:** CHOLMOD solves with the adjoint or transpose of a sparse matrix
  ([JuliaLang/julia#38397]).
* **New:** Products of sparse triangular matrices with sparse matrices and vectors, which
  return sparse results ([JuliaLang/julia#35659]).
* **Changed:** Large sparse matrices display as a Unicode "spy" plot of their nonzero
  pattern, and small ones in a `Matrix`-like 2d layout of their contents
  ([JuliaLang/julia#33821]).
* **Faster:** `extrema` of a sparse array visits only the stored entries
  ([JuliaLang/julia#37429]).
* **Faster:** `any` and `all` with a predicate over the columns of a sparse matrix
  short-circuit ([JuliaLang/julia#37848]).
* **Faster:** Sparse matrix constructors that convert the element or index type
  ([JuliaLang/julia#38315]).
* **Faster:** `blockdiag` of sparse matrices with the same element and index types
  ([JuliaLang/julia#36013]).

## Julia v1.5

Released with Julia v1.5.0 on 2020-08-01. SuiteSparse 5.4.0.

* **New:** `lu!` accepts an `UmfpackLU` to reuse its symbolic factorization
  ([JuliaLang/julia#33738]).
* **New:** `qr` of a sparse matrix accepts a user-selected fill-reducing ordering
  ([JuliaLang/julia#35599]).
* **Changed:** The sum of two `Symmetric` or `Hermitian` sparse matrices is sparse
  ([JuliaLang/julia#35325]).
* **Changed:** The CHOLMOD wrapper is thread safe ([JuliaLang/julia#34546]).
* **Changed:** `issparse` is `true` for the `Adjoint` and `Transpose` of a sparse array
  ([JuliaLang/julia#34266]).
* **Changed:** `SparseMatrixCSC` and `SparseVector` constructors from sparse arrays return a
  copy that does not share storage with their argument ([JuliaLang/julia#34647]).
* **Faster:** Right division of a sparse matrix by a `Diagonal` ([JuliaLang/julia#35533]).
* **Faster:** `copyto!` from a sparse array into a dense one ([JuliaLang/julia#33589]).
* **Removed:** The `trim` keyword argument of `fkeep!`, `tril!`, `triu!`, `droptol!`,
  `dropzeros!` and `dropzeros` is removed, and these functions always trim; calling them
  with `trim=false` could produce invalid sparse arrays ([JuliaLang/julia#32972]).

## Julia v1.4

Released with Julia v1.4.0 on 2020-03-21. SuiteSparse 5.4.0.

* **New:** Products involving sparse arrays accept more general element types, such as those
  of StaticArrays ([JuliaLang/julia#33205]).
* **New:** `AbstractSparseMatrixCSC`, an abstract type for compressed sparse column matrices
  that the SparseArrays functions accept ([JuliaLang/julia#33039]).
* **New:** Indexing a `SparseMatrixCSC` with an array of `CartesianIndex`
  ([JuliaLang/julia#33225]).
* **New:** Sparse methods for the three-argument `dot(x, A, y)` ([JuliaLang/julia#32739]).
* **New:** SPQR promotes the element type of its input to one it supports
  ([JuliaLang/julia#33601]).
* **Changed:** `zero(x::AbstractSparseArray)` stores no zeros, and takes constant time
  ([JuliaLang/julia#34209]).
* **Changed:** `sparse(I, J, V, m, n)` throws an error when the index type cannot hold the
  indices, instead of widening it ([JuliaLang/julia#33083]).
* **Changed:** `UmfpackLU` prints a summary in the REPL ([JuliaLang/julia#33705]).

## Julia v1.3

Released with Julia v1.3.0 on 2019-11-26. SuiteSparse 5.4.0.

* **New:** `spdiagm(m, n, pairs...)` builds a non-square matrix ([JuliaLang/julia#31654]).
* **New:** `SparseMatrixCSC` constructors for `Diagonal`, `Bidiagonal`, `Tridiagonal` and
  `SymTridiagonal` matrices ([JuliaLang/julia#32466], [JuliaLang/julia#32492]).
* **New:** Sparse matrices and vectors can hold non-numeric values
  ([JuliaLang/julia#30580]).
* **New:** Sparse methods for the five-argument `mul!(C, A, B, α, β)`
  ([JuliaLang/julia#29634]).
* **Changed:** `kron` of a sparse matrix or vector and a `Diagonal` returns a sparse matrix
  ([JuliaLang/julia#32793]).
* **Changed:** `SparseMatrixCSC(m, n, colptr, rowval, nzval)` checks its arguments: `colptr`
  must be properly populated, and the lengths of `colptr`, `rowval` and `nzval` must be
  compatible with `m`, `n` and `eltype(colptr)` ([JuliaLang/julia#31724]).
* **Changed:** `sparse(I, J, V, m, n)` checks that `I`, `J` and `V` have equal lengths,
  compatible with `eltype(I)`, `m` and `n` ([JuliaLang/julia#31724]).
* **Faster:** Products of sparse and strided dense matrices ([JuliaLang/julia#32195]).

## Julia v1.2

Released with Julia v1.2.0 on 2019-08-19. SuiteSparse 5.4.0.

* **New:** `rank` of a sparse matrix, computed with a sparse QR factorization
  ([JuliaLang/julia#30415]).
* **New:** `copy!` and `copyto!` into a `SparseVector` from any array
  ([JuliaLang/julia#30531], [JuliaLang/julia#30776]).
* **Changed:** Outer products of sparse vectors and column views, in the forms `kron(u,
  v')`, `u * v'` and `u .* v'`, return a sparse matrix ([JuliaLang/julia#24980]).
* **Changed:** The random stream of `sprand` and `sprandn` has changed
  ([JuliaLang/julia#30494]).
* **Changed:** `oneunit` of a sparse matrix returns a sparse matrix
  ([JuliaLang/julia#30228]).
* **Changed:** A sparse Cholesky solve with a vector right-hand side returns a `Vector`
  ([JuliaLang/julia#30416]).
* **Changed:** Sparse matrices print with less whitespace ([JuliaLang/julia#31461]).
* **Faster:** Sparse matrix-matrix multiplication ([JuliaLang/julia#30372]).
* **Faster:** `sprand`, 2 to 5 times faster ([JuliaLang/julia#30494]).
* **Faster:** Converting wrapped sparse matrices, such as transposes and triangular views,
  to `SparseMatrixCSC` ([JuliaLang/julia#30552]).
* **Faster:** `circshift!` of a sparse matrix ([JuliaLang/julia#30317]).

## Julia v1.1

Released with Julia v1.1.0 on 2019-01-21. SuiteSparse 4.4.5.

* **New:** `sprandn` accepts the element type of its result ([JuliaLang/julia#30090]).
* **New:** `lmul!` and `rmul!` of a sparse matrix with a `Diagonal`
  ([JuliaLang/julia#29296]).
* **Faster:** Products and divisions involving sparse triangular matrices use sparse methods
  instead of a generic fallback, 300 to 1000 times faster ([JuliaLang/julia#28507]).
* **Faster:** Indexing a sparse matrix with a vector, and sorting the row indices of its
  columns ([JuliaLang/julia#29682], [JuliaLang/julia#29696]).

## Julia v1.0

Released with Julia v1.0.0 on 2018-08-08. SuiteSparse 4.4.5.

* **Changed:** SparseArrays is a standard library, moved out of Base in Julia v0.7; load it
  with `using SparseArrays`. Julia v1.0 is v0.7 without its deprecations, so the changes
  from v0.6 are in the [Julia v0.7 release notes].
* **Faster:** `\` and `ldiv!` with the transpose or adjoint of a triangular sparse matrix
  use a specialized solver ([JuliaLang/julia#28242]).
* **Removed:** The SparseArrays deprecations of Julia v0.7 are removed
  ([JuliaLang/julia#28469]).

[Julia v0.7 release notes]: https://github.com/JuliaLang/julia/blob/master/HISTORY.md#julia-v070-release-notes
[#79]: https://github.com/JuliaSparse/SparseArrays.jl/pull/79
[#92]: https://github.com/JuliaSparse/SparseArrays.jl/pull/92
[#93]: https://github.com/JuliaSparse/SparseArrays.jl/pull/93
[#95]: https://github.com/JuliaSparse/SparseArrays.jl/pull/95
[#98]: https://github.com/JuliaSparse/SparseArrays.jl/pull/98
[#102]: https://github.com/JuliaSparse/SparseArrays.jl/pull/102
[#144]: https://github.com/JuliaSparse/SparseArrays.jl/pull/144
[#156]: https://github.com/JuliaSparse/SparseArrays.jl/pull/156
[#157]: https://github.com/JuliaSparse/SparseArrays.jl/pull/157
[#158]: https://github.com/JuliaSparse/SparseArrays.jl/pull/158
[#165]: https://github.com/JuliaSparse/SparseArrays.jl/pull/165
[#175]: https://github.com/JuliaSparse/SparseArrays.jl/pull/175
[#176]: https://github.com/JuliaSparse/SparseArrays.jl/pull/176
[#177]: https://github.com/JuliaSparse/SparseArrays.jl/pull/177
[#179]: https://github.com/JuliaSparse/SparseArrays.jl/pull/179
[#195]: https://github.com/JuliaSparse/SparseArrays.jl/pull/195
[#196]: https://github.com/JuliaSparse/SparseArrays.jl/pull/196
[#202]: https://github.com/JuliaSparse/SparseArrays.jl/pull/202
[#203]: https://github.com/JuliaSparse/SparseArrays.jl/pull/203
[#206]: https://github.com/JuliaSparse/SparseArrays.jl/pull/206
[#211]: https://github.com/JuliaSparse/SparseArrays.jl/pull/211
[#222]: https://github.com/JuliaSparse/SparseArrays.jl/pull/222
[#224]: https://github.com/JuliaSparse/SparseArrays.jl/pull/224
[#227]: https://github.com/JuliaSparse/SparseArrays.jl/pull/227
[#228]: https://github.com/JuliaSparse/SparseArrays.jl/pull/228
[#236]: https://github.com/JuliaSparse/SparseArrays.jl/pull/236
[#240]: https://github.com/JuliaSparse/SparseArrays.jl/pull/240
[#244]: https://github.com/JuliaSparse/SparseArrays.jl/pull/244
[#248]: https://github.com/JuliaSparse/SparseArrays.jl/pull/248
[#259]: https://github.com/JuliaSparse/SparseArrays.jl/pull/259
[#260]: https://github.com/JuliaSparse/SparseArrays.jl/pull/260
[#263]: https://github.com/JuliaSparse/SparseArrays.jl/pull/263
[#269]: https://github.com/JuliaSparse/SparseArrays.jl/pull/269
[#271]: https://github.com/JuliaSparse/SparseArrays.jl/pull/271
[#272]: https://github.com/JuliaSparse/SparseArrays.jl/pull/272
[#273]: https://github.com/JuliaSparse/SparseArrays.jl/pull/273
[#276]: https://github.com/JuliaSparse/SparseArrays.jl/pull/276
[#284]: https://github.com/JuliaSparse/SparseArrays.jl/pull/284
[#296]: https://github.com/JuliaSparse/SparseArrays.jl/pull/296
[#303]: https://github.com/JuliaSparse/SparseArrays.jl/pull/303
[#310]: https://github.com/JuliaSparse/SparseArrays.jl/pull/310
[#312]: https://github.com/JuliaSparse/SparseArrays.jl/pull/312
[#314]: https://github.com/JuliaSparse/SparseArrays.jl/pull/314
[#315]: https://github.com/JuliaSparse/SparseArrays.jl/pull/315
[#317]: https://github.com/JuliaSparse/SparseArrays.jl/pull/317
[#320]: https://github.com/JuliaSparse/SparseArrays.jl/pull/320
[#322]: https://github.com/JuliaSparse/SparseArrays.jl/pull/322
[#325]: https://github.com/JuliaSparse/SparseArrays.jl/pull/325
[#327]: https://github.com/JuliaSparse/SparseArrays.jl/pull/327
[#328]: https://github.com/JuliaSparse/SparseArrays.jl/pull/328
[#331]: https://github.com/JuliaSparse/SparseArrays.jl/pull/331
[#340]: https://github.com/JuliaSparse/SparseArrays.jl/pull/340
[#346]: https://github.com/JuliaSparse/SparseArrays.jl/pull/346
[#347]: https://github.com/JuliaSparse/SparseArrays.jl/pull/347
[#351]: https://github.com/JuliaSparse/SparseArrays.jl/pull/351
[#366]: https://github.com/JuliaSparse/SparseArrays.jl/pull/366
[#380]: https://github.com/JuliaSparse/SparseArrays.jl/pull/380
[#384]: https://github.com/JuliaSparse/SparseArrays.jl/pull/384
[#397]: https://github.com/JuliaSparse/SparseArrays.jl/pull/397
[#398]: https://github.com/JuliaSparse/SparseArrays.jl/pull/398
[#407]: https://github.com/JuliaSparse/SparseArrays.jl/pull/407
[#416]: https://github.com/JuliaSparse/SparseArrays.jl/pull/416
[#422]: https://github.com/JuliaSparse/SparseArrays.jl/pull/422
[#423]: https://github.com/JuliaSparse/SparseArrays.jl/pull/423
[#427]: https://github.com/JuliaSparse/SparseArrays.jl/pull/427
[#432]: https://github.com/JuliaSparse/SparseArrays.jl/pull/432
[#437]: https://github.com/JuliaSparse/SparseArrays.jl/pull/437
[#450]: https://github.com/JuliaSparse/SparseArrays.jl/pull/450
[#456]: https://github.com/JuliaSparse/SparseArrays.jl/pull/456
[#458]: https://github.com/JuliaSparse/SparseArrays.jl/pull/458
[#474]: https://github.com/JuliaSparse/SparseArrays.jl/pull/474
[#476]: https://github.com/JuliaSparse/SparseArrays.jl/pull/476
[#487]: https://github.com/JuliaSparse/SparseArrays.jl/pull/487
[#511]: https://github.com/JuliaSparse/SparseArrays.jl/pull/511
[#523]: https://github.com/JuliaSparse/SparseArrays.jl/pull/523
[#532]: https://github.com/JuliaSparse/SparseArrays.jl/pull/532
[#533]: https://github.com/JuliaSparse/SparseArrays.jl/pull/533
[#547]: https://github.com/JuliaSparse/SparseArrays.jl/pull/547
[#550]: https://github.com/JuliaSparse/SparseArrays.jl/pull/550
[#557]: https://github.com/JuliaSparse/SparseArrays.jl/pull/557
[#562]: https://github.com/JuliaSparse/SparseArrays.jl/pull/562
[#564]: https://github.com/JuliaSparse/SparseArrays.jl/pull/564
[#565]: https://github.com/JuliaSparse/SparseArrays.jl/pull/565
[#567]: https://github.com/JuliaSparse/SparseArrays.jl/pull/567
[#586]: https://github.com/JuliaSparse/SparseArrays.jl/pull/586
[#590]: https://github.com/JuliaSparse/SparseArrays.jl/pull/590
[#597]: https://github.com/JuliaSparse/SparseArrays.jl/pull/597
[#600]: https://github.com/JuliaSparse/SparseArrays.jl/pull/600
[#603]: https://github.com/JuliaSparse/SparseArrays.jl/pull/603
[#616]: https://github.com/JuliaSparse/SparseArrays.jl/pull/616
[#626]: https://github.com/JuliaSparse/SparseArrays.jl/pull/626
[#640]: https://github.com/JuliaSparse/SparseArrays.jl/pull/640
[#643]: https://github.com/JuliaSparse/SparseArrays.jl/pull/643
[#645]: https://github.com/JuliaSparse/SparseArrays.jl/pull/645
[#649]: https://github.com/JuliaSparse/SparseArrays.jl/pull/649
[#650]: https://github.com/JuliaSparse/SparseArrays.jl/pull/650
[#652]: https://github.com/JuliaSparse/SparseArrays.jl/pull/652
[#666]: https://github.com/JuliaSparse/SparseArrays.jl/pull/666
[#668]: https://github.com/JuliaSparse/SparseArrays.jl/pull/668
[#672]: https://github.com/JuliaSparse/SparseArrays.jl/pull/672
[#677]: https://github.com/JuliaSparse/SparseArrays.jl/pull/677
[#679]: https://github.com/JuliaSparse/SparseArrays.jl/pull/679
[#684]: https://github.com/JuliaSparse/SparseArrays.jl/pull/684
[#692]: https://github.com/JuliaSparse/SparseArrays.jl/pull/692
[#693]: https://github.com/JuliaSparse/SparseArrays.jl/pull/693
[#694]: https://github.com/JuliaSparse/SparseArrays.jl/pull/694
[#697]: https://github.com/JuliaSparse/SparseArrays.jl/pull/697
[#708]: https://github.com/JuliaSparse/SparseArrays.jl/pull/708
[#710]: https://github.com/JuliaSparse/SparseArrays.jl/pull/710
[#722]: https://github.com/JuliaSparse/SparseArrays.jl/pull/722
[#727]: https://github.com/JuliaSparse/SparseArrays.jl/pull/727
[#732]: https://github.com/JuliaSparse/SparseArrays.jl/pull/732
[#734]: https://github.com/JuliaSparse/SparseArrays.jl/pull/734
[#736]: https://github.com/JuliaSparse/SparseArrays.jl/pull/736
[#751]: https://github.com/JuliaSparse/SparseArrays.jl/pull/751
[#753]: https://github.com/JuliaSparse/SparseArrays.jl/pull/753
[#754]: https://github.com/JuliaSparse/SparseArrays.jl/pull/754
[#765]: https://github.com/JuliaSparse/SparseArrays.jl/pull/765
[#766]: https://github.com/JuliaSparse/SparseArrays.jl/pull/766
[#767]: https://github.com/JuliaSparse/SparseArrays.jl/pull/767
[#768]: https://github.com/JuliaSparse/SparseArrays.jl/pull/768
[#769]: https://github.com/JuliaSparse/SparseArrays.jl/pull/769
[#770]: https://github.com/JuliaSparse/SparseArrays.jl/pull/770
[#772]: https://github.com/JuliaSparse/SparseArrays.jl/pull/772
[#776]: https://github.com/JuliaSparse/SparseArrays.jl/pull/776
[#784]: https://github.com/JuliaSparse/SparseArrays.jl/pull/784
[#786]: https://github.com/JuliaSparse/SparseArrays.jl/pull/786
[#798]: https://github.com/JuliaSparse/SparseArrays.jl/pull/798
[#799]: https://github.com/JuliaSparse/SparseArrays.jl/pull/799
[#803]: https://github.com/JuliaSparse/SparseArrays.jl/pull/803
[#804]: https://github.com/JuliaSparse/SparseArrays.jl/pull/804
[#805]: https://github.com/JuliaSparse/SparseArrays.jl/pull/805
[#806]: https://github.com/JuliaSparse/SparseArrays.jl/pull/806
[#807]: https://github.com/JuliaSparse/SparseArrays.jl/pull/807
[#828]: https://github.com/JuliaSparse/SparseArrays.jl/pull/828
[#830]: https://github.com/JuliaSparse/SparseArrays.jl/pull/830
[#831]: https://github.com/JuliaSparse/SparseArrays.jl/pull/831
[#833]: https://github.com/JuliaSparse/SparseArrays.jl/pull/833
[#836]: https://github.com/JuliaSparse/SparseArrays.jl/pull/836
[#855]: https://github.com/JuliaSparse/SparseArrays.jl/pull/855
[#857]: https://github.com/JuliaSparse/SparseArrays.jl/pull/857
[#858]: https://github.com/JuliaSparse/SparseArrays.jl/pull/858
[#859]: https://github.com/JuliaSparse/SparseArrays.jl/pull/859
[#860]: https://github.com/JuliaSparse/SparseArrays.jl/pull/860
[#861]: https://github.com/JuliaSparse/SparseArrays.jl/pull/861
[#864]: https://github.com/JuliaSparse/SparseArrays.jl/pull/864
[#865]: https://github.com/JuliaSparse/SparseArrays.jl/pull/865
[#866]: https://github.com/JuliaSparse/SparseArrays.jl/pull/866
[#870]: https://github.com/JuliaSparse/SparseArrays.jl/pull/870
[#886]: https://github.com/JuliaSparse/SparseArrays.jl/pull/886
[#887]: https://github.com/JuliaSparse/SparseArrays.jl/pull/887
[#888]: https://github.com/JuliaSparse/SparseArrays.jl/pull/888
[#890]: https://github.com/JuliaSparse/SparseArrays.jl/pull/890
[#891]: https://github.com/JuliaSparse/SparseArrays.jl/pull/891
[#892]: https://github.com/JuliaSparse/SparseArrays.jl/pull/892
[#893]: https://github.com/JuliaSparse/SparseArrays.jl/pull/893
[#896]: https://github.com/JuliaSparse/SparseArrays.jl/pull/896
[#897]: https://github.com/JuliaSparse/SparseArrays.jl/pull/897
[#898]: https://github.com/JuliaSparse/SparseArrays.jl/pull/898
[#906]: https://github.com/JuliaSparse/SparseArrays.jl/pull/906
[#914]: https://github.com/JuliaSparse/SparseArrays.jl/pull/914
[#917]: https://github.com/JuliaSparse/SparseArrays.jl/pull/917
[#919]: https://github.com/JuliaSparse/SparseArrays.jl/pull/919
[#920]: https://github.com/JuliaSparse/SparseArrays.jl/pull/920
[#921]: https://github.com/JuliaSparse/SparseArrays.jl/pull/921
[#922]: https://github.com/JuliaSparse/SparseArrays.jl/pull/922
[#923]: https://github.com/JuliaSparse/SparseArrays.jl/pull/923
[#927]: https://github.com/JuliaSparse/SparseArrays.jl/pull/927
[#928]: https://github.com/JuliaSparse/SparseArrays.jl/pull/928
[#932]: https://github.com/JuliaSparse/SparseArrays.jl/pull/932
[#933]: https://github.com/JuliaSparse/SparseArrays.jl/pull/933
[#935]: https://github.com/JuliaSparse/SparseArrays.jl/pull/935
[#944]: https://github.com/JuliaSparse/SparseArrays.jl/pull/944
[#945]: https://github.com/JuliaSparse/SparseArrays.jl/pull/945
[#949]: https://github.com/JuliaSparse/SparseArrays.jl/pull/949
[#950]: https://github.com/JuliaSparse/SparseArrays.jl/pull/950
[#953]: https://github.com/JuliaSparse/SparseArrays.jl/pull/953
[#954]: https://github.com/JuliaSparse/SparseArrays.jl/pull/954
[#955]: https://github.com/JuliaSparse/SparseArrays.jl/pull/955
[#969]: https://github.com/JuliaSparse/SparseArrays.jl/pull/969
[#972]: https://github.com/JuliaSparse/SparseArrays.jl/pull/972
[#973]: https://github.com/JuliaSparse/SparseArrays.jl/pull/973
[#975]: https://github.com/JuliaSparse/SparseArrays.jl/pull/975
[#978]: https://github.com/JuliaSparse/SparseArrays.jl/pull/978
[#982]: https://github.com/JuliaSparse/SparseArrays.jl/pull/982
[#983]: https://github.com/JuliaSparse/SparseArrays.jl/pull/983
[#985]: https://github.com/JuliaSparse/SparseArrays.jl/pull/985
[#988]: https://github.com/JuliaSparse/SparseArrays.jl/pull/988
[#989]: https://github.com/JuliaSparse/SparseArrays.jl/pull/989
[#1002]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1002
[#1004]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1004
[#1006]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1006
[#1008]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1008
[#1011]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1011
[#1013]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1013
[#1019]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1019
[#1024]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1024
[#1025]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1025
[#1028]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1028
[#1030]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1030
[#1031]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1031
[#1032]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1032
[#1036]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1036
[#1041]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1041
[#1045]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1045
[#1050]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1050
[#1055]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1055
[#1056]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1056
[#1057]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1057
[#1058]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1058
[#1059]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1059
[#1060]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1060
[#1062]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1062
[#1063]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1063
[#1065]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1065
[#1066]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1066
[#1067]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1067
[#1075]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1075
[#1077]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1077
[#1078]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1078
[#1081]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1081
[#1084]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1084
[#1086]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1086
[#1101]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1101
[#1102]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1102
[#1103]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1103
[#1105]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1105
[#1109]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1109
[#1110]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1110
[#1113]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1113
[#1116]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1116
[#1118]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1118
[#1119]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1119
[#1124]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1124
[#1126]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1126
[#1127]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1127
[#1128]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1128
[#1131]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1131
[#1137]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1137
[#1139]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1139
[#1142]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1142
[#1143]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1143
[#1145]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1145
[#1157]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1157
[#1160]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1160
[JuliaLang/julia#24980]: https://github.com/JuliaLang/julia/pull/24980
[JuliaLang/julia#28242]: https://github.com/JuliaLang/julia/pull/28242
[JuliaLang/julia#28469]: https://github.com/JuliaLang/julia/pull/28469
[JuliaLang/julia#28507]: https://github.com/JuliaLang/julia/pull/28507
[JuliaLang/julia#29296]: https://github.com/JuliaLang/julia/pull/29296
[JuliaLang/julia#29634]: https://github.com/JuliaLang/julia/pull/29634
[JuliaLang/julia#29682]: https://github.com/JuliaLang/julia/pull/29682
[JuliaLang/julia#29696]: https://github.com/JuliaLang/julia/pull/29696
[JuliaLang/julia#29884]: https://github.com/JuliaLang/julia/pull/29884
[JuliaLang/julia#30090]: https://github.com/JuliaLang/julia/pull/30090
[JuliaLang/julia#30228]: https://github.com/JuliaLang/julia/pull/30228
[JuliaLang/julia#30317]: https://github.com/JuliaLang/julia/pull/30317
[JuliaLang/julia#30372]: https://github.com/JuliaLang/julia/pull/30372
[JuliaLang/julia#30415]: https://github.com/JuliaLang/julia/pull/30415
[JuliaLang/julia#30416]: https://github.com/JuliaLang/julia/pull/30416
[JuliaLang/julia#30494]: https://github.com/JuliaLang/julia/pull/30494
[JuliaLang/julia#30531]: https://github.com/JuliaLang/julia/pull/30531
[JuliaLang/julia#30552]: https://github.com/JuliaLang/julia/pull/30552
[JuliaLang/julia#30580]: https://github.com/JuliaLang/julia/pull/30580
[JuliaLang/julia#30776]: https://github.com/JuliaLang/julia/pull/30776
[JuliaLang/julia#31069]: https://github.com/JuliaLang/julia/pull/31069
[JuliaLang/julia#31461]: https://github.com/JuliaLang/julia/pull/31461
[JuliaLang/julia#31654]: https://github.com/JuliaLang/julia/pull/31654
[JuliaLang/julia#31724]: https://github.com/JuliaLang/julia/pull/31724
[JuliaLang/julia#32195]: https://github.com/JuliaLang/julia/pull/32195
[JuliaLang/julia#32466]: https://github.com/JuliaLang/julia/pull/32466
[JuliaLang/julia#32492]: https://github.com/JuliaLang/julia/pull/32492
[JuliaLang/julia#32739]: https://github.com/JuliaLang/julia/pull/32739
[JuliaLang/julia#32793]: https://github.com/JuliaLang/julia/pull/32793
[JuliaLang/julia#32972]: https://github.com/JuliaLang/julia/pull/32972
[JuliaLang/julia#33039]: https://github.com/JuliaLang/julia/pull/33039
[JuliaLang/julia#33083]: https://github.com/JuliaLang/julia/pull/33083
[JuliaLang/julia#33205]: https://github.com/JuliaLang/julia/pull/33205
[JuliaLang/julia#33225]: https://github.com/JuliaLang/julia/pull/33225
[JuliaLang/julia#33589]: https://github.com/JuliaLang/julia/pull/33589
[JuliaLang/julia#33601]: https://github.com/JuliaLang/julia/pull/33601
[JuliaLang/julia#33705]: https://github.com/JuliaLang/julia/pull/33705
[JuliaLang/julia#33738]: https://github.com/JuliaLang/julia/pull/33738
[JuliaLang/julia#33821]: https://github.com/JuliaLang/julia/pull/33821
[JuliaLang/julia#34209]: https://github.com/JuliaLang/julia/pull/34209
[JuliaLang/julia#34266]: https://github.com/JuliaLang/julia/pull/34266
[JuliaLang/julia#34308]: https://github.com/JuliaLang/julia/pull/34308
[JuliaLang/julia#34546]: https://github.com/JuliaLang/julia/pull/34546
[JuliaLang/julia#34647]: https://github.com/JuliaLang/julia/pull/34647
[JuliaLang/julia#35325]: https://github.com/JuliaLang/julia/pull/35325
[JuliaLang/julia#35533]: https://github.com/JuliaLang/julia/pull/35533
[JuliaLang/julia#35599]: https://github.com/JuliaLang/julia/pull/35599
[JuliaLang/julia#35659]: https://github.com/JuliaLang/julia/pull/35659
[JuliaLang/julia#36013]: https://github.com/JuliaLang/julia/pull/36013
[JuliaLang/julia#36177]: https://github.com/JuliaLang/julia/pull/36177
[JuliaLang/julia#37429]: https://github.com/JuliaLang/julia/pull/37429
[JuliaLang/julia#37684]: https://github.com/JuliaLang/julia/pull/37684
[JuliaLang/julia#37848]: https://github.com/JuliaLang/julia/pull/37848
[JuliaLang/julia#38315]: https://github.com/JuliaLang/julia/pull/38315
[JuliaLang/julia#38397]: https://github.com/JuliaLang/julia/pull/38397
[JuliaLang/julia#38876]: https://github.com/JuliaLang/julia/pull/38876
[JuliaLang/julia#39557]: https://github.com/JuliaLang/julia/pull/39557
[JuliaLang/julia#39645]: https://github.com/JuliaLang/julia/pull/39645
[JuliaLang/julia#39886]: https://github.com/JuliaLang/julia/pull/39886
[JuliaLang/julia#39889]: https://github.com/JuliaLang/julia/pull/39889
[JuliaLang/julia#40515]: https://github.com/JuliaLang/julia/pull/40515
[JuliaLang/julia#40523]: https://github.com/JuliaLang/julia/pull/40523
[JuliaLang/julia#40560]: https://github.com/JuliaLang/julia/pull/40560
[JuliaLang/julia#40716]: https://github.com/JuliaLang/julia/pull/40716
[JuliaLang/julia#40836]: https://github.com/JuliaLang/julia/pull/40836
[JuliaLang/julia#40988]: https://github.com/JuliaLang/julia/pull/40988
[JuliaLang/julia#41150]: https://github.com/JuliaLang/julia/pull/41150
[JuliaLang/julia#42181]: https://github.com/JuliaLang/julia/pull/42181
[JuliaLang/julia#42577]: https://github.com/JuliaLang/julia/pull/42577
[JuliaLang/julia#42825]: https://github.com/JuliaLang/julia/pull/42825
[JuliaLang/julia#43127]: https://github.com/JuliaLang/julia/pull/43127
[JuliaLang/julia#44247]: https://github.com/JuliaLang/julia/pull/44247
[JuliaLang/julia#48979]: https://github.com/JuliaLang/julia/pull/48979
[JuliaLang/julia#49266]: https://github.com/JuliaLang/julia/pull/49266
