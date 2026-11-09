# SparseArrays.jl release history

SparseArrays is a Julia standard library. Each Julia release ships one fixed version of it,
numbered like the release: SparseArrays v1.14 ships with Julia v1.14. Its code lived in the
Julia repository until Julia v1.8, and the entries for those releases link to Julia pull
requests.

Each change is listed under the first Julia v1.x.0 release that shipped it. Fixes
backported to patch releases are not listed again, and a bug fix is listed only when it
changes a result that code could have relied on.

## Unreleased

Changes on `main` that no release branch has yet.

### Behavior changes

* `sprand` and `sprandn` accept a matrix of 2^63 or more entries again, drawing it column
  by column, instead of throwing an `OverflowError` ([#1143]).

### Performance

* The package image holds native code for `show`, `\`, products, broadcast, concatenation
  and `sprand` of sparse `Float64` and `ComplexF64` arrays, so their first call compiles
  less ([#1103]).
* `transpose(A) * x`, `A' * x` and `mul!(y, A', x)` with a sparse matrix `A` and a sparse
  vector `x` no longer merge `x` with every column of `A`, which is much faster when `A`
  has many columns ([#1139]).

## Julia v1.14

Ships with Julia v1.14.0, not yet released. SuiteSparse 7.14.0.

### New features

#### Construction and conversion

* `sparse` accepts a vector of `CartesianIndex` positions in place of row and column index
  vectors ([#600]).
* `sparse!` and `spzeros!` are public ([#668]).
* `promote_type` and `promote` of `SparseMatrixCSC` and `SparseVector` types with different
  element or index types return a concrete sparse type ([#807]).
* `SparseMatrixCSC{Tv}(M::AbstractMatrix)` converts the element type to `Tv` ([#927]).

#### Indexing and assignment

* Sparse `getindex` lowers its indices with `to_indices`, as `setindex!` does, so it
  accepts custom index types such as `InvertedIndices.Not` ([#857]).
* `getindex` of a sparse matrix accepts `StepRangeLen` row ranges, a vector of
  `CartesianIndex{1}` as a row or column index, and a `Bool` unit range as a mask
  ([#1057], [#1058], [#1062]).

#### Views, wrappers and storage

* `AbstractSparseMatrixCSC`, `getcolptr`, `getrowval`, `getnzval` and `indtype` are exported
  and documented; `getrowval` and `getnzval` also accept sparse vectors and their views
  ([#767], [#805]).
* `nonzeros`, `rowvals` and `nzrange` of a view of some columns of a sparse matrix describe
  only the entries inside the view, and `nnz` of a triangular, `Symmetric` or `Hermitian`
  wrapper of a sparse matrix counts the entries of its triangle ([#893]).
* `getcolptr`, `getrowval` and `getnzval` are the storage interface of
  `AbstractSparseMatrixCSC`: a subtype that defines them works with every kernel of the
  package ([#1008]).

#### Arithmetic and broadcasting

* `map` and `broadcast` over sparse arrays accept functions that return values of a type
  without `zero`, such as `String`, `Tuple`, `Pair` or `nothing` ([#1055]).

#### Reductions, sorting and comparison

* `findnz` and `isequal` accept adjoints and transposes of sparse matrices and vectors
  ([#765], [#768]).
* `sort` and `sort!` of a `SparseMatrixCSC` along a dimension ([#784], [#786]).
* Reductions along a dimension (`sum`, `prod`, `maximum`, `minimum`, `count`, `any`, `all`
  and `mapreduce`) take a `sparse` keyword that returns a sparse result instead of a dense
  one ([#798]).

#### Concatenation

* `sparse_hcat`, `sparse_vcat` and `sparse_hvcat` accept `UniformScaling` arguments
  ([#920]).

#### Products and linear algebra

* `opnorm(A)` (the 2-norm) of a sparse matrix returns an estimate of the largest singular
  value instead of throwing ([#859]).
* `kron!` accepts a dense vector operand next to a sparse one ([#988]).

#### Factorizations and solvers

* `rcond` of a `CHOLMOD.Factor` or an `UmfpackLU` returns the solver's reciprocal condition
  number estimate ([#776]).
* `ldiv!` with a CHOLMOD, UMFPACK or SPQR factorization takes an optional `workspace`
  keyword, so that repeated solves can run without allocating ([#727], [#914]).
* Solving with the adjoint of a sparse QR factorization, `qr(A)' \ b` ([#804]).
* `LibSuiteSparse.set_libdir!` and the `JULIA_SUITESPARSE_LIBDIR` environment variable load
  the SuiteSparse libraries from another directory ([#806]).
* Products of the `Q` of a sparse QR factorization with sparse operands, including its
  adjoint and transpose, apply the Householder reflectors instead of forming `Q` or throwing
  ([#830]).
* `lq` of a sparse matrix, computed as the adjoint of the sparse QR of the adjoint ([#833]).
* `ldlt`, `lu` and `inv` accept `Adjoint`, `Transpose`, `Symmetric`, `Hermitian` and view
  wrappers of sparse matrices; `lu` of a wrapped sparse matrix returns a sparse
  factorization ([#855]).
* One `UmfpackLU`, `QRSparse` or `CHOLMOD.Factor` can be used to solve from several tasks
  in parallel, and the reads of an `UmfpackLU` (`det`, `logabsdet`, its factors) take no
  lock. A factorization must not be refactorized while another task uses it; such a task
  works on its own `copy(F)` ([#906], [#914], [#1041], [#1042], [#1043], [#1045]).

#### Compilation

* Dense and sparse `cat` and `hvcat`, sparse block literals, the SuiteSparse library
  paths and the main sparse operations can be compiled with `juliac --trim=safe`
  ([#972], [#973], [#974], [#1028]).

### Behavior changes

#### Construction and conversion

* `one(A)` of a sparse matrix whose element type has a multiplicative identity other than
  `1` uses that identity on the diagonal ([#732]).
* `sparsevec` of a sparse matrix returns a `SparseVector` instead of a lazy reshape
  ([#860]).
* `repeat` of a sparse matrix and sparse column slices such as `A[:, j]` keep the matrix's
  index type ([#919], [#945]).
* `SparseMatrixCSC` of a view of a sparse vector or column returns an `n x 1` matrix
  ([#1113]).
* `sprand` with a value function honours the eltype `T` and the documented forms of
  `rfn` ([#1109]).

#### Indexing and assignment

* Scalar and nonscalar `setindex!` store `-0.0` and `missing` but not a heap-allocated zero
  such as `big(0)`, consistently ([#891], [#896], [#898]).
* A `Bool` is no longer accepted as a scalar index in sparse `getindex`, `setindex!` and
  `dropstored!` ([#1067]), and scalar `setindex!` at an index that does not fit the index
  type throws a `BoundsError` ([#1145]).
* `getindex` of a sparse array with an index array of zero or three or more dimensions
  returns a dense array ([#1059]).
* Reductions and linear indexing handle sparse matrices of 2^63 or more entries; linear
  indexing of one that `Int` cannot index throws an `OverflowError`, and so does `sprand`
  of such a size ([#1128], [#1137]).
* In-place sparse kernels (`copyto!`, `setindex!`, `map!`, `broadcast!`, `mul!`, `ldiv!`,
  `circshift!`, `sparse!`, `fkeep!`, `dropstored!`, `lu!` and others) check sizes,
  aliasing, value conversion and index-type overflow before writing, and throw with the
  destination unchanged ([#1019], [#1052], [#1060], [#1071], [#1073], [#1080], [#1082],
  [#1110], [#1111], [#1114], [#1115], [#1120], [#1121], [#1123], [#1125]).
* Fixed-pattern arrays (experimental, unexported `SparseArrays.fixed`) are a write
  destination only: results computed from them are plain sparse arrays, and functions
  whose purpose is to change the pattern throw ([#1033]).

#### Views, wrappers and storage

* `nonzeroinds` is the function from LinearAlgebra ([#678]).

#### Arithmetic and broadcasting

* Broadcasting a sparse array against a view of a dense array returns a sparse result, as
  against the dense array itself ([#772]).
* Broadcasting a sparse array against a dense array does not store `0 / x`-style zeros
  where the sparse argument has no entry ([#886]).
* `+` and `-` of a sparse vector and a dense vector return a dense `Vector`, as for
  matrices ([#887]).
* `map` and broadcast over a single sparse array keep its stored entries, including stored
  zeros and entries where `f` returns zero, as `float(A)`, `-A` and `2A` do ([#892]).
* Broadcasting over sparse views of whole columns, or of a sparse vector, returns a sparse
  result ([#917]).
* A result that would store every entry is dense: sparse `+` and `-` with any strided
  dense matrix, including its views, adjoints and transposes, broadcast `+` and `-` of a
  sparse array with a dense array or a scalar, and a broadcast over sparse arrays of a
  function that is not zero at zero (such as `cos.(A)` or `A ./ A`) return a dense array
  ([#949], [#1004], [#1005], [#1086]).
* Scaling a sparse array by `NaN` or `Inf` with `rmul!`, `lmul!`, `mul!` or `x * NaN`,
  the outer product `x .* y'`, and `prod` propagate the non-finite value to the unstored
  entries, as dense arrays do ([#1065], [#1077], [#1101]).
* `kron` with a view of a sparse matrix, and `diff` of a sparse view, transpose or
  adjoint, return sparse results ([#953], [#1078], [#1118]).

#### Reductions, sorting and comparison

* `findmin` and `findmax` of a `SparseVector` return an `Int` index, and `findall` returns a
  `Vector{Int}` for every predicate ([#922], [#923]).
* `maximum`, `minimum` and `extrema` along a dimension of an empty sparse array return a
  dense `Array` ([#928]).
* `sum`, `prod` and `mapreduce` with `+` or `*` over sparse arrays of small integers widen
  the unstored entries as for dense arrays, so their result types match dense input
  ([#932]).
* Reductions along a dimension of the adjoint or transpose of a sparse vector return an
  `Array` ([#935]).
* `mapreduce` over sparse arrays folds the entries in element order unless `op` is known
  to commute, so a non-commutative `op` gives the dense result ([#1006], [#1063]).
* `==` and `isequal` of two sparse arrays compare the implicit zeros of their eltypes
  ([#985]).
* `any` and `all` of a sparse array throw a `TypeError` for a non-Boolean value, as dense
  does ([#1056]).

#### Concatenation

* Loading SparseArrays no longer makes `vcat`, `hcat` and `hvcat` methods of other packages'
  array types ambiguous, and a leading number in a sparse concatenation fills its block as
  in dense concatenation ([#890]).
* `hcat`, `vcat` and `cat` of sparse arrays with dense arrays, numbers, `I` or a
  `Diagonal` keep the stored zeros of the sparse arguments ([#1102]). The index type of a
  mixed sparse and dense concatenation is taken from its sparse arguments only ([#1127]).
  A concatenation with a leading sparse array and an array of non-`Number` eltype returns
  a dense array instead of throwing ([#975]).
* Loading SparseArrays no longer changes the result or the speed of dense `hcat`, `vcat`,
  `hvcat` and `cat` ([#1157], [#1160]).

#### Products and linear algebra

* Products of a sparse matrix with a structured or dense matrix take their result type from
  the non-sparse factor: sparse times dense returns dense, sparse times a banded structured
  matrix stays sparse ([#693]).
* Solves of structured matrices with sparse matrices return dense arrays, consistently for
  left and right division ([#708]).
* Sparse `D \ A` and `A / D` with a singular `Diagonal` `D` throw `SingularException`
  ([#944]).
* Products and solves of a wrapped sparse or dense operand (adjoints and transposes of
  triangular, symmetric or Hermitian wrappers, and views of rows) return the same
  container whichever side the operand is on: dense with a dense factor, sparse with a
  banded one ([#1075]).
* `cond` of a singular sparse matrix returns `Inf` instead of throwing, in the same float
  type as for a nonsingular one, and `cond` of an empty sparse matrix returns zero
  ([#989], [#1126], [#1131]).

#### Factorizations and solvers

* `A \ b` for a wide sparse `A` returns the minimum-norm solution, as dense `\` does
  ([#831]).
* Sparse `\` and `factorize` treat a Hermitian matrix as Hermitian only for floating-point
  element types; integer matrices go through the general sparse path instead of a dense
  exact factorization ([#921]).
* `\` with a sparse right-hand side, and `sparse(F.L)` of a factorization, return a
  `SparseMatrixCSC` ([#933]).
* Sparse `\`, its adjoint and transpose forms, and `qr(A) \ b` return the eltype of dense
  `\`, whatever the shape of `A` ([#950], [#1124]).
* `factorize` of a sparse indefinite Hermitian matrix whose LDLt fails falls back to `lu`
  instead of throwing `ZeroPivotException` ([#1116]).
* The sparse triangular solve throws `SingularException` for a stored zero on the
  diagonal, as for a missing diagonal entry ([#1066]).
* `copy` of an `UmfpackLU` or a `QRSparse` and `deepcopy` of a `CHOLMOD.Factor` return
  independent factorizations, and `copy` of an adjoint `CHOLMOD.Factor` is defined
  ([#1030], [#1031], [#1032]).

#### Display

* `show` of a sparse matrix uses Base's display pipeline: the Unicode spy plot uses the
  full display width and is chosen from the display size, and stored zeros print centered
  at a fixed width ([#736]).
* Sparse views and `Hermitian`, `Symmetric`, triangular and `Diagonal` wrappers of sparse
  arrays print with the sparse display ([#978]). The `Q` factor of a sparse `qr` or `lq`
  prints its entries like any other `AbstractQ` ([#1025], [#1036]).

### Performance

#### Construction and conversion

* `stack` of sparse vectors builds the sparse result directly ([#836]).

#### Indexing and assignment

* Row indexing of the adjoint or transpose of a sparse matrix ([#734]).
* `copyto!` of a sparse source into a view of a sparse array splices the stored entries in
  one pass ([#799]).
* `A[I, J]` with a sorted vector `I` selecting few columns, or on a hypersparse matrix,
  no longer allocates a cache of length `size(A, 1)` ([#870]).
* Ranged `fill!` on a sparse view merges in place ([#897]).

#### Views, wrappers and storage

* `Matrix` and `Vector` of views, adjoints and transposes of sparse arrays take time
  proportional to the length plus the number of stored entries ([#828]).
* Views of sparse matrices and vectors, and lazy adjoints and transposes, use the O(nnz)
  kernels in reductions, products, `norm`, `diag`, `istriu`, `istril`, `issymmetric`,
  `ishermitian`, `count`, `==`, `reverse`, `circshift`, `permutedims`, `sort`,
  `findnext`, `findprev`, `spdiagm` and conversion, instead of reading every element
  ([#969], [#979], [#1024], [#1119], [#1133], [#1134], [#1135]).

#### Arithmetic and broadcasting

* Sparse broadcast grows its result storage on demand instead of allocating the upper bound
  on the number of stored entries ([#803]).
* Broadcast that scales the rows of a sparse matrix by a vector runs in time proportional
  to the number of stored entries ([#888]).
* `+` and `-` of sparse vectors and their views are one O(nnz) merge ([#954]).
* `reverse`, `circshift` and rotations of sparse arrays run in O(nnz) ([#955]).
* Sparse plus dense and sparse-plus-scalar broadcasts such as `A .+ 1` skip the dense
  copy of the sparse operand ([#982], [#983], [#1105]).
* `map!` and `broadcast!` into a sparse array do not allocate to unalias their inputs
  ([#1013]).

#### Reductions, sorting and comparison

* `isequal`, `findnext`, `findprev` and `hash` of sparse arrays visit only the stored
  entries ([#766], [#769]).
* `mapreduce(f, op, A, B)` over two sparse arrays reduces without building
  `map(f, A, B)` ([#1002]).

#### Products and linear algebra

* Products with `Diagonal` and `dot` with adjoints and transposes of sparse matrices,
  mixed adjoint and transpose operands, and column views ([#770], [#858], [#866]).
* Products of dense matrices with `Symmetric`, `Hermitian` and triangular sparse matrices,
  of `Symmetric` and `Hermitian` sparse matrices with sparse operands, and `mul!` of sparse
  matrices into dense or sparse destinations use the sparse kernels instead of generic
  element-wise fallbacks ([#861], [#864], [#865], [#867], [#868], [#869], [#871], [#873],
  [#875], [#877]).
* Sparse `A*B` sizes its result from a sample of its columns and releases spare
  capacity, which cuts the memory it holds several times over where rows collide, as on
  a stencil ([#1142]).
* Products of a sparse matrix with a `Bidiagonal`, `Tridiagonal` or `SymTridiagonal`
  run in O(nnz) ([#1084]).

#### Factorizations and solvers

* The sparse triangular solve finds the diagonal of a stored triangle without a search
  ([#1081]).

### Deprecated or removed

* SparseArrays requires SuiteSparse 7.0 or later ([#806]).
* The ~560 `cholmod_*`, `CHOLMOD_*` and `umfpack_*` names of `LibSuiteSparse` are no longer
  bindings in `SparseArrays` itself; use them from `SparseArrays.LibSuiteSparse` ([#947]).
* Implementing `rowvals` and `nonzeros` instead of `getrowval` and `getnzval` on a
  subtype of `AbstractSparseMatrixCSC` is deprecated ([#1011]).
* `rowvals`, `nonzeros` and `nzrange` of an `UpperTriangular` or `LowerTriangular` wrapper
  of a sparse matrix throw a `MethodError` instead of returning the parent's storage,
  entries outside the triangle included ([#893], [#1050]).

## Julia v1.13

Released with Julia v1.13.0 on 2026-09-09. SuiteSparse 7.10.1.

### New features

* `ldiv!` solves in place with a sparse `qr` factorization (`SPQR.QRSparse`) ([#679]).
* 3-argument `dot(x, A, y)` with a sparse `Symmetric` or `Hermitian` `A` accepts element
  types other than real and complex numbers, such as quaternions ([#643]).
* `propertynames` of a CHOLMOD factorization lists its components, and the `cholesky` and
  `ldlt` docstrings describe them ([#645]).
* `sparse`, `spdiagm`, `sparsevec`, `mapreduce` and the SPQR factorization can be compiled
  with `juliac --trim` ([#751], [#753], [#754], [#756]).
* `LibSuiteSparse.init_suitesparse` is public, for packages that wrap other SuiteSparse
  libraries and must initialize them the same way ([#677]).

### Behavior changes

* `A \ b` with a sparse `Float32` or `Float16` matrix returns a result of that precision,
  instead of `Float64` ([#684]).
* `zero` of a sparse array keeps its index type ([#710]).
* `findmin` and `findmax` of sparse arrays treat `NaN` as dense arrays do ([#722]).

### Performance

* Sparse-times-dense and dense-times-sparse matrix multiplication is faster ([#666]), and
  so is multiplication of a sparse `Symmetric` or `Hermitian` matrix by an adjoint or
  transposed dense matrix ([#692]).
* `isdiag` of a sparse matrix runs in time proportional to the number of stored entries
  ([#672]).
* 5-argument `mul!` with a `Diagonal` and a sparse matrix, and 3-argument `dot(x, A, y)`
  with sparse vectors and a dense matrix, visit only the stored entries ([#603], [#640]).
* `ldiv!` with a `QRSparse` does not allocate a square copy of `R` ([#697]).
* CHOLMOD and the other SuiteSparse libraries are initialized on first use instead of at
  load time ([#626], [#677]).
* Loading SparseArrays causes fewer invalidations ([#694]).

## Julia v1.12

Released with Julia v1.12.0 on 2025-10-07. SuiteSparse 7.8.3.

### New features

* `ldiv!` solves in place with a CHOLMOD `cholesky` or `ldlt` factorization ([#547]).
* `qr` of a sparse `Float32` or `ComplexF32` matrix factorizes in single precision;
  `Float16` and `ComplexF16` matrices are converted to single precision ([#586], [#597]).
* `CHOLMOD.symbolic` takes a `nested_dissection` keyword to enable nested dissection
  ordering ([#550]).
* `SparseArrays.@allowscalar` toggles scalar indexing and takes effect in the same world
  age ([#583]).
* `SparseMatrixCSC{Tv,Ti}(undef, (m, n))` and `SparseVector{Tv,Ti}(undef, (n,))` accept a
  tuple of dimensions, so `similar(SparseMatrixCSC{Tv,Ti}, dims)` works ([#523]).
* `ldiv!` and `rdiv!` with a `Diagonal` and a sparse matrix accept differing element types
  when the result can be stored in the destination ([#616]).

### Behavior changes

* CHOLMOD does not try nested dissection ordering by default, which could be slow for
  large matrices ([#550]).
* The default `tol` of sparse `qr` uses `eps()` of `Float64` for every element type, as
  SPQR itself does ([#557]).

### Performance

* `repeat` of sparse matrices and vectors builds the result column by column ([#532]).
* `copytrito!` from a sparse source visits only the stored entries ([#533]).
* `Diagonal * S * Diagonal` with a sparse `S`, `Diagonal` times a sparse vector, 3-argument
  `dot(x, D::Diagonal, y)` with a sparse vector, and `ldiv!(D::Diagonal, x)` with a sparse
  vector run in time proportional to the number of stored entries ([#562], [#564], [#565],
  [#613]).
* `istriu(S, k)` and `istril(S, k)` with a band index use the sparse kernel ([#590]).
* 5-argument `mul!` of a sparse matrix and a dense vector or matrix does not allocate
  ([#567]).
* Loading SparseArrays causes fewer invalidations of the system image ([#649], [#650],
  [#652], [#653]).

### Deprecated or removed

* `UMFPACK.UmfpackLU` no longer has methods of `lock`, `trylock` and `unlock`; its internal
  lock is not part of the interface ([#617]).

## Julia v1.11

Released with Julia v1.11.0 on 2024-10-07. SuiteSparse 7.7.0.

### New features

* CHOLMOD factorizations (`cholesky`, `ldlt`) accept `Float32` and `ComplexF32` matrices and
  factorize them in single precision ([#487]).
* `lu`, `qr` and `cholesky` of wrapped sparse matrices, such as `Symmetric` and `Hermitian`
  wrappers, reach the sparse solvers through LinearAlgebra's factorization dispatch
  ([#437]).
* `+`, `-` and other algebraic operators on column views of a `SparseMatrixCSC`
  (`view(S, :, cols)`) return sparse results ([#458]).
* `kron` of a sparse matrix with an adjoint or transposed dense matrix returns a sparse
  matrix ([#474]).
* `UMFPACK.JL_UMFPACK_PIVOT_TOLERANCE` indexes the pivot tolerance in the UMFPACK control
  vector ([#447]).

### Behavior changes

* `show` of a `SparseVector` in compact or one-line contexts, such as inside a container,
  prints a round-trippable `sparsevec(indices, values, n)` ([#427]).
* Views of a contiguous range of a sparse vector behave as sparse vectors: `nnz`,
  `nonzeros`, `nonzeroinds` and the sparse kernels apply to them ([#416]).
* `reverse` of a sparse vector or matrix returns a sparse array with the same number of
  stored entries, instead of storing explicit zeros ([#450]).

### Performance

* Adding or subtracting a sparse matrix and a dense `Array` does not materialize the sparse
  operand as a dense matrix first ([#456]).
* Multiplying by a column view of a sparse matrix with a non-contiguous column selection
  uses the sparse kernels instead of a generic fallback ([#476]).
* `findmin` and `findmax` of sparse matrices, with or without `dims`, do not allocate for
  every comparison ([#511]).
* Concatenation of sparse vectors infers its result type ([#422]).

## Julia v1.10

Released with Julia v1.10.0 on 2023-12-25. SuiteSparse 7.2.1.

### New features

* `spzeros([T,] I, J, m, n)` creates a sparse matrix with structural zeros at the positions
  `(I[k], J[k])`, and `SparseArrays.spzeros!` does so into preallocated buffers ([#284],
  [#310], [#315]).
* `SparseArrays.sparse!(I, J, V)` and `SparseArrays.spzeros!(I, J)` build a sparse matrix
  reusing the memory of the input vectors ([#397]).
* CHOLMOD and SPQR accept `Int32` index types as well as `Int64` ([#380]).
* Solving with the adjoint of a real CHOLMOD `Factor` accepts a complex right-hand side
  ([#327]).
* `sparse`, `SparseMatrixCSC` and `SparseMatrixCSC{Tv,Ti}` construct a sparse matrix from
  an `AbstractQ` ([#203], [#346]).

### Behavior changes

* SparseArrays and the SuiteSparse libraries are no longer in the default system image, so
  the core language no longer contains GPL libraries; they still ship with the standard
  binary distribution ([JuliaLang/julia#44247], [JuliaLang/julia#48979],
  [JuliaLang/julia#49266]).
* `kron` with one sparse and one dense factor returns a sparse matrix ([#331]).
* `float(S)` and `complex(S)` return `S` itself when its eltype is already floating point
  or complex, like for dense arrays ([#340]).
* Solving with a CHOLMOD factorization and a vector right-hand side returns a vector
  ([#320]).
* `adjoint` and `transpose` of an `UmfpackLU` or a CHOLMOD `Factor` return LinearAlgebra's
  `AdjointFactorization` and `TransposeFactorization`, and the `Q` of a sparse QR
  factorization is an `AbstractQ`, following LinearAlgebra ([#203], [#269], [#347]).
* `map` keeps the index type of its sparse arguments ([#351]).
* `sort!` of a sparse vector keeps its zeros implicit instead of storing them ([#303]).
* Sparse matrices nested inside other containers print according to the `IOContext`, for
  example compactly ([#423]).

### Performance

* `collect`, `Vector` and `Matrix` of sparse arrays, and 3-argument `dot` with a sparse
  `Hermitian` or `Symmetric` matrix, are faster ([#312], [#328], [#398]).
* Products of an `AbstractQ` with a sparse array no longer fall back to elementwise generic
  multiplication ([#317]).
* Concatenation of dense arrays and numbers is faster with SparseArrays loaded, and
  concatenation of many arguments infers better ([#384], [#407], [#432]).

## Julia v1.9

Released with Julia v1.9.0 on 2023-05-07. SuiteSparse 5.10.1.

### New features

* The SuiteSparse solver wrappers moved from the SuiteSparse standard library into
  SparseArrays and are available as its submodules `SparseArrays.CHOLMOD`,
  `SparseArrays.UMFPACK` and `SparseArrays.SPQR`; SuiteSparse.jl re-exports them ([#95],
  [#144]).
* The experimental `SparseArrays.allowscalar(::Bool)` disables or enables scalar indexing of
  sparse arrays, to help find accidental scalar indexing, a common source of slow code
  ([#200]).
* `lu(A; q)` and `lu!(F, A; q)` take a user-supplied column permutation `q` for the UMFPACK
  symbolic factorization ([#165]).
* `lu!(F::UmfpackLU)` refactorizes with the existing symbolic factorization, and
  `UmfpackLU` gains a constructor that does not factorize ([#224], [#276]).
* `copy(F::UmfpackLU; copynumeric, copysymbolic)` controls whether the numeric and symbolic
  factorizations are copied ([#259]).
* UMFPACK `lu` and `lu!` accept any `AbstractSparseMatrixCSC`, not only `SparseMatrixCSC`
  ([#202]).
* `rank` of a sparse matrix and of a sparse QR factorization takes a tolerance `tol`
  ([#195], [#228]).
* `SparseMatrixCSC{Tv,Ti}(undef, m, n)` and `SparseVector{Tv,Ti}(undef, n)` create empty
  sparse arrays ([#263]).
* `findnz` and reductions work for any `AbstractSparseVector` ([#175]).
* `reinterpret` works on sparse arrays ([#296]).

### Behavior changes

* UMFPACK and CHOLMOD are safer to use from several threads: they no longer keep their
  workspaces and settings in globals, and multithreaded `ldiv!` with an `UmfpackLU` is safe
  ([#98], [#179], [#206]).
* Serializing an `UmfpackLU` no longer serializes pointers or the workspace ([#224]).
* `length(::SparseVector{Tv,Ti})` returns a `Ti` ([#176]).
* `map` on sparse arrays determines the result eltype from the values `f` returns, and calls
  `f` on zero only when the result needs it ([#177], [#222]).
* `circshift` shifts in the same direction as for dense arrays ([#260], [#272]).
* Assigning `-0.0` (or any value that is `==` but not `===` to zero) to a sparse array
  stores it ([#296]).
* Adding a `UniformScaling` to a sparse matrix keeps its index type ([#158]).
* `all` and `any` of an empty sparse array, `findmin`, `findmax`, and reductions with an
  `init` value return the same results as for dense arrays ([#93], [#271], [#366]).
* `sparse!` resizes its output buffers to the lengths the result needs ([#314]).
* Solving `Hermitian` or `Symmetric` sparse systems with a sparse right-hand side throws an
  `ArgumentError` suggesting a dense right-hand side, and the polyalgorithm for sparse
  self-adjoint `\` no longer tries `ldlt!` ([#322], [#325]).
* Sparse matrices shown as a Braille spy plot have a border, and printing of large sparse
  vectors and of adjoints and transposes of sparse matrices is fixed ([#79], [#102],
  [#211]).

### Performance

* `lu!` on an `UmfpackLU` reuses its working buffers, and complex UMFPACK `ldiv!` allocates
  less ([#156], [#179]).
* `==` between sparse arrays and reductions such as `sum` and `maximum` take time
  proportional to the stored entries ([#227], [#244], [#248]).
* `kron` with `Bidiagonal`, `Tridiagonal` and `SymTridiagonal` matrices, and
  `SparseMatrixCSC(::AbstractQ)` and products with `AbstractQ`, use the structure of their
  arguments ([#196], [#236]).
* `transpose!` and `adjoint!` no longer allocate, `cat` of sparse arrays is type stable,
  and `getindex` elides bounds checks it has already made ([#92], [#157], [#273]).

### Deprecated or removed

* `fkeep!(A, f)` is deprecated in favor of `fkeep!(f, A)` ([#240]).

## Julia v1.8

Released with Julia v1.8.0 on 2022-08-17. SuiteSparse 5.10.1.

### New features

* `sparse_hcat`, `sparse_vcat` and `sparse_hvcat` return a `SparseMatrixCSC` whatever the
  types of their arguments. They provide the concatenation behavior in which a special
  "sparse" matrix argument, such as a `Diagonal`, made the result sparse, which is no
  longer possible now that LinearAlgebra is independent of SparseArrays
  ([JuliaLang/julia#43127]).

### Behavior changes

* The code for SparseArrays moved from the Julia repository to
  https://github.com/JuliaSparse/SparseArrays.jl; usage is unchanged
  ([JuliaLang/julia#43813]).
* LinearAlgebra is independent of SparseArrays, so its methods no longer return sparse
  arrays for Base or LinearAlgebra arguments: concatenations involving `Diagonal`,
  `Bidiagonal`, `Tridiagonal` and `SymTridiagonal` return dense matrices, and so does
  three-argument `similar` of these types ([JuliaLang/julia#43127]).
* `sparse` of a `Diagonal` stores only its nonzero entries ([JuliaLang/julia#42577]).
* `kron` of the adjoint or transpose of a sparse matrix returns a sparse matrix
  ([JuliaLang/julia#42181]).
* The adjoint and transpose of a large sparse matrix display as a Braille spy plot
  ([JuliaLang/julia#36177]), and two-argument `show` of a small sparse matrix prints it
  like any other `AbstractArray` ([JuliaLang/julia#41150]).

### Performance

* `findmax` and `findmin` of a sparse vector are O(nnz) ([JuliaLang/julia#42825]).

## Julia v1.7

Released with Julia v1.7.0 on 2021-11-30. SuiteSparse 5.10.1.

### New features

* `sizehint!(::SparseMatrixCSC, ::Integer)` ([JuliaLang/julia#40523]).
* `spzeros` accepts its dimensions as a tuple ([JuliaLang/julia#39886],
  [JuliaLang/julia#40515]).
* `logabsdet` of an `UmfpackLU` ([JuliaLang/julia#40716]).
* Conversion of a sparse matrix to `Diagonal`, `Tridiagonal`, `SymTridiagonal`,
  `LowerTriangular` and `UpperTriangular` ([JuliaLang/julia#40988]).
* `sum`, `minimum` and `maximum` of a sparse vector use the sparse method for any function
  `f` ([JuliaLang/julia#29884]).

### Behavior changes

* `cholesky` preserves the user-specified permutation ([JuliaLang/julia#40560]).
* `issparse` applies consistently to all wrapper arrays, including nested ones, by
  checking `issparse` on the wrapped parent array ([JuliaLang/julia#34308]).
* `SparseMatrixCSC` keeps `length(nonzeros(A)) == nnz(A)` for the arrays returned by
  exported functions ([JuliaLang/julia#40523]).
* The length of a `SparseVector` has its index type ([JuliaLang/julia#39645]).
* `sparse(x')` of a sparse vector `x` keeps the index type of `x`
  ([JuliaLang/julia#40836]).
* The SuiteSparse wrappers moved from Julia to the external SuiteSparse.jl repository
  ([JuliaLang/julia#40770]).

### Performance

* Products of dense and sparse matrices are faster ([JuliaLang/julia#38876]).
* `dot` of a dense vector, a sparse matrix and a dense vector is faster
  ([JuliaLang/julia#39889]).
* Products of sparse arrays with `BitMatrix` and `BitVector` use sparse multiplication
  ([JuliaLang/julia#39557]).

## Julia v1.6

Released with Julia v1.6.0 on 2021-03-24. SuiteSparse 5.4.0.

### New features

* `spdiagm([m, n,] v::AbstractVector)` methods, equivalent to `spdiagm([m, n,] 0 => v)`
  like their dense `diagm` counterparts ([JuliaLang/julia#37684]).
* `kron!`, an in-place Kronecker product, with sparse methods ([JuliaLang/julia#31069]).
* CHOLMOD solves with the adjoint or transpose of a sparse matrix
  ([JuliaLang/julia#38397]).
* Products of sparse triangular matrices with sparse matrices and vectors, which return
  sparse results ([JuliaLang/julia#35659]).

### Behavior changes

* Large sparse matrices display as a Unicode "spy" plot of their nonzero pattern, and
  small ones in a `Matrix`-like 2d layout of their contents ([JuliaLang/julia#33821]).

### Performance

* `extrema` of a sparse array is O(nnz) ([JuliaLang/julia#37429]).
* `any` and `all` with a predicate over the columns of a sparse matrix short-circuit
  ([JuliaLang/julia#37848]).
* Sparse matrix constructors that convert the element or index type are faster
  ([JuliaLang/julia#38315]).
* `blockdiag` of sparse matrices with the same element and index types is faster
  ([JuliaLang/julia#36013]).

## Julia v1.5

Released with Julia v1.5.0 on 2020-08-01. SuiteSparse 5.4.0.

### New features

* `lu!` accepts an `UmfpackLU` to reuse its symbolic factorization
  ([JuliaLang/julia#33738]).
* `qr` of a sparse matrix accepts a user-selected fill-reducing ordering
  ([JuliaLang/julia#35599]).

### Behavior changes

* The sum of two `Symmetric` or `Hermitian` sparse matrices is sparse
  ([JuliaLang/julia#35325]).
* The CHOLMOD wrapper is thread safe ([JuliaLang/julia#34546]).
* `issparse` is `true` for the `Adjoint` and `Transpose` of a sparse array
  ([JuliaLang/julia#34266]).
* `SparseMatrixCSC` and `SparseVector` constructors from sparse arrays return a copy that
  does not share storage with their argument ([JuliaLang/julia#34647]).

### Performance

* Right division of a sparse matrix by a `Diagonal` is faster ([JuliaLang/julia#35533]).
* `copyto!` from a sparse array into a dense one is faster ([JuliaLang/julia#33589]).

### Deprecated or removed

* The `trim` keyword argument of `fkeep!`, `tril!`, `triu!`, `droptol!`, `dropzeros!`
  and `dropzeros` is removed, and these functions always trim; calling them with
  `trim=false` could produce invalid sparse arrays ([JuliaLang/julia#32972]).

## Julia v1.4

Released with Julia v1.4.0 on 2020-03-21. SuiteSparse 5.4.0.

### New features

* Products involving sparse arrays accept more general element types, such as those of
  StaticArrays ([JuliaLang/julia#33205]).
* `AbstractSparseMatrixCSC`, an abstract type for compressed sparse column matrices that
  the SparseArrays functions accept ([JuliaLang/julia#33039]).
* Indexing a `SparseMatrixCSC` with an array of `CartesianIndex` ([JuliaLang/julia#33225]).
* Sparse methods for the three-argument `dot(x, A, y)` ([JuliaLang/julia#32739]).
* SPQR promotes the element type of its input to one it supports
  ([JuliaLang/julia#33601]).

### Behavior changes

* `zero(x::AbstractSparseArray)` stores no zeros, and takes constant time
  ([JuliaLang/julia#34209]).
* `sparse(I, J, V, m, n)` throws an error when the index type cannot hold the indices,
  instead of widening it ([JuliaLang/julia#33083]).
* `UmfpackLU` prints a summary in the REPL ([JuliaLang/julia#33705]).

## Julia v1.3

Released with Julia v1.3.0 on 2019-11-26. SuiteSparse 5.4.0.

### New features

* `spdiagm(m, n, pairs...)` builds a non-square matrix ([JuliaLang/julia#31654]).
* `SparseMatrixCSC` constructors for `Diagonal`, `Bidiagonal`, `Tridiagonal` and
  `SymTridiagonal` matrices ([JuliaLang/julia#32466], [JuliaLang/julia#32492]).
* Sparse matrices and vectors can hold non-numeric values ([JuliaLang/julia#30580]).
* Sparse methods for the five-argument `mul!(C, A, B, α, β)` ([JuliaLang/julia#29634]).

### Behavior changes

* `kron` of a sparse matrix or vector and a `Diagonal` returns a sparse matrix
  ([JuliaLang/julia#32793]).
* `SparseMatrixCSC(m, n, colptr, rowval, nzval)` checks its arguments: `colptr` must be
  properly populated, and the lengths of `colptr`, `rowval` and `nzval` must be
  compatible with `m`, `n` and `eltype(colptr)` ([JuliaLang/julia#31724]).
* `sparse(I, J, V, m, n)` checks that `I`, `J` and `V` have equal lengths, compatible
  with `eltype(I)`, `m` and `n` ([JuliaLang/julia#31724]).

### Performance

* Products of sparse and strided dense matrices are faster ([JuliaLang/julia#32195]).

## Julia v1.2

Released with Julia v1.2.0 on 2019-08-19. SuiteSparse 5.4.0.

### New features

* `rank` of a sparse matrix, computed with a sparse QR factorization
  ([JuliaLang/julia#30415]).
* `copy!` and `copyto!` into a `SparseVector` from any array ([JuliaLang/julia#30531],
  [JuliaLang/julia#30776]).

### Behavior changes

* Outer products of sparse vectors and column views, in the forms `kron(u, v')`, `u * v'`
  and `u .* v'`, return a sparse matrix ([JuliaLang/julia#24980]).
* The random stream of `sprand` and `sprandn` has changed ([JuliaLang/julia#30494]).
* `oneunit` of a sparse matrix returns a sparse matrix ([JuliaLang/julia#30228]).
* A sparse Cholesky solve with a vector right-hand side returns a `Vector`
  ([JuliaLang/julia#30416]).
* Sparse matrices print with less whitespace ([JuliaLang/julia#31461]).

### Performance

* Sparse matrix-matrix multiplication is faster ([JuliaLang/julia#30372]).
* `sprand` is 2 to 5 times faster ([JuliaLang/julia#30494]).
* Converting wrapped sparse matrices, such as transposes and triangular views, to
  `SparseMatrixCSC` is faster ([JuliaLang/julia#30552]).
* `circshift!` of a sparse matrix is faster ([JuliaLang/julia#30317]).

## Julia v1.1

Released with Julia v1.1.0 on 2019-01-21. SuiteSparse 4.4.5.

### New features

* `sprandn` accepts the element type of its result ([JuliaLang/julia#30090]).
* `lmul!` and `rmul!` of a sparse matrix with a `Diagonal` ([JuliaLang/julia#29296]).

### Performance

* Products and divisions involving sparse triangular matrices use sparse methods instead
  of a generic fallback, 300 to 1000 times faster ([JuliaLang/julia#28507]).
* Indexing a sparse matrix with a vector, and sorting the row indices of its columns, are
  faster ([JuliaLang/julia#29682], [JuliaLang/julia#29696]).

## Julia v1.0

Released with Julia v1.0.0 on 2018-08-08. SuiteSparse 4.4.5.

### Behavior changes

* SparseArrays is a standard library, moved out of Base in Julia v0.7; load it with
  `using SparseArrays`. Julia v1.0 is v0.7 without its deprecations, so the changes from
  v0.6 are in the [Julia v0.7 release notes].

### Performance

* `\` and `ldiv!` with the transpose or adjoint of a triangular sparse matrix use a
  specialized solver ([JuliaLang/julia#28242]).

### Deprecated or removed

* The SparseArrays deprecations of Julia v0.7 are removed ([JuliaLang/julia#28469]).

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
[#200]: https://github.com/JuliaSparse/SparseArrays.jl/pull/200
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
[#447]: https://github.com/JuliaSparse/SparseArrays.jl/pull/447
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
[#583]: https://github.com/JuliaSparse/SparseArrays.jl/pull/583
[#586]: https://github.com/JuliaSparse/SparseArrays.jl/pull/586
[#590]: https://github.com/JuliaSparse/SparseArrays.jl/pull/590
[#597]: https://github.com/JuliaSparse/SparseArrays.jl/pull/597
[#600]: https://github.com/JuliaSparse/SparseArrays.jl/pull/600
[#603]: https://github.com/JuliaSparse/SparseArrays.jl/pull/603
[#613]: https://github.com/JuliaSparse/SparseArrays.jl/pull/613
[#616]: https://github.com/JuliaSparse/SparseArrays.jl/pull/616
[#617]: https://github.com/JuliaSparse/SparseArrays.jl/pull/617
[#626]: https://github.com/JuliaSparse/SparseArrays.jl/pull/626
[#640]: https://github.com/JuliaSparse/SparseArrays.jl/pull/640
[#643]: https://github.com/JuliaSparse/SparseArrays.jl/pull/643
[#645]: https://github.com/JuliaSparse/SparseArrays.jl/pull/645
[#649]: https://github.com/JuliaSparse/SparseArrays.jl/pull/649
[#650]: https://github.com/JuliaSparse/SparseArrays.jl/pull/650
[#652]: https://github.com/JuliaSparse/SparseArrays.jl/pull/652
[#653]: https://github.com/JuliaSparse/SparseArrays.jl/pull/653
[#666]: https://github.com/JuliaSparse/SparseArrays.jl/pull/666
[#668]: https://github.com/JuliaSparse/SparseArrays.jl/pull/668
[#672]: https://github.com/JuliaSparse/SparseArrays.jl/pull/672
[#677]: https://github.com/JuliaSparse/SparseArrays.jl/pull/677
[#678]: https://github.com/JuliaSparse/SparseArrays.jl/pull/678
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
[#756]: https://github.com/JuliaSparse/SparseArrays.jl/pull/756
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
[#867]: https://github.com/JuliaSparse/SparseArrays.jl/pull/867
[#868]: https://github.com/JuliaSparse/SparseArrays.jl/pull/868
[#869]: https://github.com/JuliaSparse/SparseArrays.jl/pull/869
[#870]: https://github.com/JuliaSparse/SparseArrays.jl/pull/870
[#871]: https://github.com/JuliaSparse/SparseArrays.jl/pull/871
[#873]: https://github.com/JuliaSparse/SparseArrays.jl/pull/873
[#875]: https://github.com/JuliaSparse/SparseArrays.jl/pull/875
[#877]: https://github.com/JuliaSparse/SparseArrays.jl/pull/877
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
[#947]: https://github.com/JuliaSparse/SparseArrays.jl/pull/947
[#949]: https://github.com/JuliaSparse/SparseArrays.jl/pull/949
[#950]: https://github.com/JuliaSparse/SparseArrays.jl/pull/950
[#953]: https://github.com/JuliaSparse/SparseArrays.jl/pull/953
[#954]: https://github.com/JuliaSparse/SparseArrays.jl/pull/954
[#955]: https://github.com/JuliaSparse/SparseArrays.jl/pull/955
[#969]: https://github.com/JuliaSparse/SparseArrays.jl/pull/969
[#972]: https://github.com/JuliaSparse/SparseArrays.jl/pull/972
[#973]: https://github.com/JuliaSparse/SparseArrays.jl/pull/973
[#974]: https://github.com/JuliaSparse/SparseArrays.jl/pull/974
[#975]: https://github.com/JuliaSparse/SparseArrays.jl/pull/975
[#978]: https://github.com/JuliaSparse/SparseArrays.jl/pull/978
[#979]: https://github.com/JuliaSparse/SparseArrays.jl/pull/979
[#982]: https://github.com/JuliaSparse/SparseArrays.jl/pull/982
[#983]: https://github.com/JuliaSparse/SparseArrays.jl/pull/983
[#985]: https://github.com/JuliaSparse/SparseArrays.jl/pull/985
[#988]: https://github.com/JuliaSparse/SparseArrays.jl/pull/988
[#989]: https://github.com/JuliaSparse/SparseArrays.jl/pull/989
[#1002]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1002
[#1004]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1004
[#1005]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1005
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
[#1033]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1033
[#1036]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1036
[#1041]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1041
[#1042]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1042
[#1043]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1043
[#1045]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1045
[#1050]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1050
[#1052]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1052
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
[#1071]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1071
[#1073]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1073
[#1075]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1075
[#1077]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1077
[#1078]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1078
[#1080]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1080
[#1081]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1081
[#1082]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1082
[#1084]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1084
[#1086]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1086
[#1101]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1101
[#1102]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1102
[#1103]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1103
[#1105]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1105
[#1109]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1109
[#1110]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1110
[#1111]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1111
[#1113]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1113
[#1114]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1114
[#1115]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1115
[#1116]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1116
[#1118]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1118
[#1119]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1119
[#1120]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1120
[#1121]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1121
[#1123]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1123
[#1124]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1124
[#1125]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1125
[#1126]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1126
[#1127]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1127
[#1128]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1128
[#1131]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1131
[#1133]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1133
[#1134]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1134
[#1135]: https://github.com/JuliaSparse/SparseArrays.jl/pull/1135
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
[JuliaLang/julia#40770]: https://github.com/JuliaLang/julia/pull/40770
[JuliaLang/julia#40836]: https://github.com/JuliaLang/julia/pull/40836
[JuliaLang/julia#40988]: https://github.com/JuliaLang/julia/pull/40988
[JuliaLang/julia#41150]: https://github.com/JuliaLang/julia/pull/41150
[JuliaLang/julia#42181]: https://github.com/JuliaLang/julia/pull/42181
[JuliaLang/julia#42577]: https://github.com/JuliaLang/julia/pull/42577
[JuliaLang/julia#42825]: https://github.com/JuliaLang/julia/pull/42825
[JuliaLang/julia#43127]: https://github.com/JuliaLang/julia/pull/43127
[JuliaLang/julia#43813]: https://github.com/JuliaLang/julia/pull/43813
[JuliaLang/julia#44247]: https://github.com/JuliaLang/julia/pull/44247
[JuliaLang/julia#48979]: https://github.com/JuliaLang/julia/pull/48979
[JuliaLang/julia#49266]: https://github.com/JuliaLang/julia/pull/49266
