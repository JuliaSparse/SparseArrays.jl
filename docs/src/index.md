```@meta
EditURL = "https://github.com/JuliaSparse/SparseArrays.jl/blob/main/docs/src/index.md"
```

# Sparse Arrays

```@meta
DocTestSetup = :(using SparseArrays, LinearAlgebra)
```

The `SparseArrays` stdlib module provides sparse vectors and
[sparse matrices](https://en.wikipedia.org/wiki/Sparse_matrix): arrays with enough zeros
that storing only the other entries saves space and time compared to dense arrays.

```@contents
Pages = ["index.md", "SparseArrays.md"]
Depth = 2:4
```

## [Sparse Vectors and Matrices](@id man-sparse-vectors-matrices)

### [Compressed Sparse Column (CSC) Sparse Matrix Storage](@id man-csc)

In Julia, sparse matrices are stored in the [Compressed Sparse Column (CSC) format](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_.28CSC_or_CCS.29).
Julia sparse matrices have the type [`SparseMatrixCSC{Tv,Ti}`](@ref), where `Tv` is the
type of the stored values, and `Ti` is the integer type for storing column pointers and
row indices. The internal representation of `SparseMatrixCSC` is as follows:

```julia
struct SparseMatrixCSC{Tv,Ti<:Integer} <: AbstractSparseMatrixCSC{Tv,Ti}
    m::Int                  # Number of rows
    n::Int                  # Number of columns
    colptr::Vector{Ti}      # Column j is in colptr[j]:(colptr[j+1]-1)
    rowval::Vector{Ti}      # Row indices of stored values
    nzval::Vector{Tv}       # Stored values, typically nonzeros
end
```

CSC storage makes access to the entries of a column fast, and access by row considerably
slower. Inserting previously unstored entries one at a time is slow as well, because every
entry beyond the point of insertion has to be moved one place over. See
[Performance tips](@ref man-sparse-performance).

A `SparseMatrixCSC` may store explicit zeros. [`nnz`](@ref) counts the stored entries,
including stored zeros, and [`count(!iszero, x)`](@ref) counts the numerical nonzeros.
[`dropzeros`](@ref), and the in-place [`dropzeros!`](@ref), remove stored zeros.

```jldoctest
julia> A = sparse([1, 1, 2, 3], [1, 3, 2, 3], [0, 1, 2, 0])
3×3 SparseMatrixCSC{Int64, Int64} with 4 stored entries:
 0  ⋅  1
 ⋅  2  ⋅
 ⋅  ⋅  0

julia> dropzeros(A)
3×3 SparseMatrixCSC{Int64, Int64} with 2 stored entries:
 ⋅  ⋅  1
 ⋅  2  ⋅
 ⋅  ⋅  ⋅
```

Arrays in CSC format from another application or library can be wrapped, without copying,
with `SparseMatrixCSC(m, n, colptr, rowval, nzval)`. Within each column the row indices
must be sorted and unique, which the constructor does *not* check; the
[`SparseMatrixCSC`](@ref) docstring lists the invariants. Arrays from a 0-based source,
such as C or SciPy, need `1` added to `colptr` and `rowval`. To sort the row indices of a
matrix `A` wrapped this way, transpose it twice, `copy(transpose(copy(transpose(A))))`.

The arrays of an `m × n` matrix in compressed sparse row (CSR) format are the CSC arrays of
its transpose, so build the `n × m` matrix from them and transpose it:

```jldoctest
julia> rowptr = [1, 3, 4]; colval = [1, 3, 2]; nzval = [1.0, 2.0, 3.0];  # 2 × 3 CSR

julia> copy(transpose(SparseMatrixCSC(3, 2, rowptr, colval, nzval)))
2×3 SparseMatrixCSC{Float64, Int64} with 3 stored entries:
 1.0   ⋅   2.0
  ⋅   3.0   ⋅
```

### Sparse Vector Storage

Sparse vectors are stored in a close analog of the CSC format. They have the type
[`SparseVector{Tv,Ti}`](@ref), where `Tv` is the type of the stored values and `Ti` the
integer type of the indices. The internal representation is as follows:

```julia
struct SparseVector{Tv,Ti<:Integer} <: AbstractSparseVector{Tv,Ti}
    n::Ti               # Length of the sparse vector
    nzind::Vector{Ti}   # Indices of stored values
    nzval::Vector{Tv}   # Stored values, typically nonzeros
end
```

Like a `SparseMatrixCSC`, a `SparseVector` can store explicit zeros. Its length is stored
in the index type, so `length(x)` is a `Ti` rather than an `Int`.

### Sparse Vector and Matrix Constructors

The simplest way to create a sparse array is [`spzeros`](@ref), the equivalent of
[`zeros`](@ref) for dense arrays:

```jldoctest
julia> spzeros(3)
3-element SparseVector{Float64, Int64} with 0 stored entries
```

[`sparse`](@ref) builds a sparse matrix from a vector `I` of row indices, a vector `J` of
column indices and a vector `V` of stored values, known as the
[COO (coordinate) format](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_.28COO.29):
`sparse(I,J,V)` returns `S` such that `S[I[k], J[k]] = V[k]`. Its vector equivalent
[`sparsevec`](@ref) takes the index vector `I` and the values `V`, and returns `R` such that
`R[I[k]] = V[k]`. Values given for the same position are added.

```jldoctest sparse_function
julia> I = [1, 4, 3, 5]; J = [4, 7, 18, 9]; V = [1, 2, -5, 3];

julia> S = sparse(I,J,V)
5×18 SparseMatrixCSC{Int64, Int64} with 4 stored entries:
 ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅   ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅   ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  -5
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  2  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅   ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  3  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅   ⋅

julia> R = sparsevec(I,V)
5-element SparseVector{Int64, Int64} with 4 stored entries:
  [1]  =  1
  [3]  =  -5
  [4]  =  2
  [5]  =  3
```

[`findnz`](@ref) is the inverse of [`sparse`](@ref) and [`sparsevec`](@ref): it returns the
indices and values of the stored entries, including stored zeros.
[`findall(!iszero, x)`](@ref) returns the indices of the nonzero entries of `x`, excluding
stored zeros, and `sparse` also accepts the positions as such a vector of `CartesianIndex`es.

```jldoctest sparse_function
julia> findnz(S)
([1, 4, 5, 3], [4, 7, 9, 18], [1, 2, 3, -5])

julia> findnz(R)
([1, 3, 4, 5], [1, -5, 2, 3])

julia> sparse(findall(!iszero, S), [1, 2, 3, -5]) == S
true
```

[`sparse`](@ref) also converts a dense array to a sparse one, the [`Array`](@ref)
constructor converts back, and [`issparse`](@ref) tells whether an array is sparse.

```jldoctest
julia> sparse([1.0, 0.0, 1.0])
3-element SparseVector{Float64, Int64} with 2 stored entries:
  [1]  =  1.0
  [3]  =  1.0

julia> issparse(spzeros(5))
true
```

Other constructors build common matrices directly. `sparse(I, n, n)` is the identity,
[`spdiagm`](@ref) places vectors on diagonals, and [`blockdiag`](@ref) puts matrices along
the diagonal of a larger one. [`sprand`](@ref) and [`sprandn`](@ref) return a random vector
or matrix in which each entry is stored with probability `d`, with uniformly or normally
distributed values.

```jldoctest
julia> spdiagm(0 => [1, 2, 3], 1 => [4, 5])
3×3 SparseMatrixCSC{Int64, Int64} with 5 stored entries:
 1  4  ⋅
 ⋅  2  5
 ⋅  ⋅  3

julia> blockdiag(sparse(2I, 2, 2), sparse([3 4])) == [2 0 0 0; 0 2 0 0; 0 0 3 4]
true
```

### Sparse matrix operations

Arithmetic, indexing, assignment and concatenation work on sparse arrays as they do on
dense arrays. A result is sparse unless it would store every entry:

| Operation | Result |
|:----------|:-------|
| `A + B`, `A - B`, `A * B`, `kron(A, B)` of sparse arrays | sparse |
| `A + D`, `A - D`, `A * D` with a dense array `D` | dense |
| `A[I, J]`, `A[:, j]`, `A[i, :]` | sparse |
| `[A B]`, `[A; B]` with at least one sparse argument | sparse |
| `sum(A; dims)` and the other reductions along a dimension | dense, or sparse with `sparse = true` |

A reduction along a dimension with `sparse = true`, such as `sum(A; dims = 2, sparse = true)`,
stores an entry only for the rows of `A` that store one, and its cost is proportional to
the number of stored entries. `prod`, `maximum`, `minimum`, `count`, `any`, `all` and
`mapreduce` accept the keyword too.

[`permute`](@ref) applies a row and a column permutation in one pass, and
[`indtype`](@ref) returns the index type `Ti` of a sparse array.

Indexing, and especially assignment, is expensive one element at a time. See
[Performance tips](@ref man-sparse-performance).

#### [Broadcasting and `map`](@id man-sparse-broadcast)

[`broadcast`](@ref) (including dot syntax such as `A .* B`) and [`map`](@ref) over sparse
arrays return a sparse result when the function maps zeros to zero. To decide, the function
is evaluated once on the zeros of the arguments' element types:

| Arguments | `f(0, 0, ...)` is zero | `f(0, 0, ...)` is not zero |
|:----------|:-----------------------|:---------------------------|
| one sparse array | sparse, with the stored entries of the argument | `Array` |
| several sparse arrays | sparse, storing only the nonzero results | `Array` |
| sparse arrays and a scalar | the scalar is folded into `f`; then as above | sparse with every entry stored; `Array` if `f` is made only of `+` and `-` |

So `2 .* A`, `abs.(A)` and `Float64.(A)` keep the stored zeros of `A`, just as `2A`, `-A`
and `float(A)` do, while a combination of several sparse arrays drops the zeros it computes:

```jldoctest sparsebroadcast
julia> A = sparse([1, 1, 2, 3], [1, 2, 2, 3], [1, 0, -2, 3])
3×3 SparseMatrixCSC{Int64, Int64} with 4 stored entries:
 1   0  ⋅
 ⋅  -2  ⋅
 ⋅   ⋅  3

julia> B = sparse([1, 1, 3], [1, 3, 3], [1, 5, -3]);

julia> A .+ B
3×3 SparseMatrixCSC{Int64, Int64} with 3 stored entries:
 2   ⋅  5
 ⋅  -2  ⋅
 ⋅   ⋅  ⋅

julia> iszero.(A)
3×3 BitMatrix:
 0  1  1
 1  0  1
 1  1  0

julia> A .+ 2
3×3 Matrix{Int64}:
 3  2  2
 2  0  2
 2  2  5
```

A result such as `2 .* A .+ 1` or `A .^ 0`, sparse with every entry stored, needs more
memory than the equivalent `Array`; convert to dense first when that is what is meant.

Arguments that are not sparse arrays are handled as follows:

| Argument | Treatment |
|:---------|:----------|
| a `Vector` or `Matrix`, or its adjoint or transpose | converted to sparse, so `A .* D` is sparse; a broadcast made only of `+` and `-` returns an `Array`, as `A + D` does |
| a `Diagonal`, `Bidiagonal`, `Tridiagonal` or `SymTridiagonal` matrix | converted to sparse |
| a view of whole columns of a sparse matrix, or of a range of a sparse vector | treated as a sparse array |
| anything else: a tuple, a range, a triangular or `Symmetric` wrapper, a view of some rows, an array of more than two dimensions | the generic broadcast, which visits every element and returns an `Array` |

```jldoctest sparsebroadcast
julia> A .* Diagonal([1, 2, 3])
3×3 SparseMatrixCSC{Int64, Int64} with 3 stored entries:
 1   ⋅  ⋅
 ⋅  -4  ⋅
 ⋅   ⋅  9

julia> typeof(A .* (1:3))
Matrix{Int64} (alias for Array{Int64, 2})
```

Whether a result is sparse depends on the value `f(0, 0, ...)`, so its type is inferred
only when the compiler can evaluate `f` there, as it does for the standard number types.
A broadcast made only of `+`, `-` and `*` is always inferable.

`map` over sparse arrays always returns a sparse array, with every entry stored when
`f(0, 0, ...)` is not zero. It requires all arguments to have the same shape, whereas
`broadcast` expands singleton dimensions, with a sparse vector behaving as a one-column
matrix.

[`broadcast!`](@ref), [`map!`](@ref) and `.=` with a sparse destination overwrite its stored
pattern with that of the result. The destination may also be one of the arguments, as in
`C .= C .+ C'`. One exception: `C .= 0` calls [`fill!`](@ref), which keeps the pattern of `C`
and sets the stored values to zero.

To apply a function to the stored values only, leaving the pattern untouched whatever the
function returns for zero, broadcast over [`nonzeros`](@ref) instead:

```jldoctest sparsebroadcast
julia> C = A .* B;

julia> nonzeros(C) .= nonzeros(C) .+ 9;

julia> C
3×3 SparseMatrixCSC{Int64, Int64} with 2 stored entries:
 10  ⋅  ⋅
  ⋅  ⋅  ⋅
  ⋅  ⋅  0
```

#### [Products and norms](@id man-sparse-products)

The functions of LinearAlgebra have methods for sparse arrays. Products, `dot`, `kron` and
the norms are implemented in this package. The factorizations, and the solves that use
them, call [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) and are
described on the Sparse Linear Solvers page.

The result of `*` follows the other factor:

| Product of a sparse matrix with | Result |
|:--------------------------------|:-------|
| a `Vector` or a `Matrix` | dense |
| a sparse vector | `SparseVector` |
| a sparse matrix | `SparseMatrixCSC` |
| a `Diagonal`, `Bidiagonal`, `Tridiagonal` or `SymTridiagonal` matrix | `SparseMatrixCSC` |

`mul!(y, A, x)` writes the product into `y`, and `mul!(y, A, x, α, β)` computes
`A*x*α + y*β` in place. Products with `A'` and `transpose(A)` do not materialize the
transpose when the other factor is dense or a sparse vector: see
[Lazy `transpose` and `adjoint`](@ref man-sparse-lazy-transpose).

- `dot(x, y)` of sparse or dense vectors and matrices visits the stored entries only, and
  `dot(x, A, y)` computes `x' * A * y` without forming `A * y`.
- `kron` of two sparse arrays, or of a sparse array and a dense or `Diagonal` matrix, is
  sparse.
- `norm(A, p)` treats `A` as a vector of its entries, so `norm(A)` is the Frobenius norm.
  [`opnorm(A, p)`](@ref SparseArrays.opnorm(::SparseArrays.AbstractSparseMatrixCSC, ::Real))
  is the operator norm for `p` equal to `1`, `2` or `Inf`; for `p = 2` it is an iterative
  estimate.
- `cond(A, 1)` and `cond(A, Inf)` estimate the norm of the inverse from an `lu`
  factorization. The 2-norm condition number is not
  available; use `cond(Array(A))` for a matrix small enough.
- `tr`, `diag`, `triu` and `tril` work on the stored entries, and the last three return
  sparse results.

```jldoctest
julia> A = sparse([2.0 1 0; 0 3 1; 1 0 4]); x = [1.0, 2.0, 3.0];

julia> y = similar(x); mul!(y, A, x)
3-element Vector{Float64}:
  4.0
  9.0
 13.0

julia> dot(x, A, x)
61.0

julia> norm(A, 1), opnorm(A, 1), opnorm(A, Inf)
(12.0, 5.0, 5.0)
```

```@docs
SparseArrays.opnorm(::SparseArrays.AbstractSparseMatrixCSC, ::Real)
```

### [Performance tips](@id man-sparse-performance)

Sparse code is fast when its cost is proportional to the number of stored entries, and the
[CSC layout](@ref man-csc) determines which operations have that property.

#### Iterate over stored entries by column

The stored entries of column `j` are located at the positions [`nzrange(A, j)`](@ref nzrange) of
[`rowvals(A)`](@ref rowvals) and [`nonzeros(A)`](@ref nonzeros). Looping over the columns, and over
that range within each column, visits every stored entry once and in memory order, without
searching. For example, a matrix-vector product can be written as:

```jldoctest perftips
julia> A = sparse([1, 1, 2, 3], [1, 3, 2, 3], [1.0, 2.0, 3.0, 4.0]);

julia> function mymul(A::SparseMatrixCSC, x::Vector)
           y = zeros(promote_type(eltype(A), eltype(x)), size(A, 1))
           rows, vals = rowvals(A), nonzeros(A)
           for j in axes(A, 2), k in nzrange(A, j)
               y[rows[k]] += vals[k] * x[j]
           end
           return y
       end;

julia> mymul(A, [1.0, 10.0, 100.0]) == A * [1.0, 10.0, 100.0]
true
```

The same three accessors work on a [`SparseVector`](@ref), which is treated as a single
column, and on a view of whole columns such as `@view A[:, j]` or `@view A[:, 2:3]`. They
are not defined for a triangular, `Symmetric` or `Hermitian` wrapper; work on its `parent`.
[`getcolptr`](@ref), [`getrowval`](@ref) and [`getnzval`](@ref) return the three storage
vectors of a matrix themselves.

In contrast, scalar indexing `A[i, j]` has to do a binary search of column `j` for row `i`. A
loop over all `(i, j)` of an `m`-by-`n` matrix therefore performs `m * n` searches, however few
entries are stored.

#### Build a matrix from its entries in one call

Storing a value at a position that has no stored entry yet moves all the later entries, so
filling `spzeros(m, n)` one element at a time takes time proportional to `nnz(A)` for each
insertion. `sizehint!` avoids reallocating, but not moving the entries. Instead, collect the
row indices, column indices and values in three vectors and call [`sparse`](@ref) once.
Entries with the same position are added together, or combined with the function passed as
the last argument, as in `sparse(I, J, V, m, n, max)`.

To change many entries of an existing matrix, convert it to that form with
[`findnz`](@ref), work on the three vectors, and rebuild it.

#### Slice columns, not rows

`A[:, j]` copies one contiguous range of the stored entries. `A[i, :]` has to search every column for
row `i`, so its cost grows with the number of columns even when the row is empty. When an algorithm
works row by row, transpose the matrix once and work on the columns of the result. `permutedims(A)`
and `copy(transpose(A))` build the transposed matrix in time proportional to `nnz(A)`.

```jldoctest perftips
julia> At = permutedims(A);

julia> At[:, 1] == A[1, :]
true
```

#### Choose a smaller index type

The index type `Ti` is used for the column pointers and for one row index per stored entry. With `Float64`
values, `Int32` indices reduce the memory per stored entry from 16 to 12 bytes. `Ti` is taken from the
index vectors given to [`sparse`](@ref), can be given to [`spzeros`](@ref), and an existing matrix is
converted with the type constructor. Both dimensions have to be at most `typemax(Ti)`, and the number of
stored entries has to be less than `typemax(Ti)`. Operations between matrices with different
index types promote to the wider one, so use one index type consistently.

```jldoctest perftips
julia> A32 = SparseMatrixCSC{Float64,Int32}(A)
3×3 SparseMatrixCSC{Float64, Int32} with 4 stored entries:
 1.0   ⋅   2.0
  ⋅   3.0   ⋅
  ⋅    ⋅   4.0

julia> spzeros(Float64, Int8, 200, 200)
ERROR: ArgumentError: number of rows (m = 200) does not fit in Ti = Int8
[...]
```

#### [Lazy `transpose` and `adjoint`](@id man-sparse-lazy-transpose)

`transpose(A)` and `A'` do not copy; they return `Transpose` and `Adjoint` wrappers around
`A`. What happens to a wrapper depends on the function:

- Products with dense vectors and matrices (including `mul!`) and with sparse vectors, `\`,
  `==`, and `A'[i, :]`, which is a column slice of `A`, work on the wrapper without
  materializing the transpose.
- Products with another sparse matrix, broadcasting (which includes `+` and `-`), `map`,
  concatenation, `kron` and `findnz` first copy the wrapper into a new `SparseMatrixCSC`,
  in time proportional to `nnz(A)`, on every call.
- Other functions, such as `sum`, `norm` and indexing other than by row, reach generic
  `AbstractMatrix` methods that visit the wrapper element by element.

When a transposed matrix is used more than once, or is passed to code outside the first
group, materialize it with `copy`, or with `sparse` when the argument may or may not be a
wrapper.

```jldoctest perftips
julia> A' * [1.0, 10.0, 100.0]
3-element Vector{Float64}:
   1.0
  30.0
 402.0

julia> copy(A')
3×3 SparseMatrixCSC{Float64, Int64} with 4 stored entries:
 1.0   ⋅    ⋅
  ⋅   3.0   ⋅
 2.0   ⋅   4.0
```

#### Keep results free of stored zeros

Assigning zero to a stored entry, and cancellation in a matrix product, leave explicitly
stored zeros behind. They are harmless for correctness but are visited by every kernel.
[`dropzeros!`](@ref) removes them, [`droptol!`](@ref) removes entries of small magnitude,
and [`fkeep!`](@ref) keeps the entries for which a predicate of `(i, j, v)` is true, all in
place.

```jldoctest perftips
julia> B = copy(A); B[1, 1] = 0; nnz(B)
4

julia> nnz(dropzeros!(B))
3

julia> fkeep!((i, j, v) -> i == j, B)
3×3 SparseMatrixCSC{Float64, Int64} with 2 stored entries:
 ⋅   ⋅    ⋅
 ⋅  3.0   ⋅
 ⋅   ⋅   4.0
```

## [Specialized Sparse Arrays](@id man-sparse-specialized)

### [User-defined element types](@id man-sparse-eltype)

A sparse array can store values of any type `T`. Building one from indices and values,
copying, slicing, concatenating and permuting it, and reading a stored entry call nothing
on the elements. Everything else rests on three things:

- **`zero(T)`** is the value of an entry that is not stored. It is called on the type, not
  on a value, by whatever has to produce such an entry: indexing, `setindex!`,
  `Matrix(A)`, a product with a dense array or a sparse vector, `A + B`, `map`,
  broadcasting and the reductions.
- **`iszero(x)`** decides whether a value has to be stored. [`sparse`](@ref) of a dense
  array, [`dropzeros!`](@ref), `A + B`, `map` and broadcasting call it. It falls back to
  `x == zero(x)`.
- **The arithmetic of the operation itself**: `+` and `*` for a product, `isless` for
  `maximum`, `minimum` and `sort`, and so on. `sum` adds the `n` entries that are not
  stored in one step, as `zero(T) * n`, so it also needs the product of an element with an
  integer.

The sparse kernels skip the entries that are not stored, so they assume that `zero(T)`
behaves as a zero: `x + zero(T) == x`, and a product with `zero(T)` is zero. A type that
defines `zero`, `+` and `*` is enough for most of the package, and every `Number`
qualifies. For other element types:

- Multiplying two sparse matrices needs only `*` and `+`, not `zero(T)`. So a matrix of
  elements that have no `zero(T)`, such as `Matrix{Float64}`, can be stored and multiplied
  by another, but not indexed at an entry that is not stored.
- `c * A`, `A * c` and `A / c` are defined for a scalar `c` that is a `Number`. For any
  other element type, use `map` or broadcasting.
- The sparse factorizations work in real and complex floating
  point. A matrix of integers or rationals is converted to `Float64` first; a matrix of
  another element type, such as `BigFloat`, is rejected.

### [Sparse arrays with a fixed pattern](@id man-sparse-fixed)

!!! warning "Experimental"
    `fixed`, `FixedSparseCSC` and `FixedSparseVector` are experimental and not exported.

Writing into a `SparseMatrixCSC` can change which entries it stores. `W .= M .- γ .* J`
drops an entry of `W` wherever the difference is zero, and `W[i, j] = v` inserts one. Code
that has derived something from the pattern of `W`, such as a symbolic factorization or
the positions of its entries in `nonzeros(W)`, is then silently out of date.

A fixed sparse array rules that out. It is a **destination whose sparsity pattern never
changes**: the stored values can be overwritten, and no entry can be added or removed.
[`SparseArrays.fixed`](@ref) makes one, with the arguments of [`sparse`](@ref) or from a
sparse array, which it copies. Three rules describe everything a fixed array does; the
[`SparseArrays.FixedSparseCSC`](@ref) docstring lists the functions each one covers.

**A write lands inside the pattern or throws.** A write, such as `W .= ...`, `copyto!`,
`mul!` or `W[i, j] = v`, stores a value at each stored entry, a zero included, and leaves
the pattern alone. What is written need not fill the pattern: the stored entries it leaves
out become zero. A write that needs an entry where the pattern has none throws an
`ArgumentError`, and the kernels of this package check for that before they change any
value. Writing through `nonzeros(W)` is always allowed.

```jldoctest fixedpattern
julia> J = sparse([1.0 2 0; 0 3 0; 4 0 5]); M = sparse(1.0I, 3, 3);

julia> nnz(M .- J)   # the entry at (1, 1) cancels and is not stored
4

julia> W = SparseArrays.fixed(M + J);

julia> W .= M .- J
3×3 SparseArrays.FixedSparseCSC{Float64, Int64} with 5 stored entries:
  0.0  -2.0    ⋅
   ⋅   -2.0    ⋅
 -4.0    ⋅   -4.0

julia> W[1, 3] = 1.0
ERROR: ArgumentError: cannot store a new entry at (1, 3) in a FixedSparseCSC, its sparsity pattern is read-only
[...]
```

**A function that exists to change the pattern throws.** [`dropzeros!`](@ref),
[`droptol!`](@ref), [`fkeep!`](@ref), `tril!`, `sort!`, `permute!` and the like refuse a
fixed array as the argument they change. Their out-of-place forms work, because of the next
rule.

```jldoctest fixedpattern
julia> dropzeros!(W)
ERROR: ArgumentError: cannot `dropzeros!` a FixedSparseCSC: it would change the sparsity pattern, which is read-only; apply it to a writable copy, `SparseMatrixCSC(A)`
[...]
```

**Reading a fixed array gives an ordinary result.** As an operand, a fixed array behaves
as the `SparseMatrixCSC` or `SparseVector` with the same entries, and what comes back is
what that array would give. Being fixed is a property of one array, not of its values, so
it does not carry over to indexing, arithmetic, broadcasting, concatenation or `similar`.
Only `copy(W)` is fixed, and shares the pattern of `W`; `SparseArrays.fixed(A)` fixes any
other result, and `SparseMatrixCSC(W)` gives a writable copy.

```jldoctest fixedpattern
julia> typeof(2W), typeof(similar(W))
(SparseMatrixCSC{Float64, Int64}, SparseMatrixCSC{Float64, Int64})

julia> V = copy(W); rowvals(V) === rowvals(W)
true
```

The factorizations accept a fixed matrix, which is what makes reusing a symbolic
factorization safe: see "Reusing a factorization" on the Sparse Linear Solvers page.

```@meta
DocTestSetup = nothing
```

## [SparseArrays API](@id stdlib-sparse-arrays)

```@docs
SparseArrays
```

### Types

```@docs
SparseArrays.AbstractSparseArray
SparseArrays.AbstractSparseVector
SparseArrays.AbstractSparseMatrix
SparseArrays.AbstractSparseMatrixCSC
SparseArrays.SparseVector
SparseArrays.SparseMatrixCSC
```

### Constructors

```@docs
SparseArrays.sparse
SparseArrays.sparse!
SparseArrays.sparsevec
SparseArrays.spzeros
SparseArrays.spzeros!
SparseArrays.spdiagm
SparseArrays.blockdiag
SparseArrays.sprand
SparseArrays.sprandn
Base.similar(::SparseArrays.AbstractSparseMatrixCSC, ::Type)
```

### Stored entries

```@docs
SparseArrays.issparse
SparseArrays.nnz
SparseArrays.findnz
SparseArrays.nonzeros
SparseArrays.rowvals
SparseArrays.nzrange
SparseArrays.getcolptr
SparseArrays.getrowval
SparseArrays.getnzval
SparseArrays.indtype
```

### Removing stored entries

```@docs
SparseArrays.dropzeros
SparseArrays.dropzeros!
SparseArrays.droptol!
SparseArrays.fkeep!
```

### Permuting, transposing and sorting

```@docs
SparseArrays.permute
Base.permute!(::SparseArrays.AbstractSparseMatrixCSC{Tv,Ti}, ::SparseArrays.AbstractSparseMatrixCSC{Tv,Ti}, ::AbstractVector{<:Integer}, ::AbstractVector{<:Integer}) where {Tv,Ti}
SparseArrays.ftranspose!
SparseArrays.transpose!(::SparseArrays.AbstractSparseMatrixCSC{Tv,Ti}, ::SparseArrays.AbstractSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
SparseArrays.adjoint!(::SparseArrays.AbstractSparseMatrixCSC{Tv,Ti}, ::SparseArrays.AbstractSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
Base.sort!(::SparseArrays.AbstractSparseMatrixCSC)
Base.sort(::SparseArrays.AbstractSparseMatrixCSC)
Base.sort!(::Union{SparseArrays.AbstractCompressedVector, SparseArrays.SparseColumnView})
```

### Concatenation

```@docs
SparseArrays.sparse_hcat
SparseArrays.sparse_vcat
SparseArrays.sparse_hvcat
```

### Fixed pattern (experimental)

```@docs
SparseArrays.fixed
SparseArrays.FixedSparseCSC
SparseArrays.FixedSparseVector
```

## Noteworthy External Sparse Packages

Other Julia packages provide further sparse formats and operations:

1. [SuiteSparseGraphBLAS.jl](https://github.com/JuliaSparse/SuiteSparseGraphBLAS.jl) is a wrapper over the fast, multithreaded SuiteSparse:GraphBLAS C library.

2. [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) exposes the [CUSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html) library for GPU sparse matrix operations.

3. [SparseMatricesCSR.jl](https://github.com/gridap/SparseMatricesCSR.jl) provides a Julia native implementation of the Compressed Sparse Rows (CSR) format.

4. [MKLSparse.jl](https://github.com/JuliaSparse/MKLSparse.jl) accelerates SparseArrays sparse-dense matrix operations using Intel's MKL library.

5. [SparseArrayKit.jl](https://github.com/Jutho/SparseArrayKit.jl) provides multidimensional sparse arrays.

6. [LuxurySparse.jl](https://github.com/QuantumBFS/LuxurySparse.jl) provides static sparse array formats, as well as a coordinate format.

7. [ExtendableSparse.jl](https://github.com/j-fu/ExtendableSparse.jl) enables fast insertion into sparse matrices using a lazy approach to new stored indices.

8. [Finch.jl](https://github.com/willow-ahrens/Finch.jl) supports extensive multidimensional sparse array formats and operations through a mini tensor language and compiler, all in native Julia. Support for COO, CSF, CSR, CSC and more, as well as operations like broadcast, reduce, etc. and custom operations.

External packages providing sparse direct solvers:
1. [KLU.jl](https://github.com/JuliaSparse/KLU.jl)
2. [Pardiso.jl](https://github.com/JuliaSparse/Pardiso.jl/)

External packages providing solvers for iterative solution of eigensystems and singular value decompositions:
1. [ArnoldiMethod.jl](https://github.com/JuliaLinearAlgebra/ArnoldiMethod.jl)
2. [KrylovKit](https://github.com/Jutho/KrylovKit.jl)
3. [Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl)

External packages for working with graphs:
1. [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl)
