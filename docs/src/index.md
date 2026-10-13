```@meta
EditURL = "https://github.com/JuliaSparse/SparseArrays.jl/blob/main/docs/src/index.md"
```

# Sparse Arrays

```@meta
DocTestSetup = :(using SparseArrays, LinearAlgebra)
```

The `SparseArrays` stdlib module provides sparse vectors and [sparse
matrices](https://en.wikipedia.org/wiki/Sparse_matrix): arrays with enough zeros that
storing only the other entries saves space and time compared to dense arrays.

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
entry beyond the point of insertion has to be moved one place over. See [Performance
tips](@ref man-sparse-performance).

Data in CSC format from another application or library can be wrapped directly with
`SparseMatrixCSC(m, n, colptr, rowval, nzval)`. The arrays are not copied, so the matrix
aliases them; pass copies to avoid that. They must satisfy these invariants:

  * `colptr` has length `n + 1`, starts at `1`, and is nondecreasing;
  * `rowval` and `nzval` both have length `colptr[end] - 1`, and `rowval` has the same
    element type as `colptr`;
  * within each column, the row indices are sorted, unique, and in `1:m`.

The constructor throws an `ArgumentError` if the first two are violated, but it does *not*
inspect the row indices. A matrix with unsorted, repeated or out-of-range row indices is
constructed silently and then gives inconsistent results.

Arrays from C, Python (SciPy's `indptr` and `indices`) and other 0-based sources need `1`
added to `colptr` and `rowval`.

To sort the row indices, transpose twice, with a `copy` to materialize each lazy transpose.
Alternatively, rebuild the matrix from its coordinates with [`findnz`](@ref) and
[`sparse`](@ref), which also adds up repeated entries:

```jldoctest cscimport
julia> colptr = [1, 3, 4, 6]; rowval = [3, 1, 2, 3, 1]; nzval = [20.0, 10.0, 30.0, 50.0, 40.0];

julia> A = SparseMatrixCSC(3, 3, colptr, rowval, nzval);  # unsorted row indices

julia> B = copy(transpose(copy(transpose(A))))
3×3 SparseMatrixCSC{Float64, Int64} with 5 stored entries:
 10.0    ⋅   40.0
   ⋅   30.0    ⋅
 20.0    ⋅   50.0

julia> B == sparse(findnz(A)..., size(A)...)
true
```

The arrays of an `m × n` matrix in compressed sparse row (CSR) format are the CSC arrays of
its transpose, so build the `n × m` matrix from them and transpose it:

```jldoctest
julia> rowptr = [1, 3, 4]; colval = [1, 3, 2]; nzval = [1.0, 2.0, 3.0];  # 2 × 3 CSR

julia> copy(transpose(SparseMatrixCSC(3, 2, rowptr, colval, nzval)))
2×3 SparseMatrixCSC{Float64, Int64} with 3 stored entries:
 1.0   ⋅   2.0
  ⋅   3.0   ⋅
```

A `SparseMatrixCSC` may store explicit zeros. Functions accept them, and many routines treat
them as structural nonzeros, but mutating operations are not guaranteed to preserve them.
[`nnz`](@ref) counts the stored entries, including stored zeros; [`count(!iszero, x)`](@ref)
counts the numerical nonzeros by inspecting every stored entry. [`dropzeros`](@ref), and the
in-place [`dropzeros!`](@ref), remove stored zeros.

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

Like [`SparseMatrixCSC`](@ref), a `SparseVector` can contain explicitly stored zeros (see
[Sparse Matrix Storage](@ref man-csc)).

The length is stored in the index type, so `length(x)` and `size(x)` of a
`SparseVector{Tv,Ti}` return values of type `Ti`, not `Int`. A vector can then be longer than
`typemax(Int)` when `Ti` is `BigInt`, but arithmetic on the length of a vector with a small
index type can overflow: for an `Int8`-indexed vector of length 100, `length(x) + length(x)`
is `-56`. Convert with `Int(length(x))` before computing with it.

### Sparse Vector and Matrix Constructors

The simplest way to create a sparse array is [`spzeros`](@ref), the equivalent of
[`zeros`](@ref) for dense arrays:

```jldoctest
julia> spzeros(3)
3-element SparseVector{Float64, Int64} with 0 stored entries
```

[`sparse`](@ref) builds a sparse matrix from a vector `I` of row indices, a vector `J` of
column indices and a vector `V` of stored values, known as the [COO (coordinate)
format](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_.28COO.29):
`sparse(I,J,V)` returns `S` such that `S[I[k], J[k]] = V[k]`. Its vector equivalent
[`sparsevec`](@ref) takes the index vector `I` and the values `V`, and returns `R` such that
`R[I[k]] = V[k]`.

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
stored zeros.

```jldoctest sparse_function
julia> findnz(S)
([1, 4, 5, 3], [4, 7, 9, 18], [1, 2, 3, -5])

julia> findall(!iszero, S)
4-element Vector{CartesianIndex{2}}:
 CartesianIndex(1, 4)
 CartesianIndex(4, 7)
 CartesianIndex(5, 9)
 CartesianIndex(3, 18)

julia> findnz(R)
([1, 3, 4, 5], [1, -5, 2, 3])

julia> findall(!iszero, R)
4-element Vector{Int64}:
 1
 3
 4
 5
```

The positions can also be given to [`sparse`](@ref) as one vector of `CartesianIndex`es, as
`findall` returns them:

```jldoctest sparse_function
julia> sparse(findall(!iszero, S), [1, 2, 3, -5]) == S
true
```

[`sparse`](@ref) also converts a dense array to a sparse one:

```jldoctest
julia> sparse(Matrix(1.0I, 5, 5))
5×5 SparseMatrixCSC{Float64, Int64} with 5 stored entries:
 1.0   ⋅    ⋅    ⋅    ⋅
  ⋅   1.0   ⋅    ⋅    ⋅
  ⋅    ⋅   1.0   ⋅    ⋅
  ⋅    ⋅    ⋅   1.0   ⋅
  ⋅    ⋅    ⋅    ⋅   1.0

julia> sparse([1.0, 0.0, 1.0])
3-element SparseVector{Float64, Int64} with 2 stored entries:
  [1]  =  1.0
  [3]  =  1.0
```

The [`Array`](@ref) constructor converts back, and [`issparse`](@ref) tells whether an array
is sparse.

```jldoctest
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

Arithmetic, indexing, assignment and concatenation work on sparse matrices as they do on
dense matrices. Indexing, and especially assignment, is expensive one element at a time. It
is often better to convert the matrix to `(I,J,V)` format with [`findnz`](@ref), manipulate
those dense vectors, and rebuild the matrix with [`sparse`](@ref).

Reductions along a dimension, such as `sum(S; dims = 2)`, return a dense `Matrix`, as for dense
input. To keep the result sparse, pass `sparse = true`: `sum(S; dims = 2, sparse = true)` stores an
entry only for the rows of `S` that store one, at a cost proportional to the number of stored
entries rather than to the number of rows. `prod`, `maximum`, `minimum`, `count`, `any`, `all` and
`mapreduce` accept the keyword in the same way, as do adjoints and transposes of sparse matrices,
views of a subset of their columns, and sparse vectors, for which the result is a `SparseVector`.

#### [Broadcasting and `map`](@id man-sparse-broadcast)

[`broadcast`](@ref) (including dot syntax such as `A .* B`) and [`map`](@ref) over sparse vectors
and matrices return a sparse result when the function maps zeros to zero. To decide which entries to
store, the function is first evaluated once on the zeros of the arguments' element types. If `f(0, 0, ...)` is zero, as for
`A .* B`, `abs.(A)` or `2 .* A`, only positions where some argument has a stored entry are visited.
With a single sparse argument the result has exactly that argument's stored entries, so `2 .* A`,
`abs.(A)` and `Float64.(A)` keep the stored zeros of `A`, just as `2A`, `-A` and `float(A)` do:

```jldoctest sparsebroadcast
julia> A = sparse([1, 1, 2, 3], [1, 2, 2, 3], [1, 0, -2, 3])
3×3 SparseMatrixCSC{Int64, Int64} with 4 stored entries:
 1   0  ⋅
 ⋅  -2  ⋅
 ⋅   ⋅  3

julia> Float64.(A)
3×3 SparseMatrixCSC{Float64, Int64} with 4 stored entries:
 1.0   0.0   ⋅
  ⋅   -2.0   ⋅
  ⋅     ⋅   3.0
```

With two or more sparse arguments, only the results that are nonzero are stored:

```jldoctest sparsebroadcast
julia> B = sparse([1, 1, 3], [1, 3, 3], [1, 5, -3]);

julia> A .* B
3×3 SparseMatrixCSC{Int64, Int64} with 2 stored entries:
 1  ⋅   ⋅
 ⋅  ⋅   ⋅
 ⋅  ⋅  -9

julia> A .+ B
3×3 SparseMatrixCSC{Int64, Int64} with 3 stored entries:
 2   ⋅  5
 ⋅  -2  ⋅
 ⋅   ⋅  ⋅
```

The entry `A[3, 3] + B[3, 3]` cancels to zero and is dropped rather than stored, and so is the
stored zero `A[1, 2]`, because `A[1, 2] + B[1, 2]` computes to zero.

If `f(0, 0, ...)` is not zero, as for `cos.(A)`, `iszero.(A)` or `A ./ B` (where `0/0` is `NaN`), no
entry of the result is a structural zero, and a broadcast over sparse arrays alone returns the
`Array` that the same broadcast over dense arrays would:

```jldoctest sparsebroadcast
julia> iszero.(A)
3×3 BitMatrix:
 0  1  1
 1  0  1
 1  1  0
```

Whether such a result is sparse depends on a value, `f(0, 0, ...)`, so its type can be inferred only
when the compiler can evaluate `f` there. It does for the standard number types such as `Float64`,
`Int` and `ComplexF64`, but not for a `BigFloat`. A broadcast made only of `+`, `-` and `*` is
always sparse, and always inferable.

With a scalar among the arguments, as in `A .^ 0` or `2 .* A .+ 1`, or with a function that
carries a value of its own, such as the closure in `c = 1; (x -> x + c).(A)`, the result is still a
sparse array, but every entry is stored, including any that happen to compute to zero. Such a result needs
more memory than the equivalent `Array`, so convert to dense first when this is intended. Sums are
the exception: a broadcast made only of `+` and `-` that has a scalar among its arguments returns an
`Array`:

```jldoctest sparsebroadcast
julia> A .+ 2
3×3 Matrix{Int64}:
 3  2  2
 2  0  2
 2  2  5
```

`map` over sparse arrays always returns a sparse array, with every entry stored when `f(0, 0, ...)`
is not zero. It requires all arguments to have the same shape and throws a `DimensionMismatch`
otherwise, whereas `broadcast` expands singleton dimensions. A sparse vector
behaves as a one-column matrix, and combining it with a sparse matrix or with the adjoint or
transpose of a sparse vector gives a `SparseMatrixCSC`:

```jldoctest sparsebroadcast
julia> v = sparsevec([1, 3], [1, 2], 3);

julia> A .+ v
3×3 SparseMatrixCSC{Int64, Int64} with 7 stored entries:
 2   1  1
 ⋅  -2  ⋅
 2   2  5

julia> v .* v'
3×3 SparseMatrixCSC{Int64, Int64} with 4 stored entries:
 1  ⋅  2
 ⋅  ⋅  ⋅
 2  ⋅  4
```

Arguments that are not sparse arrays are handled as follows:

- Outside the sums above, scalars and `Ref`s are folded into the function before these rules
  are applied, so `2 .* A` is sparse and `2 .* A .+ 1` is sparse with every entry stored.
- A `Vector`, a `Matrix`, the adjoint or transpose of either, and a `Diagonal`,
  `Bidiagonal`, `Tridiagonal` or `SymTridiagonal` matrix are converted to sparse first, so
  the result is sparse, as in `A .* ones(3, 3)`. Sums are again the exception: a broadcast
  made only of `+` and `-` with a `Vector` or a `Matrix` (or a view, adjoint or transpose of
  one) among its arguments returns an `Array`, as `A + ones(3, 3)` does. So
  `A .+ ones(3, 3)` and `A .- v .+ ones(3)` are dense, while `2 .* A .+ ones(3, 3)` is
  sparse.
- A view of whole columns of a sparse matrix, such as `@view A[:, 2:3]` or `@view A[:, j]`,
  or of a range of a sparse vector, is copied to a sparse array in time proportional to its
  stored entries. Broadcasting over it is sparse too, even with no other sparse argument.
- Any other argument, such as a tuple, a range, a triangular or `Symmetric` wrapper, a view
  that selects some of the rows of a sparse matrix, or an array with more than two
  dimensions, makes the broadcast fall back to the generic implementation, which visits
  every element and returns an `Array`.

`map` accepts the same structured matrices alongside sparse matrices, and falls back to a
dense result otherwise.

```jldoctest sparsebroadcast
julia> A .* Diagonal([1, 2, 3])
3×3 SparseMatrixCSC{Int64, Int64} with 3 stored entries:
 1   ⋅  ⋅
 ⋅  -4  ⋅
 ⋅   ⋅  9

julia> typeof(A .* (1:3))
Matrix{Int64} (alias for Array{Int64, 2})
```

[`broadcast!`](@ref), [`map!`](@ref) and `.=` with a sparse destination overwrite its stored
pattern with that of the result, growing or shrinking its buffers as needed, so a preallocated
destination saves allocations only when its buffers are already large enough. The destination may
also be one of the arguments (they are copied first if they share memory with it), as in
`C .= C .+ C'`. One exception: `C .= 0` calls [`fill!`](@ref), which keeps `C`'s stored pattern and
sets the stored values to zero.

```jldoctest sparsebroadcast
julia> C = spzeros(Int, 3, 3);

julia> C .= A .* B;

julia> C
3×3 SparseMatrixCSC{Int64, Int64} with 2 stored entries:
 1  ⋅   ⋅
 ⋅  ⋅   ⋅
 ⋅  ⋅  -9
```

To apply a function to the stored values only, leaving the pattern untouched whatever the function
returns for zero, broadcast over [`nonzeros`](@ref) instead:

```jldoctest sparsebroadcast
julia> nonzeros(C) .= nonzeros(C) .+ 9;

julia> C
3×3 SparseMatrixCSC{Int64, Int64} with 2 stored entries:
 10  ⋅  ⋅
  ⋅  ⋅  ⋅
  ⋅  ⋅  0
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

The same three accessors work on a [`SparseVector`](@ref) and on a column view `@view A[:, j]`,
which are treated as a single column: `nzrange(x, 1)` covers all the stored entries, and
`rowvals(x)` holds their indices.

```jldoctest perftips
julia> x = sparsevec([2, 5], [1.5, 2.5], 6);

julia> [(rowvals(x)[k], nonzeros(x)[k]) for k in nzrange(x, 1)]
2-element Vector{Tuple{Int64, Float64}}:
 (2, 1.5)
 (5, 2.5)
```

A view of all rows and some columns, `@view A[:, 2:3]` or `@view A[:, [3, 1]]`, is a matrix
in its own right. Its `nonzeros` and `rowvals` are views of the parent's vectors, with `nnz`
elements each, holding only its own entries. Its `nzrange` indexes these views, and writing
to them mutates the parent. The accessors are O(1) for a range of columns. For an arbitrary
column subset, `nonzeros` and `rowvals` gather the positions of the entries once, and
`nzrange(V, j)` costs O(`j`). An `UpperTriangular`, `LowerTriangular`, `Symmetric` or
`Hermitian` wrapper of a sparse matrix has no `nonzeros`, `rowvals` or `nzrange`; convert it
with `sparse`, or work on its `parent`.

In contrast, scalar indexing `A[i, j]` has to do a binary search of column `j` for row `i`. A
loop over all `(i, j)` of an `m`-by-`n` matrix therefore performs `m * n` searches, however few
entries are stored, instead of visiting the `nnz(A)` stored entries once.

#### Build a matrix from its entries in one call

Storing a value at a position that has no stored entry yet moves all the later entries of `rowvals(A)`
and `nonzeros(A)` and updates the column pointers, so filling `spzeros(m, n)` one element at a time takes
time proportional to `nnz(A)` for each insertion. `sizehint!(A, n)` reserves room for
`n` stored entries, which avoids reallocating those vectors but not moving the entries. Instead, collect
the row indices, column indices and values in three vectors and call [`sparse`](@ref) once. Entries with
the same position are added together, or combined with the function passed as the last argument:

```jldoctest perftips
julia> I = [1, 1, 2]; J = [1, 1, 2]; V = [1.0, 2.0, 5.0];

julia> sparse(I, J, V)
2×2 SparseMatrixCSC{Float64, Int64} with 2 stored entries:
 3.0   ⋅
  ⋅   5.0

julia> sparse(I, J, V, 2, 2, max)
2×2 SparseMatrixCSC{Float64, Int64} with 2 stored entries:
 2.0   ⋅
  ⋅   5.0
```

#### Slice columns, not rows

`A[:, j]` copies one contiguous range of the stored entries. `A[i, :]` has to search every column for
row `i`, so its cost grows with the number of columns even when the row is empty. When an algorithm
works row by row, transpose the matrix once and work on the columns of the result. `permutedims(A)`
and `copy(transpose(A))` (equivalently `sparse(transpose(A))`) build the transposed matrix in time
proportional to `nnz(A)`.

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
stored entries has to be less than `typemax(Ti)`.

```jldoctest perftips
julia> A32 = SparseMatrixCSC{Float64,Int32}(A)
3×3 SparseMatrixCSC{Float64, Int32} with 4 stored entries:
 1.0   ⋅   2.0
  ⋅   3.0   ⋅
  ⋅    ⋅   4.0

julia> typeof(sparse(Int32[1, 2], Int32[1, 2], [1.0, 2.0])) == typeof(spzeros(Float64, Int32, 2, 2))
true

julia> spzeros(Float64, Int8, 200, 200)
ERROR: ArgumentError: number of rows (m = 200) does not fit in Ti = Int8
[...]
```

Operations between matrices with different index types promote to the wider one, so use one index type
consistently.

#### [Lazy `transpose` and `adjoint`](@id man-sparse-lazy-transpose)

`transpose(A)` and `A'` do not copy; they return `Transpose` and `Adjoint` wrappers around
`A`. What happens to a wrapper depends on the function:

- Products with dense vectors and matrices (including `mul!`) and with sparse vectors, `\`,
  `==`, and `A'[i, :]`, which is a column slice of `A`, work on the wrapper without
  materializing the transpose.
- Products with another sparse matrix, broadcasting (which includes `+` and `-`), `map`,
  concatenation, `kron` and `findnz` first copy the wrapper into a new `SparseMatrixCSC`.
  The copy takes time proportional to `nnz(A)`, on every call.
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

#### Keep results sparse and free of stored zeros

A result stays sparse only if the operation maps zeros to zeros. `A .+ 1` and `exp.(A)` return a
`Matrix`; apply such functions to `nonzeros(A)` instead when only the stored entries are meant. Assigning zero to a stored entry,
and cancellation in a matrix product, leave explicitly stored zeros behind. They are harmless for correctness but
are visited by every kernel. [`dropzeros!`](@ref) removes them, [`droptol!`](@ref) removes entries of small
magnitude, and [`fkeep!`](@ref) keeps the entries for which a predicate of `(i, j, v)` is true, all in place.

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
  `x == zero(x)`, so a type that defines `zero` of a value and `==` need not define it.
- **The arithmetic of the operation itself**: `+` and `*` for a product, `isless` for
  `maximum`, `minimum` and `sort`, and so on. As with dense arrays, `A'` and
  `transpose(A)` apply `adjoint` and `transpose` to the elements. `sum` adds the `n`
  entries that are not stored in one step, as `zero(T) * n`, so it also needs the product
  of an element with an integer.

The sparse kernels skip the entries that are not stored, so they assume that `zero(T)`
behaves as a zero: `x + zero(T) == x`, and a product with `zero(T)` is zero. `map` and
broadcasting do not assume this of an arbitrary function, and evaluate it on `zero(T)`, as
described in [Broadcasting and `map`](@ref man-sparse-broadcast).

A type that defines `zero`, `+` and `*` is enough for most of the package, and every
`Number` qualifies. Some limits apply to the other element types:

- Multiplying two sparse matrices needs only `*` and `+`. It is the one arithmetic
  operation that works without `zero(T)`.
- An element type whose zero depends on the value, such as `Matrix{Float64}`, whose size
  is not part of its type, has no `zero(T)`. A sparse matrix of them can be stored and
  multiplied by another, but not indexed at an entry that is not stored.
- `c * A`, `A * c` and `A / c` are defined for a scalar `c` that is a `Number`. For any
  other element type, use `map` or broadcasting.
- The [sparse factorizations](@ref stdlib-sparse-linalg) `lu`, `qr`, `cholesky` and `ldlt`
  call SuiteSparse, which works in real and complex floating point. A matrix of integers
  or rationals is converted to `Float64` first; a matrix of another element type, such
  as `BigFloat`, is rejected.

### [Sparse arrays with a fixed pattern](@id man-sparse-fixed)

!!! warning "Experimental"
    `fixed`, `FixedSparseCSC` and `FixedSparseVector` are experimental and not exported.

Writing into a `SparseMatrixCSC` can change which entries it stores. `W .= M .- γ .* J`
drops an entry of `W` wherever the difference is zero, and `W[i, j] = v` inserts one. Code
that has derived something from the pattern of `W`, such as a symbolic factorization, the
positions of its entries in `nonzeros(W)`, or a copy of its index arrays on another
device, is then silently out of date.

A fixed sparse array rules that out. It is a **destination whose sparsity pattern never
changes**: the stored values can be overwritten, and no entry can be added or removed.
[`SparseArrays.fixed`](@ref) makes one, with the arguments of [`sparse`](@ref) or from a
sparse array, which it copies.

```jldoctest fixedpattern
julia> J = sparse([1.0 2 0; 0 3 0; 4 0 5]); M = sparse(1.0I, 3, 3);

julia> nnz(M .- J)   # the entry at (1, 1) cancels and is not stored
4

julia> W = SparseArrays.fixed(M + J)
3×3 SparseArrays.FixedSparseCSC{Float64, Int64} with 5 stored entries:
 2.0  2.0   ⋅
  ⋅   4.0   ⋅
 4.0   ⋅   6.0

julia> W .= M .- J
3×3 SparseArrays.FixedSparseCSC{Float64, Int64} with 5 stored entries:
  0.0  -2.0    ⋅
   ⋅   -2.0    ⋅
 -4.0    ⋅   -4.0
```

Three rules describe everything a fixed array does.

**A write lands inside the pattern or throws.** `W .= ...`, `map!`, `copyto!`, `mul!`,
`fill!`, `rmul!`, `lmul!`, `rdiv!`, `W[i, j] = v` and assignment through a column view store
a value at each stored entry, a zero included, and leave the pattern alone. What is written
need not fill the pattern: every value that `setindex!` would store must fall on a stored
entry, and the stored entries it leaves out become zero, so `W .= M` and `W .= 2 .* M` both
work. That holds for a result written into a separate fixed destination as well, as by
`transpose!(W, A)`, `permute!(W, A, p, q)` or `circshift!(W, A, s)`. A write that needs an
entry where the pattern has none throws an `ArgumentError`, and the kernels of this package
check for that before they change any value. That covers a block assignment such as
`W[:, j] = x`, `W[I, J] = B` or `w[inds] = x`. Where this package has no method for a sparse
destination, Julia's generic one runs and writes one entry at a time: a broadcast into a
`view` of `W`, a dense source as in `W .= D` or `copyto!(W, D)`, `axpy!(a, A, W)`, or `mul!`
with a dense factor. Such a
write cannot change the pattern either, but it stops at the first entry outside it, with
the entries before it already written. Writing through `nonzeros(W)` is always allowed.

```jldoctest fixedpattern
julia> W[1, 3] = 1.0
ERROR: ArgumentError: cannot store a new entry at (1, 3) in a FixedSparseCSC, its sparsity pattern is read-only
[...]

julia> W .= J .+ 1
ERROR: ArgumentError: cannot store a nonzero f(0) into a FixedSparseCSC, its sparsity pattern is read-only
[...]
```

**A function that exists to change the pattern throws.** [`dropzeros!`](@ref),
[`droptol!`](@ref), [`SparseArrays.fkeep!`](@ref), `tril!`, `triu!`,
[`SparseArrays.dropstored!`](@ref), `sort!`, `permute!`, `invpermute!`, `reverse!`,
`circshift!`, `SparseArrays.swaprows!` and `SparseArrays.swapcols!` refuse a fixed array
as the argument they change. None of them does something else that happens to fit the
pattern. Their out-of-place forms (`dropzeros`, `tril`, `sort`, ...) work, because of the
next rule. Applied to a `view` of a fixed array, those that have no sparse method for a
view are generic writes, under the previous rule.

```jldoctest fixedpattern
julia> dropzeros!(W)
ERROR: ArgumentError: cannot `dropzeros!` a FixedSparseCSC: it would change the sparsity pattern, which is read-only; apply it to a writable copy, `SparseMatrixCSC(A)`
[...]
```

**Reading a fixed array gives an ordinary result.** As an operand, a fixed array behaves
as the `SparseMatrixCSC` or `SparseVector` with the same entries, and what comes back is
what that array would give: a `SparseMatrixCSC`, a `SparseVector`, a dense array or a
number. Being fixed is a property of one array, not of its values, so it does not carry
over to indexing, arithmetic, broadcasting, concatenation or `similar`. A result is fixed
only where you ask for it:

- `copy(W)` is fixed and shares the pattern of `W`, so that `rowvals(copy(W)) === rowvals(W)`,
  and so is the result of a generic function of another package that returns `copy(W)`;
- a function that returns its argument, such as `float(W)` here or `+W`, returns `W`;
- `SparseArrays.fixed(A)` fixes any other result;
- `SparseMatrixCSC(W)` and `SparseVector(x)` give a writable copy.

```jldoctest fixedpattern
julia> 2W
3×3 SparseMatrixCSC{Float64, Int64} with 5 stored entries:
  0.0  -4.0    ⋅
   ⋅   -4.0    ⋅
 -8.0    ⋅   -8.0

julia> W[:, 1]
3-element SparseVector{Float64, Int64} with 2 stored entries:
  [1]  =  0.0
  [3]  =  -4.0

julia> typeof(similar(W))
SparseMatrixCSC{Float64, Int64}

julia> V = copy(W); rowvals(V) === rowvals(W)
true
```

Note that an out-of-place `map` or broadcast over two or more arrays drops the entries it
computes as zero, as it does for other sparse arrays (over a single array it keeps the
pattern of its argument); write into a fixed destination to keep them.

The factorizations accept a fixed matrix, which is what makes reusing a symbolic
factorization safe: see [Reusing a factorization](@ref man-sparse-refactorize).

## [Sparse Linear Algebra](@id stdlib-sparse-linalg)

The functions of LinearAlgebra have methods for sparse arrays. Products, `dot`, `kron` and
the norms are implemented in this package. The factorizations, and the solves that use
them, call [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse).

### [Products and norms](@id man-sparse-products)

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
- `cond(A, 1)` and `cond(A, Inf)` estimate the norm of the inverse from an
  [`lu`](@ref SparseArrays.UMFPACK.lu) factorization. The 2-norm condition number is not
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

julia> kron(A, sparse(I, 2, 2)) isa SparseMatrixCSC
true

julia> norm(A, 1), opnorm(A, 1), opnorm(A, Inf)
(12.0, 5.0, 5.0)
```

### [Solving linear systems](@id man-sparse-solving)

Sparse factorizations call SuiteSparse:

| Function | Returns | Library |
|:---------|:--------|:--------|
| [`cholesky`](@ref SparseArrays.CHOLMOD.cholesky), [`ldlt`](@ref SparseArrays.CHOLMOD.ldlt) | [`CHOLMOD.Factor`](@ref SparseArrays.CHOLMOD.Factor) | CHOLMOD |
| [`lu`](@ref SparseArrays.UMFPACK.lu) | [`UMFPACK.UmfpackLU`](@ref SparseArrays.UMFPACK.UmfpackLU) | UMFPACK |
| [`qr`](@ref SparseArrays.SPQR.qr) | [`SPQR.QRSparse`](@ref SparseArrays.SPQR.QRSparse) | SPQR |
| [`lq`](@ref SparseArrays.SPQR.lq) | [`SPQR.AdjointQRSparse`](@ref SparseArrays.SPQR.AdjointQRSparse), the adjoint of `qr(A')` | SPQR |

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

#### [Reusing a factorization](@id man-sparse-refactorize)

To solve several systems with one matrix, factorize once. `F \ B` takes a vector or a
matrix of right-hand sides, and `ldiv!(x, F, b)` writes into `x`.

`ldiv!` allocates scratch space on each call. To avoid that in a loop, create a workspace
once and pass it with the `workspace` keyword:
[`UMFPACK.UmfpackWS(F)`](@ref SparseArrays.UMFPACK.UmfpackWS) for `lu`,
[`CHOLMOD.CholmodWS(F)`](@ref SparseArrays.CHOLMOD.CholmodWS) for `cholesky` and `ldlt`,
and [`SPQR.SpqrWS(F)`](@ref SparseArrays.SPQR.SpqrWS) for `qr`. This works with `F`, `F'`
and `transpose(F)`, and in `ldiv!(F, b)`. A workspace grows as needed and can be reused,
but not by two calls at once; `similar(ws)` makes another one. The right-hand side of
`ldiv!` may be complex when the factorization is real.

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

The symbolic analysis in `F` is valid only for the pattern it was computed from, and an
in-place update of a `SparseMatrixCSC` can change that pattern: `A .= M .- c .* B` drops
an entry wherever the difference is zero. A [fixed-pattern matrix](@ref man-sparse-fixed) cannot
lose or gain an entry, so it can be refilled and refactorized any number of times:

```jldoctest
julia> A = SparseArrays.fixed(sparse([2.0 1 0; 0 3 1; 1 0 4])); b = [1.0, 2.0, 3.0];

julia> F = lu(A); x = F \ b;

julia> A .= 2 .* A; nnz(A)
6

julia> lu!(F, A);

julia> F \ b ≈ x / 2
true
```

#### Extracting the factors

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

#### Failures

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

### Multithreading and thread safety

A solve (`\` or `ldiv!`, with `F`, `F'` or `transpose(F)`) only reads the factorization and
takes no lock, and neither do `det`, `logabsdet`, `logdet` and the factor properties. Any
number of tasks can therefore use one factorization at the same time:

```julia
using LinearAlgebra, SparseArrays

F = lu(A)                       # or cholesky(A), or qr(A)
X = similar(B)
Threads.@threads for j in axes(B, 2)
    ldiv!(view(X, :, j), F, view(B, :, j))
end
```

Each `ldiv!` above allocates its scratch space. To avoid that, give every task a workspace
of its own, made with `similar(ws)` from one that exists.

Solves with a supernodal `cholesky` factor, which CHOLMOD chooses for larger matrices, do
not currently speed up when run from several threads, and can slow down: they make many
small BLAS calls, which contend with each other in OpenBLAS
([SparseArrays.jl#1044](https://github.com/JuliaSparse/SparseArrays.jl/issues/1044)). Solves
with `ldlt`, `lu` and `qr` factors are not affected.

A `QRSparse` is never modified once `qr` has returned, and has no lock. An `UmfpackLU` and a
`CHOLMOD.Factor` can be refactorized in place, and each has an internal lock, which the
calls that change or copy it hold for the whole call:

| Type | Calls that take the lock of `F` |
|:-----|:--------------------------------|
| `UMFPACK.UmfpackLU` | `lu!`, `copy`, `show_umf_ctrl` and `show_umf_info` |
| `SPQR.QRSparse` | none |
| `CHOLMOD.Factor` | [`cholesky!`](@ref SparseArrays.CHOLMOD.cholesky!), `ldlt!`, the in-place low-rank updates, `copy`, and what copies the factor internally: `sparse(F)`, `nnz(F)`, and a solve that has to convert the factor to the element type of the right-hand side |

The lock keeps those calls apart from each other, not from a call that reads: a
factorization must not be refactorized or updated while another task solves with it or reads
its factors. A task that does so needs its own `copy(F)`, which is independent of `F` and
duplicates the memory of the factors. `deepcopy(F)` is equivalent, and `copy` of `F'` or
`transpose(F)` wraps `copy(F)`.

UMFPACK reports statistics of each solve, such as the refinement steps it took. A solve
writes them to the `info` of the [`UMFPACK.UmfpackWS`](@ref SparseArrays.UMFPACK.UmfpackWS)
it was given, not to `F.info`, which describes the factorization only.

CHOLMOD and SPQR keep their parameters, statistics and error state in a `cholmod_common`
structure. SparseArrays creates one lazily for each Julia task (and index type) and keeps
it in task-local storage, so factorizing different matrices from different tasks needs no
coordination.

### Tuning the factorizations

The defaults suit most problems. These keywords change them.

#### `cholesky` and `ldlt`

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

#### `lu`

- `q`: an initial column ordering, which UMFPACK may still refine. `F.q` is the final one.
- `control`: UMFPACK's `Control` array. Get the defaults with
  `SparseArrays.UMFPACK.get_umfpack_control(Tv, Ti)` for the element and index types of
  `A`, and index it with the one-based constants `SparseArrays.UMFPACK.JL_UMFPACK_*`,
  such as `JL_UMFPACK_PIVOT_TOLERANCE` or `JL_UMFPACK_ORDERING`. The UMFPACK user guide
  describes each entry. The factorization keeps a copy, which later solves and
  [`lu!`](@ref SparseArrays.UMFPACK.lu!) use.

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

#### `qr`

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

### Using a different SuiteSparse build

By default the solvers use the SuiteSparse libraries bundled with Julia. To use another
build, such as one with GPU support, point SparseArrays at a directory holding all of
`libsuitesparseconfig`, `libamd`, `libcamd`, `libcolamd`, `libccolamd`, `libcholmod`,
`libspqr` and `libumfpack`, with the bundled file names and the same major SuiteSparse
version. Either call
[`SparseArrays.LibSuiteSparse.set_libdir!(dir)`](@ref SparseArrays.LibSuiteSparse.set_libdir!)
before the first solver call, or set `JULIA_SUITESPARSE_LIBDIR` before starting Julia; `set_libdir!` wins.
Packages that call `SuiteSparse_jll` directly are not affected.

```@meta
DocTestSetup = nothing
```

## [SparseArrays API](@id stdlib-sparse-arrays)

```@docs
SparseArrays
SparseArrays.AbstractSparseArray
SparseArrays.AbstractSparseVector
SparseArrays.AbstractSparseMatrix
SparseArrays.AbstractSparseMatrixCSC
SparseArrays.AbstractCompressedVector
SparseArrays.SparseVector
SparseArrays.SparseMatrixCSC
SparseArrays.sparse
SparseArrays.sparse!
SparseArrays.sparsevec
Base.similar(::SparseArrays.AbstractSparseMatrixCSC, ::Type)
SparseArrays.issparse
SparseArrays.nnz
SparseArrays.findnz
SparseArrays.iternz
SparseArrays.spzeros
SparseArrays.spzeros!
SparseArrays.spdiagm
SparseArrays.sparse_hcat
SparseArrays.sparse_vcat
SparseArrays.sparse_hvcat
SparseArrays.blockdiag
SparseArrays.sprand
SparseArrays.sprandn
SparseArrays.getcolptr
SparseArrays.getrowval
SparseArrays.getnzval
SparseArrays.indtype
SparseArrays.nonzeros
SparseArrays.rowvals
SparseArrays.nzrange
SparseArrays.droptol!
SparseArrays.dropzeros!
SparseArrays.dropzeros
SparseArrays.dropstored!
SparseArrays.fkeep!
SparseArrays.permute
Base.permute!(::SparseArrays.AbstractSparseMatrixCSC{Tv,Ti}, ::SparseArrays.AbstractSparseMatrixCSC{Tv,Ti}, ::AbstractVector{<:Integer}, ::AbstractVector{<:Integer}) where {Tv,Ti}
SparseArrays.halfperm!
SparseArrays.ftranspose!
SparseArrays.transpose!(::SparseArrays.AbstractSparseMatrixCSC{Tv,Ti}, ::SparseArrays.AbstractSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
SparseArrays.adjoint!(::SparseArrays.AbstractSparseMatrixCSC{Tv,Ti}, ::SparseArrays.AbstractSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
Base.sort!(::SparseArrays.AbstractSparseMatrixCSC)
Base.sort(::SparseArrays.AbstractSparseMatrixCSC)
Base.sort!(::Union{SparseArrays.AbstractCompressedVector, SparseArrays.SparseColumnView})
SparseArrays.fixed
SparseArrays.FixedSparseCSC
SparseArrays.FixedSparseVector
SparseArrays.allowscalar
```

## [Sparse Linear Algebra API](@id stdlib-sparse-linalg-api)

### Norms

```@docs
SparseArrays.opnorm(::SparseArrays.AbstractSparseMatrixCSC, ::Real)
```

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
SparseArrays.CHOLMOD.lowrankupdowndate!
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
SparseArrays.SPQR.AdjointQRSparse
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
```

## Internal API

The helpers below are not part of the API: they are unexported and may change or disappear
in any release. They are documented, incompletely, as a companion to the public API.

```@docs
SparseArrays.ReadOnly
SparseArrays.ColumnIndices
SparseArrays.iswrsparse
SparseArrays.depth
SparseArrays.sparse_with_lmul
SparseArrays.rowcheck_index
SparseArrays.mergeinds!
SparseArrays.move_fixed
SparseArrays._unsafe_unfix
SparseArrays.@RCI
SparseArrays._densify!
SparseArrays.HigherOrderFns._map_zeropres!
SparseArrays.HigherOrderFns._map_notzeropres!
SparseArrays._spsetz_setindex!
SparseArrays._spsetnz_setindex!
SparseArrays.unchecked_noalias_permute!
SparseArrays.unchecked_aliasing_permute!
SparseArrays._computecolptrs_permute!
SparseArrays._checkargs_sourcecompatperms_permute!
SparseArrays._checkargs_permutationsvalid_permute!
SparseArrays._checkargs_sourcecompatdest_permute!
SparseArrays._checkargs_sourcecompatworkmat_permute!
SparseArrays._checkargs_sourcecompatworkcolptr_permute!
SparseArrays._computecolptrs_halfperm!
SparseArrays._distributevals_halfperm!
SparseArrays.LibSuiteSparse.init_suitesparse
```

## Noteworthy External Sparse Packages

Several other Julia packages provide sparse matrix implementations that should be mentioned:

1. [SuiteSparseGraphBLAS.jl](https://github.com/JuliaSparse/SuiteSparseGraphBLAS.jl) is a wrapper over the fast, multithreaded SuiteSparse:GraphBLAS C library.

2. [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) exposes the [CUSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html) library for GPU sparse matrix operations.

3. [SparseMatricesCSR.jl](https://github.com/gridap/SparseMatricesCSR.jl) provides a Julia native implementation of the Compressed Sparse Rows (CSR) format.

4. [MKLSparse.jl](https://github.com/JuliaSparse/MKLSparse.jl) accelerates SparseArrays sparse-dense matrix operations using Intel's MKL library.

5. [SparseArrayKit.jl](https://github.com/Jutho/SparseArrayKit.jl) available for multidimensional sparse arrays.

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
