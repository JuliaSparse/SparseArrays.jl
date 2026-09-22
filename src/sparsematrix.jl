# This file is a part of Julia. License is MIT: https://julialang.org/license

# Compressed sparse columns data structure
# No assumptions about stored zeros in the data structure
# Assumes that row values in rowval for each column are sorted
#      issorted(rowval[colptr[i]:(colptr[i+1]-1)]) == true
# Assumes that 1 <= colptr[i] <= colptr[i+1] for i in 1..n
# Assumes that nnz <= length(rowval) < typemax(Ti)
# Assumes that 0   <= length(nzval) < typemax(Ti)

"""
    SparseMatrixCSC{Tv,Ti<:Integer} <: AbstractSparseMatrixCSC{Tv,Ti}

Matrix type for storing sparse matrices in the
[Compressed Sparse Column](@ref man-csc) format. The standard way
of constructing SparseMatrixCSC is through the [`sparse`](@ref) function.
See also [`spzeros`](@ref), [`spdiagm`](@ref) and [`sprand`](@ref).

    SparseMatrixCSC(m::Integer, n::Integer, colptr::Vector, rowval::Vector, nzval::Vector)

Wrap existing CSC arrays as an `m × n` matrix without copying them. Column `j` holds the
entries `nzval[k]` in rows `rowval[k]` for `k in colptr[j]:(colptr[j+1] - 1)`.

Only `colptr` and the lengths of `rowval` and `nzval` are checked. The row indices are
not inspected, and must be strictly increasing within each column and lie in `1:m`.
The kernels in this package rely on that. A matrix with unsorted, repeated or
out-of-range row indices is constructed silently and then gives inconsistent results.
To construct a matrix from repeated indices, use [`sparse`](@ref), which adds up their
values.
See [the manual](@ref man-csc) for how to repair such arrays.

# Examples
```jldoctest
julia> SparseMatrixCSC(3, 2, [1, 3, 4], [1, 3, 2], [1.0, 2.0, 3.0])
3×2 SparseMatrixCSC{Float64, Int64} with 3 stored entries:
 1.0   ⋅
  ⋅   3.0
 2.0   ⋅
```
"""
struct SparseMatrixCSC{Tv,Ti<:Integer} <: AbstractSparseMatrixCSC{Tv,Ti}
    m::Int                  # Number of rows
    n::Int                  # Number of columns
    colptr::Vector{Ti}      # Column i is in colptr[i]:(colptr[i+1]-1)
    rowval::Vector{Ti}      # Row indices of stored values
    nzval::Vector{Tv}       # Stored values, typically nonzeros

    function SparseMatrixCSC{Tv,Ti}(m::Integer, n::Integer, colptr::Vector{Ti},
                            rowval::Vector{Ti}, nzval::Vector{Tv}) where {Tv,Ti<:Integer}
        sparse_check_Ti(m, n, Ti)
        _goodbuffers(Int(m), Int(n), colptr, rowval, nzval) ||
            throw(ArgumentError("Invalid buffers for SparseMatrixCSC construction n=$n, colptr=$(summary(colptr)), rowval=$(summary(rowval)), nzval=$(summary(nzval))"))
        new(Int(m), Int(n), colptr, rowval, nzval)
    end
end
function SparseMatrixCSC(m::Integer, n::Integer, colptr::Vector, rowval::Vector, nzval::Vector)
    Tv = eltype(nzval)
    Ti = promote_type(eltype(colptr), eltype(rowval))
    sparse_check_Ti(m, n, Ti)
    sparse_check(n, colptr, rowval, nzval)
    # silently shorten rowval and nzval to usable index positions.
    maxlen = abs(widemul(m, n))
    isbitstype(Ti) && (maxlen = min(maxlen, typemax(Ti) - 1))
    length(rowval) > maxlen && resize!(rowval, maxlen)
    length(nzval) > maxlen && resize!(nzval, maxlen)
    SparseMatrixCSC{Tv,Ti}(m, n, colptr, rowval, nzval)
end

SparseMatrixCSC(m, n, colptr::ReadOnly, rowval::ReadOnly, nzval::Vector) =
    SparseMatrixCSC(m, n, copy(parent(colptr)), copy(parent(rowval)), nzval)

"""
    SparseMatrixCSC{Tv,Ti}(::UndefInitializer, m::Integer, n::Integer)
    SparseMatrixCSC{Tv,Ti}(::UndefInitializer, (m,n)::NTuple{2,Integer})

Creates an empty sparse matrix with element type `Tv` and integer type `Ti` of size `m × n`.
"""
SparseMatrixCSC{Tv,Ti}(::UndefInitializer, m::Integer, n::Integer) where {Tv, Ti} = spzeros(Tv, Ti, m, n)
SparseMatrixCSC{Tv,Ti}(::UndefInitializer, mn::NTuple{2,Integer}) where {Tv, Ti} = spzeros(Tv, Ti, mn...)

"""
    SparseMatrixCSC(x::FixedSparseCSC)

Get a writable copy of x. See `_unsafe_unfix(x)`
"""
SparseMatrixCSC(x::FixedSparseCSC) = SparseMatrixCSC(size(x, 1), size(x, 2),
    copy(parent(getcolptr(x))),
    copy(parent(rowvals(x))),
    copy(nonzeros(x)))

function sparse_check_Ti(m::Integer, n::Integer, Ti::Type)
        @noinline throwTi(str, lbl, k) =
            throw(ArgumentError("$str ($lbl = $k) does not fit in Ti = $(Ti)"))
        0 ≤ m && (!isbitstype(Ti) || m ≤ typemax(Ti)) || throwTi("number of rows", "m", m)
        0 ≤ n && (!isbitstype(Ti) || n ≤ typemax(Ti)) || throwTi("number of columns", "n", n)
end

function sparse_check(n::Integer, colptr::Vector{Ti}, rowval, nzval) where Ti
    # String interpolation is a performance bottleneck when it's part of the same function,
    # ensure we only do it once committed to the error.
    throwstart(ckp) = throw(ArgumentError("$ckp == colptr[1] != 1"))
    throwmonotonic(ckp, ck, k) = throw(ArgumentError("$ckp == colptr[$(k-1)] > colptr[$k] == $ck"))

    sparse_check_length("colptr", colptr, n+1, String) # don't check upper bound
    ckp = colptr[1]
    ckp == Ti(1) || throwstart(ckp)
    @inbounds for k = 2:n+1
        ck = colptr[k]
        ckp <= ck || throwmonotonic(ckp, ck, k)
        ckp = ck
    end
    sparse_check_length("rowval", rowval, ckp-1, Ti)
    sparse_check_length("nzval", nzval, 0, Ti) # we allow empty nzval !!!
end
function sparse_check_length(rowstr, rowval, minlen, Ti)
    throwmin(len, minlen, rowstr) = throw(ArgumentError("$len == length($rowstr) < $minlen"))
    throwmax(len, max, rowstr) = throw(ArgumentError("$len == length($rowstr) >= $max"))

    len = length(rowval)
    len >= minlen || throwmin(len, minlen, rowstr)
    !isbitstype(Ti) || len < typemax(Ti) || throwmax(len, typemax(Ti), rowstr)
end

size(S::SparseMatrixCSC) = (getfield(S, :m), getfield(S, :n))
size(S::FixedSparseCSC) = (getfield(S, :m), getfield(S, :n))

_goodbuffers(S::AbstractSparseMatrixCSC) = _goodbuffers(size(S)..., getcolptr(S), getrowval(S), nonzeros(S))
_checkbuffers(S::AbstractSparseMatrixCSC) = (@assert _goodbuffers(S); S)
_checkbuffers(S::Union{Adjoint, Transpose}) = (_checkbuffers(parent(S)); S)

function _goodbuffers(m, n, colptr, rowval, nzval)
    (length(colptr) == n + 1 && colptr[end] - 1 == length(rowval) == length(nzval))
    # stronger check for debugging purposes
    # && all(issorted(@view rowval[colptr[i]:colptr[i+1]-1]) for i=1:n)
end

getcolptr(S::SparseMatrixCSC) = getfield(S, :colptr)
getcolptr(S::FixedSparseCSC) = getfield(S, :colptr)
getcolptr(S::SparseMatrixCSCView) = view(getcolptr(parent(S)), first(parentindices(S)[2]):(last(parentindices(S)[2]) + 1))
getcolptr(S::SparseMatrixCSCColumnSubset) = error("getcolptr not well-defined for $(typeof(S))")

# The kernels address the storage behind `S` through `getcolptr`, `getrowval`, `getnzval`
# and `getnzrange`: for a view or a wrapper these are the parent's vectors and the
# positions of column `j` of `S` within them, so `getnzval(S)[getnzrange(S, j)]` holds
# column `j` whatever `S` is. The public `nonzeros`, `rowvals` and `nzrange` instead
# describe the entries of `S` itself, which for a view or wrapper are a subset of the
# parent's; for a `SparseMatrixCSC` the two agree.
Base.@propagate_inbounds getnzrange(S::AbstractSparseMatrixCSC, col::Integer) = nzrange(S, col)
Base.@propagate_inbounds getnzrange(S::SparseMatrixCSCColumnSubset, col::Integer) = getnzrange(parent(S), parentindices(S)[2][col])
getnzrange(S::UpperTriangular{<:Any,<:SparseMatrixCSCOrView}, i::Integer) = nzrangeup(S.data, i)
getnzrange(S::LowerTriangular{<:Any,<:SparseMatrixCSCOrView}, i::Integer) = nzrangelo(S.data, i)
"""
    getrowval(A)

Return the vector of row indices of the structural nonzeros of sparse array `A`.
For a [`SparseMatrixCSC`](@ref) this is the `rowval` field; for a
[`SparseVector`](@ref) it is the `nzind` field. Any modifications to the returned
vector will mutate `A` as well. Providing access to how the row indices are
stored internally can be useful in conjunction with iterating over structural
nonzero values. See also [`getnzval`](@ref) and [`nzrange`](@ref).

For a `SparseMatrixCSC` or a `SparseVector`, `getrowval` is equivalent to
[`rowvals`](@ref). For a column view or a triangular wrapper of a sparse matrix it
returns the parent's vector, of which `rowvals(A)` is the part belonging to `A`.

# Examples
```jldoctest
julia> A = sparse(2I, 3, 3)
3×3 SparseMatrixCSC{Int64, Int64} with 3 stored entries:
 2  ⋅  ⋅
 ⋅  2  ⋅
 ⋅  ⋅  2

julia> getrowval(A)
3-element Vector{Int64}:
 1
 2
 3

julia> getrowval(sparsevec([2, 5], [3.0, 4.0]))
2-element Vector{Int64}:
 2
 5
```
"""
getrowval(S::AbstractSparseMatrixCSC) = rowvals(S)
getrowval(S::SparseMatrixCSCColumnSubset) = rowvals(parent(S))
getrowval(S::UpperTriangular{<:Any,<:SparseMatrixCSCOrView}) = getrowval(S.data)
getrowval(S::LowerTriangular{<:Any,<:SparseMatrixCSCOrView}) = getrowval(S.data)

"""
    getnzval(A)

Return the vector of structural nonzero values of sparse array `A`, i.e. the `nzval`
field of a [`SparseMatrixCSC`](@ref) or [`SparseVector`](@ref). This includes zeros
that are explicitly stored in the sparse array. The returned vector points directly
to the internal nonzero storage of `A`, and any modifications to the returned vector
will mutate `A` as well. See also [`getrowval`](@ref) and [`nzrange`](@ref).

For a `SparseMatrixCSC` or a `SparseVector`, `getnzval` is equivalent to
[`nonzeros`](@ref). For a column view or a triangular wrapper of a sparse matrix it
returns the parent's vector, of which `nonzeros(A)` is the part belonging to `A`.

# Examples
```jldoctest
julia> A = sparse(2I, 3, 3)
3×3 SparseMatrixCSC{Int64, Int64} with 3 stored entries:
 2  ⋅  ⋅
 ⋅  2  ⋅
 ⋅  ⋅  2

julia> getnzval(A)
3-element Vector{Int64}:
 2
 2
 2

julia> getnzval(sparsevec([2, 5], [3.0, 4.0]))
2-element Vector{Float64}:
 3.0
 4.0
```
"""
getnzval( S::AbstractSparseMatrixCSC) = nonzeros(S)
getnzval( S::SparseMatrixCSCColumnSubset) = nonzeros(parent(S))
getnzval( S::UpperTriangular{<:Any,<:SparseMatrixCSCOrView}) = getnzval(S.data)
getnzval( S::LowerTriangular{<:Any,<:SparseMatrixCSCOrView}) = getnzval(S.data)
nzvalview(S::AbstractSparseMatrixCSC) = view(nonzeros(S), 1:nnz(S))
nzvalview(S::SparseMatrixCSCColumnSubset) = nonzeros(S)
# where the stored entries of `S` sit in `getnzval(S)`: a contiguous range for a column range
_storedinds(S::AbstractSparseMatrixCSC) = 1:nnz(S)
function _storedinds(S::SparseMatrixCSCView)
    cols = parentindices(S)[2]
    isempty(cols) && return 1:0   # an empty range need not lie within the parent's columns
    colptr = getcolptr(parent(S))
    return Int(colptr[first(cols)]):Int(colptr[last(cols)+1]) - 1
end
function _storedinds(S::SparseMatrixCSCColumnSubset)
    inds = Int[]
    for col in axes(S, 2)
        append!(inds, getnzrange(S, col))
    end
    return inds
end
widelength(S::SparseMatrixCSCColumnSubset) = prod(Int64.(size(S)))

"""
    nnz(A)

Returns the number of stored (filled) elements in a sparse array.

# Examples
```jldoctest
julia> A = sparse(2I, 3, 3)
3×3 SparseMatrixCSC{Int64, Int64} with 3 stored entries:
 2  ⋅  ⋅
 ⋅  2  ⋅
 ⋅  ⋅  2

julia> nnz(A)
3
```
"""
nnz(S::AbstractSparseMatrixCSC) = @inbounds Int(getcolptr(S)[size(S, 2) + 1]) - 1
nnz(S::ReshapedArray{<:Any,1,<:AbstractSparseMatrixCSC}) = nnz(parent(S))
nnz(S::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}) = nnz(parent(S))
nnz(S::UpperTriangular{<:Any,<:SparseMatrixCSCOrView}) = nnz1(S)
nnz(S::LowerTriangular{<:Any,<:SparseMatrixCSCOrView}) = nnz1(S)
nnz(S::SparseMatrixCSCColumnSubset) = nnz1(S)
nnz(S::SparseMatrixCSCView) = length(_storedinds(S))
nnz1(S) = @inbounds sum(length.(getnzrange.(Ref(S), axes(S, 2))))

function Base._simple_count(pred, S::SparseMatrixCSCOrColumnSubset, init::T) where T
    init + T(count(pred, nzvalview(S)) + pred(zero(eltype(S)))*(prod(size(S)) - nnz(S)))
end

"""
    nonzeros(A)

Return a vector of the structural nonzero values in sparse array `A`. This
includes zeros that are explicitly stored in the sparse array. The returned
vector points directly to the internal nonzero storage of `A`, and any
modifications to the returned vector will mutate `A` as well. See
[`rowvals`](@ref) and [`nzrange`](@ref).

For a view of all rows and some columns of a sparse matrix, `nonzeros` returns a view
of the parent's storage holding the entries of those columns only, so `length(nonzeros(A))`
is `nnz(A)` and writes to it mutate the parent. For a view of a range of columns the
view is contiguous; for other column subsets it gathers the positions of the stored
entries, which takes time proportional to `nnz(A)`.

# Examples
```jldoctest
julia> A = sparse(2I, 3, 3)
3×3 SparseMatrixCSC{Int64, Int64} with 3 stored entries:
 2  ⋅  ⋅
 ⋅  2  ⋅
 ⋅  ⋅  2

julia> nonzeros(A)
3-element Vector{Int64}:
 2
 2
 2

julia> nonzeros(view(A, :, 2:3))
2-element view(::Vector{Int64}, 2:3) with eltype Int64:
 2
 2
```
"""
nonzeros(S::SparseMatrixCSC) = getfield(S, :nzval)
nonzeros(S::FixedSparseCSC) = getfield(S, :nzval)
nonzeros(S::SparseMatrixCSCColumnSubset) = view(getnzval(S), _storedinds(S))
nonzeros(S::UpperTriangular{<:Any,<:SparseMatrixCSCOrView}) = getnzval(S.data)
nonzeros(S::LowerTriangular{<:Any,<:SparseMatrixCSCOrView}) = getnzval(S.data)

"""
    rowvals(A)

Return a vector of the row indices of sparse array `A`. Any modifications to the returned
vector will mutate `A` as well. Providing access to how the row indices are
stored internally can be useful in conjunction with iterating over structural
nonzero values. See also [`nonzeros`](@ref) and [`nzrange`](@ref).

# Examples
```jldoctest
julia> A = sparse(2I, 3, 3)
3×3 SparseMatrixCSC{Int64, Int64} with 3 stored entries:
 2  ⋅  ⋅
 ⋅  2  ⋅
 ⋅  ⋅  2

julia> rowvals(A)
3-element Vector{Int64}:
 1
 2
 3
```

For a view of all rows and some columns of a sparse matrix, `rowvals` returns a view of
the parent's vector holding the row indices of those columns' entries only, matching
[`nonzeros`](@ref). For a sparse vector or a column view of a sparse matrix, `rowvals`
returns the indices of the stored entries:

```jldoctest
julia> rowvals(sparsevec([2, 5], [1.5, 2.5], 6))
2-element Vector{Int64}:
 2
 5
```
"""
rowvals(S::SparseMatrixCSC) = getfield(S, :rowval)
rowvals(S::FixedSparseCSC) = getfield(S, :rowval)
rowvals(S::SparseMatrixCSCColumnSubset) = view(getrowval(S), _storedinds(S))
rowvals(S::UpperTriangular{<:Any,<:SparseMatrixCSCOrView}) = getrowval(S.data)
rowvals(S::LowerTriangular{<:Any,<:SparseMatrixCSCOrView}) = getrowval(S.data)

"""
    nzrange(A, col::Integer)

Return the range of indices to the structural nonzero values of column `col`
of sparse array `A`. In conjunction with [`nonzeros`](@ref) and
[`rowvals`](@ref), this allows for convenient iterating over a sparse matrix :

    A = sparse(I,J,V)
    rows = rowvals(A)
    vals = nonzeros(A)
    m, n = size(A)
    for j = 1:n
       for i in nzrange(A, j)
          row = rows[i]
          val = vals[i]
          # perform sparse wizardry...
       end
    end

The same loop works on a view of all rows and some columns of `A`, whose `nzrange`
indexes the `rowvals` and `nonzeros` of the view. For a view of a range of columns each
call is O(1); for other column subsets `nzrange(A, col)` sums the lengths of the preceding
columns, so it is O(`col`).

!!! warning
    Adding or removing nonzero elements to the matrix may invalidate the `nzrange`, one should not mutate the matrix while iterating.
"""
Base.@propagate_inbounds nzrange(S::AbstractSparseMatrixCSC, col::Integer) = getcolptr(S)[col]:(getcolptr(S)[col+1]-1)
Base.@propagate_inbounds function nzrange(S::SparseMatrixCSCView, col::Integer)
    r = getnzrange(S, col)
    off = first(_storedinds(S)) - 1
    return (first(r) - off):(last(r) - off)
end
function nzrange(S::SparseMatrixCSCColumnSubset, col::Integer)
    @boundscheck checkbounds(axes(S, 2), col)
    off = 0
    @inbounds for k in 1:col-1
        off += length(getnzrange(S, k))
    end
    return (off + 1):(off + length(@inbounds getnzrange(S, col)))
end
nzrange(S::UpperTriangular{<:Any,<:SparseMatrixCSCOrView}, i::Integer) = nzrangeup(S.data, i)
nzrange(S::LowerTriangular{<:Any,<:SparseMatrixCSCOrView}, i::Integer) = nzrangelo(S.data, i)
# positions in `getnzval(A)` of the stored entries of column `i` up to (and including
# if excl=false) the diagonal
function nzrangeup(A, i, excl=false)
    r = getnzrange(A, i); r1 = r.start; r2 = r.stop
    rv = getrowval(A)
    @inbounds r2 < r1 || rv[r2] <= i - excl ? r : r1:(searchsortedlast(view(rv, r1:r2), i - excl) + r1-1)
end
# the same from the diagonal (included if excl=false) to the end
function nzrangelo(A, i, excl=false)
    r = getnzrange(A, i); r1 = r.start; r2 = r.stop
    rv = getrowval(A)
    @inbounds r2 < r1 || rv[r1] >= i + excl ? r : (searchsortedfirst(view(rv, r1:r2), i + excl) + r1-1):r2
end
# how the stored triangle of a symmetric/Hermitian wrapper is walked: its range within a
# column, and the maps applied to diagonal and to mirrored off-diagonal entries
_symherm_ops(t::AbstractChar) = (_isuppercase(t) ? nzrangeup : nzrangelo,
    _uppercase(t) == 'S' ? identity : real, _uppercase(t) == 'S' ? transpose : adjoint)
_symherm_ops(A::HermOrSym) = _symherm_ops(LinearAlgebra.wrapper_char(A))

indtype(S::SparseMatrixCSCColumnSubset{<:Any,Ti}) where {Ti} = Ti

function Base.isstored(A::AbstractSparseMatrixCSC, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    rows = rowvals(A)
    @inbounds for istored in nzrange(A, j) # could do binary search if the row indices are sorted?
        i == rows[istored] && return true
    end
    return false
end

function Base.isstored(A::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    cols = rowvals(parent(A))
    @inbounds for istored in nzrange(parent(A), i)
        j == cols[istored] && return true
    end
    return false
end

Base.replace_in_print_matrix(A::SparseMatrixCSCMaybeAdjOrTrans, i::Integer, j::Integer, s::AbstractString) =
    Base.isstored(A, i, j) ? s : Base.replace_with_centered_mark(s)

function Base.array_summary(io::IO, S::SparseMatrixCSCMaybeAdjOrTrans, dims::Tuple{Vararg{Base.OneTo}})
    _checkbuffers(S)

    xnnz = nnz(S)
    m, n = size(S)
    print(io, m, "×", n, " ", typeof(S), " with ", xnnz, " stored ",
              xnnz == 1 ? "entry" : "entries")
    nothing
end

# called by `show(io, MIME("text/plain"), ::SparseMatrixCSCMaybeAdjOrTrans)`
function Base.print_array(io::IO, S::SparseMatrixCSCMaybeAdjOrTrans)
    if max(size(S)...) < 16
        Base.print_matrix(io, S)
    else
        _show_with_braille_patterns(io, S)
    end
end

"""
    ColumnIndices(S::AbstractSparseMatrixCSC)

Return the column indices of the stored values in `S`.
This is an internal type that is used in displaying sparse matrices,
and is not a part of the public interface.
"""
struct ColumnIndices{Ti,S<:AbstractSparseMatrixCSC{<:Any,Ti}} <: AbstractVector{Ti}
    arr :: S
end

size(C::ColumnIndices) = (nnz(C.arr),)
# returns the column index of the n-th non-zero value from the column pointer
@inline function getindex(C::ColumnIndices, i::Int)
    @boundscheck checkbounds(C, i)
    colptr = getcolptr(C.arr)
    ind = searchsortedlast(colptr, i)
    eltype(C)(ind)
end

# always show matrices as `sparse(I, J, K)`
function Base.show(io::IO, _S::SparseMatrixCSCMaybeAdjOrTrans)
    _checkbuffers(_S)
    # can't use `findnz`, because that expects all values not to be #undef
    S = _S isa Adjoint || _S isa Transpose ? parent(_S) : _S
    I = rowvals(S)
    K = nonzeros(S)
    m, n = size(S)
    if _S isa Adjoint
        print(io, "adjoint(")
    elseif _S isa Transpose
        print(io, "transpose(")
    end
    print(io, "sparse(", I, ", ")
    show(io, ColumnIndices(S))
    print(io, ", ", K, ", ", m, ", ", n, ")")
    if _S isa Adjoint || _S isa Transpose
        print(io, ")")
    end
end

const brailleBlocks = UInt16['⠁', '⠂', '⠄', '⡀', '⠈', '⠐', '⠠', '⢀']
function _show_with_braille_patterns(io::IO, S::SparseMatrixCSCMaybeAdjOrTrans)
    m, n = size(S)
    (m == 0 || n == 0) && return show(io, MIME("text/plain"), S)

    # The maximal number of characters we allow to display the matrix
    local maxHeight::Int, maxWidth::Int
    maxHeight = displaysize(io)[1] - 4 # -4 from [Prompt, header, newline after elements, new prompt]
    maxWidth = displaysize(io)[2] ÷ 2

    # In the process of generating the braille pattern to display the nonzero
    # structure of `S`, we need to be able to scale the matrix `S` to a
    # smaller matrix with the same aspect ratio as `S`, but fits on the
    # available screen space. The size of that smaller matrix is stored
    # in the variables `scaleHeight` and `scaleWidth`. If no scaling is needed,
    # we can use the size `m × n` of `S` directly.
    # We determine if scaling is needed and set the scaling factors
    # `scaleHeight` and `scaleWidth` accordingly. Note that each available
    # character can contain up to 4 braille dots in its height (⡇) and up to
    # 2 braille dots in its width (⠉).
    if get(io, :limit, true) && (m > 4maxHeight || n > 2maxWidth)
        s = min(2maxWidth / n, 4maxHeight / m)
        scaleHeight = floor(Int, s * m)
        scaleWidth = floor(Int, s * n)
    else
        scaleHeight = m
        scaleWidth = n
    end

    # Make sure that the matrix size is big enough to be able to display all
    # the corner border characters
    if scaleHeight < 8
        scaleHeight = 8
    end
    if scaleWidth < 4
        scaleWidth = 4
    end

    # `brailleGrid` is used to store the needed braille characters for
    # the matrix `S`. Each row of the braille pattern to print is stored
    # in a column of `brailleGrid`.
    brailleGrid = fill(UInt16(10240), (scaleWidth - 1) ÷ 2 + 4, (scaleHeight - 1) ÷ 4 + 1)
    brailleGrid[1,:] .= '⎢'
    brailleGrid[end-1,:] .= '⎥'
    brailleGrid[1,1] = '⎡'
    brailleGrid[1,end] = '⎣'
    brailleGrid[end-1,1] = '⎤'
    brailleGrid[end-1,end] = '⎦'
    brailleGrid[end, :] .= '\n'

    rvals = rowvals(parent(S))
    rowscale = max(1, scaleHeight - 1) / max(1, m - 1)
    colscale = max(1, scaleWidth - 1) / max(1, n - 1)
    if isa(S, AbstractSparseMatrixCSC)
        @inbounds for j in axes(S,2)
            # Scale the column index `j` to the best matching column index
            # of a matrix of size `scaleHeight × scaleWidth`
            sj = round(Int, (j - 1) * colscale + 1)
            for x in nzrange(S, j)
                # Scale the row index `i` to the best matching row index
                # of a matrix of size `scaleHeight × scaleWidth`
                si = round(Int, (rvals[x] - 1) * rowscale + 1)

                # Given the index pair `(si, sj)` of the scaled matrix,
                # calculate the corresponding triple `(k, l, p)` such that the
                # element at `(si, sj)` can be found at position `(k, l)` in the
                # braille grid `brailleGrid` and corresponds to the 1-dot braille
                # character `brailleBlocks[p]`
                k = (sj - 1) ÷ 2 + 2
                l = (si - 1) ÷ 4 + 1
                p = ((sj - 1) % 2) * 4 + ((si - 1) % 4 + 1)

                brailleGrid[k, l] |= brailleBlocks[p]
            end
        end
    else
        # If `S` is a adjoint or transpose of a sparse matrix we invert the
        # roles of the indices `i` and `j`
        @inbounds for i = 1:m
            si = round(Int, (i - 1) * rowscale + 1)
            for x in nzrange(parent(S), i)
                sj = round(Int, (rvals[x] - 1) * colscale + 1)
                k = (sj - 1) ÷ 2 + 2
                l = (si - 1) ÷ 4 + 1
                p = ((sj - 1) % 2) * 4 + ((si - 1) % 4 + 1)
                brailleGrid[k, l] |= brailleBlocks[p]
            end
        end
    end
    foreach(c -> print(io, Char(c)), @view brailleGrid[1:end-1])
end

# The dense-operand methods in LinearAlgebra accept the thin shapes, so the sparse operands
# get them too. See SparseQMatOperand.
for QT in (:LinAlgLeftQs, :LQPackedQ), Q in (QT, :(AdjointQ{<:Any,<:$QT}))
    @eval begin
        (*)(Q::$Q, B::SparseQMatOperand) = Q * Matrix(B)
        (*)(Q::$Q, b::SparseQVecOperand) = Q * Vector(b)
        (*)(A::SparseQMatOperand, Q::$Q) = Matrix(A) * Q
        (*)(a::SparseQVecOperand, Q::$Q) = Vector(a) * Q
    end
end

## Reshape

function sparse_compute_reshaped_colptr_and_rowval!(colptrS::Vector{Ti}, rowvalS::Vector{Ti},
                                                   mS::Int, nS::Int, colptrA::Vector{Ti},
                                                   rowvalA::Vector{Ti}, mA::Int, nA::Int) where Ti
    lrowvalA = length(rowvalA)
    maxrowvalA = (lrowvalA > 0) ? maximum(rowvalA) : zero(Ti)
    ((length(colptrA) == (nA+1)) && (maximum(colptrA) <= (lrowvalA+1)) && (maxrowvalA <= mA)) || throw(BoundsError())

    colptrS[1] = 1
    colA = 1
    colS = 1
    ptr = 1

    @inbounds while colA <= nA
        offsetA = (colA - 1) * mA
        while ptr <= colptrA[colA+1]-1
            rowA = rowvalA[ptr]
            i = offsetA + rowA - 1
            colSn = div(i, mS) + 1
            rowS = mod(i, mS) + 1
            while colS < colSn
                colptrS[colS+1] = ptr
                colS += 1
            end
            rowvalS[ptr] = rowS
            ptr += 1
        end
        colA += 1
    end
    @inbounds while colS <= nS
        colptrS[colS+1] = ptr
        colS += 1
    end
end

function copy(ra::ReshapedArray{<:Any,2,<:AbstractSparseMatrixCSC})
    mS,nS = size(ra)
    a = parent(ra)
    mA,nA = size(a)
    numnz = nnz(a)
    colptr = similar(getcolptr(a), nS+1)
    rowval = similar(rowvals(a))
    nzval = copy(nonzeros(a))

    sparse_compute_reshaped_colptr_and_rowval!(colptr, rowval, mS, nS, getcolptr(a), rowvals(a), mA, nA)

    return SparseMatrixCSC(mS, nS, colptr, rowval, nzval)
end

## Alias detection and prevention
using Base: dataids, unaliascopy
Base.dataids(S::AbstractSparseMatrixCSC) = _is_fixed(S) ? dataids(nonzeros(S)) : (dataids(getcolptr(S))..., dataids(rowvals(S))..., dataids(nonzeros(S))...)
Base.unaliascopy(S::AbstractSparseMatrixCSC) = typeof(S)(size(S, 1), size(S, 2),
    _is_fixed(S) ? getcolptr(S) : unaliascopy(getcolptr(S)),
    _is_fixed(S) ? rowvals(S) : unaliascopy(rowvals(S)),
    unaliascopy(nonzeros(S)))

## Constructors

copy(S::AbstractSparseMatrixCSC) =
    SparseMatrixCSC(size(S, 1), size(S, 2), copy(getcolptr(S)), copy(rowvals(S)), copy(nonzeros(S)))
copy(S::FixedSparseCSC) =
    FixedSparseCSC(size(S, 1), size(S, 2), getcolptr(S), rowvals(S), copy(nonzeros(S)))
function copyto!(A::AbstractSparseMatrixCSC, B::AbstractSparseMatrixCSC)
    _is_fixed(A) && return _copyto_fixed!(A, B)
    # If the two matrices have the same length then all the
    # elements in A will be overwritten.
    if widelength(A) == widelength(B)
        resize!(nonzeros(A), length(nonzeros(B)))
        resize!(rowvals(A), length(rowvals(B)))
        if size(A) == size(B)
            # Simple case: we can simply copy the internal fields of B to A.
            copyto!(getcolptr(A), getcolptr(B))
            copyto!(rowvals(A), rowvals(B))
        else
            # This is like a "reshape B into A".
            sparse_compute_reshaped_colptr_and_rowval!(getcolptr(A), rowvals(A), size(A, 1), size(A, 2), getcolptr(B), rowvals(B), size(B, 1), size(B, 2))
        end
    else
        widelength(A) >= widelength(B) || throw(BoundsError())
        lB = widelength(B)
        nnzA = nnz(A)
        nnzB = nnz(B)
        # Up to which col, row, and ptr in rowval/nzval will A be overwritten?
        lastmodcolA = Int(div(lB - 1, size(A, 1))) + 1
        lastmodrowA = Int(mod(lB - 1, size(A, 1))) + 1
        lastmodptrA = getcolptr(A)[lastmodcolA]
        while lastmodptrA < getcolptr(A)[lastmodcolA+1] && rowvals(A)[lastmodptrA] <= lastmodrowA
            lastmodptrA += 1
        end
        lastmodptrA -= 1
        if lastmodptrA >= nnzB
            # A will have fewer non-zero elements; unmodified elements are kept at the end.
            deleteat!(rowvals(A), nnzB+1:lastmodptrA)
            deleteat!(nonzeros(A), nnzB+1:lastmodptrA)
        else
            # A will have more non-zero elements; unmodified elements are kept at the end.
            resize!(rowvals(A), nnzB + nnzA - lastmodptrA)
            resize!(nonzeros(A), nnzB + nnzA - lastmodptrA)
            copyto!(rowvals(A), nnzB+1, rowvals(A), lastmodptrA+1, nnzA-lastmodptrA)
            copyto!(nonzeros(A), nnzB+1, nonzeros(A), lastmodptrA+1, nnzA-lastmodptrA)
        end
        # Adjust colptr accordingly.
        @inbounds for i in 2:length(getcolptr(A))
            getcolptr(A)[i] += nnzB - lastmodptrA
        end
        sparse_compute_reshaped_colptr_and_rowval!(getcolptr(A), rowvals(A), size(A, 1), lastmodcolA-1, getcolptr(B), rowvals(B), size(B, 1), size(B, 2))
    end
    copyto!(nonzeros(A), nonzeros(B))
    return _checkbuffers(A)
end

function copyto!(dest::AbstractMatrix, Rdest::CartesianIndices{2},
                 src::AbstractSparseMatrixCSC{T}, Rsrc::CartesianIndices{2}) where {T}
    isempty(Rdest) && return dest
    if size(Rdest) != size(Rsrc)
        throw(ArgumentError("source and destination must have same size (got $(size(Rsrc)) and $(size(Rdest)))"))
    end
    checkbounds(dest, Rdest)
    checkbounds(src, Rsrc)
    src′ = Base.unalias(dest, src)
    for I in Rdest
        @inbounds dest[I] = zero(T) # implicitly convert to eltype(dest), throw if not possible
    end
    rows, cols = Rsrc.indices
    lin = LinearIndices(Base.IdentityUnitRange.(Rsrc.indices))
    @inbounds for col in cols, ptr in nzrange(src′, col)
        row = rowvals(src′)[ptr]
        if row in rows
            val = nonzeros(src′)[ptr]
            I = Rdest[lin[row, col]]
            dest[I] = val
        end
    end
    return dest
end

# Faster version for non-abstract Array and SparseMatrixCSC
function Base.copyto!(A::Array{T}, S::SparseMatrixCSC{<:Number}) where {T<:Number}
    _checkbuffers(S)
    isempty(S) && return A
    length(A) < length(S) && throw(BoundsError())

    # Zero elements that are also in S, don't change rest of A
    @inbounds for i in 1:length(S)
        A[i] = zero(T)
    end
    # Copy the structural nonzeros from S to A using
    # the linear indices (to work when size(A)!=size(S))
    num_rows = size(S,1)
    rowval = getrowval(S)
    nzval = getnzval(S)
    linear_index_col0 = 0   # Linear index before column (linear index = linear_index_col0 + row)
    @inbounds for col in axes(S, 2)
        for i in nzrange(S, col)
            row = rowval[i]
            val = nzval[i]
            A[linear_index_col0+row] = val
        end
        linear_index_col0 += num_rows
    end
    return A
end

## similar
#
# parent method for similar that preserves stored-entry structure (for when new and old dims match)
function _sparsesimilar(S::AbstractSparseMatrixCSC, ::Type{TvNew}, ::Type{TiNew}) where {TvNew,TiNew}
    newcolptr = copyto!(similar(getcolptr(S), TiNew), getcolptr(S))
    newrowval = copyto!(similar(rowvals(S), TiNew), rowvals(S))
    return SparseMatrixCSC(size(S, 1), size(S, 2), newcolptr, newrowval, similar(nonzeros(S), TvNew))
end
# parent methods for similar that preserves only storage space (for when new dims are 2-d)
_sparsesimilar(S::AbstractSparseMatrixCSC, ::Type{TvNew}, ::Type{TiNew}, dims::Dims{2}) where {TvNew,TiNew} =
    sizehint!(spzeros(TvNew, TiNew, dims...), length(nonzeros(S)))
# parent method for similar that allocates an empty sparse vector (for when new dims are 1-d)
_sparsesimilar(S::AbstractSparseMatrixCSC, ::Type{TvNew}, ::Type{TiNew}, dims::Dims{1}) where {TvNew,TiNew} =
    SparseVector(dims..., similar(rowvals(S), TiNew, 0), similar(nonzeros(S), TvNew, 0))

# The following methods hook into the AbstractArray similar hierarchy. The first method
# covers similar(A[, Tv]) calls, which preserve stored-entry structure, and the latter
# methods cover similar(A[, Tv], shape...) calls, which partially preserve
# storage space when the shape calls for a two-dimensional result.

"""
    similar(A::AbstractSparseMatrixCSC{Tv,Ti}, [::Type{TvNew}, ::Type{TiNew}, m::Integer, n::Integer]) where {Tv,Ti}

Create an uninitialized mutable array with the given element type,
index type, and size, based upon the given source
`SparseMatrixCSC`. The new sparse matrix maintains the structure of
the original sparse matrix, except in the case where dimensions of the
output matrix are different from the output.

The output matrix has zeros in the same locations as the input, but
uninitialized values for the nonzero locations. A `FixedSparseCSC` input keeps
its fixed pattern only in the structure-preserving form; the forms taking a
shape return a `SparseMatrixCSC`.
"""
similar(S::AbstractSparseMatrixCSC{<:Any,Ti}, ::Type{TvNew}) where {Ti,TvNew} =
    @if_move_fixed S _sparsesimilar(S, TvNew, Ti)

# a new shape carries no pattern over, so the result is never fixed
similar(S::AbstractSparseMatrixCSC{<:Any,Ti}, ::Type{TvNew}, dims::Union{Dims{1},Dims{2}}) where {Ti,TvNew} =
    _sparsesimilar(S, TvNew, Ti, dims)

# The following methods cover similar(A, Tv, Ti[, shape...]) calls, which specify the
# result's index type in addition to its entry type, and aren't covered by the hooks above.
# The calls without shape again preserve stored-entry structure, whereas those with shape
# preserve storage space when the shape calls for a two-dimensional result.
similar(S::AbstractSparseMatrixCSC, ::Type{TvNew}, ::Type{TiNew}) where{TvNew,TiNew} =
    _sparsesimilar(S, TvNew, TiNew)
similar(S::AbstractSparseMatrixCSC, ::Type{TvNew}, ::Type{TiNew}, dims::Union{Dims{1},Dims{2}}) where {TvNew,TiNew} =
    _sparsesimilar(S, TvNew, TiNew, dims)
similar(S::AbstractSparseMatrixCSC, ::Type{TvNew}, ::Type{TiNew}, m::Integer) where {TvNew,TiNew} =
    _sparsesimilar(S, TvNew, TiNew, (m,))
similar(S::AbstractSparseMatrixCSC, ::Type{TvNew}, ::Type{TiNew}, m::Integer, n::Integer) where {TvNew,TiNew} =
    _sparsesimilar(S, TvNew, TiNew, (m, n))

function Base.sizehint!(S::SparseMatrixCSC, n::Integer)
    nhint = min(n, widelength(S))
    sizehint!(getrowval(S), nhint)
    sizehint!(nonzeros(S),  nhint)
    return S
end

## Transposition and permutation methods

"""
    halfperm!(X::AbstractSparseMatrixCSC{Tv,Ti}, A::AbstractSparseMatrixCSC{TvA,Ti},
              q::AbstractVector{<:Integer}, f::Function = identity) where {Tv,TvA,Ti}

Column-permute and transpose `A`, simultaneously applying `f` to each entry of `A`, storing
the result `(f(A)Q)^T` (`map(f, transpose(A[:,q]))`) in `X`.

Element type `Tv` of `X` must match `f(::TvA)`, where `TvA` is the element type of `A`.
`X`'s dimensions must match those of `transpose(A)` (`size(X, 1) == size(A, 2)` and
`size(X, 2) == size(A, 1)`), and `X` must have enough storage to accommodate all allocated
entries in `A` (`length(rowvals(X)) >= nnz(A)` and `length(nonzeros(X)) >= nnz(A)`).
Column-permutation `q`'s length must match `A`'s column count (`length(q) == size(A, 2)`).

This method is the parent of several methods performing transposition and permutation
operations on [`SparseMatrixCSC`](@ref)s. As this method performs no argument checking,
prefer the safer child methods (`[c]transpose[!]`, `permute[!]`) to direct use.

This method implements the `HALFPERM` algorithm described in F. Gustavson, "Two fast
algorithms for sparse matrices: multiplication and permuted transposition," ACM TOMS 4(3),
250-269 (1978). The algorithm runs in `O(size(A, 1), size(A, 2), nnz(A))` time and requires no space
beyond that passed in.
"""
function halfperm!(X::AbstractSparseMatrixCSC{Tv,Ti}, A::AbstractSparseMatrixCSC{TvA,Ti},
        q::AbstractVector{<:Integer}, f::F = identity) where {Tv,TvA,Ti,F<:Function}
    _computecolptrs_halfperm!(X, A)
    _distributevals_halfperm!(X, A, q, f)
    return X
end
"""
Helper method for `halfperm!`. Computes `transpose(A[:,q])`'s column pointers, storing them
shifted one position forward in `getcolptr(X)`; `_distributevals_halfperm!` fixes this shift.
"""
function _computecolptrs_halfperm!(X::AbstractSparseMatrixCSC{Tv,Ti}, A::AbstractSparseMatrixCSC{TvA,Ti}) where {Tv,TvA,Ti}
    # Compute `transpose(A[:,q])`'s column counts. Store shifted forward one position in getcolptr(X).
    fill!(getcolptr(X), 0)
    @inbounds for k in 1:nnz(A)
        getcolptr(X)[rowvals(A)[k] + 1] += 1
    end
    # Compute `transpose(A[:,q])`'s column pointers. Store shifted forward one position in getcolptr(X).
    getcolptr(X)[1] = 1
    countsum = 1
    @inbounds for k in 2:(size(A, 1) + 1)
        overwritten = getcolptr(X)[k]
        getcolptr(X)[k] = countsum
        countsum += overwritten
    end
end
"""
Helper method for `halfperm!`. With `transpose(A[:,q])`'s column pointers shifted one
position forward in `getcolptr(X)`, computes `map(f, transpose(A[:,q]))` by appropriately
distributing `rowvals(A)` and `f`-transformed `nonzeros(A)` into `rowvals(X)` and `nonzeros(X)`
respectively. Simultaneously fixes the one-position-forward shift in `getcolptr(X)`.
"""
@noinline function _distributevals_halfperm!(X::AbstractSparseMatrixCSC{Tv,Ti},
        A::AbstractSparseMatrixCSC{TvA,Ti}, q::AbstractVector{<:Integer}, f::F) where {Tv,TvA,Ti,F<:Function}
    resize!(nonzeros(X), nnz(A))
    resize!(rowvals(X), nnz(A))
    @inbounds for Xi in axes(A,2)
        Aj = q[Xi]
        for Ak in nzrange(A, Aj)
            Ai = rowvals(A)[Ak]
            Xk = getcolptr(X)[Ai + 1]
            rowvals(X)[Xk] = Xi
            nonzeros(X)[Xk] = f(nonzeros(A)[Ak])
            getcolptr(X)[Ai + 1] += 1
        end
    end
    return # kill potential type instability
end
"""
    ftranspose!(X::AbstractSparseMatrixCSC{Tv,Ti}, A::AbstractSparseMatrixCSC{Tv,Ti}, f::Function) where {Tv,Ti}

Transpose `A` and store it in `X` while applying the function `f` to the non-zero elements.
Does not remove the zeros created by `f`. `size(X)` must be equal to `size(transpose(A))`.
No additional memory is allocated other than resizing the rowval and nzval of `X`, if needed.

See `halfperm!`
"""
function ftranspose!(X::AbstractSparseMatrixCSC{Tv,Ti}, A::AbstractSparseMatrixCSC{Tv,Ti}, f::F) where {Tv,Ti,F<:Function}
    # Check compatibility of source argument A and destination argument X
    if size(X, 2) != size(A, 1)
        throw(DimensionMismatch(string("destination argument `X`'s column count, ",
            "`size(X, 2) (= $(size(X, 2)))`, must match source argument `A`'s row count, `size(A, 1) (= $(size(A, 1)))`")))
    elseif size(X, 1) != size(A, 2)
        throw(DimensionMismatch(string("destination argument `X`'s row count, ",
            "`size(X, 1) (= $(size(X, 1)))`, must match source argument `A`'s column count, `size(A, 2) (= $(size(A, 2)))`")))
    # halfperm! overwrites X's buffers while reading A's. With nnz(A) == 0 only the colptr is
    # written, and the empty rowval/nzval buffers would falsely alias (they share one `Memory`)
    elseif nnz(A) > 0 ? Base.mightalias(X, A) : Base.mightalias(getcolptr(X), getcolptr(A))
        throw(ArgumentError("destination argument `X` must not share memory with source argument `A`"))
    end
    halfperm!(X, A, axes(A,2), f)
end

"""
    transpose!(X::AbstractSparseMatrixCSC{Tv,Ti}, A::AbstractSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}

Transpose the matrix `A` and stores it in the matrix `X`.
`size(X)` must be equal to `size(transpose(A))`.
No additional memory is allocated other than resizing the rowval and nzval of `X`, if needed.

See `halfperm!`
"""
transpose!(X::AbstractSparseMatrixCSC{Tv,Ti}, A::AbstractSparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = ftranspose!(X, A, identity)

"""
    adjoint!(X::AbstractSparseMatrixCSC{Tv,Ti}, A::AbstractSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}

Transpose the matrix `A` and stores the adjoint of the elements in the matrix `X`.
`size(X)` must be equal to `size(transpose(A))`.
No additional memory is allocated other than resizing the rowval and nzval of `X`, if needed.

See `halfperm!`
"""
adjoint!(X::AbstractSparseMatrixCSC{Tv,Ti}, A::AbstractSparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = ftranspose!(X, A, conj)

# manually specifying eltype allows to avoid calling return_type of f on TvA
function ftranspose(A::AbstractSparseMatrixCSC{TvA,Ti}, f::Function, eltype::Type{Tv} = TvA) where {Tv,TvA,Ti}
    X = SparseMatrixCSC(size(A, 2), size(A, 1),
                        ones(Ti, size(A, 1)+1),
                        Vector{Ti}(undef, 0),
                        Vector{Tv}(undef, 0))
    sizehint!(X, nnz(A))
    return @if_move_fixed A halfperm!(X, A, axes(A,2), f)
end

adjoint(A::AbstractSparseMatrixCSC) = Adjoint(A)
transpose(A::AbstractSparseMatrixCSC) = Transpose(A)
_adjtrans_fun(::Adjoint) = x -> adjoint(copy(x))
_adjtrans_fun(::Transpose) = x -> transpose(copy(x))
Base.copy(A::Adjoint{<:Any,<:AbstractSparseMatrixCSC}) =
    ftranspose(parent(A), _adjtrans_fun(A), eltype(A))
Base.copy(A::Transpose{<:Any,<:AbstractSparseMatrixCSC}) =
    ftranspose(parent(A), _adjtrans_fun(A), eltype(A))
function Base.permutedims(A::AbstractSparseMatrixCSC, (a,b))
    (a, b) == (2, 1) && return ftranspose(A, identity)
    (a, b) == (1, 2) && return copy(A)
    throw(ArgumentError("no valid permutation of dimensions"))
end

"""
    unchecked_noalias_permute!(X::AbstractSparseMatrixCSC{Tv,Ti},
        A::AbstractSparseMatrixCSC{Tv,Ti}, p::AbstractVector{<:Integer},
        q::AbstractVector{<:Integer}, C::AbstractSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}

See [`permute!`](@ref) for basic usage. Parent of `permute[!]`
methods operating on `SparseMatrixCSC`s that assume none of `X`, `A`, and `C` alias each
other. As this method performs no argument checking, prefer the safer child methods
(`permute[!]`) to direct use.

This method consists of two major steps: (1) Column-permute (`Q`,`I[:,q]`) and transpose `A`
to generate intermediate result `(AQ)^T` (`transpose(A[:,q])`) in `C`. (2) Column-permute
(`P^T`, I[:,p]) and transpose intermediate result `(AQ)^T` to generate result
`((AQ)^T P^T)^T = PAQ` (`A[p,q]`) in `X`.

The first step is a call to `halfperm!`, and the second is a variant on `halfperm!` that
avoids an unnecessary length-`nnz(A)` array-sweep and associated recomputation of column
pointers. See [`halfperm!`](:func:SparseArrays.halfperm!) for additional algorithmic
information.

See also `unchecked_aliasing_permute!`.
"""
function unchecked_noalias_permute!(X::AbstractSparseMatrixCSC{Tv,Ti},
        A::AbstractSparseMatrixCSC{Tv,Ti}, p::AbstractVector{<:Integer},
        q::AbstractVector{<:Integer}, C::AbstractSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    halfperm!(C, A, q)
    _computecolptrs_permute!(X, A, q, getcolptr(X))
    _distributevals_halfperm!(X, C, p, identity)
    return X
end
"""
    unchecked_aliasing_permute!(A::AbstractSparseMatrixCSC{Tv,Ti},
        p::AbstractVector{<:Integer}, q::AbstractVector{<:Integer},
        C::AbstractSparseMatrixCSC{Tv,Ti}, workcolptr::Vector{Ti}) where {Tv,Ti}

See [`permute!`](@ref) for basic usage. Parent of `permute!`
methods operating on [`SparseMatrixCSC`](@ref)s where the source and destination matrices
are the same. See `unchecked_noalias_permute!`
for additional information; these methods are identical but for this method's requirement of
the additional `workcolptr`, `length(workcolptr) >= size(A, 2) + 1`, which enables efficient
handling of the source-destination aliasing.
"""
function unchecked_aliasing_permute!(A::AbstractSparseMatrixCSC{Tv,Ti},
        p::AbstractVector{<:Integer}, q::AbstractVector{<:Integer},
        C::AbstractSparseMatrixCSC{Tv,Ti}, workcolptr::Vector{Ti}) where {Tv,Ti}
    halfperm!(C, A, q)
    _computecolptrs_permute!(A, A, q, workcolptr)
    _distributevals_halfperm!(A, C, p, identity)
    return A
end
"""
Helper method for `unchecked_noalias_permute!` and `unchecked_aliasing_permute!`.
Computes `PAQ`'s column pointers, storing them shifted one position forward in `getcolptr(X)`;
`_distributevals_halfperm!` fixes this shift. Saves some work relative to
`_computecolptrs_halfperm!` as described in `uncheckednoalias_permute!`'s documentation.
"""
function _computecolptrs_permute!(X::AbstractSparseMatrixCSC{Tv,Ti},
        A::AbstractSparseMatrixCSC{Tv,Ti}, q::AbstractVector{<:Integer}, workcolptr::Vector{Ti}) where {Tv,Ti}
    # Compute `A[p,q]`'s column counts. Store shifted forward one position in workcolptr.
    @inbounds for k in axes(A,2)
        workcolptr[k+1] = getcolptr(A)[q[k] + 1] - getcolptr(A)[q[k]]
    end
    # Compute `A[p,q]`'s column pointers. Store shifted forward one position in getcolptr(X).
    getcolptr(X)[1] = 1
    countsum = 1
    @inbounds for k in 2:(size(X, 2) + 1)
        overwritten = workcolptr[k]
        getcolptr(X)[k] = countsum
        countsum += overwritten
    end
end

"""
Helper method for `permute` and `permute!` methods operating on `SparseMatrixCSC`s.
Checks compatibility of source argument `A`, row-permutation argument `p`, and
column-permutation argument `q`.
"""
function _checkargs_sourcecompatperms_permute!(A::AbstractSparseMatrixCSC,
        p::AbstractVector{<:Integer}, q::AbstractVector{<:Integer})
    require_one_based_indexing(p, q)
    if length(q) != size(A, 2)
         throw(DimensionMismatch(string("the length of column-permutation argument `q`, ",
             "`length(q) (= $(length(q)))`, must match source argument `A`'s column ",
             "count, `size(A, 2) (= $(size(A, 2)))`")))
     elseif length(p) != size(A, 1)
         throw(DimensionMismatch(string("the length of row-permutation argument `p`, ",
             "`length(p) (= $(length(p)))`, must match source argument `A`'s row count, ",
             "`size(A, 1) (= $(size(A, 1)))`")))
     end
end
"""
Helper method for `permute` and `permute!` methods operating on `SparseMatrixCSC`s.
Checks whether row- and column- permutation arguments `p` and `q` are valid permutations.
"""
function _checkargs_permutationsvalid_permute!(
        p::AbstractVector{<:Integer}, pcheckspace::Vector{Ti},
        q::AbstractVector{<:Integer}, qcheckspace::Vector{Ti}) where Ti<:Integer
    if !_ispermutationvalid_permute!(p, pcheckspace)
        throw(ArgumentError("row-permutation argument `p` must be a valid permutation"))
    elseif !_ispermutationvalid_permute!(q, qcheckspace)
        throw(ArgumentError("column-permutation argument `q` must be a valid permutation"))
    end
end
function _ispermutationvalid_permute!(perm::AbstractVector{<:Integer},
        checkspace::Vector{<:Integer})
    require_one_based_indexing(perm)
    n = length(perm)
    checkspace[1:n] .= 0
    for k in perm
        (0 < k ≤ n) && ((checkspace[k] ⊻= 1) == 1) || return false
    end
    return true
end
"""
Helper method for `permute` and `permute!` methods operating on `SparseMatrixCSC`s.
Checks compatibility of source argument `A` and destination argument `X`.
"""
function _checkargs_sourcecompatdest_permute!(A::AbstractSparseMatrixCSC{Tv,Ti},
        X::AbstractSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    if size(X, 1) != size(A, 1)
        throw(DimensionMismatch(string("destination argument `X`'s row count, ",
            "`size(X, 1) (= $(size(X, 1)))`, must match source argument `A`'s row count, `size(A, 1) (= $(size(A, 1)))`")))
    elseif size(X, 2) != size(A, 2)
        throw(DimensionMismatch(string("destination argument `X`'s column count, ",
            "`size(X, 2) (= $(size(X, 2)))`, must match source argument `A`'s column count, `size(A, 2) (= $(size(A, 2)))`")))
    elseif length(rowvals(X)) < nnz(A)
        throw(ArgumentError(string("the length of destination argument `X`'s `rowval` ",
            "array, `length(rowvals(X)) (= $(length(rowvals(X))))`, must be greater than or ",
            "equal to source argument `A`'s allocated entry count, `nnz(A) (= $(nnz(A)))`")))
    elseif length(nonzeros(X)) < nnz(A)
        throw(ArgumentError(string("the length of destination argument `X`'s `nzval` ",
            "array, `length(nonzeros(X)) (= $(length(nonzeros(X))))`, must be greater than or ",
            "equal to source argument `A`'s allocated entry count, `nnz(A) (= $(nnz(A)))`")))
    end
end
"""
Helper method for `permute` and `permute!` methods operating on `SparseMatrixCSC`s.
Checks compatibility of source argument `A` and intermediate result argument `C`.
"""
function _checkargs_sourcecompatworkmat_permute!(A::AbstractSparseMatrixCSC{Tv,Ti},
        C::AbstractSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    if size(C, 2) != size(A, 1)
        throw(DimensionMismatch(string("intermediate result argument `C`'s column count, ",
            "`size(C, 2) (= $(size(C, 2)))`, must match source argument `A`'s row count, `size(A, 1) (= $(size(A, 1)))`")))
    elseif size(C, 1) != size(A, 2)
        throw(DimensionMismatch(string("intermediate result argument `C`'s row count, ",
            "`size(C, 1) (= $(size(C, 1)))`, must match source argument `A`'s column count, `size(A, 2) (= $(size(A, 2)))`")))
    elseif length(rowvals(C)) < nnz(A)
        throw(ArgumentError(string("the length of intermediate result argument `C`'s ",
            "`rowval` array, `length(rowvals(C)) (= $(length(rowvals(C))))`, must be greater than ",
            "or equal to source argument `A`'s allocated entry count, `nnz(A) (= $(nnz(A)))`")))
    elseif length(nonzeros(C)) < nnz(A)
        throw(ArgumentError(string("the length of intermediate result argument `C`'s ",
            "`rowval` array, `length(nonzeros(C)) (= $(length(nonzeros(C))))`, must be greater than ",
            "or equal to source argument `A`'s allocated entry count, `nnz(A)` (= $(nnz(A)))")))
    end
end
"""
Helper method for `permute` and `permute!` methods operating on `SparseMatrixCSC`s.
Checks compatibility of source argument `A` and workspace argument `workcolptr`.
"""
function _checkargs_sourcecompatworkcolptr_permute!(A::AbstractSparseMatrixCSC{Tv,Ti},
        workcolptr::Vector{Ti}) where {Tv,Ti}
    if length(workcolptr) <= size(A, 2)
        throw(DimensionMismatch(string("argument `workcolptr`'s length, ",
            "`length(workcolptr) (= $(length(workcolptr)))`, must exceed source argument ",
            "`A`'s column count, `size(A, 2) (= $(size(A, 2)))`")))
    end
end
"""
    permute!(X::AbstractSparseMatrixCSC{Tv,Ti}, A::AbstractSparseMatrixCSC{Tv,Ti},
             p::AbstractVector{<:Integer}, q::AbstractVector{<:Integer},
             [C::AbstractSparseMatrixCSC{Tv,Ti}]) where {Tv,Ti}

Bilaterally permute `A`, storing result `PAQ` (`A[p,q]`) in `X`. Stores intermediate result
`(AQ)^T` (`transpose(A[:,q])`) in optional argument `C` if present. Requires that none of
`X`, `A`, and, if present, `C` alias each other; to store result `PAQ` back into `A`, use
the following method lacking `X`:

    permute!(A::AbstractSparseMatrixCSC{Tv,Ti}, p::AbstractVector{<:Integer},
             q::AbstractVector{<:Integer}[, C::AbstractSparseMatrixCSC{Tv,Ti},
             [workcolptr::Vector{Ti}]]) where {Tv,Ti}

`X`'s dimensions must match those of `A` (`size(X, 1) == size(A, 1)` and `size(X, 2) == size(A, 2)`), and `X` must
have enough storage to accommodate all allocated entries in `A` (`length(rowvals(X)) >= nnz(A)`
and `length(nonzeros(X)) >= nnz(A)`). Column-permutation `q`'s length must match `A`'s column
count (`length(q) == size(A, 2)`). Row-permutation `p`'s length must match `A`'s row count
(`length(p) == size(A, 1)`).

`C`'s dimensions must match those of `transpose(A)` (`size(C, 1) == size(A, 2)` and `size(C, 2) == size(A, 1)`), and `C`
must have enough storage to accommodate all allocated entries in `A` (`length(rowvals(C)) >= nnz(A)`
and `length(nonzeros(C)) >= nnz(A)`).

For additional (algorithmic) information, and for versions of these methods that forgo
argument checking, see (unexported) parent methods `unchecked_noalias_permute!`
and `unchecked_aliasing_permute!`.

See also [`permute`](@ref).
"""
function permute!(X::AbstractSparseMatrixCSC{Tv,Ti}, A::AbstractSparseMatrixCSC{Tv,Ti},
        p::AbstractVector{<:Integer}, q::AbstractVector{<:Integer}) where {Tv,Ti}
    _checkargs_sourcecompatdest_permute!(A, X)
    _checkargs_sourcecompatperms_permute!(A, p, q)
    # bypass strict buffer checking
    C = spzeros(Tv, Ti, size(A,2), size(A,1))
    resize!(getrowval(C), nnz(A))
    resize!(getnzval(C), nnz(A))

    _checkargs_permutationsvalid_permute!(p, getcolptr(C), q, getcolptr(X))
    unchecked_noalias_permute!(X, A, p, q, C)
end
function permute!(X::AbstractSparseMatrixCSC{Tv,Ti}, A::AbstractSparseMatrixCSC{Tv,Ti},
        p::AbstractVector{<:Integer}, q::AbstractVector{<:Integer},
        C::AbstractSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    _checkargs_sourcecompatdest_permute!(A, X)
    _checkargs_sourcecompatperms_permute!(A, p, q)
    _checkargs_sourcecompatworkmat_permute!(A, C)
    _checkargs_permutationsvalid_permute!(p, getcolptr(C), q, getcolptr(X))
    unchecked_noalias_permute!(X, A, p, q, C)
end
function permute!(A::AbstractSparseMatrixCSC{Tv,Ti}, p::AbstractVector{<:Integer},
        q::AbstractVector{<:Integer}) where {Tv,Ti}
    _checkargs_sourcecompatperms_permute!(A, p, q)
    C = spzeros(Tv, Ti, size(A,2), size(A,1))
    resize!(getrowval(C), nnz(A))
    resize!(getnzval(C), nnz(A))
    workcolptr = Vector{Ti}(undef, size(A, 2) + 1)
    _checkargs_permutationsvalid_permute!(p, getcolptr(C), q, workcolptr)
    unchecked_aliasing_permute!(A, p, q, C, workcolptr)
end
function permute!(A::AbstractSparseMatrixCSC{Tv,Ti}, p::AbstractVector{<:Integer},
        q::AbstractVector{<:Integer}, C::AbstractSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    _checkargs_sourcecompatperms_permute!(A, p, q)
    _checkargs_sourcecompatworkmat_permute!(A, C)
    workcolptr = Vector{Ti}(undef, size(A, 2) + 1)
    _checkargs_permutationsvalid_permute!(p, getcolptr(C), q, workcolptr)
    unchecked_aliasing_permute!(A, p, q, C, workcolptr)
end
function permute!(A::AbstractSparseMatrixCSC{Tv,Ti}, p::AbstractVector{<:Integer},
        q::AbstractVector{<:Integer}, C::AbstractSparseMatrixCSC{Tv,Ti},
        workcolptr::Vector{Ti}) where {Tv,Ti}
    _checkargs_sourcecompatperms_permute!(A, p, q)
    _checkargs_sourcecompatworkmat_permute!(A, C)
    _checkargs_sourcecompatworkcolptr_permute!(A, workcolptr)
    _checkargs_permutationsvalid_permute!(p, getcolptr(C), q, workcolptr)
    unchecked_aliasing_permute!(A, p, q, C, workcolptr)
end
"""
    permute(A::AbstractSparseMatrixCSC{Tv,Ti}, p::AbstractVector{<:Integer},
            q::AbstractVector{<:Integer}) where {Tv,Ti}

Bilaterally permute `A`, returning `PAQ` (`A[p,q]`). Column-permutation `q`'s length must
match `A`'s column count (`length(q) == size(A, 2)`). Row-permutation `p`'s length must match `A`'s
row count (`length(p) == size(A, 1)`).

For expert drivers and additional information, see [`permute!`](@ref).

# Examples
```jldoctest
julia> A = spdiagm(0 => [1, 2, 3, 4], 1 => [5, 6, 7])
4×4 SparseMatrixCSC{Int64, Int64} with 7 stored entries:
 1  5  ⋅  ⋅
 ⋅  2  6  ⋅
 ⋅  ⋅  3  7
 ⋅  ⋅  ⋅  4

julia> permute(A, [4, 3, 2, 1], [1, 2, 3, 4])
4×4 SparseMatrixCSC{Int64, Int64} with 7 stored entries:
 ⋅  ⋅  ⋅  4
 ⋅  ⋅  3  7
 ⋅  2  6  ⋅
 1  5  ⋅  ⋅

julia> permute(A, [1, 2, 3, 4], [4, 3, 2, 1])
4×4 SparseMatrixCSC{Int64, Int64} with 7 stored entries:
 ⋅  ⋅  5  1
 ⋅  6  2  ⋅
 7  3  ⋅  ⋅
 4  ⋅  ⋅  ⋅
```
"""
function permute(A::AbstractSparseMatrixCSC{Tv,Ti}, p::AbstractVector{<:Integer},
        q::AbstractVector{<:Integer}) where {Tv,Ti}
    _checkargs_sourcecompatperms_permute!(A, p, q)
    # bypass strict buffer checking
    X = spzeros(Tv, Ti, size(A,1), size(A,2))
    resize!(getrowval(X), nnz(A))
    resize!(getnzval(X), nnz(A))
    # bypass strict buffer checking
    C = spzeros(Tv, Ti, size(A,2), size(A,1))
    resize!(getrowval(C), nnz(A))
    resize!(getnzval(C), nnz(A))
    _checkargs_permutationsvalid_permute!(p, getcolptr(C), q, getcolptr(X))
    unchecked_noalias_permute!(X, A, p, q, C)
end

## Sorting

#sorting TODO: integrate with `Base.Sort.IEEEFloatOptimization`'s partitioning by zero
searchsortedfirst_discard_keywords(v::AbstractVector, x; lt=isless, by=identity,
    rev::Union{Bool,Nothing}=nothing, order::Base.Order.Ordering=Forward, kws...) =
        searchsortedfirst(v, x, Base.Order.ord(lt,by,rev,order))

"""
    sort!(A::AbstractSparseMatrixCSC; dims::Integer, kws...)

Sort `A` in place along dimension `dims`, moving its stored entries to their sorted
positions without adding new stored entries, so that `nnz(A)` is unchanged. Within each
column (or row), stored values that compare equal to zero under the ordering are grouped
after the structural zeros, so the result may differ from the dense `sort!` for orderings
that do not distinguish stored values from zero (such as `by = iszero`).

`A` may not be a `FixedSparseCSC`, since its row indices are read-only; use
[`sort`](@ref) instead.

The remaining keyword arguments are those of `sort!` for a `Vector`.
"""
function Base.sort!(A::AbstractSparseMatrixCSC; dims::Integer, kws...)
    if _is_fixed(A)
        throw(ArgumentError("cannot sort! a FixedSparseCSC in place, its row indices are read-only"))
    end
    if dims == 1
        _sortcolumns!(A; kws...)
    elseif dims == 2
        # the rows of `A` are the columns of `transpose(A)`, which is cheap to form and
        # cheap to transpose back once its columns are sorted
        At = ftranspose(A, identity)
        _sortcolumns!(At; kws...)
        transpose!(A, At)
    else
        throw(ArgumentError(lazy"dimension out of range, got dims = $dims, expected 1 or 2"))
    end
    return A
end

# each column view is sorted through the sparse vector `sort!`; one scratch buffer is
# shared between the columns so that Base does not allocate a fresh one per column
function _sortcolumns!(A::AbstractSparseMatrixCSC; scratch=nothing, kws...)
    require_one_based_indexing(A)
    scratch = something(scratch, Vector{eltype(A)}(undef, 0))
    for j in axes(A, 2)
        sort!(view(A, :, j); scratch, kws...)
    end
    # with no columns there is nothing to sort, but the keywords are still validated
    size(A, 2) == 0 && sort!(view(nonzeros(A), 1:0); scratch, kws...)
    return A
end

"""
    sort(A::AbstractSparseMatrixCSC; dims::Integer, kws...)

Return a sorted copy of `A` along dimension `dims` as a `SparseMatrixCSC`, keeping only the
stored entries of `A`. See [`sort!`](@ref) for the treatment of stored values that compare
equal to zero.
"""
Base.sort(A::AbstractSparseMatrixCSC; kws...) =
    # the generic `Base.sort` for matrices goes through `permutedims`/`reshape` and does
    # not return a `SparseMatrixCSC` for `dims = 1`; `copy` of a `FixedSparseCSC` shares
    # its read-only structure, so convert to a writable `SparseMatrixCSC` in that case
    sort!(_is_fixed(A) ? SparseMatrixCSC(A) : copy(A); kws...)

## fkeep! and children tril!, triu!, droptol!, dropzeros[!]

function _fkeep!(f::F, A::AbstractSparseMatrixCSC) where F<:Function
    An = size(A, 2)
    Acolptr = getcolptr(A)
    Arowval = rowvals(A)
    Anzval = nonzeros(A)

    # Sweep through columns, rewriting kept elements in their new positions
    # and updating the column pointers accordingly as we go.
    Awritepos = 1
    oldAcolptrAj = 1
    @inbounds for Aj in 1:An
        for Ak in oldAcolptrAj:(Acolptr[Aj+1]-1)
            Ai = Arowval[Ak]
            Ax = Anzval[Ak]
            # If this element should be kept, rewrite in new position
            if f(Ai, Aj, Ax)
                if Awritepos != Ak
                    Arowval[Awritepos] = Ai
                    Anzval[Awritepos] = Ax
                end
                Awritepos += 1
            end
        end
        oldAcolptrAj = Acolptr[Aj+1]
        Acolptr[Aj+1] = Awritepos
    end

    # Trim A's storage if necessary
    Annz = Acolptr[end] - 1
    resize!(Arowval, Annz)
    resize!(Anzval, Annz)

    return A
end

"""
    fkeep!(f, A::AbstractSparseArray)

Keep elements of `A` for which test `f` returns `true`. `f`'s signature should be

    f(i::Integer, [j::Integer,] x) -> Bool

where `i` and `j` are an element's row and column indices and `x` is the element's
value. This method makes a single sweep
through `A`, requiring `O(size(A, 2), nnz(A))`-time for matrices and `O(nnz(A))`-time for vectors
and no space beyond that passed in.

# Examples
```jldoctest
julia> A = sparse(Diagonal([1, 2, 3, 4]))
4×4 SparseMatrixCSC{Int64, Int64} with 4 stored entries:
 1  ⋅  ⋅  ⋅
 ⋅  2  ⋅  ⋅
 ⋅  ⋅  3  ⋅
 ⋅  ⋅  ⋅  4

julia> SparseArrays.fkeep!((i, j, v) -> isodd(v), A)
4×4 SparseMatrixCSC{Int64, Int64} with 2 stored entries:
 1  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  3  ⋅
 ⋅  ⋅  ⋅  ⋅
```
"""
fkeep!(f::F, A::AbstractSparseMatrixCSC) where F<:Function = _is_fixed(A) ? _fkeep!_fixed(f, A) : _fkeep!(f, A)

# deprecated syntax
function fkeep!(x::SparseVecOrMat, f::F) where F<:Function
    Base.depwarn("`fkeep!(x, f::Function)` is deprecated, use `fkeep!(f::Function, x)` instead.", :fkeep!)
    return fkeep!(f, x)
end


tril!(A::AbstractSparseMatrixCSC, k::Integer = 0) =
    fkeep!((i, j, x) -> i + k >= j, A)
triu!(A::AbstractSparseMatrixCSC, k::Integer = 0) =
    fkeep!((i, j, x) -> j >= i + k, A)

"""
    droptol!(A::AbstractSparseMatrixCSC, tol)

Removes stored values from `A` whose absolute value is less than or equal to `tol`.
"""
droptol!(A::AbstractSparseMatrixCSC, tol) =
    fkeep!((i, j, x) -> abs(x) > tol, A)

"""
    dropzeros!(A::AbstractSparseMatrixCSC;)

Removes stored numerical zeros from `A`.

For an out-of-place version, see [`dropzeros`](@ref). For
algorithmic information, see `fkeep!`.
"""

dropzeros!(A::AbstractSparseMatrixCSC) = _is_fixed(A) ? A : fkeep!((i, j, x) -> _isnotzero(x), A)

"""
    dropzeros(A::AbstractSparseMatrixCSC;)

Generates a copy of `A` and removes stored numerical zeros from that copy.

For an in-place version and algorithmic information, see [`dropzeros!`](@ref).

# Examples
```jldoctest
julia> A = sparse([1, 2, 3], [1, 2, 3], [1.0, 0.0, 1.0])
3×3 SparseMatrixCSC{Float64, Int64} with 3 stored entries:
 1.0   ⋅    ⋅
  ⋅   0.0   ⋅
  ⋅    ⋅   1.0

julia> dropzeros(A)
3×3 SparseMatrixCSC{Float64, Int64} with 2 stored entries:
 1.0   ⋅    ⋅
  ⋅    ⋅    ⋅
  ⋅    ⋅   1.0
```
"""
dropzeros(A::AbstractSparseMatrixCSC) = dropzeros!(copy(A))

## Find methods

function findall(S::AbstractSparseMatrixCSC)
    return findall(identity, S)
end

function findall(p::Function, S::AbstractSparseMatrixCSC)
    if p(zero(eltype(S)))
        return invoke(findall, Tuple{Function, Any}, p, S)
    end

    numnz = nnz(S)
    inds = Vector{CartesianIndex{2}}(undef, numnz)

    count = 0
    @inbounds for col = 1 : size(S, 2), k = nzrange(S, col)
        if p(nonzeros(S)[k])
            count += 1
            inds[count] = CartesianIndex(rowvals(S)[k], col)
        end
    end

    resize!(inds, count)

    return inds
end
findall(p::Base.Fix2{typeof(in)}, x::AbstractSparseMatrixCSC) =
    invoke(findall, Tuple{Base.Fix2{typeof(in)}, AbstractArray}, p, x)

function findnz(S::AbstractSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    numnz = nnz(S)
    I = Vector{Ti}(undef, numnz)
    J = Vector{Ti}(undef, numnz)
    V = Vector{Tv}(undef, numnz)

    count = 1
    @inbounds for col = 1 : size(S, 2), k = nzrange(S, col)
        I[count] = rowvals(S)[k]
        J[count] = col
        V[count] = nonzeros(S)[k]
        count += 1
    end

    return (I, J, V)
end

# Materializing the (conjugate) transpose is linear in `nnz` and yields the
# indices in column-major order of the wrapped matrix, consistent with the
# `AbstractSparseMatrixCSC` method above.
findnz(S::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}) = findnz(copy(S))

function _sparse_findnextnz(m::AbstractSparseMatrixCSC, ij::CartesianIndex{2})
    row, col = Tuple(ij)
    col > size(m, 2) && return nothing

    lo, hi = getcolptr(m)[col], getcolptr(m)[col+1]
    n = searchsortedfirst(view(rowvals(m), lo:hi-1), row) + lo - 1
    if lo <= n <= hi-1
        return CartesianIndex(rowvals(m)[n], col)
    end
    nextcol = searchsortedfirst(view(getcolptr(m), col+1:length(getcolptr(m))), hi + 1) + col
    nextcol > length(getcolptr(m)) && return nothing
    nextlo = getcolptr(m)[nextcol-1]
    return CartesianIndex(rowvals(m)[nextlo], nextcol - 1)
end

function _sparse_findprevnz(m::AbstractSparseMatrixCSC, ij::CartesianIndex{2})
    row, col = Tuple(ij)
    iszero(col) && return nothing

    lo, hi = getcolptr(m)[col], getcolptr(m)[col+1]
    n = searchsortedlast(view(rowvals(m), lo:hi-1), row) + lo - 1
    if lo <= n <= hi-1
        return CartesianIndex(rowvals(m)[n], col)
    end
    prevcol = searchsortedlast(view(getcolptr(m), 1:col-1), lo - 1)
    prevcol < 1 && return nothing
    prevhi = getcolptr(m)[prevcol+1]
    return CartesianIndex(rowvals(m)[prevhi-1], prevcol)
end


Base.iszero(A::AbstractSparseMatrixCSC) = iszero(nzvalview(A))

function Base.isone(A::AbstractSparseMatrixCSC)
    m, n = size(A)
    m == n && getcolptr(A)[n+1] >= n+1 || return false
    for j in axes(A,2)
        founddiag = false
        for k in nzrange(A, j)
            i, x = rowvals(A)[k], nonzeros(A)[k]
            if i == j
                isone(x) || return false
                founddiag = true
            else
                iszero(x) || return false
            end
        end
        # every column must have a stored diagonal entry equal to one
        founddiag || return false
    end
    return true
end

# TODO: More appropriate location?
function conj!(A::AbstractSparseMatrixCSC)
    map!(conj, nzvalview(A), nzvalview(A))
    return A
end
function (-)(A::AbstractSparseMatrixCSC)
    nzval = similar(nonzeros(A), typeof(-zero(eltype(A))))
    map!(-, view(nzval, 1:nnz(A)), nzvalview(A))
    return SparseMatrixCSC(size(A, 1), size(A, 2), copy(getcolptr(A)), copy(rowvals(A)), nzval)
end

# the rest of real, conj, imag are handled correctly via AbstractArray methods
function conj(A::AbstractSparseMatrixCSC{<:Complex})
    nzval = similar(nonzeros(A))
    map!(conj, view(nzval, 1:nnz(A)), nzvalview(A))
    return SparseMatrixCSC(size(A, 1), size(A, 2), copy(getcolptr(A)), copy(rowvals(A)), nzval)
end
imag(A::SparseMatrixCSCOrView{Tv,Ti}) where {Tv<:Real,Ti} = spzeros(Tv, Ti, size(A, 1), size(A, 2))

## Binary arithmetic and boolean operators
(+)(A::SparseMatrixCSCOrView, B::SparseMatrixCSCOrView) = map(+, A, B)
(-)(A::SparseMatrixCSCOrView, B::SparseMatrixCSCOrView) = map(-, A, B)

function (+)(A::SparseMatrixCSCOrView, B::Array)
    Base.promote_shape(axes(A), axes(B))
    C = Ref(zero(eltype(A))) .+ B
    rowinds, nzvals = getrowval(A), getnzval(A)
    for j in axes(A,2)
        @inbounds for i in getnzrange(A, j)
            rowidx = rowinds[i]
            C[rowidx,j] = nzvals[i] + B[rowidx,j]
        end
    end
    return C
end
function (+)(A::Array, B::SparseMatrixCSCOrView)
    Base.promote_shape(axes(A), axes(B))
    C = A .+ Ref(zero(eltype(B)))
    rowinds, nzvals = getrowval(B), getnzval(B)
    for j in axes(B,2)
        @inbounds for i in getnzrange(B, j)
            rowidx = rowinds[i]
            C[rowidx,j] = A[rowidx,j] + nzvals[i]
        end
    end
    return C
end
function (-)(A::SparseMatrixCSCOrView, B::Array)
    Base.promote_shape(axes(A), axes(B))
    C = Ref(zero(eltype(A))) .- B
    rowinds, nzvals = getrowval(A), getnzval(A)
    for j in axes(A,2)
        @inbounds for i in getnzrange(A, j)
            rowidx = rowinds[i]
            C[rowidx,j] = nzvals[i] - B[rowidx,j]
        end
    end
    return C
end
function (-)(A::Array, B::SparseMatrixCSCOrView)
    Base.promote_shape(axes(A), axes(B))
    C = A .- Ref(zero(eltype(B)))
    rowinds, nzvals = getrowval(B), getnzval(B)
    for j in axes(B,2)
        @inbounds for i in getnzrange(B, j)
            rowidx = rowinds[i]
            C[rowidx,j] = A[rowidx,j] - nzvals[i]
        end
    end
    return C
end

## full equality
# Compare two CSC matrices by walking their stored entries only. `eq` is the elementwise
# predicate (`==` or `isequal`); stored entries without a counterpart are compared against
# the implicit zero of the other matrix so that e.g. `isequal(-0.0, 0.0)` and
# `isequal(NaN, NaN)` behave as they do for dense arrays.
function _iseq(eq::F, A1::AbstractSparseMatrixCSC, A2::AbstractSparseMatrixCSC) where {F}
    size(A1) != size(A2) && return false
    @inbounds for i in axes(A1, 2)
        nz1, nz2 = nzrange(A1,i), nzrange(A2,i)
        j1, j2 = first(nz1), first(nz2)
        # step through the rows of both matrices at once:
        while j1 <= last(nz1) && j2 <= last(nz2)
            r1, r2 = rowvals(A1)[j1], rowvals(A2)[j2]
            if r1 == r2
                eq(nonzeros(A1)[j1], nonzeros(A2)[j2]) || return false
                j1 += 1
                j2 += 1
            elseif r1 < r2
                _iszero_under(eq, nonzeros(A1)[j1]) || return false
                j1 += 1
            else # r1 > r2
                _iszero_under(eq, nonzeros(A2)[j2]) || return false
                j2 += 1
            end
        end
        # finish off any left-overs:
        for j = j1:last(nz1)
            _iszero_under(eq, nonzeros(A1)[j]) || return false
        end
        for j = j2:last(nz2)
            _iszero_under(eq, nonzeros(A2)[j]) || return false
        end
    end
    return true
end

==(A1::AbstractSparseMatrixCSC, A2::AbstractSparseMatrixCSC) = _iseq(==, A1, A2)
Base.isequal(A1::AbstractSparseMatrixCSC, A2::AbstractSparseMatrixCSC) = _iseq(isequal, A1, A2)

## Explicit efficient comparisons with transposed arrays

# Check whether all nonzero elements of A are equal to the respective elements in B
# under the elementwise predicate `eq` (`==` or `isequal`)
function nzeq(eq::F, A::AbstractSparseMatrixCSC, B::AbstractMatrix) where {F}
    @inbounds for j in axes(A,2)
        for k in nzrange(A, j)
            i = rowvals(A)[k]
            val = nonzeros(A)[k]
            eq(val, B[i,j]) || return false
        end
    end
    return true
end
# Peel off `Adjoint` and `Transpose` from first argument
# `B` may be a nested wrapper such as `Adjoint{<:Any,<:Transpose}` (from `A' == transpose(B)`),
# hence the loose `AbstractMatrix` bound: `B` is only ever indexed
nzeq(eq::F, A::Adjoint{<:Any,<:SparseMatrixCSCMaybeAdjOrTrans},
     B::AbstractMatrix) where {F} =
    nzeq(eq, A', B')
nzeq(eq::F, A::Transpose{<:Any,<:SparseMatrixCSCMaybeAdjOrTrans},
     B::AbstractMatrix) where {F} =
    nzeq(eq, transpose(A), transpose(B))

# Compare by walking both matrices
# (We could further optimize the case `AbstractSparseMatrixCSC ==
# Adjoint(Transpose(AbstractSparseMatrixCSC))` more efficiently, i.e.
# the case where the RHS is both adjoint and transposed, i.e. where it
# is in CSC format again.)
function _iseq(eq::F, A::AbstractSparseMatrixCSC,
               B::AdjOrTrans{<:Any,<:SparseMatrixCSCMaybeAdjOrTrans}) where {F}
    # Different sizes are always different
    size(A) ≠ size(B) && return false
    # Compare nonzero elements
    return nzeq(eq, A, B) && nzeq(eq, B, A)
end
==(A::AbstractSparseMatrixCSC, B::AdjOrTrans{<:Any,<:SparseMatrixCSCMaybeAdjOrTrans}) =
    _iseq(==, A, B)
Base.isequal(A::AbstractSparseMatrixCSC, B::AdjOrTrans{<:Any,<:SparseMatrixCSCMaybeAdjOrTrans}) =
    _iseq(isequal, A, B)
# Peel off `Adjoint` and `Transpose` from first argument
==(A::Adjoint{<:Any,<:SparseMatrixCSCMaybeAdjOrTrans}, B::SparseMatrixCSCMaybeAdjOrTrans) =
    A' == B'
==(A::Transpose{<:Any,<:SparseMatrixCSCMaybeAdjOrTrans}, B::SparseMatrixCSCMaybeAdjOrTrans) =
    transpose(A) == transpose(B)
Base.isequal(A::Adjoint{<:Any,<:SparseMatrixCSCMaybeAdjOrTrans}, B::SparseMatrixCSCMaybeAdjOrTrans) =
    isequal(A', B')
Base.isequal(A::Transpose{<:Any,<:SparseMatrixCSCMaybeAdjOrTrans}, B::SparseMatrixCSCMaybeAdjOrTrans) =
    isequal(transpose(A), transpose(B))

## Structure query functions
issymmetric(A::AbstractSparseMatrixCSC) = is_hermsym(A, transpose)

ishermitian(A::AbstractSparseMatrixCSC) = is_hermsym(A, adjoint)

function is_hermsym(A::AbstractSparseMatrixCSC, check::Function)
    m, n = size(A)
    if m != n; return false; end

    colptr = getcolptr(A)
    rowval = rowvals(A)
    nzval = nonzeros(A)
    tracker = copy(getcolptr(A))
    @inbounds for col in axes(A,2)
        # `tracker` is updated such that, for symmetric matrices,
        # the loop below starts from an element at or below the
        # diagonal element of column `col`"
        for p = tracker[col]:colptr[col+1]-1
            val = nzval[p]
            row = rowval[p]

            # Ignore stored zeros
            if iszero(val)
                continue
            end

            # If the matrix was symmetric we should have updated
            # the tracker to start at the diagonal or below. Here
            # we are above the diagonal so the matrix can't be symmetric.
            if row < col
                return false
            end

            # Diagonal element
            if row == col
                if val != check(val)
                    return false
                end
            else
                # if nzrange(A, row) is empty, then A[:, row] is all zeros.
                # Specifically, A[col, row] is zero.
                # However, we know at this point that A[row, col] is not zero
                # This means that the matrix is not symmetric
                isempty(nzrange(A, row)) && return false

                offset = tracker[row]

                # If the matrix is unsymmetric, there might not exist
                # a rowval[offset]
                if offset > colptr[row+1] - 1
                    return false
                end

                row2 = rowval[offset]

                # row2 can be less than col if the tracker didn't
                # get updated due to stored zeros in previous elements.
                # We therefore "catch up" here while making sure that
                # the elements are actually zero.
                while row2 < col
                    if _isnotzero(nzval[offset])
                        return false
                    end
                    offset += 1
                    tracker[row] += 1
                    # Column `row` ran out of stored entries before
                    # reaching row `col`, so A[col, row] does not exist
                    if offset > colptr[row+1] - 1
                        return false
                    end
                    row2 = rowval[offset]
                end

                # Non zero A[i,j] exists but A[j,i] does not exist
                if row2 > col
                    return false
                end

                # A[i,j] and A[j,i] exists
                if row2 == col
                    if val != check(nzval[offset])
                        return false
                    end
                    tracker[row] += 1
                end
            end
        end
    end
    return true
end

function istriu(A::AbstractSparseMatrixCSC, k::Integer=0)
    m, n = size(A)
    rowval = rowvals(A)
    nzval  = nonzeros(A)

    @inbounds for col = 1:min(n, m-1)
        for i in reverse(nzrange(A, col))
            if rowval[i] <= col - k
                # rows preceeding the index would also lie above the band
                break
            end
            if _isnotzero(nzval[i])
                return false
            end
        end
    end
    return true
end

function istril(A::AbstractSparseMatrixCSC, k::Integer=0)
    m, n = size(A)
    rowval = rowvals(A)
    nzval  = nonzeros(A)

    @inbounds for col = 2:n
        for i = nzrange(A, col)
            if rowval[i] >= col - k
                # subsequent rows would also lie below the band
                break
            end
            if _isnotzero(nzval[i])
                return false
            end
        end
    end
    return true
end

function isdiag(A::AbstractSparseMatrixCSC)
    m, n = size(A)
    rowval = rowvals(A)
    nzval = nonzeros(A)
    @inbounds for col in 1:n
        for k in nzrange(A, col)
            if rowval[k] != col && _isnotzero(nzval[k])
                return false
            end
        end
    end
    return true
end

## expand a colptr or rowptr into a dense index vector
function expandptr(V::Vector{<:Integer})
    if V[1] != 1 throw(ArgumentError("first index must be one")) end
    res = similar(V, (Int64(V[end]-1),))
    for i in 1:(length(V)-1), j in V[i]:(V[i+1] - 1); res[j] = i end
    res
end


function diag(A::AbstractSparseMatrixCSC{Tv,Ti}, d::Integer=0) where {Tv,Ti}
    m, n = size(A)
    k = Int(d)
    l = k < 0 ? min(m+k,n) : min(n-k,m)
    r, c = k <= 0 ? (-k, 0) : (0, k) # start row/col -1
    ind = Vector{Ti}()
    val = Vector{Tv}()
    for i in 1:l
        r += 1; c += 1
        r1 = Int(first(nzrange(A, c)))
        r2 = Int(last(nzrange(A, c)))
        r1 > r2 && continue
        r1 += searchsortedfirst(view(rowvals(A), r1:r2), r) - 1
        ((r1 > r2) || (rowvals(A)[r1] != r)) && continue
        push!(ind, i)
        push!(val, nonzeros(A)[r1])
    end
    return SparseVector{Tv,Ti}(l, ind, val)
end

function tr(A::AbstractSparseMatrixCSC{Tv}) where Tv
    n = checksquare(A)
    s = zero(Tv)
    for i in 1:n
        s += A[i,i]
    end
    return s
end

## rotations

function rot180(A::AbstractSparseMatrixCSC)
    I,J,V = findnz(A)
    m,n = size(A)
    for i=1:length(I)
        I[i] = m - I[i] + 1
        J[i] = n - J[i] + 1
    end
    return sparse(I,J,V,m,n)
end

function rotr90(A::AbstractSparseMatrixCSC)
    I,J,V = findnz(A)
    m,n = size(A)
    #old col inds are new row inds
    for i=1:length(I)
        I[i] = m - I[i] + 1
    end
    return sparse(J, I, V, n, m)
end

function rotl90(A::AbstractSparseMatrixCSC)
    I,J,V = findnz(A)
    m,n = size(A)
    #old row inds are new col inds
    for i=1:length(J)
        J[i] = n - J[i] + 1
    end
    return sparse(J, I, V, n, m)
end

## Uniform matrix arithmetic

(+)(A::AbstractSparseMatrixCSC{Tv, Ti}, J::UniformScaling{T}) where {T<:Number, Tv, Ti} =
    A + sparse(T, Ti, J, size(A)...)
(+)(J::UniformScaling{T}, A::AbstractSparseMatrixCSC{Tv, Ti}) where {T<:Number, Tv, Ti} =
    sparse(T, Ti, J, size(A)...) + A
(-)(A::AbstractSparseMatrixCSC{Tv, Ti}, J::UniformScaling{T}) where {T<:Number, Tv, Ti} =
    A - sparse(T, Ti, J, size(A)...)
(-)(J::UniformScaling{T}, A::AbstractSparseMatrixCSC{Tv, Ti}) where {T<:Number, Tv, Ti} =
    sparse(T, Ti, J, size(A)...) - A



## circular shift

function circshift!(O::AbstractSparseMatrixCSC, X::AbstractSparseMatrixCSC, (r,c)::Base.DimsInteger{2})
    nnz = length(nonzeros(X))

    iszero(nnz) && return copy!(O, X)

    ##### column shift
    c = mod(c, size(X, 2))
    if iszero(c)
        copy!(O, X)
    else
        ##### readjust output
        resize!(getcolptr(O), size(X, 2) + 1)
        resize!(rowvals(O), nnz)
        resize!(nonzeros(O), nnz)
        getcolptr(O)[size(X, 2) + 1] = nnz + 1

        # exchange left and right blocks
        nleft = getcolptr(X)[size(X, 2) - c + 1] - 1
        nright = nnz - nleft
        @inbounds for i=c+1:size(X, 2)
            getcolptr(O)[i] = getcolptr(X)[i-c] + nright
        end
        @inbounds for i=1:c
            getcolptr(O)[i] = getcolptr(X)[size(X, 2) - c + i] - nleft
        end
        # rotate rowval and nzval by the right number of elements
        circshift!(rowvals(O), rowvals(X), (nright,))
        circshift!(nonzeros(O), nonzeros(X), (nright,))
    end
    ##### row shift
    r = mod(r, size(X, 1))
    iszero(r) && return O
    @inbounds for i in axes(O, 2)
        subvector_shifter!(rowvals(O), nonzeros(O), first(nzrange(O, i)), last(nzrange(O, i)), size(O, 1), r)
    end
    return _checkbuffers(O)
end

circshift!(O::AbstractSparseMatrixCSC, X::AbstractSparseMatrixCSC, (r,)::Base.DimsInteger{1}) = circshift!(O, X, (r,0))
circshift!(O::AbstractSparseMatrixCSC, X::AbstractSparseMatrixCSC, r::Real) = circshift!(O, X, (Integer(r),0))
# a fixed X keeps its pattern under `similar`, so shift into a plain copy instead
circshift(X::AbstractSparseMatrixCSC, s::Base.DimsInteger) = circshift!(similar(_unsafe_unfix(X)), X, s)
circshift(X::AbstractSparseMatrixCSC, s::Real) = circshift!(similar(_unsafe_unfix(X)), X, (Integer(s),))

## swaprows! / swapcols!
macro swap(a, b)
    esc(:(($a, $b) = ($b, $a)))
end

function Base.swapcols!(A::AbstractSparseMatrixCSC, i, j)
    i == j && return

    # For simplicity, let i denote the smaller of the two columns
    j < i && @swap(i, j)

    colptr = getcolptr(A)
    irow = nzrange(A, i)
    jrow = nzrange(A, j)

    function rangeexchange!(arr, irow, jrow)
        if length(irow) == length(jrow)
            for (a, b) in zip(irow, jrow)
                @inbounds @swap(arr[a], arr[b])
            end
            return
        end
        # This is similar to the triple-reverse tricks for
        # circshift!, except that we have three ranges here,
        # so it ends up being 4 reverse calls (but still
        # 2 overall reversals for the memory range). Like
        # circshift!, there's also a cycle chasing algorithm
        # with optimal memory complexity, but the performance
        # tradeoffs against this implementation are non-trivial,
        # so let's just do this simple thing for now.
        # See https://github.com/JuliaLang/julia/pull/42676 for
        # discussion of circshift!-like algorithms.
        reverse!(@view arr[irow])
        reverse!(@view arr[jrow])
        reverse!(@view arr[(last(irow)+1):(first(jrow)-1)])
        reverse!(@view arr[first(irow):last(jrow)])
    end
    rangeexchange!(rowvals(A), irow, jrow)
    rangeexchange!(nonzeros(A), irow, jrow)

    if length(irow) != length(jrow)
        @inbounds colptr[i+1:j] .+= length(jrow) - length(irow)
    end
    return nothing
end

function Base.swaprows!(A::AbstractSparseMatrixCSC, i, j)
    # For simplicity, let i denote the smaller of the two rows
    j < i && @swap(i, j)

    rows = rowvals(A)
    vals = nonzeros(A)
    for col in axes(A,2)
        rr = nzrange(A, col)
        iidx = searchsortedfirst(@view(rows[rr]), i)
        has_i = iidx <= length(rr) && rows[rr[iidx]] == i

        jrange = has_i ? (rr[iidx]:last(rr)) : rr
        jidx = searchsortedlast(@view(rows[jrange]), j)
        has_j = jidx != 0 && rows[jrange[jidx]] == j

        if !has_j && !has_i
            # Has neither row - nothing to do
            continue
        elseif has_i && has_j
            # This column had both i and j rows - swap them
            @swap(vals[rr[iidx]], vals[jrange[jidx]])
        elseif has_i
            # Update the rowval and then rotate both nonzeros
            # and the remaining rowvals into the correct place
            rows[rr[iidx]] = j
            rotate_range = rr[iidx]:jrange[jidx]
            circshift!(@view(vals[rotate_range]), -1)
            circshift!(@view(rows[rotate_range]), -1)
        else
            # Same as i, but in the opposite direction
            @assert has_j
            rows[jrange[jidx]] = i
            rotate_range = rr[iidx]:jrange[jidx]
            circshift!(@view(vals[rotate_range]), 1)
            circshift!(@view(rows[rotate_range]), 1)
        end
    end
    return nothing
end

reverse(A::AbstractSparseMatrixCSC; dims=:) = _reverse(A, dims)
function _reverse(A::AbstractSparseMatrixCSC, ::Colon)
    rowinds, colinds, nzval = findnz(A)
    rowinds .= (size(A,1) + 1) .- rowinds
    colinds .= (size(A,2) + 1) .- colinds
    sparse!(rowinds, colinds, nzval, size(A)...)
end
function _reverse(A::AbstractSparseMatrixCSC, dims::Integer)
    dims ∈ (1,2) || throw(ArgumentError("invalid dimension $dims in reverse"))
    rowinds, colinds, nzval = findnz(A)
    if dims == 1
        rowinds .= (size(A,1) + 1) .- rowinds
    else # dims == 2
        colinds .= (size(A,2) + 1) .- colinds
    end
    sparse!(rowinds, colinds, nzval, size(A)...)
end
function _reverse(A::AbstractSparseMatrixCSC, dims::Tuple{Integer,Integer})
    dims == (1,2) || dims == (2,1) || throw(ArgumentError("invalid dimension $dims in reverse"))
    _reverse(A, :)
end

reverse(S::SparseMatrixCSC; dims...) = reverse!(copy(S); dims...)
reverse!(S::SparseMatrixCSC; dims=:) = _reverse!(S, dims)
function _reverse!(S::SparseMatrixCSC, ::Colon)
    rowinds, nzval = rowvals(S), nonzeros(S)
    colptr = getcolptr(S)
    rowinds .= (size(S,1) + 1) .- rowinds
    reverse!(rowinds)
    colptr .= (nnz(S) + 2) .- colptr
    reverse!(colptr)
    reverse!(nzval)
    return S
end
function _reverse!(S::SparseMatrixCSC, dims::Integer)
    dims ∈ (1,2) || throw(ArgumentError("invalid dimension $dims in reverse"))
    rowinds, nzval = rowvals(S), nonzeros(S)
    colptr = getcolptr(S)
    nzrs = nzrange.(Ref(S), axes(S,2))
    if dims == 1
        for col in axes(S,2)
            nzr = nzrs[col]
            reverse!(@views nzval[nzr])
            rowinds_col = @view rowinds[nzr]
            rowinds_col .= (size(S,1) + 1) .- rowinds_col
            reverse!(rowinds_col)
        end
    else # dims == 2
        colptr .= (nnz(S) + 2) .- colptr
        reverse!(colptr)
        for col in axes(S,2)
            nzr = nzrs[col]
            reverse!(@views nzval[nzr])
            reverse!(@views rowinds[nzr])
        end
        reverse!(nzval)
        reverse!(rowinds)
    end
    return S
end
function _reverse!(A::SparseMatrixCSC, dims::Tuple{Integer,Integer})
    dims == (1,2) || dims == (2,1) || throw(ArgumentError("invalid dimension $dims in reverse"))
    _reverse!(A, :)
end

function copytrito!(M::AbstractMatrix, S::AbstractSparseMatrixCSC, uplo::Char)
    Base.require_one_based_indexing(M, S)
    if !(uplo == 'U' || uplo == 'L')
        throw(ArgumentError(lazy"uplo argument must be 'U' (upper) or 'L' (lower), got '$uplo'"))
    end
    m,n = size(S)
    m1,n1 = size(M)
    (m1 < m || n1 < n) && throw(DimensionMismatch("dest of size ($m1,$n1) should have at least the same number of rows and columns than src of size ($m,$n)"))

    rv = rowvals(S)
    nz = nonzeros(S)
    @inbounds for col in axes(S,2)
        trirange = uplo == 'U' ? (1:min(col, size(S,1))) : (col:size(S,1))
        fill!(view(M, trirange, col), zero(eltype(S)))
        for i in nzrange(S, col)
            row = rv[i]
            (uplo == 'U' && row <= col) || (uplo == 'L' && row >= col) || continue
            M[row, col] = nz[i]
        end
    end
    return M
end
