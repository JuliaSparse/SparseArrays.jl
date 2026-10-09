# This file is a part of Julia. License is MIT: https://julialang.org/license

# `hcat`, `vcat`, `hvcat`, `blockdiag` and `repeat` for sparse matrices and vectors.

# Sparse concatenation

promote_idxtype(::AbstractSparseMatrixCSC{<:Any, Ti}) where {Ti} = Ti
promote_idxtype(::AbstractSparseMatrixCSC{<:Any, Ti}, X::AbstractSparseMatrixCSC...) where {Ti} =
    promote_type(Ti, promote_idxtype(X...))

vcat(X::AbstractSparseMatrixCSC...) = _vcat_csc(promote_eltype(X...), promote_idxtype(X...), X...)
function _vcat_csc(::Type{Tv}, ::Type{Ti}, X::AbstractSparseMatrixCSC...) where {Tv,Ti}
    num = length(X)
    mX = Int[ size(x, 1) for x in X ]
    nX = Int[ size(x, 2) for x in X ]
    m = sum(mX)
    n = nX[1]

    for i = 2 : num
        if nX[i] != n
            throw(DimensionMismatch("All inputs to vcat should have the same number of columns"))
        end
    end

    nnzX = Int[ nnz(x) for x in X ]
    nnz_res = sum(nnzX)
    colptr = Vector{Ti}(undef, n+1)
    rowval = Vector{Ti}(undef, nnz_res)
    nzval  = Vector{Tv}(undef, nnz_res)

    colptr[1] = 1
    for c = 1:n
        mX_sofar = 0
        ptr_res = colptr[c]
        for i = 1 : num
            colptrXi = getcolptr(X[i])
            col_length = colptrXi[c + 1] - colptrXi[c]
            ptr_Xi = colptrXi[c]

            ptr_res = stuffcol!(rowval, nzval, ptr_res, getrowval(X[i]), getnzval(X[i]), ptr_Xi,
                                col_length, mX_sofar)
            mX_sofar += mX[i]
        end
        colptr[c + 1] = ptr_res
    end
    SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

@inline function stuffcol!(rowval, nzval, ptr_res, rowvalXi, nzvalXi, ptr_Xi,
                           col_length, mX_sofar)
    for k=ptr_res:(ptr_res + col_length - 1)
        @inbounds rowval[k] = rowvalXi[ptr_Xi] + mX_sofar
        @inbounds nzval[k]  = nzvalXi[ptr_Xi]
        ptr_Xi += 1
    end
    return ptr_res + col_length
end

function hcat(X::AbstractSparseMatrixCSC...)
    num = length(X)
    mX = Int[ size(x, 1) for x in X ]
    nX = Int[ size(x, 2) for x in X ]
    m = mX[1]
    for i = 2 : num
        if mX[i] != m; throw(DimensionMismatch("")); end
    end
    n = sum(nX)

    Tv = promote_eltype(X...)
    Ti = promote_idxtype(X...)

    colptr = Vector{Ti}(undef, n+1)
    nnzX = Int[ nnz(x) for x in X ]
    nnz_res = sum(nnzX)
    rowval = Vector{Ti}(undef, nnz_res)
    nzval = Vector{Tv}(undef, nnz_res)

    nnz_sofar = 0
    nX_sofar = 0
    @inbounds for i = 1 : num
        XI = X[i]
        colptr[(1 : nX[i] + 1) .+ nX_sofar] = getcolptr(XI) .+ nnz_sofar
        if nnzX[i] == length(getrowval(XI))
            rowval[(1 : nnzX[i]) .+ nnz_sofar] = getrowval(XI)
            nzval[(1 : nnzX[i]) .+ nnz_sofar] = getnzval(XI)
        else
            rowval[(1 : nnzX[i]) .+ nnz_sofar] = getrowval(XI)[1:nnzX[i]]
            nzval[(1 : nnzX[i]) .+ nnz_sofar] = getnzval(XI)[1:nnzX[i]]
        end
        nnz_sofar += nnzX[i]
        nX_sofar += nX[i]
    end

    SparseMatrixCSC(m, n, colptr, rowval, nzval)
end


# Efficient repetition of sparse matrices

function Base.repeat(A::AbstractSparseMatrixCSC, m)
    nnz_new = nnz(A) * m
    colptr = similar(getcolptr(A), length(getcolptr(A)))
    rowval = similar(getrowval(A), nnz_new)
    nzval = similar(getnzval(A), nnz_new)

    colptr[1] = 1
    for c = 1 : size(A, 2)
        ptr_res = colptr[c]
        ptr_source = getcolptr(A)[c]
        col_length = getcolptr(A)[c + 1] - ptr_source
        for index_repetition = 0 : (m - 1)
            row_offset = index_repetition * size(A, 1)
            ptr_res = stuffcol!(rowval, nzval, ptr_res, getrowval(A), getnzval(A), ptr_source,
                                col_length, row_offset)
        end
        colptr[c + 1] = ptr_res
    end
    @assert colptr[end] == nnz_new + 1

    SparseMatrixCSC(size(A, 1) * m, size(A, 2), colptr, rowval, nzval)
end

function Base.repeat(A::AbstractSparseMatrixCSC, m, n)
    B = repeat(A, m)
    nB = size(B, 2)
    nnzB = nnz(B)
    colptrB = getcolptr(B)
    colptr = similar(colptrB, nB * n + 1)
    colptr[1] = 1
    for k = 0 : (n - 1), c = 1 : nB
        colptr[k * nB + c + 1] = colptrB[c + 1] + k * nnzB
    end
    rowval = repeat(getrowval(B), n)
    nzval = repeat(getnzval(B), n)
    SparseMatrixCSC(size(B, 1), nB * n, colptr, rowval, nzval)
end


"""
    blockdiag(A...)

Concatenate matrices block-diagonally. Currently only implemented for sparse matrices.

# Examples
```jldoctest
julia> blockdiag(sparse(2I, 3, 3), sparse(4I, 2, 2))
5×5 SparseMatrixCSC{Int64, Int64} with 5 stored entries:
 2  ⋅  ⋅  ⋅  ⋅
 ⋅  2  ⋅  ⋅  ⋅
 ⋅  ⋅  2  ⋅  ⋅
 ⋅  ⋅  ⋅  4  ⋅
 ⋅  ⋅  ⋅  ⋅  4
```
"""
blockdiag() = spzeros(promote_type(), Int, 0, 0)

function blockdiag(X::AbstractSparseMatrixCSC{Tv, Ti}...) where {Tv, Ti <: Integer}
    _blockdiag(Tv, Ti, X...)
end

function blockdiag(X::AbstractSparseMatrixCSC...)
    Tv = promote_type(map(x->eltype(getnzval(x)), X)...)
    Ti = promote_type(map(x->eltype(getrowval(x)), X)...)
    _blockdiag(Tv, Ti, X...)
end

function _blockdiag(::Type{Tv}, ::Type{Ti}, X::AbstractSparseMatrixCSC...) where {Tv, Ti <: Integer}
    num = length(X)
    mX = Int[ size(x, 1) for x in X ]
    nX = Int[ size(x, 2) for x in X ]
    m = sum(mX)
    n = sum(nX)

    colptr = Vector{Ti}(undef, n+1)
    nnzX = Int[ nnz(x) for x in X ]
    nnz_res = sum(nnzX)
    rowval = Vector{Ti}(undef, nnz_res)
    nzval = Vector{Tv}(undef, nnz_res)

    nnz_sofar = 0
    nX_sofar = 0
    mX_sofar = 0
    for i = 1 : num
        colptr[(1 : nX[i] + 1) .+ nX_sofar] = getcolptr(X[i]) .+ nnz_sofar
        rowval[(1 : nnzX[i]) .+ nnz_sofar] = getrowval(X[i]) .+ mX_sofar
        nzval[(1 : nnzX[i]) .+ nnz_sofar] = getnzval(X[i])
        nnz_sofar += nnzX[i]
        nX_sofar += nX[i]
        mX_sofar += mX[i]
    end
    colptr[n+1] = nnz_sofar + 1

    SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

### Concatenation

function hcat(Xin::AbstractSparseVector...)
    X = map(_unsafe_unfix, Xin)
    Tv = promote_type(map(eltype, X)...)
    Ti = promote_type(map(indtype, X)...)
    r = stack(SparseVector{Tv,Ti}[X...])
    return @if_move_fixed Xin... r
end

# `stack` of sparse vectors along a new trailing dimension. Base's generic loop calls
# `copyto!(B, offset, x)` per slice, which copies every element, including the stored
# zeros, through `getindex` and `setindex!` on sparse arrays. Building the CSC arrays
# directly costs O(nnz + n) instead of O(m * n).
function Base._typed_stack(::Colon, ::Type{Tv}, ::Type{S}, A, Aax::Tuple{Any}) where {Tv,S<:SparseVectorOrView}
    X = A isa AbstractArray ? A : collect(A)
    isempty(X) && return Base._empty_stack(:, Tv, S, A)
    Ti = mapreduce(indtype, promote_type, X)
    n = length(X)
    m = length(first(X))
    tnnz = 0
    for x in X
        length(x) == m ||
            throw(DimensionMismatch("Inconsistent column lengths."))
        tnnz += nnz(x)
    end

    colptr = Vector{Ti}(undef, n+1)
    nzrow = Vector{Ti}(undef, tnnz)
    nzval = Vector{Tv}(undef, tnnz)
    roff = 1
    j = 0
    @inbounds for x in X
        j += 1
        colptr[j] = roff
        copyto!(nzrow, roff, nonzeroinds(x))
        copyto!(nzval, roff, nonzeros(x))
        roff += nnz(x)
    end
    colptr[n+1] = roff
    return SparseMatrixCSC{Tv,Ti}(m, n, colptr, nzrow, nzval)
end
# Sparse vector slices only reach `_dim_stack` with `dims` other than 2.
function Base._dim_stack(dims::Integer, ::Type{Tv}, ::Type{S}, A) where {Tv,S<:SparseVectorOrView}
    dims == 1 || throw(ArgumentError(LazyString("cannot stack slices ndims(x) = 1 along dims = ", dims)))
    return permutedims(Base._typed_stack(:, Tv, S, A, (Base._vec_axis(A),)), (2, 1))
end

function vcat(Xin::AbstractSparseVector...)
    X = map(_unsafe_unfix, Xin)
    Tv = promote_type(map(eltype, X)...)
    Ti = promote_type(map(indtype, X)...)
    r = (function (::Type{SV}) where SV
            _absspvec_vcat(map(x -> convert(SV, x), X)...)
        end)(SparseVector{Tv,Ti})
    return @if_move_fixed Xin... r
end
function _absspvec_vcat(X1::AbstractSparseVector{Tv,Ti}, Xs::AbstractSparseVector{Tv,Ti}...) where {Tv,Ti}
    X = (X1, Xs...)
    # check sizes
    n = length(X)
    tnnz = 0
    for j = 1:n
        tnnz += nnz(X[j])
    end

    # construction
    rnzind = Vector{Ti}(undef, tnnz)
    rnzval = Vector{Tv}(undef, tnnz)
    ir = 0
    len = 0
    @inbounds for j = 1:n
        xj = X[j]
        xnzind = nonzeroinds(xj)
        xnzval = nonzeros(xj)
        xnnz = length(xnzind)
        for i = 1:xnnz
            rnzind[ir + i] = xnzind[i] + len
        end
        copyto!(rnzval, ir+1, xnzval)
        ir += xnnz
        len += length(xj)
    end
    SparseVector(len, rnzind, rnzval)
end

### Concatenation of un/annotated sparse/special/dense vectors/matrices
# by type-pirating and subverting the Base.cat design by making these a subtype of the normal methods for it
# and re-defining all of it here. See https://github.com/JuliaLang/julia/issues/2326
# for what would have been a more principled way of doing this.

# Concatenations involving un/annotated sparse/special matrices/vectors should yield sparse arrays

# the output array type is determined by the first element of the to be concatenated objects
# if this is a Number, the output would be dense by the fallback abstractarray.jl code (see cat_similar)
# so make sure that if that happens, the "array" is sparse (if more sparse arrays are involved, of course)
_sparse(x::Number) = sparsevec([1], [x], 1)
_sparse(A) = _makesparse(A)
_makesparse(x::Number) = x
_makesparse(x::AbstractVector) = convert(SparseVector, x)::SparseVector
_makesparse(x::AbstractMatrix) = convert(SparseMatrixCSC, x)::SparseMatrixCSC
# a `UniformScaling` has no size of its own: `LinearAlgebra._hcat`/`_vcat`/`_hvcat` size it
# from its neighbours and then call `promote_to_arrays_` below to make it sparse
_makesparse(J::UniformScaling) = J
anysparse() = false
anysparse(X) = X isa AbstractArray && issparse(X)
anysparse(X, Xs...) = anysparse(X) || anysparse(Xs...)
anysparse(X::T, Xs::T...) where {T} = anysparse(X)

# The result is sparse only when some input is sparse and every input has a `Number`
# eltype; otherwise `zero` may not exist and Base's dense `cat` handles it (#71).
_concatsparse(X...) = anysparse(X...) && _allnumeric(X...)
_allnumeric() = true
_allnumeric(X, Xs...) = eltype(X) <: Number && _allnumeric(Xs...)
# Base's dense concatenation allocates its result with `similar` of the first array. When
# that array is sparse the result is too, and filling it needs `zero`, which a non-`Number`
# eltype may not have, so the sparse inputs are made dense before Base sees them (#71)
_densesparse(x) = anysparse(x) ? Array(x) : x

const _SparseVecConcatGroup = Union{Vector, AbstractSparseVector}
function hcat(X::_SparseVecConcatGroup...)
    if _concatsparse(X...)
        return cat(map(sparse, X)...; dims=Val(2))
    end
    return cat(map(_densesparse, X)...; dims=Val(2))
end
function vcat(X::_SparseVecConcatGroup...)
    if _concatsparse(X...)
        return cat(map(sparse, X)...; dims=Val(1))
    end
    return cat(map(_densesparse, X)...; dims=Val(1))
end

# Type piracy of Base's `cat` design; see https://github.com/JuliaLang/julia/issues/2326 for
# what a principled hook would look like. Each entry point below mirrors one of Base's own
# `Vararg` signatures with a fixed first argument, which makes it more specific than Base's
# method but less specific than a package's `vcat(::AbstractMatrix, ::MyArray)`, so no
# ambiguity is introduced for arrays that are not sparse (#431).
const _SparseConcatGroup = Union{AbstractVecOrMat,Number}

# Base's `_cat_t` takes the output type from its first argument, so with a leading number
# it would build a dense array. Choose the destination from the first array instead, and
# keep the number as is so that it fills its block the way it does in dense
# concatenation (#383). A leading number still widens the index type to at least `Int`,
# as the one-element sparse vector standing in for it used to. `X` has already been
# through `_makesparse`.
_catleader(X1::AbstractArray, X...) = X1
_catleader(X1::Number, X...) = _catleader(X...)
_catleader(X1::Number) = _sparse(X1)
_catdest(::Type{T}, shape, X1::AbstractArray, X...) where {T} = similar(X1, T, shape)
function _catdest(::Type{T}, shape, X1::Number, X...) where {T}
    A = _catleader(X1, X...)
    return similar(A, T, promote_type(Int, indtype(A)), shape)
end
# The compiled method serves every value of `dims`, so `dims2cat(dims)` has an unknown
# length, and `similar` of a sparse array cannot be resolved for a shape of unknown length,
# which `juliac --trim` rejects. A sparse result has one or two dimensions, so each gets a
# branch with a concrete `catdims`; more dimensions give a dense result, which Base builds.
Base.@constprop :aggressive function _sparse_cat_t(dims, ::Type{T}, X...) where {T}
    catdims = Base.dims2cat(dims)
    if length(catdims) == 1
        return _sparse_cat_t_shaped((catdims[1],), T, X...)
    elseif length(catdims) == 2
        return _sparse_cat_t_shaped((catdims[1], catdims[2]), T, X...)
    end
    return Base._cat_t(dims, T, map(_catdense, X)...)
end
function _sparse_cat_t_shaped(catdims::Tuple{Vararg{Bool}}, ::Type{T}, X...) where {T}
    shape = Base.cat_size_shape(catdims, X...)
    A = _catdest(T, shape, X...)
    if count(catdims) > 1
        fill!(A, zero(T))
    end
    return Base.__cat(A, shape, catdims, X...)
end
_catdense(x::AbstractArray) = Array(x)
_catdense(x) = x
# with only arrays, `typed_hcat`/`typed_vcat` reach the same destination through `similar`
# of the first, now sparse, array; a number among them takes the `cat` path as in Base
_sparse_typed_hcat(::Type{T}, X::AbstractVecOrMat...) where {T} = Base.typed_hcat(T, X...)
_sparse_typed_hcat(::Type{T}, X...) where {T} = _sparse_cat_t(Val(2), T, X...)
_sparse_typed_vcat(::Type{T}, X::AbstractVector...) where {T} = Base.typed_vcat(T, X...)
_sparse_typed_vcat(::Type{T}, X::AbstractVecOrMat...) where {T} = Base.typed_vcat(T, map(_introws, X)...)
# Base stacks vectors on matrices with `size(x, 1)::Int`, and the length of a sparse vector
# is of its index type; as a one-column matrix it has `Int` dimensions
_introws(x::AbstractCompressedVector) = size(x, 1) isa Int ? x : SparseMatrixCSC(x)
_introws(x) = x
_sparse_typed_vcat(::Type{T}, X...) where {T} = _sparse_cat_t(Val(1), T, X...)

# `Vararg{_SparseConcatGroup,N}` makes Julia compile `cat_internal` for each argument
# count. Otherwise, past a few arguments, it is compiled for an unknown count and the splat
# into `Base._cat_t` is left unresolved, which `juliac --trim` rejects.
# `@constprop :aggressive` allows `dims` to be propagated as constant improving return type inference
Base.@constprop :aggressive function cat_internal(dims, X1::_SparseConcatGroup, X::Vararg{_SparseConcatGroup,N}) where {N}
    T = promote_eltype(X1, X...)
    if _concatsparse(X1, X...)
        return _sparse_cat_t(dims, T, _makesparse(X1), map(_makesparse, X)...)
    end
    return Base._cat_t(dims, T, _densesparse(X1), map(_densesparse, X)...)
end
function hcat_internal(X1::_SparseConcatGroup, X::_SparseConcatGroup...)
    T = promote_eltype(X1, X...)
    if _concatsparse(X1, X...)
        return _sparse_typed_hcat(T, _makesparse(X1), map(_makesparse, X)...)
    end
    return Base.typed_hcat(T, _densesparse(X1), map(_densesparse, X)...)
end
function vcat_internal(X1::_SparseConcatGroup, X::_SparseConcatGroup...)
    T = promote_eltype(X1, X...)
    if _concatsparse(X1, X...)
        return _sparse_typed_vcat(T, _makesparse(X1), map(_makesparse, X)...)
    end
    return Base.typed_vcat(T, _densesparse(X1), map(_densesparse, X)...)
end
# `Vararg{_SparseConcatGroup,N}` for the same reason as in `cat_internal`
function hvcat_internal(rows::Tuple{Vararg{Int}}, X1::_SparseConcatGroup, X::Vararg{_SparseConcatGroup,N}) where {N}
    if _concatsparse(X1, X...)
        return _sparse_hvcat(rows, _makesparse(X1), map(_makesparse, X)...)
    end
    return Base.typed_hvcat(Base.promote_eltypeof(X1, X...), rows, _densesparse(X1), map(_densesparse, X)...)
end
# `_hvcat_csc` reads the blocks by index. Splitting them into a tuple per block row would
# give tuples whose length depends on the value of `rows`, so their types could not be
# inferred and `juliac --trim` could not resolve the calls on them.
function _sparse_hvcat(rows::Tuple{Vararg{Int}}, X::Vararg{AbstractSparseMatrixCSC,N}) where {N}
    return _hvcat_csc(promote_eltype(X...), promote_idxtype(X...), rows, X...)
end
# A vector is an `n×1` block and a number a `1×1` block, as in dense `hvcat`. A number is
# stored unless scalar `setindex!` would leave it implicit, and a sparse vector keeps its
# stored zeros, as sparse matrix blocks do. A leading number widens the index type to at
# least `Int`, as it does for `hcat` and `vcat` (#383).
function _sparse_hvcat(rows::Tuple{Vararg{Int}}, X::Vararg{Any,N}) where {N}
    Tv = promote_eltype(X...)
    Ti = _hvcat_idxtype(X...)
    return _hvcat_csc(Tv, Ti, rows, map(x -> _hvcat_block(Tv, Ti, x), X)...)
end
_hvcat_idxtype(X1::Number, X...) = promote_type(Int, _blocks_idxtype(X...))
_hvcat_idxtype(X...) = _blocks_idxtype(X...)
_blocks_idxtype() = Union{}
_blocks_idxtype(x::Number, X...) = _blocks_idxtype(X...)
_blocks_idxtype(x::AbstractSparseArray, X...) = promote_type(indtype(x), _blocks_idxtype(X...))
_hvcat_block(::Type, ::Type, B::AbstractSparseMatrixCSC) = B
function _hvcat_block(::Type, ::Type, x::AbstractSparseVector{<:Any,Ti}) where {Ti}
    return SparseMatrixCSC(length(x), 1, Ti[1, nnz(x) + 1], nonzeroinds(x), nonzeros(x))
end
function _hvcat_block(::Type{Tv}, ::Type{Ti}, x::Number) where {Tv,Ti}
    v = convert(Tv, x)
    _isimplicitzero(v, Tv) && return SparseMatrixCSC(1, 1, Ti[1, 1], Ti[], Tv[])
    return SparseMatrixCSC(1, 1, Ti[1, 2], Ti[1], Tv[v])
end
function _hvcat_csc(::Type{Tv}, ::Type{Ti}, rows::Tuple{Vararg{Int}}, X::Vararg{AbstractSparseMatrixCSC,N}) where {Tv,Ti,N}
    nblocks = 0
    for r in rows
        r > 0 || throw(ArgumentError("length of block row must be positive, got $r"))
        nblocks += r
    end
    nblocks == length(X) ||
        throw(DimensionMismatch(lazy"block rows $rows take $nblocks blocks, got $(length(X))"))
    m, n, k = 0, 0, 0
    for (b, r) in enumerate(rows)
        h, w = size(X[k + 1], 1), 0
        for i in k+1:k+r
            size(X[i], 1) == h || throw(DimensionMismatch(
                lazy"block $i has $(size(X[i], 1)) rows, but block row $b has $h"))
            w += size(X[i], 2)
        end
        b == 1 || w == n ||
            throw(DimensionMismatch(lazy"block row $b has $w columns, but block row 1 has $n"))
        m, n, k = m + h, w, k + r
    end
    # Build the result column by column, so that it is written in order. Block `blk[b]` of
    # block row `b` covers the current column, and `lastcol[b]` is the last column it covers.
    nbr = length(rows)
    blk, lastcol, rowoff = Vector{Int}(undef, nbr), Vector{Int}(undef, nbr), Vector{Int}(undef, nbr)
    k, i0 = 0, 0
    for (b, r) in enumerate(rows)
        blk[b], lastcol[b], rowoff[b] = k + 1, size(X[k + 1], 2), i0
        i0 += size(X[k + 1], 1)
        k += r
    end
    nnzres = 0
    for x in X
        nnzres += nnz(x)
    end
    colptr = Vector{Ti}(undef, n + 1)
    rowval = Vector{Ti}(undef, nnzres)
    nzval = Vector{Tv}(undef, nnzres)
    colptr[1] = p = 1
    # the shape checks above keep `blk[b]` within the blocks of block row `b`
    @inbounds for j in 1:n
        for b in 1:nbr
            while j > lastcol[b]
                blk[b] += 1
                lastcol[b] += size(X[blk[b]], 2)
            end
            B = X[blk[b]]
            p = _hvcat_copycol!(rowval, nzval, p, B, j - lastcol[b] + size(B, 2), rowoff[b])
        end
        colptr[j + 1] = p
    end
    return SparseMatrixCSC(m, n, colptr, rowval, nzval)
end
# `c` is a column of `B` and `p` stays within the `nnz` of all the blocks, by the shape
# checks in `_hvcat_csc`
function _hvcat_copycol!(rowval, nzval, p, B, c, i0)
    cp, rv, nz = getcolptr(B), getrowval(B), getnzval(B)
    @inbounds for q in cp[c]:(cp[c + 1] - 1)
        rowval[p] = rv[q] + i0
        nzval[p] = nz[q]
        p += 1
    end
    return p
end

# `cat` is not overloaded by packages the way `vcat` and `hcat` are, so its hook keeps the
# narrower numeric group, which avoids invalidating Base's `cat` on non-numeric vectors
const _NumericSparseConcatGroup = Union{AbstractVecOrMat{<:Number},Number}
Base.@constprop :aggressive Base._cat(dims, X1::_NumericSparseConcatGroup, X::_NumericSparseConcatGroup...) =
    cat_internal(dims, X1, X...)
# With a non-`Number` array, `cat` does not reach the hook above, and Base's `cat` would
# allocate a sparse result when a sparse array comes first; see `_densesparse`. This is a
# method of `cat`, which Base defines only for `A...`, because a method of `_cat` taking a
# sparse array first is ambiguous with Base's `_cat(dims, A::AbstractArray{T}...)`.
const _SparseCatLeader = Union{AbstractSparseVecOrMat,AdjOrTrans{<:Any,<:AbstractSparseVecOrMat}}
@inline function Base.cat(X1::_SparseCatLeader, X::Vararg{Any,N}; dims) where {N}
    _allnumeric(X1, X...) && return Base._cat(dims, X1, X...)
    return Base._cat(dims, _densesparse(X1), map(_densesparse, X)...)
end
for f in (:hcat, :vcat)
    f_internal = Symbol(f, :_internal)
    @eval begin
        $f(X1::AbstractVecOrMat{T}, X::AbstractVecOrMat{T}...) where {T} = $f_internal(X1, X...)
        $f(X1::AbstractVecOrMat, X::AbstractVecOrMat...) = $f_internal(X1, X...)
        $f(X1::_SparseConcatGroup, X::_SparseConcatGroup...) = $f_internal(X1, X...)
        # disambiguation against Base's `Vararg{Number}` and `Vararg{T<:Number}` methods
        $f(n1::Number, ns::Vararg{Number}) = invoke($f, Tuple{Vararg{Number}}, n1, ns...)
        $f(n1::N, ns::Vararg{N}) where {N<:Number} = invoke($f, Tuple{Vararg{N}}, n1, ns...)
    end
end
# `vcat` alone has `Vararg{AbstractVector}` methods in Base
vcat(X1::AbstractVector{T}, X::AbstractVector{T}...) where {T} = vcat_internal(X1, X...)
vcat(X1::AbstractVector, X::AbstractVector...) = vcat_internal(X1, X...)
hvcat(rows::Tuple{Vararg{Int}}, X1::AbstractVecOrMat{T}, X::AbstractVecOrMat{T}...) where {T} = hvcat_internal(rows, X1, X...)
hvcat(rows::Tuple{Vararg{Int}}, X1::AbstractVecOrMat, X::AbstractVecOrMat...) = hvcat_internal(rows, X1, X...)
hvcat(rows::Tuple{Vararg{Int}}, X1::_SparseConcatGroup, X::_SparseConcatGroup...) = hvcat_internal(rows, X1, X...)
hvcat(rows::Tuple{Vararg{Int}}, n1::Number, ns::Vararg{Number}) = hvcat_internal(rows, n1, ns...)
hvcat(rows::Tuple{Vararg{Int}}, n1::N, ns::Vararg{N}) where {N<:Number} = hvcat_internal(rows, n1, ns...)

### Efficient repetition of sparse vectors

function Base.repeat(v::AbstractSparseVector, m)
    nnz_source = nnz(v)
    nnz_new = nnz_source * m

    nzind = similar(nonzeroinds(v), nnz_new)
    nzval = similar(nonzeros(v), nnz_new)

    ptr_res = 1
    for index_repetition = 0:(m-1)
        row_offset = index_repetition * length(v)
        ptr_res = stuffcol!(nzind, nzval, ptr_res, nonzeroinds(v), nonzeros(v), 1, nnz_source, row_offset)
    end
    @assert ptr_res == nnz_new + 1

    SparseVector(length(v) * m, nzind, nzval)
end

function Base.repeat(v::AbstractSparseVector, m, n)
    w = repeat(v, m)
    colptr = Vector{eltype(nonzeroinds(w))}(1 .+ nnz(w) * (0:n))
    rowval = repeat(nonzeroinds(w), n)
    nzval = repeat(nonzeros(w), n)
    SparseMatrixCSC(length(w), n, colptr, rowval, nzval)
end


# make sure UniformScaling objects are converted to sparse matrices for concatenation
promote_to_array_type(A::Tuple{Vararg{Union{_SparseConcatGroup,UniformScaling}}}) = _concatsparse(A...) ? SparseMatrixCSC : Matrix
promote_to_arrays_(n::Int, ::Type{SparseMatrixCSC}, J::UniformScaling) = sparse(J, n, n)

"""
    sparse_hcat(A...)

Concatenate along dimension 2. Return a SparseMatrixCSC object.

!!! compat "Julia 1.8"
    This method was added in Julia 1.8. It mimics previous concatenation behavior, where
    the concatenation with specialized "sparse" matrix types from LinearAlgebra.jl
    automatically yielded sparse output even in the absence of any SparseArray argument.
"""
sparse_hcat(Xin::Union{AbstractVecOrMat,Number}...) = _sparse_cat_t(Val(2), promote_eltype(Xin...), map(_makesparse, Xin)...)
function sparse_hcat(X::Union{AbstractVecOrMat,UniformScaling,Number}...)
    LinearAlgebra._hcat(_sparse(first(X)), map(_makesparse, Base.tail(X))...; array_type = SparseMatrixCSC)
end

"""
    sparse_vcat(A...)

Concatenate along dimension 1. Return a SparseMatrixCSC object.

!!! compat "Julia 1.8"
    This method was added in Julia 1.8. It mimics previous concatenation behavior, where
    the concatenation with specialized "sparse" matrix types from LinearAlgebra.jl
    automatically yielded sparse output even in the absence of any SparseArray argument.
"""
sparse_vcat(Xin::Union{AbstractVecOrMat,Number}...) = _sparse_cat_t(Val(1), promote_eltype(Xin...), map(_makesparse, Xin)...)
function sparse_vcat(X::Union{AbstractVecOrMat,UniformScaling,Number}...)
    LinearAlgebra._vcat(_sparse(first(X)), map(_makesparse, Base.tail(X))...; array_type = SparseMatrixCSC)
end

"""
    sparse_hvcat(rows::Tuple{Vararg{Int}}, values...)

Sparse horizontal and vertical concatenation in one call. This function is called
for block matrix syntax. The first argument specifies the number of
arguments to concatenate in each block row.

!!! compat "Julia 1.8"
    This method was added in Julia 1.8. It mimics previous concatenation behavior, where
    the concatenation with specialized "sparse" matrix types from LinearAlgebra.jl
    automatically yielded sparse output even in the absence of any SparseArray argument.
"""
function sparse_hvcat(rows::Tuple{Vararg{Int}}, Xin::Union{AbstractVecOrMat,Number}...)
    hvcat(rows, _sparse(first(Xin)), map(_makesparse, Base.tail(Xin))...)
end
function sparse_hvcat(rows::Tuple{Vararg{Int}}, X::Union{AbstractVecOrMat,UniformScaling,Number}...)
    LinearAlgebra._hvcat(rows, _sparse(first(X)), map(_makesparse, Base.tail(X))...; array_type = SparseMatrixCSC)
end
