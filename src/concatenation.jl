# This file is a part of Julia. License is MIT: https://julialang.org/license

# `hcat`, `vcat`, `hvcat`, `blockdiag` and `repeat` for sparse matrices and vectors.

# Sparse concatenation

promote_idxtype(::AbstractSparseMatrixCSC{<:Any, Ti}) where {Ti} = Ti
promote_idxtype(::AbstractSparseMatrixCSC{<:Any, Ti}, X::AbstractSparseMatrixCSC...) where {Ti} =
    promote_type(Ti, promote_idxtype(X...))

function vcat(X::AbstractSparseMatrixCSC...)
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

    Tv = promote_eltype(X...)
    Ti = promote_idxtype(X...)

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

            ptr_res = stuffcol!(rowval, nzval, ptr_res, rowvals(X[i]), nonzeros(X[i]), ptr_Xi,
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
        if nnzX[i] == length(rowvals(XI))
            rowval[(1 : nnzX[i]) .+ nnz_sofar] = rowvals(XI)
            nzval[(1 : nnzX[i]) .+ nnz_sofar] = nonzeros(XI)
        else
            rowval[(1 : nnzX[i]) .+ nnz_sofar] = rowvals(XI)[1:nnzX[i]]
            nzval[(1 : nnzX[i]) .+ nnz_sofar] = nonzeros(XI)[1:nnzX[i]]
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
    rowval = similar(rowvals(A), nnz_new)
    nzval = similar(nonzeros(A), nnz_new)

    colptr[1] = 1
    for c = 1 : size(A, 2)
        ptr_res = colptr[c]
        ptr_source = getcolptr(A)[c]
        col_length = getcolptr(A)[c + 1] - ptr_source
        for index_repetition = 0 : (m - 1)
            row_offset = index_repetition * size(A, 1)
            ptr_res = stuffcol!(rowval, nzval, ptr_res, rowvals(A), nonzeros(A), ptr_source,
                                col_length, row_offset)
        end
        colptr[c + 1] = ptr_res
    end
    @assert colptr[end] == nnz_new + 1

    SparseMatrixCSC(size(A, 1) * m, size(A, 2), colptr, rowval, nzval)
end

function Base.repeat(A::AbstractSparseMatrixCSC, m, n)
    B = repeat(A, m)
    nnz_per_column = diff(getcolptr(B))
    colptr = cumsum(vcat(1, repeat(nnz_per_column, n)))
    rowval = repeat(rowvals(B), n)
    nzval = repeat(nonzeros(B), n)
    SparseMatrixCSC(size(B, 1), size(B, 2) * n, colptr, rowval, nzval)
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
    Tv = promote_type(map(x->eltype(nonzeros(x)), X)...)
    Ti = promote_type(map(x->eltype(rowvals(x)), X)...)
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
        rowval[(1 : nnzX[i]) .+ nnz_sofar] = rowvals(X[i]) .+ mX_sofar
        nzval[(1 : nnzX[i]) .+ nnz_sofar] = nonzeros(X[i])
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

const _SparseVecConcatGroup = Union{Vector, AbstractSparseVector}
function hcat(X::_SparseVecConcatGroup...)
    if _concatsparse(X...)
        X = map(sparse, X)
    end
    return cat(X...; dims=Val(2))
end
function vcat(X::_SparseVecConcatGroup...)
    if _concatsparse(X...)
        X = map(sparse, X)
    end
    return cat(X...; dims=Val(1))
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
Base.@constprop :aggressive function _sparse_cat_t(dims, ::Type{T}, X...) where {T}
    catdims = Base.dims2cat(dims)
    shape = Base.cat_size_shape(catdims, X...)
    A = _catdest(T, shape, X...)
    if count(!iszero, catdims)::Int > 1
        fill!(A, zero(T))
    end
    return Base.__cat(A, shape, catdims, X...)
end
# with only arrays, `typed_hcat`/`typed_vcat` reach the same destination through `similar`
# of the first, now sparse, array; a number among them takes the `cat` path as in Base
_sparse_typed_hcat(::Type{T}, X::AbstractVecOrMat...) where {T} = Base.typed_hcat(T, X...)
_sparse_typed_hcat(::Type{T}, X...) where {T} = _sparse_cat_t(Val(2), T, X...)
_sparse_typed_vcat(::Type{T}, X::AbstractVecOrMat...) where {T} = Base.typed_vcat(T, X...)
_sparse_typed_vcat(::Type{T}, X...) where {T} = _sparse_cat_t(Val(1), T, X...)

# `@constprop :aggressive` allows `dims` to be propagated as constant improving return type inference
Base.@constprop :aggressive function cat_internal(dims, X1::_SparseConcatGroup, X::_SparseConcatGroup...)
    T = promote_eltype(X1, X...)
    if _concatsparse(X1, X...)
        return _sparse_cat_t(dims, T, _makesparse(X1), map(_makesparse, X)...)
    end
    return Base._cat_t(dims, T, X1, X...)
end
function hcat_internal(X1::_SparseConcatGroup, X::_SparseConcatGroup...)
    T = promote_eltype(X1, X...)
    if _concatsparse(X1, X...)
        return _sparse_typed_hcat(T, _makesparse(X1), map(_makesparse, X)...)
    end
    return Base.typed_hcat(T, X1, X...)
end
function vcat_internal(X1::_SparseConcatGroup, X::_SparseConcatGroup...)
    T = promote_eltype(X1, X...)
    if _concatsparse(X1, X...)
        return _sparse_typed_vcat(T, _makesparse(X1), map(_makesparse, X)...)
    end
    return Base.typed_vcat(T, X1, X...)
end
function hvcat_internal(rows::Tuple{Vararg{Int}}, X1::_SparseConcatGroup, X::_SparseConcatGroup...)
    if _concatsparse(X1, X...)
        vcat(_hvcat_rows(rows, X1, X...)...)
    else
        Base.typed_hvcat(Base.promote_eltypeof(X1, X...), rows, X1, X...)
    end
end
function _hvcat_rows((row1, rows...)::Tuple{Vararg{Int}}, X::_SparseConcatGroup...)
    if row1 ≤ 0
        throw(ArgumentError("length of block row must be positive, got $row1"))
    end
    # assert `X` is non-empty so that inference of `eltype` won't include `Type{Union{}}`
    T = eltype(X::Tuple{Any,Vararg{Any}})
    # inference of `getindex` may be imprecise in case `row1` is not const-propagated up
    # to here, so help inference with the following type-assertions
    return (
        hcat(X[1 : row1]::Tuple{typeof(X[1]),Vararg{T}}...),
        _hvcat_rows(rows, X[row1+1:end]::Tuple{Vararg{T}}...)...
    )
end
_hvcat_rows(::Tuple{}, X::_SparseConcatGroup...) = ()

# `cat` is not overloaded by packages the way `vcat` and `hcat` are, so its hook keeps the
# narrower numeric group, which avoids invalidating Base's `cat` on non-numeric vectors
const _NumericSparseConcatGroup = Union{AbstractVecOrMat{<:Number},Number}
Base.@constprop :aggressive Base._cat(dims, X1::_NumericSparseConcatGroup, X::_NumericSparseConcatGroup...) =
    cat_internal(dims, X1, X...)
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
