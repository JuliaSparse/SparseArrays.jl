# This file is a part of Julia. License is MIT: https://julialang.org/license

# The experimental fixed-pattern arrays: the read-only index vector behind them, the types,
# `fixed`/`move_fixed`, and the kernels that exist only for them. Checks for a fixed operand
# inside general kernels stay with those kernels.

"""
   `ReadOnly{T,N,<:AbstractArray{T,N,V<:AbstractArray{T,N}}} <: AbstractArray{T,N}`

Internal. Wrapper around an `AbstractArray` that blocks change. Practically no-op operations
are not blocked. For instance, `setindex!(x, getindex(x, i...), i...)` or `resize!(x, length(x))`
for `x isa ReadOnly`.
"""
struct ReadOnly{T,N,V<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::V
end
# ReadOnly of ReadOnly is meaningless
ReadOnly(x::ReadOnly) = x
Base.getproperty(x::ReadOnly, s::Symbol) = Base.getproperty(parent(x), s)
@inline Base.parent(x::ReadOnly) = getfield(x, :parent)

for i in [:length, :first, :last, :axes, :size]
    @eval Base.@propagate_inbounds @inline Base.$i(x::ReadOnly) = Base.$i(parent(x))
end
for i in [:iterate, :getindex, :strides]
    @eval(Base.@propagate_inbounds @inline Base.$i(x::ReadOnly, y...) = Base.$i(parent(x), y...))
end

Base.eachindex(i::IndexLinear, x::ReadOnly) = eachindex(i, parent(x))
Base.eachindex(i::IndexCartesian, x::ReadOnly) = eachindex(i, parent(x))

Base.unsafe_convert(x::Type{Ptr{T}}, A::ReadOnly) where T = Base.unsafe_convert(x, parent(A))
Base.elsize(::Type{ReadOnly{T,N,V}}) where {T,N,V} = Base.elsize(V)
@noinline _readonly_error(x) =
    throw(ArgumentError("cannot modify a $(nameof(typeof(x))) array, the sparsity pattern of a fixed sparse array is read-only"))
Base.@propagate_inbounds @inline Base.setindex!(x::ReadOnly, v, ind::Vararg{Integer}) =
    v == getindex(parent(x), ind...) ? v : _readonly_error(x)
for i in [:IteratorSize, :IndexStyle]
    @eval(@inline Base.$i(::Type{ReadOnly{T,N,V}}) where {T,N,V} = Base.$i(V))
end
@inline Base.resize!(x::ReadOnly, l) = l == length(parent(x)) ? x : _readonly_error(x)
Base.copy(x::ReadOnly) = ReadOnly(copy(parent(x)))
(==)(x::ReadOnly, y::AbstractVector) = parent(x) == y
(==)(x::AbstractVector, y::ReadOnly) = x == parent(y)
(==)(x::ReadOnly, y::ReadOnly) = parent(x) == parent(y)
# disambiguation
(==)(x::ReadOnly{T,1,<:AbstractVector{T}}, y::ReadOnly{S,1,<:AbstractVector{S}}) where {T,S} =
    parent(x) == parent(y)

Base.dataids(::ReadOnly) = tuple()

# Forward the sparse array interface to the parent, so that a `ReadOnly` wrapping a
# sparse array behaves like one. `issparse` already forwards through `parent`.
nnz(x::ReadOnly) = nnz(parent(x))
indtype(x::ReadOnly) = indtype(parent(x))

@inline _is_fixed(::AbstractArray) = false
@inline _is_fixed(A::AbstractArray, Bs::Vararg{Any,N}) where N = _is_fixed(A) || (N > 0 && _is_fixed(Bs...))
@noinline _throwfixedinsert(A, I...) =
    throw(ArgumentError("cannot store a new entry at ($(join(I, ", "))) in a $(nameof(typeof(A))), its sparsity pattern is read-only"))
macro if_move_fixed(a...)
    length(a) <= 1 && error("@if_move_fixed needs at least two arguments")
    h, v = esc.(a[1:end - 1]), esc(a[end])
    :(_is_fixed($(h...)) ? move_fixed($v) : $v)
end

"""
    FixedSparseCSC{Tv,Ti<:Integer} <: AbstractSparseMatrixCSC{Tv,Ti}

Experimental AbstractSparseMatrixCSC whose non-zero index are fixed.
"""
struct FixedSparseCSC{Tv,Ti<:Integer} <: AbstractSparseMatrixCSC{Tv,Ti}
    m::Int                  # Number of rows
    n::Int                  # Number of columns
    colptr::ReadOnly{Ti,1,Vector{Ti}} # Column i is in colptr[i]:(colptr[i+1]-1)
    rowval::ReadOnly{Ti,1,Vector{Ti}} # Row indices of stored values
    nzval::Vector{Tv}       # Stored values, typically nonzeros

    function FixedSparseCSC{Tv,Ti}(m::Integer, n::Integer,
                            colptr::ReadOnly{Ti,1,Vector{Ti}},
                            rowval::ReadOnly{Ti,1,Vector{Ti}},
                            nzval::Vector{Tv}) where {Tv,Ti<:Integer}
        sparse_check_Ti(m, n, Ti)
        _goodbuffers(Int(m), Int(n), parent(colptr), parent(rowval), nzval) ||
            throw(ArgumentError("Invalid buffers for FixedSparseCSC construction n=$n, colptr=$(summary(colptr)), rowval=$(summary(rowval)), nzval=$(summary(nzval))"))
        new(Int(m), Int(n), colptr, rowval, nzval)
    end
end
@inline _is_fixed(::FixedSparseCSC) = true
FixedSparseCSC(m::Integer, n::Integer,
    colptr::ReadOnly{Ti,1,Vector{Ti}},
    rowval::ReadOnly{Ti,1,Vector{Ti}},
    nzval::Vector{Tv}) where {Tv,Ti<:Integer} =
    FixedSparseCSC{Tv,Ti}(m, n, colptr, rowval, nzval)
FixedSparseCSC{Tv,Ti}(m::Integer, n::Integer, colptr::Vector{Ti}, rowval::Vector{Ti}, nzval::Vector{Tv}) where {Tv,Ti} =
    FixedSparseCSC{Tv,Ti}(m, n, ReadOnly(colptr), ReadOnly(rowval), nzval)
FixedSparseCSC(m::Integer, n::Integer, colptr::Vector{Ti}, rowval::Vector{Ti}, nzval::Vector{Tv}) where {Tv,Ti} =
    FixedSparseCSC{Tv,Ti}(m, n, ReadOnly(colptr), ReadOnly(rowval), nzval)
FixedSparseCSC(x::AbstractSparseMatrixCSC{Tv,Ti}) where {Tv,Ti} =
    FixedSparseCSC{Tv,Ti}(size(x, 1), size(x, 2),
        getcolptr(x), rowvals(x), nonzeros(x))
# shares x's buffers when the types already match, converts them otherwise
function FixedSparseCSC{Tv,Ti}(x::AbstractSparseMatrixCSC) where {Tv,Ti}
    y = _unsafe_unfix(x)
    FixedSparseCSC{Tv,Ti}(size(y, 1), size(y, 2),
        convert(Vector{Ti}, getcolptr(y)), convert(Vector{Ti}, rowvals(y)), convert(Vector{Tv}, nonzeros(y)))
end

"""
    fixed(x...)

Experimental. Like `sparse` but returns a sparse array whose sparsity pattern is read-only:
stored entries can change value, but none can be added or removed.
"""
fixed(x...) = move_fixed(sparse(x...))
fixed(x::AbstractSparseMatrixCSC) = FixedSparseCSC(x)

"""
    move_fixed(x::AbstractSparseMatrixCSC)

Experimental, unsafe. Make a `FixedSparseCSC` by reusing the colptr, rowvals and nonzeros of `x`.
"""
move_fixed(x::AbstractSparseMatrixCSC) = FixedSparseCSC(size(x)..., getcolptr(x), rowvals(x), nonzeros(x))
"""
    _unsafe_unfix(x)

Experimental, unsafe. Returns a modifiable version of `x` for compatibility with this codebase.
"""
_unsafe_unfix(x::FixedSparseCSC) = SparseMatrixCSC(size(x)..., parent(getcolptr(x)), parent(rowvals(x)), nonzeros(x))
_unsafe_unfix(x::AbstractSparseMatrixCSC) = x

# A fixed destination keeps its pattern: B's stored entries must lie in it and A's other
# entries become zero. The pattern is checked in full before anything is written.
function _copyto_fixed!(A::AbstractSparseMatrixCSC, B::AbstractSparseMatrixCSC)
    size(A) == size(B) || throw(DimensionMismatch(lazy"cannot copy a matrix of size $(size(B)) into a fixed one of size $(size(A))"))
    Arv, Brv, Anz, Bnz = rowvals(A), rowvals(B), nonzeros(A), nonzeros(B)
    for write in (false, true)
        write && fill!(Anz, zero(eltype(A)))
        @inbounds for j in axes(A, 2)
            k, kend = Int(first(nzrange(A, j))), Int(last(nzrange(A, j)))
            for p in nzrange(B, j)
                i = Brv[p]
                while k <= kend && Arv[k] < i; k += 1; end
                (k <= kend && Arv[k] == i) || _throwfixedinsert(A, i, j)
                write && (Anz[k] = Bnz[p])
            end
        end
    end
    return A
end

function _fkeep!_fixed(f::F, A::AbstractSparseMatrixCSC) where F<:Function
    @inbounds for j in axes(A,2)
        for k in nzrange(A, j)
            # If this element should be kept, rewrite in new position
            if !f(rowvals(A)[k], j, nonzeros(A)[k])
                nonzeros(A)[k] = zero(eltype(A))
            end
        end
    end
    return A
end

"""
    FixedSparseVector{Tv,Ti<:Integer} <: AbstractCompressedVector{Tv,Ti}

Experimental AbstractCompressedVector whose non-zero index are fixed.
"""
struct FixedSparseVector{Tv,Ti<:Integer} <: AbstractCompressedVector{Tv,Ti}
    n::Ti              # Length of the sparse vector
    nzind::ReadOnly{Ti,1,Vector{Ti}}   # Indices of stored values
    nzval::Vector{Tv}   # Stored values, typically nonzeros

    function FixedSparseVector{Tv,Ti}(n::Integer, nzind::ReadOnly{Ti,1,Vector{Ti}}, nzval::Vector{Tv}) where {Tv,Ti<:Integer}
        n >= 0 || throw(ArgumentError("The number of elements must be non-negative."))
        length(nzind) == length(nzval) ||
            throw(ArgumentError("index and value vectors must be the same length"))
        new(convert(Ti, n), nzind, nzval)
    end
end
@inline _is_fixed(::FixedSparseVector) = true
FixedSparseVector(n::Integer, nzind::ReadOnly{Ti,1,Vector{Ti}}, nzval::Vector{Tv}) where {Tv,Ti<:Integer} =
    FixedSparseVector{Tv,Ti}(n, nzind, nzval)

FixedSparseVector(n::Integer, nzind::Vector{<:Integer}, nzval::Vector) =
    FixedSparseVector(n, ReadOnly(nzind), nzval)

FixedSparseVector(s::AbstractSparseVector) = FixedSparseVector(length(s), copy(nonzeroinds(s)), copy(nonzeros(s)))
FixedSparseVector{Tv,Ti}(s::AbstractSparseVector) where {Tv,Ti} =
    FixedSparseVector{Tv,Ti}(length(s), ReadOnly(Vector{Ti}(nonzeroinds(s))), Vector{Tv}(nonzeros(s)))

"""
inverse of fixed, should not allocate
"""
_unsafe_unfix(s::AbstractSparseVector) = s
_unsafe_unfix(s::FixedSparseVector) = SparseVector(length(s), parent(nonzeroinds(s)), nonzeros(s))

fixed(x::AbstractSparseVector) = FixedSparseVector(x)
move_fixed(x::AbstractSparseVector) = FixedSparseVector(length(x), nonzeroinds(x), nonzeros(x))

# see `_copyto_fixed!` for matrices
function _copyto_fixed!(A::AbstractCompressedVector, B::AbstractCompressedVector)
    length(A) == length(B) || throw(DimensionMismatch(lazy"cannot copy a vector of length $(length(B)) into a fixed one of length $(length(A))"))
    Ai, Bi, Anz, Bnz = nonzeroinds(A), nonzeroinds(B), nonzeros(A), nonzeros(B)
    for write in (false, true)
        write && fill!(Anz, zero(eltype(A)))
        k = 1
        @inbounds for p in eachindex(Bi)
            i = Bi[p]
            while k <= length(Ai) && Ai[k] < i; k += 1; end
            (k <= length(Ai) && Ai[k] == i) || _throwfixedinsert(A, i)
            write && (Anz[k] = Bnz[p])
        end
    end
    return A
end
