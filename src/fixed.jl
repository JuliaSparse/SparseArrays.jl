# This file is a part of Julia. License is MIT: https://julialang.org/license

# The experimental fixed-pattern arrays: the read-only index vector behind them, the types,
# `fixed`/`move_fixed`, and the kernels that exist only for them. A fixed array is a
# destination whose pattern never changes. As an operand it reads as the plain array that
# shares its buffers and the result is a plain array, so only kernels that write their
# argument check for one, and those checks stay with the kernels.

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
@noinline _throwfixedinsert(A, I::Vararg{Integer,N}) where {N} =
    throw(ArgumentError("cannot store a new entry at ($(join(I, ", "))) in a $(nameof(typeof(A))), its sparsity pattern is read-only"))
# `f!` exists to change the pattern of its argument, which a fixed array does not allow
@noinline _throwfixedchange(A, f!::Symbol) =
    throw(ArgumentError("cannot `$(f!)` a $(nameof(typeof(A))): it would change the sparsity pattern, which is read-only; apply it to a writable copy, `$(A isa AbstractVector ? :SparseVector : :SparseMatrixCSC)(A)`"))
@inline _checkpatternwritable(A, f!::Symbol) = _is_fixed(A) && _throwfixedchange(A, f!)
# `copy` of a fixed array is fixed and shares its pattern; an out-of-place kernel that
# goes on to change the pattern of its copy starts from this one
_writablecopy(A::Union{AbstractSparseMatrixCSC,AbstractSparseVector}) = copy(A)

"""
    FixedSparseCSC{Tv,Ti<:Integer} <: AbstractSparseMatrixCSC{Tv,Ti}

Experimental. A sparse matrix in CSC format whose sparsity pattern is read-only: stored
values can be overwritten, but no entry can be added or removed.

A `FixedSparseCSC` is a destination for in-place writes. As an operand it behaves as the
[`SparseMatrixCSC`](@ref) with the same entries, and results are not fixed; only `copy`
returns a `FixedSparseCSC`, which shares the pattern. See
[Sparse arrays with a fixed pattern](@ref man-sparse-fixed).
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
# These copy `x`: a pattern shared with a writable `x` could be changed through it.
# `move_fixed` reuses the buffers.
FixedSparseCSC(x::AbstractSparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = FixedSparseCSC{Tv,Ti}(x)
# a fixed `x` shares its read-only pattern, as with `copy`
FixedSparseCSC{Tv,Ti}(x::FixedSparseCSC{Tv,Ti}) where {Tv,Ti} = copy(x)
function FixedSparseCSC{Tv,Ti}(x::AbstractSparseMatrixCSC) where {Tv,Ti}
    y = _unsafe_unfix(x)
    FixedSparseCSC{Tv,Ti}(size(y, 1), size(y, 2),
        Vector{Ti}(getcolptr(y)), Vector{Ti}(getrowval(y)), Vector{Tv}(getnzval(y)))
end

"""
    fixed(x...)

Experimental. Like `sparse` but returns a sparse array whose sparsity pattern is read-only:
stored entries can change value, but none can be added or removed.

`fixed(A)` of a writable sparse array `A` copies it, so the result shares no memory with
`A`. Use `SparseArrays.move_fixed` to reuse the buffers of `A` instead.
`fixed(F)` of a fixed array is `copy(F)`: it copies the values and shares the pattern.

A write into the result keeps every stored entry and throws an `ArgumentError`, before
changing anything, when it needs an entry outside the pattern. A function whose purpose is
to change the pattern, such as `dropzeros!` or `tril!`, throws. Operations that read the
result return ordinary arrays: of the out-of-place functions only `copy` returns a fixed
array, sharing the pattern. See [Sparse arrays with a fixed pattern](@ref man-sparse-fixed).

# Examples
```jldoctest
julia> W = SparseArrays.fixed(sparse([1.0 2; 0 3]));

julia> W .= 0 .* W
2×2 SparseArrays.FixedSparseCSC{Float64, Int64} with 3 stored entries:
 0.0  0.0
  ⋅   0.0

julia> W[2, 1] = 1.0
ERROR: ArgumentError: cannot store a new entry at (2, 1) in a FixedSparseCSC, its sparsity pattern is read-only
[...]

julia> typeof(W + W)
SparseMatrixCSC{Float64, Int64}
```
"""
fixed(x...) = move_fixed(sparse(x...))
fixed(x::AbstractSparseMatrixCSC) = FixedSparseCSC(x)

"""
    move_fixed(x::AbstractSparseMatrixCSC)

Experimental, unsafe. Make a `FixedSparseCSC` by reusing the colptr, rowvals and nonzeros of `x`.
The pattern of the result is read-only only as long as `x` is not used to change it.
"""
move_fixed(x::AbstractSparseMatrixCSC) = FixedSparseCSC(size(x)..., getcolptr(x), getrowval(x), getnzval(x))
"""
    _unsafe_unfix(x)

Experimental, unsafe. Returns a modifiable version of `x` for compatibility with this codebase.
"""
_unsafe_unfix(x::FixedSparseCSC) = SparseMatrixCSC(size(x)..., parent(getcolptr(x)), parent(getrowval(x)), getnzval(x))
_unsafe_unfix(x::AbstractSparseMatrixCSC) = x
_writablecopy(A::FixedSparseCSC) = SparseMatrixCSC(A)

# A fixed destination keeps its pattern: A's entries that B does not store become zero, and
# an entry of B outside the pattern must be one that `setindex!` would not store. The
# pattern and the values are checked in full before anything is written.
function _copyto_fixed!(A::AbstractSparseMatrixCSC, B::AbstractSparseMatrixCSC)
    size(A) == size(B) || throw(DimensionMismatch(lazy"cannot copy a matrix of size $(size(B)) into a fixed one of size $(size(A))"))
    Arv, Brv, Anz, Bnz = getrowval(A), getrowval(B), getnzval(A), getnzval(B)
    _checkconvertible(eltype(A), Bnz)
    for write in (false, true)
        if write
            # B's pattern lies in A's, so with one value buffer the two patterns are equal
            Anz === Bnz && return A
            Bnz = Base.unalias(Anz, Bnz)
            fill!(Anz, zero(eltype(A)))
        end
        @inbounds for j in axes(A, 2)
            k, kend = Int(first(nzrange(A, j))), Int(last(nzrange(A, j)))
            for p in nzrange(B, j)
                i = Brv[p]
                while k <= kend && Arv[k] < i; k += 1; end
                if !(k <= kend && Arv[k] == i)
                    _isimplicitzero(convert(eltype(A), Bnz[p]), eltype(A)) || _throwfixedinsert(A, i, j)
                    continue
                end
                write && (Anz[k] = Bnz[p])
            end
        end
    end
    return A
end

"""
    FixedSparseVector{Tv,Ti<:Integer} <: AbstractCompressedVector{Tv,Ti}

Experimental. A sparse vector whose sparsity pattern is read-only: stored values can be
overwritten, but no entry can be added or removed.

It follows the rules of [`SparseArrays.FixedSparseCSC`](@ref): a destination for in-place
writes, an ordinary [`SparseVector`](@ref) as an operand.
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
# a fixed `s` shares its read-only pattern, as with `copy`
FixedSparseVector(s::FixedSparseVector) = copy(s)
FixedSparseVector{Tv,Ti}(s::FixedSparseVector{Tv,Ti}) where {Tv,Ti} = copy(s)
FixedSparseVector{Tv,Ti}(s::AbstractSparseVector) where {Tv,Ti} =
    FixedSparseVector{Tv,Ti}(length(s), ReadOnly(Vector{Ti}(nonzeroinds(s))), Vector{Tv}(nonzeros(s)))

"""
inverse of fixed, should not allocate
"""
_unsafe_unfix(s::AbstractSparseVector) = s
_unsafe_unfix(s::FixedSparseVector) = SparseVector(length(s), parent(nonzeroinds(s)), nonzeros(s))
_writablecopy(s::FixedSparseVector) = copy(_unsafe_unfix(s))

fixed(x::AbstractSparseVector) = FixedSparseVector(x)
move_fixed(x::AbstractSparseVector) = FixedSparseVector(length(x), nonzeroinds(x), nonzeros(x))

# see `_copyto_fixed!` for matrices
function _copyto_fixed!(A::AbstractCompressedVector, B::AbstractCompressedVector)
    length(A) == length(B) || throw(DimensionMismatch(lazy"cannot copy a vector of length $(length(B)) into a fixed one of length $(length(A))"))
    Ai, Bi, Anz, Bnz = nonzeroinds(A), nonzeroinds(B), nonzeros(A), nonzeros(B)
    _checkconvertible(eltype(A), Bnz)
    for write in (false, true)
        if write
            Anz === Bnz && return A
            Bnz = Base.unalias(Anz, Bnz)
            fill!(Anz, zero(eltype(A)))
        end
        k = 1
        @inbounds for p in eachindex(Bi)
            i = Bi[p]
            while k <= length(Ai) && Ai[k] < i; k += 1; end
            if !(k <= length(Ai) && Ai[k] == i)
                _isimplicitzero(convert(eltype(A), Bnz[p]), eltype(A)) || _throwfixedinsert(A, i)
                continue
            end
            write && (Anz[k] = Bnz[p])
        end
    end
    return A
end

Base.setindex!(x::FixedSparseVector, v::AbstractVector, ::Colon) = setindex!(x, v, axes(x, 1))
# Base's methods never reach a sparse kernel that could check: `circshift!` goes through
# `reverse!` and `permute!` assigns a permuted copy
Base.circshift!(x::FixedSparseVector, ::Integer) = _throwfixedchange(x, :circshift!)
Base.permute!(x::FixedSparseVector, ::AbstractVector) = _throwfixedchange(x, :permute!)
Base.invpermute!(x::FixedSparseVector, ::AbstractVector) = _throwfixedchange(x, :invpermute!)
sparsevec(x::FixedSparseVector) = _writablecopy(x)
