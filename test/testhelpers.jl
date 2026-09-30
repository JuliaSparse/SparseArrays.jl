# This file is a part of Julia. License is MIT: https://julialang.org/license

# Helpers shared by the test suites. This is a plain definition file, not a module:
# every suite module `include`s it at its top so that all suites run under the same
# accessor guard and share one set of fixtures. `ambiguous.jl` does not include it,
# so that the `getproperty` overrides below do not show up in its piracy report.

using Test
using LinearAlgebra: LinearAlgebra
using SparseArrays: SparseArrays, SparseMatrixCSC, SparseVector, AbstractSparseMatrixCSC,
    AbstractSparseVector, FixedSparseCSC, FixedSparseVector, getcolptr, rowvals, nonzeros,
    nonzeroinds

# Field access on the sparse types is an error under test, so that kernels and tests go
# through the accessors. `ReadOnly` has a `getproperty` of its own and stays out. Every
# suite module includes this file, and each definition is skipped when an earlier include
# already installed it: `hasmethod` cannot tell, because the generic `Base.getproperty`
# fallback for `Any` always applies, so the check is whether `which` still returns that
# fallback.
let fallback = which(Base.getproperty, Tuple{Any, Symbol})
    for T in (SparseMatrixCSC, SparseVector, FixedSparseCSC, FixedSparseVector)
        if which(Base.getproperty, Tuple{T, Symbol}) === fallback
            @eval Base.getproperty(::$T, ::Symbol) = error("use accessor function")
        end
    end
end

# Whether sparse arrays have the same shape and stored pattern, whatever their values.
same_pattern(A::AbstractSparseMatrixCSC, B::AbstractSparseMatrixCSC) =
    size(A) == size(B) && getcolptr(A) == getcolptr(B) && rowvals(A) == rowvals(B)
same_pattern(x::AbstractSparseVector, y::AbstractSparseVector) =
    length(x) == length(y) && nonzeroinds(x) == nonzeroinds(y)
same_pattern(A, B, C...) = same_pattern(A, B) && same_pattern(B, C...)

# Whether two sparse vectors agree in eltype, index type, pattern and stored values,
# stored zeros included.
exact_equal(x::AbstractSparseVector, y::AbstractSparseVector) =
    eltype(x) == eltype(y) && eltype(nonzeroinds(x)) == eltype(nonzeroinds(y)) &&
    same_pattern(x, y) && nonzeros(x) == nonzeros(y)

# An `AbstractSparseVector` that is not an `AbstractCompressedVector`, so that the generic
# sparse-vector paths are reached rather than the compressed-vector specializations.
struct WrappedSparseVector{Tv,Ti} <: SparseArrays.AbstractSparseVector{Tv,Ti}
    x::SparseVector{Tv,Ti}
end
Base.size(v::WrappedSparseVector) = size(v.x)
Base.getindex(v::WrappedSparseVector, i::Int) = v.x[i]
SparseArrays.nonzeros(v::WrappedSparseVector) = SparseArrays.nonzeros(v.x)
SparseArrays.nonzeroinds(v::WrappedSparseVector) = SparseArrays.nonzeroinds(v.x)

# An eltype that counts its scalar multiplications and comparisons, so that a kernel
# touching only the stored entries and a generic fallback visiting every element are told
# apart by an operation count rather than by timing.
struct OpCount{T} <: Number
    x::T
end
OpCount{T}(a::OpCount{T}) where {T} = a
const MULCOUNT = Ref(0)
const EQCOUNT = Ref(0)
Base.:*(a::OpCount, b::OpCount) = (MULCOUNT[] += 1; OpCount(a.x * b.x))
Base.:(==)(a::OpCount, b::OpCount) = (EQCOUNT[] += 1; a.x == b.x)
Base.isequal(a::OpCount, b::OpCount) = (EQCOUNT[] += 1; isequal(a.x, b.x))
Base.:+(a::OpCount, b::OpCount) = OpCount(a.x + b.x)
Base.:-(a::OpCount, b::OpCount) = OpCount(a.x - b.x)
Base.:-(a::OpCount) = OpCount(-a.x)
Base.zero(::Type{OpCount{T}}) where {T} = OpCount(zero(T))
Base.zero(a::OpCount) = zero(typeof(a))
Base.one(::Type{OpCount{T}}) where {T} = OpCount(one(T))
Base.conj(a::OpCount) = OpCount(conj(a.x))
Base.adjoint(a::OpCount) = conj(a)
Base.transpose(a::OpCount) = a
Base.iszero(a::OpCount) = iszero(a.x)
Base.isone(a::OpCount) = isone(a.x)
Base.promote_rule(::Type{OpCount{T}}, ::Type{OpCount{U}}) where {T,U} = OpCount{promote_type(T, U)}
# the number of scalar multiplications, or of `==`/`isequal` comparisons, performed by `f()`
mulcount(f) = (MULCOUNT[] = 0; f(); MULCOUNT[])
eqcount(f) = (EQCOUNT[] = 0; f(); EQCOUNT[])
# the same sparse array with `OpCount` entries
opcount_sparse(S::SparseMatrixCSC) =
    SparseMatrixCSC(size(S)..., copy(getcolptr(S)), copy(rowvals(S)), OpCount.(nonzeros(S)))
opcount_sparse(x::SparseVector) =
    SparseVector(length(x), copy(nonzeroinds(x)), OpCount.(nonzeros(x)))

# An array that counts its element reads, to tell a kernel that reads each element once
# from one that rereads it for every stored entry.
struct CountedReads{T,N} <: AbstractArray{T,N}
    parent::Array{T,N}
    reads::Base.RefValue{Int}
end
CountedReads(A::AbstractArray) = CountedReads(Array(A), Ref(0))
Base.size(c::CountedReads) = size(c.parent)
Base.IndexStyle(::Type{<:CountedReads}) = IndexLinear()
Base.getindex(c::CountedReads, i::Int) = (c.reads[] += 1; c.parent[i])

# Whether inference gives some local or SSA value of `f(::types...)` a `Union` that
# contains both `T1` and `T2`. A counter seeded from an `Int32` index array and then
# incremented with an `Int` literal is the usual way such a union appears; the public
# call still infers, so `@inferred` cannot see it.
function hasunionlocal(f, types, T1, T2)
    ci, _ = only(Base.code_typed(f, types; optimize=false))
    slots = ci.slottypes === nothing ? Any[] : ci.slottypes
    any(T -> T isa Union && T1 <: T && T2 <: T, Iterators.flatten((slots, ci.ssavaluetypes)))
end

# Base's Quaternions.jl test helper, a noncommutative eltype. It is loaded into `Main`
# once, so that every suite in a process sees the same type, and at top level, because
# a test that loaded it itself would run in a world where the type is not yet defined.
isdefined(Main, :Quaternions) ||
    Base.include(Main, joinpath(Sys.BINDIR, "..", "share", "julia", "test", "testhelpers", "Quaternions.jl"))
quaternion_type() = Main.Quaternions.Quaternion

# A statically sized matrix eltype, in the spirit of StaticArrays, whose products and sums
# fix the result size in the type.
struct SimpleSMatrix{N,M,T} <: AbstractMatrix{T}
    m::Matrix{T}
end

SimpleSMatrix{N,M}(m::AbstractMatrix{T}) where {N,M,T} =
    size(m) == (N, M) ? SimpleSMatrix{N,M,T}(m) : throw(error("Wrong matrix size"))

Base.:*(a::SimpleSMatrix{N,O}, b::SimpleSMatrix{O,M}) where {N,O,M} =
    SimpleSMatrix{N,M}(a.m * b.m)

Base.:*(a::LinearAlgebra.Adjoint{<:Any, <:SimpleSMatrix{O,N}}, b::SimpleSMatrix{O,M}) where {N,O,M} =
    SimpleSMatrix{N,M}(adjoint(parent(a).m) * b.m)

Base.:*(a::SimpleSMatrix{N,O}, b::LinearAlgebra.Adjoint{<:Any, <:SimpleSMatrix{M,O}}) where {N,O,M} =
    SimpleSMatrix{N,M}(a.m * adjoint(parent(b).m))

Base.:+(a::SimpleSMatrix{N,M}, b::SimpleSMatrix{N,M}) where {N,M} =
    SimpleSMatrix{N,M}(a.m + b.m)

Base.:+(a::LinearAlgebra.Adjoint{<:Any, <:SimpleSMatrix{M,N}}, b::SimpleSMatrix{N,M}) where {N,M} =
    SimpleSMatrix{N,M}(adjoint(parent(a).m) + b.m)

Base.:+(a::LinearAlgebra.Adjoint{<:Any, <:SimpleSMatrix}, b::LinearAlgebra.Adjoint{<:Any, <:SimpleSMatrix}) =
    (a' + b')'

Base.:+(a::SimpleSMatrix{N,M}, b::LinearAlgebra.Adjoint{<:Any, <:SimpleSMatrix{M,N}}) where {N,M} =
    SimpleSMatrix{N,M}(a.m + adjoint(parent(b).m))

Base.:-(a::SimpleSMatrix{N,M}, b::SimpleSMatrix{N,M}) where {N,M} =
    SimpleSMatrix{N,M}(a.m - b.m)

Base.:-(a::LinearAlgebra.Adjoint{<:Any, <:SimpleSMatrix{M,N}}, b::SimpleSMatrix{N,M}) where {N,M} =
    SimpleSMatrix{N,M}(adjoint(parent(a).m) - b.m)

Base.:-(a::LinearAlgebra.Adjoint{<:Any, <:SimpleSMatrix}, b::LinearAlgebra.Adjoint{<:Any, <:SimpleSMatrix}) =
    (a' - b')'

Base.:-(a::SimpleSMatrix{N,M}, b::LinearAlgebra.Adjoint{<:Any, <:SimpleSMatrix{M,N}}) where {N,M} =
    SimpleSMatrix{N,M}(a.m - adjoint(parent(b).m))

Base.size(a::SimpleSMatrix{N,M}) where {N,M} = (N, M)

Base.length(a::SimpleSMatrix{N,M}) where {N,M} = N * M

Base.zero(::Type{S}) where {N,M,T,S<:SimpleSMatrix{N,M,T}} = SimpleSMatrix{N,M}(zeros(T, N, M))

Base.getindex(s::SimpleSMatrix, inds...) = getindex(s.m, inds...)

Base.convert(::Type{S}, value::Matrix) where {N,M,T,S<:SimpleSMatrix{N,M,T}} =
    SimpleSMatrix{N,M}(T.(value))
