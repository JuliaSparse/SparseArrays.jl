# This file is a part of Julia. License is MIT: https://julialang.org/license

# The types, helpers and constants shared by the test suites. `testhelpers.jl` loads this
# module into `Main` once per process, so that every suite sees the same types and a
# kernel compiled for one of them in one suite is reused by the next. A type or helper a
# test needs is defined here, not in the suite.
module SparseTestHelpers

export COMPREHENSIVE, STD_ELTYPES, itypes, core_itypes, eachvalue, pairwise,
    same_pattern, exact_equal, WrappedSparseVector, OpCount, mulcount, eqcount,
    opcount_sparse, CountedReads, hasunionlocal, quaternion_type, SimpleSMatrix,
    NonCSCSparse, ConcatArray, AllBut, MockTropical, Variable, Expression, Meters,
    OneSided, Tagged, CustomType, UndefElt, Positive,
    check_trisolve, check_scalar_broadcast,
    show_plain, show_contents,
    mismatch, fixture, fixturevec, fixturepair, fixturedense, fixturestrided, FIXTURE_SHAPES

using Test
using LinearAlgebra: LinearAlgebra
using SparseArrays: SparseArrays, SparseMatrixCSC, SparseVector, AbstractSparseMatrix,
    AbstractSparseMatrixCSC, AbstractSparseVector, FixedSparseCSC, FixedSparseVector,
    getcolptr, rowvals, nonzeros, nonzeroinds

# Field access on the sparse types is an error under test, so that kernels and tests go
# through the accessors. `ReadOnly` has a `getproperty` of its own and stays out.
for T in (SparseMatrixCSC, SparseVector, FixedSparseCSC, FixedSparseVector)
    @eval Base.getproperty(::$T, ::Symbol) = error("use accessor function")
end

# Whether the comprehensive tests run as well: the issue regressions and the wider corner
# cases, which the suites guard with `@static if COMPREHENSIVE`, so that a standard run
# does not even lower them. Set by `SPARSEARRAYS_TEST_COMPREHENSIVE=true` in the
# environment, which `runtests.jl` also sets for the `--comprehensive` argument.
const COMPREHENSIVE = get(ENV, "SPARSEARRAYS_TEST_COMPREHENSIVE", "false") == "true"

# The element types of standard mode. Test time is compilation, and one process runs
# every suite, so a kernel compiled for these types in one suite is reused by the others.
# Another type appears in standard mode only where that type is the point of the test.
const STD_ELTYPES = (Float64, ComplexF64)
# The C index types of the SuiteSparse solvers, and the one a build's `Int` selects.
const itypes = sizeof(Int) == 4 ? (Int32,) : (Int32, Int64)
const core_itypes = sizeof(Int) == 4 ? (Int32,) : (Int64,)

# A subset of the Cartesian product of `dims` in which every value of every dimension
# appears, as tuples: as many cases as the longest dimension, the shorter ones cycling.
function eachvalue(dims...)
    cols = map(d -> collect(Any, d), dims)
    n = maximum(length, cols)
    Any[map(c -> c[mod1(k, length(c))], cols) for k in 1:n]
end

# A subset of the Cartesian product of `dims` in which every pair of values of every two
# dimensions appears together, as tuples. Chosen greedily and deterministically.
function pairwise(dims...)
    cols = map(d -> collect(Any, d), dims)
    n = length(cols)
    all = vec(collect(Iterators.product(map(eachindex, cols)...)))
    n <= 2 && return Any[map(getindex, cols, c) for c in all]
    pairs(c) = ((i, j, c[i], c[j]) for i in 1:n for j in i+1:n)
    uncovered = Set{NTuple{4,Int}}()
    foreach(c -> union!(uncovered, pairs(c)), all)
    chosen = Any[]
    while !isempty(uncovered)
        c = all[argmax([count(in(uncovered), pairs(c)) for c in all])]
        setdiff!(uncovered, pairs(c))
        push!(chosen, map(getindex, cols, c))
    end
    chosen
end

# Whether sparse arrays have the same shape and stored pattern, whatever their values.
same_pattern(@nospecialize(A::AbstractSparseMatrixCSC), @nospecialize(B::AbstractSparseMatrixCSC)) =
    size(A) == size(B) && getcolptr(A) == getcolptr(B) && rowvals(A) == rowvals(B)
same_pattern(@nospecialize(x::AbstractSparseVector), @nospecialize(y::AbstractSparseVector)) =
    length(x) == length(y) && nonzeroinds(x) == nonzeroinds(y)
same_pattern(A, B, C...) = same_pattern(A, B) && same_pattern(B, C...)

# Whether two sparse vectors agree in eltype, index type, pattern and stored values,
# stored zeros included.
exact_equal(@nospecialize(x::AbstractSparseVector), @nospecialize(y::AbstractSparseVector)) =
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
SparseArrays.nnz(v::WrappedSparseVector) = SparseArrays.nnz(v.x)

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
Base.:/(a::OpCount, b::OpCount) = OpCount(a.x / b.x)
Base.abs(a::OpCount) = OpCount(abs(a.x))
Base.isless(a::OpCount, b::OpCount) = isless(a.x, b.x)
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
function hasunionlocal(@nospecialize(f), @nospecialize(types), @nospecialize(T1), @nospecialize(T2))
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

# A sparse matrix type that is not CSC, with nothing but `size` and scalar `getindex`.
struct NonCSCSparse{Tv,Ti} <: AbstractSparseMatrix{Tv,Ti}
    A::SparseMatrixCSC{Tv,Ti}
end
Base.size(S::NonCSCSparse) = size(S.A)
Base.getindex(S::NonCSCSparse, i::Int, j::Int) = S.A[i, j]

# An array type from another package that owns the `vcat`/`hcat`/`hvcat` of its own
# arrays with anything.
struct ConcatArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end
ConcatArray(x::AbstractArray) = ConcatArray(Array(x))
Base.size(x::ConcatArray) = size(x.data)
Base.getindex(x::ConcatArray, i::Int) = x.data[i]
Base.IndexStyle(::Type{<:ConcatArray}) = IndexLinear()
Base.vcat(x::AbstractMatrix, y::ConcatArray{<:Any,2}) = ConcatArray(vcat(x, y.data))
Base.hcat(x::AbstractMatrix, y::ConcatArray{<:Any,2}) = ConcatArray(hcat(x, y.data))
Base.hvcat(rows::Tuple{Vararg{Int}}, x::AbstractMatrix, y::ConcatArray{<:Any,2}) =
    ConcatArray(hvcat(rows, x, y.data))
Base.vcat(x::AbstractVector, y::ConcatArray{<:Any,1}) = ConcatArray(vcat(x, y.data))
Base.hcat(x::AbstractVector, y::ConcatArray{<:Any,1}) = ConcatArray(hcat(x, y.data))

# An index type lowered by `to_indices`, like `InvertedIndices.Not`.
struct AllBut; i::Int; end
Base.to_indices(A, inds, I::Tuple{AllBut,Vararg}) =
    (setdiff(inds[1], I[1].i), to_indices(A, Base.tail(inds), Base.tail(I))...)

# A semiring eltype whose `zero` is not the additive identity of its field.
struct MockTropical{T} <: Number
    n::T
end
MockTropical{T}(x::MockTropical{T}) where {T} = x
Base.zero(::Type{MockTropical{T}}) where {T} = MockTropical{T}(typemin(T))
Base.zero(x::MockTropical{T}) where {T} = zero(MockTropical{T})
Base.one(::Type{MockTropical{T}}) where {T} = MockTropical{T}(zero(T))
Base.one(x::MockTropical{T}) where {T} = one(MockTropical{T})
Base.:*(a::MockTropical{T}, b::MockTropical{T}) where {T} = MockTropical{T}(a.n + b.n)
Base.:+(a::MockTropical{T}, b::MockTropical{T}) where {T} = MockTropical{T}(max(a.n, b.n))

# An element type whose zero, sums and differences are of another type.
struct Variable
    x::Int
end
mutable struct Expression
    x::Int
end
Base.:(==)(a::Expression, b::Expression) = a.x == b.x
Base.zero(::Type{Variable}) = Expression(0)
Base.convert(::Type{Expression}, v::Variable) = Expression(v.x)
Base.promote_rule(::Type{Variable}, ::Type{Expression}) = Expression
Base.transpose(a::Union{Variable,Expression}) = a
for op in (:+, :-)
    @eval Base.$op(a::Union{Variable,Expression}, b::Int) = Expression($op(a.x, b))
    @eval Base.$op(a::Int, b::Union{Variable,Expression}) = Expression($op(a, b.x))
end

# A quantity with a unit: `one` is the dimensionless identity, `oneunit` keeps the unit.
struct Meters <: Number
    x::Int
    Meters(x::Int) = new(x)
end
Base.zero(::Type{Meters}) = Meters(0)
Base.one(::Type{Meters}) = 1

# An element type that equals a number from the left only, as an affine expression of an
# optimization model does.
struct OneSided
    x::Int
end
Base.zero(::Type{OneSided}) = OneSided(0)
Base.:(==)(a::OneSided, b::Number) = a.x == b
Base.transpose(a::OneSided) = a

# A number whose zero keeps the unit of a value, so that the type alone has none.
struct Tagged <: Number
    x::Int
    unit::Symbol
end
Base.zero(a::Tagged) = Tagged(0, a.unit)
Base.:(==)(a::Tagged, b::Tagged) = a.x == b.x && a.unit == b.unit

# A non-numeric eltype with a `zero` and an order.
struct CustomType
    x::String
end
Base.zero(::Type{CustomType}) = CustomType("")
Base.zero(x::CustomType) = zero(CustomType)
Base.isless(x::CustomType, y::CustomType) = isless(x.x, y.x)

# A non-isbits eltype, so that uninitialized storage holds `#undef`.
mutable struct UndefElt end
Base.zero(::Type{UndefElt}) = UndefElt()
Base.zero(x::UndefElt) = UndefElt()

# A callable that is not a `Function`.
struct Positive end
(::Positive)(x) = x > 0

# `mat \ spvec` and, where the eltype allows solving in place, `ldiv!(mat, spvec)` for a
# triangular `mat` and a sparse vector, against the solves with the dense vector. A unit
# triangle does not divide, so an integer solve stays integer.
function check_trisolve(@nospecialize(mat), @nospecialize(spvec))
    fspvec = Array(spvec)
    T = typeof(zero(eltype(mat))*zero(eltype(spvec)) + zero(eltype(mat))*zero(eltype(spvec)))
    if !(mat isa Union{LinearAlgebra.UnitLowerTriangular,LinearAlgebra.UnitUpperTriangular})
        T = typeof(zero(T)/one(eltype(mat)))
    end
    # a sparse triangle gives a sparse solution, a dense one a dense solution
    R = SparseArrays.issparse(mat) ? SparseArrays.SparseVector{T,SparseArrays.indtype(spvec)} : Vector{T}
    @test (mat \ spvec)::R ≈ mat \ fspvec
    if eltype(spvec) == T
        @test LinearAlgebra.ldiv!(mat, copy(spvec)) ≈ LinearAlgebra.ldiv!(mat, copy(fspvec))
    end
end

# `broadcast` and `broadcast!` of `f` over `sparseargs`, a mix of scalars and sparse arrays,
# against the dense result: value, type, inference, and the allocation of the in-place form.
# Specialized on its arguments, which `@inferred` and `@allocated` need.
function check_scalar_broadcast(f, sparseargs, alloc_limit=1028)
    denseargs = map(x -> x isa AbstractArray ? Array(x) : x, sparseargs)
    fX = broadcast(f, denseargs...)
    X = @inferred broadcast(f, sparseargs...)
    @test X == SparseArrays.sparse(fX)
    @test typeof(X) === typeof(SparseArrays.sparse(fX))
    @test (@inferred broadcast!(f, X, sparseargs...)) === X
    @test X == SparseArrays.sparse(fX)
    X = SparseArrays.sparse(fX)
    # Transposed sparse inputs require materializing CSC copies.
    extra = sum(x -> x isa LinearAlgebra.Transpose ? @allocated(SparseMatrixCSC(x)) + 128 : 0, sparseargs)
    @test (@allocated broadcast!(f, X, sparseargs...)) <= extra + alloc_limit
end

# The `text/plain` display of `X` on a limited screen, and that display without its
# summary line.
show_plain(@nospecialize(X); displaysize=(24, 80)) =
    sprint(show, "text/plain", X; context=(:limit=>true, :displaysize=>displaysize))
show_contents(@nospecialize(X); kwargs...) = last(split(show_plain(X; kwargs...), '\n'; limit=2))

# What is wrong with the stored structure of `S`, as a message, or `nothing`: the buffers
# must cover what the column pointers claim, and the indices of a column must be in range,
# sorted and unique. A kernel that gets this wrong can still compare equal to dense.
function structure_error(@nospecialize(S::AbstractSparseMatrixCSC))
    m, n = size(S)
    colptr, rowval, nzval = getcolptr(S), rowvals(S), nonzeros(S)
    length(colptr) == n + 1 || return "colptr has length $(length(colptr)) for $n columns"
    colptr[1] == 1 || return "colptr[1] is $(colptr[1])"
    for j in 1:n
        colptr[j] <= colptr[j+1] || return "colptr decreases at column $j"
    end
    stored = colptr[n+1] - 1
    length(rowval) >= stored && length(nzval) >= stored ||
        return "$stored stored entries, but $(length(rowval)) row indices and $(length(nzval)) values"
    for j in 1:n, k in colptr[j]:colptr[j+1]-1
        1 <= rowval[k] <= m || return "row index $(rowval[k]) in column $j is out of range"
        k > colptr[j] && rowval[k-1] >= rowval[k] &&
            return "the row indices of column $j are not sorted and unique"
    end
    return nothing
end
function structure_error(@nospecialize(x::AbstractSparseVector))
    inds = nonzeroinds(x)
    length(inds) == length(nonzeros(x)) ||
        return "$(length(inds)) indices, but $(length(nonzeros(x))) values"
    for k in eachindex(inds)
        1 <= inds[k] <= length(x) || return "index $(inds[k]) is out of range"
        k > 1 && inds[k-1] >= inds[k] && return "the indices are not sorted and unique"
    end
    return nothing
end

# The first way in which `S` is not a well-formed sparse result equal to the dense
# reference `D`, as a message, or `nothing`. Comparing `S == D` checks the values only: a
# result that is dense, has unsorted or repeated indices, or has the wrong element or index
# type passes it. `Tv` is the expected element type, that of `D` unless given (`nothing`
# skips the check); `Ti` the expected index type, checked when given; `approx` compares
# with `≈`. Write `@test mismatch(S, D) === nothing`, so that a failure shows the message.
function mismatch(@nospecialize(S), @nospecialize(D); Tv=eltype(D), Ti=nothing, approx::Bool=false)
    S isa Union{AbstractSparseMatrixCSC,AbstractSparseVector} ||
        return "the result is a $(typeof(S)), not a sparse array"
    size(S) == size(D) || return "size $(size(S)), expected $(size(D))"
    msg = structure_error(S)
    msg === nothing || return msg
    Tv === nothing || eltype(S) === Tv || return "eltype $(eltype(S)), expected $Tv"
    Ti === nothing || SparseArrays.indtype(S) === Ti ||
        return "index type $(SparseArrays.indtype(S)), expected $Ti"
    A = Array(S)
    (approx ? isapprox(A, D) : A == D || isequal(A, D)) ||
        return "the values differ from the dense reference"
    return nothing
end

# Fixtures for what a square random real matrix does not exercise. `fixture(T, m, n)` has
# an empty row, an empty column and a stored zero, and for a complex `T` no entry equals
# its conjugate, so that a swapped dimension, a stored zero taken for a structural one and
# a missing or extra `conj` each change a result. The values are fixed, not random.
const FIXTURE_SHAPES = ((5, 3), (3, 5), (4, 4))
fixturevalue(::Type{T}, i, j) where {T<:Complex} = T(i + j, i == 2j ? 1 : i - 2j)
fixturevalue(::Type{T}, i, j) where {T} = T(i + 3j)
function fixture(::Type{T}, m::Integer, n::Integer) where {T}
    I, J, V = Int[], Int[], T[]
    for j in 1:n, i in 1:m
        ((m > 2 && i == m - 1) || (n > 2 && j == 2) || (i + j) % 3 == 0) && continue
        push!(I, i); push!(J, j)
        push!(V, (i, j) == (1, 1) ? zero(T) : fixturevalue(T, i, j))
    end
    return SparseArrays.sparse(I, J, V, m, n)
end
# a sparse vector with gaps, an unstored last entry and a stored zero first
function fixturevec(::Type{T}, n::Integer) where {T}
    I = [i for i in 1:n-1 if i % 3 != 0]
    return SparseArrays.sparsevec(I, T[i == 1 ? zero(T) : fixturevalue(T, i, 1) for i in I], n)
end
# a fixture and its dense copy, each call a fresh pair, so that a testset owns its inputs
fixturepair(::Type{T}, m::Integer, n::Integer) where {T} = (A = fixture(T, m, n); (A, Matrix(A)))
fixturepair(::Type{T}, n::Integer) where {T} = (x = fixturevec(T, n); (x, Vector(x)))

# Fixed dense operands and right-hand sides: values of either sign, none zero and, for a
# complex `T`, none real, so that they cannot hide a dropped term or a missing `conj`.
fixturedensevalue(::Type{T}, k) where {T} = T <: Complex ? T(cos(k^2), sin(3k)) : T(cos(k^2))
fixturedense(::Type{T}, dims::Integer...) where {T} =
    reshape(T[fixturedensevalue(T, k) for k in 1:prod(dims)], dims)
# A sparse matrix that stores every `s`-th entry in column-major order, for a test that
# needs a density of `1/s` or a size that `fixture` does not suit.
fixturestrided(::Type{T}, m, n, s) where {T} =
    SparseArrays.sparse([mod1(k, m) for k in 1:s:m*n], [cld(k, m) for k in 1:s:m*n],
                        T[fixturedensevalue(T, k) for k in 1:s:m*n], m, n)

end # module
