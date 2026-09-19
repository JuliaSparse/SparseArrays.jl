# Deterministic replacement for wall-clock guards (see #781): an eltype that records every
# scalar multiplication, so that a kernel touching only the stored entries and a generic
# fallback visiting every element are told apart by the count rather than by timing.
struct MulCount{T} <: Number
    x::T
end
const MULCOUNT = Ref(0)
Base.:*(a::MulCount, b::MulCount) = (MULCOUNT[] += 1; MulCount(a.x * b.x))
Base.:+(a::MulCount, b::MulCount) = MulCount(a.x + b.x)
Base.:-(a::MulCount, b::MulCount) = MulCount(a.x - b.x)
Base.:-(a::MulCount) = MulCount(-a.x)
Base.zero(::Type{MulCount{T}}) where {T} = MulCount(zero(T))
Base.zero(a::MulCount) = zero(typeof(a))
Base.one(::Type{MulCount{T}}) where {T} = MulCount(one(T))
Base.conj(a::MulCount) = MulCount(conj(a.x))
Base.adjoint(a::MulCount) = conj(a)
Base.transpose(a::MulCount) = a
Base.:(==)(a::MulCount, b::MulCount) = a.x == b.x
Base.iszero(a::MulCount) = iszero(a.x)
Base.isone(a::MulCount) = isone(a.x)
Base.promote_rule(::Type{MulCount{T}}, ::Type{MulCount{U}}) where {T,U} = MulCount{promote_type(T, U)}
# number of scalar multiplications performed by `f()`
mulcount(f) = (MULCOUNT[] = 0; f(); MULCOUNT[])
# the same sparse matrix with `MulCount` entries
mulcount_sparse(S::SparseMatrixCSC) =
    SparseMatrixCSC(size(S)..., copy(getcolptr(S)), copy(rowvals(S)), MulCount.(nonzeros(S)))
