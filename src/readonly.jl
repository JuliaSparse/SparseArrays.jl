"""
   `ReadOnly{T,N<:AbstractArray{T,N,V<:AbstractArray{T,N}}} <: AbstractArray{T,N}`

Internal. Wrapper around an abstract vector, blocks changing elements.
`setindex!(x, getindex(x, i...), i...) ` does not error. Resizing
a vector to its original length will not throw an error either.
"""
struct ReadOnly{T,N,V<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::V
end
# ReadOnly of ReadOnly is meaningless
ReadOnly(x::ReadOnly) = x
Base.getproperty(::ReadOnly, ::Symbol) = error("Use parent instead.")
@inline Base.parent(x::ReadOnly) = getfield(x, :parent)

for i in [:length, :first, :last, :eachindex, :firstindex, :lastindex, :eltype]
    @eval Base.@propagate_inbounds @inline Base.$i(x::ReadOnly) = Base.$i(parent(x))
end
for i in [:iterate, :axes, :getindex, :size, :strides]
    @eval(Base.@propagate_inbounds @inline Base.$i(x::ReadOnly, y...) = Base.$i(parent(x), y...))
end

Base.unsafe_convert(x::Type{Ptr{T}}, A::ReadOnly) where T = Base.unsafe_convert(x, parent(A))
Base.elsize(::Type{ReadOnly{T,N,V}}) where {T,N,V} = Base.elsize(V)
Base.@propagate_inbounds @inline Base.setindex!(x::ReadOnly, v, ind...) = if v == getindex(parent(x), ind...)
        v
    else
        error("Can't change $(typeof(x)).")
    end
for i in [:IteratorSize, :IndexStyle]
    @eval(@inline Base.$i(::Type{ReadOnly{T,N,V}}) where {T,N,V} = Base.$i(V))
end
@inline Base.resize!(x::ReadOnly, l) = l == length(parent(x)) ? x : error("can't resize $(typeof(x))")
Base.copy(x::ReadOnly) = ReadOnly(copy(parent(x)))
(==)(x::ReadOnly, y::AbstractVector) = parent(x) == y
(==)(x::AbstractVector, y::ReadOnly) = x == parent(y)
(==)(x::ReadOnly, y::ReadOnly) = parent(x) == parent(y)

Base.dataids(S::ReadOnly) = tuple()