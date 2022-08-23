

"""
    ReadOnly{T,<:AbstractVector{T}} <: AbstractVector{T}

Wrapper around an abstract vector, blocks changing elements (but not `setindex!`).
This is done by only throwing errors when the modifying operations changes the structure.
For instance resizing a vector to its original length will not throw an error as it is basically a no op.
"""
struct ReadOnly{T,V<:AbstractVector{T}} <: AbstractVector{T}
    parent::V
end
# ReadOnly of ReadOnly is meaningless
ReadOnly(x::ReadOnly) = x
Base.getproperty(::ReadOnly, ::Symbol) = error("Use parent instead.")
@inline Base.parent(x::ReadOnly) = getfield(x, :parent)

for i in [:length, :first, :last, :eachindex, :firstindex, :lastindex, :eltype]
    @eval Base.@propagate_inbounds @inline Base.$i(x::ReadOnly) = Base.$i(parent(x))
end
for i in [:iterate, :axes, :getindex, :similar, :size, :strides]
    @eval(Base.@propagate_inbounds @inline Base.$i(x::ReadOnly, y...) = Base.$i(parent(x), y...))
end
Base.unsafe_convert(x::Type{Ptr{T}}, A::ReadOnly) where T = Base.unsafe_convert(x, parent(A))
Base.elsize(::Type{ReadOnly{T,V}}) where {T,V} = Base.elsize(V)
Base.@propagate_inbounds @inline Base.setindex!(x::ReadOnly, v, ind...) = if v == getindex(parent(x), ind...)
        v
    else
        error("Can't change $(typeof(x)).")
    end
for i in [:IteratorSize, :IndexStyle]
    @eval(@inline Base.$i(::Type{ReadOnly{T,V}}) where {T,V} = Base.$i(V))
end
@inline Base.resize!(x::ReadOnly, l) = l == length(parent(x)) ? x : error("can't resize $(typeof(x))")
Base.copy(x::ReadOnly) = ReadOnly(copy(parent(x)))
(==)(x::ReadOnly, y::AbstractVector) = parent(x) == y
(==)(x::AbstractVector, y::ReadOnly) = x == parent(y)
(==)(x::ReadOnly, y::ReadOnly) = parent(x) == parent(y)

