

"""
    ReadOnly{T,<:AbstractVector{T}} <: AbstractVector{T}

Wrapper around an abstract vector, blocks changing elements (but not `setindex!`).
This is done by only throwing errors when the modifying operations changes the structure.
For instance resizing a vector to its original length will not throw an error as it is basically a no op.
"""
struct ReadOnly{T,V<:AbstractVector{T}} <: AbstractVector{T}
    parent::V
end
Base.getproperty(::ReadOnly, ::Symbol) = error("Use parent instead.")
@inline Base.parent(x::ReadOnly) = getfield(x, :parent)
Base.@propagate_inbounds @inline getindex(x::ReadOnly, v...) = getindex(parent(x), v...)
for i in [:length, :lastindex]
    Base.@propagate_inbounds @eval Base.$i(x::ReadOnly) = Base.$i(parent(x))
end
Base.@propagate_inbounds Base.setindex!(x::ReadOnly, v, ind...) = if v == getindex(parent(x), ind...)
    v
else
    error("Can't change $(typeof(x)).")
end
Base.resize!(x::ReadOnly, l) = l == length(parent(x)) ? x : error("can't resize $(typeof(x))")
Base.copy(x::ReadOnly) = ReadOnly(copy(parent(x)))
(==)(x::ReadOnly, y::AbstractVector) = parent(x) == y
(==)(x::AbstractVector, y::ReadOnly) = x == parent(y)
(==)(x::ReadOnly, y::ReadOnly) = parent(x) == parent(y)
size(x::ReadOnly) = size(parent(x))
size(x::ReadOnly, i) = size(parent(x), i)

