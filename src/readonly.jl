

"""
    ReadOnly{T,<:AbstractVector{T}} <: AbstractVector{T}

Wrapper around an abstract vector, blocks changing elements (but not `setindex!`). 
This is done by only throwing errors when the modifying operations changes the structure.
For instance resizing a vector to its original length will not throw an error as it is basically a no op.
"""
struct ReadOnly{T,V<:AbstractVector{T}} <: AbstractVector{T}
    x::V
end
@inline inner(x::ReadOnly) = x.x
getindex(x::ReadOnly, v...) = getindex(inner(x), v...)
for i in [:length, :lastindex]
    @eval Base.$i(x::ReadOnly) = Base.$i(inner(x))
end
Base.setindex!(x::ReadOnly, v, ind...) = if v == getindex(inner(x), ind...)
    v
else
    error("Can't change $(typeof(x)).")
end
Base.resize!(x::ReadOnly, l) = l == length(inner(x)) ? x : error("can't resize $(typeof(x))")
Base.copy(x::ReadOnly) = ReadOnly(copy(inner(x)))
(==)(x::ReadOnly, y::AbstractVector) = inner(x) == y
(==)(x::AbstractVector, y::ReadOnly) = x == inner(y)
(==)(x::ReadOnly, y::ReadOnly) = inner(x) == inner(y)
size(x::ReadOnly) = size(inner(x))
size(x::ReadOnly, i) = size(inner(x), i)

