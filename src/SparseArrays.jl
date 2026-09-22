# This file is a part of Julia. License is MIT: https://julialang.org/license

"""
Support for sparse arrays. Provides `AbstractSparseArray` and subtypes.
"""
module SparseArrays

using Base: ReshapedArray, promote_op, setindex_shape_check, to_shape, tail,
    require_one_based_indexing, promote_eltype, @propagate_inbounds, &, |
using Base.Order: Forward
using LinearAlgebra
using LinearAlgebra: AdjOrTrans, AdjointFactorization, TransposeFactorization, matprod,
    AbstractQ, AdjointQ, HessenbergQ, QRCompactWYQ, QRPackedQ, LQPackedQ,
    UpperOrLowerTriangular, UnitUpperOrUnitLowerTriangular, UpperOrUnitUpperTriangular,
    LowerOrUnitLowerTriangular, HermOrSym, BiTriSym, BandedMatrix, isbanded


import Base: +, -, *, \, /, ==, zero
import Base: Matrix, Vector
import LinearAlgebra: mul!, ldiv!, rdiv!, cholesky, adjoint!, diag, eigen, dot,
    issymmetric, istril, istriu, lu, tr, transpose!, tril!, triu!, isdiag,
    cond, factorize, ishermitian, norm, opnorm, lmul!, rmul!, tril, triu,
    matop_dest, copytrito!, nonzeroinds

import Base: adjoint, argmin, argmax, Array, broadcast, circshift, circshift!, complex, Complex,
    conj, conj!, convert, copy, copy!, copyto!, count, diff, findall, findmax, findmin, findnext, findprev,
    float, getindex, imag, inv, kron, kron!, length, map, maximum, minimum, permute!, promote_rule, real,
    rot180, rotl90, rotr90, setindex!, show, similar, size, sum, transpose,
    vcat, hcat, hvcat, cat, vec, reverse, reverse!

using Random: default_rng, AbstractRNG, randsubseq, randsubseq!

export AbstractSparseArray, AbstractSparseMatrix, AbstractSparseVector, AbstractSparseMatrixCSC,
    SparseMatrixCSC, SparseVector, blockdiag, droptol!, dropzeros!, dropzeros,
    issparse, nonzeros, nzrange, rowvals, sparse, sparsevec, spdiagm,
    sprand, sprandn, spzeros, nnz, indtype, permute, findnz,  fkeep!, ftranspose!,
    sparse_hcat, sparse_vcat, sparse_hvcat, getcolptr, getrowval, getnzval

public sparse!, spzeros!

# helper function needed in sparsematrix, sparsevector and higherorderfns
# `iszero` and `!iszero` don't guarantee to return a boolean but we need one that does
# to remove the handle the structure of the array.
@inline _iszero(x) = iszero(x) === true
@inline _iszero(x::Number) = iszero(x)
@inline _iszero(x::AbstractArray) = iszero(x)
@inline _isnotzero(x) = iszero(x) !== true # like `!iszero(x)`, but handles `x::Missing`
@inline _isnotzero(x::Number) = !iszero(x)
@inline _isnotzero(x::AbstractArray) = !iszero(x)

## Functions to switch to 0-based indexing to call external sparse solvers

# Convert from 1-based to 0-based indices
function decrement!(A::AbstractArray{T}) where T<:Integer
    for i in eachindex(A); A[i] -= oneunit(T) end
    A
end
decrement(A::AbstractArray) = let y = Array(A)
    y .= y .- oneunit(eltype(A))
end

"""
    AbstractSparseArray{Tv,Ti,N}

Supertype for `N`-dimensional sparse arrays (or array-like types) with elements
of type `Tv` and index type `Ti`. [`SparseMatrixCSC`](@ref), [`SparseVector`](@ref)
and `SuiteSparse.CHOLMOD.Sparse` are subtypes of this.
"""
abstract type AbstractSparseArray{Tv,Ti,N} <: AbstractArray{Tv,N} end

"""
    AbstractSparseVector{Tv,Ti}

Supertype for one-dimensional sparse arrays (or array-like types) with elements
of type `Tv` and index type `Ti`. Alias for `AbstractSparseArray{Tv,Ti,1}`.
"""
const AbstractSparseVector{Tv,Ti} = AbstractSparseArray{Tv,Ti,1}

"""
    AbstractCompressedVector{Tv,Ti}

Supertype for vectors stored using a compressed map.
"""
abstract type AbstractCompressedVector{Tv,Ti} <: AbstractSparseVector{Tv,Ti} end

"""
    AbstractSparseMatrix{Tv,Ti}

Supertype for two-dimensional sparse arrays (or array-like types) with elements
of type `Tv` and index type `Ti`. Alias for `AbstractSparseArray{Tv,Ti,2}`.
"""
const AbstractSparseMatrix{Tv,Ti} = AbstractSparseArray{Tv,Ti,2}

"""
    AbstractSparseMatrixCSC{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}

Supertype for matrix with compressed sparse column (CSC).
"""
abstract type AbstractSparseMatrixCSC{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti} end

# ---- Type aliases used for dispatch across files ----
# Alias names use the Sparse family name and accept the abstract tier of the type they
# are built on; an alias restricted to the concrete types says so in its name.
# (Aliases for solver scalar types live with the solvers.)

const AbstractSparseVecOrMat = Union{AbstractSparseVector,AbstractSparseMatrix}
# types exposing compressed storage via nonzeros, rowvals/nonzeroinds and nzrange
const SparseVecOrMat = Union{AbstractCompressedVector,AbstractSparseMatrixCSC}

# Views of an AbstractSparseMatrixCSC taking all rows and a unit range of columns. getcolptr
# is an offset view into the parent's colptr, so kernels written against
# getcolptr/getrowval/getnzval work unchanged.
const SparseMatrixCSCView{Tv,Ti} =
    SubArray{Tv,2,<:AbstractSparseMatrixCSC{Tv,Ti},
        Tuple{Base.Slice{Base.OneTo{Int}},I}} where {I<:AbstractUnitRange{<:Integer}}
const SparseMatrixCSCOrView{Tv,Ti} = Union{AbstractSparseMatrixCSC{Tv,Ti}, SparseMatrixCSCView{Tv,Ti}}
# Views taking all rows and an arbitrary column subset (a superset of SparseMatrixCSCView):
# nzrange/getrowval/getnzval work, getcolptr does not.
const SparseMatrixCSCColumnSubset{Tv,Ti} =
    SubArray{Tv,2,<:AbstractSparseMatrixCSC{Tv,Ti},
        Tuple{Base.Slice{Base.OneTo{Int}},I}} where {I<:AbstractVector{<:Integer}}
const SparseMatrixCSCOrColumnSubset{Tv,Ti} = Union{AbstractSparseMatrixCSC{Tv,Ti}, SparseMatrixCSCColumnSubset{Tv,Ti}}

# Whole-column views of sparse matrices and whole views of sparse vectors share the
# sparse vector interface.
const SparseColumnView{Tv,Ti}  = SubArray{Tv,1,<:AbstractSparseMatrixCSC{Tv,Ti},Tuple{Base.Slice{Base.OneTo{Int}},Int},false}
const SparseVectorView{Tv,Ti}  = SubArray{Tv,1,<:AbstractSparseVector{Tv,Ti},Tuple{Base.Slice{Base.OneTo{Int}}},false}
const SparseVectorOrView{Tv,Ti} = Union{AbstractCompressedVector{Tv,Ti}, SparseColumnView{Tv,Ti}, SparseVectorView{Tv,Ti}}
const AdjOrTransSparseVectorOrView{Tv,Ti} = AdjOrTrans{Tv, <:SparseVectorOrView{Tv,Ti}}
# view of a unit range of a sparse vector's indices
const SparseVectorPartialView{Tv,Ti} = SubArray{Tv,1,<:AbstractSparseVector{Tv,Ti},<:Tuple{AbstractUnitRange},false}

# `X` or an `Adjoint`/`Transpose` of `X`, named after LinearAlgebra's StridedMaybeAdjOrTransMat
const SparseMatrixCSCMaybeAdjOrTrans = Union{AbstractSparseMatrixCSC, AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}}
const SparseVecOrMatMaybeAdjOrTrans = Union{SparseVecOrMat, AdjOrTrans{<:Any,<:SparseVecOrMat}}

# LinearAlgebra wrappers around CSC storage
const SparseTriangular{Tv,Ti} = UpperOrLowerTriangular{Tv,<:SparseMatrixCSCOrView{Tv,Ti}}
const SparseOrTri{Tv,Ti} = Union{SparseMatrixCSCOrView{Tv,Ti}, SparseTriangular{Tv,Ti}}
const SparseMatrixCSCSymmHerm{Tv,Ti} = HermOrSym{Tv,<:SparseMatrixCSCOrView{Tv,Ti}}

# dense operands of the sparse-dense kernels
const DenseMatrixUnion = Union{StridedMatrix, BitMatrix}
# LinearAlgebra wrappers of a matrix of type MT, and those plus 2-d views (for dot)
const MatrixWrappers{T,MT} = Union{AdjOrTrans{T,MT}, HermOrSym{T,MT}, UpperOrLowerTriangular{T,MT}, UpperHessenberg{T,MT}}
const MatrixWrappersOrView{T,MT} = Union{SubArray{T,2,MT}, MatrixWrappers{T,MT}}
const QuasiSparseMatrix = Union{SparseMatrixCSCOrColumnSubset, MatrixWrappers{<:Any,<:SparseMatrixCSCOrColumnSubset}}
const QuasiStridedMatrix = Union{StridedMatrix, MatrixWrappers{<:Any,<:StridedMatrix}}
# the adjoint/transpose of a sparse triangular matrix, which LinearAlgebra makes eagerly,
# and the lazy conjugate that taking both leaves
const SparseAdjOrTransTriangular = UpperOrLowerTriangular{<:Any,<:AdjOrTrans{<:Any,
    <:Union{SparseMatrixCSCOrView, AdjOrTrans{<:Any,<:SparseMatrixCSCOrView}}}}

# LinearAlgebra's BiTriSym (Bidiagonal/Tridiagonal/SymTridiagonal) and BandedMatrix (those
# plus Diagonal) are imported for the banded special matrices.
# LinearAlgebra's Q types that multiply sparse arrays via densification
const LinAlgLeftQs = Union{HessenbergQ,QRCompactWYQ,QRPackedQ}
# Sparse operands that a Q multiplies through a dense copy: Q is dense in general, so
# applying it to a dense copy of the operand is cheaper than materializing Q or a sparse
# transpose of the operand. Sparse matrices and their views, with or without an adjoint or
# transpose, and a transposed sparse vector as a one-row matrix; and sparse vectors and their
# views. An adjoint sparse vector is left to LinearAlgebra's generic (Q' * u')', which keeps
# the result an Adjoint as for a dense vector.
const SparseQMatOperand = Union{SparseMatrixCSCOrView, AdjOrTrans{<:Any,<:SparseMatrixCSCOrView},
                                Transpose{<:Any,<:SparseVectorOrView}}
const SparseQVecOperand = Union{AbstractSparseVector, SparseVectorOrView, SparseVectorPartialView}

# Former names of the above, not used here but relied on by downstream packages together
# with SparseMatrixCSCView, SparseColumnView and SparseVectorView. May be deprecated in a
# future release.
const SparseMatrixCSCUnion{Tv,Ti} = SparseMatrixCSCOrView{Tv,Ti}
const SparseVectorUnion{Tv,Ti} = SparseVectorOrView{Tv,Ti}

"""
    issparse(S)

Returns `true` if `S` is sparse or wraps a sparse array, and `false` otherwise.

`issparse` is a classification predicate. A `true` result does not guarantee
support for a particular `SparseArrays` operation (such as `nnz`, `nonzeros`,
or `findnz`), a particular sparse storage format, or that `sparse(S)` is an
identity or efficient operation. Likewise, `false` means that `S` is not recognized
as sparse by `SparseArrays`, not that it is dense: an array type that does not
subtype `AbstractSparseArray` or wrap one yields `false` regardless of its storage.
Code requiring a particular sparse interface should dispatch on the relevant
abstract type or operation instead of branching on `issparse`.

# Examples
```jldoctest
julia> sv = sparsevec([1, 4], [2.3, 2.2], 10)
10-element SparseVector{Float64, Int64} with 2 stored entries:
  [1]  =  2.3
  [4]  =  2.2

julia> issparse(sv)
true

julia> issparse(Array(sv))
false
```
"""
function issparse(A::AbstractArray)
    # Handle wrapper arrays: sparse if it is wrapping a sparse array.
    # This gets compiled away during specialization.
    p = parent(A)
    if p === A
        # have reached top of wrapping without finding a sparse array, assume it is not.
        return false
    else
        return issparse(p)
    end
end
issparse(A::DenseArray) = false
issparse(S::AbstractSparseArray) = true

"""
    indtype(S)

Return the type used to index sparse array entries.

# Examples
```jldoctest
julia> indtype(sparse(Int32[1, 2], Int32[1, 2], [1.0, 2.0]))
Int32
```
"""
indtype(S::AbstractSparseArray{<:Any,Ti}) where {Ti} = Ti
indtype(T::UpperOrLowerTriangular{<:Any,<:Union{AbstractSparseArray,SparseMatrixCSCColumnSubset}}) = indtype(parent(T))

# The following two methods should be overloaded by concrete types to avoid
# allocating the I = findall(...)
_sparse_findnextnz(v::AbstractSparseArray, i) = (I = findall(_isnotzero, v); n = searchsortedfirst(I, i); n<=length(I) ? I[n] : nothing)
_sparse_findprevnz(v::AbstractSparseArray, i) = (I = findall(_isnotzero, v); n = searchsortedlast(I, i);  _isnotzero(n) ? I[n] : nothing)

function findnext(f::Function, v::AbstractSparseArray, i)
    # short-circuit the case f == !iszero because that avoids
    # allocating e.g. zero(BigInt) for the f(zero(...)) test.
    if nnz(v) == length(v) || (f != (!iszero) && f != _isnotzero && f(zero(eltype(v))))
        return invoke(findnext, Tuple{Function,Any,Any}, f, v, i)
    end
    j = _sparse_findnextnz(v, i)
    while j !== nothing && !f(v[j])
        j = _sparse_findnextnz(v, nextind(v, j))
    end
    return j
end

function findprev(f::Function, v::AbstractSparseArray, i)
    # short-circuit the case f == !iszero because that avoids
    # allocating e.g. zero(BigInt) for the f(zero(...)) test.
    if nnz(v) == length(v) || (f != (!iszero) && f != _isnotzero && f(zero(eltype(v))))
        return invoke(findprev, Tuple{Function,Any,Any}, f, v, i)
    end
    j = _sparse_findprevnz(v, i)
    while j !== nothing && !f(v[j])
        j = _sparse_findprevnz(v, prevind(v, j))
    end
    return j
end

"""
    findnz(A::SparseMatrixCSC)

Return a tuple `(I, J, V)` where `I` and `J` are the row and column indices of the stored
("structurally non-zero") values in sparse matrix `A`, and `V` is a vector of the values.
`A` may also be the adjoint or transpose of a sparse matrix or vector, in which case the
values in `V` are correspondingly adjointed or transposed.

# Examples
```jldoctest
julia> A = sparse([1 2 0; 0 0 3; 0 4 0])
3×3 SparseMatrixCSC{Int64, Int64} with 4 stored entries:
 1  2  ⋅
 ⋅  ⋅  3
 ⋅  4  ⋅

julia> findnz(A)
([1, 1, 3, 2], [1, 2, 2, 3], [1, 2, 4, 3])
```
"""
function findnz end

widelength(x::AbstractSparseArray) = prod(Int64.(size(x)))


const _restore_scalar_indexing = Expr[]
const _destroy_scalar_indexing = Expr[]
"""
    @RCI f

records the function `f` to be overwritten (and restored) with `allowscalar(::Bool)`. This is an
experimental feature.

Note that it will evaluate the function in the top level of the package. The original code for `f`
is stored in `_restore_scalar_indexing` and a function that has the same definition as `f` but
returns an error is stored in `_destroy_scalar_indexing`.
"""
macro RCI(exp)
    # Evaluate to not push any broken code in the arrays when developing this package.
    # Ensures that restore has the exact same effect.
    # Expand macro so we can chain macros. Save the expanded version for speed
    exp = macroexpand(__module__, exp)
    @eval __module__ $exp
    if length(exp.args) == 2 && exp.head ∈ (:function, :(=))
        push!(_restore_scalar_indexing, exp)
        push!(_destroy_scalar_indexing,
            Expr(exp.head,
            exp.args[1],
            :(error("scalar indexing was turned off"))))
    else
        error("can't parse expression")
    end
    return
end

"""
    allowscalar(::Bool)

An experimental function that allows one to disable and re-enable scalar indexing for sparse matrices and vectors.

`allowscalar(false)` will disable scalar indexing for sparse matrices and vectors.
`allowscalar(true)` will restore the original scalar indexing functionality.

Since this function overwrites existing definitions, it will lead to recompilation. It is useful mainly when testing
code for devices such as [GPUs](https://cuda.juliagpu.org/stable/usage/workflow/), where the presence of scalar indexing can lead to substantial slowdowns.
Disabling scalar indexing during such tests can help identify performance bottlenecks quickly.
"""
allowscalar(p::Bool) = if p
    for i in _restore_scalar_indexing
        @eval $i
    end
else
    for i in _destroy_scalar_indexing
        @eval $i
    end
end

macro allowscalar(p)
    quote
        $(allowscalar)($(esc(p)))
        @Core.latestworld
    end
end

# seed for a sparse `dot`: zero of the result type, which is a scalar even for matrix-valued
# entries that have no `zero` themselves
_dot_zero(Ts::Type...) = zero(promote_op(dot, Ts...))

include("fixed.jl")
include("sparsematrix.jl")
include("constructors.jl")
include("indexing.jl")
include("reductions.jl")
include("sparseconvert.jl")
include("sparsevector.jl")
include("concatenation.jl")
include("higherorderfns.jl")
include("linalg.jl")
include("matmul.jl")



# Convert from 0-based to 1-based indices
function increment!(A::AbstractArray{T}) where T<:Integer
    for i in eachindex(A); A[i] += oneunit(T) end
    A
end
increment(A::AbstractArray{<:Integer}) = increment!(copy(A))

include("solvers/LibSuiteSparse.jl")
using .LibSuiteSparse

@static if Base.USE_GPL_LIBS
    include("solvers/umfpack.jl")
    include("solvers/cholmod.jl")
    include("solvers/spqr.jl")
end

zero(a::AbstractSparseArray{Tv,Ti}) where {Tv,Ti} = spzeros(Tv, Ti, size(a)...)

LinearAlgebra.diagzero(D::Diagonal{<:AbstractSparseMatrix{T}},i,j) where {T} =
    spzeros(T, size(D.diag[i], 1), size(D.diag[j], 2))

end
