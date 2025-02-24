# This file is a part of Julia. License is MIT: https://julialang.org/license

module SPQR

import Base: \, *
using Base: require_one_based_indexing
using LinearAlgebra
using LinearAlgebra: AbstractQ, AdjointQ, AdjointAbsVec, copy_similar
using ..LibSuiteSparse: SuiteSparseQR_C, SuiteSparseQR_i_C

# ordering options */
const ORDERING_FIXED   = Int32(0)
const ORDERING_NATURAL = Int32(1)
const ORDERING_COLAMD  = Int32(2)
const ORDERING_GIVEN   = Int32(3) # only used for C/C++ interface
const ORDERING_CHOLMOD = Int32(4) # CHOLMOD best-effort (COLAMD, METIS,...)
const ORDERING_AMD     = Int32(5) # AMD(A'*A)
const ORDERING_METIS   = Int32(6) # metis(A'*A)
const ORDERING_DEFAULT = Int32(7) # SuiteSparseQR default ordering
const ORDERING_BEST    = Int32(8) # try COLAMD, AMD, and METIS; pick best
const ORDERING_BESTAMD = Int32(9) # try COLAMD and AMD; pick best#
const ORDERINGS = [ORDERING_FIXED, ORDERING_NATURAL, ORDERING_COLAMD, ORDERING_CHOLMOD,
                   ORDERING_AMD, ORDERING_METIS, ORDERING_DEFAULT, ORDERING_BEST,
                   ORDERING_BESTAMD]

# Let [m n] = size of the matrix after pruning singletons.  The default
# ordering strategy is to use COLAMD if m <= 2*n.  Otherwise, AMD(A'A) is
# tried.  If there is a high fill-in with AMD then try METIS(A'A) and take
# the best of AMD and METIS. METIS is not tried if it isn't installed.

using ..SparseArrays
using ..SparseArrays: getcolptr, FixedSparseCSC, AbstractSparseMatrixCSC, _unsafe_unfix
using ..CHOLMOD
using ..CHOLMOD: change_stype!, free!

import ..LibSuiteSparse: cholmod_l_free, cholmod_free

function _qr!(ordering::Integer, tol::Real, econ::Integer, getCTX::Integer,
        A::Sparse{Tv, Ti},
        Bsparse::Union{Sparse{Tv, Ti}                      , Ptr{Cvoid}} = C_NULL,
        Bdense::Union{Dense{Tv}                        , Ptr{Cvoid}} = C_NULL,
        Zsparse::Union{Ref{Ptr{CHOLMOD.cholmod_sparse}}  , Ptr{Cvoid}} = C_NULL,
        Zdense::Union{Ref{Ptr{CHOLMOD.cholmod_dense}}  , Ptr{Cvoid}} = C_NULL,
        R::Union{Ref{Ptr{CHOLMOD.cholmod_sparse}}        , Ptr{Cvoid}} = C_NULL,
        E::Union{Ref{Ptr{Ti}}    , Ptr{Cvoid}} = C_NULL,
        H::Union{Ref{Ptr{CHOLMOD.cholmod_sparse}}        , Ptr{Cvoid}} = C_NULL,
        HPinv::Union{Ref{Ptr{Ti}}, Ptr{Cvoid}} = C_NULL,
        HTau::Union{Ref{Ptr{CHOLMOD.cholmod_dense}}    , Ptr{Cvoid}} = C_NULL) where {Ti<:CHOLMOD.ITypes, Tv<:CHOLMOD.VTypes}

    ordering ∈ ORDERINGS || error("unknown ordering $ordering")

    spqr_call = Ti === Int32 ? SuiteSparseQR_i_C : SuiteSparseQR_C
    AA   = unsafe_load(pointer(A))
    m, n = AA.nrow, AA.ncol
    rnk  = spqr_call(
        ordering,       # all, except 3:given treated as 0:fixed
        tol,            # columns with 2-norm <= tol treated as 0
        econ,           # e = max(min(m,econ),rank(A))
        getCTX,         # 0: Z=C (e-by-k), 1: Z=C', 2: Z=X (e-by-k)
        A,              # m-by-n sparse matrix to factorize
        Bsparse,        # sparse m-by-k B
        Bdense,         # dense  m-by-k B
        # /* outputs: */
        Zsparse,        # sparse Z
        Zdense,         # dense Z
        R,              # e-by-n sparse matrix */
        E,              # size n column perm, NULL if identity */
        H,              # m-by-nh Householder vectors
        HPinv,          # size m row permutation
        HTau,           # 1-by-nh Householder coefficients
        CHOLMOD.getcommon(Ti)) # /* workspace and parameters */

    if rnk < 0
        error("Sparse QR factorization failed")
    end

    e = E[]
    if e == C_NULL
        _E = Vector{Ti}()
    else
        _E = Vector{Ti}(undef, n)
        for i in 1:n
            @inbounds _E[i] = unsafe_load(e, i) + 1
        end
        # Free memory allocated by SPQR. This call will make sure that the
        # correct deallocator function is called and that the memory count in
        # the common struct is updated
        Ti === Int64 ? 
            cholmod_l_free(n, sizeof(Ti), e, CHOLMOD.getcommon(Ti)) :
            cholmod_free(n, sizeof(Ti), e, CHOLMOD.getcommon(Ti))
    end
    hpinv = HPinv[]
    if hpinv == C_NULL
        _HPinv = Vector{Ti}()
    else
        _HPinv = Vector{Ti}(undef, m)
        for i in 1:m
            @inbounds _HPinv[i] = unsafe_load(hpinv, i) + 1
        end
        # Free memory allocated by SPQR. This call will make sure that the
        # correct deallocator function is called and that the memory count in
        # the common struct is updated
        Ti === Int64 ? 
            cholmod_l_free(m, sizeof(Ti), hpinv, CHOLMOD.getcommon(Ti)) :
            cholmod_free(m, sizeof(Ti), hpinv, CHOLMOD.getcommon(Ti))
    end

    return rnk, _E, _HPinv
end

# Struct for storing sparse QR from SPQR such that
# A[invperm(rpivinv), cpiv] = (I - factors[:,1]*τ[1]*factors[:,1]')*...*(I - factors[:,k]*τ[k]*factors[:,k]')*R
# with k = size(factors, 2).
struct QRSparse{Tv,Ti} <: LinearAlgebra.Factorization{Tv}
    factors::SparseMatrixCSC{Tv,Ti}
    τ::Vector{Tv}
    R::SparseMatrixCSC{Tv,Ti}
    cpiv::Vector{Ti}
    rpivinv::Vector{Ti}
end

Base.size(F::QRSparse) = (size(F.factors, 1), size(F.R, 2))
function Base.size(F::QRSparse, i::Integer)
    if i == 1
        return size(F.factors, 1)
    elseif i == 2
        return size(F.R, 2)
    elseif i > 2
        return 1
    else
        throw(ArgumentError("second argument must be positive"))
    end
end
Base.axes(F::QRSparse) = map(Base.OneTo, size(F))

struct QRSparseQ{Tv<:CHOLMOD.VTypes,Ti<:Integer} <: AbstractQ{Tv}
    factors::SparseMatrixCSC{Tv,Ti}
    τ::Vector{Tv}
    n::Int # Number of columns in original matrix
end

Base.size(Q::QRSparseQ) = (size(Q.factors, 1), size(Q.factors, 1))
Base.axes(Q::QRSparseQ) = map(Base.OneTo, size(Q))

Matrix{T}(Q::QRSparseQ) where {T} = lmul!(Q, Matrix{T}(I, size(Q, 1), min(size(Q, 1), Q.n)))

# From SPQR manual p. 6
_default_tol(A::AbstractSparseMatrixCSC) =
    20*sum(size(A))*eps()*maximum(norm(view(A, :, i)) for i in axes(A, 2))

"""
    qr(A::SparseMatrixCSC; tol=_default_tol(A), ordering=ORDERING_DEFAULT) -> QRSparse

Compute the `QR` factorization of a sparse matrix `A`. Fill-reducing row and column permutations
are used such that `F.R = F.Q'*A[F.prow,F.pcol]`. The main application of this type is to
solve least squares or underdetermined problems with [`\\`](@ref). The function calls the C library SPQR[^ACM933].

!!! note
    `qr(A::SparseMatrixCSC)` uses the SPQR library that is part of [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse).
    As this library only supports sparse matrices with [`Float64`](@ref) or
    `ComplexF64` elements, as of Julia v1.4 `qr` converts `A` into a copy that is
    of type `SparseMatrixCSC{Float64}` or `SparseMatrixCSC{ComplexF64}` as appropriate.

# Examples
```jldoctest
julia> A = sparse([1,2,3,4], [1,1,2,2], [1.0,1.0,1.0,1.0])
4×2 SparseMatrixCSC{Float64, Int64} with 4 stored entries:
 1.0   ⋅
 1.0   ⋅
  ⋅   1.0
  ⋅   1.0

julia> qr(A)
SparseArrays.SPQR.QRSparse{Float64, Int64}
Q factor:
4×4 SparseArrays.SPQR.QRSparseQ{Float64, Int64}
R factor:
2×2 SparseMatrixCSC{Float64, Int64} with 2 stored entries:
 -1.41421    ⋅
   ⋅       -1.41421
Row permutation:
4-element Vector{Int64}:
 1
 3
 4
 2
Column permutation:
2-element Vector{Int64}:
 1
 2
```

[^ACM933]: Foster, L. V., & Davis, T. A. (2013). Algorithm 933: Reliable Calculation of Numerical Rank, Null Space Bases, Pseudoinverse Solutions, and Basic Solutions Using SuitesparseQR. ACM Trans. Math. Softw., 40(1). [doi:10.1145/2513109.2513116](https://doi.org/10.1145/2513109.2513116)
"""
function LinearAlgebra.qr(A::SparseMatrixCSC{Tv, Ti}; tol=_default_tol(A), ordering=ORDERING_DEFAULT) where {Ti<:CHOLMOD.ITypes, Tv<:CHOLMOD.VTypes}
    R     = Ref{Ptr{CHOLMOD.cholmod_sparse}}()
    E     = Ref{Ptr{Ti}}()
    H     = Ref{Ptr{CHOLMOD.cholmod_sparse}}()
    HPinv = Ref{Ptr{Ti}}()
    HTau  = Ref{Ptr{CHOLMOD.cholmod_dense}}(C_NULL)

    # SPQR doesn't accept symmetric matrices so we explicitly set the stype
    r, p, hpinv = _qr!(ordering, tol, 0, 0, Sparse(A, 0),
        C_NULL, C_NULL, C_NULL, C_NULL,
        R, E, H, HPinv, HTau)

    R_ = SparseMatrixCSC{Tv, Ti}(Sparse(R[]))
    return QRSparse(SparseMatrixCSC{Tv, Ti}(Sparse(H[])),
                    vec(Array{Tv}(CHOLMOD.Dense(HTau[]))),
                    SparseMatrixCSC{Tv, Ti}(min(size(A)...),
                                    size(R_, 2),
                                    getcolptr(R_),
                                    rowvals(R_),
                                    nonzeros(R_)),
                    p, hpinv)
end
LinearAlgebra.qr(A::SparseMatrixCSC{Float16}; tol=_default_tol(A)) =
    qr(convert(SparseMatrixCSC{Float32}, A); tol=tol)
LinearAlgebra.qr(A::SparseMatrixCSC{ComplexF16}; tol=_default_tol(A)) =
    qr(convert(SparseMatrixCSC{ComplexF32}, A); tol=tol)
LinearAlgebra.qr(A::Union{SparseMatrixCSC{T},SparseMatrixCSC{Complex{T}}};
   tol=_default_tol(A)) where {T<:AbstractFloat} =
    throw(ArgumentError(string("matrix type ", typeof(A), "not supported. ",
    "Try qr(convert(SparseMatrixCSC{Float64/ComplexF64, Int}, A)) for ",
    "sparse floating point QR using SPQR or qr(Array(A)) for generic ",
    "dense QR.")))
LinearAlgebra.qr(A::SparseMatrixCSC; tol=_default_tol(A)) = qr(Float64.(A); tol=tol)
LinearAlgebra.qr(::SparseMatrixCSC, ::LinearAlgebra.PivotingStrategy) = error("Pivoting Strategies are not supported by `SparseMatrixCSC`s")
LinearAlgebra.qr(A::FixedSparseCSC; tol=_default_tol(A), ordering=ORDERING_DEFAULT) =
    let B=A
        qr(_unsafe_unfix(B); tol, ordering)
    end

LinearAlgebra._qr(A::SparseMatrixCSC; kwargs...) = qr(A; kwargs...)
LinearAlgebra._qr(::SparseMatrixCSC, ::LinearAlgebra.PivotingStrategy; kwargs...) =
    error("Pivoting Strategies are not supported for `SparseMatrixCSC`s")

function LinearAlgebra.lmul!(Q::QRSparseQ, A::StridedVecOrMat)
    if size(A, 1) != size(Q, 1)
        throw(DimensionMismatch("size(Q) = $(size(Q)) but size(A) = $(size(A))"))
    end
    for l in size(Q.factors, 2):-1:1
        τl = -Q.τ[l]
        h = view(Q.factors, :, l)
        for j in axes(A, 2)
            a = view(A, :, j)
            axpy!(τl*dot(h, a), h, a)
        end
    end
    return A
end

function LinearAlgebra.rmul!(A::StridedMatrix, Q::QRSparseQ)
    if size(A, 2) != size(Q, 1)
        throw(DimensionMismatch("size(Q) = $(size(Q)) but size(A) = $(size(A))"))
    end
    tmp = similar(A, size(A, 1))
    for l in axes(Q.factors, 2)
        τl = -Q.τ[l]
        h = view(Q.factors, :, l)
        mul!(tmp, A, h)
        lowrankupdate!(A, tmp, h, τl)
    end
    return A
end

function LinearAlgebra.lmul!(adjQ::AdjointQ{<:Any,<:QRSparseQ}, A::StridedVecOrMat)
    Q = parent(adjQ)
    if size(A, 1) != size(Q, 1)
        throw(DimensionMismatch("size(Q) = $(size(Q)) but size(A) = $(size(A))"))
    end
    for l in axes(Q.factors, 2)
        τl = -Q.τ[l]
        h = view(Q.factors, :, l)
        for j in axes(A, 2)
            a = view(A, :, j)
            LinearAlgebra.axpy!(τl'*dot(h, a), h, a)
        end
    end
    return A
end

function LinearAlgebra.rmul!(A::StridedMatrix, adjQ::AdjointQ{<:Any,<:QRSparseQ})
    Q = parent(adjQ)
    if size(A, 2) != size(Q, 1)
        throw(DimensionMismatch("size(Q) = $(size(Q)) but size(A) = $(size(A))"))
    end
    tmp = similar(A, size(A, 1))
    for l in size(Q.factors, 2):-1:1
        τl = -Q.τ[l]
        h = view(Q.factors, :, l)
        mul!(tmp, A, h)
        lowrankupdate!(A, tmp, h, τl')
    end
    return A
end

function (*)(Q::QRSparseQ, b::AbstractVector)
    TQb = promote_type(eltype(Q), eltype(b))
    QQ = convert(AbstractQ{TQb}, Q)
    if size(Q.factors, 1) == length(b)
        bnew = copy_similar(b, TQb)
    elseif size(Q.factors, 2) == length(b)
        bnew = [b; zeros(TQb, size(Q.factors, 1) - length(b))]
    else
        throw(DimensionMismatch("vector must have length either $(size(Q.factors, 1)) or $(size(Q.factors, 2))"))
    end
    lmul!(QQ, bnew)
end
function (*)(Q::QRSparseQ, B::AbstractMatrix)
    TQB = promote_type(eltype(Q), eltype(B))
    QQ = convert(AbstractQ{TQB}, Q)
    if size(Q.factors, 1) == size(B, 1)
        Bnew = copy_similar(B, TQB)
    elseif size(Q.factors, 2) == size(B, 1)
        Bnew = [B; zeros(TQB, size(Q.factors, 1) - size(B,1), size(B, 2))]
    else
        throw(DimensionMismatch("first dimension of matrix must have size either $(size(Q.factors, 1)) or $(size(Q.factors, 2))"))
    end
    lmul!(QQ, Bnew)
end
function (*)(A::AbstractMatrix, adjQ::AdjointQ{<:Any,<:QRSparseQ})
    Q = parent(adjQ)
    TAQ = promote_type(eltype(A), eltype(adjQ))
    adjQQ = convert(AbstractQ{TAQ}, adjQ)
    if size(A,2) == size(Q.factors, 1)
        AA = copy_similar(A, TAQ)
        return rmul!(AA, adjQQ)
    elseif size(A,2) == size(Q.factors,2)
        return rmul!([A zeros(TAQ, size(A, 1), size(Q.factors, 1) - size(Q.factors, 2))], adjQQ)
    else
        throw(DimensionMismatch("matrix A has dimensions $(size(A)) but Q-matrix has dimensions $(size(adjQ))"))
    end
end
(*)(u::AdjointAbsVec, Q::AdjointQ{<:Any,<:QRSparseQ}) = (Q'u')'

(*)(Q::QRSparseQ, B::SparseMatrixCSC) = sparse(Q) * B
(*)(A::SparseMatrixCSC, Q::QRSparseQ) = A * sparse(Q)

@inline function Base.getproperty(F::QRSparse, d::Symbol)
    if d === :Q
        return QRSparseQ(F.factors, F.τ, size(F, 2))
    elseif d === :prow
        return invperm(F.rpivinv)
    elseif d === :pcol
        return F.cpiv
    else
        getfield(F, d)
    end
end

function Base.propertynames(F::QRSparse, private::Bool=false)
    public = (:R, :Q, :prow, :pcol)
    private ? ((public ∪ fieldnames(typeof(F)))...,) : public
end

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::QRSparse)
    summary(io, F); println(io)
    println(io, "Q factor:")
    show(io, mime, F.Q)
    println(io, "\nR factor:")
    show(io, mime, F.R)
    println(io, "\nRow permutation:")
    show(io, mime, F.prow)
    println(io, "\nColumn permutation:")
    show(io, mime, F.pcol)
end

"""
    rank(::QRSparse{Tv,Ti}) -> Ti

Return the rank of the QR factorization
"""
LinearAlgebra.rank(F::QRSparse) = reduce(max, view(rowvals(F.R), 1:nnz(F.R)), init = eltype(rowvals(F.R))(0))

"""
    rank(S::SparseMatrixCSC{Tv,Ti}; [tol::Real]) -> Ti

Calculate rank of `S` by calculating its QR factorization. Values smaller than `tol` are considered as zero. See SPQR's manual.
"""
LinearAlgebra.rank(S::SparseMatrixCSC; tol=_default_tol(S)) = rank(qr(S; tol))

# With a real lhs and complex rhs with the same precision, we can reinterpret
# the complex rhs as a real rhs with twice the number of columns
#
# This definition is similar to the definition in factorization.jl except that
# here we have to use \ instead of ldiv! because of limitations in SPQR

## Two helper methods
_ret_size(F::QRSparse, b::AbstractVector) = (size(F, 2),)
_ret_size(F::QRSparse, B::AbstractMatrix) = (size(F, 2), size(B, 2))

function (\)(F::QRSparse{T}, B::VecOrMat{Complex{T}}) where T<:LinearAlgebra.BlasReal
# |z1|z3|  reinterpret  |x1|x2|x3|x4|  transpose  |x1|y1|  reshape  |x1|y1|x3|y3|
# |z2|z4|      ->       |y1|y2|y3|y4|     ->      |x2|y2|     ->    |x2|y2|x4|y4|
#                                                 |x3|y3|
#                                                 |x4|y4|
    require_one_based_indexing(F, B)
    c2r = reshape(copy(transpose(reinterpret(T, reshape(B, (1, length(B)))))), size(B, 1), 2*size(B, 2))
    x = F\c2r

# |z1|z3|  reinterpret  |x1|x2|x3|x4|  transpose  |x1|y1|  reshape  |x1|y1|x3|y3|
# |z2|z4|      <-       |y1|y2|y3|y4|     <-      |x2|y2|     <-    |x2|y2|x4|y4|
#                                                 |x3|y3|
#                                                 |x4|y4|
    return collect(reshape(reinterpret(Complex{T}, copy(transpose(reshape(x, (length(x) >> 1), 2)))), _ret_size(F, B)))
end

function _ldiv_basic(F::QRSparse, B::StridedVecOrMat)
    if size(F, 1) != size(B, 1)
        throw(DimensionMismatch("size(F) = $(size(F)) but size(B) = $(size(B))"))
    end

    # The rank of F equal might be reduced
    rnk = rank(F)

    # allocate an array for the return value large enough to hold B and X
    # For overdetermined problem, B is larger than X and vice versa
    X   = similar(B, ntuple(i -> i == 1 ? max(size(F, 2), size(B, 1)) : size(B, 2), Val(ndims(B))))

    # Fill will zeros. These will eventually become the zeros in the basic solution
    # fill!(X, 0)
    # Apply left permutation to the solution and store in X
    for j in axes(B, 2)
        for i in 1:length(F.rpivinv)
            @inbounds X[F.rpivinv[i], j] = B[i, j]
        end
    end

    # Make a view into X corresponding to the size of B
    X0 = view(X, axes(B, 1), :)

    # Apply Q' to B
    lmul!(adjoint(F.Q), X0)

    # Zero out to get basic solution
    X[rnk + 1:end, :] .= 0

    # Solve R*X = B
    ldiv!(UpperTriangular(F.R[Base.OneTo(rnk), Base.OneTo(rnk)]),
                        view(X0, Base.OneTo(rnk), :))

    # Apply right permutation and extract solution from X
    # NB: cpiv == [] if SPQR was called with ORDERING_FIXED
    if length(F.cpiv) == 0
      return getindex(X, ntuple(i -> i == 1 ? (1:size(F,2)) : :, Val(ndims(B)))...)
    end
    return getindex(X, ntuple(i -> i == 1 ? invperm(F.cpiv) : :, Val(ndims(B)))...)
end

(\)(F::QRSparse{T}, B::StridedVecOrMat{T}) where {T} = _ldiv_basic(F, B)
"""
    (\\)(F::QRSparse, B::StridedVecOrMat)

Solve the least squares problem ``\\min\\|Ax - b\\|^2`` or the linear system of equations
``Ax=b`` when `F` is the sparse QR factorization of ``A``. A basic solution is returned
when the problem is underdetermined.

# Examples
```jldoctest
julia> A = sparse([1,2,4], [1,1,1], [1.0,1.0,1.0], 4, 2)
4×2 SparseMatrixCSC{Float64, Int64} with 3 stored entries:
 1.0   ⋅
 1.0   ⋅
  ⋅    ⋅
 1.0   ⋅

julia> qr(A)\\fill(1.0, 4)
2-element Vector{Float64}:
 1.0
 0.0
```
"""
(\)(F::QRSparse, B::StridedVecOrMat) = F\convert(AbstractArray{eltype(F)}, B)

end # module
