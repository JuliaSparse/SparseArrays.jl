# This file is a part of Julia. License is MIT: https://julialang.org/license

using LinearAlgebra: AbstractTriangular, UpperOrLowerTriangular,
    RealHermSymComplexHerm, checksquare, sym_uplo, wrap
using Random: rand!

import LinearAlgebra: _uppercase, _isuppercase

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

for op ∈ (:+, :-), Wrapper ∈ (:Hermitian, :Symmetric)
    @eval begin
        $op(A::AbstractSparseMatrix, B::$Wrapper{<:Any,<:AbstractSparseMatrix}) = $op(A, sparse(B))
        $op(A::$Wrapper{<:Any,<:AbstractSparseMatrix}, B::AbstractSparseMatrix) = $op(sparse(A), B)

        $op(A::AbstractSparseMatrix, B::$Wrapper) = $op(A, collect(B))
        $op(A::$Wrapper, B::AbstractSparseMatrix) = $op(collect(A), B)
    end
end
for op ∈ (:+, :-)
    @eval begin
        $op(A::Symmetric{<:Any,  <:AbstractSparseMatrix}, B::Hermitian{<:Any,  <:AbstractSparseMatrix}) = $op(sparse(A), sparse(B))
        $op(A::Hermitian{<:Any,  <:AbstractSparseMatrix}, B::Symmetric{<:Any,  <:AbstractSparseMatrix}) = $op(sparse(A), sparse(B))
        $op(A::Symmetric{<:Real, <:AbstractSparseMatrix}, B::Hermitian{<:Any,  <:AbstractSparseMatrix}) = $op(Hermitian(parent(A), sym_uplo(A.uplo)), B)
        $op(A::Hermitian{<:Any,  <:AbstractSparseMatrix}, B::Symmetric{<:Real, <:AbstractSparseMatrix}) = $op(A, Hermitian(parent(B), sym_uplo(B.uplo)))
    end
end

## triangular sparse handling
## triangular solvers
_uconvert_copyto!(c, b, oA) = (c .= Ref(oA) .\ b)
_uconvert_copyto!(c::AbstractArray{T}, b::AbstractArray{T}, _) where {T} = copyto!(c, b)

function LinearAlgebra.generic_trimatdiv!(C::StridedVecOrMat, uploc, isunitc, tfun::Function, A::SparseMatrixCSCOrView, B::AbstractVecOrMat)
    mA, nA = size(A)
    nrowB, ncolB = size(B, 1), size(B, 2)
    if nA != nrowB
        throw(DimensionMismatch("second dimension of left hand side A, $nA, and first dimension of right hand side B, $nrowB, must be equal"))
    end
    if size(C) != size(B)
        throw(DimensionMismatch("size of output, $(size(C)), does not match size of right hand side, $(size(B))"))
    end
    C !== B && _uconvert_copyto!(C, B, oneunit(eltype(A)))
    aa = getnzval(A)
    ja = getrowval(A)
    ia = getcolptr(A)
    unit = isunitc == 'U'

    if uploc == 'L'
        if tfun === identity
            # forward substitution for LowerTriangular CSC matrices
            for k in axes(B,2)
                for j in axes(B,1)
                    i1 = ia[j]
                    i2 = ia[j + 1] - one(eltype(ia))

                    # find diagonal element
                    ii = searchsortedfirst(view(ja, i1:i2), j) + i1 - 1
                    jai = ii > i2 ? zero(eltype(ja)) : ja[ii]

                    cj = C[j,k]
                    # check for zero pivot and divide with pivot
                    if jai == j
                        if !unit
                            cj /= LinearAlgebra._ustrip(aa[ii])
                            C[j,k] = cj
                        end
                        ii += 1
                    elseif !unit
                        throw(LinearAlgebra.SingularException(j))
                    end

                    # update remaining part
                    for i = ii:i2
                        C[ja[i],k] -= cj * LinearAlgebra._ustrip(aa[i])
                    end
                end
            end
        else # tfun in (adjoint, transpose)
            # backward substitution for adjoint and transpose of LowerTriangular CSC matrices
            for k in axes(B,2)
                for j in reverse(axes(B,1))
                    i1 = ia[j]
                    i2 = ia[j + 1] - 1
                    akku = B[j,k]
                    done = false

                    # loop through column j of A - only structural non-zeros
                    for ii = i2:-1:i1
                        jai = ja[ii]
                        if jai > j
                            akku -= C[jai,k] * tfun(aa[ii])
                        elseif jai == j
                            akku /= unit ? oneunit(eltype(A)) : tfun(aa[ii])
                            done = true
                            break
                        else
                            break
                        end
                    end
                    if !done && !unit
                        throw(LinearAlgebra.SingularException(j))
                    end
                    C[j,k] = akku
                end
            end
        end
    else # uploc == 'U'
        if tfun === identity
            # backward substitution for UpperTriangular CSC matrices
            for k in axes(B,2)
                for j in reverse(axes(B,1))
                    i1 = ia[j]
                    i2 = ia[j + 1] - one(eltype(ia))

                    # find diagonal element
                    ii = searchsortedlast(view(ja, i1:i2), j) + i1 - 1
                    jai = ii < i1 ? zero(eltype(ja)) : ja[ii]

                    cj = C[j,k]
                    # check for zero pivot and divide with pivot
                    if jai == j
                        if !unit
                            cj /= LinearAlgebra._ustrip(aa[ii])
                            C[j,k] = cj
                        end
                        ii -= 1
                    elseif !unit
                        throw(LinearAlgebra.SingularException(j))
                    end

                    # update remaining part
                    for i = ii:-1:i1
                        C[ja[i],k] -= cj * LinearAlgebra._ustrip(aa[i])
                    end
                end
            end
        else # tfun in  (adjoint, transpose)
            # forward substitution for adjoint and transpose of UpperTriangular CSC matrices
            for k in axes(B,2)
                for j in axes(B,1)
                    i1 = ia[j]
                    i2 = ia[j + 1] - 1
                    akku = B[j,k]
                    done = false

                    # loop through column j of A - only structural non-zeros
                    for ii = i1:i2
                        jai = ja[ii]
                        if jai < j
                            akku -= C[jai,k] * tfun(aa[ii])
                        elseif jai == j
                            akku /= unit ? oneunit(eltype(A)) : tfun(aa[ii])
                            done = true
                            break
                        else
                            break
                        end
                    end
                    if !done && !unit
                        throw(LinearAlgebra.SingularException(j))
                    end
                    C[j,k] = akku
                end
            end
        end
    end
    C
end
function LinearAlgebra.generic_trimatdiv!(C::StridedVecOrMat, uploc, isunitc, ::Function, xA::AdjOrTrans{<:Any,<:SparseMatrixCSCOrView}, B::AbstractVecOrMat)
    A = parent(xA)
    mA, nA = size(A)
    nrowB, ncolB = size(B, 1), size(B, 2)
    if nA != nrowB
        throw(DimensionMismatch("second dimension of left hand side A, $nA, and first dimension of right hand side B, $nrowB, must be equal"))
    end
    if size(C) != size(B)
        throw(DimensionMismatch("size of output, $(size(C)), does not match size of right hand side, $(size(B))"))
    end
    C !== B && _uconvert_copyto!(C, B, oneunit(eltype(A)))

    aa = getnzval(A)
    ja = getrowval(A)
    ia = getcolptr(A)
    unit = isunitc == 'U'

    if uploc == 'L'
        # forward substitution for LowerTriangular CSC matrices
        for k in axes(B,2)
            for j in axes(B,1)
                i1 = ia[j]
                i2 = ia[j + 1] - one(eltype(ia))

                # find diagonal element
                ii = searchsortedfirst(view(ja, i1:i2), j) + i1 - 1
                jai = ii > i2 ? zero(eltype(ja)) : ja[ii]

                cj = C[j,k]
                # check for zero pivot and divide with pivot
                if jai == j
                    if !unit
                        cj /= LinearAlgebra._ustrip(conj(aa[ii]))
                        C[j,k] = cj
                    end
                    ii += 1
                elseif !unit
                    throw(LinearAlgebra.SingularException(j))
                end

                # update remaining part
                for i = ii:i2
                    C[ja[i],k] -= cj * LinearAlgebra._ustrip(conj(aa[i]))
                end
            end
        end
    else # uploc == 'U'
        # backward substitution for UpperTriangular CSC matrices
        for k in axes(B,2)
            for j in reverse(axes(B,1))
                i1 = ia[j]
                i2 = ia[j + 1] - one(eltype(ia))

                # find diagonal element
                ii = searchsortedlast(view(ja, i1:i2), j) + i1 - 1
                jai = ii < i1 ? zero(eltype(ja)) : ja[ii]

                cj = C[j,k]
                # check for zero pivot and divide with pivot
                if jai == j
                    if !unit
                        cj /= LinearAlgebra._ustrip(conj(aa[ii]))
                        C[j,k] = cj
                    end
                    ii -= 1
                elseif !unit
                    throw(LinearAlgebra.SingularException(j))
                end

                # update remaining part
                for i = ii:-1:i1
                    C[ja[i],k] -= cj * LinearAlgebra._ustrip(conj(aa[i]))
                end
            end
        end
    end
    C
end

matop_dest(::typeof(\), A, b::AbstractSparseVector) =
    Vector{promote_op(\, eltype(A), eltype(b))}(undef, length(b))
matop_dest(::typeof(\), A::UnitUpperOrUnitLowerTriangular, b::AbstractSparseVector) =
    Vector{LinearAlgebra._inner_type_promotion(\, eltype(A), eltype(b))}(undef, length(b))
matop_dest(::typeof(\), A::Diagonal, b::AbstractSparseVector) =
    similar(b , promote_op(\, eltype(A), eltype(b)))
matop_dest(::typeof(\), A, B::QuasiSparseMatrix) =
    Matrix{promote_op(\, eltype(A), eltype(B))}(undef, size(B))
matop_dest(::typeof(\), A::Diagonal, B::QuasiSparseMatrix) =
    similar(B , promote_op(\, eltype(A), eltype(B)), size(B))
matop_dest(::typeof(\), A::UnitUpperOrUnitLowerTriangular, B::QuasiSparseMatrix) =
    Matrix{LinearAlgebra._inner_type_promotion(\, eltype(A), eltype(B))}(undef, size(B))
matop_dest(::typeof(/), A::QuasiSparseMatrix, B) =
    Matrix{promote_op(/, eltype(A), eltype(B))}(undef, size(A))
matop_dest(::typeof(/), A::QuasiSparseMatrix, B::UnitUpperOrUnitLowerTriangular) =
    Matrix{LinearAlgebra._inner_type_promotion(/, eltype(A), eltype(B))}(undef, size(A))
matop_dest(::typeof(/), A::QuasiSparseMatrix, B::Diagonal) =
    similar(A , promote_op(/, eltype(A), eltype(B)), size(A))
## end of triangular

\(A::Transpose{<:Complex,<:Hermitian{<:Complex,<:AbstractSparseMatrixCSC}}, B::Vector) = copy(A) \ B

function rdiv!(A::AbstractSparseMatrixCSC, D::Diagonal)
    dd = D.diag
    if (k = length(dd)) ≠ size(A, 2)
        throw(DimensionMismatch("size(A, 2)=$(size(A, 2)) should be size(D, 1)=$k"))
    end
    nonz = nonzeros(A)
    @inbounds for j in 1:k
        ddj = dd[j]
        if iszero(ddj)
            throw(LinearAlgebra.SingularException(j))
        end
        for i in nzrange(A, j)
            nonz[i] /= ddj
        end
    end
    A
end

function ldiv!(D::Diagonal, A::Union{AbstractSparseMatrixCSC, AbstractSparseVector})
    # require_one_based_indexing(A)
    if size(A, 1) != length(D.diag)
        throw(DimensionMismatch("diagonal matrix is $(length(D.diag)) by $(length(D.diag)) but right hand side has $(size(A, 1)) rows"))
    end
    nonz = nonzeros(A)
    Arowval = rowvals(A)
    b = D.diag
    @inbounds for i=axes(b,1)
        iszero(b[i]) && throw(SingularException(i))
    end
    @inbounds for col in axes(A,2), p in nzrange(A, col)
        nonz[p] = b[Arowval[p]] \ nonz[p]
    end
    A
end

## triu, tril

function triu(S::AbstractSparseMatrixCSC{Tv,Ti}, k::Integer=0) where {Tv,Ti}
    m,n = size(S)
    colptr = Vector{Ti}(undef, n+1)
    nnz = 0
    @inbounds for col = 1 : min(max(k+1,1), n+1)
        colptr[col] = 1
    end
    @inbounds for col = max(k+1,1) : n
        for c1 in nzrange(S, col)
            rowvals(S)[c1] > col - k && break
            nnz += 1
        end
        colptr[col+1] = nnz+1
    end
    rowval = Vector{Ti}(undef, nnz)
    nzval = Vector{Tv}(undef, nnz)
    @inbounds for col = max(k+1,1) : n
        c1 = getcolptr(S)[col]
        for c2 in colptr[col]:colptr[col+1]-1
            rowval[c2] = rowvals(S)[c1]
            nzval[c2] = nonzeros(S)[c1]
            c1 += 1
        end
    end
    SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

function tril(S::AbstractSparseMatrixCSC{Tv,Ti}, k::Integer=0) where {Tv,Ti}
    m,n = size(S)
    colptr = Vector{Ti}(undef, n+1)
    nnz = 0
    colptr[1] = 1
    @inbounds for col = 1 : min(n, m+k)
        l1 = getcolptr(S)[col+1]-1
        for c1 = 0 : (l1 - getcolptr(S)[col])
            rowvals(S)[l1 - c1] < col - k && break
            nnz += 1
        end
        colptr[col+1] = nnz+1
    end
    @inbounds for col = max(min(n, m+k)+2,1) : n+1
        colptr[col] = nnz+1
    end
    rowval = Vector{Ti}(undef, nnz)
    nzval = Vector{Tv}(undef, nnz)
    @inbounds for col = 1 : min(n, m+k)
        c1 = getcolptr(S)[col+1]-1
        l2 = colptr[col+1]-1
        for c2 = 0 : l2 - colptr[col]
            rowval[l2 - c2] = rowvals(S)[c1]
            nzval[l2 - c2] = nonzeros(S)[c1]
            c1 -= 1
        end
    end
    SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

## diff

function sparse_diff1(S::AbstractSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    m,n = size(S)
    m > 1 || return SparseMatrixCSC(0, n, fill(one(Ti),n+1), Ti[], Tv[])
    colptr = Vector{Ti}(undef, n+1)
    numnz = 2 * nnz(S) # upper bound; will shrink later
    rowval = Vector{Ti}(undef, numnz)
    nzval = Vector{Tv}(undef, numnz)
    numnz = 0
    @inbounds colptr[1] = 1
    @inbounds for col = 1 : n
        last_row = 0
        last_val = 0
        for k in nzrange(S, col)
            row = rowvals(S)[k]
            val = nonzeros(S)[k]
            if row > 1
                if row == last_row + 1
                    nzval[numnz] += val
                    nzval[numnz]==zero(Tv) && (numnz -= 1)
                else
                    numnz += 1
                    rowval[numnz] = row - 1
                    nzval[numnz] = val
                end
            end
            if row < m
                numnz += 1
                rowval[numnz] = row
                nzval[numnz] = -val
            end
            last_row = row
            last_val = val
        end
        colptr[col+1] = numnz+1
    end
    deleteat!(rowval, numnz+1:length(rowval))
    deleteat!(nzval, numnz+1:length(nzval))
    return SparseMatrixCSC(m-1, n, colptr, rowval, nzval)
end

function sparse_diff2(a::AbstractSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    m,n = size(a)
    colptr = Vector{Ti}(undef, max(n,1))
    numnz = 2 * nnz(a) # upper bound; will shrink later
    rowval = Vector{Ti}(undef, numnz)
    nzval = Vector{Tv}(undef, numnz)

    z = zero(Tv)

    colptr_a = getcolptr(a)
    rowval_a = rowvals(a)
    nzval_a = nonzeros(a)

    @inbounds begin
        ptrS = 1
        colptr[1] = 1

        n == 0 && return SparseMatrixCSC(m, n, colptr, rowval, nzval)

        startA = colptr_a[1]
        stopA = colptr_a[2]

        rA = startA : stopA - 1
        rowvalA = rowval_a[rA]
        nzvalA = nzval_a[rA]
        lA = stopA - startA
    end

    @inbounds for col = 1:n-1
        startB, stopB = startA, stopA
        startA = colptr_a[col+1]
        stopA = colptr_a[col+2]

        rowvalB = rowvalA
        nzvalB = nzvalA
        lB = lA

        rA = startA : stopA - 1
        rowvalA = rowval_a[rA]
        nzvalA = nzval_a[rA]
        lA = stopA - startA

        ptrB = 1
        ptrA = 1

        while ptrA <= lA && ptrB <= lB
            rowA = rowvalA[ptrA]
            rowB = rowvalB[ptrB]
            if rowA < rowB
                rowval[ptrS] = rowA
                nzval[ptrS] = nzvalA[ptrA]
                ptrS += 1
                ptrA += 1
            elseif rowB < rowA
                rowval[ptrS] = rowB
                nzval[ptrS] = -nzvalB[ptrB]
                ptrS += 1
                ptrB += 1
            else
                res = nzvalA[ptrA] - nzvalB[ptrB]
                if res != z
                    rowval[ptrS] = rowA
                    nzval[ptrS] = res
                    ptrS += 1
                end
                ptrA += 1
                ptrB += 1
            end
        end

        while ptrA <= lA
            rowval[ptrS] = rowvalA[ptrA]
            nzval[ptrS] = nzvalA[ptrA]
            ptrS += 1
            ptrA += 1
        end

        while ptrB <= lB
            rowval[ptrS] = rowvalB[ptrB]
            nzval[ptrS] = -nzvalB[ptrB]
            ptrS += 1
            ptrB += 1
        end

        colptr[col+1] = ptrS
    end
    deleteat!(rowval, ptrS:length(rowval))
    deleteat!(nzval, ptrS:length(nzval))
    return SparseMatrixCSC(m, n-1, colptr, rowval, nzval)
end

diff(a::AbstractSparseMatrixCSC; dims::Integer) = dims==1 ? sparse_diff1(a) : sparse_diff2(a)

## norm and rank
norm(A::AbstractSparseMatrixCSC, p::Real=2) = norm(view(nonzeros(A), 1:nnz(A)), p)

function opnorm(A::AbstractSparseMatrixCSC, p::Real=2)
    m, n = size(A)
    if m == 0 || n == 0 || isempty(A)
        return float(real(zero(eltype(A))))
    elseif m == 1
        if p == 1
            return norm(nzvalview(A), Inf)
        elseif p == 2
            return norm(nzvalview(A), 2)
        elseif p == Inf
            return norm(nzvalview(A), 1)
        end
    elseif n == 1 && p in (1, 2, Inf)
        return norm(nzvalview(A), p)
    else
        Tnorm = typeof(float(real(zero(eltype(A)))))
        Tsum = promote_type(Float64,Tnorm)
        if p==1
            nA::Tsum = 0
            @inbounds for j in axes(A,2)
                colSum::Tsum = 0
                for i in nzrange(A, j)
                    colSum += abs(nonzeros(A)[i])
                end
                nA = max(nA, colSum)
            end
            return convert(Tnorm, nA)
        elseif p==2
            throw(ArgumentError("2-norm not yet implemented for sparse matrices. Try opnorm(Array(A)) or opnorm(A, p) where p=1 or Inf."))
        elseif p==Inf
            rowSum = zeros(Tsum,m)
            @inbounds for i in axes(nonzeros(A),1)
                rowSum[rowvals(A)[i]] += abs(nonzeros(A)[i])
            end
            return convert(Tnorm, maximum(rowSum))
        end
    end
    throw(ArgumentError("invalid operator p-norm p=$p. Valid: 1, Inf"))
end

# TODO rank

# cond
function cond(A::AbstractSparseMatrixCSC, p::Real=2)
    if p == 1
        normAinv = opnormestinv(A)
        normA = opnorm(A, 1)
        return normA * normAinv
    elseif p == Inf
        normAinv = opnormestinv(copy(A'))
        normA = opnorm(A, Inf)
        return normA * normAinv
    elseif p == 2
        throw(ArgumentError("only 1- and Inf-norm condition numbers are implemented for sparse matrices, for 2-norm try cond(Array(A), 2) instead"))
    else
        throw(ArgumentError("second argument must be either 1 or Inf, got $p"))
    end
end

function opnormestinv(A::AbstractSparseMatrixCSC{T}, t::Integer = min(2,maximum(size(A)))) where T
    maxiter = 5
    # Check the input
    n = checksquare(A)
    F = factorize(A)
    if t <= 0
        throw(ArgumentError("number of blocks must be a positive integer"))
    end
    if t > n
        throw(ArgumentError("number of blocks must not be greater than $n"))
    end
    ind = Vector{Int64}(undef, n)
    ind_hist = Vector{Int64}(undef, maxiter * t)

    Ti = typeof(float(zero(T)))

    S = zeros(T <: Real ? Int : Ti, n, t)

    function _any_abs_eq(v,n::Int)
        for vv in v
            if abs(vv)==n
                return true
            end
        end
        return false
    end

    # Generate the block matrix
    X = Matrix{Ti}(undef, n, t)
    @inbounds X[1:n,1] .= 1
    @inbounds for j = 2:t
        while true
            rand!(view(X,1:n,j), (-1, 1))
            yaux = X[1:n,j]' * X[1:n,1:j-1]
            if !_any_abs_eq(yaux,n)
                break
            end
        end
    end
    rmul!(X, inv(n))

    iter = 0
    local est
    local est_old
    est_ind = 0
    while iter < maxiter
        iter += 1
        Y = F \ X
        est = zero(real(eltype(Y)))
        est_ind = 0
        for i = 1:t
            y = norm(Y[1:n,i], 1)
            if y > est
                est = y
                est_ind = i
            end
        end
        if iter == 1
            est_old = est
        end
        if est > est_old || iter == 2
            ind_best = est_ind
        end
        if iter >= 2 && est <= est_old
            est = est_old
            break
        end
        est_old = est
        S_old = copy(S)
        for j = 1:t
            for i = 1:n
                S[i,j] = Y[i,j]==0 ? one(Y[i,j]) : sign(Y[i,j])
            end
        end

        if T <: Real
            # Check whether cols of S are parallel to cols of S or S_old
            for j = 1:t
                while true
                    repeated = false
                    if j > 1
                        saux = S[1:n,j]' * S[1:n,1:j-1]
                        if _any_abs_eq(saux,n)
                            repeated = true
                        end
                    end
                    if !repeated && 2^(n-1) ≥ 2t #we need enough non-parallel ±1 vectors
                        saux2 = S[1:n,j]' * S_old[1:n,1:t]
                        if _any_abs_eq(saux2,n)
                            repeated = true
                        end
                    end
                    if repeated
                        rand!(view(S,1:n,j), (-1, 1))
                    else
                        break
                    end
                end
            end
        end

        # Use the conjugate transpose
        Z = F' \ S
        h_max = zero(real(eltype(Z)))
        h = zeros(real(eltype(Z)), n)
        h_ind = 0
        for i in axes(A,1)
            h[i] = norm(Z[i,1:t], Inf)
            if h[i] > h_max
                h_max = h[i]
                h_ind = i
            end
            ind[i] = i
        end
        if iter >=2 && ind_best == h_ind
            break
        end
        p = sortperm(h, rev=true)
        h = h[p]
        permute!(ind, p)
        if t > 1
            addcounter = t
            elemcounter = 0
            while addcounter > 0 && elemcounter < n
                elemcounter = elemcounter + 1
                current_element = ind[elemcounter]
                found = false
                for i = 1:t * (iter - 1)
                    if current_element == ind_hist[i]
                        found = true
                        break
                    end
                end
                if !found
                    addcounter = addcounter - 1
                    for i = 1:current_element - 1
                        X[i,t-addcounter] = 0
                    end
                    X[current_element,t-addcounter] = 1
                    for i = current_element + 1:n
                        X[i,t-addcounter] = 0
                    end
                    ind_hist[iter * t - addcounter] = current_element
                else
                    if elemcounter == t && addcounter == t
                        break
                    end
                end
            end
        else
            ind_hist[1:t] = ind[1:t]
            for j = 1:t
                for i = 1:ind[j] - 1
                    X[i,j] = 0
                end
                X[ind[j],j] = 1
                for i = ind[j] + 1:n
                    X[i,j] = 0
                end
            end
        end
    end
    return est
end

## det, inv, cond

inv(A::AbstractSparseMatrixCSC) = error("The inverse of a sparse matrix can often be dense and can cause the computer to run out of memory. If you are sure you have enough memory, please either convert your matrix to a dense matrix, e.g. by calling `Matrix` or if `A` can be factorized, use `\\` on the dense identity matrix, e.g. `A \\ Matrix{eltype(A)}(I, size(A)...)` restrictions of `\\` on sparse lhs applies. Alternatively, `A\\b` is generally preferable to `inv(A)*b`")

# TODO

## scale methods

# Copy colptr and rowval from one sparse matrix to another
function copyinds!(C::AbstractSparseMatrixCSC, A::AbstractSparseMatrixCSC; copy_rows=true, copy_cols=true)
    if copy_cols && getcolptr(C) !== getcolptr(A)
        resize!(getcolptr(C), length(getcolptr(A)))
        copyto!(getcolptr(C), getcolptr(A))
    end
    if copy_rows && rowvals(C) !== rowvals(A)
        resize!(rowvals(C), length(rowvals(A)))
        copyto!(rowvals(C), rowvals(A))
    end
end

"""
    rowcheck_index(A::AbstractSparseMatrixCSC, row::Integer, col::Integer)

Check if A[row, col] is a stored value, and return the index of the row in `rowvals(A)`.
Returns `(row_exists, row_ind)`, where `row_exists::Bool` signifies
whether the corresponding index is populated, and `row_ind` is the index.
If `row_exists` is `false`, the `row_ind` is the index where the value should be inserted into
`rowvals(A)` such that the subarray `@view rowvals(A)[nzrange(A, col)]` remains sorted.
"""
@inline function rowcheck_index(A::AbstractSparseMatrixCSC, row::Integer, col::Integer)
    nzinds = nzrange(A, col)
    rows_col = @view rowvals(A)[nzinds]
    # faster implementation of row ∈ rows_col and obtaining the index,
    # assuming that rows_col is sorted
    row_ind_col = searchsortedfirst(rows_col, row)
    row_exists = row_ind_col ∈ axes(rows_col,1) && rows_col[row_ind_col] == row
    row_ind = row_ind_col + first(nzinds) - firstindex(nzinds)
    row_exists, row_ind
end

"""
    mergeinds!(C::AbstractSparseMatrixCSC, A::AbstractSparseMatrixCSC)

Update `C` to contain stored values corresponding to the stored indices of `A`.
Stored indices common to `C` and `A` are not touched. Indices of `A` at which
`C` did not have a stored value are populated with zeros after the call.

# Examples
```jldoctest
julia> A = spzeros(3,3);

julia> A[4:4:8] .= 1;

julia> A
3×3 SparseMatrixCSC{Float64, Int64} with 2 stored entries:
  ⋅   1.0   ⋅
  ⋅    ⋅   1.0
  ⋅    ⋅    ⋅

julia> C = spzeros(3,3);

julia> C[2:4:6] .= 2;

julia> C
3×3 SparseMatrixCSC{Float64, Int64} with 2 stored entries:
  ⋅    ⋅    ⋅
 2.0   ⋅    ⋅
  ⋅   2.0   ⋅

julia> SparseArrays.mergeinds!(C, A)
3×3 SparseMatrixCSC{Float64, Int64} with 4 stored entries:
  ⋅   0.0   ⋅
 2.0   ⋅   0.0
  ⋅   2.0   ⋅
```
"""
function mergeinds!(C::AbstractSparseMatrixCSC, A::AbstractSparseMatrixCSC)
    C_colptr = getcolptr(C)
    for col in axes(A,2)
        n_extra = 0
        for ind in @inbounds nzrange(A, col)
            row = @inbounds rowvals(A)[ind]
            row_exists, ind = rowcheck_index(C, row, col)
            if !row_exists
                _is_fixed(C) && throw(ArgumentError(lazy"cannot store entry ($row, $col) in a fixed sparse matrix whose pattern lacks it"))
                n_extra += 1
                insert!(rowvals(C), ind, row)
                insert!(nonzeros(C), ind, zero(eltype(C)))
                C_colptr[col+1] += 1
            end
        end
        if !iszero(n_extra)
            @views C_colptr[col+2:end] .+= n_extra
        end
    end
    C
end

function ldiv!(C::AbstractSparseMatrixCSC, D::Diagonal, A::AbstractSparseMatrixCSC)
    m, n = size(A)
    b    = D.diag
    lb = length(b)
    m==lb || throw(DimensionMismatch("D has size ($lb, $lb) but A has size ($m, $n)"))
    szC = size(C)
    size(A) == szC || throw(DimensionMismatch("A has size ($m, $n), D has size ($lb, $lb), C has size $szC"))
    copyinds!(C, A)
    Cnzval = nonzeros(C)
    Anzval = nonzeros(A)
    Arowval = rowvals(A)
    resize!(Cnzval, length(Anzval))
    for col in axes(A,2), p in nzrange(A, col)
        @inbounds Cnzval[p] = b[Arowval[p]] \ Anzval[p]
    end
    C
end

function LinearAlgebra._rdiv!(C::AbstractSparseMatrixCSC, A::AbstractSparseMatrixCSC, D::Diagonal)
    m, n = size(A)
    b    = D.diag
    lb = length(b)
    n == lb || throw(DimensionMismatch("A has size ($m, $n) but D has size ($lb, $lb)"))
    szC = size(C)
    size(A) == szC || throw(DimensionMismatch("A has size ($m, $n), D has size ($lb, $lb), C has size $szC"))
    copyinds!(C, A)
    Cnzval = nonzeros(C)
    Anzval = nonzeros(A)
    resize!(Cnzval, length(Anzval))
    for col in axes(A,2), p in nzrange(A, col)
        @inbounds Cnzval[p] = Anzval[p] / b[col]
    end
    C
end

function \(A::AbstractSparseMatrixCSC, B::AbstractVecOrMat)
    require_one_based_indexing(A, B)
    m, n = size(A)
    if m == n
        if istril(A)
            if istriu(A)
                return \(Diagonal(Vector(diag(A))), B)
            else
                return \(LowerTriangular(A), B)
            end
        elseif istriu(A)
            return \(UpperTriangular(A), B)
        end
        if ishermitian(A)
            return \(Hermitian(A), B)
        end
        return convert(AbstractArray{typeof(one(eltype(A)) \ one(eltype(B)))}, \(lu(A), B))
    elseif m > n
        return \(qr(A), B)
    else
        # A is wide, so the LQ factorization gives the minimum-norm solution
        return \(lq(A), B)
    end
end
for (xformtype, xformop) in ((:Adjoint, :adjoint), (:Transpose, :transpose))
    @eval begin
        function \(xformA::($xformtype){<:Any,<:AbstractSparseMatrixCSC}, B::AbstractVecOrMat)
            A = parent(xformA)
            require_one_based_indexing(A, B)
            m, n = size(A)
            if m == n
                if istril(A)
                    if istriu(A)
                        return \(Diagonal(($xformop.(diag(A)))), B)
                    else
                        return \(UpperTriangular($xformop(A)), B)
                    end
                elseif istriu(A)
                    return \(LowerTriangular($xformop(A)), B)
                end
                if ishermitian(A)
                    return \($xformop(Hermitian(A)), B)
                end
                return \($xformop(lu(A)), B)
            elseif m > n
                # A' is wide, so solve the underdetermined system with the
                # factorization of A itself, which gives the minimum-norm solution
                return \($xformop(qr(A)), B)
            else
                # A' is tall, so the least squares solve needs a factorization of A'
                return \(qr($xformop(A)), B)
            end
        end
    end
end

function factorize(A::AbstractSparseMatrixCSC)
    m, n = size(A)
    if m == n
        if istril(A)
            if istriu(A)
                return Diagonal(A)
            else
                return LowerTriangular(A)
            end
        elseif istriu(A)
            return UpperTriangular(A)
        end
        if ishermitian(A)
            return factorize(Hermitian(A))
        end
        return lu(A)
    elseif m > n
        return qr(A)
    else
        # A is wide, so solving with the LQ factorization gives the minimum-norm solution
        return lq(A)
    end
end

function factorize(A::RealHermSymComplexHerm{<:Union{Float32,Float64},<:AbstractSparseMatrixCSC})
    F = cholesky(A; check = false)
    if LinearAlgebra.issuccess(F)
        return F
    else
        ldlt!(F, A)
        return F
    end
end

eigen(A::AbstractSparseMatrixCSC) =
    error("eigen(A) not supported for sparse matrices. Use for example eigs(A) from the Arpack package instead.")
