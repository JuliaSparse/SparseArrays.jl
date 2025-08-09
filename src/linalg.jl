# This file is a part of Julia. License is MIT: https://julialang.org/license

using LinearAlgebra: AbstractTriangular, StridedMaybeAdjOrTransMat, UpperOrLowerTriangular,
    RealHermSymComplexHerm, checksquare, sym_uplo, wrap
using Random: rand!

const tilebufsize = 10800  # Approximately 32k/3

# In matrix-vector multiplication, the correct orientation of the vector is assumed.
const DenseMatrixUnion = Union{StridedMatrix, BitMatrix}
const DenseTriangular  = UpperOrLowerTriangular{<:Any,<:DenseMatrixUnion}
const DenseInputVector = Union{StridedVector, BitVector}
const DenseVecOrMat = Union{DenseMatrixUnion, DenseInputVector}

matprod_dest(A::SparseMatrixCSCUnion2, B::DenseTriangular, TS) =
    similar(B, TS, (size(A, 1), size(B, 2)))
matprod_dest(A::AdjOrTrans{<:Any,<:SparseMatrixCSCUnion2}, B::DenseTriangular, TS) =
    similar(B, TS, (size(A, 1), size(B, 2)))
matprod_dest(A::StridedMaybeAdjOrTransMat, B::SparseMatrixCSCUnion2, TS) =
    similar(A, TS, (size(A, 1), size(B, 2)))
matprod_dest(A::Union{BitMatrix,AdjOrTrans{<:Any,BitMatrix}}, B::SparseMatrixCSCUnion2, TS) =
    similar(A, TS, (size(A, 1), size(B, 2)))
matprod_dest(A::DenseTriangular, B::SparseMatrixCSCUnion2, TS) =
    similar(A, TS, (size(A, 1), size(B, 2)))
matprod_dest(A::StridedMaybeAdjOrTransMat, B::AdjOrTrans{<:Any,<:SparseMatrixCSCUnion2}, TS) =
    similar(A, TS, (size(A, 1), size(B, 2)))
matprod_dest(A::Union{BitMatrix,AdjOrTrans{<:Any,BitMatrix}}, B::AdjOrTrans{<:Any,<:SparseMatrixCSCUnion2}, TS) =
    similar(A, TS, (size(A, 1), size(B, 2)))
matprod_dest(A::DenseTriangular, B::AdjOrTrans{<:Any,<:SparseMatrixCSCUnion2}, TS) =
    similar(A, TS, (size(A, 1), size(B, 2)))

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

@inline generic_matmatmul!(C::StridedMatrix, tA, tB, A::SparseMatrixCSCUnion2, B::DenseMatrixUnion, alpha::Number, beta::Number) =
    spdensemul!(C, tA, tB, A, B, alpha, beta)
@inline generic_matmatmul!(C::StridedMatrix, tA, tB, A::SparseMatrixCSCUnion2, B::AbstractTriangular, alpha::Number, beta::Number) =
    spdensemul!(C, tA, tB, A, B, alpha, beta)
@inline generic_matvecmul!(C::StridedVecOrMat, tA, A::SparseMatrixCSCUnion2, B::DenseInputVector, alpha::Number, beta::Number) =
    spdensemul!(C, tA, 'N', A, B, alpha, beta)

Base.@constprop :aggressive function spdensemul!(C, tA, tB, A, B, alpha, beta)
    tA_uc, tB_uc = uppercase(tA), uppercase(tB)
    if tA_uc == 'N'
        _spmatmul!(C, A, wrap(B, tB), alpha, beta)
    elseif tA_uc == 'T'
        _At_or_Ac_mul_B!(transpose, C, A, wrap(B, tB), alpha, beta)
    elseif tA_uc == 'C'
        _At_or_Ac_mul_B!(adjoint, C, A, wrap(B, tB), alpha, beta)
    elseif tA_uc in ('S', 'H') && tB_uc == 'N'
        rangefun = isuppercase(tA) ? nzrangeup : nzrangelo
        diagop = tA_uc == 'S' ? identity : real
        odiagop = tA_uc == 'S' ? transpose : adjoint
        T = eltype(C)
        _mul!(rangefun, diagop, odiagop, C, A, B, T(alpha), T(beta))
    else
        @stable_muladdmul LinearAlgebra._generic_matmatmul!(C, 'N', 'N', wrap(A, tA), wrap(B, tB), MulAddMul(alpha, beta))
    end
    return C
end

function _spmatmul!(C, A, B, α, β)
    size(A, 2) == size(B, 1) ||
        throw(DimensionMismatch("second dimension of A, $(size(A,2)), does not match the first dimension of B, $(size(B,1))"))
    size(A, 1) == size(C, 1) ||
        throw(DimensionMismatch("first dimension of A, $(size(A,1)), does not match the first dimension of C, $(size(C,1))"))
    size(B, 2) == size(C, 2) ||
        throw(DimensionMismatch("second dimension of B, $(size(B,2)), does not match the second dimension of C, $(size(C,2))"))
    nzv = nonzeros(A)
    rv = rowvals(A)
    β != one(β) && LinearAlgebra._rmul_or_fill!(C, β)
    for k in axes(C, 2)
        @inbounds for col in axes(A,2)
            αxj = B[col,k] * α
            for j in nzrange(A, col)
                C[rv[j], k] += nzv[j]*αxj
            end
        end
    end
    C
end

function _At_or_Ac_mul_B!(tfun::Function, C, A, B, α, β)
    size(A, 2) == size(C, 1) ||
        throw(DimensionMismatch("second dimension of A, $(size(A,2)), does not match the first dimension of C, $(size(C,1))"))
    size(A, 1) == size(B, 1) ||
        throw(DimensionMismatch("first dimension of A, $(size(A,1)), does not match the first dimension of B, $(size(B,1))"))
    size(B, 2) == size(C, 2) ||
        throw(DimensionMismatch("second dimension of B, $(size(B,2)), does not match the second dimension of C, $(size(C,2))"))
    nzv = nonzeros(A)
    rv = rowvals(A)
    β != one(β) && LinearAlgebra._rmul_or_fill!(C, β)
    for k in axes(C, 2)
        @inbounds for col in axes(A,2)
            tmp = zero(eltype(C))
            for j in nzrange(A, col)
                tmp += tfun(nzv[j])*B[rv[j],k]
            end
            C[col,k] += tmp * α
        end
    end
    C
end

Base.@constprop :aggressive function generic_matmatmul!(C::StridedMatrix, tA, tB, A::DenseMatrixUnion, B::SparseMatrixCSCUnion2, alpha::Number, beta::Number)
    transA = tA == 'N' ? identity : tA == 'T' ? transpose : adjoint
    if tB == 'N'
        _spmul!(C, transA(A), B, alpha, beta)
    elseif tB == 'T'
        _A_mul_Bt_or_Bc!(transpose, C, transA(A), B, alpha, beta)
    else # tB == 'C'
        _A_mul_Bt_or_Bc!(adjoint, C, transA(A), B, alpha, beta)
    end
    return C
end
function _spmul!(C::StridedMatrix, X::DenseMatrixUnion, A::SparseMatrixCSCUnion2, α::Number, β::Number)
    mX, nX = size(X)
    nX == size(A, 1) ||
        throw(DimensionMismatch("second dimension of X, $nX, does not match the first dimension of A, $(size(A,1))"))
    mX == size(C, 1) ||
        throw(DimensionMismatch("first dimension of X, $mX, does not match the first dimension of C, $(size(C,1))"))
    size(A, 2) == size(C, 2) ||
        throw(DimensionMismatch("second dimension of A, $(size(A,2)), does not match the second dimension of C, $(size(C,2))"))
    rv = rowvals(A)
    nzv = nonzeros(A)
    β != one(β) && LinearAlgebra._rmul_or_fill!(C, β)
    @inbounds for col in axes(A,2), k in nzrange(A, col)
        Aiα = nzv[k] * α
        rvk = rv[k]
        @simd for multivec_row in axes(X,1)
            C[multivec_row, col] += X[multivec_row, rvk] * Aiα
        end
    end
    C
end
function _spmul!(C::StridedMatrix, X::AdjOrTrans{<:Any,<:DenseMatrixUnion}, A::SparseMatrixCSCUnion2, α::Number, β::Number)
    mX, nX = size(X)
    nX == size(A, 1) ||
        throw(DimensionMismatch("second dimension of X, $nX, does not match the first dimension of A, $(size(A,1))"))
    mX == size(C, 1) ||
        throw(DimensionMismatch("first dimension of X, $mX, does not match the first dimension of C, $(size(C,1))"))
    size(A, 2) == size(C, 2) ||
        throw(DimensionMismatch("second dimension of A, $(size(A,2)), does not match the second dimension of C, $(size(C,2))"))
    rv = rowvals(A)
    nzv = nonzeros(A)
    β != one(β) && LinearAlgebra._rmul_or_fill!(C, β)
    for multivec_row in axes(X,1), col in axes(C, 2)
        @inbounds for k in nzrange(A, col)
            C[multivec_row, col] += X[multivec_row, rv[k]] * nzv[k] * α
        end
    end
    C
end

function _A_mul_Bt_or_Bc!(tfun::Function, C::StridedMatrix, A::AbstractMatrix, B::SparseMatrixCSCUnion2, α::Number, β::Number)
    mA, nA = size(A)
    nA == size(B, 2) ||
        throw(DimensionMismatch("second dimension of A, $nA, does not match the second dimension of B, $(size(B,2))"))
    mA == size(C, 1) ||
        throw(DimensionMismatch("first dimension of A, $mA, does not match the first dimension of C, $(size(C,1))"))
    size(B, 1) == size(C, 2) ||
        throw(DimensionMismatch("first dimension of B, $(size(B,2)), does not match the second dimension of C, $(size(C,2))"))
    rv = rowvals(B)
    nzv = nonzeros(B)
    β != one(β) && LinearAlgebra._rmul_or_fill!(C, β)
    @inbounds for col in axes(B, 2), k in nzrange(B, col)
        Biα = tfun(nzv[k]) * α
        rvk = rv[k]
        @simd for multivec_col in axes(A,1)
            C[multivec_col, rvk] += A[multivec_col, col] * Biα
        end
    end
    C
end

function *(A::Diagonal, b::AbstractSparseVector)
    if size(A, 2) != length(b)
        throw(
            DimensionMismatch(lazy"The dimension of the matrix A $(size(A)) and of the vector b $(length(b))")
        )
    end
    T = promote_eltype(A, b)
    res = similar(b, T)
    nzind_b = nonzeroinds(b)
    nzval_b = nonzeros(b)
    nzval_res = nonzeros(res)
    for idx in eachindex(nzind_b)
        nzval_res[idx] = A.diag[nzind_b[idx]] * nzval_b[idx]
    end
    return res
end

# Sparse matrix multiplication as described in [Gustavson, 1978]:
# http://dl.acm.org/citation.cfm?id=355796

const SparseTriangular{Tv,Ti} = Union{UpperTriangular{Tv,<:SparseMatrixCSCUnion{Tv,Ti}},LowerTriangular{Tv,<:SparseMatrixCSCUnion{Tv,Ti}}}
const SparseOrTri{Tv,Ti} = Union{SparseMatrixCSCUnion{Tv,Ti},SparseTriangular{Tv,Ti}}

*(A::SparseOrTri, B::AbstractSparseVector) = spmatmulv(A, B)
*(A::SparseOrTri, B::SparseColumnView) = spmatmulv(A, B)
*(A::SparseOrTri, B::SparseVectorView) = spmatmulv(A, B)
*(A::SparseMatrixCSCUnion, B::SparseMatrixCSCUnion) = spmatmul(A,B)
*(A::SparseTriangular, B::SparseMatrixCSCUnion) = spmatmul(A,B)
*(A::SparseMatrixCSCUnion, B::SparseTriangular) = spmatmul(A,B)
*(A::SparseTriangular, B::SparseTriangular) = spmatmul1(A,B)
*(A::SparseOrTri, B::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}) = spmatmul(A, copy(B))
*(A::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}, B::SparseOrTri) = spmatmul(copy(A), B)
*(A::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}, B::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}) = spmatmul(copy(A), copy(B))

(*)(Da::Diagonal, A::Union{SparseMatrixCSCUnion, AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}}, Db::Diagonal) = Da * (A * Db)
function (*)(Da::Diagonal, A::SparseMatrixCSC, Db::Diagonal)
    (size(Da, 2) == size(A,1) && size(A,2) == size(Db,1)) ||
        throw(DimensionMismatch("incompatible sizes"))
    T = promote_op(matprod, eltype(Da), promote_op(matprod, eltype(A), eltype(Db)))
    dest = similar(A, T)
    vals_dest = nonzeros(dest)
    rows = rowvals(A)
    vals = nonzeros(A)
    da, db = map(parent, (Da, Db))
    for col in axes(A,2)
        dbcol = db[col]
        for i in nzrange(A, col)
            row = rows[i]
            val = vals[i]
            vals_dest[i] = da[row] * val * dbcol
        end
    end
    dest
end

# Gustavson's matrix multiplication algorithm revisited.
# The result rowval vector is already sorted by construction.
# The auxiliary Vector{Ti} xb is replaced by a Vector{Bool} of same length.
# The optional argument controlling a sorting algorithm is obsolete.
# depending on expected execution speed the sorting of the result column is
# done by a quicksort of the row indices or by a full scan of the dense result vector.
# The last is faster, if more than ≈ 1/32 of the result column is nonzero.
# TODO: extend to SparseMatrixCSCUnion to allow for SubArrays (view(X, :, r)).
function spmatmul(A::SparseOrTri, B::Union{SparseOrTri,AbstractCompressedVector,SubArray{<:Any,<:Any,<:AbstractSparseArray}})
    Tv = promote_op(matprod, eltype(A), eltype(B))
    Ti = promote_type(indtype(A), indtype(B))
    mA, nA = size(A)
    nB = size(B, 2)
    mB = size(B, 1)
    nA == mB || throw(DimensionMismatch("second dimension of A, $nA, does not match the first dimension of B, $mB"))

    nnzC = min(estimate_mulsize(mA, nnz(A), nA, nnz(B), nB) * 11 ÷ 10 + mA, mA*nB)
    colptrC = Vector{Ti}(undef, nB+1)
    rowvalC = Vector{Ti}(undef, nnzC)
    nzvalC = Vector{Tv}(undef, nnzC)

    @inbounds begin
        ip = 1
        xb = fill(false, mA)
        for i in axes(B,2)
            if ip + mA - 1 > nnzC
                nnzC += max(mA, nnzC>>2)
                resize!(rowvalC, nnzC)
                resize!(nzvalC, nnzC)
            end
            colptrC[i] = ip
            ip = spcolmul!(rowvalC, nzvalC, xb, i, ip, A, B)
        end
        colptrC[nB+1] = ip
    end

    resize!(rowvalC, ip - 1)
    resize!(nzvalC, ip - 1)

    # This modification of Gustavson algorithm has sorted row indices
    C = SparseMatrixCSC(mA, nB, colptrC, rowvalC, nzvalC)
    return C
end

# process single rhs column
function spcolmul!(rowvalC, nzvalC, xb, i, ip, A, B)
    rowvalA = rowvals(A); nzvalA = nonzeros(A)
    rowvalB = rowvals(B); nzvalB = nonzeros(B)
    mA = size(A, 1)
    ip0 = ip
    k0 = ip - 1
    @inbounds begin
        for jp in nzrange(B, i)
            nzB = nzvalB[jp]
            j = rowvalB[jp]
            for kp in nzrange(A, j)
                nzC = nzvalA[kp] * nzB
                k = rowvalA[kp]
                if xb[k]
                    nzvalC[k+k0] += nzC
                else
                    nzvalC[k+k0] = nzC
                    xb[k] = true
                    rowvalC[ip] = k
                    ip += 1
                end
            end
        end
        if ip > ip0
            if prefer_sort(ip-k0, mA)
                # in-place sort of indices. Effort: O(nnz*ln(nnz)).
                sort!(rowvalC, ip0, ip-1, QuickSort, Base.Order.Forward)
                for vp = ip0:ip-1
                    k = rowvalC[vp]
                    xb[k] = false
                    nzvalC[vp] = nzvalC[k+k0]
                end
            else
                # scan result vector (effort O(mA))
                for k in axes(A,1)
                    if xb[k]
                        xb[k] = false
                        rowvalC[ip0] = k
                        nzvalC[ip0] = nzvalC[k+k0]
                        ip0 += 1
                    end
                end
            end
        end
    end
    return ip
end

# special cases of same twin Upper/LowerTriangular
spmatmul1(A, B) = spmatmul(A, B)
function spmatmul1(A::UpperTriangular, B::UpperTriangular)
    UpperTriangular(spmatmul(A, B))
end
function spmatmul1(A::LowerTriangular, B::LowerTriangular)
    LowerTriangular(spmatmul(A, B))
end
# exploit spmatmul for sparse vectors and column views
function spmatmulv(A, B)
    spmatmul(A, B)[:,1]
end

# estimated number of non-zeros in matrix product
# it is assumed, that the non-zero indices are distributed independently and uniformly
# in both matrices. Over-estimation is possible if that is not the case.
function estimate_mulsize(m::Integer, nnzA::Integer, n::Integer, nnzB::Integer, k::Integer)
    p = (nnzA / (m * n)) * (nnzB / (n * k))
    p >= 1 ? m*k : p > 0 ? Int(ceil(-expm1(log1p(-p) * n)*m*k)) : 0 # (1-(1-p)^n)*m*k
end

Base.@constprop :aggressive function generic_matmatmul!(C::SparseMatrixCSCUnion2, tA, tB, A::SparseMatrixCSCUnion2,
                            B::SparseMatrixCSCUnion2, alpha::Number, beta::Number)
    tA_uc, tB_uc = uppercase(tA), uppercase(tB)
    Anew, ta = tA_uc in ('S', 'H') ? (wrap(A, tA), oftype(tA, 'N')) : (A, tA)
    Bnew, tb = tB_uc in ('S', 'H') ? (wrap(B, tB), oftype(tB, 'N')) : (B, tB)
    @stable_muladdmul _generic_matmatmul!(C, ta, tb, Anew, Bnew, MulAddMul(alpha, beta))
end
function _generic_matmatmul!(C::SparseMatrixCSCUnion2, tA, tB, A::AbstractVecOrMat,
                                B::AbstractVecOrMat, _add::MulAddMul)
    @assert tA in ('N', 'T', 'C') && tB in ('N', 'T', 'C')
    require_one_based_indexing(C, A, B)
    R = eltype(C)
    T = eltype(A)
    S = eltype(B)

    mA, nA = LinearAlgebra.lapack_size(tA, A)
    mB, nB = LinearAlgebra.lapack_size(tB, B)
    if mB != nA
        throw(DimensionMismatch(lazy"matrix A has dimensions ($mA,$nA), matrix B has dimensions ($mB,$nB)"))
    end
    if size(C,1) != mA || size(C,2) != nB
        throw(DimensionMismatch(lazy"result C has dimensions $(size(C)), needs ($mA,$nB)"))
    end

    if iszero(_add.alpha) || isempty(A) || isempty(B)
        return LinearAlgebra._rmul_or_fill!(C, _add.beta)
    end

    tile_size = 0
    if isbitstype(R) && isbitstype(T) && isbitstype(S) && (tA == 'N' || tB != 'N')
        tile_size = floor(Int, sqrt(tilebufsize / max(sizeof(R), sizeof(S), sizeof(T), 1)))
    end
    @inbounds begin
    if tile_size > 0
        sz = (tile_size, tile_size)
        Atile = Array{T}(undef, sz)
        Btile = Array{S}(undef, sz)

        z1 = zero(A[1, 1]*B[1, 1] + A[1, 1]*B[1, 1])
        z = convert(promote_type(typeof(z1), R), z1)

        if mA < tile_size && nA < tile_size && nB < tile_size
            copy_transpose!(Atile, 1:nA, 1:mA, tA, A, 1:mA, 1:nA)
            copyto!(Btile, 1:mB, 1:nB, tB, B, 1:mB, 1:nB)
            for j = 1:nB
                boff = (j-1)*tile_size
                for i = 1:mA
                    aoff = (i-1)*tile_size
                    s = z
                    for k = 1:nA
                        s += Atile[aoff+k] * Btile[boff+k]
                    end
                    LinearAlgebra._modify!(_add, s, C, (i,j))
                end
            end
        else
            Ctile = Array{R}(undef, sz)
            for jb = 1:tile_size:nB
                jlim = min(jb+tile_size-1,nB)
                jlen = jlim-jb+1
                for ib = 1:tile_size:mA
                    ilim = min(ib+tile_size-1,mA)
                    ilen = ilim-ib+1
                    fill!(Ctile, z)
                    for kb = 1:tile_size:nA
                        klim = min(kb+tile_size-1,mB)
                        klen = klim-kb+1
                        copy_transpose!(Atile, 1:klen, 1:ilen, tA, A, ib:ilim, kb:klim)
                        copyto!(Btile, 1:klen, 1:jlen, tB, B, kb:klim, jb:jlim)
                        for j=1:jlen
                            bcoff = (j-1)*tile_size
                            for i = 1:ilen
                                aoff = (i-1)*tile_size
                                s = z
                                for k = 1:klen
                                    s += Atile[aoff+k] * Btile[bcoff+k]
                                end
                                Ctile[bcoff+i] += s
                            end
                        end
                    end
                    if isone(_add.alpha) && iszero(_add.beta)
                        copyto!(C, ib:ilim, jb:jlim, Ctile, 1:ilen, 1:jlen)
                    else
                        C[ib:ilim, jb:jlim] .= @views _add.(Ctile[1:ilen, 1:jlen], C[ib:ilim, jb:jlim])
                    end
                end
            end
        end
    else
        # Multiplication for non-plain-data uses the naive algorithm
        if tA == 'N'
            if tB == 'N'
                for i = 1:mA, j = 1:nB
                    z2 = zero(A[i, 1]*B[1, j] + A[i, 1]*B[1, j])
                    Ctmp = convert(promote_type(R, typeof(z2)), z2)
                    for k = 1:nA
                        Ctmp += A[i, k]*B[k, j]
                    end
                    LinearAlgebra._modify!(_add, Ctmp, C, (i,j))
                end
            elseif tB == 'T'
                for i = 1:mA, j = 1:nB
                    z2 = zero(A[i, 1]*transpose(B[j, 1]) + A[i, 1]*transpose(B[j, 1]))
                    Ctmp = convert(promote_type(R, typeof(z2)), z2)
                    for k = 1:nA
                        Ctmp += A[i, k] * transpose(B[j, k])
                    end
                    LinearAlgebra._modify!(_add, Ctmp, C, (i,j))
                end
            else
                for i = 1:mA, j = 1:nB
                    z2 = zero(A[i, 1]*B[j, 1]' + A[i, 1]*B[j, 1]')
                    Ctmp = convert(promote_type(R, typeof(z2)), z2)
                    for k = 1:nA
                        Ctmp += A[i, k]*B[j, k]'
                    end
                    LinearAlgebra._modify!(_add, Ctmp, C, (i,j))
                end
            end
        elseif tA == 'T'
            if tB == 'N'
                for i = 1:mA, j = 1:nB
                    z2 = zero(transpose(A[1, i])*B[1, j] + transpose(A[1, i])*B[1, j])
                    Ctmp = convert(promote_type(R, typeof(z2)), z2)
                    for k = 1:nA
                        Ctmp += transpose(A[k, i]) * B[k, j]
                    end
                    LinearAlgebra._modify!(_add, Ctmp, C, (i,j))
                end
            elseif tB == 'T'
                for i = 1:mA, j = 1:nB
                    z2 = zero(transpose(A[1, i])*transpose(B[j, 1]) + transpose(A[1, i])*transpose(B[j, 1]))
                    Ctmp = convert(promote_type(R, typeof(z2)), z2)
                    for k = 1:nA
                        Ctmp += transpose(A[k, i]) * transpose(B[j, k])
                    end
                    LinearAlgebra._modify!(_add, Ctmp, C, (i,j))
                end
            else
                for i = 1:mA, j = 1:nB
                    z2 = zero(transpose(A[1, i])*B[j, 1]' + transpose(A[1, i])*B[j, 1]')
                    Ctmp = convert(promote_type(R, typeof(z2)), z2)
                    for k = 1:nA
                        Ctmp += transpose(A[k, i]) * adjoint(B[j, k])
                    end
                    LinearAlgebra._modify!(_add, Ctmp, C, (i,j))
                end
            end
        else
            if tB == 'N'
                for i = 1:mA, j = 1:nB
                    z2 = zero(A[1, i]'*B[1, j] + A[1, i]'*B[1, j])
                    Ctmp = convert(promote_type(R, typeof(z2)), z2)
                    for k = 1:nA
                        Ctmp += A[k, i]'B[k, j]
                    end
                    LinearAlgebra._modify!(_add, Ctmp, C, (i,j))
                end
            elseif tB == 'T'
                for i = 1:mA, j = 1:nB
                    z2 = zero(A[1, i]'*transpose(B[j, 1]) + A[1, i]'*transpose(B[j, 1]))
                    Ctmp = convert(promote_type(R, typeof(z2)), z2)
                    for k = 1:nA
                        Ctmp += adjoint(A[k, i]) * transpose(B[j, k])
                    end
                    LinearAlgebra._modify!(_add, Ctmp, C, (i,j))
                end
            else
                for i = 1:mA, j = 1:nB
                    z2 = zero(A[1, i]'*B[j, 1]' + A[1, i]'*B[j, 1]')
                    Ctmp = convert(promote_type(R, typeof(z2)), z2)
                    for k = 1:nA
                        Ctmp += A[k, i]'B[j, k]'
                    end
                    LinearAlgebra._modify!(_add, Ctmp, C, (i,j))
                end
            end
        end
    end
    end # @inbounds
    C
end

if VERSION < v"1.10.0-DEV.299"
    top_set_bit(x::Base.BitInteger) = 8 * sizeof(x) - leading_zeros(x)
else
    top_set_bit(x::Base.BitInteger) = Base.top_set_bit(x)
end
# determine if sort! shall be used or the whole column be scanned
# based on empirical data on i7-3610QM CPU
# measuring runtimes of the scanning and sorting loops of the algorithm.
# The parameters 6 and 3 might be modified for different architectures.
prefer_sort(nz::Integer, m::Integer) = m > 6 && 3 * top_set_bit(nz) * nz < m

# Frobenius dot/inner product: trace(A'B)
function dot(A::AbstractSparseMatrixCSC{T1,S1},B::AbstractSparseMatrixCSC{T2,S2}) where {T1,T2,S1,S2}
    m, n = size(A)
    size(B) == (m,n) || throw(DimensionMismatch("matrices must have the same dimensions"))
    r = dot(zero(T1), zero(T2))
    @inbounds for j in axes(A,2)
        ia = getcolptr(A)[j]; ia_nxt = getcolptr(A)[j+1]
        ib = getcolptr(B)[j]; ib_nxt = getcolptr(B)[j+1]
        if ia < ia_nxt && ib < ib_nxt
            ra = rowvals(A)[ia]; rb = rowvals(B)[ib]
            while true
                if ra < rb
                    ia += oneunit(S1)
                    ia < ia_nxt || break
                    ra = rowvals(A)[ia]
                elseif ra > rb
                    ib += oneunit(S2)
                    ib < ib_nxt || break
                    rb = rowvals(B)[ib]
                else # ra == rb
                    r += dot(nonzeros(A)[ia], nonzeros(B)[ib])
                    ia += oneunit(S1); ib += oneunit(S2)
                    ia < ia_nxt && ib < ib_nxt || break
                    ra = rowvals(A)[ia]; rb = rowvals(B)[ib]
                end
            end
        end
    end
    return r
end

function dot(x::AbstractVector{T1}, A::AbstractSparseMatrixCSC{T2}, y::AbstractVector{T3}) where {T1,T2,T3}
    require_one_based_indexing(x, y)
    m, n = size(A)
    (length(x) == m && n == length(y)) ||
        throw(DimensionMismatch("x has length $(length(x)), A has size ($m, $n), y has length $(length(y))"))
    s = dot(zero(T1), zero(T2), zero(T3))
    T = typeof(s)
    (iszero(m) || iszero(n)) && return s

    rowvals = getrowval(A)
    nzvals = getnzval(A)

    @inbounds @simd for col in axes(A,2)
        ycol = y[col]
        for j in nzrange(A, col)
            row = rowvals[j]
            val = nzvals[j]
            s += dot(x[row], val, ycol)
        end
    end
    return s
end
function dot(x::SparseVector, A::AbstractSparseMatrixCSC, y::SparseVector)
    m, n = size(A)
    length(x) == m && n == length(y) ||
        throw(DimensionMismatch("x has length $(length(x)), A has size ($m, $n), y has length $(length(y))"))
    if iszero(m) || iszero(n)
        return dot(zero(eltype(x)), zero(eltype(A)), zero(eltype(y)))
    end
    r = zero(promote_type(eltype(x), eltype(A), eltype(y)))
    xnzind = nonzeroinds(x)
    xnzval = nonzeros(x)
    ynzind = nonzeroinds(y)
    ynzval = nonzeros(y)
    Acolptr = getcolptr(A)
    Arowval = getrowval(A)
    Anzval = getnzval(A)
    for (yi, yv) in zip(ynzind, ynzval)
        A_ptr_lo = Acolptr[yi]
        A_ptr_hi = Acolptr[yi+1] - 1
        if A_ptr_lo <= A_ptr_hi
            r += _spdot(dot, 1, length(xnzind), xnzind, xnzval,
                                            A_ptr_lo, A_ptr_hi, Arowval, Anzval) * yv
        end
    end
    r
end

const WrapperMatrixTypes{T,MT} = Union{
    SubArray{T,2,MT},
    Adjoint{T,MT},
    Transpose{T,MT},
    UpperOrLowerTriangular{T,MT},
    UpperHessenberg{T,MT},
    Symmetric{T,MT},
    Hermitian{T,MT},
}

function dot(A::Union{DenseMatrixUnion,WrapperMatrixTypes{<:Any,<:Union{DenseMatrixUnion,AbstractSparseMatrix}}}, B::AbstractSparseMatrixCSC)
    T = promote_type(eltype(A), eltype(B))
    (m, n) = size(A)
    if (m, n) != size(B)
        throw(DimensionMismatch("A has size ($m, $n) but B has size $(size(B))"))
    end
    s = zero(T)
    if m * n == 0
        return s
    end
    rows = rowvals(B)
    vals = nonzeros(B)
    @inbounds for j in axes(A,2)
        for ridx in nzrange(B, j)
            i = rows[ridx]
            v = vals[ridx]
            s += dot(A[i,j], v)
        end
    end
    return s
end

function dot(A::AbstractSparseMatrixCSC, B::Union{DenseMatrixUnion,WrapperMatrixTypes{<:Any,<:Union{DenseMatrixUnion,AbstractSparseMatrix}}})
    return conj(dot(B, A))
end

function dot(x::AbstractSparseVector, D::Diagonal, y::AbstractVector)
    d = D.diag
    if length(x) != length(y) || length(y) != length(d)
        throw(
            DimensionMismatch("Vectors and matrix have different dimensions, x has a length $(length(x)), y has a length $(length(y)), D has side dimension $(length(d))")
        )
    end
    nzvals = nonzeros(x)
    nzinds = nonzeroinds(x)
    s = zero(typeof(dot(first(x), first(D), first(y))))
    @inbounds for nzidx in eachindex(nzvals)
        s += dot(nzvals[nzidx], d[nzinds[nzidx]], y[nzinds[nzidx]])
    end
    return s
end

dot(x::AbstractVector, D::Diagonal, y::AbstractSparseVector) = adjoint(dot(y, D', x))

function dot(x::AbstractSparseVector, D::Diagonal, y::AbstractSparseVector)
    d = D.diag
    if length(y) != length(x) || length(y) != length(d)
        throw(
            DimensionMismatch("Vectors and matrix have different dimensions, x has a length $(length(x)), y has a length $(length(y)), Q has side dimension $(length(d))")
        )
    end
    xnzind = nonzeroinds(x)
    ynzind = nonzeroinds(y)
    xnzval = nonzeros(x)
    ynzval = nonzeros(y)
    s = zero(typeof(dot(first(x), first(D), first(y))))
    if isempty(xnzind) || isempty(ynzind)
        return s
    end

    x_idx = 1
    y_idx = 1
    x_idx_last = length(xnzind)
    y_idx_last = length(ynzind)

    # go through the nonzero indices of a and b simultaneously
    @inbounds while x_idx <= x_idx_last && y_idx <= y_idx_last
        ix = xnzind[x_idx]
        iy = ynzind[y_idx]
        if ix == iy
            s += dot(xnzval[x_idx], d[ix], ynzval[y_idx])
            x_idx += 1
            y_idx += 1
        elseif ix < iy
            x_idx += 1
        else
            y_idx += 1
        end
    end
    return s
end

## triangular sparse handling
## triangular multiplication
function LinearAlgebra.generic_trimatmul!(C::StridedVecOrMat, uploc, isunitc, tfun::Function, A::SparseMatrixCSCUnion, B::AbstractVecOrMat)
    require_one_based_indexing(A, C)
    nrowC = size(C, 1)
    ncol = checksquare(A)
    if nrowC != ncol
        throw(DimensionMismatch("A has $(ncol) columns and B has $(nrowC) rows"))
    end
    nrowB, ncolB  = size(B, 1), size(B, 2)
    C !== B && copyto!(C, B)
    aa = getnzval(A)
    ja = getrowval(A)
    ia = getcolptr(A)
    joff = 0
    unit = isunitc == 'U'
    Z = zero(eltype(C))

    if uploc == 'U'
        if tfun === identity
            # forward multiplication for UpperTriangular SparseCSC matrices
            for k in axes(B,2)
                for j in axes(B,1)
                    i1 = ia[j]
                    i2 = ia[j + 1] - 1
                    done = unit

                    bj = B[joff + j]
                    for ii = i1:i2
                        jai = ja[ii]
                        aii = aa[ii]
                        if jai < j
                            C[joff + jai] += aii * bj
                        elseif jai == j
                            if !unit
                                C[joff + j] = aii * bj
                                done = true
                            end
                        else
                            break
                        end
                    end
                    if !done
                        C[joff + j] = Z
                    end
                end
                joff += nrowB
            end
        else # tfun in (adjoint, transpose)
            # backward multiplication with adjoint and transpose of LowerTriangular CSC matrices
            for k in axes(B,2)
                for j in reverse(axes(B,1))
                    i1 = ia[j]
                    i2 = ia[j + 1] - 1
                    akku = Z
                    j0 = !unit ? j : j - 1

                    # loop through column j of A - only structural non-zeros
                    for ii = i1:i2
                        jai = ja[ii]
                        if jai <= j0
                            akku += tfun(aa[ii]) * B[joff + jai]
                        else
                            break
                        end
                    end
                    if unit
                        akku += oneunit(eltype(A)) * B[joff + j]
                    end
                    C[joff + j] = akku
                end
                joff += nrowB
            end
        end
    else # uploc == 'L'
        if tfun === identity
            # backward multiplication for LowerTriangular SparseCSC matrices
            for k in axes(B,2)
                for j in reverse(axes(B,1))
                    i1 = ia[j]
                    i2 = ia[j + 1] - 1
                    done = unit

                    bj = B[joff + j]
                    for ii = i2:-1:i1
                        jai = ja[ii]
                        aii = aa[ii]
                        if jai > j
                            C[joff + jai] += aii * bj
                        elseif jai == j
                            if !unit
                                C[joff + j] = aii * bj
                                done = true
                            end
                        else
                            break
                        end
                    end
                    if !done
                        C[joff + j] = Z
                    end
                end
                joff += nrowB
            end
        else # tfun in (adjoint, transpose)
            # forward multiplication for adjoint and transpose of LowerTriangular CSC matrices
            for k in axes(B,2)
                for j in axes(B,1)
                    i1 = ia[j]
                    i2 = ia[j + 1] - 1
                    akku = Z
                    j0 = !unit ? j : j + 1

                    # loop through column j of A - only structural non-zeros
                    for ii = i2:-1:i1
                        jai = ja[ii]
                        if jai >= j0
                            akku += tfun(aa[ii]) * B[joff + jai]
                        else
                            break
                        end
                    end
                    if unit
                        akku += oneunit(eltype(A)) * B[joff + j]
                    end
                    C[joff + j] = akku
                end
                joff += nrowB
            end
        end
    end
    return C
end
function LinearAlgebra.generic_trimatmul!(C::StridedVecOrMat, uploc, isunitc, ::Function, xA::AdjOrTrans{<:Any,<:SparseMatrixCSCUnion}, B::AbstractVecOrMat)
    A = parent(xA)
    nrowC = size(C, 1)
    ncol = checksquare(A)
    if nrowC != ncol
        throw(DimensionMismatch("A has $(ncol) columns and B has $(nrowC) rows"))
    end
    C !== B && copyto!(C, B)
    nrowB, ncolB  = size(B, 1), size(B, 2)
    aa = getnzval(A)
    ja = getrowval(A)
    ia = getcolptr(A)
    joff = 0
    unit = isunitc == 'U'
    Z = zero(eltype(C))

    if uploc == 'U'
        for k in axes(B,2)
            for j in axes(B,1)
                i1 = ia[j]
                i2 = ia[j + 1] - 1
                done = unit

                bj = B[joff + j]
                for ii = i1:i2
                    jai = ja[ii]
                    aii = conj(aa[ii])
                    if jai < j
                        C[joff + jai] += aii * bj
                    elseif jai == j
                        if !unit
                            C[joff + j] = aii * bj
                            done = true
                        end
                    else
                        break
                    end
                end
                if !done
                    C[joff + j] = Z
                end
            end
            joff += nrowB
        end
    else # uploc == 'L'
        for k in axes(B,2)
            for j in reverse(axes(B,1))
                i1 = ia[j]
                i2 = ia[j + 1] - 1
                done = unit

                bj = B[joff + j]
                for ii = i2:-1:i1
                    jai = ja[ii]
                    aii = conj(aa[ii])
                    if jai > j
                        C[joff + jai] += aii * bj
                    elseif jai == j
                        if !unit
                            C[joff + j] = aii * bj
                            done = true
                        end
                    else
                        break
                    end
                end
                if !done
                    C[joff + j] = Z
                end
            end
            joff += nrowB
        end
    end
    return C
end

## triangular solvers
_uconvert_copyto!(c, b, oA) = (c .= Ref(oA) .\ b)
_uconvert_copyto!(c::AbstractArray{T}, b::AbstractArray{T}, _) where {T} = copyto!(c, b)

function LinearAlgebra.generic_trimatdiv!(C::StridedVecOrMat, uploc, isunitc, tfun::Function, A::SparseMatrixCSCUnion, B::AbstractVecOrMat)
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
function LinearAlgebra.generic_trimatdiv!(C::StridedVecOrMat, uploc, isunitc, ::Function, xA::AdjOrTrans{<:Any,<:SparseMatrixCSCUnion}, B::AbstractVecOrMat)
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

function (\)(A::Union{UpperTriangular,LowerTriangular}, B::AbstractSparseMatrixCSC)
    require_one_based_indexing(B)
    TAB = promote_op(\, eltype(A), eltype(B))
    ldiv!(Matrix{TAB}(undef, size(B)), A, B)
end
function (\)(A::Union{UnitUpperTriangular,UnitLowerTriangular}, B::AbstractSparseMatrixCSC)
    require_one_based_indexing(B)
    TAB = LinearAlgebra._inner_type_promotion(\, eltype(A), eltype(B))
    ldiv!(Matrix{TAB}(undef, size(B)), A, B)
end
# (*)(L::DenseTriangular, B::AbstractSparseMatrixCSC) = lmul!(L, Array(B))

## end of triangular

# symmetric/Hermitian

function _mul!(nzrang::Function, diagop::Function, odiagop::Function, C::StridedVecOrMat{T}, A, B, α, β) where T
    n = size(A, 2)
    m = size(B, 2)
    n == size(B, 1) == size(C, 1) && m == size(C, 2) ||
        throw(DimensionMismatch("A has size $(size(A)), B has size $(size(B)), C has size $(size(C))"))
    rv = rowvals(A)
    nzv = nonzeros(A)
    let z = T(0), sumcol=z, αxj=z, aarc=z, α = α
        β != one(β) && LinearAlgebra._rmul_or_fill!(C, β)
        @inbounds for k in axes(B,2)
            for col in axes(B,1)
                αxj = B[col,k] * α
                sumcol = z
                for j = nzrang(A, col)
                    row = rv[j]
                    aarc = nzv[j]
                    if row == col
                        sumcol += diagop(aarc) * B[row,k]
                    else
                        C[row,k] += aarc * αxj
                        sumcol += odiagop(aarc) * B[row,k]
                    end
                end
                C[col,k] += α * sumcol
            end
        end
    end
    C
end

# row range up to (and including if excl=false) diagonal
function nzrangeup(A, i, excl=false)
    r = nzrange(A, i); r1 = r.start; r2 = r.stop
    rv = rowvals(A)
    @inbounds r2 < r1 || rv[r2] <= i - excl ? r : r1:(searchsortedlast(view(rv, r1:r2), i - excl) + r1-1)
end
# row range from diagonal (included if excl=false) to end
function nzrangelo(A, i, excl=false)
    r = nzrange(A, i); r1 = r.start; r2 = r.stop
    rv = rowvals(A)
    @inbounds r2 < r1 || rv[r1] >= i + excl ? r : (searchsortedfirst(view(rv, r1:r2), i + excl) + r1-1):r2
end

dot(x::AbstractVector, A::RealHermSymComplexHerm{<:Real,<:AbstractSparseMatrixCSC}, y::AbstractVector) =
    _dot(x, parent(A), y, A.uplo == 'U' ? nzrangeup : nzrangelo, A isa Symmetric ? identity : real, A isa Symmetric ? transpose : adjoint)
function _dot(x::AbstractVector, A::AbstractSparseMatrixCSC, y::AbstractVector, rangefun::Function, diagop::Function, odiagop::Function)
    require_one_based_indexing(x, y)
    m, n = size(A)
    (length(x) == m && n == length(y)) ||
        throw(DimensionMismatch("x has length $(length(x)), A has size ($m, $n), y has length $(length(y))"))
    if iszero(m) || iszero(n)
        return dot(zero(eltype(x)), zero(eltype(A)), zero(eltype(y)))
    end
    T = promote_type(eltype(x), eltype(A), eltype(y))
    r = zero(T)
    rvals = getrowval(A)
    nzvals = getnzval(A)
    @inbounds for col in axes(A,2)
        ycol = y[col]
        xcol = x[col]
        if _isnotzero(ycol) && _isnotzero(xcol)
            for k in rangefun(A, col)
                i = rvals[k]
                Aij = nzvals[k]
                if i != col
                    r += dot(x[i], Aij, ycol)
                    r += dot(xcol, odiagop(Aij), y[i])
                else
                    r += dot(x[i], diagop(Aij), ycol)
                end
            end
        end
    end
    return r
end
dot(x::SparseVector, A::RealHermSymComplexHerm{<:Real,<:AbstractSparseMatrixCSC}, y::SparseVector) =
    _dot(x, parent(A), y, A.uplo == 'U' ? nzrangeup : nzrangelo, A isa Symmetric ? identity : real)
function _dot(x::SparseVector, A::AbstractSparseMatrixCSC, y::SparseVector, rangefun::Function, diagop::Function)
    m, n = size(A)
    length(x) == m && n == length(y) ||
        throw(DimensionMismatch("x has length $(length(x)), A has size ($m, $n), y has length $(length(y))"))
    if iszero(m) || iszero(n)
        return dot(zero(eltype(x)), zero(eltype(A)), zero(eltype(y)))
    end
    r = zero(promote_type(eltype(x), eltype(A), eltype(y)))
    xnzind = nonzeroinds(x)
    xnzval = nonzeros(x)
    ynzind = nonzeroinds(y)
    ynzval = nonzeros(y)
    Arowval = getrowval(A)
    Anzval = getnzval(A)
    Acolptr = getcolptr(A)
    isempty(Arowval) && return r
    # plain triangle without diagonal
    for (yi, yv) in zip(ynzind, ynzval)
        A_ptr_lo = first(rangefun(A, yi, true))
        A_ptr_hi = last(rangefun(A, yi, true))
        if A_ptr_lo <= A_ptr_hi
            # dot is conjugated in the first argument, so double conjugate a's
            r += dot(_spdot((x, a) -> a'x, 1, length(xnzind), xnzind, xnzval,
                                            A_ptr_lo, A_ptr_hi, Arowval, Anzval), yv)
        end
    end
    # view triangle without diagonal
    for (xi, xv) in zip(xnzind, xnzval)
        A_ptr_lo = first(rangefun(A, xi, true))
        A_ptr_hi = last(rangefun(A, xi, true))
        if A_ptr_lo <= A_ptr_hi
            r += dot(xv, _spdot((a, y) -> a'y, A_ptr_lo, A_ptr_hi, Arowval, Anzval,
                                            1, length(ynzind), ynzind, ynzval))
        end
    end
    # diagonal
    for i in axes(A,1)
        r1 = Int(Acolptr[i])
        r2 = Int(Acolptr[i+1]-1)
        r1 > r2 && continue
        r1 += searchsortedfirst(view(Arowval, r1:r2), i) - 1
        ((r1 > r2) || (Arowval[r1] != i)) && continue
        r += dot(x[i], diagop(Anzval[r1]), y[i])
    end
    r
end
## end of symmetric/Hermitian

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
    for i=axes(b,1)
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
    for col = 1 : min(max(k+1,1), n+1)
        colptr[col] = 1
    end
    for col = max(k+1,1) : n
        for c1 in nzrange(S, col)
            rowvals(S)[c1] > col - k && break
            nnz += 1
        end
        colptr[col+1] = nnz+1
    end
    rowval = Vector{Ti}(undef, nnz)
    nzval = Vector{Tv}(undef, nnz)
    for col = max(k+1,1) : n
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
    for col = 1 : min(n, m+k)
        l1 = getcolptr(S)[col+1]-1
        for c1 = 0 : (l1 - getcolptr(S)[col])
            rowvals(S)[l1 - c1] < col - k && break
            nnz += 1
        end
        colptr[col+1] = nnz+1
    end
    for col = max(min(n, m+k)+2,1) : n+1
        colptr[col] = nnz+1
    end
    rowval = Vector{Ti}(undef, nnz)
    nzval = Vector{Tv}(undef, nnz)
    for col = 1 : min(n, m+k)
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
    colptr[1] = 1
    for col = 1 : n
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

    ptrS = 1
    colptr[1] = 1

    n == 0 && return SparseMatrixCSC(m, n, colptr, rowval, nzval)

    startA = colptr_a[1]
    stopA = colptr_a[2]

    rA = startA : stopA - 1
    rowvalA = rowval_a[rA]
    nzvalA = nzval_a[rA]
    lA = stopA - startA

    for col = 1:n-1
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
            for j in axes(A,2)
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
            for i in axes(nonzeros(A),1)
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
    X[1:n,1] .= 1
    for j = 2:t
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
                    if !repeated
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

## kron
const _SparseArraysCSC = Union{AbstractCompressedVector, AbstractSparseMatrixCSC}
const _SparseKronArrays = Union{_SparseArraysCSC, AdjOrTrans{<:Any,<:_SparseArraysCSC}}

const _Symmetric_SparseKronArrays = Symmetric{<:Any,<:_SparseKronArrays}
const _Hermitian_SparseKronArrays = Hermitian{<:Any,<:_SparseKronArrays}
const _Triangular_SparseKronArrays = UpperOrLowerTriangular{<:Any,<:_SparseKronArrays}
const _Annotated_SparseKronArrays = Union{_Triangular_SparseKronArrays, _Symmetric_SparseKronArrays, _Hermitian_SparseKronArrays}
const _SparseKronGroup = Union{_SparseKronArrays, _Annotated_SparseKronArrays}

const _SpecialArrays = Union{Diagonal, Bidiagonal, Tridiagonal, SymTridiagonal}
const _Symmetric_DenseArrays{T,A<:Matrix} = Symmetric{T,A}
const _Hermitian_DenseArrays{T,A<:Matrix} = Hermitian{T,A}
const _Triangular_DenseArrays{T,A<:Matrix} = UpperOrLowerTriangular{<:Any,A} # AbstractTriangular{T,A}
const _Annotated_DenseArrays = Union{_SpecialArrays, _Triangular_DenseArrays, _Symmetric_DenseArrays, _Hermitian_DenseArrays}
const _DenseKronGroup = Union{Number, Vector, Matrix, AdjOrTrans{<:Any,<:VecOrMat}, _Annotated_DenseArrays}

@inline function kron!(C::SparseMatrixCSC, A::AbstractSparseMatrixCSC, B::AbstractSparseMatrixCSC)
    mA, nA = size(A); mB, nB = size(B)
    mC, nC = mA*mB, nA*nB
    @boundscheck size(C) == (mC, nC) || throw(DimensionMismatch("target matrix needs to have size ($mC, $nC)," *
        " but has size $(size(C))"))
    rowvalC = rowvals(C)
    nzvalC = nonzeros(C)
    colptrC = getcolptr(C)

    nnzC = nnz(A)*nnz(B)
    resize!(nzvalC, nnzC)
    resize!(rowvalC, nnzC)

    col = 1
    @inbounds for j in axes(A,2)
        startA = getcolptr(A)[j]
        stopA = getcolptr(A)[j+1] - 1
        lA = stopA - startA + 1
        for i in axes(B,2)
            startB = getcolptr(B)[i]
            stopB = getcolptr(B)[i+1] - 1
            lB = stopB - startB + 1
            ptr_range = (1:lB) .+ (colptrC[col]-1)
            colptrC[col+1] = colptrC[col] + lA*lB
            col += 1
            for ptrA = startA : stopA
                ptrB = startB
                for ptr = ptr_range
                    rowvalC[ptr] = (rowvals(A)[ptrA]-1)*mB + rowvals(B)[ptrB]
                    nzvalC[ptr] = nonzeros(A)[ptrA] * nonzeros(B)[ptrB]
                    ptrB += 1
                end
                ptr_range = ptr_range .+ lB
            end
        end
    end
    return C
end
@inline function kron!(z::SparseVector, x::SparseVector, y::SparseVector)
    @boundscheck length(z) == length(x)*length(y) || throw(DimensionMismatch("length of " *
        "target vector needs to be $(length(x)*length(y)), but has length $(length(z))"))
    nnzx, nnzy = nnz(x), nnz(y)
    nzind = nonzeroinds(z)
    nzval = nonzeros(z)

    nnzz = nnzx*nnzy
    resize!(nzind, nnzz)
    resize!(nzval, nnzz)

    @inbounds for i = 1:nnzx, j = 1:nnzy
        this_ind = (i-1)*nnzy+j
        nzind[this_ind] = (nonzeroinds(x)[i]-1)*length(y) + nonzeroinds(y)[j]
        nzval[this_ind] = nonzeros(x)[i] * nonzeros(y)[j]
    end
    return z
end
kron!(C::SparseMatrixCSC, A::_SparseKronGroup, B::_DenseKronGroup) =
    kron!(C, convert(SparseMatrixCSC, A), convert(SparseMatrixCSC, B))
kron!(C::SparseMatrixCSC, A::_DenseKronGroup, B::_SparseKronGroup) =
    kron!(C, convert(SparseMatrixCSC, A), convert(SparseMatrixCSC, B))
kron!(C::SparseMatrixCSC, A::_SparseKronGroup, B::_SparseKronGroup) =
    kron!(C, convert(SparseMatrixCSC, A), convert(SparseMatrixCSC, B))
kron!(C::SparseMatrixCSC, A::_SparseVectorUnion, B::_AdjOrTransSparseVectorUnion) =
    broadcast!(*, C, A, B)
# disambiguation
kron!(C::SparseMatrixCSC, A::_SparseKronGroup, B::Diagonal) =
    kron!(C, convert(SparseMatrixCSC, A), convert(SparseMatrixCSC, B))
kron!(C::SparseMatrixCSC, A::Diagonal, B::_SparseKronGroup) =
    kron!(C, convert(SparseMatrixCSC, A), convert(SparseMatrixCSC, B))
kron!(C::SparseMatrixCSC, A::AbstractCompressedVector, B::AdjOrTrans{<:Any,<:AbstractCompressedVector}) =
    broadcast!(*, C, A, B)
kron!(c::SparseMatrixCSC, a::Number, b::_SparseKronGroup) = mul!(c, a, b)
kron!(c::SparseMatrixCSC, a::_SparseKronGroup, b::Number) = mul!(c, a, b)

function kron(A::AbstractSparseMatrixCSC, B::AbstractSparseMatrixCSC)
    mA, nA = size(A)
    mB, nB = size(B)
    mC, nC = mA*mB, nA*nB
    Tv = typeof(oneunit(eltype(A))*oneunit(eltype(B)))
    Ti = promote_type(indtype(A), indtype(B))
    C = spzeros(Tv, Ti, mC, nC)
    sizehint!(C, nnz(A)*nnz(B))
    return @inbounds kron!(C, A, B)
end
function kron(x::AbstractCompressedVector, y::AbstractCompressedVector)
    nnzx, nnzy = nnz(x), nnz(y)
    nnzz = nnzx*nnzy # number of nonzeros in new vector
    nzind = Vector{promote_type(indtype(x), indtype(y))}(undef, nnzz) # the indices of nonzeros
    nzval = Vector{typeof(oneunit(eltype(x))*oneunit(eltype(y)))}(undef, nnzz) # the values of nonzeros
    z = SparseVector(length(x)*length(y), nzind, nzval)
    return @inbounds kron!(z, x, y)
end
# extend to annotated sparse arrays, but leave out the (dense ⊗ dense)-case
kron(A::_SparseKronGroup, B::_SparseKronGroup) =
    kron(convert(SparseMatrixCSC, A), convert(SparseMatrixCSC, B))
kron(A::_SparseKronGroup, B::_DenseKronGroup) = kron(A, sparse(B))
kron(A::_DenseKronGroup, B::_SparseKronGroup) = kron(sparse(A), B)
kron(A::_SparseVectorUnion, B::_AdjOrTransSparseVectorUnion) = A .* B
# disambiguation
kron(A::AbstractCompressedVector, B::AdjOrTrans{<:Any,<:AbstractCompressedVector}) = A .* B
kron(a::Number, b::_SparseKronGroup) = a * b
kron(a::_SparseKronGroup, b::Number) = a * b

## det, inv, cond

inv(A::AbstractSparseMatrixCSC) = error("The inverse of a sparse matrix can often be dense and can cause the computer to run out of memory. If you are sure you have enough memory, please either convert your matrix to a dense matrix, e.g. by calling `Matrix` or if `A` can be factorized, use `\\` on the dense identity matrix, e.g. `A \\ Matrix{eltype(A)}(I, size(A)...)` restrictions of `\\` on sparse lhs applies. Alternatively, `A\\b` is generally preferable to `inv(A)*b`")

# TODO

## scale methods

# Copy colptr and rowval from one sparse matrix to another
function copyinds!(C::AbstractSparseMatrixCSC, A::AbstractSparseMatrixCSC)
    if getcolptr(C) !== getcolptr(A)
        resize!(getcolptr(C), length(getcolptr(A)))
        copyto!(getcolptr(C), getcolptr(A))
    end
    if rowvals(C) !== rowvals(A)
        resize!(rowvals(C), length(rowvals(A)))
        copyto!(rowvals(C), rowvals(A))
    end
end

# multiply by diagonal matrix as vector
function mul!(C::AbstractSparseMatrixCSC, A::AbstractSparseMatrixCSC, D::Diagonal)
    m, n = size(A)
    b    = D.diag
    lb = length(b)
    n == lb || throw(DimensionMismatch("A has size ($m, $n) but D has size ($lb, $lb)"))
    size(A)==size(C) || throw(DimensionMismatch("A has size ($m, $n), D has size ($lb, $lb), C has size $(size(C))"))
    copyinds!(C, A)
    Cnzval = nonzeros(C)
    Anzval = nonzeros(A)
    resize!(Cnzval, length(Anzval))
    for col in axes(A,2), p in nzrange(A, col)
        @inbounds Cnzval[p] = Anzval[p] * b[col]
    end
    C
end

function mul!(C::AbstractSparseMatrixCSC, D::Diagonal, A::AbstractSparseMatrixCSC)
    m, n = size(A)
    b    = D.diag
    lb = length(b)
    m == lb || throw(DimensionMismatch("D has size ($lb, $lb) but A has size ($m, $n)"))
    size(A)==size(C) || throw(DimensionMismatch("A has size ($m, $n), D has size ($lb, $lb), C has size $(size(C))"))
    copyinds!(C, A)
    Cnzval = nonzeros(C)
    Anzval = nonzeros(A)
    Arowval = rowvals(A)
    resize!(Cnzval, length(Anzval))
    for col in axes(A,2), p in nzrange(A, col)
        @inbounds Cnzval[p] = b[Arowval[p]] * Anzval[p]
    end
    C
end

function mul!(C::AbstractSparseMatrixCSC, A::AbstractSparseMatrixCSC, b::Number)
    size(A)==size(C) || throw(DimensionMismatch("A has size $(size(A)) but C has size $(size(C))"))
    copyinds!(C, A)
    resize!(nonzeros(C), length(nonzeros(A)))
    mul!(nonzeros(C), nonzeros(A), b)
    C
end

function mul!(C::AbstractSparseMatrixCSC, b::Number, A::AbstractSparseMatrixCSC)
    size(A)==size(C) || throw(DimensionMismatch("A has size $(size(A)) but C has size $(size(C))"))
    copyinds!(C, A)
    resize!(nonzeros(C), length(nonzeros(A)))
    mul!(nonzeros(C), b, nonzeros(A))
    C
end

function rmul!(A::AbstractSparseMatrixCSC, b::Number)
    rmul!(nonzeros(A), b)
    return A
end

function lmul!(b::Number, A::AbstractSparseMatrixCSC)
    lmul!(b, nonzeros(A))
    return A
end

function rmul!(A::AbstractSparseMatrixCSC, D::Diagonal)
    m, n = size(A)
    szD = size(D, 1)
    (n == szD) || throw(DimensionMismatch("A has size ($m, $n) but D has size ($szD, $szD)"))
    Anzval = nonzeros(A)
    @inbounds for col in axes(A,2), p in nzrange(A, col)
         Anzval[p] = Anzval[p] * D.diag[col]
    end
    return A
end

function lmul!(D::Diagonal, A::AbstractSparseMatrixCSC)
    m, n = size(A)
    ds2 = size(D, 2)
    (m == ds2) || throw(DimensionMismatch("D has size ($ds2, $ds2) but A has size ($m, $n)"))
    Anzval = nonzeros(A)
    Arowval = rowvals(A)
    @inbounds for col in axes(A,2), p in nzrange(A, col)
        Anzval[p] = D.diag[Arowval[p]] * Anzval[p]
    end
    return A
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
        return \(lu(A), B)
    else
        return \(qr(A), B)
    end
end
for (xformtype, xformop) in ((:Adjoint, :adjoint), (:Transpose, :transpose))
    @eval begin
        function \(xformA::($xformtype){<:Any,<:AbstractSparseMatrixCSC}, B::AbstractVecOrMat)
            A = xformA.parent
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
            else
                return \($xformop(qr(A)), B)
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
    else
        return qr(A)
    end
end

# function factorize(A::Symmetric{Float64,AbstractSparseMatrixCSC{Float64,Ti}}) where Ti
#     F = cholesky(A)
#     if LinearAlgebra.issuccess(F)
#         return F
#     else
#         ldlt!(F, A)
#         return F
#     end
# end
function factorize(A::LinearAlgebra.RealHermSymComplexHerm{Float64,<:AbstractSparseMatrixCSC})
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
