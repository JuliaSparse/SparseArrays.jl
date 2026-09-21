# This file is a part of Julia. License is MIT: https://julialang.org/license

# Products: `*`, `mul!`, `lmul!`, `rmul!`, `dot` and `kron`, and the kernels behind them.

_fix_size(M, nrow, ncol) = M

# An immutable fixed size wrapper for matrices to work around
# the performance issue caused by https://github.com/JuliaLang/julia/issues/60409
# This is more-of-less a stripped down version of FixedSizeArrays
# which we can't easily use without pulling that into the standard library.
struct _FixedSizeMatrix{Trans,R}
    ref::R
    nrow::Int
    ncol::Int
    function _FixedSizeMatrix{Trans}(ref::R, nrow, ncol) where {Trans,R}
        new{Trans,R}(ref, nrow, ncol)
    end
end
@inline Base.getindex(A::_FixedSizeMatrix{'N'}, i, j) =
    @inbounds Core.memoryrefnew(A.ref, A.nrow * (j - 1) + i, false)[]
@inline Base.setindex!(A::_FixedSizeMatrix{'N'}, v, i, j) =
    @inbounds Core.memoryrefnew(A.ref, A.nrow * (j - 1) + i, false)[] = v

@inline Base.getindex(A::_FixedSizeMatrix{'T'}, i, j) =
    @inbounds transpose(Core.memoryrefnew(A.ref, A.ncol * (i - 1) + j, false)[])
@inline Base.setindex!(A::_FixedSizeMatrix{'T'}, v, i, j) =
    @inbounds Core.memoryrefnew(A.ref, A.ncol * (i - 1) + j, false)[] = transpose(v)

@inline Base.getindex(A::_FixedSizeMatrix{'C'}, i, j) =
    @inbounds adjoint(Core.memoryrefnew(A.ref, A.ncol * (i - 1) + j, false)[])
@inline Base.setindex!(A::_FixedSizeMatrix{'C'}, v, i, j) =
    @inbounds Core.memoryrefnew(A.ref, A.ncol * (i - 1) + j, false)[] = adjoint(v)

@inline _fix_size(A::Matrix, nrow, ncol) = _FixedSizeMatrix{'N'}(A.ref, nrow, ncol)
@inline _fix_size(A::Transpose{<:Any,<:Matrix}, nrow, ncol) =
    _FixedSizeMatrix{'T'}(parent(A).ref, nrow, ncol)
@inline _fix_size(A::Adjoint{<:Any,<:Matrix}, nrow, ncol) =
    _FixedSizeMatrix{'C'}(parent(A).ref, nrow, ncol)

const tilebufsize = 10800  # Approximately 32k/3


matop_dest(::typeof(*), A::QuasiStridedMatrix, b::AbstractSparseVector) =
    Vector{promote_op(matprod, eltype(A), eltype(b))}(undef, size(A, 1))
matop_dest(::typeof(*), A, B::Union{QuasiSparseMatrix,SparseAdjOrTransTriangular}) =
    similar(A, promote_op(matprod, eltype(A), eltype(B)), (size(A, 1), size(B, 2)))
# sparse products with banded matrices should return sparse arrays
matop_dest(::typeof(*), A::BiTriSym, B::Union{QuasiSparseMatrix,SparseAdjOrTransTriangular}) =
    similar(B, promote_op(matprod, eltype(A), eltype(B)), size(B))
# needed for disambiguation with LinearAlgebra
matop_dest(::typeof(*), A::Diagonal, B::Union{QuasiSparseMatrix,SparseAdjOrTransTriangular}) =
    similar(B, promote_op(matprod, eltype(A), eltype(B)), size(B))
# a `Diagonal` product keeps the structure of the sparse operand, so a fixed operand gets
# a fixed destination with that structure up front, which `mul!` then only has to fill
# (an empty fixed destination could not take the indices); the adjoint/transpose of a
# sparse matrix gets an empty, writable destination, since its structure is not that of
# the parent
matop_dest(::typeof(*), A::Diagonal, B::AbstractSparseMatrixCSC) =
    _is_fixed(B) ? similar(B, promote_op(matprod, eltype(A), eltype(B))) :
                   similar(B, promote_op(matprod, eltype(A), eltype(B)), size(B))
matop_dest(::typeof(*), A::Diagonal, B::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}) =
    _adjtrans_dest(B, promote_op(matprod, eltype(A), eltype(B)))
matop_dest(::typeof(*), A::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}, B::Diagonal) =
    _adjtrans_dest(A, promote_op(matprod, eltype(A), eltype(B)))
function _adjtrans_dest(A::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}, ::Type{T}) where T
    P = parent(A)
    return sizehint!(spzeros(T, indtype(P), size(A)...), nnz(P))
end
matop_dest(::typeof(*), A::QuasiSparseMatrix, B::BiTriSym) =
    similar(A, promote_op(matprod, eltype(A), eltype(B)), (size(A, 1), size(B, 2)))


# The dense factor arrives with its adjoint/transpose/symmetric/Hermitian wrapper stripped,
# so it is any matrix with fast scalar `getindex`: strided or not, or a structured dense
# type without a product of its own, such as `UpperHessenberg`.
mul!(C::StridedMatrix, tA, tB, A::SparseMatrixCSCOrColumnSubset, B::AbstractMatrix, alpha::Number, beta::Number) =
    spdensemul!(C, tA, tB, A, B, alpha, beta)
mul!(C::StridedMatrix, tA, tB, A::AbstractMatrix, B::SparseMatrixCSCOrColumnSubset, alpha::Number, beta::Number) =
    densespmul!(C, tA, tB, A, B, alpha, beta)
Base.@constprop :aggressive function mul!(C::StridedMatrix, tA, tB, A::SparseMatrixCSCOrColumnSubset, B::SparseMatrixCSCOrColumnSubset, alpha::Number, beta::Number)
    # `A' * B' == (B * A)'` walks columns instead of looking up rows, for commutative eltypes
    if tA == tB && tA in ('T', 'C') && eltype(A) <: Union{Real,Complex} && eltype(B) <: Union{Real,Complex}
        tfun = tA == 'T' ? transpose : adjoint
        _spmatmul!(tfun(C), B, A, tfun(alpha), tfun(beta))
        return C
    end
    # with both symmetric/Hermitian, row lookups in two half-stored factors cost more than the generic sweep
    if _uppercase(tA) in ('S', 'H') && _uppercase(tB) in ('S', 'H')
        return LinearAlgebra._generic_matmatmul!(C, wrap(A, tA), wrap(B, tB), alpha, beta)
    end
    spdensemul!(C, tA, tB, A, B, alpha, beta)
end
LinearAlgebra._mul!(C::StridedMatrix, A::QuasiSparseMatrix, B::AbstractTriangular, alpha::Number, beta::Number) =
    spdensemul!(C, LinearAlgebra.wrapper_char(A), LinearAlgebra.wrapper_char(B), LinearAlgebra._unwrap(A), B, alpha, beta)
mul!(C::StridedVecOrMat, tA, A::SparseMatrixCSCOrColumnSubset, B::AbstractVector, alpha::Number, beta::Number) =
    spdensemul!(C, tA, 'N', A, B, alpha, beta)
# LinearAlgebra materializes the second of two symmetric/Hermitian factors, elementwise
# when it is sparse; the kernels take both wrappers as they are
LinearAlgebra.mul(A::HermOrSym{<:Any,<:DenseMatrixUnion}, B::SparseMatrixCSCSymmHerm) = LinearAlgebra._mul(A, B)
LinearAlgebra.mul(A::SparseMatrixCSCSymmHerm, B::HermOrSym{<:Any,<:DenseMatrixUnion}) = LinearAlgebra._mul(A, B)

Base.@constprop :aggressive function spdensemul!(C, tA, tB, A, B, alpha, beta)
    tA_uc, tB_uc = _uppercase(tA), _uppercase(tB)
    if tA_uc == 'N'
        _spmatmul!(C, A, wrap(B, tB), alpha, beta)
    elseif tA_uc == 'T'
        _At_or_Ac_mul_B!(transpose, C, A, wrap(B, tB), alpha, beta)
    elseif tA_uc == 'C'
        _At_or_Ac_mul_B!(adjoint, C, A, wrap(B, tB), alpha, beta)
    elseif tA_uc in ('S', 'H')
        rangefun = _isuppercase(tA) ? nzrangeup : nzrangelo
        diagop = tA_uc == 'S' ? identity : real
        odiagop = tA_uc == 'S' ? transpose : adjoint
        T = eltype(C)
        _symherm_mul!(rangefun, diagop, odiagop, C, A, wrap(B, tB), T(alpha), T(beta))
    else
        LinearAlgebra._generic_matmatmul!(C, wrap(A, tA), wrap(B, tB), alpha, beta)
    end
    return C
end

# Slow non-inlined functions for throwing the error without messing up the caller
@noinline function _matmul_size_error(mC, nC, mA, nA, mB, nB, At, Bt)
    if At == 'N'
        Anames = "first", "second"
    else
        Anames = "second", "first"
    end
    if Bt == 'N'
        Bnames = "first", "second"
    else
        Bnames = "second", "first"
    end
    nA == mB ||
        throw(DimensionMismatch("$(Anames[2]) dimension of A, $nA, does not match the $(Bnames[1]) dimension of B, $mB"))
    mA == mC ||
        throw(DimensionMismatch("$(Anames[1]) dimension of A, $mA, does not match the first dimension of C, $mC"))
    nB == nC ||
        throw(DimensionMismatch("$(Bnames[2]) dimension of B, $nB, does not match the second dimension of C, $nC"))
    # unreachable
    throw(DimensionMismatch("Unknown dimension mismatch"))
end

@inline function _matmul_size(C, A, B, ::Val{At}, ::Val{Bt}) where {At,Bt}
    mC = size(C, 1)
    nC = size(C, 2)
    mA = size(A, 1)
    nA = size(A, 2)
    mB = size(B, 1)
    nB = size(B, 2)

    _mA, _nA = At == 'N' ? (mA, nA) : (nA, mA)
    _mB, _nB = Bt == 'N' ? (mB, nB) : (nB, mB)

    if (_nA != _mB) | (_mA != mC) | (_nB != nC)
        _matmul_size_error(mC, nC, _mA, _nA, _mB, _nB, At, Bt)
    end
    return mC, nC, mA, nA, mB, nB
end

@inline _matmul_size_AB(C, A, B) = _matmul_size(C, A, B, Val('N'), Val('N'))
@inline _matmul_size_AtB(C, A, B) = _matmul_size(C, A, B, Val('T'), Val('N'))
@inline _matmul_size_ABt(C, A, B) = _matmul_size(C, A, B, Val('N'), Val('T'))

function _spmatmul!(C, A, B, α, β)
    Cax2 = axes(C, 2)
    Aax2 = axes(A, 2)
    mC, nC, mA, nA, mB, nB = _matmul_size_AB(C, A, B)
    nzv = nonzeros(A)
    rv = rowvals(A)
    isone(β) || LinearAlgebra._rmul_or_fill!(C, β)
    if α isa Bool && !α
        return
    end
    B = _fix_size(B, mB, nB)
    C = _fix_size(C, mC, nC)
    for k in Cax2
        @inbounds for col in Aax2
            αxj = α isa Bool ? B[col,k] : B[col,k] * α
            for j in nzrange(A, col)
                rvj = rv[j]
                C[rvj, k] = muladd(nzv[j], αxj, C[rvj, k])
            end
        end
    end
end

function _At_or_Ac_mul_B!(tfun::Function, C, A, B, α, β)
    Cax2 = axes(C, 2)
    Aax2 = axes(A, 2)
    mC, nC, mA, nA, mB, nB = _matmul_size_AtB(C, A, B)
    nzv = nonzeros(A)
    rv = rowvals(A)
    isone(β) || LinearAlgebra._rmul_or_fill!(C, β)
    if α isa Bool && !α
        return
    end
    C0 = zero(eltype(C)) # Pre-allocate for BigFloat/BigInt etc
    B = _fix_size(B, mB, nB)
    C = _fix_size(C, mC, nC)
    for k in Cax2
        @inbounds for col in Aax2
            tmp = C0
            for j in nzrange(A, col)
                tmp = muladd(tfun(nzv[j]), B[rv[j], k], tmp)
            end
            C[col, k] = α isa Bool ? tmp + C[col, k] : muladd(tmp, α, C[col, k])
        end
    end
end

Base.@constprop :aggressive function densespmul!(C, tA, tB, A, B, alpha, beta)
    X = wrap(A, tA)
    tB_uc = _uppercase(tB)
    if tB_uc == 'N'
        _spmul!(C, X, B, alpha, beta)
    elseif tB_uc == 'T'
        _A_mul_Bt_or_Bc!(transpose, C, X, B, alpha, beta)
    elseif tB_uc == 'C'
        _A_mul_Bt_or_Bc!(adjoint, C, X, B, alpha, beta)
    else # tB_uc in ('S', 'H')
        rangefun = _isuppercase(tB) ? nzrangeup : nzrangelo
        diagop = tB_uc == 'S' ? identity : real
        odiagop = tB_uc == 'S' ? transpose : adjoint
        _A_mul_symherm!(rangefun, diagop, odiagop, C, X, B, alpha, beta)
    end
    return C
end

function _spmul!(C::StridedMatrix, X::AbstractMatrix, A::SparseMatrixCSCOrColumnSubset, α::Number, β::Number)
    Aax2 = axes(A, 2)
    Xax1 = axes(X, 1)
    mC, nC, mX, nX, mA, nA = _matmul_size_AB(C, X, A)
    rv = rowvals(A)
    nzv = nonzeros(A)
    isone(β) || LinearAlgebra._rmul_or_fill!(C, β)
    if α isa Bool && !α
        return
    end
    C = _fix_size(C, mC, nC)
    X = _fix_size(X, mX, nX)
    @inbounds for col in Aax2, k in nzrange(A, col)
        Aiα = α isa Bool ? nzv[k] : nzv[k] * α
        rvk = rv[k]
        @simd for multivec_row in Xax1
            C[multivec_row, col] = muladd(X[multivec_row, rvk], Aiα,
                                          C[multivec_row, col])
        end
    end
end
function _spmul!(C::StridedMatrix, X::AdjOrTrans, A::SparseMatrixCSCOrColumnSubset, α::Number, β::Number)
    Xax1 = axes(X, 1)
    Cax2 = axes(C, 2)
    mC, nC, mX, nX, mA, nA = _matmul_size_AB(C, X, A)
    rv = rowvals(A)
    nzv = nonzeros(A)
    isone(β) || LinearAlgebra._rmul_or_fill!(C, β)
    if α isa Bool && !α
        return
    end
    C = _fix_size(C, mC, nC)
    X = _fix_size(X, mX, nX)
    @inbounds for multivec_row in Xax1, col in Cax2
        nzrng = nzrange(A, col)
        if isempty(nzrng)
            continue
        end
        tmp = C[multivec_row, col]
        for k in nzrng
            tmp = muladd(X[multivec_row, rv[k]],
                         (α isa Bool ? nzv[k] : nzv[k] * α), tmp)
        end
        C[multivec_row, col] = tmp
    end
end

function _A_mul_Bt_or_Bc!(tfun::Function, C::StridedMatrix, A::AbstractMatrix, B::SparseMatrixCSCOrColumnSubset, α::Number, β::Number)
    Bax2 = axes(B, 2)
    Aax1 = axes(A, 1)
    mC, nC, mA, nA, mB, nB = _matmul_size_ABt(C, A, B)
    rv = rowvals(B)
    nzv = nonzeros(B)
    isone(β) || LinearAlgebra._rmul_or_fill!(C, β)
    if α isa Bool && !α
        return
    end
    C = _fix_size(C, mC, nC)
    A = _fix_size(A, mA, nA)
    @inbounds for col in Bax2, k in nzrange(B, col)
        Biα = α isa Bool ? tfun(nzv[k]) : tfun(nzv[k]) * α
        rvk = rv[k]
        @simd for multivec_col in Aax1
            C[multivec_col, rvk] = muladd(A[multivec_col, col], Biα, C[multivec_col, rvk])
        end
    end
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

# spmatmul handles compressed vectors and whole-column/whole-vector views; other
# AbstractSparseVectors take the generic product. Plain CSC times a compressed vector is
# defined with the sparse vector products below.
*(A::SparseTriangular, B::SparseVectorOrView) = spmatmulv(A, B)
*(A::SparseMatrixCSCView, B::SparseVectorOrView) = spmatmulv(A, B)
*(A::AbstractSparseMatrixCSC, B::Union{SparseColumnView,SparseVectorView}) = spmatmulv(A, B)
*(A::SparseMatrixCSCOrView, B::SparseMatrixCSCOrView) = spmatmul(A,B)
*(A::SparseTriangular, B::SparseMatrixCSCOrView) = spmatmul(A,B)
*(A::SparseMatrixCSCOrView, B::SparseTriangular) = spmatmul(A,B)
*(A::SparseTriangular, B::SparseTriangular) = spmatmul1(A,B)
*(A::SparseOrTri, B::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}) = spmatmul(A, copy(B))
*(A::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}, B::SparseOrTri) = spmatmul(copy(A), B)
*(A::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}, B::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}) = spmatmul(copy(A), copy(B))
# a symmetric/Hermitian sparse factor is materialized, which is O(nnz) like the copies above
*(A::SparseMatrixCSCSymmHerm, B::Union{SparseOrTri,AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}}) = sparse(A) * B
*(A::Union{SparseOrTri,AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}}, B::SparseMatrixCSCSymmHerm) = A * sparse(B)
*(A::SparseMatrixCSCSymmHerm, B::SparseMatrixCSCSymmHerm) = sparse(A) * sparse(B)
*(A::SparseMatrixCSCSymmHerm, x::SparseVectorOrView) = sparse(A) * x

(*)(Da::Diagonal, A::Union{SparseMatrixCSCOrView, AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}}, Db::Diagonal) = Da * (A * Db)
function (*)(Da::Diagonal, A::SparseMatrixCSC, Db::Diagonal)
    (size(Da, 2) == size(A,1) && size(A,2) == size(Db,1)) ||
        throw(DimensionMismatch("incompatible sizes"))
    T = promote_op(matprod, eltype(Da), promote_op(matprod, eltype(A), eltype(Db)))
    dest = similar(A, T)
    vals_dest = nonzeros(dest)
    rows = rowvals(A)
    vals = nonzeros(A)
    da, db = map(parent, (Da, Db))
    @inbounds for col in axes(A,2)
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
# Unit triangular wrappers keep their diagonal implicitly, so they are materialized first.
# The added diagonal may not fit the parent's index type, in which case a wider one is used
# for the temporary; the result keeps the index type of the operands.
_explicitdiag(A) = A
_explicitdiag(A::UnitUpperTriangular{<:Any,<:SparseMatrixCSCOrView}) = sparse(UnitUpperTriangular(_fitsdiag(parent(A))))
_explicitdiag(A::UnitLowerTriangular{<:Any,<:SparseMatrixCSCOrView}) = sparse(UnitLowerTriangular(_fitsdiag(parent(A))))
# colptr[end] is one past the stored count
_fitsdiag(S) = nnz(S) + size(S, 2) < typemax(indtype(S)) ? S : SparseMatrixCSC{eltype(S),Int}(S)
function spmatmul(A::SparseOrTri, B::Union{SparseOrTri,AbstractCompressedVector,SubArray{<:Any,<:Any,<:AbstractSparseArray}})
    Tv = promote_op(matprod, eltype(A), eltype(B))
    Ti = promote_type(indtype(A), indtype(B))
    A = _explicitdiag(A)
    B = _explicitdiag(B)
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
spmatmul1(A::UpperOrUnitUpperTriangular, B::UpperOrUnitUpperTriangular) = UpperTriangular(spmatmul(A, B))
spmatmul1(A::UnitUpperTriangular, B::UnitUpperTriangular) = UnitUpperTriangular(spmatmul(A, B))
spmatmul1(A::LowerOrUnitLowerTriangular, B::LowerOrUnitLowerTriangular) = LowerTriangular(spmatmul(A, B))
spmatmul1(A::UnitLowerTriangular, B::UnitLowerTriangular) = UnitLowerTriangular(spmatmul(A, B))
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

Base.@constprop :aggressive function mul!(C::SparseMatrixCSCOrColumnSubset, tA, tB, A::SparseMatrixCSCOrColumnSubset,
                            B::SparseMatrixCSCOrColumnSubset, alpha::Number, beta::Number)
    tA_uc, tB_uc = _uppercase(tA), _uppercase(tB)
    Anew, ta = tA_uc in ('S', 'H') ? (wrap(A, tA), oftype(tA, 'N')) : (A, tA)
    Bnew, tb = tB_uc in ('S', 'H') ? (wrap(B, tB), oftype(tB, 'N')) : (B, tB)
    @stable_muladdmul _generic_spmatmatmul!(C, ta, tb, Anew, Bnew, MulAddMul(alpha, beta))
end
# A writable sparse destination takes the sparse product; its pattern becomes that of
# `A*B*α + C*β`. A fixed destination is checked against that pattern before it is written.
Base.@constprop :aggressive function mul!(C::AbstractSparseMatrixCSC, tA, tB, A::SparseMatrixCSCOrColumnSubset,
                            B::SparseMatrixCSCOrColumnSubset, alpha::Number, beta::Number)
    mA, nA = LinearAlgebra.lapack_size(_uppercase(tA) in ('S', 'H') ? 'N' : tA, A)
    mB, nB = LinearAlgebra.lapack_size(_uppercase(tB) in ('S', 'H') ? 'N' : tB, B)
    nA == mB || throw(DimensionMismatch(lazy"matrix A has dimensions ($mA,$nA), matrix B has dimensions ($mB,$nB)"))
    size(C) == (mA, nB) || throw(DimensionMismatch(lazy"result C has dimensions $(size(C)), needs ($mA,$nB)"))
    iszero(alpha) && return LinearAlgebra._rmul_or_fill!(C, beta)
    P = _unwrapped_sparse(A, tA) * _unwrapped_sparse(B, tB)
    isone(alpha) || (P = P * alpha)
    R = iszero(beta) ? P : isone(beta) ? P + C : P + C * beta
    # converting first leaves `C` untouched if its eltype cannot hold the result
    return copyto!(C, convert(SparseMatrixCSC{eltype(C),indtype(C)}, R))
end
# only contiguous column views have a sparse product of their own
_unwrapped_sparse(A, t) = t == 'N' && A isa SparseMatrixCSCOrView ? A : sparse(wrap(A, t))

# Sparse-destination counterpart of `LinearAlgebra._generic_matmatmul!` (which this file also
# calls, qualified, for dense destinations); named distinctly so the two are not confused.
function _generic_spmatmatmul!(C::SparseMatrixCSCOrColumnSubset, tA, tB, A::AbstractVecOrMat,
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

# determine if sort! shall be used or the whole column be scanned
# based on empirical data on i7-3610QM CPU
# measuring runtimes of the scanning and sorting loops of the algorithm.
# The parameters 6 and 3 might be modified for different architectures.
prefer_sort(nz::Integer, m::Integer) = m > 6 && 3 * Base.top_set_bit(nz) * nz < m


## triangular multiplication
function LinearAlgebra.generic_trimatmul!(C::StridedVecOrMat, uploc, isunitc, tfun::Function, A::SparseMatrixCSCOrView, B::AbstractVecOrMat)
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
function LinearAlgebra.generic_trimatmul!(C::StridedVecOrMat, uploc, isunitc, ::Function, xA::AdjOrTrans{<:Any,<:SparseMatrixCSCOrView}, B::AbstractVecOrMat)
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

LinearAlgebra.generic_mattrimul!(C::StridedMatrix, uploc, isunitc, tfun::Function, A::AbstractMatrix, B::SparseMatrixCSCOrView) =
    _mattrimul!(C, uploc == 'U', isunitc == 'U', tfun, A, B)
LinearAlgebra.generic_mattrimul!(C::StridedMatrix, uploc, isunitc, ::Function, A::AbstractMatrix, xB::AdjOrTrans{<:Any,<:SparseMatrixCSCOrView}) =
    _mattrimul!(C, uploc == 'U', isunitc == 'U', conj, A, parent(xB))

# C = X * M, where M is the `upper` or lower triangle of B, elementwise `f` of it for
# `identity` and `conj`, or the `transpose`/`adjoint` `f` of it. The first kind gathers
# column `col` of C, the second scatters column `col` of X; either way the columns are
# visited so that none is written before its last read, and C may be X.
function _mattrimul!(C, upper::Bool, unit::Bool, f::Function, X, B)
    require_one_based_indexing(C, X)
    n = checksquare(B)
    size(X, 2) == n ||
        throw(DimensionMismatch(lazy"A has $(size(X, 2)) columns and B has $n rows"))
    size(C) == size(X) ||
        throw(DimensionMismatch(lazy"C has size $(size(C)), A * B has size $(size(X))"))
    rv = rowvals(B)
    nzv = nonzeros(B)
    rows = axes(X, 1)
    gather = f === identity || f === conj
    @inbounds for col in (upper == gather ? (n:-1:1) : (1:n))
        rng = upper ? nzrangeup(B, col) : nzrangelo(B, col)
        kd = upper ? last(rng) : first(rng)
        hasdiag = !isempty(rng) && rv[kd] == col
        offdiag = !hasdiag ? rng : upper ? (first(rng):kd-1) : (kd+1:last(rng))
        if !gather
            for k in offdiag
                a = f(nzv[k])
                row = rv[k]
                @simd for i in rows
                    C[i, row] = muladd(X[i, col], a, C[i, row])
                end
            end
        end
        d = unit ? oneunit(eltype(B)) : hasdiag ? f(nzv[kd]) : zero(eltype(B))
        @simd for i in rows
            C[i, col] = X[i, col] * d
        end
        if gather
            for k in offdiag
                a = f(nzv[k])
                row = rv[k]
                @simd for i in rows
                    C[i, col] = muladd(X[i, row], a, C[i, col])
                end
            end
        end
    end
    return C
end


function _symherm_mul!(rangefun::Function, diagop::Function, odiagop::Function, C::StridedVecOrMat{T}, A, B, α, β) where T
    n = size(A, 2)
    m = size(B, 2)
    n == size(B, 1) == size(C, 1) && m == size(C, 2) ||
        throw(DimensionMismatch("A has size $(size(A)), B has size $(size(B)), C has size $(size(C))"))
    rv = rowvals(A)
    nzv = nonzeros(A)
    let z = T(0), sumcol=z, αxj=z, aarc=z, α = α
        isone(β) || LinearAlgebra._rmul_or_fill!(C, β)
        @inbounds for k in axes(B,2)
            for col in axes(B,1)
                αxj = B[col,k] * α
                sumcol = z
                for j = rangefun(A, col)
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
end

function _A_mul_symherm!(rangefun::Function, diagop::Function, odiagop::Function, C::StridedMatrix, X::AbstractMatrix, A, α::Number, β::Number)
    Aax2 = axes(A, 2)
    Xax1 = axes(X, 1)
    mC, nC, mX, nX, mA, nA = _matmul_size_AB(C, X, A)
    rv = rowvals(A)
    nzv = nonzeros(A)
    isone(β) || LinearAlgebra._rmul_or_fill!(C, β)
    if α isa Bool && !α
        return
    end
    C = _fix_size(C, mC, nC)
    X = _fix_size(X, mX, nX)
    @inbounds for col in Aax2, k in rangefun(A, col)
        row = rv[k]
        if row == col
            Aiα = α isa Bool ? diagop(nzv[k]) : diagop(nzv[k]) * α
            @simd for i in Xax1
                C[i, col] = muladd(X[i, col], Aiα, C[i, col])
            end
        else
            Aiα = α isa Bool ? nzv[k] : nzv[k] * α
            Atiα = α isa Bool ? odiagop(nzv[k]) : odiagop(nzv[k]) * α
            @simd for i in Xax1
                C[i, col] = muladd(X[i, row], Aiα, C[i, col])
            end
            @simd for i in Xax1
                C[i, row] = muladd(X[i, col], Atiα, C[i, row])
            end
        end
    end
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

# multiply by diagonal matrix as vector
function mul!(C::AbstractSparseMatrixCSC, A::AbstractSparseMatrixCSC, D::Diagonal, alpha::Number, beta::Number)
    m, n = size(A)
    b = D.diag
    lb = length(b)
    n == lb || throw(DimensionMismatch(lazy"A has size ($m, $n) but D has size ($lb, $lb)"))
    size(A)==size(C) || throw(DimensionMismatch(lazy"A has size ($m, $n), D has size ($lb, $lb), C has size $(size(C))"))
    iszero(alpha) && (LinearAlgebra._rmul_or_fill!(nonzeros(C), beta); return C)
    beta_is_zero = iszero(beta)
    rows_match = rowvals(C) == rowvals(A)
    cols_match = getcolptr(C) == getcolptr(A)
    identical_nzinds = rows_match && cols_match
    Cnzval = nonzeros(C)
    Anzval = nonzeros(A)
    if identical_nzinds || (beta_is_zero && !_is_fixed(C))
        identical_nzinds || copyinds!(C, A, copy_rows = !rows_match, copy_cols = !cols_match)
        resize!(Cnzval, length(Anzval))
        @inbounds if beta_is_zero
            if isone(alpha)
                for col in axes(A,2), p in nzrange(A, col)
                    Cnzval[p] = Anzval[p] * b[col]
                end
            else
                for col in axes(A,2), p in nzrange(A, col)
                    Cnzval[p] = Anzval[p] * b[col] * alpha
                end
            end
        else
            if isone(alpha)
                for col in axes(A,2), p in nzrange(A, col)
                    Cnzval[p] = Anzval[p] * b[col] + Cnzval[p] * beta
                end
            else
                for col in axes(A,2), p in nzrange(A, col)
                    Cnzval[p] = Anzval[p] * b[col] * alpha + Cnzval[p] * beta
                end
            end
        end
    else
        mergeinds!(C, A)
        beta_is_zero && fill!(Cnzval, zero(eltype(C)))
        for col in axes(C,2), p in @inbounds nzrange(C, col)
            row = @inbounds rowvals(C)[p]
            # check if the index (row, col) is stored in A
            row_exists, row_ind_A = rowcheck_index(A, row, col)
            if row_exists
                if isone(alpha)
                    @inbounds Cnzval[p] = Anzval[row_ind_A] * b[col] + Cnzval[p] * beta
                else
                    @inbounds Cnzval[p] = Anzval[row_ind_A] * b[col] * alpha + Cnzval[p] * beta
                end
            else # A[row,col] == 0
                @inbounds Cnzval[p] = Cnzval[p] * beta
            end
        end
    end
    C
end

# Adjoint/transpose of a sparse matrix with a `Diagonal`: the generic
# `Diagonal` kernel in LinearAlgebra visits every element of `C`. With `beta == 0` the
# adjoint is formed directly in `C` (one `halfperm!`, O(nnz)) and scaled in place;
# otherwise it is materialized once and handed to the CSC kernels above, which also
# covers `alpha == 0`, a destination that shares storage with the parent, one whose index
# type or fixed structure `halfperm!` cannot write, and eltypes of `C`, `A` and the product
# that differ, so that no operand is converted before multiplying, as for dense (a real
# `Inf` times a complex one is not `complex(Inf)` times it).
function _adjtrans_into!(C::AbstractSparseMatrixCSC, A::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC})
    P = parent(A)
    return halfperm!(C, P, axes(P, 2), _adjtrans_fun(A))
end
_adjtrans_direct(C::AbstractSparseMatrixCSC, A::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}, D::Diagonal, alpha, beta) =
    !iszero(alpha) && iszero(beta) && !Base.mightalias(C, parent(A)) && !_is_fixed(C) &&
    indtype(C) === indtype(parent(A)) &&
    eltype(C) === eltype(A) === promote_op(matprod, eltype(A), eltype(D))

function mul!(C::AbstractSparseMatrixCSC, A::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}, D::Diagonal, alpha::Number, beta::Number)
    m, n = size(A)
    lb = length(D.diag)
    n == lb || throw(DimensionMismatch(lazy"A has size ($m, $n) but D has size ($lb, $lb)"))
    size(C) == (m, n) || throw(DimensionMismatch(lazy"A has size ($m, $n), D has size ($lb, $lb), C has size $(size(C))"))
    _adjtrans_direct(C, A, D, alpha, beta) || return mul!(C, copy(A), D, alpha, beta)
    rmul!(_adjtrans_into!(C, A), D)
    isone(alpha) || rmul!(C, alpha)
    return C
end

function mul!(C::AbstractSparseMatrixCSC, D::Diagonal, A::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}, alpha::Number, beta::Number)
    m, n = size(A)
    lb = length(D.diag)
    m == lb || throw(DimensionMismatch(lazy"D has size ($lb, $lb) but A has size ($m, $n)"))
    size(C) == (m, n) || throw(DimensionMismatch(lazy"A has size ($m, $n), D has size ($lb, $lb), C has size $(size(C))"))
    _adjtrans_direct(C, A, D, alpha, beta) || return mul!(C, D, copy(A), alpha, beta)
    lmul!(D, _adjtrans_into!(C, A))
    isone(alpha) || rmul!(C, alpha)
    return C
end

function mul!(C::AbstractSparseMatrixCSC, D::Diagonal, A::AbstractSparseMatrixCSC, alpha::Number, beta::Number)
    m, n = size(A)
    b    = D.diag
    lb = length(b)
    m == lb || throw(DimensionMismatch(lazy"D has size ($lb, $lb) but A has size ($m, $n)"))
    size(A)==size(C) || throw(DimensionMismatch(lazy"A has size ($m, $n), D has size ($lb, $lb), C has size $(size(C))"))
    iszero(alpha) && (LinearAlgebra._rmul_or_fill!(nonzeros(C), beta); return C)
    beta_is_zero = iszero(beta)
    rows_match = rowvals(C) == rowvals(A)
    cols_match = getcolptr(C) == getcolptr(A)
    identical_nzinds = rows_match && cols_match
    Cnzval = nonzeros(C)
    Anzval = nonzeros(A)
    Arowval = rowvals(A)
    if identical_nzinds || (beta_is_zero && !_is_fixed(C))
        identical_nzinds || copyinds!(C, A, copy_rows = !rows_match, copy_cols = !cols_match)
        resize!(Cnzval, length(Anzval))
        if beta_is_zero
            if isone(alpha)
                for col in axes(A,2), p in nzrange(A, col)
                    @inbounds Cnzval[p] = b[Arowval[p]] * Anzval[p]
                end
            else
                for col in axes(A,2), p in nzrange(A, col)
                    @inbounds Cnzval[p] = b[Arowval[p]] * Anzval[p] * alpha
                end
            end
        else
            if isone(alpha)
                for col in axes(A,2), p in nzrange(A, col)
                    @inbounds Cnzval[p] = b[Arowval[p]] * Anzval[p] + Cnzval[p] * beta
                end
            else
                for col in axes(A,2), p in nzrange(A, col)
                    @inbounds Cnzval[p] = b[Arowval[p]] * Anzval[p] * alpha + Cnzval[p] * beta
                end
            end
        end
    else
        mergeinds!(C, A)
        beta_is_zero && fill!(Cnzval, zero(eltype(C)))
        for col in axes(C,2), p in nzrange(C, col)
            row = rowvals(C)[p]
            # check if the index (row, col) is stored in A
            row_exists, row_ind_A = rowcheck_index(A, row, col)
            if row_exists
                if isone(alpha)
                    @inbounds Cnzval[p] = b[row] * Anzval[row_ind_A] + Cnzval[p] * beta
                else
                    @inbounds Cnzval[p] = b[row] * Anzval[row_ind_A] * alpha + Cnzval[p] * beta
                end
            else # A[row,col] == 0
                @inbounds Cnzval[p] = Cnzval[p] * beta
            end
        end
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

## sparse vectors

# scaling

function rmul!(x::SparseVectorOrView, a::Real)
    rmul!(nonzeros(x), a)
    return x
end
function rmul!(x::SparseVectorOrView, a::Complex)
    rmul!(nonzeros(x), a)
    return x
end
function lmul!(a::Real, x::SparseVectorOrView)
    rmul!(nonzeros(x), a)
    return x
end
function lmul!(a::Complex, x::SparseVectorOrView)
    rmul!(nonzeros(x), a)
    return x
end

(*)(x::SparseVectorOrView, a::Number) =
    @if_move_fixed x SparseVector(length(x), copy(nonzeroinds(x)), nonzeros(x) * a)
(*)(a::Number, x::SparseVectorOrView) =
    @if_move_fixed x SparseVector(length(x), copy(nonzeroinds(x)), a * nonzeros(x))

# * and mul!

_fliptri(A::UpperTriangular) = LowerTriangular(parent(parent(A)))
_fliptri(A::UnitUpperTriangular) = UnitLowerTriangular(parent(parent(A)))
_fliptri(A::LowerTriangular) = UpperTriangular(parent(parent(A)))
_fliptri(A::UnitLowerTriangular) = UnitUpperTriangular(parent(parent(A)))

Base.@constprop :aggressive function mul!(y::AbstractVector, tA, A::StridedMatrix, x::AbstractSparseVector,
                                                        alpha::Number, beta::Number)
    if tA == 'N'
        _spmul!(y, A, x, alpha, beta)
    elseif tA == 'T'
        _At_or_Ac_mul_B!(transpose, y, A, x, alpha, beta)
    elseif tA == 'C'
        _At_or_Ac_mul_B!(adjoint, y, A, x, alpha, beta)
    else
        _spmul!(y, wrap(A, tA), x, alpha, beta)
    end
    return y
end

LinearAlgebra._mul!(y::AbstractVector, A::UpperOrLowerTriangular, x::AbstractSparseVector,
                    alpha::Number, beta::Number) = mul!(y, 'N', A, x, alpha, beta)
function mul!(y::AbstractVector, tA, A::UpperOrLowerTriangular, x::AbstractSparseVector,
                alpha::Number, beta::Number)
    @assert tA == 'N'
    Adata = parent(A)
    if Adata isa Transpose
        _At_or_Ac_mul_B!(transpose, y, _fliptri(A), x, alpha, beta)
    elseif Adata isa Adjoint
        _At_or_Ac_mul_B!(adjoint, y, _fliptri(A), x, alpha, beta)
    else # Adata is plain
        _spmul!(y, A, x, alpha, beta)
    end
    return y
end
function _spmul!(y::AbstractVector, A::AbstractMatrix, x::AbstractSparseVector, α::Number, β::Number)
    require_one_based_indexing(y, A, x)
    m, n = size(A)
    length(x) == n || throw(DimensionMismatch(
        "Matrix A has $n columns, but vector x has a length $(length(x))"))
    length(y) == m || throw(DimensionMismatch(
        "Matrix A has $m rows, but vector y has a length $(length(y))"))
    m == 0 && return
    β != one(β) && LinearAlgebra._rmul_or_fill!(y, β)
    _iszero(α) && return

    xnzind = nonzeroinds(x)
    xnzval = nonzeros(x)
    @inbounds for i = 1:length(xnzind)
        v = xnzval[i]
        if _isnotzero(v)
            j = xnzind[i]
            αv = v * α
            for r = 1:m
                y[r] += A[r,j] * αv
            end
        end
    end
end

function _At_or_Ac_mul_B!(tfun::Function,
                            y::AbstractVector, A::Union{StridedMatrix,UpperOrLowerTriangular}, x::AbstractSparseVector,
                            α::Number, β::Number)
    require_one_based_indexing(y, A, x)
    n, m = size(A)
    length(x) == n || throw(DimensionMismatch(
        "Matrix A has $n rows, but vector x has a length $(length(x))"))
    length(y) == m || throw(DimensionMismatch(
        "Matrix A has $m columns, but vector y has a length $(length(y))"))
    m == 0 && return
    β != one(β) && LinearAlgebra._rmul_or_fill!(y, β)
    _iszero(α) && return

    xnzind = nonzeroinds(x)
    xnzval = nonzeros(x)
    _nnz = length(xnzind)
    _nnz == 0 && return

    Ty = promote_op(matprod, eltype(A), eltype(x))
    @inbounds for j = 1:m
        s = zero(Ty)
        for i = 1:_nnz
            s += tfun(A[xnzind[i], j]) * xnzval[i]
        end
        y[j] += s * α
    end
    return
end

### BLAS-2 / sparse A * sparse x -> dense y

function densemv(A::AbstractSparseMatrixCSC, x::AbstractSparseVector; trans::AbstractChar='N')
    local xlen::Int, ylen::Int
    require_one_based_indexing(A, x)
    m, n = size(A)
    if trans == 'N' || trans == 'n'
        xlen = n; ylen = m
        xaxis = "columns"
    elseif trans == 'T' || trans == 't' || trans == 'C' || trans == 'c'
        xlen = m; ylen = n
        xaxis = "rows"
    else
        throw(ArgumentError("Invalid trans character $trans"))
    end
    xlen == length(x) || throw(DimensionMismatch(
        "Matrix A has $xlen $xaxis, but vector x has a length $(length(x))"))
    T = promote_op(matprod, eltype(A), eltype(x))
    y = Vector{T}(undef, ylen)
    if trans == 'N' || trans == 'n'
        mul!(y, A, x)
    elseif trans == 'T' || trans == 't'
        mul!(y, transpose(A), x)
    else # trans == 'C' || trans == 'c'
        mul!(y, adjoint(A), x)
    end
    y
end

# * and mul!
mul!(y::AbstractVector, tA, A::AbstractSparseMatrixCSC, x::AbstractSparseVector, alpha::Number, beta::Number) =
    _spmatspvecmul!(y, tA, A, x, alpha, beta)
# disambiguates against the sparse matrix times dense vector method
mul!(y::StridedVector, tA, A::AbstractSparseMatrixCSC, x::AbstractSparseVector, alpha::Number, beta::Number) =
    _spmatspvecmul!(y, tA, A, x, alpha, beta)
Base.@constprop :aggressive function _spmatspvecmul!(y, tA, A, x, alpha, beta)
    if tA == 'N'
        _spmul!(y, A, x, alpha, beta)
    elseif tA == 'T'
        _At_or_Ac_mul_B!((a,b) -> transpose(a) * b, y, A, x, alpha, beta)
    elseif tA == 'C'
        _At_or_Ac_mul_B!((a,b) -> adjoint(a) * b, y, A, x, alpha, beta)
    else
        LinearAlgebra._generic_matvecmul!(y, 'N', wrap(A, tA), x, alpha, beta)
    end
    return y
end

function _spmul!(y::AbstractVector, A::AbstractSparseMatrixCSC, x::AbstractSparseVector, α::Number, β::Number)
    require_one_based_indexing(y, A, x)
    m, n = size(A)
    length(x) == n || throw(DimensionMismatch(
        "Matrix A has $n columns, but vector x has a length $(length(x))"))
    length(y) == m || throw(DimensionMismatch(
        "Matrix A has $m rows, but vector y has a length $(length(y))"))
    m == 0 && return
    β != one(β) && LinearAlgebra._rmul_or_fill!(y, β)
    _iszero(α) && return

    xnzind = nonzeroinds(x)
    xnzval = nonzeros(x)
    Arowval = rowvals(A)
    Anzval = nonzeros(A)

    @inbounds for i = 1:length(xnzind)
        v = xnzval[i]
        if _isnotzero(v)
            αv = v * α
            j = xnzind[i]
            for r = nzrange(A, j)
                y[Arowval[r]] += Anzval[r] * αv
            end
        end
    end
end

function _At_or_Ac_mul_B!(tfun::Function,
                          y::AbstractVector, A::AbstractSparseMatrixCSC, x::AbstractSparseVector,
                          α::Number, β::Number)
    require_one_based_indexing(y, A, x)
    m, n = size(A)
    length(x) == m || throw(DimensionMismatch(
        "Matrix A has $n columns, but vector x has a length $(length(x))"))
    length(y) == n || throw(DimensionMismatch(
        "Matrix A has $m rows, but vector y has a length $(length(y))"))
    n == 0 && return
    β != one(β) && LinearAlgebra._rmul_or_fill!(y, β)
    _iszero(α) && return

    xnzind = nonzeroinds(x)
    xnzval = nonzeros(x)
    Arowval = rowvals(A)
    Anzval = nonzeros(A)
    mx = length(xnzind)

    for j = 1:n
        # s <- dot(A[:,j], x)
        s = _spdot(tfun, Int(first(nzrange(A, j))), Int(last(nzrange(A, j))), Arowval, Anzval,
                   1, mx, xnzind, xnzval)
        @inbounds y[j] += s * α
    end
end


### BLAS-2 / sparse A * sparse x -> dense y

function *(A::AbstractSparseMatrixCSC, x::AbstractSparseVector)
    require_one_based_indexing(A, x)
    y = densemv(A, x)
    initcap = min(nnz(A), size(A,1))
    _dense2sparsevec(y, initcap)
end

*(xA::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}, x::AbstractSparseVector) =
    _At_or_Ac_mul_B((a,b) -> wrapperop(xA)(a) * b, parent(xA), x, promote_op(matprod, eltype(xA), eltype(x)))

function _At_or_Ac_mul_B(tfun::Function, A::AbstractSparseMatrixCSC{TvA,TiA}, x::AbstractSparseVector{TvX,TiX},
                         Tv = promote_op(matprod, TvA, TvX)) where {TvA,TiA,TvX,TiX}
    require_one_based_indexing(A, x)
    m, n = size(A)
    length(x) == m || throw(DimensionMismatch(
        "Matrix A has $m rows, but vector x has a length $(length(x))"))
    Ti = promote_type(TiA, TiX)

    xnzind = nonzeroinds(x)
    xnzval = nonzeros(x)
    Arowval = rowvals(A)
    Anzval = nonzeros(A)
    mx = length(xnzind)

    ynzind = Vector{Ti}(undef, n)
    ynzval = Vector{Tv}(undef, n)

    jr = 0
    for j = 1:n
        s = _spdot(tfun, Int(first(nzrange(A, j))), Int(last(nzrange(A, j))), Arowval, Anzval,
                   1, mx, xnzind, xnzval)
        if s != zero(s)
            jr += 1
            ynzind[jr] = j
            ynzval[jr] = s
        end
    end
    if jr < n
        resize!(ynzind, jr)
        resize!(ynzval, jr)
    end
    return @if_move_fixed A x SparseVector(n, ynzind, ynzval)
end

## dot

function dot(x::AbstractVector, y::SparseVectorOrView)
    require_one_based_indexing(x, y)
    n = length(x)
    length(y) == n || throw(DimensionMismatch(
        "Vector x has a length $n but y has a length $(length(y))"))
    nzind = nonzeroinds(y)
    nzval = nonzeros(y)
    s = _dot_zero(eltype(x), eltype(y))
    @inbounds for i = 1:length(nzind)
        s += dot(x[nzind[i]], nzval[i])
    end
    return s
end

function dot(x::SparseVectorOrView, y::AbstractVector)
    require_one_based_indexing(x, y)
    n = length(y)
    length(x) == n || throw(DimensionMismatch(
        "Vector x has a length $(length(x)) but y has a length $n"))
    nzind = nonzeroinds(x)
    nzval = nonzeros(x)
    s = _dot_zero(eltype(x), eltype(y))
    @inbounds for i = 1:length(nzind)
        s += dot(nzval[i], y[nzind[i]])
    end
    return s
end

function _spdot(f::Function,
                xj::Int, xj_last::Int, xnzind, xnzval,
                yj::Int, yj_last::Int, ynzind, ynzval)
    # dot product between ranges of non-zeros,
    s = zero(promote_op(f, eltype(xnzval), eltype(ynzval)))
    @inbounds while xj <= xj_last && yj <= yj_last
        ix = xnzind[xj]
        iy = ynzind[yj]
        if ix == iy
            s += f(xnzval[xj], ynzval[yj])
            xj += 1
            yj += 1
        elseif ix < iy
            xj += 1
        else
            yj += 1
        end
    end
    s
end

function dot(x::SparseVectorOrView, y::SparseVectorOrView)
    x === y && return sum(abs2, x)
    n = length(x)
    length(y) == n || throw(DimensionMismatch(
        "Vector x has a length $n but y has a length $(length(y))"))

    xnzind = nonzeroinds(x)
    ynzind = nonzeroinds(y)
    xnzval = nonzeros(x)
    ynzval = nonzeros(y)

    _spdot(dot,
           1, length(xnzind), xnzind, xnzval,
           1, length(ynzind), ynzind, ynzval)
end

# Frobenius dot/inner product: trace(A'B)
dot(A::AbstractSparseMatrixCSC, B::AbstractSparseMatrixCSC) = _dot_walk(dot, A, B)
# column views, spelled out so that each is more specific than the dense-operand methods below
dot(A::SparseMatrixCSCColumnSubset, B::AbstractSparseMatrixCSC) = _dot_walk(dot, A, B)
dot(A::AbstractSparseMatrixCSC, B::SparseMatrixCSCColumnSubset) = _dot_walk(dot, A, B)
dot(A::SparseMatrixCSCColumnSubset, B::SparseMatrixCSCColumnSubset) = _dot_walk(dot, A, B)

# The wrappers differ, so LinearAlgebra cannot strip them; with matching positions in the
# parents, only the elementwise operation changes.
dot(A::Adjoint{<:Any,<:SparseMatrixCSCOrColumnSubset}, B::Transpose{<:Any,<:SparseMatrixCSCOrColumnSubset}) =
    _dot_walk((a, b) -> dot(adjoint(a), transpose(b)), parent(A), parent(B))
dot(A::Transpose{<:Any,<:SparseMatrixCSCOrColumnSubset}, B::Adjoint{<:Any,<:SparseMatrixCSCOrColumnSubset}) =
    _dot_walk((a, b) -> dot(transpose(a), adjoint(b)), parent(A), parent(B))

# first stored index of column `j` and of the column after it; unlike `nzrange` this builds
# no range, whose empty-range normalization is measurable in per-column loops
Base.@propagate_inbounds _colbounds(A::AbstractSparseMatrixCSC, j) = (getcolptr(A)[j], getcolptr(A)[j+1])
Base.@propagate_inbounds _colbounds(A::SparseMatrixCSCColumnSubset, j) = _colbounds(parent(A), A.indices[2][j])

# `Σ f(A[i,j], B[i,j])` over the entries stored in both `A` and `B`
function _dot_walk(f::F, A::SparseMatrixCSCOrColumnSubset{T1,S1}, B::SparseMatrixCSCOrColumnSubset{T2,S2}) where {F,T1,T2,S1,S2}
    m, n = size(A)
    size(B) == (m,n) || throw(DimensionMismatch("matrices must have the same dimensions"))
    r = _dot_zero(T1, T2)
    @inbounds for j in axes(A,2)
        ia, ia_nxt = _colbounds(A, j)
        ib, ib_nxt = _colbounds(B, j)
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
                    r += f(nonzeros(A)[ia], nonzeros(B)[ib])
                    ia += oneunit(S1); ib += oneunit(S2)
                    ia < ia_nxt && ib < ib_nxt || break
                    ra = rowvals(A)[ia]; rb = rowvals(B)[ib]
                end
            end
        end
    end
    return r
end

function dot(x::AbstractVector{T1}, A::SparseMatrixCSCOrColumnSubset{T2}, y::AbstractVector{T3}) where {T1,T2,T3}
    require_one_based_indexing(x, y)
    m, n = size(A)
    (length(x) == m && n == length(y)) ||
        throw(DimensionMismatch("x has length $(length(x)), A has size ($m, $n), y has length $(length(y))"))
    s = _dot_zero(T1, T2, T3)
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
function dot(x::AbstractSparseVector, A::SparseMatrixCSCOrColumnSubset, y::AbstractSparseVector)
    m, n = size(A)
    length(x) == m && n == length(y) ||
        throw(DimensionMismatch("x has length $(length(x)), A has size ($m, $n), y has length $(length(y))"))
    r = _dot_zero(eltype(x), eltype(A), eltype(y))
    (iszero(m) || iszero(n)) && return r
    xnzind = nonzeroinds(x)
    xnzval = nonzeros(x)
    ynzind = nonzeroinds(y)
    ynzval = nonzeros(y)
    Arowval = getrowval(A)
    Anzval = getnzval(A)
    for (yi, yv) in zip(ynzind, ynzval)
        A_ptr_lo = Int(first(nzrange(A, yi)))
        A_ptr_hi = Int(last(nzrange(A, yi)))
        if A_ptr_lo <= A_ptr_hi
            r += _spdot((xv, av) -> dot(xv, av, yv), 1, length(xnzind), xnzind, xnzval,
                                            A_ptr_lo, A_ptr_hi, Arowval, Anzval)
        end
    end
    r
end

function dot(A::Union{DenseMatrixUnion,MatrixWrappersOrView{<:Any,<:Union{DenseMatrixUnion,AbstractSparseMatrix}}}, B::SparseMatrixCSCOrColumnSubset)
    (m, n) = size(A)
    if (m, n) != size(B)
        throw(DimensionMismatch("A has size ($m, $n) but B has size $(size(B))"))
    end
    s = _dot_zero(eltype(A), eltype(B))
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

function dot(A::SparseMatrixCSCOrColumnSubset, B::Union{DenseMatrixUnion,MatrixWrappersOrView{<:Any,<:Union{DenseMatrixUnion,AbstractSparseMatrix}}})
    return conj(dot(B, A))
end
dot(A::SparseMatrixCSCOrColumnSubset, B::AdjOrTrans{<:Any,<:SparseMatrixCSCColumnSubset}) = conj(dot(B, A))

# Frobenius dot of the adjoint/transpose of a CSC matrix with a CSC matrix.
# With `P = parent(A)`, `dot(A, B) = Σ dot(op(P[j,i]), B[i,j])`, so the stored entries of
# one operand are matched against those of the other at transposed positions. Walking the
# operand with fewer stored entries and columns, with one cursor per column of the other,
# keeps the work at O(nnz(P) + nnz(B) + n) with O(n) extra memory, where `n` counts the
# columns of the other operand; a binary search per stored entry is used instead when the
# other operand is far denser, since the cursors would then sweep all of its entries.
dot(A::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}, B::AbstractSparseMatrixCSC) = _dot_transposed(A, B)
dot(A::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}, B::SparseMatrixCSCColumnSubset) = _dot_transposed(A, B)
dot(A::AdjOrTrans{<:Any,<:SparseMatrixCSCColumnSubset}, B::AbstractSparseMatrixCSC) = _dot_transposed(A, B)
dot(A::AdjOrTrans{<:Any,<:SparseMatrixCSCColumnSubset}, B::SparseMatrixCSCColumnSubset) = _dot_transposed(A, B)
function _dot_transposed(A, B)
    m, n = size(A)
    size(B) == (m, n) || throw(DimensionMismatch(lazy"A has size ($m, $n) but B has size $(size(B))"))
    P = parent(A)
    op = LinearAlgebra.wrapperop(A)
    r = _dot_zero(eltype(A), eltype(B))
    (iszero(nnz(P)) || iszero(nnz(B))) && return r
    if nnz(B) + size(B, 2) <= nnz(P) + size(P, 2)
        return _dot_transposed_walk((b, p) -> dot(op(p), b), B, P, r)
    else
        return _dot_transposed_walk((p, b) -> dot(op(p), b), P, B, r)
    end
end

# `r + Σ f(X[i,j], Y[j,i])` over the stored entries of `X` that have a stored counterpart
# in `Y`. Walking the columns of `X` in order, the row index `j` looked for in column `i`
# of `Y` is nondecreasing, so one cursor per column of `Y` suffices. The cursors visit
# every stored entry of `Y`, so once `Y` holds well over an order of magnitude more entries
# (or columns) than `X`, a binary search per entry of `X` is cheaper; the crossover is at
# a ratio of about 20-50 in measurements.
# a fresh vector of the first stored index of each column
_colstarts(Y::AbstractSparseMatrixCSC) = getcolptr(Y)[1:size(Y, 2)]
_colstarts(Y::SparseMatrixCSCColumnSubset) = [first(_colbounds(Y, i)) for i in axes(Y, 2)]
# `(v, off)` with `v[i+off]` one past the last stored index of column `i`; indexing `colptr`
# at an offset is measurably faster in the walk than a view of it
_colstops(Y::AbstractSparseMatrixCSC) = (getcolptr(Y), 1)
_colstops(Y::SparseMatrixCSCColumnSubset) = ([last(_colbounds(Y, i)) for i in axes(Y, 2)], 0)

function _dot_transposed_walk(f::F, X::SparseMatrixCSCOrColumnSubset, Y::SparseMatrixCSCOrColumnSubset, r) where F
    Xrows, Xvals = rowvals(X), nonzeros(X)
    Yrows, Yvals = rowvals(Y), nonzeros(Y)
    if size(Y, 2) + nnz(Y) > 32 * nnz(X)
        @inbounds for j in axes(X, 2), k in nzrange(X, j)
            i = Xrows[k]
            rng = nzrange(Y, i)
            p = searchsortedfirst(view(Yrows, rng), j) + first(rng) - 1
            if p <= last(rng) && Yrows[p] == j
                r += f(Xvals[k], Yvals[p])
            end
        end
        return r
    end
    cursor = _colstarts(Y)   # cursor[i] indexes into column i of Y
    pends, off = _colstops(Y)
    @inbounds for j in axes(X, 2), k in nzrange(X, j)
        i = Xrows[k]
        p = cursor[i]
        pend = pends[i+off]
        while p < pend && Yrows[p] < j
            p += 1
        end
        cursor[i] = p
        if p < pend && Yrows[p] == j
            r += f(Xvals[k], Yvals[p])
        end
    end
    return r
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
    s = _dot_zero(eltype(x), eltype(D), eltype(y))
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
    s = _dot_zero(eltype(x), eltype(D), eltype(y))
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

function dot(
    a::AbstractSparseVector,
    Q::Union{DenseMatrixUnion,MatrixWrappersOrView{<:Any,<:DenseMatrixUnion}},
    b::AbstractSparseVector,
)
    return _dot_quadratic_form(a, Q, b)
end

function dot(
    a::AbstractSparseVector,
    Q::LinearAlgebra.Transpose{<:Real,<:DenseMatrixUnion},
    b::AbstractSparseVector,
)
    return _dot_quadratic_form(a, Q, b)
end

function dot(
    a::AbstractSparseVector,
    Q::LinearAlgebra.Transpose{<:Real,<:MatrixWrappersOrView{<:Real,<:DenseMatrixUnion}},
    b::AbstractSparseVector,
)
    return _dot_quadratic_form(a, Q, b)
end

function dot(
    a::AbstractSparseVector,
    Q::LinearAlgebra.RealHermSymComplexHerm{<:Real,<:DenseMatrixUnion},
    b::AbstractSparseVector)
    return _dot_quadratic_form(a, Q, b)
end

function dot(
    a::AbstractSparseVector,
    Q::Union{
        LinearAlgebra.Hermitian{<:Real,<:DenseMatrixUnion}, LinearAlgebra.Symmetric{<:Real,<:DenseMatrixUnion}
    },
    b::AbstractSparseVector)
    return _dot_quadratic_form(a, Q, b)
end

# actual function implementation called by the method dispatch
function _dot_quadratic_form(a, Q, b)
    n = length(a)
    m = length(b)
    if size(Q) != (n, m)
        throw(DimensionMismatch("Matrix has a size $(size(Q)) but vectors have length $n, $m"))
    end
    anzind = nonzeroinds(a)
    bnzind = nonzeroinds(b)
    anzval = nonzeros(a)
    bnzval = nonzeros(b)
    s = zero(Base.promote_eltype(a, Q, b))
    if isempty(anzind) || isempty(bnzind)
        return s
    end
    @inbounds for a_idx in eachindex(anzind)
        for b_idx in eachindex(bnzind)
            ia = anzind[a_idx]
            ib = bnzind[b_idx]
            s += dot(anzval[a_idx], Q[ia, ib], bnzval[b_idx])
        end
    end
    return s
end

# symmetric/Hermitian


dot(x::AbstractVector, A::HermOrSym{<:Any,<:SparseMatrixCSCOrColumnSubset}, y::AbstractVector) =
    _dot(x, parent(A), y, A.uplo == 'U' ? nzrangeup : nzrangelo, A isa Symmetric ? identity : real, A isa Symmetric ? transpose : adjoint)
# disambiguation
dot(x::AbstractVector, A::RealHermSymComplexHerm{<:Real,<:SparseMatrixCSCOrColumnSubset}, y::AbstractVector) =
    _dot(x, parent(A), y, A.uplo == 'U' ? nzrangeup : nzrangelo, A isa Symmetric ? identity : real, A isa Symmetric ? transpose : adjoint)
function _dot(x::AbstractVector, A::SparseMatrixCSCOrColumnSubset, y::AbstractVector, rangefun::Function, diagop::Function, odiagop::Function)
    require_one_based_indexing(x, y)
    m, n = size(A)
    (length(x) == m && n == length(y)) ||
        throw(DimensionMismatch("x has length $(length(x)), A has size ($m, $n), y has length $(length(y))"))
    r = _dot_zero(eltype(x), eltype(A), eltype(y))
    (iszero(m) || iszero(n)) && return r
    rvals = getrowval(A)
    nzvals = getnzval(A)
    @inbounds for col in axes(A,2)
        ycol = y[col]
        xcol = x[col]
        if _isnotzero(ycol) || _isnotzero(xcol)
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
dot(x::AbstractSparseVector, A::HermOrSym{<:Any,<:SparseMatrixCSCOrColumnSubset}, y::AbstractSparseVector) =
    _dot(x, parent(A), y, A.uplo == 'U' ? nzrangeup : nzrangelo, A isa Symmetric ? identity : real, A isa Symmetric ? transpose : adjoint)
# disambiguation
dot(x::AbstractSparseVector, A::RealHermSymComplexHerm{<:Real,<:SparseMatrixCSCOrColumnSubset}, y::AbstractSparseVector) =
    _dot(x, parent(A), y, A.uplo == 'U' ? nzrangeup : nzrangelo, A isa Symmetric ? identity : real, A isa Symmetric ? transpose : adjoint)
function _dot(x::AbstractSparseVector, A::SparseMatrixCSCOrColumnSubset, y::AbstractSparseVector, rangefun::Function, diagop::Function, odiagop::Function)
    m, n = size(A)
    length(x) == m && n == length(y) ||
        throw(DimensionMismatch("x has length $(length(x)), A has size ($m, $n), y has length $(length(y))"))
    r = _dot_zero(eltype(x), eltype(A), eltype(y))
    (iszero(m) || iszero(n)) && return r
    xnzind = nonzeroinds(x)
    xnzval = nonzeros(x)
    ynzind = nonzeroinds(y)
    ynzval = nonzeros(y)
    Arowval = getrowval(A)
    Anzval = getnzval(A)
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
            r += dot(xv, _spdot((a, y) -> odiagop(a)*y, A_ptr_lo, A_ptr_hi, Arowval, Anzval,
                                            1, length(ynzind), ynzind, ynzval))
        end
    end
    # diagonal
    @inbounds for i in axes(A,1)
        r1 = Int(first(nzrange(A, i)))
        r2 = Int(last(nzrange(A, i)))
        r1 > r2 && continue
        r1 += searchsortedfirst(view(Arowval, r1:r2), i) - 1
        ((r1 > r2) || (Arowval[r1] != i)) && continue
        r += dot(x[i], diagop(Anzval[r1]), y[i])
    end
    r
end
## end of symmetric/Hermitian

## kron
const _SparseKronGroup = Union{SparseVecOrMatMaybeAdjOrTrans,
                               HermOrSym{<:Any,<:SparseVecOrMatMaybeAdjOrTrans},
                               UpperOrLowerTriangular{<:Any,<:SparseVecOrMatMaybeAdjOrTrans}}
const _DenseKronGroup = Union{Number, Vector, Matrix, AdjOrTrans{<:Any,<:VecOrMat}, BandedMatrix,
                              HermOrSym{<:Any,<:Matrix}, UpperOrLowerTriangular{<:Any,<:Matrix}}

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
        lA = length(nzrange(A, j))
        for i in axes(B,2)
            startB = first(nzrange(B, i))
            lB = length(nzrange(B, i))
            ptr_range = (1:lB) .+ (colptrC[col]-1)
            colptrC[col+1] = colptrC[col] + lA*lB
            col += 1
            for ptrA = nzrange(A, j)
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
kron!(C::SparseMatrixCSC, A::SparseVectorOrView, B::AdjOrTrans{<:Any,<:SparseVectorOrView}) =
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
kron(A::SparseVectorOrView, B::AdjOrTrans{<:Any,<:SparseVectorOrView}) = A .* B
# disambiguation
kron(A::AbstractCompressedVector, B::AdjOrTrans{<:Any,<:AbstractCompressedVector}) = A .* B
kron(a::Number, b::_SparseKronGroup) = a * b
kron(a::_SparseKronGroup, b::Number) = a * b
