# This file is a part of Julia. License is MIT: https://julialang.org/license

using LinearAlgebra: AbstractTriangular, UpperOrLowerTriangular,
    RealHermSymComplexHerm, checksquare, sym_uplo, wrap
using Random: rand!, Xoshiro

import LinearAlgebra: _uppercase, _isuppercase

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
Base.@propagate_inbounds _colbounds(A::SparseMatrixCSCColumnSubset, j) = _colbounds(parent(A), parentindices(A)[2][j])

# `Σ f(A[i,j], B[i,j])` over the entries stored in both `A` and `B`
function _dot_walk(f::F, A::SparseMatrixCSCOrColumnSubset{T1,S1}, B::SparseMatrixCSCOrColumnSubset{T2,S2}) where {F,T1,T2,S1,S2}
    m, n = size(A)
    size(B) == (m,n) || throw(DimensionMismatch("matrices must have the same dimensions"))
    r = _dot_zero(T1, T2)
    @inbounds for j in axes(A,2)
        ia, ia_nxt = _colbounds(A, j)
        ib, ib_nxt = _colbounds(B, j)
        if ia < ia_nxt && ib < ib_nxt
            ra = getrowval(A)[ia]; rb = getrowval(B)[ib]
            while true
                if ra < rb
                    ia += oneunit(S1)
                    ia < ia_nxt || break
                    ra = getrowval(A)[ia]
                elseif ra > rb
                    ib += oneunit(S2)
                    ib < ib_nxt || break
                    rb = getrowval(B)[ib]
                else # ra == rb
                    r += f(getnzval(A)[ia], getnzval(B)[ib])
                    ia += oneunit(S1); ib += oneunit(S2)
                    ia < ia_nxt && ib < ib_nxt || break
                    ra = getrowval(A)[ia]; rb = getrowval(B)[ib]
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
        for j in getnzrange(A, col)
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
        A_ptr_lo = Int(first(getnzrange(A, yi)))
        A_ptr_hi = Int(last(getnzrange(A, yi)))
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
    rows = getrowval(B)
    vals = getnzval(B)
    @inbounds for j in axes(A,2)
        for ridx in getnzrange(B, j)
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
    Xrows, Xvals = getrowval(X), getnzval(X)
    Yrows, Yvals = getrowval(Y), getnzval(Y)
    if size(Y, 2) + nnz(Y) > 32 * nnz(X)
        @inbounds for j in axes(X, 2), k in getnzrange(X, j)
            i = Xrows[k]
            rng = getnzrange(Y, i)
            p = searchsortedfirst(view(Yrows, rng), j) + first(rng) - 1
            if p <= last(rng) && Yrows[p] == j
                r += f(Xvals[k], Yvals[p])
            end
        end
        return r
    end
    cursor = _colstarts(Y)   # cursor[i] indexes into column i of Y
    pends, off = _colstops(Y)
    @inbounds for j in axes(X, 2), k in getnzrange(X, j)
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

## triangular sparse handling
## triangular solvers
_uconvert_copyto!(c, b, oA) = (c .= Ref(oA) .\ b)
_uconvert_copyto!(c::AbstractArray{T}, b::AbstractArray{T}, _) where {T} = copyto!(c, b)

LinearAlgebra.generic_trimatdiv!(C::StridedVecOrMat, uploc, isunitc, tfun::Function, A::SparseMatrixCSCOrView, B::AbstractVecOrMat) =
    _trimatdiv!(C, uploc == 'U', isunitc == 'U', tfun, A, B)
LinearAlgebra.generic_trimatdiv!(C::StridedVecOrMat, uploc, isunitc, ::Function, xA::AdjOrTrans{<:Any,<:SparseMatrixCSCOrView}, B::AbstractVecOrMat) =
    _trimatdiv!(C, uploc == 'U', isunitc == 'U', conj, parent(xA), B)

# C = M \ B for a triangle M of `f(A)`, solved column by column of A; C may be B.
# `F` keeps the method specialized on the forwarded `f`.
function _trimatdiv!(C, upper::Bool, unit::Bool, f::F, A, B) where {F<:Function}
    n = size(A, 2)
    size(B, 1) == n ||
        throw(DimensionMismatch(lazy"second dimension of left hand side A, $n, and first dimension of right hand side B, $(size(B, 1)), must be equal"))
    size(C) == size(B) ||
        throw(DimensionMismatch(lazy"size of output, $(size(C)), does not match size of right hand side, $(size(B))"))
    if C === B
        # solving in place overwrites the right-hand side, so a singular triangle is found first
        if !unit
            j = _trisingular(upper, A)
            j == 0 || throw(LinearAlgebra.SingularException(j))
        end
    else
        _uconvert_copyto!(C, B, oneunit(eltype(A)))
    end
    return upper ? _trimatdiv!(C, Val(true), unit, f, A, B) : _trimatdiv!(C, Val(false), unit, f, A, B)
end
# The position in the storage of `A` of the diagonal entry of column `j`, which is the last
# entry of the upper triangle and the first of the lower one, and whether it is stored. A
# matrix that stores one triangle only has it at the end of the column, with no search.
@inline function _tridiagindex(upper::Bool, ja, i1::Int, i2::Int, j::Int)
    i1 > i2 && return (upper ? i2 : i1), false
    ii = upper ? i2 : i1
    @inbounds ja[ii] == j && return ii, true
    ii = upper ? searchsortedlast(ja, j, i1, i2, Base.Order.Forward) :
                 searchsortedfirst(ja, j, i1, i2, Base.Order.Forward)
    return ii, i1 <= ii <= i2 && @inbounds(ja[ii]) == j
end
# The first column of `A` whose diagonal entry is not stored or is a stored zero, or 0.
function _trisingular(upper::Bool, A)
    aa = getnzval(A)
    ja = getrowval(A)
    ia = getcolptr(A)
    for j in 1:size(A, 2)
        ii, hasdiag = _tridiagindex(upper, ja, Int(ia[j]), Int(ia[j + 1]) - 1, j)
        hasdiag && !iszero(aa[ii]) || return j
    end
    return 0
end
# A diagonal entry that is not stored or is a stored zero is singular, as a zero on the
# diagonal of a dense triangle is. The elements of `A` multiply and divide from the left,
# as they do in `M * X == B`.
function _trimatdiv!(C, ::Val{upper}, unit::Bool, f::Function, A, B) where {upper}
    n = size(A, 2)
    aa = getnzval(A)
    ja = getrowval(A)
    ia = getcolptr(A)
    direct = f === identity || f === conj
    for k in axes(B, 2), j in (upper == direct ? (n:-1:1) : (1:n))
        i1 = Int(ia[j])
        i2 = Int(ia[j + 1]) - 1
        if direct
            ii, hasdiag = _tridiagindex(upper, ja, i1, i2, j)
            cj = C[j,k]
            if hasdiag
                if !unit
                    d = f(aa[ii])
                    iszero(d) && throw(LinearAlgebra.SingularException(j))
                    cj = LinearAlgebra._ustrip(d) \ cj
                    C[j,k] = cj
                end
                ii += upper ? -1 : 1
            elseif !unit
                throw(LinearAlgebra.SingularException(j))
            end
            for i in (upper ? (ii:-1:i1) : (ii:i2))
                C[ja[i],k] -= LinearAlgebra._ustrip(f(aa[i])) * cj
            end
        else
            akku = C[j,k]
            done = false
            for ii in (upper ? (i1:i2) : (i2:-1:i1))
                jai = Int(ja[ii])
                d = upper ? j - jai : jai - j
                if d > 0
                    akku -= f(aa[ii]) * C[jai,k]
                elseif d == 0
                    if unit
                        akku = oneunit(eltype(A)) \ akku
                    else
                        a = f(aa[ii])
                        iszero(a) && throw(LinearAlgebra.SingularException(j))
                        akku = a \ akku
                    end
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
    return C
end

matop_dest(::typeof(\), A, b::AbstractSparseVector) =
    Vector{promote_op(\, eltype(A), eltype(b))}(undef, length(b))
matop_dest(::typeof(\), A::UnitUpperOrUnitLowerTriangular, b::AbstractSparseVector) =
    Vector{LinearAlgebra._inner_type_promotion(\, eltype(A), eltype(b))}(undef, length(b))
matop_dest(::typeof(\), A::Diagonal, b::AbstractSparseVector) =
    similar(b , promote_op(\, eltype(A), eltype(b)))
# Solves return dense. A triangle of a lazy adjoint or transpose of a sparse matrix is a
# sparse operand like the other wrappers.
matop_dest(::typeof(\), A, B::Union{QuasiSparseMatrix,SparseAdjOrTransTriangular}) =
    Matrix{promote_op(\, eltype(A), eltype(B))}(undef, size(B))
# a `Diagonal` keeps the structure of the sparse operand, and its triangle
matop_dest(::typeof(\), A::Diagonal, B::Union{QuasiSparseMatrix,SparseAdjOrTransTriangular}) =
    B isa SparseAdjOrTransTriangular ? similar(B, promote_op(\, eltype(A), eltype(B))) :
                                       similar(B, promote_op(\, eltype(A), eltype(B)), size(B))
matop_dest(::typeof(\), A::UnitUpperOrUnitLowerTriangular, B::Union{QuasiSparseMatrix,SparseAdjOrTransTriangular}) =
    Matrix{LinearAlgebra._inner_type_promotion(\, eltype(A), eltype(B))}(undef, size(B))
matop_dest(::typeof(/), A::Union{QuasiSparseMatrix,SparseAdjOrTransTriangular}, B) =
    Matrix{promote_op(/, eltype(A), eltype(B))}(undef, size(A))
matop_dest(::typeof(/), A::Union{QuasiSparseMatrix,SparseAdjOrTransTriangular}, B::UnitUpperOrUnitLowerTriangular) =
    Matrix{LinearAlgebra._inner_type_promotion(/, eltype(A), eltype(B))}(undef, size(A))
matop_dest(::typeof(/), A::Union{QuasiSparseMatrix,SparseAdjOrTransTriangular}, B::Diagonal) =
    A isa SparseAdjOrTransTriangular ? similar(A, promote_op(/, eltype(A), eltype(B))) :
                                       similar(A, promote_op(/, eltype(A), eltype(B)), size(A))
## end of triangular

# symmetric/Hermitian


dot(x::AbstractVector, A::HermOrSym{<:Any,<:SparseMatrixCSCOrColumnSubset}, y::AbstractVector) =
    _dot(x, parent(A), y, _symherm_ops(A)...)
# disambiguation
dot(x::AbstractVector, A::RealHermSymComplexHerm{<:Real,<:SparseMatrixCSCOrColumnSubset}, y::AbstractVector) =
    _dot(x, parent(A), y, _symherm_ops(A)...)
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
    _dot(x, parent(A), y, _symherm_ops(A)...)
# disambiguation
dot(x::AbstractSparseVector, A::RealHermSymComplexHerm{<:Real,<:SparseMatrixCSCOrColumnSubset}, y::AbstractSparseVector) =
    _dot(x, parent(A), y, _symherm_ops(A)...)
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
        r1 = Int(first(getnzrange(A, i)))
        r2 = Int(last(getnzrange(A, i)))
        r1 > r2 && continue
        r1 += searchsortedfirst(view(Arowval, r1:r2), i) - 1
        ((r1 > r2) || (Arowval[r1] != i)) && continue
        r += dot(x[i], diagop(Anzval[r1]), y[i])
    end
    r
end
## end of symmetric/Hermitian

\(A::Transpose{<:Complex,<:Hermitian{<:Complex,<:AbstractSparseMatrixCSC}}, B::Vector) = copy(A) \ B

## diagonal scaling

# Scale the stored entries of `A` into `C`, which either is `A` or takes its pattern:
# `C[i, j] = op(A[i, j], d[j])` columnwise, `C[i, j] = op(d[i], A[i, j])` rowwise. The
# argument order of `op` is kept for non-commutative eltypes. The callers check that
# `length(d)` matches the scaled dimension and that `size(C) == size(A)`.
function _scalecols!(C::AbstractSparseMatrixCSC, A::AbstractSparseMatrixCSC, d::AbstractVector, op::F) where {F}
    copyinds!(C, A)
    Cnzval = getnzval(C)
    Anzval = getnzval(A)
    resize!(Cnzval, length(Anzval))
    @inbounds for col in axes(A, 2)
        dcol = d[col]
        for p in nzrange(A, col)
            Cnzval[p] = op(Anzval[p], dcol)
        end
    end
    return C
end

function _scalerows!(C::AbstractSparseMatrixCSC, A::AbstractSparseMatrixCSC, d::AbstractVector, op::F) where {F}
    copyinds!(C, A)
    Cnzval = getnzval(C)
    Anzval = getnzval(A)
    Arowval = getrowval(A)
    resize!(Cnzval, length(Anzval))
    @inbounds for col in axes(A, 2), p in nzrange(A, col)
        Cnzval[p] = op(d[Arowval[p]], Anzval[p])
    end
    return C
end

function _scalerows!(x::AbstractSparseVector, d::AbstractVector, op::F) where {F}
    xnzval = nonzeros(x)
    xnzind = nonzeroinds(x)
    @inbounds for p in eachindex(xnzind, xnzval)
        xnzval[p] = op(d[xnzind[p]], xnzval[p])
    end
    return x
end

function _checkscaledims(C, A, D::Diagonal, dim::Int)
    n = size(A, dim)
    k = length(D.diag)
    n == k || throw(DimensionMismatch("A has size $(size(A)) but D has size ($k, $k)"))
    size(C) == size(A) || throw(DimensionMismatch("A has size $(size(A)) but C has size $(size(C))"))
    return nothing
end

function _checknonsingular(D::Diagonal)
    i = findfirst(iszero, D.diag)
    isnothing(i) || throw(SingularException(i))
    return nothing
end

function LinearAlgebra._rdiv!(C::AbstractSparseMatrixCSC, A::AbstractSparseMatrixCSC, D::Diagonal)
    _checkscaledims(C, A, D, 2)
    _checknonsingular(D)
    return _scalecols!(C, A, D.diag, /)
end

rdiv!(A::AbstractSparseMatrixCSC, D::Diagonal) = LinearAlgebra._rdiv!(A, A, D)

function ldiv!(C::AbstractSparseMatrixCSC, D::Diagonal, A::AbstractSparseMatrixCSC)
    _checkscaledims(C, A, D, 1)
    _checknonsingular(D)
    return _scalerows!(C, A, D.diag, \)
end

ldiv!(D::Diagonal, A::AbstractSparseMatrixCSC) = ldiv!(A, D, A)

function ldiv!(D::Diagonal, x::AbstractSparseVector)
    _checkscaledims(x, x, D, 1)
    _checknonsingular(D)
    return _scalerows!(x, D.diag, \)
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
            getrowval(S)[c1] > col - k && break
            nnz += 1
        end
        colptr[col+1] = nnz+1
    end
    rowval = Vector{Ti}(undef, nnz)
    nzval = Vector{Tv}(undef, nnz)
    @inbounds for col = max(k+1,1) : n
        c1 = Int(getcolptr(S)[col])
        for c2 in colptr[col]:colptr[col+1]-1
            rowval[c2] = getrowval(S)[c1]
            nzval[c2] = getnzval(S)[c1]
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
            getrowval(S)[l1 - c1] < col - k && break
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
            rowval[l2 - c2] = getrowval(S)[c1]
            nzval[l2 - c2] = getnzval(S)[c1]
            c1 -= 1
        end
    end
    SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

## diff

# Append the first difference along one stored column (rows `rowval_a[rng]`, values
# `nzval_a[rng]`, length `m`) to `rowval`/`nzval` after entry `numnz`, and return the
# new count. Each stored entry `(i, v)` contributes `+v` at `i-1` and `-v` at `i`; the
# two contributions at the same row of consecutive stored entries merge in place.
function _sparse_diff1_column!(rowval, nzval, numnz, rowval_a, nzval_a, rng, m)
    last_row = 0
    @inbounds for k in rng
        row = rowval_a[k]
        val = nzval_a[k]
        if row > 1
            if row == last_row + 1
                nzval[numnz] += val
                iszero(nzval[numnz]) && (numnz -= 1)
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
    end
    return numnz
end

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
        numnz = _sparse_diff1_column!(rowval, nzval, numnz, getrowval(S), getnzval(S), nzrange(S, col), m)
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
    rowval_a = getrowval(a)
    nzval_a = getnzval(a)

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

function diff(a::AbstractSparseMatrixCSC; dims::Integer)
    1 <= dims <= 2 || throw(ArgumentError("dimension $dims out of range (1:2)"))
    return dims == 1 ? sparse_diff1(a) : sparse_diff2(a)
end

function _sparse_diff(x, dims)
    dims == 1 || throw(ArgumentError("dimension $dims out of range (1:1)"))
    n = length(x)
    nzind_x = nonzeroinds(x)
    nzval_x = nonzeros(x)
    Ti, Tv = eltype(nzind_x), eltype(nzval_x)
    n > 1 || return SparseVector(0, Ti[], Tv[])
    numnz = 2 * length(nzind_x) # upper bound; will shrink later
    nzind = Vector{Ti}(undef, numnz)
    nzval = Vector{Tv}(undef, numnz)
    numnz = _sparse_diff1_column!(nzind, nzval, 0, nzind_x, nzval_x, eachindex(nzind_x), n)
    deleteat!(nzind, numnz+1:length(nzind))
    deleteat!(nzval, numnz+1:length(nzval))
    return SparseVector(n-1, nzind, nzval)
end
diff(x::AbstractSparseVector; dims::Integer=1) = _sparse_diff(x, dims)
diff(x::Union{SparseColumnView, SparseVectorView}; dims::Integer=1) = _sparse_diff(x, dims)

## norm and rank
function norm(A::SparseMatrixCSCOrView, p::Real=2)
    v = nzvalview(A)
    r = norm(v, p)
    # an implicit zero makes a norm with p < 0 zero, but `min` keeps a stored NaN
    return p < 0 && length(v) < length(A) ? min(r, zero(r)) : r
end

"""
    opnorm(A::AbstractSparseMatrixCSC, p::Real=2)

Operator `p`-norm of the sparse matrix `A`, for `p` equal to `1`, `2` or `Inf`.

For `p = 2` the norm is an iterative estimate from Lanczos bidiagonalization rather than a
full singular value decomposition. It approaches the norm from below, to a relative accuracy
of about `1e-10`, and is computed in `Float64` arithmetic even when the element type is
wider. Each iteration costs a product with `A` and one with `A'`. Matrices whose largest
singular values are tightly clustered may need a number of iterations comparable to their
size; use `opnorm(Array(A))` when such a matrix is small enough.
"""
function opnorm(A::AbstractSparseMatrixCSC, p::Real=2)
    m, n = size(A)
    p in (1, 2, Inf) || throw(ArgumentError("invalid operator p-norm p=$p. Valid: 1, 2, Inf"))
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
                    colSum += abs(getnzval(A)[i])
                end
                nA = max(nA, colSum)
            end
            return convert(Tnorm, nA)
        elseif p==2
            return convert(Tnorm, opnorm2est(A))
        elseif p==Inf
            rowSum = zeros(Tsum,m)
            @inbounds for i in axes(getnzval(A),1)
                rowSum[getrowval(A)[i]] += abs(getnzval(A)[i])
            end
            return convert(Tnorm, maximum(rowSum))
        end
    end
    throw(ArgumentError("invalid operator p-norm p=$p. Valid: 1, 2, Inf"))
end

# Largest singular value by Golub-Kahan-Lanczos bidiagonalization, stopped once the residual
# of the leading Ritz pair stays below `tol` relative to it on two checks in a row. The
# estimate converges from below. With residual `r` and a gap `g` to the next singular value
# its error is of the order of `min(r, r^2/g)`, so singular values clustered more tightly
# than `tol` are not told apart. The start vector is seeded so that the result is
# reproducible; a fixed one such as `ones` may lie in the null space. The recurrence runs on
# `A/s`, with `s` the largest stored magnitude, which keeps its coefficients representable
# as `Float64` whatever the range of the element type.
function opnorm2est(A::AbstractSparseMatrixCSC, tol::Real=1e-10, maxiter::Integer=max(100, 2*minimum(size(A))))
    Tnorm = typeof(float(real(zero(eltype(A)))))
    s = convert(Tnorm, norm(nzvalview(A), Inf))
    (iszero(s) || !isfinite(s)) && return s
    T = promote_type(Float64, eltype(A))
    v = convert(Vector{T}, normalize!(randn(Xoshiro(0x2a), size(A, 2))))
    u = A * v
    α, β = [Float64(norm(u) / s)], Float64[]
    σ = α[1]
    passed = false
    for k in 1:maxiter
        if iszero(α[k])
            σ = first(_leading_ritz(α, β, k - 1))
            break
        end
        u ./= α[k] * s
        mul!(v, A', u, true, -α[k] * s)
        push!(β, norm(v) / s)
        # past the first steps the check runs only now and then, or when the recurrence is
        # about to break down
        if k <= 32 || k % 8 == 0 || k == maxiter || β[k] <= tol * σ
            σ, uk = _leading_ritz(α, β, k)
            converged = β[k] * uk <= tol * σ
            converged && (passed || iszero(β[k])) && break
            passed = converged
        end
        v ./= β[k] * s
        mul!(u, A, v, true, -β[k] * s)
        push!(α, norm(u) / s)
    end
    return s * convert(Tnorm, σ)
end

# Largest singular value of the upper bidiagonal matrix with diagonal `α[1:k]` and
# superdiagonal `β[1:k-1]`, and the magnitude of the last entry of its left singular vector.
# Both come from one eigenpair of the Golub-Kahan tridiagonal matrix, at O(k) cost, where a
# full SVD would cost O(k^2) memory.
function _leading_ritz(α::Vector{Float64}, β::Vector{Float64}, k::Integer)
    k == 0 && return 0.0, 1.0
    k == 1 && return α[1], 1.0
    ev = Vector{Float64}(undef, 2k - 1)
    for i in 1:k-1
        ev[2i-1] = α[i]
        ev[2i] = β[i]
    end
    ev[2k-1] = α[k]
    # one eigenpair needs no sorting; LinearAlgebra's `sorteig!` does not specialize on
    # `sortby`, so `juliac --trim` cannot resolve its sorting calls
    F = eigen(SymTridiagonal(zeros(2k), ev), 2k:2k; sortby = nothing)
    return F.values[1], sqrt(2) * abs(F.vectors[2k, 1])
end

# TODO rank

# cond
function cond(A::AbstractSparseMatrixCSC, p::Real=2)
    # as for a dense matrix; the estimator of the inverse's norm needs at least one column
    (p == 1 || p == Inf) && checksquare(A) == 0 && return zero(real(float(eltype(A))))
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

## kron
const _SparseKronGroup = Union{SparseVecOrMatMaybeAdjOrTrans, SparseMatrixCSCSubArray,
                               HermOrSym{<:Any,<:SparseVecOrMatMaybeAdjOrTrans},
                               UpperOrLowerTriangular{<:Any,<:SparseVecOrMatMaybeAdjOrTrans}}
const _DenseKronGroup = Union{Number, Vector, Matrix, AdjOrTrans{<:Any,<:VecOrMat}, BandedMatrix,
                              HermOrSym{<:Any,<:Matrix}, UpperOrLowerTriangular{<:Any,<:Matrix}}

@inline function kron!(C::SparseMatrixCSC, A::AbstractSparseMatrixCSC, B::AbstractSparseMatrixCSC)
    mA, nA = size(A); mB, nB = size(B)
    mC, nC = mA*mB, nA*nB
    @boundscheck size(C) == (mC, nC) || throw(DimensionMismatch("target matrix needs to have size ($mC, $nC)," *
        " but has size $(size(C))"))
    rowvalC = getrowval(C)
    nzvalC = getnzval(C)
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
                    rowvalC[ptr] = (getrowval(A)[ptrA]-1)*mB + getrowval(B)[ptrB]
                    nzvalC[ptr] = getnzval(A)[ptrA] * getnzval(B)[ptrB]
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
# `sparse` of the dense operand, as in `kron`: a dense vector has no conversion to a matrix
kron!(C::SparseMatrixCSC, A::_SparseKronGroup, B::_DenseKronGroup) = kron!(C, A, sparse(B))
kron!(C::SparseMatrixCSC, A::_DenseKronGroup, B::_SparseKronGroup) = kron!(C, sparse(A), B)
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
# a view times a number would be dense
kron(a::Number, b::SparseMatrixCSCSubArray) = a * convert(SparseMatrixCSC, b)
kron(a::SparseMatrixCSCSubArray, b::Number) = convert(SparseMatrixCSC, a) * b

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
    if copy_rows && getrowval(C) !== getrowval(A)
        resize!(getrowval(C), length(getrowval(A)))
        copyto!(getrowval(C), getrowval(A))
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
    rows_col = @view getrowval(A)[nzinds]
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
 ⋅  1.0   ⋅
 ⋅   ⋅   1.0
 ⋅   ⋅    ⋅

julia> C = spzeros(3,3);

julia> C[2:4:6] .= 2;

julia> C
3×3 SparseMatrixCSC{Float64, Int64} with 2 stored entries:
  ⋅    ⋅   ⋅
 2.0   ⋅   ⋅
  ⋅   2.0  ⋅

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
            row = @inbounds getrowval(A)[ind]
            row_exists, ind = rowcheck_index(C, row, col)
            if !row_exists
                _is_fixed(C) && throw(ArgumentError(lazy"cannot store entry ($row, $col) in a fixed sparse matrix whose pattern lacks it"))
                n_extra += 1
                insert!(getrowval(C), ind, row)
                insert!(getnzval(C), ind, zero(eltype(C)))
                C_colptr[col+1] += 1
            end
        end
        if !iszero(n_extra)
            @views C_colptr[col+2:end] .+= n_extra
        end
    end
    C
end

# For an integer eltype the Hermitian branches of `\` and `factorize` below would reach
# LinearAlgebra's generic `factorize(::HermOrSym)`, a dense Bunch-Kaufman in
# `Rational{BigInt}`, so they take the `lu` branch instead, which converts to floating
# point like dense `\` does. Every other eltype keeps its path: floating point goes to
# the sparse Cholesky/LDLt, and `Rational` stays exact through the generic factorization.
_hermitian_solve(A::AbstractSparseMatrixCSC) =
    !(eltype(A) <: Union{Integer, Complex{<:Integer}}) && ishermitian(A)

# `\` and `factorize` choose the same method from the structure of `A`; `f` (or `flu`, for
# the LU factorization) is applied in each branch so that the result stays inferrable. A
# Hermitian matrix is only wrapped, because `\` and `factorize` finish it differently.
function _sparse_factorize(f, A::AbstractSparseMatrixCSC, flu = f)
    m, n = size(A)
    if m == n
        if istril(A)
            return istriu(A) ? f(Diagonal(diag(A))) : f(LowerTriangular(A))
        elseif istriu(A)
            return f(UpperTriangular(A))
        elseif _hermitian_solve(A)
            return f(Hermitian(A))
        end
        return flu(lu(A))
    elseif m > n
        return f(qr(A))
    else
        # A is wide, so the LQ factorization gives the minimum-norm solution
        return f(lq(A))
    end
end

# the solution is dense, so a diagonal is densified for the solve
_solve_factor(F) = F
_solve_factor(D::Diagonal) = Diagonal(Vector(diag(D)))
_sparse_solve(op, F, B) = \(op(_solve_factor(F)), B)
# the sparse LU works in double precision, so its solution is converted to the eltype that
# dense `\` gives
_sparse_lusolve(op, F, A, B) =
    convert(AbstractArray{typeof(one(eltype(A)) \ one(eltype(B)))}, _sparse_solve(op, F, B))

function \(A::AbstractSparseMatrixCSC, B::AbstractVecOrMat)
    require_one_based_indexing(A, B)
    return _sparse_factorize(F -> _sparse_solve(identity, F, B), A,
                             F -> _sparse_lusolve(identity, F, A, B))
end
for (xformtype, xformop) in ((:Adjoint, :adjoint), (:Transpose, :transpose))
    @eval function \(xformA::($xformtype){<:Any,<:AbstractSparseMatrixCSC}, B::AbstractVecOrMat)
        A = parent(xformA)
        require_one_based_indexing(A, B)
        m, n = size(A)
        if m < n
            # A' is tall, so the least squares solve needs a factorization of A' itself
            return _sparse_solve(identity, qr($xformop(A)), B)
        end
        # Otherwise the transformed choice for A serves. For a wide A' that is the
        # factorization of A, which gives the minimum-norm solution.
        return _sparse_factorize(F -> _sparse_solve($xformop, F, B), A,
                                 F -> _sparse_lusolve($xformop, F, A, B))
    end
end

_factorize_choice(F) = F
_factorize_choice(H::Hermitian) = factorize(H)
factorize(A::AbstractSparseMatrixCSC) = _sparse_factorize(_factorize_choice, A)

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
