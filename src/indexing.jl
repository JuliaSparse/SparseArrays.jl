# This file is a part of Julia. License is MIT: https://julialang.org/license

# `getindex`, `setindex!` and `dropstored!` for `SparseMatrixCSC`, including the slices and
# linear and logical indexing that return a `SparseVector`.

## getindex
function rangesearch(haystack::AbstractRange, needle)
    (i,rem) = divrem(needle - first(haystack), step(haystack))
    (rem==0 && 1<=i+1<=length(haystack)) ? i+1 : 0
end

@RCI @propagate_inbounds getindex(A::AbstractSparseMatrixCSC, I::Tuple{Integer,Integer}) = getindex(A, I[1], I[2])

@RCI @propagate_inbounds function getindex(A::AbstractSparseMatrixCSC{T}, i0::Integer, i1::Integer) where T
    @boundscheck checkbounds(A, i0, i1)
    r1 = Int(@inbounds first(nzrange(A, i1)))
    r2 = Int(@inbounds last(nzrange(A, i1)))
    (r1 > r2) && return zero(T)
    r1 = searchsortedfirst(view(rowvals(A), r1:r2), i0) + r1 - 1
    ((r1 > r2) || (rowvals(A)[r1] != i0)) ? zero(T) : nonzeros(A)[r1]
end

# Colon translation
# Lower indices as Base does. Bounds are checked first because a mask lowers to a
# `Base.LogicalIndex`, which `ensure_indexable` collects, dropping the mask's length.
@inline function _lower_indices(A, I...)
    L = to_indices(A, I)
    @boundscheck checkbounds(A, L...)
    return Base.ensure_indexable(L)
end

# Index types no method below matches (`CartesianIndex`, custom indices) are lowered and
# dispatched again; indices that are already lowered fall back to Base.
@propagate_inbounds getindex(A::AbstractSparseMatrixCSC{Tv,Ti}, I...) where {Tv,Ti} =
    _getindex_lowered(A, I, _lower_indices(A, I...))
@propagate_inbounds getindex(x::AbstractSparseVector{Tv,Ti}, I...) where {Tv,Ti} =
    _getindex_lowered(x, I, _lower_indices(x, I...))
@propagate_inbounds _getindex_lowered(A, I, L) = A[L...]
@propagate_inbounds _getindex_lowered(A, I::T, ::T) where {T} =
    invoke(getindex, Tuple{AbstractArray,Vararg{Any}}, A, I...)

getindex(A::AbstractSparseMatrixCSC, ::Colon, ::Colon) = copy(A)
getindex(A::AbstractSparseMatrixCSC, i, ::Colon)       = getindex(A, i, axes(A,2))
getindex(A::AbstractSparseMatrixCSC, ::Colon, i)       = getindex(A, axes(A,1), i)

function getindex_cols(A::AbstractSparseMatrixCSC{Tv,Ti}, J::AbstractVector) where {Tv,Ti}
    require_one_based_indexing(A, J)
    # for indexing whole columns
    (m, n) = size(A)
    nJ = length(J)

    colptrA = getcolptr(A); rowvalA = rowvals(A); nzvalA = nonzeros(A)

    colptrS = Vector{Ti}(undef, nJ+1)
    colptrS[1] = 1
    nnzS = 0

    @inbounds for j = 1:nJ
        col = J[j]
        1 <= col <= n || throw(BoundsError())
        nnzS += colptrA[col+1] - colptrA[col]
        colptrS[j+1] = nnzS + 1
    end

    rowvalS = Vector{Ti}(undef, nnzS)
    nzvalS  = Vector{Tv}(undef, nnzS)
    ptrS = 0

    @inbounds for j = 1:nJ
        col = J[j]
        for k = nzrange(A, col)
            ptrS += 1
            rowvalS[ptrS] = rowvalA[k]
            nzvalS[ptrS] = nzvalA[k]
        end
    end
    return @if_move_fixed A SparseMatrixCSC(m, nJ, colptrS, rowvalS, nzvalS)
end

getindex_traverse_col(::AbstractUnitRange, lo::Integer, hi::Integer) = lo:hi
getindex_traverse_col(I::StepRange, lo::Integer, hi::Integer) = step(I) > 0 ? (lo:1:hi) : (hi:-1:lo)

function getindex(A::AbstractSparseMatrixCSC{Tv,Ti}, I::AbstractRange, J::AbstractVector) where {Tv,Ti<:Integer}
    require_one_based_indexing(A, I, J)
    I, J = _lower_indices(A, I, J)
    # Ranges for indexing rows
    (m, n) = size(A)
    # whole columns:
    if I == 1:m
        return getindex_cols(A, J)
    end

    nI = length(I)
    nI == 0 || (minimum(I) >= 1 && maximum(I) <= m) || throw(BoundsError())
    nJ = length(J)
    rowvalA = rowvals(A); nzvalA = nonzeros(A)
    colptrS = Vector{Ti}(undef, nJ+1)
    colptrS[1] = 1
    nnzS = 0

    # Form the structure of the result and compute space
    @inbounds for j = 1:nJ
        col = J[j]
        1 <= col <= n || throw(BoundsError())
        @simd for k in nzrange(A, col)
            nnzS += rowvalA[k] in I # `in` is fast for ranges
        end
        colptrS[j+1] = nnzS+1
    end

    # Populate the values in the result
    rowvalS = Vector{Ti}(undef, nnzS)
    nzvalS  = Vector{Tv}(undef, nnzS)
    ptrS    = 1

    @inbounds for j = 1:nJ
        col = J[j]
        for k = getindex_traverse_col(I, first(nzrange(A, col)), last(nzrange(A, col)))
            rowA = rowvalA[k]
            i = rangesearch(I, rowA)
            if i > 0
                rowvalS[ptrS] = i
                nzvalS[ptrS] = nzvalA[k]
                ptrS += 1
            end
        end
    end

    return @if_move_fixed A SparseMatrixCSC(nI, nJ, colptrS, rowvalS, nzvalS)
end

function getindex_I_sorted(A::AbstractSparseMatrixCSC{Tv,Ti}, I::AbstractVector, J::AbstractVector) where {Tv,Ti}
    require_one_based_indexing(A, I, J)
    # Sorted vectors for indexing rows.
    # Similar to getindex_general but without the transpose trick.
    (m, n) = size(A)

    nI   = length(I)
    nzA  = nnz(A)
    avgM = div(nzA,n)
    # Heuristics based on experiments discussed in:
    # https://github.com/JuliaLang/julia/issues/12860
    # https://github.com/JuliaLang/julia/pull/12934
    alg = ((m > nzA) && (m > nI)) ? 0 :
          ((nI - avgM) > 2^8) ? 1 :
          ((avgM - nI) > 2^10) ? 0 : 2

    nzJ = sum(j -> length(nzrange(A, j)), J)
    avgJ = nzJ ÷ length(J)
    # `bsearch_A` walks `I` for every column, which only suits an `I` much shorter than the
    # selected columns.
    if alg == 0
        (nI > 16 + avgJ ÷ 16) && return getindex_I_sorted_nocache(A, I, J)
        return getindex_I_sorted_bsearch_A(A, I, J)
    end
    # The other two kernels set up a cache of length `m`, which only pays off once the
    # searches it saves are comparable to `m`. The factor is where the cache breaks even
    # in its best case, every selected column storing the same rows.
    gap = nI ÷ max(avgJ, 1)
    (32 * nzJ * ndigits(gap + 1, base=2) < m) && return getindex_I_sorted_nocache(A, I, J)
    (alg == 1) ? getindex_I_sorted_bsearch_I(A, I, J) :
    return getindex_I_sorted_linear(A, I, J)
end

function getindex_I_sorted_bsearch_A(A::AbstractSparseMatrixCSC{Tv,Ti}, I::AbstractVector, J::AbstractVector) where {Tv,Ti}
    require_one_based_indexing(A, I, J)
    nI = length(I)
    nJ = length(J)

    rowvalA = rowvals(A); nzvalA = nonzeros(A)
    colptrS = Vector{Ti}(undef, nJ+1)
    colptrS[1] = 1

    ptrS = 1
    # determine result size
    @inbounds for j = 1:nJ
        col = J[j]
        ptrI::Int = 1 # runs through I
        ptrA::Int = first(nzrange(A, col))
        stopA::Int = last(nzrange(A, col))
        if ptrA <= stopA
            while ptrI <= nI
                rowI = I[ptrI]
                ptrI += 1
                (rowvalA[ptrA] > rowI) && continue
                ptrA += searchsortedfirst(view(rowvalA, ptrA:stopA), rowI) - 1
                (ptrA <= stopA) || break
                if rowvalA[ptrA] == rowI
                    ptrS += 1
                end
            end
        end
        colptrS[j+1] = ptrS
    end

    rowvalS = Vector{Ti}(undef, ptrS-1)
    nzvalS  = Vector{Tv}(undef, ptrS-1)

    # fill the values
    ptrS = 1
    @inbounds for j = 1:nJ
        col = J[j]
        ptrI::Int = 1 # runs through I
        ptrA::Int = first(nzrange(A, col))
        stopA::Int = last(nzrange(A, col))
        if ptrA <= stopA
            while ptrI <= nI
                rowI = I[ptrI]
                if rowvalA[ptrA] <= rowI
                    ptrA += searchsortedfirst(view(rowvalA, ptrA:stopA), rowI) - 1
                    (ptrA <= stopA) || break
                    if rowvalA[ptrA] == rowI
                        rowvalS[ptrS] = ptrI
                        nzvalS[ptrS] = nzvalA[ptrA]
                        ptrS += 1
                    end
                end
                ptrI += 1
            end
        end
    end
    return @if_move_fixed A SparseMatrixCSC(nI, nJ, colptrS, rowvalS, nzvalS)
end

function getindex_I_sorted_linear(A::AbstractSparseMatrixCSC{Tv,Ti}, I::AbstractVector, J::AbstractVector) where {Tv,Ti}
    require_one_based_indexing(A, I, J)
    nI = length(I)
    nJ = length(J)

    colptrA = getcolptr(A); rowvalA = rowvals(A); nzvalA = nonzeros(A)
    colptrS = Vector{Ti}(undef, nJ+1)
    colptrS[1] = 1
    cacheI = zeros(Int, size(A, 1))

    ptrS   = 1
    # build the cache and determine result size
    @inbounds for j = 1:nJ
        col = J[j]
        ptrI::Int = 1 # runs through I
        ptrA::Int = colptrA[col]
        stopA::Int = colptrA[col+1]
        while ptrI <= nI && ptrA < stopA
            rowA = rowvalA[ptrA]
            rowI = I[ptrI]

            if rowI > rowA
                ptrA += 1
            elseif rowI < rowA
                ptrI += 1
            else
                (cacheI[rowA] == 0) && (cacheI[rowA] = ptrI)
                ptrS += 1
                ptrI += 1
            end
        end
        colptrS[j+1] = ptrS
    end

    rowvalS = Vector{Ti}(undef, ptrS-1)
    nzvalS  = Vector{Tv}(undef, ptrS-1)

    # fill the values
    ptrS = 1
    @inbounds for j = 1:nJ
        col = J[j]
        ptrA::Int = colptrA[col]
        stopA::Int = colptrA[col+1]
        while ptrA < stopA
            rowA = rowvalA[ptrA]
            ptrI = cacheI[rowA]
            if ptrI > 0
                while ptrI <= nI && I[ptrI] == rowA
                    rowvalS[ptrS] = ptrI
                    nzvalS[ptrS] = nzvalA[ptrA]
                    ptrS += 1
                    ptrI += 1
                end
            end
            ptrA += 1
        end
    end
    return @if_move_fixed A SparseMatrixCSC(nI, nJ, colptrS, rowvalS, nzvalS)
end

function getindex_I_sorted_bsearch_I(A::AbstractSparseMatrixCSC{Tv,Ti}, I::AbstractVector, J::AbstractVector) where {Tv,Ti}
    require_one_based_indexing(A, I, J)
    nI = length(I)
    nJ = length(J)

    colptrA = getcolptr(A); rowvalA = rowvals(A); nzvalA = nonzeros(A)
    colptrS = Vector{Ti}(undef, nJ+1)
    colptrS[1] = 1

    m = size(A, 1)

    # cacheI is used first to store num occurrences of each row in columns of interest
    # and later to store position of first occurrence of each row in I
    cacheI = zeros(Int, m)

    # count rows
    @inbounds for j = 1:nJ
        col = J[j]
        for ptrA in nzrange(A, col)
            cacheI[rowvalA[ptrA]] += 1
        end
    end

    # fill cache and count nnz
    ptrS::Int = 0
    ptrI::Int = 1
    @inbounds for j = 1:m
        cval = cacheI[j]
        (cval == 0) && continue
        ptrI += searchsortedfirst(view(I, ptrI:nI), j) - 1
        cacheI[j] = ptrI
        while ptrI <= nI && I[ptrI] == j
            ptrS += cval
            ptrI += 1
        end
        if ptrI > nI
            @simd for i=(j+1):m; @inbounds cacheI[i]=ptrI; end
            break
        end
    end
    rowvalS = Vector{Ti}(undef, ptrS)
    nzvalS  = Vector{Tv}(undef, ptrS)
    colptrS[nJ+1] = ptrS+1

    # fill the values
    ptrS = 1
    @inbounds for j = 1:nJ
        col = J[j]
        ptrA::Int = colptrA[col]
        stopA::Int = colptrA[col+1]
        while ptrA < stopA
            rowA = rowvalA[ptrA]
            ptrI = cacheI[rowA]
            (ptrI > nI) && break
            if ptrI > 0
                while I[ptrI] == rowA
                    rowvalS[ptrS] = ptrI
                    nzvalS[ptrS] = nzvalA[ptrA]
                    ptrS += 1
                    ptrI += 1
                    (ptrI > nI) && break
                end
            end
            ptrA += 1
        end
        colptrS[j+1] = ptrS
    end
    return @if_move_fixed A SparseMatrixCSC(nI, nJ, colptrS, rowvalS, nzvalS)
end

# First position at or after `ptrI` in the sorted `I` whose value is not less than `row`.
# Gallops, so the cost grows with the log of the distance moved rather than of `length(I)`.
@inline function _advance_I(I::AbstractVector, ptrI::Int, nI::Int, row)
    hi = ptrI
    step = 1
    @inbounds while hi <= nI && I[hi] < row
        ptrI = hi + 1
        hi += step
        step <<= 1
    end
    hi = min(hi, nI)
    (hi - ptrI > 8) && return ptrI + searchsortedfirst(view(I, ptrI:hi), row) - 1
    @inbounds while ptrI <= hi && I[ptrI] < row
        ptrI += 1
    end
    return ptrI
end

@inline function _nzrange_from(A, rowvalA, col, minrow)
    r = nzrange(A, col)
    return (first(r) + searchsortedfirst(view(rowvalA, r), minrow) - 1):last(r)
end

# Same results as `getindex_I_sorted_bsearch_I` and `getindex_I_sorted_linear` without their
# cache of length `size(A, 1)`, so the cost depends only on `I` and the selected columns.
function getindex_I_sorted_nocache(A::AbstractSparseMatrixCSC{Tv,Ti}, I::AbstractVector, J::AbstractVector) where {Tv,Ti}
    require_one_based_indexing(A, I, J)
    nI = length(I)
    nJ = length(J)

    rowvalA = rowvals(A); nzvalA = nonzeros(A)
    colptrS = Vector{Ti}(undef, nJ+1)
    colptrS[1] = 1
    # stored entries above the first requested row are skipped in one search per column
    minI = nI == 0 ? size(A, 1) + 1 : I[1]

    ptrS = 1
    # determine result size
    @inbounds for j = 1:nJ
        ptrI = 1
        for ptrA in _nzrange_from(A, rowvalA, J[j], minI)
            rowA = rowvalA[ptrA]
            ptrI = _advance_I(I, ptrI, nI, rowA)
            while ptrI <= nI && I[ptrI] == rowA
                ptrS += 1
                ptrI += 1
            end
            (ptrI > nI) && break
        end
        colptrS[j+1] = ptrS
    end

    rowvalS = Vector{Ti}(undef, ptrS-1)
    nzvalS  = Vector{Tv}(undef, ptrS-1)

    # fill the values
    ptrS = 1
    @inbounds for j = 1:nJ
        ptrI = 1
        for ptrA in _nzrange_from(A, rowvalA, J[j], minI)
            rowA = rowvalA[ptrA]
            ptrI = _advance_I(I, ptrI, nI, rowA)
            while ptrI <= nI && I[ptrI] == rowA
                rowvalS[ptrS] = ptrI
                nzvalS[ptrS] = nzvalA[ptrA]
                ptrS += 1
                ptrI += 1
            end
            (ptrI > nI) && break
        end
    end
    return @if_move_fixed A SparseMatrixCSC(nI, nJ, colptrS, rowvalS, nzvalS)
end

function permute_rows!(S::AbstractSparseMatrixCSC{Tv,Ti}, pI::Vector{Int}) where {Tv,Ti}
    (m, n) = size(S)
    colptrS = getcolptr(S); rowvalS = rowvals(S); nzvalS = nonzeros(S)
    # preallocate temporary sort space
    nr = min(nnz(S), m)

    rowperm = Vector{Int}(undef, nr)
    rowval_temp = Vector{Ti}(undef, nr)
    rnzval_temp = Vector{Tv}(undef, nr)
    perm = Base.Perm(Base.ord(isless, identity, false, Base.Order.Forward), rowval_temp)

    @inbounds for j in axes(S,2)
        rowrange = nzrange(S, j)
        nr = length(rowrange)
        resize!(rowperm, nr)
        resize!(rowval_temp, nr)
        (nr > 0) || continue
        k = 1
        for i in rowrange
            rowA = rowvalS[i]
            rowval_temp[k] = pI[rowA]
            rnzval_temp[k] = nzvalS[i]
            k += 1
        end

        if nr <= 16
            alg = Base.Sort.InsertionSort
        else
            alg = Base.Sort.QuickSort
        end

        # Reset permutation
        rowperm .= 1:nr
        sort!(rowperm, alg, perm)

        k = 1
        for i in rowrange
            kperm = rowperm[k]
            rowvalS[i] = rowval_temp[kperm]
            nzvalS[i] = rnzval_temp[kperm]
            k += 1
        end
    end
    return _checkbuffers(S)
end

function getindex_general(A::AbstractSparseMatrixCSC, I::AbstractVector, J::AbstractVector)
    require_one_based_indexing(A, I, J)
    pI = sortperm(I)
    @inbounds Is = I[pI]
    return permute_rows!(getindex_I_sorted(A, Is, J), pI)
end

# the general case:
function getindex(A::AbstractSparseMatrixCSC{Tv,Ti}, I::AbstractVector, J::AbstractVector) where {Tv,Ti}
    require_one_based_indexing(A, I, J)
    I, J = _lower_indices(A, I, J)

    if isempty(I) || isempty(J) || (0 == nnz(A))
        return spzeros(Tv, Ti, length(I), length(J))
    end

    if issorted(I)
        return getindex_I_sorted(A, I, J)
    else
        return getindex_general(A, I, J)
    end
end

function getindex(A::AbstractSparseMatrixCSC{Tv,Ti}, I::AbstractArray) where {Tv,Ti}
    require_one_based_indexing(A, I)
    szA = size(A)
    nA = szA[1]*szA[2]
    rowvalA = rowvals(A)
    nzvalA = nonzeros(A)

    n = length(I)
    outm = size(I,1)
    outn = size(I,2)
    szB = (outm, outn)
    colptrB = zeros(Ti, outn+1)
    rowvalB = Vector{Ti}(undef, n)
    nzvalB = Vector{Tv}(undef, n)

    colB = 1
    rowB = 1
    colptrB[colB] = 1
    idxB = 1

    CartIndsA = CartesianIndices(szA)
    CartIndsB = CartesianIndices(szB)

    for i in 1:n
        @boundscheck checkbounds(A, I[i])
        row,col = Tuple(CartIndsA[I[i]])
        for r in nzrange(A, col)
            @inbounds if rowvalA[r] == row
                rowB,colB = Tuple(CartIndsB[i])
                colptrB[colB+1] += 1
                rowvalB[idxB] = rowB
                nzvalB[idxB] = nzvalA[r]
                idxB += 1
                break
            end
        end
    end
    cumsum!(colptrB,colptrB)
    if n > (idxB-1)
        deleteat!(nzvalB, idxB:n)
        deleteat!(rowvalB, idxB:n)
    end
    @if_move_fixed A SparseMatrixCSC(outm, outn, colptrB, rowvalB, nzvalB)
end

# a range of `Bool` is a mask, not a range of rows
getindex(A::AbstractSparseMatrixCSC{<:Any,<:Integer}, I::AbstractRange{Bool}, J::AbstractVector) = A[collect(I), J]

## setindex!

# dispatch helper for #29034
@RCI setindex!(A::AbstractSparseMatrixCSC, _v, _i::Integer, _j::Integer) = _setindex_scalar!(A, _v, _i, _j)

function _setindex_scalar!(A::AbstractSparseMatrixCSC{Tv,Ti}, _v, _i::Integer, _j::Integer) where {Tv,Ti<:Integer}
    v = convert(Tv, _v)
    i = convert(Ti, _i)
    j = convert(Ti, _j)
    if !((1 <= i <= size(A, 1)) & (1 <= j <= size(A, 2)))
        throw(BoundsError(A, (i,j)))
    end
    coljfirstk = Int(getcolptr(A)[j])
    coljlastk = Int(getcolptr(A)[j+1] - 1)
    searchk = searchsortedfirst(view(rowvals(A), coljfirstk:coljlastk), i) + coljfirstk - 1
    if searchk <= coljlastk && rowvals(A)[searchk] == i
        # Column j contains entry A[i,j]. Update and return
        nonzeros(A)[searchk] = v
        return A
    end
    # Column j does not contain entry A[i,j].
    if !_isimplicitzero(v, Tv)
        nz = getcolptr(A)[size(A, 2)+1]
        # throw exception before state is partially modified
        !isbitstype(Ti) || nz < typemax(Ti) ||
            throw(ArgumentError("nnz(A) going to exceed typemax(Ti) = $(typemax(Ti))"))

        _is_fixed(A) && _throwfixedinsert(A, i, j)
        # if nnz(A) < length(rowval/nzval): no need to grow rowval and preserve values
        _insert!(rowvals(A), searchk, i, nz)
        _insert!(nonzeros(A), searchk, v, nz)
        @simd for m in (j + 1):(size(A, 2) + 1)
            @inbounds getcolptr(A)[m] += Ti(1)
        end
    end
    return A
end

# insert item at position pos, shifting only from pos+1 to nz
function _insert!(v::Vector, pos::Integer, item, nz::Integer)
    if nz > length(v)
        insert!(v, pos, item)
    else # nz < length(v)
        Base.unsafe_copyto!(v, pos+1, v, pos, nz - pos)
        v[pos] = item
        v
    end
end

function Base.fill!(V::SubArray{Tv, <:Any, <:AbstractSparseMatrixCSC{Tv}, <:Tuple{Vararg{Union{Integer, AbstractVector{<:Integer}},2}}}, x) where Tv
    A = parent(V)
    I, J = V.indices
    if isempty(I) || isempty(J); return A; end
    if _is_fixed(A)   # the scalar path keeps the pattern and throws outside it
        for j in J, i in I
            A[i, j] = x
        end
        return V
    end
    # lt=≤ to check for strict sorting
    if !issorted(I, lt=≤); I = sort!(unique(I)); end
    if !issorted(J, lt=≤); J = sort!(unique(J)); end
    if (I[1] < 1 || I[end] > size(A, 1)) || (J[1] < 1 || J[end] > size(A, 2))
        throw(BoundsError(A, (I, J)))
    end
    x = convert(Tv, x)
    if _isimplicitzero(x, Tv)
        _spsetz_setindex!(A, I, J)
    else
        _spsetnz_setindex!(A, x, I, J)
    end
    _checkbuffers(A)
    V
end
"""
Helper method for immediately preceding fill! method. For all (i,j) such that i in I and
j in J, assigns zero to A[i,j] if A[i,j] is a presently-stored entry, and otherwise does nothing.
"""
function _spsetz_setindex!(A::AbstractSparseMatrixCSC,
        I::Union{Integer, AbstractVector{<:Integer}}, J::Union{Integer, AbstractVector{<:Integer}})
    require_one_based_indexing(A, I, J)
    lengthI = length(I)
    for j in J
        coljAfirstk = getcolptr(A)[j]
        coljAlastk = getcolptr(A)[j+1] - 1
        coljAfirstk > coljAlastk && continue
        kA = coljAfirstk
        kI = 1
        entrykArow = rowvals(A)[kA]
        entrykIrow = I[kI]
        while true
            if entrykArow < entrykIrow
                kA += 1
                kA > coljAlastk && break
                entrykArow = rowvals(A)[kA]
            elseif entrykArow > entrykIrow
                kI += 1
                kI > lengthI && break
                entrykIrow = I[kI]
            else # entrykArow == entrykIrow
                nonzeros(A)[kA] = zero(eltype(A))
                kA += 1
                kI += 1
                (kA > coljAlastk || kI > lengthI) && break
                entrykArow = rowvals(A)[kA]
                entrykIrow = I[kI]
            end
        end
    end
end
"""
Helper method for immediately preceding fill! method. For all (i,j) such that i in I
and j in J, assigns x to A[i,j] if A[i,j] is a presently-stored entry, and allocates and
assigns x to A[i,j] if A[i,j] is not presently stored.
"""
function _spsetnz_setindex!(A::AbstractSparseMatrixCSC{Tv}, x::Tv,
        I::Union{Integer, AbstractVector{<:Integer}}, J::Union{Integer, AbstractVector{<:Integer}}) where Tv
    require_one_based_indexing(A, I, J)
    m, n = size(A)
    lenI = length(I)

    nnzold = nnz(A)
    nnzA = nnzold + lenI * length(J)

    rowvalA = rowvals(A)
    nzvalA = nonzeros(A)

    rowidx = 1
    nadd = 0
    shift = 0   # the unread old entries sit `shift` places further along once the buffers grow
    @inbounds for col in axes(A,2)
        rrange = nzrange(A, col)
        if nadd > 0
            getcolptr(A)[col] = getcolptr(A)[col] + nadd
        end

        if col in J
            if isempty(rrange) # set new vals only
                nincl = lenI
                if nadd == 0
                    shift = _spsetnz_makeroom!(rowvalA, nzvalA, first(rrange), nnzold, nnzA)
                end
                r = rowidx:(rowidx+nincl-1)
                rowvalA[r] .= I
                for rr in r
                    nzvalA[rr] = x
                end
                rowidx += nincl
                nadd += nincl
            else # set old + new vals
                old_ptr = rrange[1]
                old_stop = rrange[end]
                new_ptr = 1
                new_stop = lenI

                while true
                    old_row = rowvalA[old_ptr+shift]
                    new_row = I[new_ptr]
                    if old_row < new_row
                        rowvalA[rowidx] = old_row
                        nzvalA[rowidx] = nzvalA[old_ptr+shift]
                        rowidx += 1
                        old_ptr += 1
                    else
                        if old_row == new_row
                            old_ptr += 1
                        else
                            if nadd == 0
                                shift = _spsetnz_makeroom!(rowvalA, nzvalA, old_ptr, nnzold, nnzA)
                            end
                            nadd += 1
                        end
                        rowvalA[rowidx] = new_row
                        nzvalA[rowidx] = x
                        rowidx += 1
                        new_ptr += 1
                    end

                    if old_ptr > old_stop
                        if new_ptr <= new_stop
                            if nadd == 0
                                shift = _spsetnz_makeroom!(rowvalA, nzvalA, old_ptr, nnzold, nnzA)
                            end
                            r = rowidx:(rowidx+(new_stop-new_ptr))
                            rowvalA[r] .= I isa Number ? I : I[new_ptr:new_stop]
                            for rr in r
                                nzvalA[rr] = x
                            end
                            rowidx += length(r)
                            nadd += length(r)
                        end
                        break
                    end

                    if new_ptr > new_stop
                        nincl = old_stop-old_ptr+1
                        copyto!(rowvalA, rowidx, rowvalA, old_ptr+shift, nincl)
                        copyto!(nzvalA, rowidx, nzvalA, old_ptr+shift, nincl)
                        rowidx += nincl
                        break
                    end
                end
            end
        elseif !isempty(rrange) # set old vals only
            nincl = length(rrange)
            if nadd > 0
                copyto!(rowvalA, rowidx, rowvalA, rrange[1]+shift, nincl)
                copyto!(nzvalA, rowidx, nzvalA, rrange[1]+shift, nincl)
            end
            rowidx += nincl
        end
    end

    if nadd > 0
        getcolptr(A)[n+1] = rowidx
        deleteat!(rowvalA, rowidx:nnzA)
        deleteat!(nzvalA, rowidx:nnzA)
    end
    return A
end

# Grow the buffers to `nnzA`, the most entries the assignment can produce, and move the old
# entries not yet merged, `from:nnzold`, to the end. The merge writes at most `shift` places
# ahead of where it reads, so the moved entries are never overwritten before they are read.
function _spsetnz_makeroom!(rowvalA, nzvalA, from::Integer, nnzold::Integer, nnzA::Integer)
    resize!(rowvalA, nnzA)
    resize!(nzvalA, nnzA)
    shift = nnzA - nnzold
    nmove = nnzold - from + 1
    copyto!(rowvalA, from+shift, rowvalA, from, nmove)
    copyto!(nzvalA, from+shift, nzvalA, from, nmove)
    return shift
end

# Nonscalar A[I,J] = B: Convert B to a SparseMatrixCSC of the appropriate shape first
# (reshape also fixes a 1×n V assigned to A[:, j], which the shape check allows; see #569)
# a dense `V` keeps what scalar `setindex!` would store, such as `-0.0`, which `sparse` drops
function _to_same_csc(::AbstractSparseMatrixCSC{Tv, Ti}, V::AbstractVecOrMat, I, J) where {Tv,Ti}
    M = reshape(V, length(I), length(J))
    nz = count(x -> !_isimplicitzero(convert(Tv, x), Tv), M)
    colptr = Vector{Ti}(undef, size(M, 2) + 1)
    rowval = Vector{Ti}(undef, nz)
    nzval = Vector{Tv}(undef, nz)
    colptr[1] = 1
    k = 1
    for j in axes(M, 2)
        for i in axes(M, 1)
            v = convert(Tv, M[i, j])
            if !_isimplicitzero(v, Tv)
                rowval[k] = i
                nzval[k] = v
                k += 1
            end
        end
        colptr[j+1] = k
    end
    return SparseMatrixCSC{Tv,Ti}(size(M)..., colptr, rowval, nzval)
end
# a sparse `V` is copied through its storage; converting the lazy reshape would visit every element
_to_same_csc(::AbstractSparseMatrixCSC{Tv, Ti}, V::AbstractSparseMatrixCSC, I, J) where {Tv,Ti} =
    SparseMatrixCSC{Tv,Ti}(size(V) == (length(I), length(J)) ? V : copy(reshape(V, length(I), length(J))))

# The positions of `I` that a sorted merge has to visit, in increasing order of index:
# of a repeated index only the last position, whose write wins. `nothing` if `I` is
# already strictly increasing.
function _setindex_lastwrites(I)
    issorted(I, lt=≤) && return nothing
    # a decreasing range has no repeats, and indexing by its range permutation is cheaper
    I isa AbstractRange && !iszero(step(I)) && return sortperm(I)
    p = issorted(I) ? collect(eachindex(I)) : sortperm(I)   # stable, so repeats keep their order
    n = length(p)
    k = 0
    @inbounds for t in 1:n
        (t < n && I[p[t]] == I[p[t+1]]) && continue
        p[k += 1] = p[t]
    end
    return resize!(p, k)
end

setindex!(A::AbstractSparseMatrixCSC{Tv}, B::AbstractVecOrMat, I::Integer, J::Integer) where {Tv} = _setindex_scalar!(A, B, I, J)

function setindex!(A::AbstractSparseMatrixCSC{Tv,Ti}, V::AbstractVecOrMat, Ix::Union{Integer, AbstractVector{<:Integer}, Colon}, Jx::Union{Integer, AbstractVector{<:Integer}, Colon}) where {Tv,Ti<:Integer}
    require_one_based_indexing(A, V, Ix, Jx)
    (I, J) = Base.ensure_indexable(to_indices(A, (Ix, Jx)))
    checkbounds(A, I, J)
    nJ = length(J)
    Base.setindex_shape_check(V, length(I), nJ)
    if _is_fixed(A)   # the scalar path keeps the pattern and throws outside it
        k = 0
        for j in J, i in I
            A[i, j] = V[k += 1]
        end
        return A
    end
    B = _to_same_csc(A, V, I, J)

    m, n = size(A)
    if (!isempty(I) && (I[1] < 1 || I[end] > m)) || (!isempty(J) && (J[1] < 1 || J[end] > n))
        throw(BoundsError(A, (I, J)))
    end
    if isempty(I) || isempty(J)
        return A
    end

    pI = _setindex_lastwrites(I)
    pJ = _setindex_lastwrites(J)
    if pI !== nothing && pJ !== nothing
        I = I[pI]; J = J[pJ]
        B = B[pI, pJ]
    elseif pI !== nothing
        I = I[pI]
        B = B[pI, :]
    elseif pJ !== nothing
        J = J[pJ]
        B = B[:, pJ]
    end
    nJ = length(J)

    colptrA = getcolptr(A); rowvalA = rowvals(A); nzvalA = nonzeros(A)
    colptrB = getcolptr(B); rowvalB = rowvals(B); nzvalB = nonzeros(B)

    nnzS = nnz(A) + nnz(B)

    colptrS = copy(getcolptr(A))
    rowvalS = copy(rowvals(A))
    nzvalS = copy(nonzeros(A))

    resize!(rowvalA, nnzS)
    resize!(nzvalA, nnzS)

    colB = 1

    I_asgn = falses(m)
    fill!(view(I_asgn, I), true)

    ptrS = 1

    @inbounds for col in axes(A,2)

        # Copy column of A if it is not being assigned into
        if colB > nJ || col != J[colB]
            colptrA[col+1] = colptrA[col] + (colptrS[col+1]-colptrS[col])

            for k = colptrS[col]:colptrS[col+1]-1
                rowvalA[ptrS] = rowvalS[k]
                nzvalA[ptrS] = nzvalS[k]
                ptrS += 1
            end
            continue
        end

        ptrA::Int  = colptrS[col]
        stopA::Int = colptrS[col+1]
        ptrB::Int  = colptrB[colB]
        stopB::Int = colptrB[colB+1]

        while ptrA < stopA && ptrB < stopB
            rowA = rowvalS[ptrA]
            rowB = I[rowvalB[ptrB]]
            if rowA < rowB
                rowvalA[ptrS] = rowA
                nzvalA[ptrS] = I_asgn[rowA] ? zero(Tv) : nzvalS[ptrA]
                ptrS += 1
                ptrA += 1
            elseif rowB < rowA
                if !_isimplicitzero(nzvalB[ptrB], Tv)
                    rowvalA[ptrS] = rowB
                    nzvalA[ptrS] = nzvalB[ptrB]
                    ptrS += 1
                end
                ptrB += 1
            else
                rowvalA[ptrS] = rowB
                nzvalA[ptrS] = nzvalB[ptrB]
                ptrS += 1
                ptrB += 1
                ptrA += 1
            end
        end

        while ptrA < stopA
            rowA = rowvalS[ptrA]
            rowvalA[ptrS] = rowA
            nzvalA[ptrS] = I_asgn[rowA] ? zero(Tv) : nzvalS[ptrA]
            ptrS += 1
            ptrA += 1
        end

        while ptrB < stopB
            rowB = I[rowvalB[ptrB]]
            if !_isimplicitzero(nzvalB[ptrB], Tv)
                rowvalA[ptrS] = rowB
                nzvalA[ptrS] = nzvalB[ptrB]
                ptrS += 1
            end
            ptrB += 1
        end

        colptrA[col+1] = ptrS
        colB += 1
    end

    deleteat!(rowvalA, colptrA[end]:length(rowvalA))
    deleteat!(nzvalA, colptrA[end]:length(nzvalA))

    return _checkbuffers(A)
end

# Logical setindex!

setindex!(A::Matrix, x::AbstractSparseMatrixCSC, I::Integer, J::AbstractVector{Bool}) = setindex!(A, Array(x), I, findall(J))
setindex!(A::Matrix, x::AbstractSparseMatrixCSC, I::AbstractVector{Bool}, J::Integer) = setindex!(A, Array(x), findall(I), J)
setindex!(A::Matrix, x::AbstractSparseMatrixCSC, I::AbstractVector{Bool}, J::AbstractVector{Bool}) = setindex!(A, Array(x), findall(I), findall(J))
setindex!(A::Matrix, x::AbstractSparseMatrixCSC, I::AbstractVector{<:Integer}, J::AbstractVector{Bool}) = setindex!(A, Array(x), I, findall(J))
setindex!(A::Matrix, x::AbstractSparseMatrixCSC, I::AbstractVector{Bool}, J::AbstractVector{<:Integer}) = setindex!(A, Array(x), findall(I), J)

function setindex!(A::AbstractSparseMatrixCSC, x::AbstractArray, I::AbstractMatrix{Bool})
    require_one_based_indexing(A, x, I)
    checkbounds(A, I)
    if _is_fixed(A)   # the scalar path keeps the pattern and throws outside it
        k = 0
        for ci in CartesianIndices(I)
            I[ci] && (A[ci] = x[k += 1])
        end
        return A
    end
    n = sum(I)
    (n == 0) && (return A)

    colptrA = getcolptr(A); rowvalA = rowvals(A); nzvalA = nonzeros(A)
    colptrB = colptrA; rowvalB = rowvalA; nzvalB = nzvalA
    nadd = 0
    bidx = xidx = 1
    r1 = r2 = 0

    @inbounds for col in axes(A,2)
        r1 = Int(first(nzrange(A, col)))
        r2 = Int(last(nzrange(A, col)))

        for row in axes(A,1)
            if I[row, col]
                v = convert(eltype(A), x[xidx])
                xidx += 1

                if r1 <= r2
                    copylen = searchsortedfirst(view(rowvalA, r1:r2), row) - 1
                    if (copylen > 0)
                        if (nadd > 0)
                            copyto!(rowvalB, bidx, rowvalA, r1, copylen)
                            copyto!(nzvalB, bidx, nzvalA, r1, copylen)
                        end
                        bidx += copylen
                        r1 += copylen
                    end
                end

                # 0: no change, 1: update, 2: add new
                mode = ((r1 <= r2) && (rowvalA[r1] == row)) ? 1 : (_isimplicitzero(v, eltype(A)) ? 0 : 2)

                if (mode > 1) && (nadd == 0)
                    # copy storage to take changes
                    colptrA = copy(colptrB)
                    memreq = (x == 0) ? 0 : n
                    # this x == 0 check and approach doesn't jive with use of v above
                    # and may not make sense generally, as scalar x == 0 probably
                    # means this section should never be called. also may not be generic.
                    # TODO: clean this up, maybe separate scalar and array X cases
                    rowvalA = copy(rowvalB)
                    nzvalA = copy(nzvalB)
                    resize!(rowvalB, length(rowvalA)+memreq)
                    resize!(nzvalB, length(rowvalA)+memreq)
                end
                if mode == 1
                    rowvalB[bidx] = row
                    nzvalB[bidx] = v
                    bidx += 1
                    r1 += 1
                elseif mode == 2
                    rowvalB[bidx] = row
                    nzvalB[bidx] = v
                    bidx += 1
                    nadd += 1
                end
                (xidx > n) && break
            end # if I[row, col]
        end # for row in axes(A,1)

        if (nadd != 0)
            l = r2-r1+1
            if l > 0
                copyto!(rowvalB, bidx, rowvalA, r1, l)
                copyto!(nzvalB, bidx, nzvalA, r1, l)
                bidx += l
            end
            colptrB[col+1] = bidx

            if (xidx > n) && (length(colptrB) > (col+1))
                diff = nadd
                colptrB[(col+2):end] = colptrA[(col+2):end] .+ diff
                r1 = colptrA[col+1]
                r2 = colptrA[end]-1
                l = r2-r1+1
                if l > 0
                    copyto!(rowvalB, bidx, rowvalA, r1, l)
                    copyto!(nzvalB, bidx, nzvalA, r1, l)
                    bidx += l
                end
            end
        else
            bidx = colptrA[col+1]
        end
        (xidx > n) && break
    end # for col in axes(A,2)

    if (nadd != 0)
        n = length(nzvalB)
        if n > (bidx-1)
            deleteat!(nzvalB, bidx:n)
            deleteat!(rowvalB, bidx:n)
        end
    end
    return _checkbuffers(A)
end

function setindex!(A::AbstractSparseMatrixCSC, x::AbstractArray, Ix::AbstractVector{<:Integer})
    require_one_based_indexing(A, x, Ix)
    (I,) = Base.ensure_indexable(to_indices(A, (Ix,)))
    if _is_fixed(A)   # the scalar path keeps the pattern and throws outside it
        for (k, i) in enumerate(I)
            A[i] = x[k]
        end
        return A
    end
    # We check bounds after sorting I
    n = length(I)
    (n == 0) && (return A)

    colptrA = getcolptr(A); rowvalA = rowvals(A); nzvalA = nonzeros(A); szA = size(A)
    colptrB = colptrA; rowvalB = rowvalA; nzvalB = nzvalA
    nadd = 0
    bidx = aidx = 1

    S = issorted(I) ? (1:n) : sortperm(I)
    sxidx = r1 = r2 = 0

    if (!isempty(I) && (I[S[1]] < 1 || I[S[end]] > widelength(A)))
        throw(BoundsError(A, I))
    end

    isa(x, AbstractArray) && setindex_shape_check(x, length(I))

    CartIndsA = CartesianIndices(szA)

    lastcol = 0
    (nrowA, ncolA) = szA
    @inbounds for xidx in 1:n
        sxidx = S[xidx]
        (xidx < n) && (I[sxidx] == I[S[xidx+1]]) && continue

        row,col = Tuple(CartIndsA[I[sxidx]])
        v = convert(eltype(A), x[sxidx])

        if col > lastcol
            r1 = Int(first(nzrange(A, col)))
            r2 = Int(last(nzrange(A, col)))

            # copy from last position till current column
            if (nadd > 0)
                colptrB[(lastcol+1):col] = colptrA[(lastcol+1):col] .+ nadd
                copylen = r1 - aidx
                if copylen > 0
                    copyto!(rowvalB, bidx, rowvalA, aidx, copylen)
                    copyto!(nzvalB, bidx, nzvalA, aidx, copylen)
                    aidx += copylen
                    bidx += copylen
                end
            else
                aidx = bidx = r1
            end
            lastcol = col
        end

        if r1 <= r2
            copylen = searchsortedfirst(view(rowvalA, r1:r2), row) - 1
            if (copylen > 0)
                if (nadd > 0)
                    copyto!(rowvalB, bidx, rowvalA, r1, copylen)
                    copyto!(nzvalB, bidx, nzvalA, r1, copylen)
                end
                bidx += copylen
                r1 += copylen
                aidx += copylen
            end
        end

        # 0: no change, 1: update, 2: add new
        mode = ((r1 <= r2) && (rowvalA[r1] == row)) ? 1 : (_isimplicitzero(v, eltype(A)) ? 0 : 2)

        if (mode > 1) && (nadd == 0)
            # copy storage to take changes
            colptrA = copy(colptrB)
            memreq = (x == 0) ? 0 : n
            # see comment/TODO for same statement in preceding logical setindex! method
            rowvalA = copy(rowvalB)
            nzvalA = copy(nzvalB)
            resize!(rowvalB, length(rowvalA)+memreq)
            resize!(nzvalB, length(rowvalA)+memreq)
        end
        if mode == 1
            rowvalB[bidx] = row
            nzvalB[bidx] = v
            bidx += 1
            aidx += 1
            r1 += 1
        elseif mode == 2
            rowvalB[bidx] = row
            nzvalB[bidx] = v
            bidx += 1
            nadd += 1
        end
    end

    # copy the rest
    @inbounds if (nadd > 0)
        colptrB[(lastcol+1):end] = colptrA[(lastcol+1):end] .+ nadd
        r1 = colptrA[end]-1
        copylen = r1 - aidx + 1
        if copylen > 0
            copyto!(rowvalB, bidx, rowvalA, aidx, copylen)
            copyto!(nzvalB, bidx, nzvalA, aidx, copylen)
            aidx += copylen
            bidx += copylen
        end

        n = length(nzvalB)
        if n > (bidx-1)
            deleteat!(nzvalB, bidx:n)
            deleteat!(rowvalB, bidx:n)
        end
    end
    return _checkbuffers(A)
end

## dropstored! methods
"""
    dropstored!(A::AbstractSparseMatrixCSC, i::Integer, j::Integer)

Drop entry `A[i,j]` from `A` if `A[i,j]` is stored, and otherwise do nothing.

```jldoctest
julia> A = sparse([1 2; 0 0])
2×2 SparseMatrixCSC{Int64, Int64} with 2 stored entries:
 1  2
 ⋅  ⋅

julia> SparseArrays.dropstored!(A, 1, 2); A
2×2 SparseMatrixCSC{Int64, Int64} with 1 stored entry:
 1  ⋅
 ⋅  ⋅
```
"""
function dropstored!(A::AbstractSparseMatrixCSC, i::Integer, j::Integer)
    if !((1 <= i <= size(A, 1)) & (1 <= j <= size(A, 2)))
        throw(BoundsError(A, (i,j)))
    end
    coljfirstk = Int(getcolptr(A)[j])
    coljlastk = Int(getcolptr(A)[j+1] - 1)
    searchk = searchsortedfirst(view(rowvals(A), coljfirstk:coljlastk), i) + coljfirstk - 1
    if searchk <= coljlastk && rowvals(A)[searchk] == i
        # Entry A[i,j] is stored. Drop and return.
        deleteat!(rowvals(A), searchk)
        deleteat!(nonzeros(A), searchk)
        @simd for m in (j+1):(size(A, 2) + 1)
            @inbounds getcolptr(A)[m] -= 1
        end
    end
    return _checkbuffers(A)
end
"""
    dropstored!(A::AbstractSparseMatrixCSC, I::AbstractVector{<:Integer}, J::AbstractVector{<:Integer})

For each `(i,j)` where `i in I` and `j in J`, drop entry `A[i,j]` from `A` if `A[i,j]` is
stored and otherwise do nothing. Derivative forms:

    dropstored!(A::AbstractSparseMatrixCSC, i::Integer, J::AbstractVector{<:Integer})
    dropstored!(A::AbstractSparseMatrixCSC, I::AbstractVector{<:Integer}, j::Integer)

# Examples
```jldoctest
julia> A = sparse(Diagonal([1, 2, 3, 4]))
4×4 SparseMatrixCSC{Int64, Int64} with 4 stored entries:
 1  ⋅  ⋅  ⋅
 ⋅  2  ⋅  ⋅
 ⋅  ⋅  3  ⋅
 ⋅  ⋅  ⋅  4

julia> SparseArrays.dropstored!(A, [1, 2], [1, 1])
4×4 SparseMatrixCSC{Int64, Int64} with 3 stored entries:
 ⋅  ⋅  ⋅  ⋅
 ⋅  2  ⋅  ⋅
 ⋅  ⋅  3  ⋅
 ⋅  ⋅  ⋅  4
```
"""
function dropstored!(A::AbstractSparseMatrixCSC,
        I::AbstractVector{<:Integer}, J::AbstractVector{<:Integer})
    require_one_based_indexing(A, I, J)
    m, n = size(A)
    nnzA = nnz(A)
    (nnzA == 0) && (return A)

    !issorted(I) && (I = sort(I))
    !issorted(J) && (J = sort(J))

    if (!isempty(I) && (I[1] < 1 || I[end] > m)) || (!isempty(J) && (J[1] < 1 || J[end] > n))
        throw(BoundsError(A, (I, J)))
    end

    if isempty(I) || isempty(J)
        return A
    end

    rowval = rowvalA = rowvals(A)
    nzval = nzvalA = nonzeros(A)
    rowidx = 1
    ndel = 0
    @inbounds for col in axes(A,2)
        rrange = nzrange(A, col)
        if ndel > 0
            getcolptr(A)[col] = getcolptr(A)[col] - ndel
        end

        if isempty(rrange) || !(col in J)
            nincl = length(rrange)
            if(ndel > 0) && !isempty(rrange)
                copyto!(rowvalA, rowidx, rowval, rrange[1], nincl)
                copyto!(nzvalA, rowidx, nzval, rrange[1], nincl)
            end
            rowidx += nincl
        else
            for ridx in rrange
                if rowval[ridx] in I
                    if ndel == 0
                        rowval = copy(rowvalA)
                        nzval = copy(nzvalA)
                    end
                    ndel += 1
                else
                    if ndel > 0
                        rowvalA[rowidx] = rowval[ridx]
                        nzvalA[rowidx] = nzval[ridx]
                    end
                    rowidx += 1
                end
            end
        end
    end

    if ndel > 0
        getcolptr(A)[n+1] = rowidx
        deleteat!(rowvalA, rowidx:nnzA)
        deleteat!(nzvalA, rowidx:nnzA)
    end
    return _checkbuffers(A)
end
dropstored!(A::AbstractSparseMatrixCSC, i::Integer, J::AbstractVector{<:Integer}) = dropstored!(A, [i], J)
dropstored!(A::AbstractSparseMatrixCSC, I::AbstractVector{<:Integer}, j::Integer) = dropstored!(A, I, [j])
dropstored!(A::AbstractSparseMatrixCSC, ::Colon, j::Union{Integer,AbstractVector}) = dropstored!(A, axes(A,1), j)
dropstored!(A::AbstractSparseMatrixCSC, i::Union{Integer,AbstractVector}, ::Colon) = dropstored!(A, i, axes(A,2))
dropstored!(A::AbstractSparseMatrixCSC, ::Colon, ::Colon) = dropstored!(A, axes(A,1), axes(A,2))
dropstored!(A::AbstractSparseMatrixCSC, ::Colon) = dropstored!(A, :, :)
# TODO: Several of the preceding methods are optimization candidates.
# TODO: Implement linear indexing methods for dropstored! ?
# TODO: Implement logical indexing methods for dropstored! ?

## Indexing into Matrices can return SparseVectors

# Column slices
function getindex(x::AbstractSparseMatrixCSC, ::Colon, j::Integer)
    checkbounds(x, :, j)
    nzr = nzrange(x, j)
    return @if_move_fixed x SparseVector(size(x, 1), rowvals(x)[nzr], nonzeros(x)[nzr])
end

function getindex(x::AbstractSparseMatrixCSC, I::AbstractUnitRange, j::Integer)
    checkbounds(x, I, j)
    # Get the selected column
    c1 = Int(first(nzrange(x, j)))
    c2 = Int(last(nzrange(x, j)))
    # Restrict to the selected rows
    r1 = searchsortedfirst(view(rowvals(x), c1:c2), first(I)) + c1 - 1
    r2 = searchsortedlast(view(rowvals(x), c1:c2), last(I)) + c1 - 1
    return @if_move_fixed x SparseVector(length(I), [rowvals(x)[i] - first(I) + 1 for i = r1:r2], nonzeros(x)[r1:r2])
end

# Nonscalar indexing of an adjoint or transpose indexes the parent with the indices swapped
@propagate_inbounds getindex(M::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}, I, J) =
    _getindex_adjtrans(M, (I, J), _lower_indices(M, I, J)...)
# resolves the ambiguity with LinearAlgebra's scalar method
@propagate_inbounds getindex(M::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}, i::Int, j::Int) =
    wrapperop(M)(parent(M)[j, i])
_getindex_adjtrans(M, _, i::AbstractVector, j::AbstractVector) = copy(wrapperop(M)(parent(M)[j, i]))
_getindex_adjtrans(M, _, i::Integer, j::AbstractVector) = map!(wrapperop(M), parent(M)[j, i])
_getindex_adjtrans(M, _, i::AbstractVector, j::Integer) = map!(wrapperop(M), parent(M)[j, i])
@propagate_inbounds _getindex_adjtrans(M, _, i::Integer, j::Integer) = wrapperop(M)(parent(M)[j, i])
@propagate_inbounds _getindex_adjtrans(M, I, L...) = invoke(getindex, Tuple{AbstractArray,Vararg{Any}}, M, I...)

# In the general case, we piggy back upon SparseMatrixCSC's optimized solution
@inline getindex(A::AbstractSparseMatrixCSC, I::AbstractVector, J::Integer) =
    let M = A[I, [J]]
        @if_move_fixed A SparseVector(size(M, 1), rowvals(M), nonzeros(M))
    end

# Row slices
getindex(A::AbstractSparseMatrixCSC, i::Integer, ::Colon) = A[i, 1:end]
function Base.getindex(A::AbstractSparseMatrixCSC{Tv,Ti}, i::Integer, J::AbstractVector) where {Tv,Ti}
    require_one_based_indexing(A, J)
    _, J = _lower_indices(A, i, J)
    nJ = length(J)
    rowvalA = rowvals(A); nzvalA = nonzeros(A)

    nzinds = Vector{Ti}()
    nzvals = Vector{Tv}()

    # adapted from SparseMatrixCSC's sorted_bsearch_A
    ptrI = 1
    @inbounds for j = 1:nJ
        col = J[j]
        rowI = i
        ptrA = Int(first(nzrange(A, col)))
        stopA = Int(last(nzrange(A, col)))
        if ptrA <= stopA
            if rowvalA[ptrA] <= rowI
                ptrA += searchsortedfirst(view(rowvalA, ptrA:stopA), rowI) - 1
                if ptrA <= stopA && rowvalA[ptrA] == rowI
                    push!(nzinds, j)
                    push!(nzvals, nzvalA[ptrA])
                end
            end
            ptrI += 1
        end
    end
    @if_move_fixed A SparseVector(nJ, nzinds, nzvals)
end


# Logical and linear indexing into SparseMatrices
getindex(A::AbstractSparseMatrixCSC, I::AbstractVector{Bool}) = _logical_index(A, I) # Ambiguities
getindex(A::AbstractSparseMatrixCSC, I::AbstractArray{Bool}) = _logical_index(A, I)
function _logical_index(A::AbstractSparseMatrixCSC{Tv}, I::AbstractArray{Bool}) where Tv
    require_one_based_indexing(A, I)
    checkbounds(A, I)
    mask = reshape(I, size(A))   # a vector mask indexes linearly
    n = sum(I)
    nnzB = min(n, nnz(A))

    rowvalA = rowvals(A); nzvalA = nonzeros(A)
    rowvalB = Vector{Int}(undef, nnzB)
    nzvalB = Vector{Tv}(undef, nnzB)
    c = 1
    rowB = 1

    @inbounds for col in axes(A,2)
        r1 = first(nzrange(A, col))
        r2 = last(nzrange(A, col))

        for row in axes(A,1)
            if mask[row, col]
                while (r1 <= r2) && (rowvalA[r1] < row)
                    r1 += 1
                end
                if (r1 <= r2) && (rowvalA[r1] == row)
                    nzvalB[c] = nzvalA[r1]
                    rowvalB[c] = rowB
                    c += 1
                end
                rowB += 1
                (rowB > n) && break
            end
        end
        (rowB > n) && break
    end
    if nnzB > (c-1)
        deleteat!(nzvalB, c:nnzB)
        deleteat!(rowvalB, c:nnzB)
    end
    return @if_move_fixed A I SparseVector(n, rowvalB, nzvalB)
end

# TODO: further optimizations are available for ::Colon and other types of AbstractRange
getindex(A::AbstractSparseMatrixCSC, ::Colon) = A[1:end]

function getindex(A::AbstractSparseMatrixCSC{Tv}, I::AbstractUnitRange) where Tv
    require_one_based_indexing(A, I)
    checkbounds(A, I)
    szA = size(A)
    nA = szA[1]*szA[2]
    rowvalA = rowvals(A)
    nzvalA = nonzeros(A)

    n = length(I)
    nnzB = min(n, nnz(A))
    rowvalB = Vector{Int}(undef, nnzB)
    nzvalB = Vector{Tv}(undef, nnzB)

    CartIndsA = CartesianIndices(szA)
    LinIndsA = LinearIndices(szA)

    if nnzB > 0
        rowstart,colstart = Tuple(CartIndsA[first(I)])
        rowend,colend = Tuple(CartIndsA[last(I)])

        idxB = 1
        @inbounds for col in colstart:colend
            minrow = (col == colstart ? rowstart : 1)
            maxrow = (col == colend ? rowend : szA[1])
            for r in nzrange(A, col)
                rowA = rowvalA[r]
                if minrow <= rowA <= maxrow
                    rowvalB[idxB] = LinIndsA[rowA, col] - first(I) + 1
                    nzvalB[idxB] = nzvalA[r]
                    idxB += 1
                end
            end
        end
        if nnzB > (idxB-1)
            deleteat!(nzvalB, idxB:nnzB)
            deleteat!(rowvalB, idxB:nnzB)
        end
    end
    @if_move_fixed A SparseVector(n, rowvalB, nzvalB)
end

function getindex(A::AbstractSparseMatrixCSC{Tv,Ti}, I::AbstractVector) where {Tv,Ti}
    I isa Base.LogicalIndex && return _logical_index(A, I.mask)
    require_one_based_indexing(A, I)
    @boundscheck checkbounds(A, I)
    szA = size(A)
    nA = szA[1]*szA[2]
    rowvalA = rowvals(A)
    nzvalA = nonzeros(A)

    n = length(I)
    nnzB = min(n, nnz(A))
    rowvalB = Vector{Ti}(undef, nnzB)
    nzvalB = Vector{Tv}(undef, nnzB)

    CartIndsA = CartesianIndices(szA)

    idxB = 1
    for i in 1:n
        row,col = Tuple(CartIndsA[I[i]])
        for r in nzrange(A, col)
            @inbounds if rowvalA[r] == row
                if idxB <= nnzB
                    rowvalB[idxB] = i
                    nzvalB[idxB] = nzvalA[r]
                    idxB += 1
                else # this can happen if there are repeated indices in I
                    push!(rowvalB, i)
                    push!(nzvalB, nzvalA[r])
                end
                break
            end
        end
    end
    if nnzB > (idxB-1)
        deleteat!(nzvalB, idxB:nnzB)
        deleteat!(rowvalB, idxB:nnzB)
    end
    return @if_move_fixed A SparseVector(n, rowvalB, nzvalB)
end
