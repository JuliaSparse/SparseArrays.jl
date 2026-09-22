# This file is a part of Julia. License is MIT: https://julialang.org/license

## Reductions

# Reductions along a dimension return a dense `Array`, as for dense input. A sparse result
# (issue #43) is opt-in with the keyword `sparse = true`, see `_mapreduce_dim_sparse` below.
# Covers views and adjoints so they reduce like their copy (#377), where Base's `similar`
# would otherwise give a sparse result.
function Base.reducedim_initarray(A::Union{SparseMatrixCSCOrColumnSubset,AdjOrTrans{<:Any,<:SparseMatrixCSCOrColumnSubset}},
                                  region, v0, ::Type{R}) where {R}
    fill!(Array{R}(undef, Base.to_shape(Base.reduced_indices(A, region))), v0)
end

# General mapreduce
function _mapreducezeros(f::F, op::G, ::Type{T}, nzeros::Integer, v0) where {F,G,T}
    nzeros == 0 && return v0

    # Reduce over first zero
    zeroval = f(zero(T))
    v = op(v0, zeroval)
    isequal(v, v0) && return v

    # Reduce over remaining zeros
    for i = 2:nzeros
        lastv = v
        v = op(v, zeroval)
        # Bail out early if we reach a fixed point
        isequal(v, lastv) && break
    end

    v
end

function Base._mapreduce(f::F, op::G, ::Base.IndexCartesian, A::SparseMatrixCSCOrColumnSubset{T}) where {F,G,T}
    z = nnz(A)
    n = widelength(A)
    if z == 0
        if n == 0
            Base.mapreduce_empty(f, op, T)
        else
            _mapreducezeros(f, op, T, n-z-1, f(zero(T)))
        end
    else
        _mapreducezeros(f, op, T, n-z, Base._mapreduce(f, op, nzvalview(A)))
    end
end

# Specialized mapreduce for +/*/min/max/_extrema_rf
_mapreducezeros(f::F, op::Union{typeof(Base.add_sum),typeof(+)}, ::Type{T}, nzeros::Integer, v0) where {F,T} =
    nzeros == 0 ? op(zero(v0), v0) : op(f(zero(T))*nzeros, v0)
_mapreducezeros(f::F, op::Union{typeof(Base.mul_prod),typeof(*)},::Type{T}, nzeros::Integer, v0) where {F,T} =
    nzeros == 0 ? op(one(v0), v0) : op(f(zero(T))^nzeros, v0)
_mapreducezeros(f::F, op::Union{typeof(min),typeof(max)}, ::Type{T}, nzeros::Integer, v0) where {F,T} =
    nzeros == 0 ? v0 : op(v0, f(zero(T)))
_mapreducezeros(f::Base.ExtremaMap, op::typeof(Base._extrema_rf), ::Type{T}, nzeros::Integer, v0) where {T} =
    nzeros == 0 ? v0 : op(v0, f(zero(T)))

# Specialized mapreduce for any and all
Base._any(f, A::SparseMatrixCSCOrColumnSubset, ::Colon) =
    iszero(widelength(A)) ? false : Base._mapreduce(f, |, IndexCartesian(), A)
Base._all(f, A::SparseMatrixCSCOrColumnSubset, ::Colon) =
    iszero(widelength(A)) ? true  : Base._mapreduce(f, &, IndexCartesian(), A)

function Base._mapreduce(f::F, op::Union{typeof(Base.mul_prod),typeof(*)}, ::Base.IndexCartesian, A::SparseMatrixCSCOrColumnSubset{T}) where {F,T}
    nnzA = nnz(A)
    nzeros = widelength(A) - nnzA
    if nzeros == 0
        # No zeros, so don't compute f(0) since it might throw
        Base._mapreduce(f, op, nzvalview(A))
    else
        v = f(zero(T))^(nzeros)
        # Bail out early if initial reduction value is zero or if there are no stored elements
        (_iszero(v) || nnzA == 0) ? v : v*Base._mapreduce(f, op, nzvalview(A))
    end
end

# The `sparse` keyword, the opt-in for a sparse result (issue #43): `sum(A; dims = 2, sparse = true)`
# and the other reductions along a dimension return a `SparseMatrixCSC` (a `SparseVector` for a
# sparse vector) that stores an entry only for the rows or columns of `A` that store one (all of
# them when a slice that stores
# nothing reduces to something nonzero, as for `f(0) != 0`), in time proportional to
# nnz(A) + length(result) rather than to length(A). Base's `sum`, `prod`, `maximum` and
# `minimum` forward unknown keywords to `mapreduce`, so the `mapreduce` method serves them all; `any`,
# `all` and `count` do not and get their own methods below.
for T in (:SparseMatrixCSCOrColumnSubset, :(AdjOrTrans{<:Any,<:SparseMatrixCSCOrColumnSubset}), :SparseVectorOrView)
    @eval function Base.mapreduce(f, op, A::$T; dims=:, init=Base._InitialValue(), sparse::Bool=false)
        sparse || return Base._mapreduce_dim(f, op, init, A, dims)
        dims === (:) && throw(ArgumentError("a sparse result needs a reduction along a dimension, pass `dims`"))
        return _mapreduce_dim_sparse(f, op, init, A, dims)
    end
    for (fname, _fname, op) in ((:any, :_any, :(Base.or_any)), (:all, :_all, :(Base.and_all)))
        @eval begin
            Base.$fname(A::$T; dims=:, sparse::Bool=false) = Base.$fname(identity, A; dims, sparse)
            Base.$fname(f, A::$T; dims=:, sparse::Bool=false) =
                sparse ? mapreduce(f, $op, A; dims, sparse) : Base.$_fname(f, A, dims)
        end
    end
    @eval begin
        Base.count(A::$T; dims=:, init=0, sparse::Bool=false) = count(identity, A; dims, init, sparse)
        Base.count(f, A::$T; dims=:, init=0, sparse::Bool=false) =
            sparse ? mapreduce(Base._bool(f), Base.add_sum, A; dims, init, sparse) : Base._count(f, A, dims, init)
    end
end

# The slices are reduced the way Base's dense result is: seeded with `init` when given and
# otherwise with `mapreduce_first` of their first element, with the entries a slice does not
# store folded in through `_mapreducezeros`.
_seed(f, op, ::Base._InitialValue, x) = Base.mapreduce_first(f, op, x)
_seed(f, op, init, x) = op(init, f(x))
# element type of the dense result, taken from Base's initialization of a 1 x 1 stand-in with
# `A`'s element type. Base may map the stand-in's entry, so it is a zero only if `A` has one.
_reduced_eltype(f, op, ::Base._InitialValue, A::AbstractArray{T}) where T =
    eltype(Base.reducedim_init(f, op, fill!(Matrix{T}(undef, 1, 1), nnz(A) == length(A) > 0 ? _firststored(A) : zero(T)), 1))
_reduced_eltype(f, op, init, A) = typeof(init)
_firststored(A::AbstractVector) = first(nonzeros(A))
_firststored(A::AbstractMatrix) = getnzval(A)[first(getnzrange(A, 1))]
# the reduction of a slice with no entries at all, as Base initializes the dense result
_reduced_empty(f, op, ::Base._InitialValue, ::Type{T}) where T =
    Base.reducedim_init(f, op, Matrix{T}(undef, 0, 1), 1)[1]
_reduced_empty(f, op, init, ::Type{T}) where T = init
# the reduction of a slice of length `len` that stores nothing
_reduce_unstored(f, op, init, ::Type{T}, len) where T =
    len == 0 ? _reduced_empty(f, op, init, T) : _mapreducezeros(f, op, T, len - 1, _seed(f, op, init, zero(T)))

function _sparse_reduced_eltype(f, op, init, A)
    Tr = _reduced_eltype(f, op, init, A)
    applicable(zero, Tr) || throw(ArgumentError("cannot store a sparse result of element type $Tr, " *
        "which has no zero (as for `extrema`); reduce without `sparse = true`"))
    return Tr
end

# A slice of `A'` is a slice of `A` along the other dimension. Reducing both dimensions folds
# the elements in their order, which the parent does not share.
function _mapreduce_dim_sparse(f, op, init, A::AdjOrTrans, dims)
    1 in dims && 2 in dims && return _mapreduce_dim_sparse(f, op, init, copy(A), dims)
    g = A isa Adjoint ? adjoint : transpose
    return permutedims(_mapreduce_dim_sparse(f ∘ g, op, init, parent(A), map(_switch_dim12, dims)), (2, 1))
end
_switch_dim12(d) = d == 1 ? 2 : d == 2 ? 1 : d

function _mapreduce_dim_sparse(f, op, init, A::SparseVectorOrView{T,Ti}, dims) where {T,Ti}
    Base.reduced_indices(A, dims)   # validates `dims`
    Tr = _sparse_reduced_eltype(f, op, init, A)
    # a dimension beyond 1: every entry is a slice of its own
    1 in dims || return convert(SparseVector{Tr,Ti}, map(x -> _seed(f, op, init, x), A isa SubArray ? copy(A) : A))
    v = isempty(A) ? _reduced_empty(f, op, init, T) : _seed(identity, op, init, mapreduce(f, op, A))
    return nnz(A) > 0 || !isequal(v, zero(Tr)) ? SparseVector(1, Ti[1], Tr[v]) : spzeros(Tr, Ti, 1)
end

function _mapreduce_dim_sparse(f, op, init, A::SparseMatrixCSCOrColumnSubset{T,Ti}, dims) where {T,Ti}
    m, n = size(A)
    rm, rn = map(length, Base.reduced_indices(A, dims))   # also validates `dims`
    Tr = _sparse_reduced_eltype(f, op, init, A)
    if rm == rn == 1
        R = spzeros(Tr, Ti, 1, 1)
        v = isempty(A) ? _reduced_empty(f, op, init, T) : _seed(identity, op, init, mapreduce(f, op, A))
        if nnz(A) > 0 || !isequal(v, zero(Tr))
            push!(rowvals(R), 1)
            push!(nonzeros(R), v)
            getcolptr(R)[2] = 2
        end
        return R
    elseif rm == 1
        return _mapreducerows_sparse!(f, op, init, spzeros(Tr, Ti, 1, n), A)
    elseif rn == 1
        return _mapreducecols_sparse!(f, op, init, spzeros(Tr, Ti, m, 1), A)
    else
        # a dimension beyond 2: every entry is a slice of its own
        return convert(SparseMatrixCSC{Tr,Ti}, map(x -> _seed(f, op, init, x), A isa SubArray ? copy(A) : A))
    end
end

# `R` is a structurally empty `1 x n` sparse matrix: its columns are built in order
function _mapreducerows_sparse!(f, op, init, R::SparseMatrixCSC, A::SparseMatrixCSCOrColumnSubset{T}) where T
    nzval = getnzval(A)
    m, n = size(A)
    z = zero(eltype(R))
    # the reduction of a column that stores nothing; when every column is full it is not
    # needed and f(0) is not evaluated, since it might throw
    zunstored = nnz(A) == m*n && m > 0 ? z : _reduce_unstored(f, op, init, T, m)
    store_unstored = !isequal(zunstored, z)
    Rcolptr, Rrowval, Rnzval = getcolptr(R), rowvals(R), nonzeros(R)
    nstored = store_unstored ? n : count(col -> !isempty(getnzrange(A, col)), 1:n)
    resize!(Rrowval, nstored)
    fill!(Rrowval, 1)
    resize!(Rnzval, nstored)
    k = 0
    @inbounds for col in 1:n
        rng = getnzrange(A, col)
        if isempty(rng)
            store_unstored || (Rcolptr[col+1] = k + 1; continue)
            v = zunstored
        else
            r = _seed(f, op, init, nzval[first(rng)])
            @simd for j in first(rng)+1:last(rng)
                r = op(r, f(nzval[j]))
            end
            v = _mapreducezeros(f, op, T, m - length(rng), r)
        end
        k += 1
        Rnzval[k] = v
        Rcolptr[col+1] = k + 1
    end
    return R
end

# `R` is a structurally empty `m x 1` sparse matrix. With enough stored entries the rows are
# reduced in a dense workspace that is then compressed into `R`; a hypersparse `A` instead has
# its stored entries sorted by row so that only the rows storing something are ever visited.
function _mapreducecols_sparse!(f, op, init, R::SparseMatrixCSC, A::SparseMatrixCSCOrColumnSubset{T}) where T
    m, n = size(A)
    Tr = eltype(R)
    z = zero(Tr)
    rows = view(getrowval(A), _storedinds(A))
    vals = view(getnzval(A), _storedinds(A))
    nz = length(rows)
    zunstored = nz == m*n && n > 0 ? z : _reduce_unstored(f, op, init, T, n)
    store_unstored = !isequal(zunstored, z)
    Rcolptr, Rrowval, Rnzval = getcolptr(R), rowvals(R), nonzeros(R)
    if store_unstored || 8 * nz >= m
        W = Vector{Tr}(undef, m)
        cnt = zeros(Int, m)   # stored entries seen in each row
        @inbounds for j in eachindex(rows, vals)
            i = rows[j]
            W[i] = cnt[i] == 0 ? _seed(f, op, init, vals[j]) : op(W[i], f(vals[j]))
            cnt[i] += 1
        end
        resize!(Rrowval, m)
        resize!(Rnzval, m)
        k = 0
        @inbounds for i in 1:m
            if cnt[i] > 0
                k += 1
                Rrowval[k] = i
                Rnzval[k] = _mapreducezeros(f, op, T, n - cnt[i], W[i])
            elseif store_unstored
                k += 1
                Rrowval[k] = i
                Rnzval[k] = zunstored
            end
        end
        resize!(Rrowval, k)
        resize!(Rnzval, k)
    else
        perm = sortperm(rows; alg=Base.Sort.DEFAULT_STABLE)   # keeps each row's entries in column order
        s = 1
        @inbounds while s <= nz
            row = rows[perm[s]]
            r = _seed(f, op, init, vals[perm[s]])
            t = s + 1
            while t <= nz && rows[perm[t]] == row
                r = op(r, f(vals[perm[t]]))
                t += 1
            end
            push!(Rrowval, row)
            push!(Rnzval, _mapreducezeros(f, op, T, n - (t - s), r))
            s = t
        end
    end
    Rcolptr[2] = length(Rnzval) + 1
    return R
end

# General mapreducedim
function _mapreducerows!(f, op, R::AbstractArray, A::SparseMatrixCSCOrColumnSubset{T}) where T
    require_one_based_indexing(A, R)
    rowval = getrowval(A)
    nzval = getnzval(A)
    m, n = size(A)
    @inbounds for col in axes(A,2)
        r = R[1, col]
        @simd for j in getnzrange(A, col)
            r = op(r, f(nzval[j]))
        end
        R[1, col] = _mapreducezeros(f, op, T, m-length(getnzrange(A, col)), r)
    end
    R
end

function _mapreducecols!(f, op, R::AbstractArray, A::SparseMatrixCSCOrColumnSubset{Tv,Ti}) where {Tv,Ti}
    require_one_based_indexing(A, R)
    rowval = getrowval(A)
    nzval = getnzval(A)
    m, n = size(A)
    rownz = fill(convert(Ti, n), m)
    @inbounds for col in axes(A,2)
        @simd for j in getnzrange(A, col)
            row = rowval[j]
            R[row, 1] = op(R[row, 1], f(nzval[j]))
            rownz[row] -= 1
        end
    end
    @inbounds for i = 1:m
        R[i, 1] = _mapreducezeros(f, op, Tv, Int(rownz[i]), R[i, 1])
    end
    R
end

function Base._mapreducedim!(f::F, op::G, R::AbstractArray, A::SparseMatrixCSCOrColumnSubset{T}) where {F,G,T}
    require_one_based_indexing(A, R)
    lsiz = Base.check_reducedims(R,A)
    isempty(A) && return R

    if size(R, 1) == size(R, 2) == 1
        # Reduction along both columns and rows
        R[1, 1] = op(R[1, 1], mapreduce(f, op, A))
    elseif size(R, 1) == 1
        # Reduction along rows
        _mapreducerows!(f, op, R, A)
    elseif size(R, 2) == 1
        # Reduction along columns
        _mapreducecols!(f, op, R, A)
    else
        # Reduction along a dimension > 2
        # Compute op(R, f(A))
        m, n = size(A)
        rowval = getrowval(A)
        nzval = getnzval(A)
        if nnz(A) == m*n
            # No zeros, so don't compute f(0) since it might throw
            @inbounds for col in axes(A,2)
                @simd for j in getnzrange(A, col)
                    R[rowval[j], col] = op(R[rowval[j], col], f(nzval[j]))
                end
            end
        else
            zeroval = f(zero(T))
            @inbounds for col in axes(A,2)
                lastrow = 0
                for j in getnzrange(A, col)
                    row = rowval[j]
                    @simd for i = lastrow+1:row-1 # Zeros before this nonzero
                        R[i, col] = op(R[i, col], zeroval)
                    end
                    R[row, col] = op(R[row, col], f(nzval[j]))
                    lastrow = row
                end
                @simd for i = lastrow+1:m         # Zeros at end
                    R[i, col] = op(R[i, col], zeroval)
                end
            end
        end
    end
    R
end

# LinearAlgebra forwards the commutative reductions of an adjoint or transpose to its parent;
# this covers the others. Reducing both dimensions folds the elements in their order, which
# the parent does not share.
function Base._mapreducedim!(f, op, R::AbstractMatrix, A::AdjOrTrans{<:Any,<:SparseMatrixCSCOrColumnSubset})
    if size(R, 1) == size(R, 2) == 1
        Base._mapreducedim!(f, op, R, copy(A))
    else
        Base._mapreducedim!(f ∘ (A isa Adjoint ? adjoint : transpose), op, PermutedDimsArray(R, (2, 1)), parent(A))
    end
    return R
end

# Specialized mapreducedim for + cols to avoid allocating a
# temporary array when f(0) == 0
function _mapreducecols!(f, op::typeof(+), R::AbstractArray, A::SparseMatrixCSCOrColumnSubset{Tv,Ti}) where {Tv,Ti}
    require_one_based_indexing(A, R)
    rowval = getrowval(A)
    nzval = getnzval(A)
    m, n = size(A)
    if nnz(A) == m*n
        # No zeros, so don't compute f(0) since it might throw
        @inbounds for col in axes(A,2)
            @simd for j in getnzrange(A, col)
                R[rowval[j], 1] = op(R[rowval[j], 1], f(nzval[j]))
            end
        end
    else
        zeroval = f(zero(Tv))
        if isequal(zeroval, zero(Tv))
            # Case where f(0) == 0
            @inbounds for col in axes(A,2)
                @simd for j in getnzrange(A, col)
                    R[rowval[j], 1] += f(nzval[j])
                end
            end
        else
            # Case where f(0) != 0
            rownz = fill(convert(Ti, n), m)
            @inbounds for col in axes(A,2)
                @simd for j in getnzrange(A, col)
                    row = rowval[j]
                    R[row, 1] += f(nzval[j])
                    rownz[row] -= 1
                end
            end
            for i = 1:m
                R[i, 1] += rownz[i]*zeroval
            end
        end
    end
    R
end

# any(pred, A, dims = 1) => mapreduce(pred, |, A, dims = 1)
# final argument `post` is to allow post-mapping each columnar mapreduce
function _mapreducerows!(pred::P, ::typeof(|), R::AbstractMatrix{Bool}, A::SparseMatrixCSCOrColumnSubset{Tv},
                         post::F = identity) where {P, F, Tv}
    nzval = getnzval(A)
    m, n = size(A)
    @inbounds for ii in axes(A,2)
        rng = getnzrange(A, ii)
        len = length(rng)
        # An empty column is trivial
        if len == 0
            R[1, ii] = post(pred(zero(Tv)))
            continue
        end
        # If predicate on zero is true, then sparse column can be short-circuited
        if pred(zero(Tv)) && len < m
            R[1, ii] = post(true)
            continue
        end
        # Otherwise reduce over the stored values
        r = false
        for jj in rng
            r = pred(nzval[jj])
            r && break
        end
        R[1, ii] = post(r)
    end
    return R
end
# all(pred, A, dims = 1) => mapreduce(pred, &, A, dims = 1) == .!mapreduce(!pred, |, A, dims = 1)
_mapreducerows!(pred::P, ::typeof(&), R::AbstractMatrix{Bool},
                A::SparseMatrixCSCOrColumnSubset) where {P} = _mapreducerows!(!pred, |, R, A, !)

# findmax/min and argmax/min methods
# find first zero value in sparse matrix - return linear index in full matrix
# non-structural zeros are identified by `iszero` in line with the sparse constructors.
function _findz(A::AbstractSparseMatrixCSC{Tv,Ti}, rows=axes(A,1), cols=axes(A,2)) where {Tv,Ti}
    rowval = rowvals(A); nzval = nonzeros(A)
    row = 0
    rowmin = rows[1]; rowmax = rows[end]
    allrows = (rows == axes(A,1))
    @inbounds for col in cols
        r1::Int = first(nzrange(A, col))
        r2::Int = last(nzrange(A, col))
        if !allrows && (r1 <= r2)
            r1 += searchsortedfirst(view(rowval, r1:r2), rowmin) - 1
            (r1 <= r2 ) && (r2 = searchsortedlast(view(rowval, r1:r2), rowmax) + r1 - 1)
        end
        row = rowmin
        while (r1 <= r2) && (row == rowval[r1]) && _isnotzero(nzval[r1])
            r1 += 1
            row += 1
        end
        (row <= rowmax) && (return CartesianIndex(row, col))
    end
    return CartesianIndex(0, 0)
end

function _findr(op, A::AbstractSparseMatrixCSC{Tv}, region) where {Tv}
    require_one_based_indexing(A)
    Ti = eltype(keys(A))
    i1 = first(keys(A))
    N = nnz(A)
    L = widelength(A)
    if L == 0
        if prod(map(length, Base.reduced_indices(A, region))) != 0
            throw(ArgumentError("array slices must be non-empty"))
        else
            ri = Base.reduced_indices0(A, region)
            return (zeros(Tv, ri), zeros(Ti, ri))
        end
    end

    colptr = getcolptr(A); rowval = rowvals(A); nzval = nonzeros(A); m = size(A, 1); n = size(A, 2)
    zval = zero(Tv)
    szA = size(A)

    if region == 1 || region == (1,)
        (N == 0) && (return (fill(zval,1,n), fill(i1,1,n)))
        S = Vector{Tv}(undef, n); I = Vector{Ti}(undef, n)
        @inbounds for i = 1 : n
            Sc = zval; Ic = _findz(A, 1:m, i:i)
            if Ic == CartesianIndex(0, 0)
                j = colptr[i]
                Ic = CartesianIndex(rowval[j], i)
                Sc = nzval[j]
            end
            for j = nzrange(A, i)
                if op(nzval[j], Sc)
                    Sc = nzval[j]
                    Ic = CartesianIndex(rowval[j], i)
                end
            end
            S[i] = Sc; I[i] = Ic
        end
        return(reshape(S,1,n), reshape(I,1,n))
    elseif region == 2 || region == (2,)
        (N == 0) && (return (fill(zval,m,1), fill(i1,m,1)))
        S = Vector{Tv}(undef, m)
        I = Vector{Ti}(undef, m)
        @inbounds for row in 1:m
            S[row] = zval; I[row] = _findz(A, row:row, 1:n)
            if I[row] == CartesianIndex(0, 0)
                I[row] = CartesianIndex(row, 1)
                S[row] = A[row,1]
            end
        end
        @inbounds for i = 1 : n, j = nzrange(A, i)
            row = rowval[j]
            if op(nzval[j], S[row])
                S[row] = nzval[j]
                I[row] = CartesianIndex(row, i)
            end
        end
        return (reshape(S,m,1), reshape(I,m,1))
    elseif region == (1,2)
        (N == 0) && (return (fill(zval,1,1), fill(i1,1,1)))
        hasz = nnz(A) != widelength(A)
        Sv = hasz ? zval : nzval[1]
        Iv::(Ti) = hasz ? _findz(A) : i1
        @inbounds for i = 1 : size(A, 2), j = nzrange(A, i)
            if op(nzval[j], Sv)
                Sv = nzval[j]
                Iv = CartesianIndex(rowval[j], i)
            end
        end
        return (fill(Sv,1,1), fill(Iv,1,1))
    else
        throw(ArgumentError("invalid value for region; must be 1, 2, or (1,2)"))
    end
end

_isless_fm(a, b)    =  b == b && ( a != a || isless(a, b) )
_isgreater_fm(a, b) =  b == b && ( a != a || isless(b, a) )

findmin(A::AbstractSparseMatrixCSC{Tv}, region::Union{Integer,Tuple{Integer},NTuple{2,Integer}}) where {Tv} =
    _findr(_isless_fm, A, region)
findmax(A::AbstractSparseMatrixCSC{Tv}, region::Union{Integer,Tuple{Integer},NTuple{2,Integer}}) where {Tv} =
    _findr(_isgreater_fm, A, region)
findmin(A::AbstractSparseMatrixCSC; dims::Union{Nothing,Integer,Tuple{Integer},NTuple{2,Integer}} = nothing) =
    isnothing(dims) ? (r = findmin(A, (1,2)); (r[1][1], r[2][1])) : findmin(A, dims)
findmax(A::AbstractSparseMatrixCSC; dims::Union{Nothing,Integer,Tuple{Integer},NTuple{2,Integer}} = nothing) =
    isnothing(dims) ? (r = findmax(A, (1,2)); (r[1][1], r[2][1])) : findmax(A, dims)

argmin(A::AbstractSparseMatrixCSC) = findmin(A)[2]
argmax(A::AbstractSparseMatrixCSC) = findmax(A)[2]
