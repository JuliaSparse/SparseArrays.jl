# This file is a part of Julia. License is MIT: https://julialang.org/license

## Reductions

# Reductions along a dimension return a dense `Array`, as for dense input. A sparse result
# (issue #43) is opt-in with the keyword `sparse = true`, see `_mapreduce_dim_sparse` below.
# Covers views and adjoints so they reduce like their copy (#377), where Base's `similar`
# would otherwise give a sparse result.
using Base: CommutativeOps, _InitialValue, _mapreduce_start

widelength(inds::CartesianIndices) = prod(Int64.(size(inds)))

# Reductions along a dimension return a dense `Array`, including for views and adjoints
# (#377), where Base's `similar` would give a sparse one.
const _SparseReducible = Union{AbstractSparseArray, SparseMatrixCSCSubArray, SparseVectorOrView}
Base.mapreduce_similar(A::Union{_SparseReducible, AdjOrTrans{<:Any,<:_SparseReducible}}, ::Type{R}, axs) where {R} =
    similar(1:1, R, axs)

# Fold `nzeros` unstored zeros into the accumulator `v0`. Callers that fold many runs of
# zeros can compute `fz = f(zero(T))` once and pass it in. `fz` is `nothing` when the array
# has no unstored entries.
_mapreducezeros(f::F, op::G, ::Type{T}, nzeros::Integer, v0) where {F,G,T} =
    _mapreducezeros(f, op, T, nzeros, v0, nzeros == 0 ? nothing : f(zero(T)))
_mapreducezeros(f::F, op::G, ::Type{T}, nzeros::Integer, v0, fz) where {F,G,T} =
    nzeros == 0 ? v0 : _foldzeros(op, fz, nzeros, v0)
# Fold `zeroval` into `v0` `nzeros` times
function _foldzeros(op::G, zeroval, nzeros::Integer, v0) where {G}
    nzeros == 0 && return v0

    # Reduce over first zero
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
# `nothing` means there are no unstored entries, so there's nothing to fold. We don't
# compute `f(0)` in that case because it might throw.
_foldzeros(op, ::Nothing, nzeros::Integer, v0) = v0
# `f(0)`, or `nothing` if `A` stores every entry
_mappedzero(f::F, A::AbstractArray{T}) where {F,T} = nnz(A) == widelength(A) ? nothing : f(zero(T))

# Fast paths for +, *, min, max and extrema that fold all the unstored zeros in one step:
# multiply `f(0)` by their count, or raise it to that power. `reduce_first` widens small
# integers to `Int` like the dense reductions do, and the result is converted to the type of
# `v0`, which `init` or an earlier value may already have widened.
_mapreducezeros(f::F, op::Union{typeof(Base.add_sum),typeof(+)}, ::Type{T}, nzeros::Integer, v0, fz) where {F,T} =
    nzeros == 0 ? Base.reduce_first(op, v0) : op(_addzeros(_unstored(op, fz, v0), nzeros), v0)
_mapreducezeros(f::F, op::Union{typeof(Base.mul_prod),typeof(*)},::Type{T}, nzeros::Integer, v0, fz) where {F,T} =
    nzeros == 0 ? Base.reduce_first(op, v0) : op(_unstored(op, fz, v0)^nzeros, v0)
_unstored(op, fz, v0) = _widen(Base.reduce_first(op, fz), v0)
_widen(z, v0) = _widen(z, typeof(v0))
_widen(z, ::Type{S}) where S = convert(promote_type(typeof(z), S), z)
# `nzeros` copies of `z` added up, in the type `+` keeps for them: a small integer wraps
# rather than widening to the count's `Int`, as when dense input is summed with `+`
_addzeros(z, nzeros) = z * nzeros
_addzeros(z::Base.BitInteger, nzeros) = z * (nzeros % typeof(z))
_mapreducezeros(f::F, op::Union{typeof(min),typeof(max)}, ::Type{T}, nzeros::Integer, v0, fz) where {F,T} =
    nzeros == 0 ? v0 : op(v0, fz)
_mapreducezeros(f::Base.ExtremaMap, op::typeof(Base._extrema_rf), ::Type{T}, nzeros::Integer, v0, fz) where {T} =
    nzeros == 0 ? v0 : op(v0, fz)

# The product of the stored values `vals` and `nzeros` unstored zeros, for real and complex
# scalars. An integer zero makes the whole product zero. A float zero doesn't, because a
# stored `Inf` or `NaN` has to turn the result into `NaN`. Multiplying the stored values
# together first can overflow to `Inf` even if they're all finite, so if that gives a
# nonzero result we fold them onto the zero one at a time instead.
function _prodstored(f::F, op::G, init, vals, nzeros, ::Type{T}) where {F,G,T}
    nzeros == 0 && return mapreduce(f, op, vals; init)   # no f(0), which might throw
    # The dense fold starts every chain with `op(init, f(a1))`, so `init` sets the type the
    # product is taken in (an `Int` for `init = 1` and `Int8` entries): the zeros' power and
    # the stored values' product take that type too, else both could wrap in the narrower one.
    z = Base.reduce_first(op, f(zero(T)))
    init isa _InitialValue || (z = op(init, z))
    v = z^nzeros
    if !isempty(vals) && !(_iszero(v) && Base.promote_op(f, T) <: Integer)
        r = v * mapreduce(f, op, vals; init)
        v = (_iszero(r) || !_iszero(v)) ? r : mapfoldl(f, op, vals; init=v)
    end
    return init isa _InitialValue ? v : op(init, v)
end

# `any` and `all` require the predicate to return `Bool` (or `missing`), like the dense
# versions do. Without the check, `|` and `&` would silently combine other values.
_boolean(v::Union{Bool,Missing}) = v
_boolean(v) = v::Bool
_booleanpredicate(f::F) where {F} = _boolean ∘ f

### Kernels for sparse matrices and column-subset views

# Indices into `getnzval(A)` of the stored entries in columns `cols`. This is a contiguous
# range unless `A` is a column subset.
_storedpositions(A::AbstractSparseMatrixCSC, cols::AbstractUnitRange) =
    Int(first(getnzrange(A, first(cols)))):Int(last(getnzrange(A, last(cols))))
_storedpositions(A::SparseMatrixCSCView, cols::AbstractUnitRange) =
    Int(first(getnzrange(A, first(cols)))):Int(last(getnzrange(A, last(cols))))
function _storedpositions(A::SparseMatrixCSCColumnSubset, cols::AbstractUnitRange)
    inds = Int[]
    for col in cols
        append!(inds, getnzrange(A, col))
    end
    return inds
end

function Base.mapreduce_kernel(f::F, op::CommutativeOps, A::SparseMatrixCSCOrColumnSubset{T}, init,
                               inds::CartesianIndices{2}) where {F,T}
    n = widelength(inds)
    nnz(A) == 0 && return _mapreducezeros(f, op, T, n-1, _mapreduce_start(f, op, A, init, zero(T)))
    rows, cols = inds.indices
    rowval = getrowval(A)
    nzval = getnzval(A)
    m = length(rows)
    if m == 1
        # In the common case of reducing over just one row we just need to identify
        # whether that row is stored or not
        row = only(rows)
        r = _mapreduce_start(f, op, A, init, A[row, first(cols)])
        nskipped = 0
        for col in cols[begin+1:end]
            js = getnzrange(A, col)
            j = searchsorted(view(rowval, js), row)
            if isempty(j)
                nskipped += 1
            else
                r = op(r, f(nzval[js[j[]]]))
            end
        end
        return _mapreducezeros(f, op, T, nskipped, r)
    elseif m != size(A, 1)
        # We need to constrain our column indexing by the given rows in every col
        r = _mapreduce_start(f, op, A, init, A[first(inds)])
        for col in cols
            js = getnzrange(A, col)
            j1 = searchsortedfirst(view(rowval, js), first(rows) + (col == first(cols)))
            jN = searchsortedlast(view(rowval, js[j1:end]), last(rows))
            if jN > 0
                r = op(r, mapreduce(f, op, view(nzval, js[j1:j1+jN-1]); init))
            end
            r = _mapreducezeros(f, op, T, m-jN-(col==first(cols)), r)
        end
        return r
    else
        # All rows: reduce the stored values in these columns, then fold in the zeros
        nzinds = _storedpositions(A, cols)
        isempty(nzinds) && return _mapreducezeros(f, op, T, n-1, _mapreduce_start(f, op, A, init, zero(T)))
        r = mapreduce(f, op, view(nzval, nzinds); init)
        return _mapreducezeros(f, op, T, n - length(nzinds), r)
    end
end

# Kernel for any other `op`: fold the stored values and the runs of zeros between them in
# column-major order
function Base.mapreduce_kernel(f::F, op, A::SparseMatrixCSCOrColumnSubset{T}, init, inds::CartesianIndices{2}) where {F,T}
    n = widelength(inds)
    nnz(A) == 0 && return _mapreducezeros(f, op, T, n-1, _mapreduce_start(f, op, A, init, zero(T)))
    rows, cols = inds.indices
    rowval = getrowval(A)
    nzval = getnzval(A)
    zeroval = _mappedzero(f, A)   # computed once, not for every run of zeros
    r = _mapreduce_start(f, op, A, init, A[first(inds)])
    for col in cols
        js = getnzrange(A, col)
        j1 = searchsortedfirst(view(rowval, js), first(rows) + (col == first(cols)))
        jN = searchsortedlast(view(rowval, js[j1:end]), last(rows))
        row = first(rows) - 1 + (col==first(cols))
        for j in js[j1:j1+jN-1]
            row, prev_stored_row = rowval[j], row
            if row > prev_stored_row+1
                # implicit zeros between consecutive stored values
                r = _foldzeros(op, zeroval, row-prev_stored_row-1, r)
            end
            r = op(r, f(nzval[j]))
        end
        if last(rows) > row
            # implicit zeros after the last stored value in this column
            r = _foldzeros(op, zeroval, last(rows)-row, r)
        end
    end
    return r
end

# `*` isn't in `CommutativeOps`, but it does commute for real and complex numbers, so all
# the zeros can be folded in as a single power (see `_prodstored`).
function Base.mapreduce_kernel(f::F, op::Union{typeof(*),typeof(Base.mul_prod)}, A::SparseMatrixCSCOrColumnSubset{T}, init,
                               inds::CartesianIndices{2}) where {F,T<:Union{Real,Complex}}
    rows, cols = inds.indices
    n = widelength(inds)
    nzval = getnzval(A)
    if _scalarproduct(f, T)
        if nnz(A) == 0
            return _prodstored(f, op, init, view(nzval, 1:0), n, T)
        elseif rows == axes(A, 1)
            nzinds = _storedpositions(A, cols)
            return _prodstored(f, op, init, view(nzval, nzinds), n - length(nzinds), T)
        end
    end
    # Partial-row blocks, or an `f` whose results may not commute: use the general kernel
    return invoke(Base.mapreduce_kernel,
                  Tuple{Any,Any,SparseMatrixCSCOrColumnSubset,Any,CartesianIndices{2}},
                  f, op, A, init, inds)
end

# Base would split the range into fixed-size chunks, which makes the cost proportional to
# length(A). Give the kernel the whole range instead. It still reduces the stored values
# pairwise, through `mapreduce`.
Base.mapreduce_pairwise(f::F, op::G, A::SparseMatrixCSCOrColumnSubset, init, inds::CartesianIndices{2}) where {F,G} =
    Base.mapreduce_kernel(f, op, A, init, inds)

# Base's `any`/`all` look up every element, and each lookup is a binary search here.
# Reducing with `|` and `&` only touches the stored values and keeps the documented
# `missing` behavior. Empty arrays are handled first, since `mapreduce` may have no
# `reduce_empty` for them.
Base._any(f, A::SparseMatrixCSCOrColumnSubset, ::Colon) =
    iszero(widelength(A)) ? false : mapreduce(_booleanpredicate(f), |, A)
Base._all(f, A::SparseMatrixCSCOrColumnSubset, ::Colon) =
    iszero(widelength(A)) ? true : mapreduce(_booleanpredicate(f), &, A)

### Kernels for sparse vectors, their views, and whole-column views of a matrix

nzvalview(x::SparseVectorOrView) = view(nonzeros(x), 1:nnz(x))

Base.mapreduce_kernel(f::F, op::G, A::SparseVectorOrView, init, inds::CartesianIndices{1}) where {F,G} =
    Base.mapreduce_kernel(f, op, A, init, inds.indices[1])
function Base.mapreduce_kernel(f::F, op::CommutativeOps, A::SparseVectorOrView{T}, init, inds::AbstractUnitRange) where {F,T}
    z = nnz(A)
    if z == 0
        return _mapreducezeros(f, op, T, length(inds)-1, _mapreduce_start(f, op, A, init, zero(T)))
    elseif inds == Base.axes1(A)
        return _mapreducezeros(f, op, T, length(inds)-z, mapreduce(f, op, nzvalview(A); init))
    end
    nzind = getrowval(A)
    nzval = getnzval(A)
    m = length(inds)
    r = _mapreduce_start(f, op, A, init, A[first(inds)])
    j1 = searchsortedfirst(nzind, first(inds) + 1)
    jN = searchsortedlast(view(nzind, j1:z), last(inds))
    for j in j1:j1+jN-1
        r = op(r, f(nzval[j]))
    end
    # Apart from the first element (used by `_mapreduce_start`) and the jN stored values,
    # everything in `inds` is a zero
    return _mapreducezeros(f, op, T, m-jN-1, r)
end

function Base.mapreduce_kernel(f::F, op, A::SparseVectorOrView{T}, init, inds::AbstractUnitRange) where {F,T}
    z = nnz(A)
    if z == 0
        return _mapreducezeros(f, op, T, length(inds)-1, _mapreduce_start(f, op, A, init, zero(T)))
    end
    nzind = getrowval(A)
    nzval = getnzval(A)
    zeroval = _mappedzero(f, A)   # computed once, not for every run of zeros
    r = _mapreduce_start(f, op, A, init, A[first(inds)])
    j1 = searchsortedfirst(nzind, first(inds) + 1)
    jN = searchsortedlast(view(nzind, j1:z), last(inds))
    row = first(inds)
    for j in j1:j1+jN-1
        row, prev_stored_row = nzind[j], row
        if row > prev_stored_row+1
            # implicit zeros between consecutive stored values
            r = _foldzeros(op, zeroval, row-prev_stored_row-1, r)
        end
        r = op(r, f(nzval[j]))
    end
    if last(inds) > row
        # implicit zeros after the last stored value
        r = _foldzeros(op, zeroval, last(inds)-row, r)
    end
    return r
end

function Base.mapreduce_kernel(f::F, op::Union{typeof(*),typeof(Base.mul_prod)}, A::SparseVectorOrView{T}, init,
                               inds::AbstractUnitRange) where {F,T<:Union{Real,Complex}}
    _scalarproduct(f, T) || return invoke(Base.mapreduce_kernel,
                                          Tuple{Any,Any,SparseVectorOrView,Any,AbstractUnitRange}, f, op, A, init, inds)
    z = nnz(A)
    nzval = getnzval(A)
    if z == 0
        return _prodstored(f, op, init, view(nzval, 1:0), length(inds), T)
    elseif inds == Base.axes1(A)
        return _prodstored(f, op, init, nzvalview(A), length(inds) - z, T)
    end
    nzind = getrowval(A)
    j1 = searchsortedfirst(nzind, first(inds))
    jN = searchsortedlast(view(nzind, j1:z), last(inds))
    return _prodstored(f, op, init, view(nzval, j1:j1+jN-1), length(inds) - jN, T)
end

Base.mapreduce_pairwise(f::F, op::G, A::SparseVectorOrView, init, inds::AbstractUnitRange) where {F,G} =
    Base.mapreduce_kernel(f, op, A, init, inds)
Base.mapreduce_pairwise(f::F, op::G, A::SparseVectorOrView, init, inds::CartesianIndices{1}) where {F,G} =
    Base.mapreduce_kernel(f, op, A, init, inds.indices[1])

Base._any(f, A::SparseVectorOrView, ::Colon) =
    iszero(length(A)) ? false : mapreduce(_booleanpredicate(f), |, A)
Base._all(f, A::SparseVectorOrView, ::Colon) =
    iszero(length(A)) ? true : mapreduce(_booleanpredicate(f), &, A)

### Reductions along a dimension

# Can the row-wise pass fold each row's zeros in at the end, out of order?
_rowwise_commutes(f::F, op, ::Type{T}) where {F,T} = op isa CommutativeOps ||
    (op isa Union{typeof(*),typeof(Base.mul_prod)} && _scalarproduct(f, T))
# Does `*` commute for the values `f` returns? It does for real and complex numbers. We ask
# inference because `f` could return a string or a matrix.
_scalarproduct(f::F, ::Type{T}) where {F,T} = Base.promote_op(f, T) <: Union{Real,Complex}

# Reduce along `dims = 2` in one column-major pass with an accumulator per row, which is
# O(nnz) instead of O(m·n·log(nnz)). If `_rowwise_commutes` allows it, each row's zeros are
# folded in at the end. Otherwise they're folded in where they occur, so each row is still
# reduced in order and any `op` works.
function Base.mapreducedim_rowmajor(f::F, op::OP, A::SparseMatrixCSCOrColumnSubset{T}, init,
                                    is_inner_dim, inner, outer, alloc,
                                    enforce_pairwise::Bool=true) where {F,OP,T}
    if is_inner_dim != (false, true)
        return invoke(Base.mapreducedim_rowmajor,
                      Tuple{Any,Any,Any,Any,Any,Any,Any,Any,Any},
                      f, op, A, init, is_inner_dim, inner, outer, alloc, enforce_pairwise)
    end
    rowval = getrowval(A)
    nzval = getnzval(A)
    cols = inner.indices[2]
    c1 = first(cols)
    m, ncols = size(A, 1), length(cols)
    o2 = only(outer.indices[2])  # the (possibly offset) singleton output column
    # Seed each row from its entry in the first column. `outer` visits the rows in order, so
    # one cursor can walk that column's stored values.
    js1 = getnzrange(A, c1)
    j = Ref(Int(first(js1)))
    stop = Int(last(js1)) + 1
    seeds = Base.Generator(outer) do o
        i = o[1]
        v = zero(T)
        jj = j[]
        if jj < stop && rowval[jj] == i
            v = nzval[jj]
            j[] = jj + 1
        end
        _mapreduce_start(f, op, A, init, v)
    end
    R = alloc(seeds)
    if _rowwise_commutes(f, op, T)
        # Fold in the stored values of the remaining columns and count how many zeros each
        # row has. The zeros get folded in afterwards.
        rownz = fill(ncols - 1, m)
        for col in cols[begin+1:end]
            for jj in getnzrange(A, col)
                row = rowval[jj]
                R[row, o2] = op(R[row, o2], f(nzval[jj]))
                rownz[row] -= 1
            end
        end
        if !all(iszero, rownz)
            fz = f(zero(T))   # computed once for all rows
            for row in axes(A, 1)
                R[row, o2] = _mapreducezeros(f, op, T, rownz[row], R[row, o2], fz)
            end
        end
    else
        # Fold each row in order: before each stored value, fold in the zeros since that
        # row's previous stored value
        nstored = sum(col -> length(getnzrange(A, col)), cols; init=0)
        zeroval = nstored == m * ncols ? nothing : f(zero(T))
        lastk = fill(1, m)   # index in `cols` of the last entry folded into each row
        for (k, col) in enumerate(cols)
            k == 1 && continue
            for jj in getnzrange(A, col)
                row = rowval[jj]
                R[row, o2] = _foldzeros(op, zeroval, k - lastk[row] - 1, R[row, o2])
                R[row, o2] = op(R[row, o2], f(nzval[jj]))
                lastk[row] = k
            end
        end
        for row in axes(A, 1)
            lastk[row] == ncols || (R[row, o2] = _foldzeros(op, zeroval, ncols - lastk[row], R[row, o2]))
        end
    end
    return R
end

# Base picks this naive strategy when inference can't show that the accumulator type stays
# fixed. It runs that check starting from an empty reduction, which infers as `Union{}` for
# ops with no `reduce_empty` (`max`, `min`, `extrema`, ...), so all of those end up here.
# For a row-wise reduction of a CSC matrix that's O(m·n·log(nnz)). Redo the check starting
# from an actual element, which is how a nonempty reduction starts, and use the single-pass
# row-major method if the type is concrete and stable.
function Base.mapreducedim_naive(f::F, op::OP, A::SparseMatrixCSCOrColumnSubset{T}, init,
                                 is_inner_dim, inner, outer, alloc) where {F,OP,T}
    if is_inner_dim == (false, true)
        S = Base.promote_op(a1 -> _mapreduce_start(f, op, A, init, a1), T)
        R = Base.promote_op((s, a) -> op(s, f(a)), S, T)
        if isconcretetype(S) && R <: S
            return Base.mapreducedim_rowmajor(f, op, A, init, is_inner_dim, inner, outer, alloc)
        end
    end
    return invoke(Base.mapreducedim_naive, Tuple{Any,Any,Any,Any,Any,Any,Any,Any},
                  f, op, A, init, is_inner_dim, inner, outer, alloc)
end

# LinearAlgebra already sends commutative whole-array reductions of `A'` to `parent(A)`.
# Do the same for reductions along a dimension (and in-place ones, where `dims` is
# `nothing`) by reducing the parent along the other dimension. Reducing over both dimensions
# has to visit the elements in the order of `A'`, so that case makes a copy.
const _NotColon = Union{Integer, Tuple, AbstractVector, Nothing}
function Base.mapreducedim(f, op, A::AdjOrTrans{<:Any,<:SparseMatrixCSCOrColumnSubset}, init, dims::_NotColon,
                           alloc=Base._MapReduceAllocator(A))
    g = wrapperop(A)
    if alloc isa Base._MapReduceInPlace
        R = alloc.A
        length(R) == 1 && return Base.mapreducedim(f, op, copy(A), init, dims, alloc)
        Base.mapreducedim(f ∘ g, op, parent(A), init, dims, Base._MapReduceInPlace(_switch_dim12(R), op, alloc.update))
        return R
    end
    1 in dims && 2 in dims && return Base.mapreducedim(f, op, copy(A), init, dims, alloc)
    return permutedims(Base.mapreducedim(f ∘ g, op, parent(A), init, map(_switch_dim12, dims)), (2, 1))
end
_switch_dim12(d) = d == 1 ? 2 : d == 2 ? 1 : d
# `R` with its first two dimensions swapped, to receive the parent's reduction
_switch_dim12(R::AbstractArray{<:Any,0}) = R
_switch_dim12(R::AbstractVector) = reshape(R, 1, length(R))
_switch_dim12(R::AbstractArray) = PermutedDimsArray(R, (2, 1, ntuple(Base.Fix1(+, 2), ndims(R) - 2)...))

# The parent of a row vector is a sparse vector. Reducing along dim 2 reduces the parent;
# reducing along dim 1 just maps each element.
function Base.mapreducedim(f, op, A::AdjOrTransSparseVectorOrView, init, dims::_NotColon,
                           alloc=Base._MapReduceAllocator(A))
    g = wrapperop(A)
    if alloc isa Base._MapReduceInPlace
        R = alloc.A
        Base.mapreducedim(f ∘ g, op, parent(A), init, dims, Base._MapReduceInPlace(_switch_dim12(R), op, alloc.update))
        return R
    end
    2 in dims && return fill(Base.mapreducedim(f ∘ g, op, parent(A), init, :), 1, 1)
    # Along dim 1 (or any dim > 2) each element is its own slice. An empty row has nothing
    # to seed the result with, so this throws like the dense method does.
    isempty(A) && _mapreduce_start(f, op, A, init)
    r = map(x -> _mapreduce_start(f, op, A, init, g(x)), parent(A))
    return reshape(Array(r), 1, length(r))
end

# With `sparse = true` (#43), `sum(A; dims = 2, sparse = true)` and the other reductions
# along a dimension return a `SparseMatrixCSC` (a `SparseVector` for vector input). It
# stores an entry for each slice of `A` that stores something, or for every slice if an
# empty slice reduces to a nonzero (e.g. when `f(0) != 0`), and takes
# O(nnz(A) + length(result)) time. Base's `sum`, `prod`, `maximum` and `minimum` pass
# unknown keywords on to `mapreduce`, so the `mapreduce` method covers them. `any`, `all`
# and `count` don't, so they get their own methods. Views without compressed storage are
# copied first.
for T in (:SparseMatrixCSCOrSubArray, :(AdjOrTrans{<:Any,<:SparseMatrixCSCOrColumnSubset}), :SparseVectorOrView,
          :AdjOrTransSparseVectorOrView)
    @eval Base.@constprop :aggressive function Base.mapreduce(f, op, A::$T; dims=:, init=Base._InitialValue(), sparse::Bool=false)
        sparse || return Base.mapreducedim(f, op, _compressed(A), init, dims)
        dims === (:) && throw(ArgumentError("a sparse result needs a reduction along a dimension, pass `dims`"))
        return _mapreduce_dim_sparse(f, op, init, _compressed(A), dims)
    end
    # Base's `sum` of a `Bool` array calls `reduce` with `add_sum`
    @eval Base.@constprop :aggressive Base.reduce(op, A::$T; dims=:, init=Base._InitialValue(), sparse::Bool=false) =
        mapreduce(identity, op, A; dims, init, sparse)
    for (fname, _fname, op) in ((:any, :_any, :(Base.or_any)), (:all, :_all, :(Base.and_all)))
        @eval begin
            Base.$fname(A::$T; dims=:, sparse::Bool=false) = Base.$fname(identity, A; dims, sparse)
            Base.@constprop :aggressive Base.$fname(f, A::$T; dims=:, sparse::Bool=false) =
                sparse ? mapreduce(f, $op, A; dims, sparse) : Base.$_fname(f, _compressed(A), dims)
        end
    end
    @eval begin
        Base.count(A::$T; dims=:, init=0, sparse::Bool=false) = count(identity, A; dims, init, sparse)
        Base.@constprop :aggressive Base.count(f, A::$T; dims=:, init=0, sparse::Bool=false) =
            sparse ? mapreduce(Base._bool(f), Base.add_sum, A; dims, init, sparse) : Base._count(f, _compressed(A), dims, init)
    end
end

# `mapreduce` of a binary `f` over two sparse arrays of one shape (#33), without the array
# `map(f, A, B)` that Base's method reduces. `op` is taken to be associative, and to be
# commutative only where it is known to be, so the mapped values keep the order of the elements.
# A reduction along a dimension, an empty array and a shape mismatch are left to Base's method.
for T in (:SparseVectorOrView, :SparseMatrixCSCOrColumnSubset)
    @eval function Base.mapreduce(f::F, op::G, A::$T, B::$T; dims=:, init=Base._InitialValue(), kw...) where {F,G}
        (dims === (:) && isempty(kw) && size(A) == size(B) && !isempty(A)) ||
            return reduce(op, map(f, A, B); dims, init, kw...)
        return _commutes(f, op, eltype(A), eltype(B)) ? _mapreduce_unordered(f, op, init, A, B) :
                                                         _mapreduce_ordered(f, op, init, A, B)
    end
end
# A product commutes when its factors are real or complex numbers, which is asked of inference:
# the answer picks between two kernels that agree for such factors.
_commutes(f, ::Union{typeof(+),typeof(Base.add_sum),typeof(max),typeof(min)}, ::Type, ::Type) = true
_commutes(f::F, ::Union{typeof(*),typeof(Base.mul_prod)}, ::Type{Ta}, ::Type{Tb}) where {F,Ta,Tb} =
    Base.promote_op(f, Ta, Tb) <: Union{Real,Complex}
_commutes(f, op, ::Type, ::Type) = false

# An `op` not known to commute is folded over the positions in their order. A zero is taken
# only where an entry is not stored, as an array that stores every entry need not have one.
function _mapreduce_ordered(f::F, op::G, init, A::AbstractArray{Ta}, B::AbstractArray{Tb}) where {F,G,Ta,Tb}
    rf = Base.BottomRF(op)
    Ai, Av, Bi, Bv = getrowval(A), getnzval(A), getrowval(B), getnzval(B)
    m = size(A, 1)
    v = init
    unstored = Int64(0)   # positions that neither array stores, since the last one folded
    @inbounds for col in axes(A, 2)
        ra, rb = getnzrange(A, col), getnzrange(B, col)
        ia, ib = first(ra), first(rb)
        ea, eb = last(ra), last(rb)
        prev = 0
        while ia <= ea && ib <= eb
            row = Int(min(Ai[ia], Bi[ib]))
            x, ia, ib = _merge_step(f, Ai, Av, ia, Bi, Bv, ib)
            v = rf(_foldunstored(f, rf, v, Ta, Tb, unstored + (row - prev - 1)), x)
            unstored, prev = Int64(0), row
        end
        while ia <= ea
            row = Int(Ai[ia])
            v = rf(_foldunstored(f, rf, v, Ta, Tb, unstored + (row - prev - 1)), f(Av[ia], zero(Tb)))
            unstored, prev = Int64(0), row
            ia += 1
        end
        while ib <= eb
            row = Int(Bi[ib])
            v = rf(_foldunstored(f, rf, v, Ta, Tb, unstored + (row - prev - 1)), f(zero(Ta), Bv[ib]))
            unstored, prev = Int64(0), row
            ib += 1
        end
        unstored += m - prev
    end
    v = _foldunstored(f, rf, v, Ta, Tb, unstored)
    v isa Base._InitialValue && throw(ArgumentError("reducing over an empty collection is not allowed"))
    return v
end
# `v` folded with `f` at `n` consecutive positions that neither array stores. The mapped values
# there are all one value, so regrouping them, which an associative `op` allows, folds the
# run in O(log(n)) calls of `op`.
@inline function _foldunstored(f::F, rf, v, ::Type{Ta}, ::Type{Tb}, n) where {F,Ta,Tb}
    n == 0 && return v
    z = f(zero(Ta), zero(Tb))
    v = rf(v, z)
    n == 1 && return v
    op = rf.rf
    p = r = _widen(Base.reduce_first(op, z), v)
    n -= 2
    while n > 0
        isodd(n) && (r = op(r, p))
        n >>= 1
        n > 0 && (p = op(p, p))
    end
    return op(v, r)
end

# Where `op` commutes the merge feeds two accumulators in turn, as a single chain of `op` is
# bound by its latency, which is long for `max` and `min` of floats, and the positions
# neither array stores are folded in last.
function _mapreduce_unordered(f::F, op::G, init, A::AbstractArray{Ta}, B::AbstractArray{Tb}) where {F,G,Ta,Tb}
    rf = Base.BottomRF(op)
    Ai, Av, Bi, Bv = getrowval(A), getnzval(A), getrowval(B), getnzval(B)
    v, w = init, Base._InitialValue()
    stored = 0   # positions that either array stores
    @inbounds for col in axes(A, 2)
        ra, rb = getnzrange(A, col), getnzrange(B, col)
        ia, ib = first(ra), first(rb)
        ea, eb = last(ra), last(rb)
        while ia <= ea && ib <= eb
            x, ia, ib = _merge_step(f, Ai, Av, ia, Bi, Bv, ib)
            v = rf(v, x)
            stored += 1
            (ia <= ea && ib <= eb) || break
            x, ia, ib = _merge_step(f, Ai, Av, ia, Bi, Bv, ib)
            v, w = _fold_second(rf, v, w, x, init)
            stored += 1
        end
        stored += (ea - ia + 1) + (eb - ib + 1)
        while ia < ea
            v = rf(v, f(Av[ia], zero(Tb)))
            v, w = _fold_second(rf, v, w, f(Av[ia+1], zero(Tb)), init)
            ia += 2
        end
        ia == ea && (v = rf(v, f(Av[ia], zero(Tb))))
        while ib < eb
            v = rf(v, f(zero(Ta), Bv[ib]))
            v, w = _fold_second(rf, v, w, f(zero(Ta), Bv[ib+1]), init)
            ib += 2
        end
        ib == eb && (v = rf(v, f(zero(Ta), Bv[ib])))
    end
    w isa Base._InitialValue || (v = v isa Base._InitialValue ? w : op(v, w))
    nzeros = prod(Int64.(size(A))) - stored
    if v isa Base._InitialValue   # nothing stored and no `init`: seed with one unstored position
        v = rf(v, f(zero(Ta), zero(Tb)))
        nzeros -= 1
    end
    return _mapreducezeros(z -> f(z, zero(Tb)), op, Ta, nzeros, v)
end
# With an `init` there is one accumulator: `init` may have widened it, as for small integers,
# and a second one seeded with a mapped value would not be.
@inline _fold_second(rf, v, w, x, ::Base._InitialValue) = v, rf(w, x)
@inline _fold_second(rf, v, w, x, init) = rf(v, x), w
# `f` at the next position that `A` or `B` stores, and the indices past it
Base.@propagate_inbounds function _merge_step(f::F, Ai, Av::AbstractVector{Ta}, ia, Bi, Bv::AbstractVector{Tb}, ib) where {F,Ta,Tb}
    ja, jb = Ai[ia], Bi[ib]
    if ja == jb
        return f(Av[ia], Bv[ib]), ia + 1, ib + 1
    elseif ja < jb
        return f(Av[ia], zero(Tb)), ia + 1, ib
    else
        return f(zero(Ta), Bv[ib]), ia, ib + 1
    end
end

### The sparse result of a reduction along a dimension

# The slices are reduced the way Base's dense result is: seeded with `init` when given and
# otherwise with `mapreduce_first` of their first element, with the entries a slice does not
# store folded in through `_mapreducezeros`.
_seed(f, op, ::Base._InitialValue, x) = Base.mapreduce_first(f, op, x)
_seed(f, op, init, x) = op(init, f(x))
# The element type of the dense result, which is the type of each slice's seed. We seed from
# `zero(T)` unless `A` stores every entry, in which case `T` might not have a zero, so we
# use a stored value instead.
_reduced_eltype(f, op, init, A::AbstractArray{T}) where T =
    typeof(_seed(f, op, init, nnz(A) == length(A) > 0 ? _firststored(A) : zero(T)))
_firststored(A::AbstractVector) = first(nonzeros(A))
_firststored(A::AbstractMatrix) = getnzval(A)[first(getnzrange(A, 1))]
# the reduction of a slice with no entries at all, as Base initializes the dense result
_reduced_empty(f, op, ::Base._InitialValue, ::Type{T}) where T = Base.mapreduce_empty(f, op, T)
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
# the single row is the parent: reduced along the columns it folds the parent, and reduced
# along the row alone there is an entry per element
function _mapreduce_dim_sparse(f, op, init, A::AdjOrTransSparseVectorOrView, dims)
    r = _mapreduce_dim_sparse(f ∘ wrapperop(A), op, init, parent(A), map(_switch_dim12, dims))
    return _rowmatrix(length(r), nonzeroinds(r), nonzeros(r))
end

function _mapreduce_dim_sparse(f, op, init, A::SparseVectorOrView{T,Ti}, dims) where {T,Ti}
    Base.reduced_indices(A, dims)   # validates `dims`
    # An empty array gives the empty value, or throws, like a dense array
    isempty(A) && _reduced_empty(f, op, init, T)
    Tr = _sparse_reduced_eltype(f, op, init, A)
    # a dimension beyond 1: every entry is a slice of its own
    1 in dims || return convert(SparseVector{Tr,Ti}, map(x -> _seed(f, op, init, x), A isa SubArray ? copy(A) : A))
    v = isempty(A) ? _reduced_empty(f, op, init, T) : _seed(identity, op, init, mapreduce(f, op, A))
    return nnz(A) > 0 || !isequal(v, zero(Tr)) ? SparseVector(1, Ti[1], Tr[v]) : spzeros(Tr, Ti, 1)
end

function _mapreduce_dim_sparse(f, op, init, A::SparseMatrixCSCOrColumnSubset{T,Ti}, dims) where {T,Ti}
    m, n = size(A)
    rm, rn = map(length, Base.reduced_indices(A, dims))   # also validates `dims`
    # An empty array gives the empty value, or throws, like a dense array
    isempty(A) && _reduced_empty(f, op, init, T)
    Tr = _sparse_reduced_eltype(f, op, init, A)
    if rm == rn == 1
        R = spzeros(Tr, Ti, 1, 1)
        v = isempty(A) ? _reduced_empty(f, op, init, T) : _seed(identity, op, init, mapreduce(f, op, A))
        if nnz(A) > 0 || !isequal(v, zero(Tr))
            push!(getrowval(R), 1)
            push!(getnzval(R), v)
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
    Rcolptr, Rrowval, Rnzval = getcolptr(R), getrowval(R), getnzval(R)
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
    Rcolptr, Rrowval, Rnzval = getcolptr(R), getrowval(R), getnzval(R)
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

### findmin/findmax and argmin/argmax

# Fold over the stored values in column-major order, then compare the result with the first
# entry not stored in each slice. Base only replaces the current best with a strictly better
# value, so ties go to the earliest position, and the first such entry is the only one that
# can win.

# find the first entry of a sparse matrix that is not stored, returning its CartesianIndex,
# or CartesianIndex(0, 0) when there is none. A stored zero is not one of them: it may be a
# `-0.0`, which `isless` tells from `zero(Tv)`, and `f` may tell apart too, so it is compared
# like any other stored value, at its own position.
function _findz(A::SparseMatrixCSCOrColumnSubset{Tv,Ti}, rows=axes(A,1), cols=axes(A,2)) where {Tv,Ti}
    rowval = getrowval(A)
    row = 0
    rowmin = rows[1]; rowmax = rows[end]
    allrows = (rows == axes(A,1))
    @inbounds for col in cols
        rng = getnzrange(A, col)
        r1::Int = first(rng)
        r2::Int = last(rng)
        if !allrows && (r1 <= r2)
            r1 += searchsortedfirst(view(rowval, r1:r2), rowmin) - 1
            (r1 <= r2) && (r2 = searchsortedlast(view(rowval, r1:r2), rowmax) + r1 - 1)
        end
        row = rowmin
        while (r1 <= r2) && (row == rowval[r1])
            r1 += 1
            row += 1
        end
        (row <= rowmax) && (return CartesianIndex(row, col))
    end
    return CartesianIndex(0, 0)
end

# Pick between the best stored value `(fm, im)` and the first zero `(fz, zi)`. On a tie,
# whichever comes first in column-major order wins.
function _fm_merge(cmp::C, fm, im, fz, zi) where {C}
    if (zi[2], zi[1]) < (im[2], im[1])
        return cmp(fz, fm) ? (fm, im) : (fz, zi)
    else
        return cmp(fm, fz) ? (fz, zi) : (fm, im)
    end
end

# Whole array: fold the stored values in column-major order
function _findminmax(f::F, cmp::C, A::SparseMatrixCSCOrColumnSubset{T}) where {F,C,T}
    rowval = getrowval(A); nzval = getnzval(A)
    best = nothing
    @inbounds for col in axes(A, 2), j in getnzrange(A, col)
        fx = f(nzval[j])
        if best === nothing || cmp(best[1], fx)
            best = (fx, CartesianIndex(rowval[j], col))
        end
    end
    nnz(A) == widelength(A) && return best::Tuple
    zi = _findz(A)
    fz = f(zero(T))
    best === nothing && return (fz, zi)
    return _fm_merge(cmp, best[1], best[2], fz, zi)
end

function _findminmax_dims(cmp::C, A::SparseMatrixCSCOrColumnSubset{T}, region) where {C,T}
    m, n = size(A)
    rowval = getrowval(A); nzval = getnzval(A)
    axs = Base.reduced_indices(A, region)
    rval = Base.mapreduce_similar(A, eltype(A), axs)
    rind = Base.mapreduce_similar(A, keytype(A), axs)
    if region == 1 || region == (1,)
        @inbounds for col in axes(A, 2)
            js = getnzrange(A, col)
            best = nothing
            for j in js
                fx = nzval[j]
                if best === nothing || cmp(best[1], fx)
                    best = (fx, CartesianIndex(rowval[j], col))
                end
            end
            if length(js) < m
                zi = _findz(A, axes(A, 1), col:col)
                best = best === nothing ? (zero(T), zi) :
                       _fm_merge(cmp, best[1], best[2], zero(T), zi)
            end
            rval[1, col] = best[1]
            rind[1, col] = best[2]
        end
    else # region == 2 || region == (2,)
        has = fill(false, m)
        rownnz = fill(0, m)
        @inbounds for col in axes(A, 2), j in getnzrange(A, col)
            row = rowval[j]
            fx = nzval[j]
            rownnz[row] += 1
            if !has[row] || cmp(rval[row, 1], fx)
                rval[row, 1] = fx
                rind[row, 1] = CartesianIndex(row, col)
                has[row] = true
            end
        end
        @inbounds for row in axes(A, 1)
            rownnz[row] == n && continue  # no zeros in this row
            zi = _findz(A, row:row, axes(A, 2))
            if !has[row]
                rval[row, 1] = zero(T)
                rind[row, 1] = zi
            else
                v, i = _fm_merge(cmp, rval[row, 1], rind[row, 1], zero(T), zi)
                rval[row, 1] = v
                rind[row, 1] = i
            end
        end
    end
    return (rval, rind)
end

# `_findmin`/`_findmax` implement `findmin`, `findmax`, `argmin` and `argmax`, with or
# without `dims`, for CSC matrices and views of a column subset, whose stored entries are
# those of their columns in the parent. Empty arrays, `dims` with an `f` other than
# `identity`, and unusual `dims` use Base's generic methods, which also throw the errors
# for empty input.
_fm_supported_region(region) =
    region == 1 || region == (1,) || region == 2 || region == (2,) || region == (1, 2)

function _findminmax_region(cmp::C, f::F, A::SparseMatrixCSCOrColumnSubset, region) where {C,F}
    if region == (1, 2)
        v, i = _findminmax(f, cmp, A)
        axs = Base.reduced_indices(A, region)
        rval = Base.mapreduce_similar(A, eltype(A), axs)
        rind = Base.mapreduce_similar(A, keytype(A), axs)
        rval[1, 1] = v
        rind[1, 1] = i
        return (rval, rind)
    end
    return _findminmax_dims(cmp, A, region)
end

function Base._findmin(f, A::SparseMatrixCSCOrColumnSubset, region::Colon)
    isempty(A) && return invoke(Base._findmin, Tuple{Any,Any,Colon}, f, A, region)
    return _findminmax(f, Base.isgreater, A)
end
function Base._findmax(f, A::SparseMatrixCSCOrColumnSubset, region::Colon)
    isempty(A) && return invoke(Base._findmax, Tuple{Any,Any,Colon}, f, A, region)
    return _findminmax(f, Base.isless, A)
end
function Base._findmin(f, A::SparseMatrixCSCOrColumnSubset, region)
    (isempty(A) || f !== identity || !_fm_supported_region(region)) &&
        return invoke(Base._findmin, Tuple{Any,Any,Any}, f, A, region)
    return _findminmax_region(Base.isgreater, f, A, region)
end
function Base._findmax(f, A::SparseMatrixCSCOrColumnSubset, region)
    (isempty(A) || f !== identity || !_fm_supported_region(region)) &&
        return invoke(Base._findmax, Tuple{Any,Any,Any}, f, A, region)
    return _findminmax_region(Base.isless, f, A, region)
end
