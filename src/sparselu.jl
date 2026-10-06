# This file is a part of Julia. License is MIT: https://julialang.org/license

# Solves with a sparse right-hand side, in time proportional to the arithmetic they do:
# the triangular solve and the left-looking LU factorization of J. R. Gilbert and
# T. Peierls, "Sparse partial pivoting in time proportional to arithmetic operations",
# SIAM J. Sci. Stat. Comput. 9(5), 1988, pp. 862-874, with the symmetric pruning of
# S. C. Eisenstat and J. W. H. Liu, "Exploiting structural symmetry in a sparse partial
# pivoting code", SIAM J. Sci. Comput. 14(1), 1993, pp. 253-257, applied to the diagonal
# blocks of the block triangular form that `dmperm` finds.

# Sort the row indices of every column of the CSC arrays of a matrix with `m` rows, by a
# counting sort through the rows: O(nnz + m).
function _sortcolumns!(m::Int, colptr::Vector{Int}, rowval::Vector{Int}, nzval::Vector)
    n = length(colptr) - 1
    nz = colptr[n + 1] - 1
    rowptr = zeros(Int, m + 1)
    @inbounds for k in 1:nz
        rowptr[rowval[k] + 1] += 1
    end
    rowptr[1] = 1
    @inbounds for i in 1:m
        rowptr[i + 1] += rowptr[i]
    end
    cols = Vector{Int}(undef, nz)
    vals = similar(nzval, nz)
    @inbounds for j in 1:n, k in colptr[j]:(colptr[j + 1] - 1)
        i = rowval[k]
        d = rowptr[i]
        rowptr[i] = d + 1
        cols[d] = j
        vals[d] = nzval[k]
    end
    # rowptr[i] is now one past the end of row i
    next = colptr[1:n]
    start = 1
    @inbounds for i in 1:m
        for d in start:(rowptr[i] - 1)
            j = cols[d]
            k = next[j]
            next[j] = k + 1
            rowval[k] = i
            nzval[k] = vals[d]
        end
        start = rowptr[i]
    end
    return nothing
end

## Triangular solve with a sparse right-hand side

# X = T \ B for the lower or upper triangle T of the square `A`, with a unit diagonal
# when `unit`; entries of `A` outside that triangle are ignored. The nonzeros of a column
# of X are the vertices reachable from those of the column of B in the graph of T, found
# by depth-first search; reverse postorder is a topological order, so the columns of T
# are applied in that order, to a dense vector touched at those positions only (Gilbert
# and Peierls, Lemma 1). After a pass over the columns that locates the diagonal, each
# column of B costs its arithmetic plus the sort of its pattern.
function _sptrisolve(A::SparseMatrixCSCOrView, lower::Bool, unit::Bool,
                     B::SparseMatrixCSCOrView, ::Type{T}, ::Type{Ti}) where {T,Ti}
    require_one_based_indexing(A, B)
    n = checksquare(A)
    size(B, 1) == n ||
        throw(DimensionMismatch(lazy"the matrix has $n rows, but the right-hand side has $(size(B, 1))"))
    Ai = rowvals(A)
    Ax = nonzeros(A)
    Bi = rowvals(B)
    Bx = nonzeros(B)
    # the strict triangle of column j is Ai[from[j]:to[j]]; a stored diagonal is next to
    # it, before a lower triangle and after an upper one
    from = Vector{Int}(undef, n)
    to = Vector{Int}(undef, n)
    @inbounds for j in 1:n
        r = nzrange(A, j)
        d = searchsortedfirst(Ai, j, first(r), last(r), Forward)
        stored = d <= last(r) && Ai[d] == j
        if !unit && (!stored || _iszero(Ax[d]))
            throw(LinearAlgebra.SingularException(j))
        end
        if lower
            from[j] = stored ? d + 1 : d
            to[j] = last(r)
        else
            from[j] = first(r)
            to[j] = d - 1
        end
    end
    k = size(B, 2)
    x = zeros(T, n)
    mark = zeros(Int, n)
    order = Vector{Int}(undef, n)
    nodes = Vector{Int}(undef, n)
    pos = Vector{Int}(undef, n)
    Xp = Vector{Ti}(undef, k + 1)
    Xi = Ti[]
    Xx = T[]
    @inbounds for c in 1:k
        Xp[c] = length(Xi) + 1
        top = n + 1
        for idx in nzrange(B, c)
            i = Int(Bi[idx])
            x[i] = Bx[idx]
            mark[i] == c && continue
            mark[i] = c
            head = 1
            nodes[1] = i
            pos[1] = from[i]
            while head > 0
                j = nodes[head]
                p = pos[head]
                stop = to[j]
                descended = false
                while p <= stop
                    r = Int(Ai[p])
                    p += 1
                    if mark[r] != c
                        mark[r] = c
                        pos[head] = p
                        head += 1
                        nodes[head] = r
                        pos[head] = from[r]
                        descended = true
                        break
                    end
                end
                if !descended
                    head -= 1
                    top -= 1
                    order[top] = j
                end
            end
        end
        for t in top:n
            j = order[t]
            xj = unit ? x[j] : Ax[lower ? from[j] - 1 : to[j] + 1] \ x[j]
            x[j] = xj
            for p in from[j]:to[j]
                x[Ai[p]] -= Ax[p] * xj
            end
        end
        reach = view(order, top:n)
        sort!(reach)
        for j in reach
            push!(Xi, j)
            push!(Xx, x[j])
            x[j] = zero(T)
        end
    end
    Xp[k + 1] = length(Xi) + 1
    return SparseMatrixCSC{T,Ti}(n, k, Xp, Xi, Xx)
end

## LU factorization

"""
    SparseArrays.SparseLU{Tv,Ti} <: LinearAlgebra.Factorization{Tv}

The LU factorization of a square sparse matrix that [`SparseArrays.sparselu`](@ref)
computes. For `F = SparseArrays.sparselu(A)`,

| Property | Description                                  |
|:-------- |:-------------------------------------------- |
| `F.L`    | unit lower triangular factor                 |
| `F.U`    | upper triangular factor                      |
| `F.p`    | row permutation                              |
| `F.q`    | column permutation                           |

satisfy `F.L * F.U ≈ A[F.p, F.q]`. `A[F.p, F.q]` is block upper triangular with the
irreducible diagonal blocks of [`dmperm`](@ref), `F.L` is block diagonal, and the
factorization stores the blocks of `F.U` above the diagonal blocks unfactored, as the
entries of `A`, so reading `F.U` computes them. `F.p` and `F.q` have the index type
`Ti` of `A`; the factors are indexed with `Int`, since they can hold more entries than
`Ti` counts.

`F \\ B` solves `A * X = B`, and returns a sparse `X` when `B` is sparse;
`ldiv!(F, B)` overwrites a dense `B` with the solution.
"""
struct SparseLU{Tv,Ti<:Integer} <: Factorization{Tv}
    # The factors are indexed with `Int` whatever the index type of the matrix: they can
    # hold more entries than that type counts.
    # unit lower triangular and block diagonal, with the diagonal stored
    lower::SparseMatrixCSC{Tv,Int}
    # block upper triangular: the U factors of the diagonal blocks, and above them the
    # entries of A[p, q]
    upper::SparseMatrixCSC{Tv,Int}
    p::Vector{Ti}
    q::Vector{Ti}
    pinv::Vector{Int}
    # block b is the rows and columns blockptr[b]:blockptr[b+1]-1, and blockof inverts that
    blockptr::Vector{Int}
    blockof::Vector{Int}
    # the same boundaries with every run of 1×1 blocks merged into one triangular block,
    # which a solve with a dense right-hand side sweeps in one pass
    solveptr::Vector{Int}
    # the blocks that the solution in block b feeds are graphadj[graphptr[b]:graphptr[b+1]-1]
    graphptr::Vector{Int}
    graphadj::Vector{Int}
end

Base.size(F::SparseLU) = size(getfield(F, :lower))
Base.size(F::SparseLU, d::Integer) = size(getfield(F, :lower), d)

function Base.getproperty(F::SparseLU{Tv,Ti}, s::Symbol) where {Tv,Ti}
    if s === :L
        return getfield(F, :lower)
    elseif s === :U
        return _upperfactor(F)
    end
    return getfield(F, s)
end
Base.propertynames(F::SparseLU, private::Bool = false) =
    private ? (:L, :U, fieldnames(SparseLU)...) : (:L, :U, :p, :q)

function Base.show(io::IO, mime::MIME"text/plain", F::SparseLU)
    summary(io, F)
    nb = length(getfield(F, :blockptr)) - 1
    print(io, " of a ", size(F, 1), "×", size(F, 2), " matrix with ", nb,
          nb == 1 ? " diagonal block" : " diagonal blocks")
    print(io, "\nL factor:\n")
    show(io, mime, F.L)
    print(io, "\nU factor:\n")
    show(io, mime, F.U)
end

# U with its off-diagonal blocks factored: the stored blocks, solved with L
function _upperfactor(F::SparseLU{Tv,Ti}) where {Tv,Ti}
    upper = getfield(F, :upper)
    blockptr = getfield(F, :blockptr)
    blockof = getfield(F, :blockof)
    n = size(upper, 2)
    Ui = rowvals(upper)
    Ux = nonzeros(upper)
    # split each column at the first row of its block
    Dp = Vector{Int}(undef, n + 1)
    Op = Vector{Int}(undef, n + 1)
    Di = Int[]; Dx = Tv[]; Oi = Int[]; Ox = Tv[]
    @inbounds for j in 1:n
        Dp[j] = length(Di) + 1
        Op[j] = length(Oi) + 1
        lo = blockptr[blockof[j]]
        for k in nzrange(upper, j)
            if Ui[k] < lo
                push!(Oi, Ui[k]); push!(Ox, Ux[k])
            else
                push!(Di, Ui[k]); push!(Dx, Ux[k])
            end
        end
    end
    Dp[n + 1] = length(Di) + 1
    Op[n + 1] = length(Oi) + 1
    isempty(Oi) && return SparseMatrixCSC{Tv,Int}(n, n, Dp, Di, Dx)
    off = SparseMatrixCSC{Tv,Int}(n, n, Op, Oi, Ox)
    return SparseMatrixCSC{Tv,Int}(n, n, Dp, Di, Dx) +
        _sptrisolve(getfield(F, :lower), true, true, off, Tv, Int)
end

# The arrays of a factorization in progress: the columns of L and U built so far, the
# pivot position `pinv[i]` of each row (0 until it is a pivot), the pruned end `lend` of
# each column of L, and the work arrays of a column.
struct GPState{Tv}
    Lp::Vector{Int}
    Li::Vector{Int}
    Lx::Vector{Tv}
    Up::Vector{Int}
    Ui::Vector{Int}
    Ux::Vector{Tv}
    pinv::Vector{Int}
    lend::Vector{Int}
    pruned::BitVector
    mark::Vector{Int}
    x::Vector{Tv}
    order::Vector{Int}
    nodes::Vector{Int}
    pos::Vector{Int}
    cand::Vector{Int}
end
function GPState{Tv}(n::Int, nz::Int) where {Tv}
    Li = Int[]; Lx = Tv[]; Ui = Int[]; Ux = Tv[]
    sizehint!(Li, nz + n); sizehint!(Lx, nz + n)
    sizehint!(Ui, nz + n); sizehint!(Ux, nz + n)
    return GPState{Tv}(Vector{Int}(undef, n + 1), Li, Lx, Vector{Int}(undef, n + 1), Ui, Ux,
                       zeros(Int, n), zeros(Int, n), falses(n), zeros(Int, n), zeros(Tv, n),
                       Vector{Int}(undef, n), Vector{Int}(undef, n), Vector{Int}(undef, n),
                       Vector{Int}(undef, n))
end

# Factor the diagonal blocks `first` and later of the block upper triangular `B`, whose
# blocks are the rows and columns bp[b]:bp[b+1]-1, into `S`. Each block is factored a
# column at a time (Gilbert and Peierls): the pattern of column j of L and U is what a
# depth-first search of the graph of the earlier columns of L reaches from the entries
# of B[:, j], and solving in the reverse postorder of that search costs O(flops). Rows
# keep the numbering of B until `_gpfinish!`. Entries are never dropped on cancellation,
# which pruning relies on.
#
# Pruning (Eisenstat and Liu): once column k of L has a row t that became the pivot of a
# column j with u_kj stored, every later row of column k is also in column j of L, so
# the search reaches it through t. Column k is then reordered to have its pivotal rows
# first, and the search scans only those, up to lend[k]. A column is pruned once.
#
# Returns 0 when every block is factored. When the factors of a block b outgrow
# `limit[b]` (0 for no limit), the block is taken out of `S` again and b is returned: the
# caller reorders it and calls again with `first = b`.
function _gplu!(S::GPState{Tv}, B::AbstractSparseMatrixCSC, bp::Vector{Int}, first::Int,
                limit::Vector{Int}, tol::Float64, prune::Bool, q::Vector{Int}) where {Tv}
    n = size(B, 2)
    Bi = rowvals(B)
    Bx = nonzeros(B)
    Lp = S.Lp; Li = S.Li; Lx = S.Lx; Up = S.Up; Ui = S.Ui; Ux = S.Ux
    pinv = S.pinv; lend = S.lend; pruned = S.pruned; mark = S.mark; x = S.x
    order = S.order; nodes = S.nodes; pos = S.pos; cand = S.cand
    @inbounds for b in first:(length(bp) - 1)
        lo = bp[b]
        hi = bp[b + 1] - 1
        nupper = 0      # entries of U in the block so far
        for j in lo:hi
            if limit[b] > 0 && j > lo && length(Li) - Lp[lo] + 1 + nupper > limit[b]
                resize!(Li, Lp[lo] - 1); resize!(Lx, Lp[lo] - 1)
                resize!(Ui, Up[lo] - 1); resize!(Ux, Up[lo] - 1)
                for i in lo:hi
                    pinv[i] = 0
                    mark[i] = 0
                    lend[i] = 0
                    pruned[i] = false
                end
                return b
            end
            Lp[j] = length(Li) + 1
            Up[j] = length(Ui) + 1
            top = n + 1
            ncand = 0
            for idx in nzrange(B, j)
                i = Int(Bi[idx])
                if i < lo
                    push!(Ui, pinv[i]); push!(Ux, Bx[idx])
                    continue
                end
                x[i] = Bx[idx]
                mark[i] == j && continue
                mark[i] = j
                if pinv[i] == 0
                    ncand += 1
                    cand[ncand] = i
                    continue
                end
                head = 1
                nodes[1] = i
                pos[1] = Lp[pinv[i]] + 1
                while head > 0
                    stop = lend[pinv[nodes[head]]]
                    p = pos[head]
                    descended = false
                    while p <= stop
                        r = Li[p]
                        p += 1
                        mark[r] == j && continue
                        mark[r] = j
                        if pinv[r] == 0
                            ncand += 1
                            cand[ncand] = r
                        else
                            pos[head] = p
                            head += 1
                            nodes[head] = r
                            pos[head] = Lp[pinv[r]] + 1
                            descended = true
                            break
                        end
                    end
                    if !descended
                        top -= 1
                        order[top] = nodes[head]
                        head -= 1
                    end
                end
            end
            ublock = length(Ui) + 1
            for t in top:n
                i = order[t]
                k = pinv[i]
                ukj = x[i]
                x[i] = zero(Tv)
                push!(Ui, k); push!(Ux, ukj)
                _iszero(ukj) && continue
                for p in (Lp[k] + 1):(Lp[k + 1] - 1)
                    x[Li[p]] -= Lx[p] * ukj
                end
            end
            # the pivot: the diagonal if it is within `tol` of the largest candidate
            ncand == 0 && throw(LinearAlgebra.SingularException(q[j]))
            # the largest candidate, and of equal ones the first row, whatever order the
            # search found them in
            piv = cand[1]
            amax = abs(x[piv])
            for t in 2:ncand
                i = cand[t]
                a = abs(x[i])
                if a > amax || (i < piv && a == amax)
                    amax = a
                    piv = i
                end
            end
            _iszero(amax) && throw(LinearAlgebra.SingularException(q[j]))
            if piv != j && pinv[j] == 0 && mark[j] == j
                _acceptable(abs(x[j]), amax, tol) && (piv = j)
            end
            pivot = x[piv]
            pinv[piv] = j
            push!(Ui, j); push!(Ux, pivot)
            push!(Li, piv); push!(Lx, one(Tv))
            for t in 1:ncand
                i = cand[t]
                if i != piv
                    push!(Li, i); push!(Lx, x[i] / pivot)
                end
                x[i] = zero(Tv)
            end
            lend[j] = length(Li)
            nupper += length(Ui) - ublock + 1
            prune || continue
            for t in ublock:(length(Ui) - 1)
                k = Ui[t]
                pruned[k] && continue
                head = Lp[k] + 1
                tail = lend[k]
                symmetric = false
                for p in head:tail
                    if Li[p] == piv
                        symmetric = true
                        break
                    end
                end
                symmetric || continue
                while head <= tail
                    if pinv[Li[head]] != 0
                        head += 1
                    else
                        Li[head], Li[tail] = Li[tail], Li[head]
                        Lx[head], Lx[tail] = Lx[tail], Lx[head]
                        tail -= 1
                    end
                end
                lend[k] = tail
                pruned[k] = true
            end
        end
    end
    return 0
end

# Close the columns of the finished factorization, number the rows of L by pivot position
# and sort the columns: the CSC arrays of L (unit diagonal stored) and of U with the
# entries of B above its diagonal blocks, and pinv.
function _gpfinish!(S::GPState)
    n = length(S.pinv)
    Lp = S.Lp; Li = S.Li; Up = S.Up; Ui = S.Ui; pinv = S.pinv
    Lp[n + 1] = length(Li) + 1
    Up[n + 1] = length(Ui) + 1
    @inbounds for p in eachindex(Li)
        Li[p] = pinv[Li[p]]
    end
    _sortcolumns!(n, Lp, Li, S.Lx)
    _sortcolumns!(n, Up, Ui, S.Ux)
    return Lp, Li, S.Lx, Up, Ui, S.Ux, pinv
end

## Fill-reducing ordering of the diagonal blocks

# AMD and COLAMD are the BSD-licensed ordering libraries of SuiteSparse, which every
# build of Julia ships, with or without the GPL solvers.
using .LibSuiteSparse: amd_order, amd_l_order, colamd, colamd_l, colamd_recommended,
    colamd_l_recommended, COLAMD_STATS, AMD_INFO, AMD_LNZ

# a block smaller than this is factored in its natural order: it cannot fill in much
const _ORDERING_MIN_BLOCK = 16
# `:auto` orders a block with AMD when at least this fraction of its columns have a
# diagonal entry that the pivot test accepts, and with COLAMD otherwise
const _ORDERING_DIAGONAL = 0.9

# `:auto` gives up on the AMD ordering of a block, and orders it with COLAMD instead, once
# its factors hold this many times the entries that AMD predicted
const _AMD_FILL_LIMIT = 1.5

# whether an entry of magnitude `a` may be the pivot of a column whose largest is `amax`
@inline _acceptable(a, amax, tol::Float64) = !_iszero(a) && (tol == 1 ? a >= amax : a >= tol * amax)

# The entry points for the index type that matches `Int`. COLAMD takes the row indices
# in a work array of the size it recommends and the column pointers, both zero-based, and
# returns the column order in place of the pointers; a null pointer selects its default
# knobs, and AMD's default control. AMD reports the fill it predicts in `info`.
@static if Int === Int64
    _colamd_recommended(nz::Int, n::Int) = colamd_l_recommended(nz, n, n)
    _colamd!(n::Int, work::Vector{Int}, ptr::Vector{Int}, stats::Vector{Int}) =
        colamd_l(n, n, length(work), work, ptr, C_NULL, stats)
    _amd!(n::Int, ptr::Vector{Int}, ind::Vector{Int}, perm::Vector{Int}, info::Vector{Float64}) =
        amd_l_order(n, ptr, ind, perm, C_NULL, info)
else
    _colamd_recommended(nz::Int, n::Int) = colamd_recommended(nz, n, n)
    _colamd!(n::Int, work::Vector{Int}, ptr::Vector{Int}, stats::Vector{Int}) =
        colamd(n, n, length(work), work, ptr, C_NULL, stats)
    _amd!(n::Int, ptr::Vector{Int}, ind::Vector{Int}, perm::Vector{Int}, info::Vector{Float64}) =
        amd_order(n, ptr, ind, perm, C_NULL, info)
end

# Reorder the columns of every large diagonal block of A[p, q], whose blocks are
# bp[b]:bp[b+1]-1, to reduce the fill of its factorization: with COLAMD, which orders
# the columns for any row pivoting, or with AMD on the pattern of the block plus its
# transpose, which suits pivots that stay on the diagonal; `:auto` takes AMD for a block
# where most of the diagonal entries pass the pivot test with tolerance `tol` before any
# elimination. The rows are moved with their columns, which keeps the matched diagonal.
# A block whose ordering fails keeps its natural order. Only `blocks` are reordered.
#
# AMD predicts the fill of a factorization whose pivots stay on the diagonal. For a block
# that `:auto` gives to AMD, `limit` receives the number of entries of its factors past
# which that prediction has failed, and the block is to be ordered again, with COLAMD.
function _orderblocks!(limit::Vector{Int}, p::Vector{Int}, q::Vector{Int},
                       A::AbstractSparseMatrixCSC, bp::Vector{Int}, ordering::Symbol,
                       tol::Float64, blocks::AbstractUnitRange{Int} = 1:(length(bp) - 1))
    n = length(q)
    rv = rowvals(A)
    nz = nonzeros(A)
    rowpos = Vector{Int}(undef, n)
    @inbounds for t in 1:n
        rowpos[p[t]] = t
    end
    ptr = Int[]
    ind = Int[]
    perm = Int[]
    stats = Vector{Int}(undef, COLAMD_STATS)
    info = Vector{Float64}(undef, AMD_INFO)
    @inbounds for b in blocks
        lo = bp[b]
        hi = bp[b + 1] - 1
        nblk = hi - lo + 1
        limit[b] = 0
        nblk >= _ORDERING_MIN_BLOCK || continue
        # the pattern of the block, zero-based, and its columns with a strong diagonal
        resize!(ptr, nblk + 1)
        empty!(ind)
        strong = 0
        for c in lo:hi
            ptr[c - lo + 1] = length(ind)
            r = nzrange(A, q[c])
            isempty(r) && continue
            amax = adiag = abs(zero(eltype(nz)))
            for k in r
                t = rowpos[rv[k]]
                t >= lo || continue
                push!(ind, t - lo)
                a = abs(nz[k])
                a > amax && (amax = a)
                t == c && (adiag = a)
            end
            strong += _acceptable(adiag, amax, tol)
        end
        ptr[nblk + 1] = length(ind)
        useamd = ordering === :amd || (ordering === :auto && strong >= _ORDERING_DIAGONAL * nblk)
        if !useamd
            len = Int(_colamd_recommended(length(ind), nblk))
            len == 0 && continue
            resize!(ind, len)
            _colamd!(nblk, ind, ptr, stats) == 1 || continue
            copyto!(resize!(perm, nblk), 1, ptr, 1, nblk)
        else
            resize!(perm, nblk)
            0 <= _amd!(nblk, ptr, ind, perm, info) <= 1 || continue
            if ordering === :auto
                # L and U each hold the predicted entries below the diagonal, and a diagonal
                predicted = 2 * (info[AMD_LNZ + 1] + nblk)
                limit[b] = ceil(Int, min(_AMD_FILL_LIMIT * predicted, typemax(Int) / 2))
            end
        end
        for t in 1:nblk
            ptr[t] = q[lo + perm[t]]
            perm[t] = p[lo + perm[t]]
        end
        copyto!(q, lo, ptr, 1, nblk)
        copyto!(p, lo, perm, 1, nblk)
    end
    return nothing
end

# Row and column permutations that make `A` upper triangular with a zero-free diagonal,
# or `nothing` when there are none. A column with a single entry can come first, with the
# row of that entry; removing the row may leave other columns with a single entry, and
# `A` is a permuted triangular matrix exactly when this uses up every column. O(nnz), and
# O(n) when no column has a single entry to start from.
function _triangularorder(A::AbstractSparseMatrixCSC)
    n = size(A, 2)
    rv = rowvals(A)
    count = Vector{Int}(undef, n)
    queue = Int[]
    @inbounds for j in 1:n
        count[j] = length(nzrange(A, j))
        count[j] == 1 && push!(queue, j)
    end
    isempty(queue) && return nothing
    rowptr, colind = _rowpattern(A)
    removed = falses(n)
    p = Vector{Int}(undef, n)
    q = Vector{Int}(undef, n)
    k = 0
    head = 1
    @inbounds while head <= length(queue)
        j = queue[head]
        head += 1
        count[j] == 1 || continue
        i = 0
        for t in nzrange(A, j)
            if !removed[rv[t]]
                i = Int(rv[t])
                break
            end
        end
        k += 1
        p[k] = i
        q[k] = j
        removed[i] = true
        for t in rowptr[i]:(rowptr[i + 1] - 1)
            c = colind[t]
            count[c] -= 1
            count[c] == 1 && push!(queue, c)
        end
    end
    return k == n ? (p, q) : nothing
end

"""
    SparseArrays.sparselu(A; tol = 0.1, ordering = :auto, prune = true) -> F::SparseArrays.SparseLU

Compute the LU factorization of the square sparse matrix `A`, in Julia, for any element
type with a division. `F.L * F.U ≈ A[F.p, F.q]`; see [`SparseArrays.SparseLU`](@ref).

A permutation of a triangular matrix is recognized first and needs no factorization.
Any other `A` is permuted to block upper triangular form by [`dmperm`](@ref), and only
its diagonal blocks are factored, so entries above them cause no fill. Each block is
factored a column at a time with partial pivoting, by the left-looking algorithm of
Gilbert and Peierls [^GilbertPeierls1988], in time proportional to the arithmetic,
with the symmetric pruning of Eisenstat and Liu [^EisenstatLiu1993] to shorten its
searches.

`ordering` chooses the order of the columns within a block, which decides how much the
factors fill in: `:colamd`, a minimum degree ordering of the columns that bounds the
fill for any row pivoting; `:amd`, a minimum degree ordering of the pattern of the
block plus its transpose, which gives less fill when the pattern is nearly symmetric
and the pivots stay on the diagonal; `:natural`, increasing order; or `:auto`, the
default, which takes `:amd` for a block most of whose diagonal entries are acceptable
pivots for `tol`, and `:colamd` otherwise. If the pivots of such a block then leave the
diagonal and its factors grow past $_AMD_FILL_LIMIT times what AMD predicted, `:auto`
orders the block again with COLAMD and factors it afresh. Blocks with fewer than
$_ORDERING_MIN_BLOCK columns are always taken in increasing order. AMD and COLAMD are
the SuiteSparse ordering libraries.

A column's pivot is its diagonal entry in the permuted matrix when that is at least
`tol` times the largest candidate in magnitude, and the largest candidate otherwise.
`tol = 1` is partial pivoting; a smaller `tol` keeps more of the zero-free diagonal
that `dmperm` provides, which is what the orderings assume, and bounds the entries of
`L` by `1 / tol`. Of candidates equal in magnitude the first row is taken. `prune =
false` turns pruning off, which changes the work and the order of the updates, and so
the rounding, but not the factorization computed in exact arithmetic.

Throws a `SingularException` when `A` is structurally or numerically singular.

`A \\ B` uses this factorization when `B` is sparse.

[^GilbertPeierls1988]: J. R. Gilbert and T. Peierls, "Sparse partial pivoting in time proportional to arithmetic operations", SIAM Journal on Scientific and Statistical Computing 9(5), 1988, pp. 862-874. [doi:10.1137/0909058](https://doi.org/10.1137/0909058)

[^EisenstatLiu1993]: S. C. Eisenstat and J. W. H. Liu, "Exploiting structural symmetry in a sparse partial pivoting code", SIAM Journal on Scientific Computing 14(1), 1993, pp. 253-257. [doi:10.1137/0914015](https://doi.org/10.1137/0914015)

# Examples
```jldoctest
julia> A = sparse([2.0 0.0 1.0; 4.0 1.0 0.0; 0.0 0.0 5.0]);

julia> F = SparseArrays.sparselu(A);

julia> F.L * F.U == A[F.p, F.q]
true

julia> F \\ sparsevec([3], [10.0], 3)
3-element SparseVector{Float64, Int64} with 3 stored entries:
  [1]  =  -1.0
  [2]  =  4.0
  [3]  =  2.0
```
"""
function sparselu(A::AbstractSparseMatrixCSC{TvA,Ti}; tol::Real = 0.1, ordering::Symbol = :auto,
                  prune::Bool = true) where {TvA,Ti}
    require_one_based_indexing(A)
    n = checksquare(A)
    0 <= tol <= 1 || throw(ArgumentError(lazy"the pivot tolerance must be in [0, 1], got $tol"))
    ordering in (:auto, :colamd, :amd, :natural) ||
        throw(ArgumentError(lazy"the ordering must be :auto, :colamd, :amd or :natural, got :$ordering"))
    Tv = typeof(oneunit(TvA) / oneunit(TvA))
    triangular = _triangularorder(A)
    if triangular !== nothing
        # a permuted triangular matrix is its own U factor, in n blocks of one
        p, q = triangular
        upper = convert(SparseMatrixCSC{Tv,Int}, permute(A, p, q))
        Ux = nonzeros(upper)
        @inbounds for j in 1:n
            _iszero(Ux[last(nzrange(upper, j))]) && throw(LinearAlgebra.SingularException(q[j]))
        end
        lower = SparseMatrixCSC{Tv,Int}(n, n, collect(1:(n + 1)), collect(1:n), ones(Tv, n))
        bp = collect(1:(n + 1))
        pfinal = convert(Vector{Ti}, p)
    else
        p, q, bp, _, _, _, _, colmatch = _dmperm(A)
        unmatched = findfirst(iszero, colmatch)
        unmatched === nothing || throw(LinearAlgebra.SingularException(unmatched))
        limit = zeros(Int, length(bp) - 1)
        ordering === :natural || _orderblocks!(limit, p, q, A, bp, ordering, Float64(tol))
        S = GPState{Tv}(n, nnz(A))
        first = 1
        while true
            blown = _gplu!(S, permute(A, p, q), bp, first, limit, Float64(tol), prune, q)
            blown == 0 && break
            # AMD's prediction failed for this block: its pivots left the diagonal
            _orderblocks!(limit, p, q, A, bp, :colamd, Float64(tol), blown:blown)
            first = blown
        end
        Lp, Li, Lx, Up, Ui, Ux, pinv = _gpfinish!(S)
        pfinal = Vector{Ti}(undef, n)
        @inbounds for i in 1:n
            pfinal[pinv[i]] = p[i]
        end
        lower = SparseMatrixCSC{Tv,Int}(n, n, Lp, Li, Lx)
        upper = SparseMatrixCSC{Tv,Int}(n, n, Up, Ui, Ux)
    end
    return _sparselu(lower, upper, pfinal, convert(Vector{Ti}, q), bp)
end

# the factorization object, from the factors, the permutations and the block boundaries
function _sparselu(lower::SparseMatrixCSC{Tv,Int}, upper::SparseMatrixCSC{Tv,Int}, p::Vector{Ti},
                   q::Vector{Ti}, bp::Vector{Int}) where {Tv,Ti}
    n = size(upper, 2)
    Up = getcolptr(upper)
    Ui = rowvals(upper)
    rowpos = Vector{Int}(undef, n)
    @inbounds for t in 1:n
        rowpos[p[t]] = t
    end
    nb = length(bp) - 1
    blockof = Vector{Int}(undef, n)
    solveptr = [1]
    @inbounds for b in 1:nb
        lo = bp[b]
        hi = bp[b + 1] - 1
        for j in lo:hi
            blockof[j] = b
        end
        # a block of one extends a run of them; anything else ends the run before it
        if hi > lo
            solveptr[end] == lo || push!(solveptr, lo)
            push!(solveptr, hi + 1)
        elseif b == nb
            push!(solveptr, hi + 1)
        end
    end
    # the block graph: an edge from a block to each block with a row in its columns
    graphptr = Vector{Int}(undef, nb + 1)
    graphadj = Int[]
    seen = zeros(Int, nb)
    @inbounds for b in 1:nb
        graphptr[b] = length(graphadj) + 1
        lo = bp[b]
        for j in lo:(bp[b + 1] - 1)
            for k in Up[j]:(Up[j + 1] - 1)
                i = Ui[k]
                i < lo || break
                a = blockof[i]
                if seen[a] != b
                    seen[a] = b
                    push!(graphadj, a)
                end
            end
        end
    end
    graphptr[nb + 1] = length(graphadj) + 1
    return SparseLU{Tv,Ti}(lower, upper, p, q, rowpos, bp, blockof,
                           solveptr, graphptr, graphadj)
end

## Solves with the factorization

# Solve with the diagonal block lo:hi in place in `y`, and subtract its solution times
# the entries above the block: L forward, then the columns of the stored upper matrix
# backward, whose diagonal entry is the last of each column.
@inline function _blocksolve!(y::AbstractVector, lower::SparseMatrixCSC, upper::SparseMatrixCSC,
                              lo::Int, hi::Int)
    Lp = getcolptr(lower); Li = rowvals(lower); Lx = nonzeros(lower)
    Up = getcolptr(upper); Ui = rowvals(upper); Ux = nonzeros(upper)
    @inbounds for j in lo:(hi - 1)
        yj = y[j]
        _iszero(yj) && continue
        for k in (Lp[j] + 1):(Lp[j + 1] - 1)
            y[Li[k]] -= Lx[k] * yj
        end
    end
    @inbounds for j in hi:-1:lo
        d = Up[j + 1] - 1
        _iszero(y[j]) && continue
        yj = Ux[d] \ y[j]
        y[j] = yj
        for k in Up[j]:(d - 1)
            y[Ui[k]] -= Ux[k] * yj
        end
    end
    return y
end

function _lusolve!(b::AbstractVector, F::SparseLU, y::Vector)
    lower = getfield(F, :lower)
    upper = getfield(F, :upper)
    p = getfield(F, :p)
    q = getfield(F, :q)
    solveptr = getfield(F, :solveptr)
    @inbounds for t in eachindex(y)
        y[t] = b[p[t]]
    end
    @inbounds for blk in (length(solveptr) - 1):-1:1
        _blocksolve!(y, lower, upper, Int(solveptr[blk]), Int(solveptr[blk + 1]) - 1)
    end
    @inbounds for t in eachindex(y)
        b[q[t]] = y[t]
    end
    return b
end

function ldiv!(F::SparseLU, B::StridedVecOrMat)
    require_one_based_indexing(B)
    n = size(F, 1)
    size(B, 1) == n ||
        throw(DimensionMismatch(lazy"the factorization has $n rows, but the right-hand side has $(size(B, 1))"))
    y = Vector{eltype(B)}(undef, n)
    for c in axes(B, 2)
        _lusolve!(view(B, :, c), F, y)
    end
    return B
end

# X = F \ B for a sparse B. A block of the solution is nonzero when the column of B has
# an entry in it or a nonzero later block has entries in its rows, and it is then full,
# its diagonal block being irreducible. So the blocks to solve are those reachable in the
# block graph, and reverse postorder solves each after the blocks that feed it: the
# search of the triangular solve, a block at a time.
function _lusolve(F::SparseLU, B::SparseMatrixCSCOrView, ::Type{T}, ::Type{Ti}) where {T,Ti}
    require_one_based_indexing(B)
    n = size(F, 1)
    size(B, 1) == n ||
        throw(DimensionMismatch(lazy"the factorization has $n rows, but the right-hand side has $(size(B, 1))"))
    lower = getfield(F, :lower)
    upper = getfield(F, :upper)
    q = getfield(F, :q)
    pinv = getfield(F, :pinv)
    blockptr = getfield(F, :blockptr)
    blockof = getfield(F, :blockof)
    graphptr = getfield(F, :graphptr)
    graphadj = getfield(F, :graphadj)
    nb = length(blockptr) - 1
    Bi = rowvals(B)
    Bx = nonzeros(B)
    k = size(B, 2)
    y = zeros(T, n)     # in the permuted numbering
    z = zeros(T, n)     # in the numbering of the solution
    mark = zeros(Int, nb)
    order = Vector{Int}(undef, nb)
    nodes = Vector{Int}(undef, nb)
    pos = Vector{Int}(undef, nb)
    rows = Int[]
    Xp = Vector{Ti}(undef, k + 1)
    Xi = Ti[]
    Xx = T[]
    @inbounds for c in 1:k
        Xp[c] = length(Xi) + 1
        top = nb + 1
        for idx in nzrange(B, c)
            t = Int(pinv[Bi[idx]])
            y[t] = Bx[idx]
            blk = Int(blockof[t])
            mark[blk] == c && continue
            mark[blk] = c
            head = 1
            nodes[1] = blk
            pos[1] = graphptr[blk]
            while head > 0
                blk = nodes[head]
                g = pos[head]
                stop = Int(graphptr[blk + 1]) - 1
                descended = false
                while g <= stop
                    a = Int(graphadj[g])
                    g += 1
                    if mark[a] != c
                        mark[a] = c
                        pos[head] = g
                        head += 1
                        nodes[head] = a
                        pos[head] = graphptr[a]
                        descended = true
                        break
                    end
                end
                if !descended
                    head -= 1
                    top -= 1
                    order[top] = blk
                end
            end
        end
        empty!(rows)
        for t in top:nb
            blk = order[t]
            lo = Int(blockptr[blk])
            hi = Int(blockptr[blk + 1]) - 1
            _blocksolve!(y, lower, upper, lo, hi)
            for j in lo:hi
                r = Int(q[j])
                z[r] = y[j]
                y[j] = zero(T)
                push!(rows, r)
            end
        end
        sort!(rows)
        for r in rows
            push!(Xi, r)
            push!(Xx, z[r])
            z[r] = zero(T)
        end
    end
    Xp[k + 1] = length(Xi) + 1
    return SparseMatrixCSC{T,Ti}(n, k, Xp, Xi, Xx)
end

_solve_eltype(::Type{TA}, ::Type{TB}) where {TA,TB} = typeof(oneunit(TA) \ oneunit(TB))
_rhs_indtype(::SparseMatrixCSCOrView{<:Any,Ti}) where {Ti} = Ti

# a sparse right-hand side as a matrix the kernels can walk, and the solution back in
# the shape of the right-hand side
_rhs_matrix(B::SparseMatrixCSCOrView) = B
_rhs_matrix(B::AdjOrTrans{<:Any,<:AbstractSparseMatrixCSC}) = copy(B)
function _rhs_matrix(b::SparseVectorOrView{Tv,Ti}) where {Tv,Ti}
    inds = Vector{Ti}(nonzeroinds(b))
    return SparseMatrixCSC{Tv,Ti}(length(b), 1, Ti[1, length(inds) + 1], inds, Vector{Tv}(nonzeros(b)))
end
_rhs_shape(X::SparseMatrixCSC, B::AbstractMatrix) = X
_rhs_shape(X::SparseMatrixCSC, b::AbstractVector) = SparseVector(size(X, 1), rowvals(X), nonzeros(X))

function \(F::SparseLU{<:Any,Ti}, B::SparseSolveRHS) where {Ti}
    Bm = _rhs_matrix(B)
    return _rhs_shape(_lusolve(F, Bm, _solve_eltype(eltype(F), eltype(Bm)),
                               promote_type(Ti, _rhs_indtype(Bm))), B)
end
