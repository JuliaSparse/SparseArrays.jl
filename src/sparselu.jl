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
# and Peierls, Lemma 1). After an O(n log n) pass that locates the diagonal, each column
# costs its arithmetic plus the sort of its pattern.
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
    # the strict triangle of column j is Ai[from[j]:to[j]], and its diagonal Ax[dptr[j]]
    from = Vector{Int}(undef, n)
    to = Vector{Int}(undef, n)
    dptr = Vector{Int}(undef, n)
    @inbounds for j in 1:n
        r = nzrange(A, j)
        d = searchsortedfirst(Ai, j, first(r), last(r), Forward)
        stored = d <= last(r) && Ai[d] == j
        if !unit && (!stored || _iszero(Ax[d]))
            throw(LinearAlgebra.SingularException(j))
        end
        dptr[j] = d
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
            xj = unit ? x[j] : Ax[dptr[j]] \ x[j]
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
entries of `A`, so reading `F.U` computes them.

`F \\ B` solves `A * X = B`, and returns a sparse `X` when `B` is sparse;
`ldiv!(F, B)` overwrites a dense `B` with the solution.
"""
struct SparseLU{Tv,Ti<:Integer} <: Factorization{Tv}
    # unit lower triangular and block diagonal, with the diagonal stored
    lower::SparseMatrixCSC{Tv,Ti}
    # block upper triangular: the U factors of the diagonal blocks, and above them the
    # entries of A[p, q]
    upper::SparseMatrixCSC{Tv,Ti}
    p::Vector{Ti}
    q::Vector{Ti}
    pinv::Vector{Ti}
    # block b is the rows and columns blockptr[b]:blockptr[b+1]-1, and blockof inverts that
    blockptr::Vector{Ti}
    blockof::Vector{Ti}
    # the blocks that the solution in block b feeds are graphadj[graphptr[b]:graphptr[b+1]-1]
    graphptr::Vector{Ti}
    graphadj::Vector{Ti}
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
    Dp = Vector{Ti}(undef, n + 1)
    Op = Vector{Ti}(undef, n + 1)
    Di = Ti[]; Dx = Tv[]; Oi = Ti[]; Ox = Tv[]
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
    isempty(Oi) && return SparseMatrixCSC{Tv,Ti}(n, n, Dp, Di, Dx)
    off = SparseMatrixCSC{Tv,Ti}(n, n, Op, Oi, Ox)
    return SparseMatrixCSC{Tv,Ti}(n, n, Dp, Di, Dx) +
        _sptrisolve(getfield(F, :lower), true, true, off, Tv, Ti)
end

# The factorization of the block upper triangular `B`, whose diagonal blocks are the rows
# and columns bp[b]:bp[b+1]-1. Each block is factored a column at a time (Gilbert and
# Peierls): the pattern of column j of L and U is what a depth-first search of the graph
# of the earlier columns of L reaches from the entries of B[:, j], and solving in the
# reverse postorder of that search costs O(flops). Rows keep the numbering of B until the
# end; pinv[i] is the pivot position of row i, or 0. Entries are never dropped on
# cancellation, which pruning relies on.
#
# Pruning (Eisenstat and Liu): once column k of L has a row t that became the pivot of a
# column j with u_kj stored, every later row of column k is also in column j of L, so
# the search reaches it through t. Column k is then reordered to have its pivotal rows
# first, and the search scans only those, up to lend[k]. A column is pruned once.
#
# Returns the CSC arrays of L (unit diagonal stored) and of U with the entries of B above
# its diagonal blocks, both with sorted columns and rows in pivot order, and pinv.
function _gplu(B::AbstractSparseMatrixCSC, bp::Vector{Int}, ::Type{Tv}, tol::Float64,
               prune::Bool, q::Vector{Int}) where {Tv}
    n = size(B, 2)
    Bi = rowvals(B)
    Bx = nonzeros(B)
    Lp = Vector{Int}(undef, n + 1)
    Up = Vector{Int}(undef, n + 1)
    Li = Int[]; Lx = Tv[]; Ui = Int[]; Ux = Tv[]
    sizehint!(Li, nnz(B) + n); sizehint!(Lx, nnz(B) + n)
    sizehint!(Ui, nnz(B) + n); sizehint!(Ux, nnz(B) + n)
    pinv = zeros(Int, n)
    lend = zeros(Int, n)
    pruned = falses(n)
    mark = zeros(Int, n)
    x = zeros(Tv, n)
    order = Vector{Int}(undef, n)
    nodes = Vector{Int}(undef, n)
    pos = Vector{Int}(undef, n)
    cand = Vector{Int}(undef, n)
    @inbounds for b in 1:(length(bp) - 1)
        lo = bp[b]
        hi = bp[b + 1] - 1
        for j in lo:hi
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
            piv = 0
            amax = abs(zero(Tv))
            for t in 1:ncand
                a = abs(x[cand[t]])
                if a > amax
                    amax = a
                    piv = cand[t]
                end
            end
            piv == 0 && throw(LinearAlgebra.SingularException(q[j]))
            if piv != j && pinv[j] == 0 && mark[j] == j
                a = abs(x[j])
                if !_iszero(a) && (tol == 1 ? a >= amax : a >= tol * amax)
                    piv = j
                end
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
    Lp[n + 1] = length(Li) + 1
    Up[n + 1] = length(Ui) + 1
    @inbounds for p in eachindex(Li)
        Li[p] = pinv[Li[p]]
    end
    _sortcolumns!(n, Lp, Li, Lx)
    _sortcolumns!(n, Up, Ui, Ux)
    return Lp, Li, Lx, Up, Ui, Ux, pinv
end

"""
    SparseArrays.sparselu(A; tol = 1.0, prune = true) -> F::SparseArrays.SparseLU

Compute the LU factorization of the square sparse matrix `A`, in Julia, for any element
type with a division. `F.L * F.U ≈ A[F.p, F.q]`; see [`SparseArrays.SparseLU`](@ref).

`A` is first permuted to block upper triangular form by [`dmperm`](@ref), and only its
diagonal blocks are factored, so entries above them cause no fill. Each block is
factored a column at a time with partial pivoting, by the left-looking algorithm of
Gilbert and Peierls [^GilbertPeierls1988], in time proportional to the arithmetic,
with the symmetric pruning of Eisenstat and Liu [^EisenstatLiu1993] to shorten its
searches. Within a block the columns are taken in increasing order: no fill-reducing
ordering is applied, so a large irreducible block can fill in heavily.

A column's pivot is its diagonal entry in the permuted matrix when that is at least
`tol` times the largest candidate in magnitude, and the largest candidate otherwise:
`tol = 1` is partial pivoting, and a smaller `tol` trades stability for keeping the
zero-free diagonal that `dmperm` provides. `prune = false` turns pruning off, which
changes the work but not the pivots.

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
function sparselu(A::AbstractSparseMatrixCSC{TvA,Ti}; tol::Real = 1.0, prune::Bool = true) where {TvA,Ti}
    require_one_based_indexing(A)
    n = checksquare(A)
    0 <= tol <= 1 || throw(ArgumentError(lazy"the pivot tolerance must be in [0, 1], got $tol"))
    Tv = typeof(oneunit(TvA) / oneunit(TvA))
    p, q, bp, _, _, _, _, colmatch = _dmperm(A)
    unmatched = findfirst(iszero, colmatch)
    unmatched === nothing || throw(LinearAlgebra.SingularException(unmatched))
    Lp, Li, Lx, Up, Ui, Ux, pinv = _gplu(permute(A, p, q), bp, Tv, Float64(tol), prune, q)
    pfinal = Vector{Ti}(undef, n)
    @inbounds for i in 1:n
        pfinal[pinv[i]] = p[i]
    end
    # a row of the permuted matrix is pinv[invperm(p)[row]]
    rowpos = Vector{Ti}(undef, n)
    @inbounds for t in 1:n
        rowpos[pfinal[t]] = t
    end
    nb = length(bp) - 1
    blockof = Vector{Ti}(undef, n)
    @inbounds for b in 1:nb, j in bp[b]:(bp[b + 1] - 1)
        blockof[j] = b
    end
    # the block graph: an edge from a block to each block with a row in its columns
    graphptr = Vector{Ti}(undef, nb + 1)
    graphadj = Ti[]
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
    lower = SparseMatrixCSC{Tv,Ti}(n, n, convert(Vector{Ti}, Lp), convert(Vector{Ti}, Li), Lx)
    upper = SparseMatrixCSC{Tv,Ti}(n, n, convert(Vector{Ti}, Up), convert(Vector{Ti}, Ui), Ux)
    return SparseLU{Tv,Ti}(lower, upper, pfinal, convert(Vector{Ti}, q), rowpos,
                           convert(Vector{Ti}, bp), blockof, graphptr, graphadj)
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
    blockptr = getfield(F, :blockptr)
    @inbounds for t in eachindex(y)
        y[t] = b[p[t]]
    end
    @inbounds for blk in (length(blockptr) - 1):-1:1
        _blocksolve!(y, lower, upper, Int(blockptr[blk]), Int(blockptr[blk + 1]) - 1)
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

function \(F::SparseLU, B::SparseSolveRHS)
    Bm = _rhs_matrix(B)
    return _rhs_shape(_lusolve(F, Bm, _solve_eltype(eltype(F), eltype(Bm)),
                               promote_type(indtype(getfield(F, :lower)), _rhs_indtype(Bm))), B)
end
