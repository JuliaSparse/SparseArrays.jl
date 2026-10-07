# This file is a part of Julia. License is MIT: https://julialang.org/license

# Symbolic analysis: the column order and the panels of the numeric phase, from the
# symmetric pattern of A + Aᵀ. The panels are the supernodes, with relaxed amalgamation,
# that the factorization would have if every pivot stayed on the diagonal; the numeric
# phase pivots dynamically and only uses them to group columns.
#
# Pipeline:
#   1. fill-reducing order (AMD) of the symmetric pattern
#   2. elimination tree of the permuted pattern + postorder -> final order qf
#   3. column counts of L
#   4. fundamental supernodes + relaxed amalgamation (CHOLMOD-style rule)
#
# The algorithms are implemented from their published descriptions: the elimination tree
# via union-find with path halving (Liu 1990), postorder by an explicit two-stack DFS over
# counting-sorted child arrays, and L's column counts by the row-subtree characterization
# of Gilbert–Ng–Peyton (1994): row i appears in exactly the columns on the etree paths
# from i's below-diagonal neighbours up toward i.

# Elimination tree of a symmetric off-diagonal pattern.  For each vertex k
# (ascending) and each neighbour i < k, climb from i to the current root of
# i's partially-built subtree and attach it under k.  The climb uses
# path-halving union-find (`uf[x]` points toward the subtree root), so the
# total cost is effectively O(nnz · α).  parent[k] == 0 marks a root.
function etree_sym(cp::Vector{Int}, ri::Vector{Int}, n::Int)
    parent = zeros(Int, n)
    uf = collect(1:n)                     # union-find: x -> toward its root
    @inbounds for k in 1:n
        for p in cp[k]:(cp[k + 1] - 1)
            i = ri[p]
            i < k || continue
            # find root of i's subtree with path halving
            r = i
            while uf[r] != r
                uf[r] = uf[uf[r]]
                r = uf[r]
            end
            if r != k
                parent[r] = k
                uf[r] = k                  # union into k's growing subtree
            end
        end
        uf[k] = k
    end
    return parent
end

# Children of every node as a CSR-style array pair (counting sort by parent):
# children of v are chld[chp[v]:chp[v+1]-1], in ascending order.
function _children_csr(parent::Vector{Int})
    n = length(parent)
    chp = zeros(Int, n + 2)
    @inbounds for v in 1:n
        p = parent[v]
        p != 0 && (chp[p + 2] += 1)
    end
    chp[1] = 1
    chp[2] = 1
    @inbounds for v in 1:n
        chp[v + 2] += chp[v + 1]
    end
    chld = Vector{Int}(undef, chp[n + 2] - 1)
    @inbounds for v in 1:n                # ascending v ⇒ children stay sorted
        p = parent[v]
        if p != 0
            chld[chp[p + 1]] = v
            chp[p + 1] += 1
        end
    end
    return chp, chld                       # use chp[v]:chp[v+1]-1 after shift
end

# Depth-first postorder of the forest `parent`, children visited in ascending
# order.  Explicit node stack + child-cursor stack over the CSR child arrays
# (nothing is mutated, no sibling links).
function postorder_tree(parent::Vector{Int})
    n = length(parent)
    chp, chld = _children_csr(parent)
    post = Vector{Int}(undef, n)
    nstack = Vector{Int}(undef, n)
    cstack = Vector{Int}(undef, n)
    k = 0
    @inbounds for r in 1:n
        parent[r] == 0 || continue
        top = 1
        nstack[1] = r
        cstack[1] = chp[r]
        while top > 0
            v = nstack[top]
            c = cstack[top]
            if c < chp[v + 1]
                cstack[top] = c + 1
                top += 1
                nstack[top] = chld[c]
                cstack[top] = chp[chld[c]]
            else
                k += 1
                post[k] = v
                top -= 1
            end
        end
    end
    return post
end

# Pattern of B[perm, perm] for an off-diagonal symmetric pattern, produced
# with sorted columns by a single bucket pass: because the pattern is
# symmetric, emitting entry (min-side) buckets in ascending destination-row
# order is exactly a transpose pass, which sorts every column for free — no
# comparison sort anywhere.
function permute_pattern(cp::Vector{Int}, ri::Vector{Int}, perm::Vector{Int}, n::Int)
    pinv = invperm(perm)
    nz = length(ri)
    cpN = zeros(Int, n + 1)
    @inbounds for j in 1:n                # new-column sizes (symmetric: |col|
        cpN[pinv[j] + 1] = cp[j + 1] - cp[j]   # is permutation-invariant)
    end
    cpN[1] = 1
    @inbounds for j in 1:n
        cpN[j + 1] += cpN[j]
    end
    cursor = cpN[1:n]
    riN = Vector{Int}(undef, nz)
    # walk old columns in the order of their new ROW index; scattering entry
    # (i,j) -> (pinv[i], pinv[j]) into bucket pinv[j] then fills each new
    # column's rows in ascending order
    @inbounds for inew in 1:n
        iold = perm[inew]
        for p in cp[iold]:(cp[iold + 1] - 1)
            jnew = pinv[ri[p]]
            riN[cursor[jnew]] = inew
            cursor[jnew] += 1
        end
    end
    return cpN, riN
end

# Number of entries of each column of L strictly below the diagonal, for a postordered
# pattern, by row subtrees (Gilbert–Ng–Peyton): row i's columns are the nodes visited
# climbing the etree from each neighbour j < i of vertex i, stopping at nodes already
# claimed for row i.
function colcounts(cp::Vector{Int}, ri::Vector{Int}, parent::Vector{Int}, n::Int)
    counts = zeros(Int, n)
    claimed = zeros(Int, n)
    @inbounds for i in 1:n
        claimed[i] = i
        for p in cp[i]:(cp[i + 1] - 1)
            j = ri[p]
            j < i || continue
            while claimed[j] != i
                counts[j] += 1
                claimed[j] = i
                j = parent[j]
                j == 0 && break
            end
        end
    end
    return counts
end

# Largest half-bandwidth of the symmetric pattern, or -1 as soon as it exceeds
# `bwmax` (early abort — banded detection only needs narrow bands).
function _pattern_bandwidth(cp::Vector{Int}, ri::Vector{Int}, n::Int, bwmax::Int)
    bw = 0
    @inbounds for j in 1:n
        for p in cp[j]:(cp[j + 1] - 1)
            d = abs(ri[p] - j)
            if d > bw
                d > bwmax && return -1
                bw = d
            end
        end
    end
    return bw
end

# Fundamental supernodes: column j extends the current supernode iff j-1 is
# its only child in the etree and struct(j-1) = {j} ∪ struct(j).
function fundamental_supernodes(
        parent::Vector{Int}, counts::Vector{Int}, n::Int
    )
    nchild = zeros(Int, n)
    @inbounds for j in 1:n
        p = parent[j]
        p != 0 && (nchild[p] += 1)
    end
    sstart = Int[]
    @inbounds for j in 1:n
        extend = j > 1 && parent[j - 1] == j && nchild[j] == 1 &&
            counts[j - 1] == counts[j] + 1
        extend || push!(sstart, j)
    end
    push!(sstart, n + 1)
    return sstart
end

# Relaxed amalgamation à la CHOLMOD: repeatedly merge a supernode into the
# next one when the next one is its etree parent supernode and starts at the
# following column, if the merged panel stays small or introduces few explicit
# zeros.  For such adjacent parent merges the merged update-row set is exactly
# the parent's update-row set, which keeps the bookkeeping O(1) per attempt.
function amalgamate(
        sstart::Vector{Int}, counts::Vector{Int},
        parent::Vector{Int}, n::Int;
        nrelax0::Int = 8, nrelax1::Int = 32, nrelax2::Int = 96,
        zrelax0::Float64 = 0.8, zrelax1::Float64 = 0.2, zrelax2::Float64 = 0.05,
        maxsuper::Int = 512
    )
    ns = length(sstart) - 1
    # per (current) supernode, in column order
    first_ = [sstart[s] for s in 1:ns]
    last_ = [sstart[s + 1] - 1 for s in 1:ns]
    nrows = [counts[sstart[s + 1] - 1] for s in 1:ns]  # |update rows|
    nzero = zeros(Int, ns)          # explicit zeros accumulated in the panel
    total = Vector{Int}(undef, ns)  # panel entries (L side, incl. pivot block)
    @inbounds for s in 1:ns
        np = last_[s] - first_[s] + 1
        total[s] = np * np + np * nrows[s]     # full square pivot block + rect
    end
    merged_into = collect(1:ns)     # union-find-ish forward pointer
    alive = trues(ns)
    @inbounds for s in (ns - 1):-1:1
        t = s + 1
        while !alive[t]
            t = merged_into[t]
        end
        # merge candidate: t must start right after s and be s's etree parent
        # supernode (parent of s's last column lies in t's column range)
        first_[t] == last_[s] + 1 || continue
        pc = parent[last_[s]]
        (pc >= first_[t] && pc <= last_[t]) || continue
        npS = last_[s] - first_[s] + 1
        npT = last_[t] - first_[t] + 1
        npM = npS + npT
        npM <= maxsuper || continue
        # merged panel: columns first_[s]:last_[t], update rows = t's update
        # rows; per-column heights for s's columns grow accordingly
        totM = total[t] + npS * (npM + nrows[t])
        zM = nzero[s] + nzero[t] + (npS * (npM + nrows[t]) - total[s])
        z = zM / max(totM, 1)
        ok = npM <= nrelax0 ||
            (npM <= nrelax1 && z <= zrelax0) ||
            (npM <= nrelax2 && z <= zrelax1) ||
            z <= zrelax2
        ok || continue
        # merge s into t (t keeps its identity; ranges/zeros absorb s)
        first_[t] = first_[s]
        nzero[t] = zM
        total[t] = totM
        alive[s] = false
        merged_into[s] = t
    end
    out = Int[]
    @inbounds for s in 1:ns
        alive[s] && push!(out, first_[s])
    end
    sort!(out)
    push!(out, n + 1)
    return out
end

# Relaxed amalgamation pays off when the fundamental supernodes are this wide on average:
# for the PDE-like matrices it was tried on the factorization was 3-9% faster with it,
# while on circuit matrices, whose fundamental supernodes are nearly all single columns, the
# explicit zeros it adds made the factors 2-3.4 times larger and the factorization slower.
const RELAX_MIN_WIDTH = 1.2

# Column order `qf` and panel boundaries of the numeric phase: panel P holds the columns
# panels[P]:panels[P+1]-1, at most `maxpanel` of them. The fill-reducing order is `q0` when
# given, and otherwise comes from `ordering`; with `:amd`, a matrix whose symmetric pattern
# is a densely populated narrow band keeps the natural order, which is already
# fill-optimal. Either is then postordered. `nnzl` is the number of entries of L the
# analysis predicts.
function panel_analysis(
        A::SparseMatrixCSC; ordering::Symbol = :amd, relax::Union{Symbol, Bool} = :auto,
        maxpanel::Int = 256, q0::Union{Nothing, Vector{Int}} = nothing
    )
    n = size(A, 2)
    cp, ri = sym_pattern(A)
    q = if q0 !== nothing
        q0
    elseif ordering === :amd
        bw = n >= 8 ? _pattern_bandwidth(cp, ri, n, max(n ÷ 4 - 1, 0)) : -1
        if bw >= 0 && 2 * (length(ri) + n) >= n * (2 * bw + 1)
            collect(1:n)
        else
            amd_perm(cp, ri, n)
        end
    elseif ordering === :natural
        collect(1:n)
    else
        throw(ArgumentError("unknown ordering $ordering"))
    end
    cp1, ri1 = permute_pattern(cp, ri, q, n)
    post = postorder_tree(etree_sym(cp1, ri1, n))
    qf = q[post]
    cpF, riF = permute_pattern(cp1, ri1, post, n)
    parentF = etree_sym(cpF, riF, n)
    counts = colcounts(cpF, riF, parentF, n)
    sstart = fundamental_supernodes(parentF, counts, n)
    if relax === true || (relax === :auto && n >= RELAX_MIN_WIDTH * (length(sstart) - 1))
        sstart = amalgamate(sstart, counts, parentF, n)
    end
    panels = [1]
    for s in 1:(length(sstart) - 1)
        c = sstart[s]
        while c < sstart[s + 1]
            c = min(c + maxpanel, sstart[s + 1])
            push!(panels, c)
        end
    end
    return qf, panels, n + sum(counts)
end

# Column elimination tree of A[:, q], the elimination tree of A[:, q]ᵀ·A[:, q], without
# forming that product: column j becomes the parent of the root of every subtree that holds
# an earlier column sharing a row with it (Liu 1990). `prev[i]` is the last column seen with
# row i; `ancestor` is a path-compressed pointer toward the subtree root.
function col_etree(A::SparseMatrixCSC, q::Vector{Int})
    m, n = size(A)
    Ap = getcolptr(A)
    Ai = rowvals(A)
    parent = zeros(Int, n)
    ancestor = zeros(Int, n)
    prev = zeros(Int, m)
    @inbounds for j in 1:n
        for p in Ap[q[j]]:(Ap[q[j] + 1] - 1)
            i = Ai[p]
            k = prev[i]
            while k != 0 && k != j
                knext = ancestor[k]
                ancestor[k] = j
                if knext == 0
                    parent[k] = j
                    break
                end
                k = knext
            end
            prev[i] = j
        end
    end
    return parent
end

# Subtrees of the postordered column elimination tree with at most this many columns form
# one panel (SuperLU's relaxed supernodes).
const RELAX_SUBTREE = 16

# Column order and panels for the unsymmetric strategy: COLAMD on A (or the order `q0`
# when given), the column elimination tree postordered, relaxed subtrees as panels and the
# remaining columns grouped in chains of single children, at most `maxpanel` columns each.
function rect_panel_analysis(
        A::SparseMatrixCSC; maxpanel::Int = 256, q0::Union{Nothing, Vector{Int}} = nothing
    )
    n = size(A, 2)
    q0 === nothing && (q0 = colamd_perm(A))
    post = postorder_tree(col_etree(A, q0))
    q = q0[post]
    parent = col_etree(A, q)
    size_ = ones(Int, n)
    nchild = zeros(Int, n)
    @inbounds for j in 1:n
        p = parent[j]
        if p != 0
            size_[p] += size_[j]
            nchild[p] += 1
        end
    end
    panels = [1]
    j = 1
    @inbounds while j <= n
        # the largest relaxed subtree starting at j, if any
        r = j
        best = 0
        while r <= n && r - size_[r] + 1 == j && size_[r] <= RELAX_SUBTREE
            best = r
            r = parent[r]
            r == 0 && break
        end
        if best != 0 && best > j
            last = best
        else
            last = j
            while last < n && parent[last] == last + 1 && nchild[last + 1] == 1 &&
                    last - j + 1 < maxpanel
                last += 1
            end
        end
        push!(panels, last + 1)
        j = last + 1
    end
    return q, panels, 4 * nnz(A)
end
