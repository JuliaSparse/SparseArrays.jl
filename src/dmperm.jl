# This file is a part of Julia. License is MIT: https://julialang.org/license

# Maximum matching and the Dulmage-Mendelsohn decomposition of the pattern of a sparse
# matrix, after A. Pothen and C.-J. Fan, "Computing the block triangular form of a sparse
# matrix", ACM Trans. Math. Softw. 16(4), 1990, pp. 303-324. Stored entries count as
# nonzeros whatever their value.

# the coarse part a row or column belongs to
const _DM_SQUARE = 0
const _DM_HORIZONTAL = 1
const _DM_VERTICAL = 2

# the pattern of `A` by rows: the columns of row `i` are `colind[rowptr[i]:rowptr[i+1]-1]`
function _rowpattern(A::SparseMatrixCSCOrView)
    m, n = size(A)
    rv = rowvals(A)
    rowptr = zeros(Int, m + 1)
    @inbounds for j in 1:n, k in nzrange(A, j)
        rowptr[rv[k] + 1] += 1
    end
    rowptr[1] = 1
    @inbounds for i in 1:m
        rowptr[i + 1] += rowptr[i]
    end
    next = rowptr[1:m]
    colind = Vector{Int}(undef, rowptr[m + 1] - 1)
    @inbounds for j in 1:n, k in nzrange(A, j)
        i = rv[k]
        colind[next[i]] = j
        next[i] += 1
    end
    return rowptr, colind
end

# Search for an augmenting path from the unmatched column `c` by depth-first search along
# alternating paths, visiting only rows not yet visited in this pass. Before descending
# from a column its rows are scanned for an unmatched one (lookahead); `look` never moves
# back, so that scan costs O(nnz) over the whole matching.
function _augment!(A::SparseMatrixCSCOrView, c::Int, pass::Int, rowmatch::Vector{Int},
                   colmatch::Vector{Int}, look::Vector{Int}, ptr::Vector{Int},
                   visited::Vector{Int}, colstack::Vector{Int}, rowstack::Vector{Int})
    rv = rowvals(A)
    head = 1
    @inbounds colstack[1] = c
    @inbounds ptr[c] = first(nzrange(A, c))
    @inbounds while head > 0
        col = colstack[head]
        stop = last(nzrange(A, col))
        k = look[col]
        row = 0
        while k <= stop
            i = rv[k]
            k += 1
            if rowmatch[i] == 0
                row = i
                break
            end
        end
        look[col] = k
        if row != 0
            for d in head:-1:1
                cc = colstack[d]
                rowmatch[row] = cc
                colmatch[cc] = row
                d > 1 && (row = rowstack[d - 1])
            end
            return true
        end
        k = ptr[col]
        next = 0
        while k <= stop
            i = rv[k]
            k += 1
            if visited[i] != pass
                visited[i] = pass
                rowstack[head] = i
                next = rowmatch[i]
                break
            end
        end
        ptr[col] = k
        if next == 0
            head -= 1
        else
            head += 1
            colstack[head] = next
            ptr[next] = first(nzrange(A, next))
        end
    end
    return false
end

# A maximum matching of the bipartite graph of `A`: `rowmatch[i]` is the column matched to
# row `i` and `colmatch[j]` the row matched to column `j`, or 0. A cheap pass matches each
# column to its first free row; passes of vertex-disjoint augmenting path searches then
# run until one finds no path. O(n * nnz) in the worst case.
function _maxmatching(A::SparseMatrixCSCOrView)
    m, n = size(A)
    rv = rowvals(A)
    rowmatch = zeros(Int, m)
    colmatch = zeros(Int, n)
    look = Vector{Int}(undef, n)
    unmatched = Int[]
    @inbounds for j in 1:n
        r = nzrange(A, j)
        k = first(r)
        while k <= last(r) && rowmatch[rv[k]] != 0
            k += 1
        end
        if k <= last(r)
            rowmatch[rv[k]] = j
            colmatch[j] = rv[k]
            k += 1
        else
            push!(unmatched, j)
        end
        look[j] = k
    end
    isempty(unmatched) && return rowmatch, colmatch
    ptr = Vector{Int}(undef, n)
    visited = zeros(Int, m)
    colstack = Vector{Int}(undef, n)
    rowstack = Vector{Int}(undef, n)
    pass = 0
    while true
        pass += 1
        nleft = 0
        @inbounds for t in eachindex(unmatched)
            c = unmatched[t]
            if !_augment!(A, c, pass, rowmatch, colmatch, look, ptr, visited, colstack, rowstack)
                nleft += 1
                unmatched[nleft] = c
            end
        end
        found = nleft < length(unmatched)
        resize!(unmatched, nleft)
        (found && nleft > 0) || break
    end
    return rowmatch, colmatch
end

"""
    sprank(A)

Return the structural rank of the sparse matrix `A`: the size of a maximum matching of
its rows to its columns through stored entries. It is the largest rank a matrix with
the pattern of `A` can have, so `sprank(A) >= rank(A)`. A stored zero counts as an
entry; use [`dropzeros`](@ref) first to leave it out.

See also [`dmperm`](@ref).

# Examples
```jldoctest
julia> A = sparse([1.0 1.0 0.0; 1.0 1.0 0.0; 0.0 0.0 0.0]);

julia> sprank(A)
2

julia> rank(Matrix(A))
1
```
"""
function sprank(A::SparseMatrixCSCOrView)
    require_one_based_indexing(A)
    return count(!iszero, _maxmatching(A)[2])
end

# Tarjan's strongly connected components of the square part, on the directed graph whose
# vertices are its columns, with an edge from a column to the column matched to each of
# its rows: the matching stands in for a permutation to a zero-free diagonal. Blocks are
# numbered from `nb + 1` in the order the components are completed, which puts every
# entry on or above the block diagonal. Returns the last block number.
function _strongblocks!(rowblock::Vector{Int}, colblock::Vector{Int}, A::SparseMatrixCSCOrView,
                        rowmatch::Vector{Int}, colmatch::Vector{Int}, rowpart::Vector{Int},
                        colpart::Vector{Int}, nb::Int)
    n = size(A, 2)
    rv = rowvals(A)
    index = zeros(Int, n)
    low = Vector{Int}(undef, n)
    ptr = Vector{Int}(undef, n)
    open = Int[]      # visited columns not yet assigned to a component
    path = Int[]      # the columns on the current search path
    count = 0
    @inbounds for root in 1:n
        (colpart[root] == _DM_SQUARE && index[root] == 0) || continue
        count += 1
        index[root] = low[root] = count
        ptr[root] = first(nzrange(A, root))
        push!(open, root)
        push!(path, root)
        while !isempty(path)
            c = path[end]
            stop = last(nzrange(A, c))
            k = ptr[c]
            descended = false
            while k <= stop
                i = rv[k]
                k += 1
                rowpart[i] == _DM_SQUARE || continue
                w = rowmatch[i]
                if index[w] == 0
                    count += 1
                    index[w] = low[w] = count
                    ptr[w] = first(nzrange(A, w))
                    push!(open, w)
                    push!(path, w)
                    descended = true
                    break
                elseif colblock[w] == 0
                    low[c] = min(low[c], index[w])
                end
            end
            ptr[c] = k
            descended && continue
            if low[c] == index[c]
                nb += 1
                while true
                    w = pop!(open)
                    colblock[w] = nb
                    rowblock[colmatch[w]] = nb
                    w == c && break
                end
            end
            pop!(path)
            isempty(path) || (low[path[end]] = min(low[path[end]], low[c]))
        end
    end
    return nb
end

# Connected components of the horizontal part (`part == _DM_HORIZONTAL`, searched from its
# columns) or of the vertical part (searched from its rows; pass the row pattern as the
# column pattern). `ptr`/`ind` list the neighbours of the starting side and `optr`/`oind`
# those of the other side. Blocks are numbered from `nb + 1` in the order of their first
# starting vertex. Returns the last block number.
function _components!(block::Vector{Int}, oblock::Vector{Int}, part::Int, vpart::Vector{Int},
                      opart::Vector{Int}, ptr, ind, optr, oind, nb::Int)
    queue = Int[]
    @inbounds for root in eachindex(block)
        (vpart[root] == part && block[root] == 0) || continue
        nb += 1
        block[root] = nb
        push!(queue, root)
        while !isempty(queue)
            v = pop!(queue)
            for k in ptr[v]:(ptr[v + 1] - 1)
                o = ind[k]
                (opart[o] == part && oblock[o] == 0) || continue
                oblock[o] = nb
                for l in optr[o]:(optr[o + 1] - 1)
                    w = oind[l]
                    if vpart[w] == part && block[w] == 0
                        block[w] = nb
                        push!(queue, w)
                    end
                end
            end
        end
    end
    return nb
end

# stable counting sort of 1:length(block) by block number: the permutation and the
# pointers to the start of each block in it
function _blockorder(block::Vector{Int}, nb::Int)
    ptr = zeros(Int, nb + 1)
    @inbounds for b in block
        ptr[b + 1] += 1
    end
    ptr[1] = 1
    @inbounds for b in 1:nb
        ptr[b + 1] += ptr[b]
    end
    next = ptr[1:nb]
    perm = Vector{Int}(undef, length(block))
    @inbounds for v in eachindex(block)
        b = block[v]
        perm[next[b]] = v
        next[b] += 1
    end
    return perm, ptr
end

# The decomposition in `Int` vectors: `(p, q, rowptr, colptr, nh, ns, nv, colmatch)`, with
# `nh`, `ns` and `nv` the number of horizontal, square and vertical blocks, in that order.
function _dmperm(A::SparseMatrixCSCOrView)
    m, n = size(A)
    rv = rowvals(A)
    rowmatch, colmatch = _maxmatching(A)
    rowpart = fill(_DM_SQUARE, m)
    colpart = fill(_DM_SQUARE, n)
    rowblock = zeros(Int, m)
    colblock = zeros(Int, n)
    work = Int[]
    # horizontal part: what alternating paths from the unmatched columns reach
    @inbounds for j in 1:n
        if colmatch[j] == 0
            colpart[j] = _DM_HORIZONTAL
            push!(work, j)
        end
    end
    anyhorizontal = !isempty(work)
    @inbounds while !isempty(work)
        j = pop!(work)
        for k in nzrange(A, j)
            i = rv[k]
            rowpart[i] == _DM_SQUARE || continue
            rowpart[i] = _DM_HORIZONTAL
            c = rowmatch[i]
            if colpart[c] == _DM_SQUARE
                colpart[c] = _DM_HORIZONTAL
                push!(work, c)
            end
        end
    end
    # vertical part: what alternating paths from the unmatched rows reach
    @inbounds for i in 1:m
        if rowmatch[i] == 0
            rowpart[i] = _DM_VERTICAL
            push!(work, i)
        end
    end
    anyvertical = !isempty(work)
    nh = nv = 0
    if anyhorizontal || anyvertical
        colptr = Vector{Int}(undef, n + 1)
        @inbounds for j in 1:n
            colptr[j] = first(nzrange(A, j))
        end
        colptr[n + 1] = n == 0 ? 1 : last(nzrange(A, n)) + 1
        rowptr, colind = _rowpattern(A)
        @inbounds while !isempty(work)
            i = pop!(work)
            for k in rowptr[i]:(rowptr[i + 1] - 1)
                j = colind[k]
                colpart[j] == _DM_SQUARE || continue
                colpart[j] = _DM_VERTICAL
                r = colmatch[j]
                if rowpart[r] == _DM_SQUARE
                    rowpart[r] = _DM_VERTICAL
                    push!(work, r)
                end
            end
        end
        nh = _components!(colblock, rowblock, _DM_HORIZONTAL, colpart, rowpart,
                          colptr, rv, rowptr, colind, 0)
    end
    nhs = _strongblocks!(rowblock, colblock, A, rowmatch, colmatch, rowpart, colpart, nh)
    ns = nhs - nh
    nb = nhs
    if anyvertical
        nb = _components!(rowblock, colblock, _DM_VERTICAL, rowpart, colpart,
                          rowptr, colind, colptr, rv, nhs)
        nv = nb - nhs
    end
    p, rptr = _blockorder(rowblock, nb)
    q, cptr = _blockorder(colblock, nb)
    # in a square block, put each row opposite the column it is matched to
    @inbounds for t in cptr[nh + 1]:(cptr[nhs + 1] - 1)
        p[rptr[nh + 1] + t - cptr[nh + 1]] = colmatch[q[t]]
    end
    return p, q, rptr, cptr, nh, ns, nv, colmatch
end

"""
    dmperm(A)

Compute the Dulmage-Mendelsohn decomposition of the sparse matrix `A`: row and column
permutations `p` and `q` for which `A[p, q]` is block upper triangular, with as many
diagonal blocks as the pattern of `A` allows. Only the pattern of `A` is used, and a
stored zero counts as an entry.

Return a named tuple with the fields

* `p`, `q`: the row and column permutations.
* `rowblocks`, `colblocks`: the block boundaries. Block `k` has the rows
  `p[rowblocks[k]:rowblocks[k+1]-1]` and the columns `q[colblocks[k]:colblocks[k+1]-1]`,
  and `A[p, q]` is zero below these blocks.
* `coarse`: a vector of four block numbers delimiting the coarse decomposition. Blocks
  `coarse[1]:coarse[2]-1` are the *horizontal* (underdetermined) part, whose blocks
  have more columns than rows; blocks `coarse[2]:coarse[3]-1` the *square* part; and
  blocks `coarse[3]:coarse[4]-1` the *vertical* (overdetermined) part, whose blocks
  have more rows than columns.
* `match`: a maximum matching. `match[j]` is the row matched to column `j`, or zero
  for an unmatched column; `count(!iszero, match)` is [`sprank`](@ref)`(A)`.

The blocks of the square part are square and irreducible, and have a zero-free
diagonal in `A[p, q]`: they are the strongly connected components that a symmetric
permutation of a matrix with a zero-free diagonal can separate. The horizontal and the
vertical part are block diagonal, their blocks being connected components. The three
parts and their blocks do not depend on the maximum matching chosen. Within a block
the columns are in increasing order.

A square matrix is structurally nonsingular exactly when it has a square part only.
Its linear systems can then be solved one diagonal block at a time, from the last
block to the first, which is what `\\` does for a sparse right-hand side.

The algorithm is the one of Pothen and Fan [^PothenFan1990]: a maximum matching by
depth-first search for augmenting paths, in O(`size(A, 2) * nnz(A)`) time in the worst
case and usually close to O(`nnz(A)`), followed by searches that take O(`nnz(A)`).

[^PothenFan1990]: A. Pothen and C.-J. Fan, "Computing the block triangular form of a sparse matrix", ACM Transactions on Mathematical Software 16(4), 1990, pp. 303-324. [doi:10.1145/98267.98287](https://doi.org/10.1145/98267.98287)

# Examples
```jldoctest
julia> A = sparse([1 0 0 1; 1 1 0 0; 0 1 1 0; 0 0 0 1])
4×4 SparseMatrixCSC{Int64, Int64} with 7 stored entries:
 1  ⋅  ⋅  1
 1  1  ⋅  ⋅
 ⋅  1  1  ⋅
 ⋅  ⋅  ⋅  1

julia> d = dmperm(A);

julia> A[d.p, d.q]
4×4 SparseMatrixCSC{Int64, Int64} with 7 stored entries:
 1  1  ⋅  ⋅
 ⋅  1  1  ⋅
 ⋅  ⋅  1  1
 ⋅  ⋅  ⋅  1

julia> d.rowblocks == d.colblocks == [1, 2, 3, 4, 5]
true

julia> d.coarse
4-element Vector{Int64}:
 1
 1
 5
 5
```
"""
function dmperm(A::SparseMatrixCSCOrView{<:Any,Ti}) where {Ti}
    require_one_based_indexing(A)
    p, q, rptr, cptr, nh, ns, nv, colmatch = _dmperm(A)
    return (p = convert(Vector{Ti}, p), q = convert(Vector{Ti}, q),
            rowblocks = convert(Vector{Ti}, rptr), colblocks = convert(Vector{Ti}, cptr),
            coarse = Ti[1, nh + 1, nh + ns + 1, nh + ns + nv + 1],
            match = convert(Vector{Ti}, colmatch))
end
