# This file is a part of Julia. License is MIT: https://julialang.org/license

# Numeric phase: left-looking supernodal LU with threshold partial pivoting, after
# SuperLU (Demmel, Eisenstat, Gilbert, Li, Liu, SIMAX 20(3), 1999).
#
# The factorized matrix is B = (Dr·A·Dc)[rowperm, :], A after the matching and scaling
# (the identity when they are off), and P·B[:, q] = L·U with P the row permutation given
# by `pivrow`. The columns are processed in the panels of `panel_analysis`. For each panel:
#
#   1. A depth-first search over the supernodes of L, from the rows of the panel's columns
#      of B, finds the supernodes that update the panel (Gilbert–Peierls) and the union of
#      the panel's row patterns; reverse postorder is a topological order of the updates.
#   2. Each such supernode is applied to the panel columns it reaches with a unit lower
#      triangular solve on its block rows and a product onto its rows below.
#   3. The rows already pivoted hold U; the others form a dense block that is factored
#      with threshold partial pivoting over all its rows, preferring the diagonal row of
#      B. That block becomes one supernode of L.
#
# A supernode s holds the columns xsup[s]:xsup[s+1]-1 and the rows _rows(L, s), its pivot
# rows first, as a dense column-major block _vals(L, s): the strict lower part is L with a unit
# diagonal, and the upper triangle of the leading square block is the matching part of U.
# The entries of U above a supernode's block are stored by column in Up/Ui/Ux, at pivot
# positions.

# The supernodes of L, stored contiguously: supernode s has the rows
# rowval[rowptr[s]:rowptr[s+1]-1] and the values nzval[valptr[s]:valptr[s+1]-1].
struct Supernodes{Tv}
    rowptr::Vector{Int}
    rowval::Vector{Int}
    valptr::Vector{Int}
    nzval::Vector{Tv}
end

Supernodes{Tv}() where {Tv} = Supernodes{Tv}([1], Int[], [1], Tv[])

@inline _rows(L::Supernodes, s::Int) = @inbounds view(L.rowval, L.rowptr[s]:(L.rowptr[s + 1] - 1))
@inline _vals(L::Supernodes, s::Int) = @inbounds view(L.nzval, L.valptr[s]:(L.valptr[s + 1] - 1))

function _empty!(L::Supernodes)
    resize!(L.rowptr, 1)
    resize!(L.rowval, 0)
    resize!(L.valptr, 1)
    resize!(L.nzval, 0)
    return L
end

Base.copy(L::Supernodes) = Supernodes(copy(L.rowptr), copy(L.rowval), copy(L.valptr), copy(L.nzval))

mutable struct SupernodalLU{Tv, Ti <: Integer, R <: Real} <: Factorization{Tv}
    m::Int
    n::Int
    colperm::Vector{Int}             # column order of the analysis
    panels::Vector{Int}              # panel P holds colperm[panels[P]:panels[P+1]-1]
    qfac::Vector{Int}                # factor column j is B[:, qfac[j]]
    npiv::Int
    nnzl::Int                        # entries of L the analysis predicts
    pivrow::Vector{Int}              # pivot position (1:min(m, n)) -> row of B
    xsup::Vector{Int}
    sn::Supernodes{Tv}               # the supernodes of L and the diagonal blocks of U
    Up::Vector{Int}
    Ui::Vector{Int}
    Ux::Vector{Tv}
    zeropiv::Int                     # first zero pivot position, 0 if none
    finite::Bool                     # every pivot is finite
    matchfail::Bool                  # the matching found the matrix structurally singular
    ordering::Symbol
    matching::Union{Symbol, Bool}
    qinit::Union{Nothing, Vector{Int}}  # the column order the caller gave, if any
    colptr::Vector{Ti}               # pattern of the factorized A, checked by `lu!`
    rowval::Vector{Ti}
    rowperm::Vector{Int}             # B = (Dr·A·Dc)[rowperm, :]
    rscale::Vector{R}                # Dr
    cscale::Vector{R}                # Dc
    matched::Bool
    B::SparseMatrixCSC{Tv, Ti}       # the factorized matrix
    work::NTuple{4, Vector{Tv}}      # solve workspaces
    lock::ReentrantLock
end

nsuper(F::SupernodalLU) = length(F.xsup) - 1

# Threshold for partial pivoting: the diagonal row of B is kept as the pivot when its
# magnitude is at least this fraction of the largest candidate, as in UMFPACK.
const PIVOT_TOLERANCE = 0.1
# Up to this many multiply-adds a supernode update uses scalar loops instead of BLAS.
const UPDATE_SCALAR_MAX = 64

# The workspace of one factorization.
struct PanelWork{Tv}
    mark::Vector{Int}                # row -> last panel that touched it
    relmap::Vector{Int}              # row -> panel row
    perm_r::Vector{Int}              # row -> pivot position, 0 when unpivoted
    supno::Vector{Int}               # pivot position -> supernode
    visited::Vector{Int}             # supernode -> last panel that visited it
    pattern::Vector{Int}
    post::Vector{Int}
    stack_s::Vector{Int}
    stack_p::Vector{Int}
    uloc::Vector{Int}
    cols::Vector{Int}
    ridx::Vector{Int}
    X::Vector{Tv}
    buf1::Vector{Tv}
    buf2::Vector{Tv}
    lp::Vector{Tv}                   # the dense block of a panel
    lrows::Vector{Int}               # its rows
    rowid::Vector{Int}               # its row order after pivoting
    posof::Vector{Int}
    diagloc::Vector{Int}
    corder::Vector{Int}              # its columns in factor order
    cost::Vector{Int}                # the entry count in B of each of its rows
end

PanelWork{Tv}(m::Int, n::Int) where {Tv} = PanelWork{Tv}(zeros(Int, m), zeros(Int, m),
    zeros(Int, m), zeros(Int, min(m, n)), zeros(Int, min(m, n)), Int[], Int[], Int[], Int[],
    Int[], Int[], Int[], Tv[], Tv[], Tv[], Tv[], Vector{Int}(undef, m), Vector{Int}(undef, m),
    Vector{Int}(undef, m), Int[], Int[], Vector{Int}(undef, m))

# Factor F.B into F, whose `colperm`, `panels` and `nnzl` are set. For a square or tall
# matrix column j gets pivot position j, and a column without a nonzero candidate a zero
# pivot. For a wide matrix such a column, and every column once the rows run out, is
# deferred instead: it gets no pivot, moves to the end of the column order and holds only U.
function _factor!(F::SupernodalLU{Tv}) where {Tv}
    B = F.B
    m = F.m
    n = F.n
    npos = min(m, n)
    square = m == n
    wide = m < n
    q = F.colperm
    panels = F.panels
    Bp = getcolptr(B)
    Bi = rowvals(B)
    Bx = nonzeros(B)
    ws = PanelWork{Tv}(m, n)
    rowcount = zeros(Int, m)
    @inbounds for i in Bi
        rowcount[i] += 1
    end
    (; mark, relmap, perm_r, supno, pattern, post, uloc) = ws
    pivrow = resize!(F.pivrow, npos)
    qfac = empty!(F.qfac)
    xsup = empty!(F.xsup)
    push!(xsup, 1)
    L = _empty!(F.sn)
    sizehint!(L.nzval, F.nnzl)
    Up = empty!(F.Up)
    push!(Up, 1)
    Ui = resize!(F.Ui, F.nnzl)
    Ux = resize!(F.Ux, F.nnzl)
    nu = 0
    zeropiv = 0
    finite = true
    nextfree = 1                     # rows below this are pivoted or were used as padding
    kpos = 1                         # the next pivot position
    dcols = Int[]                    # deferred columns, and their U entries
    dptr = [1]
    di = Int[]
    dx = Tv[]
    for P in 1:(length(panels) - 1)
        c1 = panels[P]
        c2 = panels[P + 1] - 1
        w = c2 - c1 + 1
        wp = clamp(npos - kpos + 1, 0, w)  # the panel's columns that get a pivot
        empty!(pattern)
        empty!(post)
        @inbounds for c in c1:c2, p in Bp[q[c]]:(Bp[q[c] + 1] - 1)
            r = Bi[p]
            if mark[r] != P
                mark[r] = P
                push!(pattern, r)
            end
        end
        _panel_dfs!(ws, P, xsup, L)
        # a structurally singular matrix can leave fewer candidate rows than columns; the
        # missing ones are padded with unpivoted rows, which become zero pivots
        nr = 0
        @inbounds for r in pattern
            perm_r[r] == 0 && (nr += 1)
        end
        r = nextfree
        @inbounds while nr < wp
            while perm_r[r] != 0 || mark[r] == P
                r += 1
            end
            mark[r] = P
            push!(pattern, r)
            nr += 1
        end
        nall = length(pattern)
        @inbounds for (li, r) in enumerate(pattern)
            relmap[r] = li
        end
        length(ws.X) < nall * w && resize!(ws.X, nall * w)
        Xm = reshape(view(ws.X, 1:(nall * w)), nall, w)
        fill!(Xm, zero(Tv))
        @inbounds for c in c1:c2, p in Bp[q[c]]:(Bp[q[c] + 1] - 1)
            Xm[relmap[Bi[p]], c - c1 + 1] = Bx[p]
        end
        for t in length(post):-1:1
            _update_panel!(Xm, post[t], xsup, L, ws)
        end
        # rows already pivoted hold U; the rest form the block to factor
        empty!(uloc)
        @inbounds for (li, r) in enumerate(pattern)
            perm_r[r] != 0 && push!(uloc, li)
        end
        length(ws.lp) < nr * w && resize!(ws.lp, nr * w)
        Lp = reshape(view(ws.lp, 1:(nr * w)), nr, w)
        lrows = ws.lrows
        cost = ws.cost
        t = 0
        @inbounds for (li, r) in enumerate(pattern)
            if perm_r[r] == 0
                t += 1
                lrows[t] = r
                cost[t] = rowcount[r]
                relmap[r] = t
                for a in 1:w
                    Lp[t, a] = Xm[li, a]
                end
            end
        end
        diagloc = fill!(resize!(ws.diagloc, wp), 0)
        if square
            @inbounds for a in 1:wp
                d = q[c1 + a - 1]
                (mark[d] == P && perm_r[d] == 0) && (diagloc[a] = relmap[d])
            end
        end
        corder = resize!(ws.corder, w)
        corder .= 1:w
        Lp0 = wide ? copy(Lp) : Lp
        zp = _panel_getrf!(Lp, diagloc, wp, ws.rowid, ws.posof, cost)
        ndef = 0
        while wide && zp != 0
            c = corder[zp]
            deleteat!(corder, zp)
            push!(corder, c)
            ndef += 1
            wp = min(w - ndef, npos - kpos + 1)
            @inbounds for a in 1:w, i in 1:nr
                Lp[i, a] = Lp0[i, corder[a]]
            end
            zp = _panel_getrf!(Lp, fill!(resize!(diagloc, wp), 0), wp, ws.rowid, ws.posof, cost)
        end
        zp != 0 && zeropiv == 0 && (zeropiv = kpos + zp - 1)
        # U above the panel, and for the columns without a pivot also the panel's own rows
        need = nu + w * (length(uloc) + wp)
        if length(Ui) < need
            resize!(Ui, max(need, 2 * length(Ui)))
            resize!(Ux, length(Ui))
        end
        @inbounds for a in 1:w
            ca = corder[a]
            if a <= wp
                for li in uloc
                    v = Xm[li, ca]
                    if !iszero(v)
                        nu += 1
                        Ui[nu] = perm_r[pattern[li]]
                        Ux[nu] = v
                    end
                end
                push!(Up, nu + 1)
                push!(qfac, q[c1 + ca - 1])
            else
                for li in uloc
                    v = Xm[li, ca]
                    if !iszero(v)
                        push!(di, perm_r[pattern[li]])
                        push!(dx, v)
                    end
                end
                for k in 1:wp
                    v = Lp[k, a]
                    if !iszero(v)
                        push!(di, kpos + k - 1)
                        push!(dx, v)
                    end
                end
                push!(dptr, length(di) + 1)
                push!(dcols, q[c1 + ca - 1])
            end
        end
        if wp > 0
            order = ws.rowid
            @inbounds for a in 1:wp
                r = lrows[order[a]]
                finite &= isfinite(Lp[a, a])
                perm_r[r] = kpos + a - 1
                pivrow[kpos + a - 1] = r
                supno[kpos + a - 1] = length(L.rowptr)
            end
            @inbounds for a in 1:nr
                push!(L.rowval, lrows[order[a]])
            end
            push!(L.rowptr, length(L.rowval) + 1)
            append!(L.nzval, view(Lp, :, 1:wp))
            push!(L.valptr, length(L.nzval) + 1)
            push!(xsup, kpos + wp)
            kpos += wp
        end
        @inbounds for r in pattern
            relmap[r] = 0
        end
        @inbounds while nextfree <= m && perm_r[nextfree] != 0
            nextfree += 1
        end
    end
    resize!(Ui, nu)
    resize!(Ux, nu)
    for t in eachindex(dcols)
        append!(Ui, view(di, dptr[t]:(dptr[t + 1] - 1)))
        append!(Ux, view(dx, dptr[t]:(dptr[t + 1] - 1)))
        push!(Up, length(Ui) + 1)
    end
    append!(qfac, dcols)
    F.npiv = kpos - 1
    resize!(pivrow, F.npiv)
    # a wide matrix without full row rank leaves positions without a pivot
    F.npiv < npos && zeropiv == 0 && (zeropiv = F.npiv + 1)
    F.zeropiv = zeropiv
    F.finite = finite
    return F
end

# Depth-first search over the supernodes of L from the pivoted rows of `ws.pattern`,
# adding every row of a visited supernode to the pattern. Leaves the visited supernodes in
# postorder in `ws.post`.
function _panel_dfs!(ws::PanelWork, P::Int, xsup::Vector{Int}, L::Supernodes)
    (; perm_r, supno, visited, pattern, post, stack_s, stack_p) = ws
    @inbounds for t in eachindex(pattern)
        k = perm_r[pattern[t]]
        k == 0 && continue
        s0 = supno[k]
        visited[s0] == P && continue
        _visit!(ws, s0, P, xsup, L)
        while !isempty(stack_s)
            s = stack_s[end]
            rows = _rows(L, s)
            pp = stack_p[end]
            pushed = false
            while pp <= length(rows)
                r = rows[pp]
                pp += 1
                k2 = perm_r[r]
                if k2 != 0
                    s2 = supno[k2]
                    if visited[s2] != P
                        stack_p[end] = pp
                        _visit!(ws, s2, P, xsup, L)
                        pushed = true
                        break
                    end
                end
            end
            if !pushed
                push!(post, s)
                pop!(stack_s)
                pop!(stack_p)
            end
        end
    end
    return nothing
end

# Mark supernode s visited for panel P, add its rows to the pattern, and push it with its
# scan starting below its block (its block rows are its own pivots).
@inline function _visit!(ws::PanelWork, s::Int, P::Int, xsup::Vector{Int}, L::Supernodes)
    (; mark, visited, pattern, stack_s, stack_p) = ws
    visited[s] = P
    @inbounds for r in _rows(L, s)
        if mark[r] != P
            mark[r] = P
            push!(pattern, r)
        end
    end
    push!(stack_s, s)
    push!(stack_p, xsup[s + 1] - xsup[s] + 1)
    return nothing
end

# Apply supernode s to the panel columns it reaches: a unit lower solve on its block rows,
# then a product onto its rows below.
function _update_panel!(Xm::AbstractMatrix{Tv}, s::Int, xsup::Vector{Int},
                        L::Supernodes{Tv}, ws::PanelWork{Tv}) where {Tv}
    (; relmap, cols, ridx, buf1, buf2) = ws
    rows = _rows(L, s)
    V = _vals(L, s)
    nr = length(rows)
    nc = xsup[s + 1] - xsup[s]
    w = size(Xm, 2)
    nb = nr - nc
    length(ridx) < nr && resize!(ridx, nr)
    @inbounds for i in 1:nr
        ridx[i] = relmap[rows[i]]
    end
    if nr * nc * w <= UPDATE_SCALAR_MAX
        @inbounds for a in 1:w, k in 1:nc
            xk = Xm[ridx[k], a]
            iszero(xk) && continue
            off = (k - 1) * nr
            for b in (k + 1):nr
                Xm[ridx[b], a] -= V[off + b] * xk
            end
        end
        return nothing
    end
    # the panel columns with a nonzero in the block rows of s, packed into Seg
    length(buf1) < nc * w && resize!(buf1, nc * w)
    empty!(cols)
    k = 0
    @inbounds for a in 1:w
        nz = false
        for i in 1:nc
            if !iszero(Xm[ridx[i], a])
                nz = true
                break
            end
        end
        nz || continue
        k += 1
        push!(cols, a)
        off = (k - 1) * nc
        for i in 1:nc
            buf1[off + i] = Xm[ridx[i], a]
        end
    end
    k == 0 && return nothing
    Seg = reshape(view(buf1, 1:(nc * k)), nc, k)
    M = reshape(V, nr, nc)
    ldiv!(UnitLowerTriangular(view(M, 1:nc, 1:nc)), Seg)
    @inbounds for (t, a) in enumerate(cols), i in 1:nc
        Xm[ridx[i], a] = Seg[i, t]
    end
    if nb > 0
        length(buf2) < nb * k && resize!(buf2, nb * k)
        T = reshape(view(buf2, 1:(nb * k)), nb, k)
        mul!(T, view(M, (nc + 1):nr, 1:nc), Seg)
        @inbounds for (t, a) in enumerate(cols), b in 1:nb
            Xm[ridx[nc + b], a] -= T[b, t]
        end
    end
    return nothing
end

# Blocked right-looking LU of the dense nr×w panel with threshold partial pivoting over all
# its rows, preferring row diagloc[a] for column a, for its first wp columns; the others
# are only updated. A column with no nonzero candidate keeps its current row as a zero
# pivot. Leaves the row order in rowid[1:nr], whose first wp rows are the pivot rows, and
# returns the first zero pivot's column, 0 if none.
function _panel_getrf!(Lp::AbstractMatrix{Tv}, diagloc::Vector{Int}, wp::Int,
                       rowid::Vector{Int}, posof::Vector{Int}, cost::Vector{Int}) where {Tv}
    nr, w = size(Lp)
    @inbounds for i in 1:nr
        rowid[i] = i
        posof[i] = i
    end
    zp = 0
    nb = 32
    @inbounds for kb in 1:nb:wp
        ke = min(kb + nb - 1, wp)
        for a in kb:ke
            pr = a
            maxv = abs(Lp[a, a])
            for r in (a + 1):nr
                v = abs(Lp[r, a])
                if v > maxv
                    maxv = v
                    pr = r
                end
            end
            if iszero(maxv)
                zp == 0 && (zp = a)
                continue
            end
            d = diagloc[a]
            pd = d == 0 ? 0 : posof[d]
            if pd >= a && abs(Lp[pd, a]) >= PIVOT_TOLERANCE * maxv
                pr = pd
            else
                # among the rows within the threshold, the one with the fewest entries in B,
                # which keeps the fill down as UMFPACK's choice of the sparsest row does
                thr = PIVOT_TOLERANCE * maxv
                best = cost[rowid[pr]]
                for r in a:nr
                    v = abs(Lp[r, a])
                    if v >= thr
                        c = cost[rowid[r]]
                        if c < best || (c == best && v > abs(Lp[pr, a]))
                            best = c
                            pr = r
                        end
                    end
                end
            end
            if pr != a
                for c in 1:w
                    Lp[a, c], Lp[pr, c] = Lp[pr, c], Lp[a, c]
                end
                ra = rowid[a]
                rp = rowid[pr]
                rowid[a] = rp
                rowid[pr] = ra
                posof[rp] = a
                posof[ra] = pr
            end
            piv = Lp[a, a]
            for r in (a + 1):nr
                Lp[r, a] /= piv
            end
            for c in (a + 1):ke
                u = Lp[a, c]
                iszero(u) && continue
                @simd for r in (a + 1):nr
                    Lp[r, c] = muladd(-Lp[r, a], u, Lp[r, c])
                end
            end
        end
        if ke < w
            ldiv!(UnitLowerTriangular(view(Lp, kb:ke, kb:ke)), view(Lp, kb:ke, (ke + 1):w))
            mul!(view(Lp, (ke + 1):nr, (ke + 1):w), view(Lp, (ke + 1):nr, kb:ke),
                 view(Lp, kb:ke, (ke + 1):w), -one(Tv), one(Tv))
        end
    end
    return zp
end

# B = (diag(r)·A·diag(c))[rowperm, :], the matrix the factorization works on when the
# matching and scaling are on.
function _scaled_matrix(A::SparseMatrixCSC{Tv}, rowperm::Vector{Int}, r::Vector, c::Vector) where {Tv}
    B = A[rowperm, :]
    Bp = getcolptr(B)
    Bi = rowvals(B)
    Bx = nonzeros(B)
    @inbounds for j in 1:size(B, 2)
        cj = c[j]
        for p in Bp[j]:(Bp[j + 1] - 1)
            Bx[p] *= r[rowperm[Bi[p]]] * cj
        end
    end
    return B
end

# A square matrix whose pattern symmetry is below this is ordered by COLAMD, without the
# matching, as UMFPACK does: AMD on the pattern of A + Aᵀ overestimates its fill.
const SYMMETRY_THRESHOLD = 0.5

# The fraction of the off-diagonal entries of the square A whose transpose is also stored.
function _pattern_symmetry(A::SparseMatrixCSC)
    noff = 0
    Ai = rowvals(A)
    @inbounds for j in 1:size(A, 2), p in getcolptr(A)[j]:(getcolptr(A)[j + 1] - 1)
        Ai[p] != j && (noff += 1)
    end
    noff == 0 && return 1.0
    # each off-diagonal entry whose transpose is stored appears once in the pattern of A + Aᵀ
    _, ri = sym_pattern(A)
    return (2 * noff - length(ri)) / noff
end

# With partial pivoting the matching is not needed for stability. On circuit matrices, whose
# diagonal has small entries but few missing ones, it made the factorization slower and less
# accurate; on a saddle-point matrix, whose zero block leaves the ordering of A + Aᵀ no
# diagonal to plan the pivots on, it kept the fill and the time several times lower. A
# symmetric pattern takes it when at least this fraction of its diagonal is missing or zero.
const MATCHING_THRESHOLD = 0.05

# Whether the square A takes the unsymmetric strategy, and whether it takes the matching,
# under `ordering = :auto` and `matching = :auto`. An unsymmetric pattern with missing
# diagonal entries is judged after a row permutation to a zero-free diagonal, which is what
# the matching would give: rows permuted away from a symmetric pattern, as in a system whose
# equations are not ordered like its unknowns, take the symmetric strategy with the matching.
function _strategy(A::SparseMatrixCSC)
    bad = _missing_diagonal_fraction(A)
    _pattern_symmetry(A) >= SYMMETRY_THRESHOLD && return (false, bad >= MATCHING_THRESHOLD)
    bad > 0 || return (true, false)
    σ = structural_matching(A)
    (σ === nothing || _pattern_symmetry(A[σ, :]) < SYMMETRY_THRESHOLD) && return (true, false)
    return (false, true)
end

_unsymmetric(A::SparseMatrixCSC) = _strategy(A)[1]

# Analyze and factor A. `ordering` is `:auto`, `:amd` (AMD on A + Aᵀ, the symmetric
# strategy), `:colamd` (the unsymmetric strategy) or `:natural`; a matrix that is not square
# always takes the unsymmetric strategy. `matching` is `:auto` (as `_strategy` decides),
# `true` or `false`.
# `qinit` replaces the fill-reducing order of either strategy.
function _analyze_and_factor(
        A::SparseMatrixCSC{Tv, Ti}, ordering::Symbol, matching::Union{Symbol, Bool},
        qinit::Union{Nothing, Vector{Int}} = nothing; relax::Union{Symbol, Bool} = :auto
    ) where {Tv, Ti}
    m, n = size(A)
    R = real(Tv)
    auto_unsym, auto_match = m == n ? _strategy(A) : (true, false)
    unsym = m != n || ordering === :colamd || (ordering === :auto && auto_unsym)
    ms = nothing
    matchfail = false
    if m == n && (matching === true || (matching === :auto && !unsym && auto_match))
        ms = mc64_matching(A)
        if !ms.ok
            matchfail = true
            ms = nothing
        end
    end
    B = ms === nothing ? A : _scaled_matrix(A, ms.rowperm, ms.r, ms.c)
    q, panels, nnzl = unsym ? rect_panel_analysis(B; q0 = qinit) :
        panel_analysis(B; ordering = ordering === :auto ? :amd : ordering, q0 = qinit, relax)
    F = SupernodalLU{Tv, Ti, R}(
        m, n, q, panels, Int[], 0, nnzl, Vector{Int}(undef, min(m, n)), Int[], Supernodes{Tv}(),
        Int[], Int[], Tv[], 0, true, matchfail, ordering, matching, qinit,
        copy(getcolptr(A)), copy(rowvals(A)),
        ms === nothing ? collect(1:m) : ms.rowperm,
        ms === nothing ? ones(R, m) : ms.r, ms === nothing ? ones(R, n) : ms.c,
        ms !== nothing, ms === nothing ? copy(A) : B, ntuple(_ -> Vector{Tv}(undef, n), 4),
        ReentrantLock()
    )
    return _factor!(F)
end

# Refactor F with the values of A, which has the pattern F was analyzed with.
function _refactor!(F::SupernodalLU{Tv}, A::SparseMatrixCSC{Tv}) where {Tv}
    F.B = F.matched ? _scaled_matrix(A, F.rowperm, F.rscale, F.cscale) : copy(A)
    return _factor!(F)
end
