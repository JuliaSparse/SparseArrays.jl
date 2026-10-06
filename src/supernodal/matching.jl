# This file is a part of Julia. License is MIT: https://julialang.org/license

# Maximum-weighted (max-product) bipartite matching with dual-variable
# scalings — the MC64 job-5 preprocessing the Schenk–Gärtner method applies
# to unsymmetric matrices (Duff & Koster 2001; Olschowka & Neumaier 1996).  Implemented as
# shortest augmenting paths with potentials (sparse Jonker–Volgenant style),
# written fresh from the published algorithm.
#
# Finds a row permutation σ maximizing ∏ |A[σ(j), j]| and scalings r, c such
# that B = diag(r)·A·diag(c) permuted by σ has |B[σ(j),j]| = 1 and all
# |B[i,j]| ≤ 1.  Factorizing that matrix makes static pivoting almost never
# fire, which is exactly why the method turns matching on by default for
# general unsymmetric systems.

struct MatchScale{R}
    rowperm::Vector{Int}   # σ: column j is matched to row rowperm[j]
    r::Vector{R}           # row scalings
    c::Vector{R}           # column scalings
    ok::Bool               # false: structurally singular, fall back to identity
end

function _identity_matchscale(::Type{R}, n::Int) where {R}
    return MatchScale(collect(1:n), ones(R, n), ones(R, n), false)
end

# Max-product matching and scaling of sparse square `A`. A structurally singular matrix
# gets the identity assignment with `ok = false`.
function mc64_matching(A::SparseMatrixCSC{Tv}) where {Tv}
    n = size(A, 2)
    Ap = getcolptr(A)
    Ai = rowvals(A)
    Ax = nonzeros(A)
    nz = nnz(A)

    # entry costs: w = log(colmax / |a|)  (>= 0, 0 for the largest entry), in Float64
    # whatever the element type
    w = Vector{Float64}(undef, nz)
    logcmax = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        cm = 0.0
        for p in Ap[j]:(Ap[j + 1] - 1)
            a = _costabs(Ax[p])
            a > cm && (cm = a)
        end
        (cm == 0 || !isfinite(cm)) && return _identity_matchscale(real(Tv), n)
        logcmax[j] = log(cm)
        for p in Ap[j]:(Ap[j + 1] - 1)
            a = _costabs(Ax[p])
            w[p] = a == 0 ? Inf : logcmax[j] - log(a)
        end
    end

    u = zeros(n)                   # column potentials
    v = zeros(n)                   # row potentials
    colmatch = zeros(Int, n)       # column j -> matched row
    rowmatch = zeros(Int, n)       # row i -> matched column

    # cheap initial assignment: zero-reduced-cost entries
    @inbounds for j in 1:n
        best = Inf
        for p in Ap[j]:(Ap[j + 1] - 1)
            w[p] < best && (best = w[p])
        end
        u[j] = best
    end
    @inbounds for j in 1:n
        for p in Ap[j]:(Ap[j + 1] - 1)
            i = Ai[p]
            if rowmatch[i] == 0 && w[p] - u[j] <= 0.0 + 1.0e-14
                rowmatch[i] = j
                colmatch[j] = i
                break
            end
        end
    end

    # shortest augmenting path for every unmatched column
    d = fill(Inf, n)               # tentative distance to each row
    pred = zeros(Int, n)           # predecessor column on the path to row i
    done = falses(n)
    touched = Int[]
    heap = Vector{Tuple{Float64, Int}}()   # (dist, row) — lazy-deletion binheap

    @inbounds for j0 in 1:n
        colmatch[j0] != 0 && continue
        empty!(heap)
        for i in touched
            d[i] = Inf
            done[i] = false
            pred[i] = 0
        end
        empty!(touched)
        jcur = j0
        dcur = 0.0
        isink = 0
        dstar = Inf
        while true
            for p in Ap[jcur]:(Ap[jcur + 1] - 1)
                i = Ai[p]
                done[i] && continue
                isfinite(w[p]) || continue
                dnew = dcur + (w[p] - u[jcur] - v[i])
                if dnew < d[i] - 1.0e-15
                    d[i] == Inf && push!(touched, i)
                    d[i] = dnew
                    pred[i] = jcur
                    _heap_push!(heap, (dnew, i))
                end
            end
            imin = 0
            while !isempty(heap)
                dm, im = _heap_pop!(heap)
                if !done[im] && dm <= d[im] + 1.0e-15
                    imin = im
                    break
                end
            end
            imin == 0 && break                 # no augmenting path
            done[imin] = true                  # already on `touched` (d was set)
            if rowmatch[imin] == 0
                isink = imin
                dstar = d[imin]
                break
            end
            dcur = d[imin]
            jcur = rowmatch[imin]
        end
        isink == 0 && return _identity_matchscale(real(Tv), n)
        # dual updates for finalized rows and their matched columns
        for i in touched
            if done[i] && i != isink
                v[i] += d[i] - dstar
                u[rowmatch[i]] += dstar - d[i]
            end
        end
        u[j0] += dstar
        # augment along the predecessor chain
        i = isink
        while true
            j = pred[i]
            inext = colmatch[j]
            colmatch[j] = i
            rowmatch[i] = j
            j == j0 && break
            i = inext
        end
    end

    # scalings: r_i = exp(v_i), c_j = exp(u_j)/colmax_j gives |B| <= 1 with
    # ones on the matched diagonal. They are clamped below √floatmax, so that a scale and
    # the product of a row and a column scale are finite in the element type; any
    # positive scale is valid.
    R = real(Tv)
    bound = (log(floatmax(R)) - 1) / 2
    r = Vector{R}(undef, n)
    c = Vector{R}(undef, n)
    @inbounds for i in 1:n
        r[i] = exp(R(clamp(v[i], -bound, bound)))
    end
    @inbounds for j in 1:n
        c[j] = exp(R(clamp(u[j] - logcmax[j], -bound, bound)))
    end
    rowperm = Vector{Int}(undef, n)
    @inbounds for j in 1:n
        rowperm[j] = colmatch[j]
    end
    return MatchScale(rowperm, r, c, true)
end

# minimal binary min-heap on (dist, row) tuples with lazy deletion
@inline function _heap_push!(h::Vector{Tuple{Float64, Int}}, x::Tuple{Float64, Int})
    push!(h, x)
    k = length(h)
    @inbounds while k > 1
        p = k >> 1
        h[p][1] <= h[k][1] && break
        h[p], h[k] = h[k], h[p]
        k = p
    end
    return nothing
end

@inline function _heap_pop!(h::Vector{Tuple{Float64, Int}})
    @inbounds top = h[1]
    @inbounds h[1] = h[end]
    pop!(h)
    k = 1
    m = length(h)
    @inbounds while true
        l = 2k
        l > m && break
        c = (l < m && h[l + 1][1] < h[l][1]) ? l + 1 : l
        h[k][1] <= h[c][1] && break
        h[k], h[c] = h[c], h[k]
        k = c
    end
    return top
end

@inline _costabs(x::Number) = Float64(abs(x))

# The fraction of the columns of A whose diagonal entry is missing or zero.
function _missing_diagonal_fraction(A::SparseMatrixCSC)
    n = size(A, 2)
    Ap = getcolptr(A)
    Ai = rowvals(A)
    Ax = nonzeros(A)
    bad = 0
    @inbounds for j in 1:n
        found = false
        for p in Ap[j]:(Ap[j + 1] - 1)
            if Ai[p] == j
                found = !iszero(Ax[p])
                break
            end
        end
        found || (bad += 1)
    end
    return n == 0 ? 0.0 : bad / n
end

# Row permutation σ such that A[σ, :] has a zero-free diagonal, from a maximum transversal
# (Duff, ACM TOMS 7(3), 1981: depth-first augmenting paths with a cheap lookahead for an
# unmatched row), or `nothing` when A is structurally singular or the search examines more
# than `budget` entries. Ignores the values; it only serves to judge the pattern symmetry
# the weighted matching would give, and a matrix whose rows are merely permuted away from a
# symmetric pattern matches mostly through the lookahead.
function structural_matching(A::SparseMatrixCSC; budget::Int = 4 * nnz(A) + size(A, 2))
    n = size(A, 2)
    Ap = getcolptr(A)
    Ai = rowvals(A)
    rowmatch = zeros(Int, n)
    colmatch = zeros(Int, n)
    cheap = Vector{Int}(undef, n)    # lookahead position in each column
    visited = zeros(Int, n)          # row -> the search that visited it
    stack = Vector{Int}(undef, n)    # columns on the current path
    pos = Vector{Int}(undef, n)      # their next row position
    @inbounds for j in 1:n
        cheap[j] = Ap[j]
    end
    work = 0
    @inbounds for j0 in 1:n
        top = 1
        stack[1] = j0
        pos[1] = Ap[j0]
        found = 0
        while top > 0
            j = stack[top]
            # lookahead: an unmatched row of column j ends the path
            while cheap[j] < Ap[j + 1]
                i = Ai[cheap[j]]
                cheap[j] += 1
                work += 1
                if rowmatch[i] == 0
                    found = i
                    break
                end
            end
            found != 0 && break
            # otherwise descend through a matched row not yet visited in this search
            descended = false
            while pos[top] < Ap[j + 1]
                i = Ai[pos[top]]
                pos[top] += 1
                (work += 1) > budget && return nothing
                if visited[i] != j0
                    visited[i] = j0
                    top += 1
                    stack[top] = rowmatch[i]
                    pos[top] = Ap[rowmatch[i]]
                    descended = true
                    break
                end
            end
            descended || (top -= 1)
        end
        found == 0 && return nothing
        # augment: each column on the path takes the row it descended through, and the last
        # takes the unmatched row
        i = found
        while top > 0
            j = stack[top]
            inext = colmatch[j]
            colmatch[j] = i
            rowmatch[i] = j
            i = inext
            top -= 1
        end
    end
    return colmatch
end
