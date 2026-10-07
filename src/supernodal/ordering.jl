# This file is a part of Julia. License is MIT: https://julialang.org/license

# Off-diagonal symmetric pattern of A + Aᵀ as 1-based CSC with sorted row indices. Three
# O(nnz) passes and no comparison sort: (1) bucket-transpose A's pattern, (2) merge column
# j of A with row j of A under a last-claimant marker (deduplicated, unsorted), (3) one
# more bucket pass, a transpose of a symmetric pattern, which sorts every column.
function sym_pattern(A::SparseMatrixCSC)
    n = size(A, 2)
    Ap = getcolptr(A)
    Ai = rowvals(A)
    nz = nnz(A)
    tp = zeros(Int, n + 1)
    @inbounds for p in 1:nz
        tp[Ai[p] + 1] += 1
    end
    tp[1] = 1
    @inbounds for i in 1:n
        tp[i + 1] += tp[i]
    end
    tcur = tp[1:n]
    ti = Vector{Int}(undef, nz)
    @inbounds for j in 1:n
        for p in Ap[j]:(Ap[j + 1] - 1)
            i = Ai[p]
            ti[tcur[i]] = j
            tcur[i] += 1
        end
    end
    mark = zeros(Int, n)
    cpU = Vector{Int}(undef, n + 1)
    riU = Vector{Int}(undef, 2 * nz)
    k = 0
    cpU[1] = 1
    @inbounds for j in 1:n
        mark[j] = j
        for p in Ap[j]:(Ap[j + 1] - 1)
            i = Ai[p]
            if mark[i] != j
                mark[i] = j
                k += 1
                riU[k] = i
            end
        end
        for p in tp[j]:(tp[j + 1] - 1)
            i = ti[p]
            if mark[i] != j
                mark[i] = j
                k += 1
                riU[k] = i
            end
        end
        cpU[j + 1] = k + 1
    end
    resize!(riU, k)
    return permute_pattern(cpU, riU, collect(1:n), n)
end

# AMD on an off-diagonal symmetric 1-based pattern with sorted columns. `perm[k]` is the
# index eliminated at step `k`.
function amd_perm(cp::Vector{Int}, ri::Vector{Int}, n::Int)
    n == 0 && return Int[]
    init_suitesparse()
    Ap = cp .- 1
    Ai = ri .- 1
    P = Vector{Int}(undef, n)
    status = if Int === Int64
        amd_l_order(n, Ap, Ai, P, C_NULL, C_NULL)
    else
        amd_order(n, Ap, Ai, P, C_NULL, C_NULL)
    end
    status >= 0 || error("AMD ordering failed with status $status")
    return P .+= 1
end

# COLAMD column ordering of the m×n `A`, for a factorization with row pivoting.
function colamd_perm(A::SparseMatrixCSC)
    m, n = size(A)
    n == 0 && return Int[]
    init_suitesparse()
    nz = nnz(A)
    alen = Int(Int === Int64 ? colamd_l_recommended(nz, m, n) : colamd_recommended(nz, m, n))
    Ai = zeros(Int, alen)
    Ri = rowvals(A)
    @inbounds for p in 1:nz
        Ai[p] = Ri[p] - 1
    end
    Ap = getcolptr(A) .- 1
    stats = zeros(Int, COLAMD_STATS)
    ok = Int === Int64 ? colamd_l(m, n, alen, Ai, Ap, C_NULL, stats) :
        colamd(m, n, alen, Ai, Ap, C_NULL, stats)
    ok == 1 ||
        error("COLAMD ordering failed with status $(stats[COLAMD_STATUS + 1])")
    return Ap[1:n] .+ 1
end
