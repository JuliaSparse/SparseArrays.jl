# This file is a part of Julia. License is MIT: https://julialang.org/license

module DMPermTests

using Test
using SparseArrays
using LinearAlgebra
using Random
include("testhelpers.jl")

# every property `dmperm` documents, checked on the permuted pattern; returns the number of
# properties that fail
function dmperm_failures(@nospecialize(A))
    d = dmperm(A)
    m, n = size(A)
    stored = falses(m, n)
    for j in 1:n, k in nzrange(A, j)
        stored[rowvals(A)[k], j] = true
    end
    P = stored[d.p, d.q]
    nb = length(d.rowblocks) - 1
    bad = 0
    bad += !(isperm(d.p) && length(d.p) == m && isperm(d.q) && length(d.q) == n)
    bad += !(length(d.colblocks) == nb + 1 && d.rowblocks[end] == m + 1 && d.colblocks[end] == n + 1)
    bad += !(d.coarse[1] == 1 && d.coarse[4] == nb + 1 && issorted(d.coarse))
    rows(k) = d.rowblocks[k]:(d.rowblocks[k + 1] - 1)
    cols(k) = d.colblocks[k]:(d.colblocks[k + 1] - 1)
    for k in 1:nb
        r, c = rows(k), cols(k)
        # zero below the block, and its columns are in increasing order
        bad += any(P[(last(r) + 1):m, 1:last(c)])
        bad += !issorted(d.q[c])
        if k < d.coarse[2]
            bad += !(length(c) > length(r))
        elseif k < d.coarse[3]
            bad += !(length(c) == length(r) && all(P[r[t], c[t]] for t in eachindex(r)))
        else
            bad += !(length(c) < length(r))
        end
    end
    # the horizontal and the vertical part are block diagonal
    for part in (d.coarse[1]:(d.coarse[2] - 1), d.coarse[3]:(d.coarse[4] - 1)), k in part, l in part
        k == l || (bad += any(P[rows(k), cols(l)]))
    end
    # the matching uses stored entries, a row at most once, and is maximum
    matched = filter(!iszero, d.match)
    bad += !(allunique(matched) && all(j -> d.match[j] == 0 || d.match[j] in rowvals(A)[nzrange(A, j)], 1:n))
    bad += !(length(matched) == sprank(A))
    return bad
end

# the number of strongly connected components of the graph of a square pattern
function strong_components(P::AbstractMatrix{Bool})
    n = size(P, 1)
    R = P .| Matrix(I, n, n)
    for k in 1:n, i in 1:n, j in 1:n
        R[i, j] |= R[i, k] & R[k, j]
    end
    return length(unique!([findall(R[i, :] .& R[:, i]) for i in 1:n]))
end

@testset "dmperm and sprank" begin
    # all three coarse parts: columns 1:3 share row 1, rows 4:6 share column 6, and the
    # square part is a 2-cycle on rows 2:3 and columns 4:5
    A = sparse([1, 1, 1, 2, 3, 2, 3, 4, 5, 6, 1, 2], [1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 6],
               1.0:12.0, 6, 6)
    d = dmperm(A)
    @test dmperm_failures(A) == 0
    @test d.coarse == [1, 2, 3, 4]
    @test (d.p, d.q) == ([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])
    @test d.rowblocks == [1, 2, 4, 7] && d.colblocks == [1, 4, 6, 7]
    @test sprank(A) == 4 == count(!iszero, d.match)
    @test d isa NamedTuple{(:p, :q, :rowblocks, :colblocks, :coarse, :match),NTuple{6,Vector{Int}}}
    # a cycle with a diagonal is one irreducible block, and without it a permutation, which
    # has n; so has a triangular matrix, in an order that makes it upper triangular
    C = sparse([2, 3, 4, 1], 1:4, 1.0)
    @test dmperm(C + I).colblocks == [1, 5] && dmperm_failures(C + I) == 0
    @test dmperm(C).colblocks == 1:5 && isdiag(C[dmperm(C).p, dmperm(C).q])
    L = sparse(LowerTriangular(ones(4, 4)))
    dL = dmperm(L)
    @test dL.colblocks == 1:5 && istriu(L[dL.p, dL.q])
end

@static if COMPREHENSIVE
@testset "dmperm: random patterns" begin
    rng = MersenneTwister(108)
    for _ in 1:400
        m, n = rand(rng, 0:9), rand(rng, 0:9)
        A = sprand(rng, m, n, 0.5 * rand(rng))
        @test dmperm_failures(A) == 0
        # the structural rank is the rank for generic values
        @test sprank(A) == rank(Matrix(A))
    end
    for _ in 1:40
        n = rand(rng, 20:60)
        A = sprand(rng, n, n, 2.5 / n) + (rand(rng, Bool) ? I : 0I)
        @test dmperm_failures(A) == 0
    end
    # the square blocks are the strongly connected components
    for _ in 1:100
        n = rand(rng, 1:12)
        A = sprand(rng, n, n, 0.2) + I
        @test length(dmperm(A).colblocks) - 1 == strong_components(Matrix(A) .!= 0)
    end
end

@testset "dmperm: corner cases" begin
    for (m, n) in ((0, 0), (0, 3), (3, 0), (3, 3))
        Z = spzeros(m, n)
        @test dmperm_failures(Z) == 0
        @test sprank(Z) == 0
    end
    # a stored zero is an entry
    Z = sparse([1, 2], [1, 2], [0.0, 1.0])
    @test sprank(Z) == 2 && sprank(dropzeros(Z)) == 1
    # the fixtures have an empty row and column and a stored zero
    for (m, n) in FIXTURE_SHAPES
        @test dmperm_failures(fixture(Float64, m, n)) == 0
    end
    # a view of a range of columns, and another index type
    A = sprand(MersenneTwister(1), 8, 12, 0.3)
    V = view(A, :, 3:9)
    @test dmperm_failures(V) == 0
    @test dmperm(V) == dmperm(A[:, 3:9]) && sprank(V) == sprank(A[:, 3:9])
    d = dmperm(SparseMatrixCSC{Float64,Int32}(A))
    @test d.p isa Vector{Int32} && d.coarse isa Vector{Int32} && d == dmperm(A)
    # patterns the cheap matching gets wrong: every column prefers row 1, and a staircase
    # whose last column needs an augmenting path through all the others
    n = 30
    W = sparse([ones(Int, n); 2:n], [1:n; 1:(n - 1)], 1.0, n, n)
    @test sprank(W) == n && dmperm_failures(W) == 0
    St = sparse([1:n; 1:(n - 1)], [1:n; 2:n], 1.0, n, n)[:, n:-1:1]
    @test sprank(St) == n && dmperm_failures(St) == 0
    # a stored diagonal is matched to itself, which keeps a symmetric permutation symmetric
    Sy = sparse(SymTridiagonal(ones(n), ones(n - 1)))
    @test dmperm(Sy).match == 1:n && dmperm(Sy).p == dmperm(Sy).q
    @inferred dmperm(A)
    @inferred sprank(A)
end
end

end # module DMPermTests
