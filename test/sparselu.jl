# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseLUTests

using Test
using SparseArrays
using SparseArrays: sparselu, SparseLU
using LinearAlgebra
using Random
include("testhelpers.jl")

# A reducible matrix: `nblocks` irreducible diagonal blocks of order `bs` in a scrambled
# order, each coupled to the next. Well conditioned, and no entry equals its conjugate.
function reducible(::Type{T}, nblocks::Int, bs::Int, rng) where {T}
    n = nblocks * bs
    A = spzeros(T, n, n)
    for b in 1:nblocks
        r = (b - 1) * bs .+ (1:bs)
        A[r, r] = rand(rng, T, bs, bs) + 4I
        b < nblocks && (A[r[1], r[end] + 1] = T <: Complex ? T(1, 2) : T(1))
    end
    p = randperm(rng, n)
    return A[p, randperm(rng, n)]
end

@testset "sparselu and sparse right-hand sides, $T" for T in STD_ELTYPES
    rng = MersenneTwister(232)
    A = reducible(T, 5, 3, rng)
    n = size(A, 1)
    D = Matrix(A)
    F = sparselu(A)
    @test F isa SparseLU{T,Int} && size(F) == (n, n)
    L, U = F.L, F.U
    @test istril(L) && istriu(U) && all(isone, diag(L))
    @test L * U ≈ A[F.p, F.q]
    # only the diagonal blocks are factored: L is block diagonal
    @test nnz(L) <= 5 * 6
    b = rand(rng, T, n)
    @test F \ b ≈ D \ b
    @test ldiv!(F, copy(b)) ≈ D \ b
    B = sprand(rng, T, n, 4, 0.1)
    @test mismatch(F \ B, D \ Matrix(B); approx=true) === nothing
    # `\` returns a sparse solution for a sparse right-hand side, of either shape
    @test mismatch(A \ B, D \ Matrix(B); approx=true, Ti=Int) === nothing
    x = sparsevec([F.p[1]], [one(T)], n)
    @test mismatch(A \ x, D \ Vector(x); approx=true, Ti=Int) === nothing
    # the first block feeds no other, so its right-hand side reaches that block only
    @test nnz(A \ x) == 3
    # a triangular matrix is solved by substitution over the reach of the right-hand side
    Lt = tril(A[F.p, F.q]) + 4I
    e = sparsevec([n - 1], [one(T)], n)
    @test mismatch(Lt \ e, Matrix(Lt) \ Vector(e); approx=true) === nothing
    @test nnz(Lt \ e) <= 2
    @test mismatch(LowerTriangular(Lt) \ B, Matrix(Lt) \ Matrix(B); approx=true) === nothing
    @test mismatch(UpperTriangular(copy(Lt')) \ B, Matrix(Lt') \ Matrix(B); approx=true) === nothing
    # `/` goes through `\` of the adjoints, and of the transposes for a triangle
    C = copy(B')
    @test mismatch(C / A, Matrix(C) / D; approx=true) === nothing
    @test mismatch(C / LowerTriangular(Lt), Matrix(C) / Matrix(Lt); approx=true) === nothing
    # a permuted triangular matrix is recognized and is its own U factor
    P = Lt[randperm(rng, n), randperm(rng, n)]
    G = sparselu(P)
    @test SparseArrays._triangularorder(P) !== nothing && SparseArrays._triangularorder(A) === nothing
    @test G.L == I && istriu(G.U) && G.U == P[G.p, G.q]
    @test G \ b ≈ Matrix(P) \ b
    @test mismatch(P \ B, Matrix(P) \ Matrix(B); approx=true) === nothing
    @test_throws SingularException sparselu(sparse(T[1 1; 1 1]))
end

@testset "sparselu: fill-reducing ordering" begin
    # an arrow pointing the wrong way fills in completely in its natural order, and AMD,
    # which `:auto` takes for a symmetric pattern, turns it around
    n = 40
    A = sparse([1:n; fill(1, n - 1); 2:n], [1:n; 2:n; fill(1, n - 1)], [fill(4.0, n); fill(1.0, 2n - 2)])
    b = collect(1.0:n)
    @test nnz(sparselu(A; ordering=:natural).L) == n * (n + 1) ÷ 2
    for ordering in (:auto, :amd)
        F = sparselu(A; ordering)
        @test nnz(F.L) == 2n - 1
        @test F.L * F.U ≈ A[F.p, F.q]
        @test F \ b ≈ Matrix(A) \ b
    end
    # COLAMD on a grid
    k = 12
    T = sparse(SymTridiagonal(fill(2.0, k), fill(-1.0, k - 1)))
    G = kron(T, sparse(I, k, k)) + kron(sparse(I, k, k), T)
    F = sparselu(G; ordering=:colamd)
    @test nnz(F.L) < 0.8 * nnz(sparselu(G; ordering=:natural).L)
    @test F \ ones(k^2) ≈ Matrix(G) \ ones(k^2)
    @test_throws ArgumentError sparselu(A; ordering=:metis)
end

@testset "sparse solves cost their reach" begin
    # a unit lower bidiagonal matrix and a right-hand side near the end: two columns apply
    for n in (20, 200)
        S = opcount_sparse(sparse([1:n; 2:n], [1:n; 1:(n - 1)], 1.0))
        b = opcount_sparse(sparsevec([n - 1], [1.0], n))
        @test mulcount(() -> UnitLowerTriangular(S) \ b) == 1
        @test mulcount(() -> LowerTriangular(S) \ b) == 1
    end
    # independent 2×2 blocks: the solve touches the block of the right-hand side only
    for nblocks in (10, 100)
        n = 2nblocks
        S = opcount_sparse(blockdiag(fill(sparse([4.0 1; 1 3]), nblocks)...))
        F = sparselu(S)
        b = opcount_sparse(sparsevec([5], [1.0], n))
        @test mulcount(() -> F \ b) <= 2
        @test nnz(F \ b) == 2
    end
end

@static if COMPREHENSIVE
@testset "sparselu: random matrices" begin
    rng = MersenneTwister(1988)
    for _ in 1:300
        n = rand(rng, 1:12)
        T = rand(rng, STD_ELTYPES)
        A = sprand(rng, T, n, n, 0.6 * rand(rng)) + (rand(rng) < 0.7 ? 2I : 0I)
        if sprank(A) < n
            @test_throws SingularException sparselu(A)
            @test_throws SingularException A \ sprand(rng, T, n, 0.5)
            continue
        end
        D = Matrix(A)
        cond(D) > 1e8 && continue
        B = sprand(rng, T, n, 3, 0.3)
        b = sprand(rng, T, n, 0.3)
        for prune in (true, false), tol in (1.0, 0.01)
            F = sparselu(A; prune, tol)
            @test F.L * F.U ≈ A[F.p, F.q]
            @test mismatch(F \ B, D \ Matrix(B); approx=true) === nothing
            @test mismatch(F \ b, D \ Vector(b); approx=true) === nothing
            @test F \ Matrix(B) ≈ D \ Matrix(B)
        end
        @test mismatch(A \ B, D \ Matrix(B); approx=true) === nothing
        @test mismatch(A' \ B, D' \ Matrix(B); approx=true) === nothing
        @test mismatch(transpose(A) \ b, transpose(D) \ Vector(b); approx=true) === nothing
    end
end

@testset "sparselu: orderings of large blocks" begin
    rng = MersenneTwister(1990)
    for _ in 1:40
        n = rand(rng, 20:80)
        T = rand(rng, STD_ELTYPES)
        A = sprand(rng, T, n, n, 3 / n) + 4I
        D = Matrix(A)
        B = sprand(rng, T, n, 2, 0.1)
        for ordering in (:auto, :colamd, :amd, :natural), tol in (1.0, 0.1)
            F = sparselu(A; ordering, tol)
            @test isperm(F.p) && isperm(F.q)
            @test F.L * F.U ≈ A[F.p, F.q]
            @test mismatch(F \ B, D \ Matrix(B); approx=true) === nothing
        end
    end
    # `:auto` takes AMD when the diagonal entries are acceptable pivots, which depends on
    # the tolerance, and COLAMD when they are not
    pattern(d) = sparse([1:60; 2:60; 3:60; 1; 1; 2], [1:60; 1:59; 1:58; 60; 59; 60], [fill(d, 60); fill(1.0, 120)])
    S = pattern(4.0)
    @test sparselu(S).q == sparselu(S; ordering=:amd).q != sparselu(S; ordering=:colamd).q
    W = pattern(0.5)
    @test sparselu(W).q == sparselu(W; ordering=:amd).q
    @test sparselu(W; tol=1).q == sparselu(W; ordering=:colamd, tol=1).q
    W = pattern(0.05)
    @test sparselu(W).q == sparselu(W; ordering=:colamd).q
    @test sparselu(W) \ ones(60) ≈ Matrix(W) \ ones(60)
    # A grid whose diagonal passes the pivot test at first, so `:auto` takes AMD, but whose
    # pivots then leave the diagonal: the factors outgrow AMD's prediction, and `:auto`
    # orders again with COLAMD. The values come from a fixed linear congruential sequence.
    k = 12
    T = sparse(SymTridiagonal(zeros(k), ones(k - 1)))
    P = kron(T, sparse(I, k, k)) + kron(sparse(I, k, k), T)
    state = 20
    V = map(1:nnz(P)) do _
        state = (1103515245 * state + 12345) % 2147483648
        u = state / 2147483648
        (u < 0.5 ? -1.0 : 1.0) * (0.5 + abs(2u - 1) / 2)
    end
    G = SparseMatrixCSC(k^2, k^2, copy(getcolptr(P)), copy(rowvals(P)), V) + 0.11I
    F, Famd = sparselu(G), sparselu(G; ordering=:amd)
    @test F.q != Famd.q
    @test nnz(F.L) + nnz(F.U) < 0.9 * (nnz(Famd.L) + nnz(Famd.U))
    @test F.L * F.U ≈ G[F.p, F.q]
    @test F \ ones(k^2) ≈ Matrix(G) \ ones(k^2)
    @test mismatch(G \ sparsevec([1], [1.0], k^2), Matrix(G) \ Vector(sparsevec([1], [1.0], k^2)); approx=true) === nothing
    # the same orderings through the 32-bit index type of the matrix
    A = SparseMatrixCSC{Float64,Int32}(sprand(rng, 50, 50, 0.1) + 4I)
    for ordering in (:colamd, :amd)
        F = sparselu(A; ordering)
        @test F isa SparseLU{Float64,Int32} && F.L * F.U ≈ A[F.p, F.q]
    end
end

@testset "sparselu: pruning changes the work, not the factors" begin
    # exact arithmetic, so that the order of the updates cannot show
    rng = MersenneTwister(1993)
    for _ in 1:30
        n = rand(rng, 30:60)
        P = sprand(rng, n, n, 3 / n) + I
        A = SparseMatrixCSC(n, n, copy(getcolptr(P)), copy(rowvals(P)),
                            Rational{BigInt}.(rand(rng, 1:9, nnz(P))))
        F = try
            sparselu(A)
        catch err
            err isa SingularException || rethrow()
            continue
        end
        G = sparselu(A; prune=false)
        @test F.L == G.L && F.U == G.U && F.p == G.p && F.q == G.q
        @test F.L * F.U == A[F.p, F.q]
        b = SparseVector(n, [1, n], Rational{BigInt}[1, 2])
        x = A \ b
        @test x isa SparseVector{Rational{BigInt},Int} && A * x == b
    end
    # a symmetric pattern is where pruning applies: every column is pruned by its first
    # off-diagonal pivot
    A = sparse(SymTridiagonal(fill(4.0, 50), fill(1.0, 49))) + sparse([1, 50], [50, 1], 1.0)
    @test sparselu(A).L ≈ sparselu(A; prune=false).L
    @test sparselu(A) \ ones(50) ≈ Matrix(A) \ ones(50)
end

@testset "sparselu: pivoting and singular matrices" begin
    # a zero on the matched diagonal forces a row exchange inside the block
    A = sparse([0.0 1 1; 1 0 1; 1 1 0])
    F = sparselu(A)
    @test F.L * F.U ≈ A[F.p, F.q] && F \ [1.0, 2, 3] ≈ Matrix(A) \ [1.0, 2, 3]
    # the diagonal is kept when it is within `tol` of the largest entry
    A = sparse([1.0 2; 3 1e-3])
    @test sparselu(sparse([1.0 0.5; 3 4])).p == [1, 2]
    @test sparselu(sparse([1.0 0.5; 3 4]); tol=1).p == [2, 1]
    @test_throws ArgumentError sparselu(A; tol=2)
    @test_throws DimensionMismatch sparselu(sprand(3, 4, 0.5))
    # numerically singular with a full structural rank, and a stored zero pivot
    @test_throws SingularException sparselu(sparse([1.0 2; 2 4]))
    @test_throws SingularException sparselu(sparse([1, 2], [1, 2], [1.0, 0.0]))
    @test_throws SingularException sparse([1, 2], [1, 2], [1.0, 0.0]) \ sparsevec([1], [1.0], 2)
    # empty
    F = sparselu(spzeros(0, 0))
    @test size(F) == (0, 0) && F \ spzeros(0, 2) == spzeros(0, 2)
    @test_throws DimensionMismatch F \ sprand(3, 2, 0.5)
    @test_throws DimensionMismatch sparse(1.0I, 3, 3) \ sprand(4, 2, 0.5)
    @test_throws DimensionMismatch LowerTriangular(sparse(1.0I, 3, 3)) \ sprand(4, 0.5)
    @test_throws DimensionMismatch ldiv!(sparselu(sparse(1.0I, 3, 3)), ones(4))
end

@testset "sparselu: permuted triangular matrices and runs of 1×1 blocks" begin
    rng = MersenneTwister(1992)
    for _ in 1:300
        n = rand(rng, 0:30)
        T = rand(rng, STD_ELTYPES)
        kind = rand(rng, 1:3)
        if kind == 1       # upper or lower triangular, scrambled
            A = (rand(rng, Bool) ? triu : tril)(sprand(rng, T, n, n, 0.3)) + 2I
        else               # triangular, irreducible, triangular: runs around a larger block
            h = n ÷ 2
            A = blockdiag(sparse(triu(sprand(rng, T, h, h, 0.4))) + 2I, sprand(rng, T, 5, 5, 0.6) + 3I,
                          sparse(triu(sprand(rng, T, 3, 3, 0.5))) + 2I)
            A[1:h, (h + 1):end] = sprand(rng, T, h, 8, 0.2)
        end
        n = size(A, 1)
        A = A[randperm(rng, n), randperm(rng, n)]
        D = Matrix(A)
        (n > 0 && cond(D) > 1e8) && continue
        F = sparselu(A)
        order = SparseArrays._triangularorder(A)
        kind == 1 && n > 0 && @test order !== nothing
        order === nothing || @test istriu(A[order[1], order[2]])
        @test F.L * F.U ≈ A[F.p, F.q]
        b = rand(rng, T, n)
        B = sprand(rng, T, n, 3, 0.3)
        @test F \ b ≈ D \ b
        @test mismatch(F \ B, D \ Matrix(B); approx=true) === nothing
        @test F \ Matrix(B) ≈ D \ Matrix(B)
    end
    # a stored zero on the diagonal of a permuted triangular matrix is singular
    Z = sparse([1, 1, 2], [1, 2, 2], [1.0, 2.0, 0.0])[[2, 1], :]
    @test_throws SingularException sparselu(Z)
    # exact, and another index type
    R = sparse(Rational{BigInt}[2 3 0; 0 1 5; 0 0 4])[[3, 1, 2], [2, 3, 1]]
    F = sparselu(R)
    @test F.L * F.U == R[F.p, F.q] && F \ Rational{BigInt}[1, 2, 3] == Matrix(R) \ [1, 2, 3]
    @test sparselu(SparseMatrixCSC{Float32,Int32}(R)) isa SparseLU{Float32,Int32}
    # a solve that reaches few entries of a permuted triangular matrix stores only those
    n = 200
    U = sparse([1:n; 1:(n - 1)], [1:n; 2:n], 1.0)[randperm(rng, n), :]
    F = sparselu(U)
    x = F \ sparsevec([F.p[3]], [1.0], n)
    @test nnz(x) == 3 && U * x ≈ sparsevec([F.p[3]], [1.0], n)
end

@testset "sparse right-hand sides: types, wrappers and views" begin
    rng = MersenneTwister(708)
    n = 8
    A = sprand(rng, n, n, 0.4) + 3I
    D = Matrix(A)
    B = sprand(rng, n, 3, 0.4)
    b = sprand(rng, n, 0.4)
    for W in (LowerTriangular, UnitLowerTriangular, UpperTriangular, UnitUpperTriangular)
        @test mismatch(W(A) \ B, W(D) \ Matrix(B); approx=true) === nothing
        @test mismatch(W(A) \ b, W(D) \ Vector(b); approx=true) === nothing
        @test mismatch(W(A)' \ B, W(D)' \ Matrix(B); approx=true) === nothing
        @test mismatch(W(view(A, :, 1:n)) \ view(B, :, 2:3), W(D) \ Matrix(B)[:, 2:3]; approx=true) === nothing
        # right division by the triangle, its adjoint and its transpose
        C = sprand(rng, 2, n, 0.5)
        @test mismatch(C / W(A), Matrix(C) / W(D); approx=true) === nothing
        @test mismatch(C / W(A)', Matrix(C) / W(D)'; approx=true) === nothing
        @test mismatch(B' / transpose(W(A)), Matrix(B') / transpose(W(D)); approx=true) === nothing
        @test mismatch(view(C, :, 1:n) / W(A), Matrix(C) / W(D); approx=true) === nothing
    end
    for M in (A, tril(A), triu(A), sparse(Diagonal(A)))
        @test mismatch(M \ view(B, :, 2:3), Matrix(M) \ Matrix(B)[:, 2:3]; approx=true) === nothing
        @test mismatch(M \ view(B, :, 2), Matrix(M) \ Matrix(B)[:, 2]; approx=true) === nothing
        @test mismatch(M \ copy(B')', Matrix(M) \ Matrix(B); approx=true) === nothing
        @test mismatch(B' / M, Matrix(B') / Matrix(M); approx=true) === nothing
        @inferred M \ B
        @inferred M \ b
        @inferred M' \ B
    end
    # a stored zero in the other triangle leaves the matrix triangular
    Z = tril(A)
    Z[1, n] = 1.0
    nonzeros(Z)[end] = 0.0
    @test mismatch(Z \ b, Matrix(Z) \ Vector(b); approx=true) === nothing
    # a missing or zero diagonal is singular, as for a dense triangle, wherever it is
    S = sparse([1.0 0 0; 1 0 0; 1 1 1])
    @test_throws SingularException LowerTriangular(S) \ sparsevec([3], [1.0], 3)
    @test_throws SingularException LowerTriangular(dropzeros(S)) \ sparsevec([3], [1.0], 3)
    @test mismatch(UnitLowerTriangular(S) \ sparsevec([1], [1.0], 3), [1.0, -1, 0]) === nothing
    # element and index types
    Ai = sparse([2 0 0; 1 2 0; 0 1 2])
    @test mismatch(Ai \ sparsevec([1], [1], 3), [0.5, -0.25, 0.125]; Ti=Int) === nothing
    @test mismatch(UnitLowerTriangular(Ai) \ sparsevec([1], [1], 3), [1, -1, 1]; Ti=Int) === nothing
    A32 = SparseMatrixCSC{Float32,Int32}(A)
    X = A32 \ SparseMatrixCSC{Float32,Int32}(B)
    @test mismatch(X, Matrix(A32) \ Matrix{Float32}(B); approx=true, Ti=Int32) === nothing
    @test sparselu(A32) isa SparseLU{Float32,Int32}
    @test mismatch(A \ SparseMatrixCSC{ComplexF64,Int32}(B), D \ Matrix(B); Tv=ComplexF64, Ti=Int, approx=true) === nothing
    # a Hermitian matrix takes the same path
    H = A + A'
    @test mismatch(H \ B, Matrix(H) \ Matrix(B); approx=true) === nothing
    # the factorization of a fixed-pattern matrix
    @test mismatch(SparseArrays.fixed(A) \ B, D \ Matrix(B); approx=true) === nothing
    # the factors may hold more entries than a narrow index type of the matrix counts
    arrow = sparse([1:15; fill(1, 14); 2:15], [1:15; 2:15; fill(1, 14)], [fill(4.0, 15); fill(1.0, 28)])
    A8 = SparseMatrixCSC{Float64,Int8}(blockdiag(arrow, arrow))
    F8 = sparselu(A8)
    @test F8 isa SparseLU{Float64,Int8} && F8.p isa Vector{Int8} && nnz(F8.L) > typemax(Int8)
    x8 = A8 \ SparseVector{Float64,Int8}(sparsevec([2], [1.0], 30))
    @test mismatch(x8, Matrix(A8) \ [0.0; 1.0; zeros(28)]; approx=true, Ti=Int8) === nothing
    # equal candidates: the pivots do not depend on the order of the search, pruned or not
    Teq = sparse([1.0 0 0 1 0 0 1 -1; 1 1 0 1 0 0 0 1; 0 0 1 -1 -1 1 0 0; 1 0 1 0 -1 0 1 1;
                  0 1 0 0 1 0 0 0; 0 0 -1 1 0 1 -1 0; -1 1 0 -1 1 0 2 0; 1 0 0 0 0 -1 1 1])
    @test sparselu(Teq; tol=1, ordering=:natural).p == sparselu(Teq; tol=1, ordering=:natural, prune=false).p
    # printing
    @test occursin("2 diagonal blocks", sprint(show, MIME"text/plain"(), sparselu(sparse([1.0 2; 0 3]))))
    @test propertynames(sparselu(A)) == (:L, :U, :p, :q)
end
end

end # module SparseLUTests
