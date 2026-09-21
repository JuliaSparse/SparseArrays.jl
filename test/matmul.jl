# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseMatmulTests
# `*`, `mul!`, `lmul!` and `rmul!`. Kept apart from linalg.jl so the two run on separate
# test workers.

using Test
using SparseArrays
using SparseArrays: AbstractSparseMatrixCSC, nonzeroinds, getcolptr, rowvals, nonzeros, fixed, _is_fixed
using LinearAlgebra
using Random
include("forbidproperties.jl")
include("mulcount.jl")

sA = sprandn(3, 7, 0.5)
sC = similar(sA)
dA = Array(sA)

const BASE_TEST_PATH = joinpath(Sys.BINDIR, "..", "share", "julia", "test")
isdefined(Main, :Quaternions) || @eval Main include(joinpath($(BASE_TEST_PATH), "testhelpers", "Quaternions.jl"))
using .Main.Quaternions

@testset "matrix-vector multiplication (non-square)" begin
    for i = 1:5
        a = sprand(10, 5, 0.5)
        b = rand(5)
        @test maximum(abs.(a*b - Array(a)*b)) < 100*eps()
    end
end

@testset "diagonal - sparse vector mutliplication" begin
    for _ in 1:10
        b = spzeros(10)
        b[1:3] .= 1:3
        A = Diagonal(randn(10))
        @test norm(A * b - A * Vector(b)) <= 10eps()
        @test norm(A * b - Array(A) * b) <= 10eps()
        Ac = Diagonal(randn(Complex{Float64}, 10))
        @test norm(Ac * b - Ac * Vector(b)) <= 10eps()
        @test norm(Ac * b - Array(Ac) * b) <= 10eps()
        @test_throws DimensionMismatch A * [b; 1]
        @test_throws DimensionMismatch A * b[1:end-1]
    end
end

@testset "sparse matrix * BitArray" begin
    A = sprand(5,5,0.3)
    MA = Array(A)
    B = trues(5)
    @test A*B ≈ MA*B
    B = trues(5,5)
    for trA in (identity, adjoint, transpose), trB in (identity, adjoint, transpose)
        @test trA(A) * trB(B) ≈ trA(MA) * trB(B)
        @test trB(B) * trA(A) ≈ trB(B) * trA(MA)
    end
end


@testset "matrix multiplication" begin
    for (m, p, n, q, k) in (
                            (10, 0.7, 5, 0.3, 15),
                            (100, 0.01, 100, 0.01, 20),
                            (100, 0.1, 100, 0.2, 100),
                           )
        a = sprand(m, n, p); ad = Array(a)
        b = sprand(n, k, q); bd = Array(b)
        as = sparse(a')
        bs = sparse(b')
        ab = a * b
        aab = ad * bd
        @test maximum(abs.(ab - aab)) < 100*eps()
        @test a*bs' == ab
        @test as'*b == ab
        @test as'*bs' == ab
        f = Diagonal(rand(n))
        @test Array(a*f) == ad*f
        @test Array(f*b) == f*bd
        A = rand(2n, 2n)
        sA = view(A, 1:2:2n, 1:2:2n); dA = Array(sA)
        @test (sA*b)::Matrix ≈ dA*bd
        @test (a*sA)::Matrix ≈ ad*dA
        @test (sA'b)::Matrix ≈ dA'*bd
        c = sprandn(ComplexF32, n, n, q); cd = Array(c)
        @test (sA*c')::Matrix ≈ dA*cd'
        @test (c'*sA)::Matrix ≈ cd'*dA
        @test (sA'c)::Matrix ≈ dA'*cd
        @test (sA'c')::Matrix ≈ dA'*cd'
    end
end

@testset "symmetric/Hermitian sparse times sparse" begin
    n = 10
    @testset "$T" for T in (Float64, ComplexF64)
        A = sprandn(T, n, n, 0.3); B = sprandn(T, n, n, 0.3)
        for S in (Symmetric(A), Hermitian(A, :L), Symmetric(view(A, :, 1:n)))
            for X in (B, B', transpose(B), UpperTriangular(B), view(B, :, 1:n), Hermitian(B), Symmetric(B, :L))
                @test (S * X)::SparseMatrixCSC ≈ Matrix(S) * Matrix(X)
                @test (X * S)::SparseMatrixCSC ≈ Matrix(X) * Matrix(S)
            end
            for x in (sprandn(T, n, 0.5), view(B, :, 2))
                @test (S * x)::SparseVector ≈ Matrix(S) * Vector(x)
            end
        end
    end
    # one multiplication per pair of matching stored entries, not per element
    P = mulcount_sparse(sparse(1.0I, n, n))
    for f in (() -> Symmetric(P) * P, () -> P * Symmetric(P), () -> P' * Symmetric(P),
              () -> UpperTriangular(P) * Symmetric(P), () -> Symmetric(P) * Symmetric(P, :L))
        @test mulcount(f) == n
    end
    x = sparsevec(fill(MulCount(1.0), n))
    @test mulcount(() -> Symmetric(P) * x) == mulcount(() -> P * x)
end

@testset "Sparse promotion in sparse matmul" begin
    A = SparseMatrixCSC{Float32, Int8}(2, 2, Int8[1, 2, 3], Int8[1, 2], Float32[1., 2.])
    MA = Array(A)
    B = SparseMatrixCSC{ComplexF32, Int32}(2, 2, Int32[1, 2, 3], Int32[1, 2], ComplexF32[1. + im, 2. - im])
    MB = Array(B)
    @test A*transpose(B)                  ≈ MA * transpose(MB)
    @test A*adjoint(B)                    ≈ MA * adjoint(MB)
    @test transpose(A)*B                  ≈ transpose(MA) * MB
    @test transpose(A)*transpose(B)       ≈ transpose(MA) * transpose(MB)
    @test adjoint(B)*A                    ≈ adjoint(MB) * MA
    @test adjoint(B)*adjoint(complex.(A)) ≈ adjoint(MB) * adjoint(Array(complex.(A)))
end

@testset "destination array density in multiplication" begin
    wrappers = (adjoint, transpose, Hermitian, Symmetric, UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular, UpperHessenberg)
    for tA in wrappers
        A = randn(5,5)
        At = tA(A)
        S = sprandn(5,5,0.3)
        St = tA(S)
        for tB in wrappers
            B = sprandn(5,5, 0.3)
            Bt = tB(B)
            C = At*Bt
            @test C ≈ Matrix(At) * Matrix(Bt)
            @test !issparse(C)
            D = St*Bt
            @test D ≈ Matrix(St) * Matrix(Bt)
            @test issparse(D)
        end
        b = sprandn(5, 0.3)
        c = At * b
        @test c ≈ Matrix(At) * Vector(b)
        @test c isa DenseVector
        d = St*b
        @test d ≈ Matrix(St) * Vector(b)
        @test d isa SparseVector
        for T in (Diagonal(randn(5)),
                    Bidiagonal(ones(5), ones(4), :U),
                    Tridiagonal(ones(4), ones(5), ones(4)),
                    SymTridiagonal(ones(5), ones(4)))
            M = St*T
            @test M ≈ Matrix(St) * Matrix(T)
            @test issparse(M)
            N = T*St
            @test N ≈ Matrix(T) * Matrix(St)
            @test issparse(N)
        end
    end
end

@testset "multiplication of special sparse with dense matrix" begin
    # this results in a call of the most generic multiplication code in LinearAlgebra.jl
    A = randn(2, 2)
    S = sparse(A)
    B = rand(1, 2)'
    @test Symmetric(S) * B ≈ Symmetric(A) * B
end

@testset "Symmetric of sparse matrix mul! dense vector" begin
    rng = Random.MersenneTwister(1)
    n = 1000
    p = 0.02
    q = 1 - sqrt(1-p)
    Areal = sprandn(rng, n, n, p)
    Breal = randn(rng, n)
    Acomplex = sprandn(rng, n, n, q) + sprandn(rng, n, n, q) * im
    Bcomplex = Breal + randn(rng, n) * im
    @testset "symmetric/Hermitian sparse multiply with $S($U)" for S in (Symmetric, Hermitian), U in (:U, :L), (A, B) in ((Areal,Breal), (Acomplex,Bcomplex))
        Asym = S(A, U)
        As = sparse(Asym) # takes most time
        # @test which(mul!, (typeof(B), typeof(Asym), typeof(B))).module == SparseArrays
        @test norm(Asym * B - As * B, Inf) <= eps() * n * p * 10
    end
end

@testset "Symmetric of view of sparse matrix mul! dense vector" begin
    rng = Random.MersenneTwister(1)
    n = 1000
    p = 0.02
    q = 1 - sqrt(1-p)
    Areal = view(sprandn(rng, n, n+10, p), :, 6:n+5)
    Breal = randn(rng, n)
    Acomplex = view(sprandn(rng, n, n+10, q) + sprandn(rng, n, n+10, q) * im, :, 6:n+5)
    Bcomplex = Breal + randn(rng, n) * im
    @testset "symmetric/Hermitian sparseview multiply with $S($U)" for S in (Symmetric, Hermitian), U in (:U, :L), (A, B) in ((Areal,Breal), (Acomplex,Bcomplex))
        Asym = S(A, U)
        As = sparse(Asym) # takes most time
        # @test which(mul!, (typeof(B), typeof(Asym), typeof(B))).module == SparseArrays
        @test norm(Asym * B - As * B, Inf) <= eps() * n * p * 10
    end
end

@testset "Dense times symmetric/Hermitian sparse matrix multiplication" begin
    A = [1 3; 2 4]
    As = sparse(A)
    B = [1 1; 1 1]
    @test mul!(copy(B), B, Hermitian(A), true, true) == mul!(copy(B), B, Hermitian(As), true, true)

    rng = Random.MersenneTwister(1)
    n = 20
    @testset "$T, $S($U)" for T in (Float64, ComplexF64), S in (Symmetric, Hermitian), U in (:U, :L)
        P = sprandn(rng, T, n, n + 2, 0.2)
        nonzeros(P)[1] = 0
        C = randn(rng, T, 3, n)
        for A in (S(P[:, 1:n], U), S(view(P, :, 2:n+1), U)),
                X in (randn(rng, T, 3, n), randn(rng, T, n, 3)', transpose(randn(rng, T, n, 3)))
            @test X * A ≈ X * Matrix(A)
            @test mul!(copy(C), X, A, 2, 3) ≈ mul!(copy(C), X, Matrix(A), 2, 3)
        end
        X = S(randn(rng, T, n, n), U)
        C = randn(rng, T, n, n)
        Q = P[:, 1:n]
        for B in (Q, Q', transpose(Q), view(P, :, 2:n+1), Symmetric(Q), Hermitian(Q, :L))
            @test X * B ≈ Matrix(X) * Matrix(B)
            @test mul!(copy(C), X, B, 2, 3) ≈ mul!(copy(C), Matrix(X), Matrix(B), 2, 3)
        end
        @test_throws DimensionMismatch mul!(zeros(T, 3, n + 1), C, S(P[:, 1:n], U))
    end
    # the sparse kernel multiplies by stored zeros, the generic fallback skips them
    @test isequal([Inf 1.0] * Symmetric(sparse([1, 2], [1, 2], [0.0, 1.0])), [NaN 1.0])
    @test isequal(Symmetric([Inf 1.0; 1.0 1.0]) * sparse([1, 2], [1, 2], [0.0, 1.0]), [NaN 1.0; 0.0 1.0])
end

@testset "sparse-dense products take the same dense factors on either side" begin
    rng = Random.MersenneTwister(1)
    n = 12
    @testset "$T" for T in (Float64, ComplexF64)
        S = sprandn(rng, T, n, n, 0.3)
        D = randn(rng, T, n, n)
        C = randn(rng, T, n, n)
        c = randn(rng, T, n)
        # one dense factor per sparse kernel rather than the full grid
        for (A, X, x) in ((S, view(D, [1:n;], :), 1.0:n),
                          (S', view(D, :, [1:n;])', view(D, [1:n;], 1)),
                          (view(S, :, [1:n;]), reshape(1.0:n^2, n, n), 1.0:n),
                          (Symmetric(S), UpperHessenberg(D), view(D, [1:n;], 1)),
                          (Hermitian(S, :L), Hermitian(D, :L), 1.0:n))
            @test X * A ≈ Matrix(X) * Matrix(A)
            @test A * X ≈ Matrix(A) * Matrix(X)
            @test mul!(copy(C), X, A, 2, 3) ≈ mul!(copy(C), Matrix(X), Matrix(A), 2, 3)
            @test mul!(copy(C), A, X, 2, 3) ≈ mul!(copy(C), Matrix(A), Matrix(X), 2, 3)
            @test mul!(copy(c), A, x, 2, 3) ≈ mul!(copy(c), Matrix(A), Vector(x), 2, 3)
            @test mul!(copy(C), A, S, 2, 3) ≈ mul!(copy(C), Matrix(A), Matrix(S), 2, 3)
            @test mul!(copy(C), A, S', 2, 3) ≈ mul!(copy(C), Matrix(A), Matrix(S'), 2, 3)
        end
    end
    # the sparse kernels multiply by stored zeros, the generic fallbacks skip them
    Z = sparse([1, 2], [1, 2], [0.0, 1.0])
    @test isequal(view([Inf 1.0], [1], :) * Z, [NaN 1.0])
    @test isequal(Z * view([Inf 1.0; 1.0 1.0], :, [1]), [NaN; 1.0;;])
    @test isequal(Z * view([Inf, 1.0], [1, 2]), [NaN, 1.0])
    @test isequal(Z * UpperHessenberg([Inf 1.0; 1.0 1.0]), [NaN 0.0; 1.0 1.0])
    @test isequal(Symmetric([Inf 1.0; 1.0 1.0]) * Hermitian(Z), [NaN 1.0; 0.0 1.0])
    @test isequal(mul!(zeros(2, 2), Z, sparse([Inf 1.0; 1.0 1.0])), [NaN 0.0; 1.0 1.0])
    @test isequal(mul!(zeros(2, 2), Z', sparse([Inf 1.0; 1.0 1.0])), [NaN 0.0; 1.0 1.0])
end

@testset "in-place sparse-sparse mul!" begin
    for n in (20, 30)
        sA = sprandn(ComplexF64, n, n, 0.1); A = Array(sA)
        sB = sprandn(ComplexF64, n, n, 0.1); B = Array(sB)
        sC = sprandn(ComplexF64, n, n, 0.1); C = Array(sC)
        a = randn(ComplexF64); b = randn(ComplexF64)
        for (sA, A) in ((sA, A), (view(sA, :, 1:1:n), A[:,1:1:n]))
            for trA in (identity, adjoint, transpose), trB in (identity, adjoint, transpose)
                @test mul!(copy(sC), trA(sA), trB(sB)) ≈ trA(A) * trB(B)
                for α in (true, false, a), β in (true, false, b)
                    @test mul!(copy(sC), trA(sA), trB(sB), α, β) ≈ C*β + trA(A) * trB(B) * α
                end
            end
        end
    end
    A = sprandn(ComplexF64, 8, 8, 0.3); B = sprandn(ComplexF64, 8, 8, 0.3); C = sprandn(ComplexF64, 8, 8, 0.3)
    for W in (Symmetric, Hermitian)
        @test mul!(copy(C), W(A), B, 2, 3) ≈ 2 * W(Matrix(A)) * Matrix(B) + 3 * Matrix(C)
        @test mul!(copy(C), A', W(B, :L), 2, 3) ≈ 2 * Matrix(A)' * W(Matrix(B), :L) + 3 * Matrix(C)
    end
    # a column-view destination is assigned through its parent
    P = sprandn(ComplexF64, 8, 10, 0.3); P0 = copy(P)
    @test mul!(view(P, :, 2:9), A, B', 2, 3) ≈ 2 * Matrix(A) * Matrix(B)' + 3 * Matrix(P0)[:, 2:9]
    @test P[:, [1, 10]] == P0[:, [1, 10]]
    W = view(P, :, [5, 3, 9]); W0 = Matrix(W)
    @test mul!(W, A, B[:, 1:3], true, true) ≈ Matrix(A) * Matrix(B)[:, 1:3] + W0
    @test mul!(view(sparse(ones(2, 2)), :, 1:2), sparse([1.0 0; 0 0]), sparse([1.0 0; 0 0])) == [1 0; 0 0]
    # the destination takes the pattern of the sparse product; the elementwise fallback keeps its own
    @test nnz(mul!(sparse(ones(2, 2)), sparse([1.0 0; 0 0]), sparse([1.0 0; 0 0]))) == 1
    @test mul!(sparse(fill(complex(NaN), 8, 8)), A, B, true, false) ≈ Matrix(A) * Matrix(B)
    X = copy(A); @test mul!(X, X, B) ≈ Matrix(A) * Matrix(B)
    X = copy(B); @test mul!(X, A, X, 2, 3) ≈ 2 * Matrix(A) * Matrix(B) + 3 * Matrix(B)
    @test_throws DimensionMismatch mul!(spzeros(8, 7), A, B)
    # nothing is written when the destination cannot hold the result
    P = sparse([1.0 2; 0 3]); Q = sparse([0.5 0; 1 1])
    Ci = sparse([1 1; 1 1]); @test_throws InexactError mul!(Ci, P, Q); @test Ci == [1 1; 1 1]
    F = SparseArrays.fixed(sparse([1.0 0; 0 1])); @test_throws ArgumentError mul!(F, P, Q); @test F == [1 0; 0 1]
    G = SparseArrays.fixed(sparse(ones(2, 2)))
    @test mul!(G, P, Q, 2, 1) == 2 * Matrix(P) * Matrix(Q) + ones(2, 2)
end

@testset "scaling with * and mul!, rmul!, and lmul!" begin
    b = randn(7)
    @test dA * Diagonal(b) == sA * Diagonal(b)
    @test dA * Diagonal(b) == mul!(sC, sA, Diagonal(b))
    @test dA * Diagonal(b) == rmul!(copy(sA), Diagonal(b))
    b = randn(3)
    @test Diagonal(b) * dA == Diagonal(b) * sA
    @test Diagonal(b) * dA == mul!(sC, Diagonal(b), sA)
    @test Diagonal(b) * dA == lmul!(Diagonal(b), copy(sA))

    # adjoint/transpose of a sparse matrix with a Diagonal (issue #619)
    for T in (Float64, ComplexF64), W in (adjoint, transpose)
        S = sprand(T, 7, 3, 0.5); M = Matrix(S)
        Dl = Diagonal(randn(T, 3)); Dr = Diagonal(randn(T, 7))
        @test W(S) * Dr isa SparseMatrixCSC
        @test Dl * W(S) isa SparseMatrixCSC
        @test W(S) * Dr ≈ W(M) * Dr
        @test Dl * W(S) ≈ Dl * W(M)
        @test Dl * W(S) * Dr ≈ Dl * W(M) * Dr
        @test_throws DimensionMismatch W(S) * Dl
        @test_throws DimensionMismatch Dr * W(S)
        # mixed eltypes promote
        Di = Diagonal(1:7)
        @test W(S) * Di ≈ W(M) * Di
        # 3- and 5-argument mul! reach the same kernels
        C = similar(W(S))
        @test mul!(C, W(S), Dr) === C
        @test C ≈ W(M) * Dr
        @test mul!(C, Dl, W(S)) === C
        @test C ≈ Dl * W(M)
        C0 = sprand(T, 3, 7, 0.5)
        @test mul!(copy(C0), W(S), Dr, 2, 3) ≈ 2 * W(M) * Dr + 3 * Matrix(C0)
        @test mul!(copy(C0), Dl, W(S), 2, 3) ≈ 2 * Dl * W(M) + 3 * Matrix(C0)
        @test mul!(copy(C0), W(S), Dr, 2, 0) ≈ 2 * W(M) * Dr
        @test_throws DimensionMismatch mul!(C, W(S), Dl)
        @test_throws DimensionMismatch mul!(similar(S), Dl, W(S))
        # a destination with another index type goes through a materialized copy
        C32 = SparseMatrixCSC{T,Int32}(spzeros(3, 7))
        @test mul!(C32, Dl, W(S)) ≈ Dl * W(M)
        # so does a destination aliasing the parent
        Q = sprand(T, 5, 5, 0.5); MQ = Matrix(Q); Dq = Diagonal(randn(T, 5))
        @test mul!(Q, W(Q), Dq) ≈ W(MQ) * Dq
        Q = sprand(T, 5, 5, 0.5); MQ = Matrix(Q)
        @test mul!(Q, Dq, W(Q)) ≈ Dq * W(MQ)
        # or sharing its storage
        Q = sprand(T, 5, 5, 0.5); MQ = Matrix(Q)
        Cs = SparseMatrixCSC(5, 5, copy(getcolptr(Q)), copy(rowvals(Q)), nonzeros(Q))
        @test mul!(Cs, W(Q), Dq) ≈ W(MQ) * Dq
        # fixed operands are read, never written
        F = fixed(S)
        @test W(F) * Dr isa AbstractSparseMatrixCSC
        @test W(F) * Dr ≈ W(M) * Dr
        @test Dl * W(F) isa AbstractSparseMatrixCSC
        @test Dl * W(F) ≈ Dl * W(M)
        @test F == S
    end
    # a Diagonal times a fixed matrix keeps the structure, and the fixedness, of the input
    F = fixed(sA)
    let Dl = Diagonal(randn(3)), Dr = Diagonal(randn(7))
        @test Dl * F ≈ Dl * dA
        @test F * Dr ≈ dA * Dr
        @test _is_fixed(Dl * F) && _is_fixed(F * Dr)
    end
    # the kernels touch only the stored entries: exactly nnz(S) scalar multiplications,
    # whereas the generic Diagonal kernel visits every element of the result
    S = mulcount_sparse(sprand(20, 30, 0.2))
    Dl = Diagonal(MulCount.(rand(30))); Dr = Diagonal(MulCount.(rand(20)))
    for W in (adjoint, transpose)
        @test mulcount(() -> W(S) * Dr) == nnz(S)
        @test mulcount(() -> Dl * W(S)) == nnz(S)
        C = similar(W(S))
        @test mulcount(() -> mul!(C, W(S), Dr)) == nnz(S)
        @test mulcount(() -> mul!(C, Dl, W(S))) == nnz(S)
    end
    # as for dense, the product is formed before conversion to the destination eltype,
    # `alpha == 0` ignores `A`, and `beta == 0` ignores `C`
    for W in (identity, adjoint, transpose)
        S = sparse([1e40;;]); D = Diagonal([1e-40])
        @test mul!(spzeros(Float32, 1, 1), W(S), D) == mul!(zeros(Float32, 1, 1), W(Matrix(S)), D)
        @test mul!(spzeros(Float32, 1, 1), D, W(S)) == mul!(zeros(Float32, 1, 1), D, W(Matrix(S)))
        S = sparse([0.5;;]); D = Diagonal([2.0])
        @test mul!(spzeros(Int, 1, 1), W(S), D) == [1;;]
        @test mul!(spzeros(Int, 1, 1), D, W(S)) == [1;;]
        S = sparse([1.0;;]); D = Diagonal(ComplexF64[Inf])
        @test mul!(spzeros(ComplexF64, 1, 1), W(S), D) == [Inf+0im;;]
        @test mul!(spzeros(ComplexF64, 1, 1), D, W(S)) == [Inf+0im;;]
        S = sparse([Inf;;]); D = Diagonal([1.0])
        @test mul!(sparse([NaN;;]), W(S), D, 0, 0) == [0.0;;]
        @test mul!(sparse([NaN;;]), D, W(S), 0, 0) == [0.0;;]
        @test mul!(sparse([3.0;;]), W(S), D, 0, 2) == [6.0;;]
        @test mul!(sparse([3.0;;]), D, W(S), 0, 2) == [6.0;;]
    end
    # a fixed destination whose pattern contains the product's is filled in place; one whose
    # pattern lacks an entry throws and is left untouched
    S = sparse([1.0 0; 0 2]); D = Diagonal([2.0, 3.0])
    for W in (identity, adjoint, transpose), (f, x, y) in ((mul!, W(S), D), (mul!, D, W(S)))
        F = fixed(sparse(ones(2, 2)))
        @test f(F, x, y) === F
        @test F == Matrix(x) * Matrix(y) && nnz(F) == 4 && _is_fixed(F)
        @test f(F, x, y, 2, 3) ≈ 5 * Matrix(x) * Matrix(y)
        G = fixed(sparse([1.0 0; 0 1]))
        S1 = sparse([1.0 1; 0 1])
        @test_throws ArgumentError f(G, W(S1), D)
        @test_throws ArgumentError f(G, W(S1), D, 2, 3)
        @test G == [1 0; 0 1]
    end

    @test dA * 0.5            == sA * 0.5
    @test dA * 0.5            == mul!(sC, sA, 0.5)
    @test dA * 0.5            == rmul!(copy(sA), 0.5)
    @test 0.5 * dA            == 0.5 * sA
    @test 0.5 * dA            == mul!(sC, sA, 0.5)
    @test 0.5 * dA            == lmul!(0.5, copy(sA))
    @test mul!(sC, 0.5, sA)   == mul!(sC, sA, 0.5)

    @testset "inverse scaling with mul!" begin
        bi = inv.(b)
        @test lmul!(Diagonal(bi), copy(dA)) ≈ ldiv!(Diagonal(b), copy(sA))
        @test lmul!(Diagonal(bi), copy(dA)) ≈ ldiv!(transpose(Diagonal(b)), copy(sA))
        @test lmul!(Diagonal(conj(bi)), copy(dA)) ≈ ldiv!(adjoint(Diagonal(b)), copy(sA))
        Aob = Diagonal(b) \ sA
        @test Aob == ldiv!(Diagonal(b), copy(sA))
        @test issparse(Aob)
        @test_throws DimensionMismatch ldiv!(Diagonal(fill(1., length(b)+1)), copy(sA))
        @test_throws LinearAlgebra.SingularException ldiv!(Diagonal(zeros(length(b))), copy(sA))

        dAt = copy(transpose(dA))
        sAt = copy(transpose(sA))
        @test rmul!(copy(dAt), Diagonal(bi)) ≈ rdiv!(copy(sAt), Diagonal(b))
        @test rmul!(copy(dAt), Diagonal(bi)) ≈ rdiv!(copy(sAt), transpose(Diagonal(b)))
        @test rmul!(copy(dAt), Diagonal(conj(bi))) ≈ rdiv!(copy(sAt), adjoint(Diagonal(b)))
        Atob = sAt / Diagonal(b)
        @test Atob == rdiv!(copy(dAt), Diagonal(b))
        @test issparse(Atob)
        @test_throws DimensionMismatch rdiv!(copy(sAt), Diagonal(fill(1., length(b)+1)))
        @test_throws LinearAlgebra.SingularException rdiv!(copy(sAt), Diagonal(zeros(length(b))))
    end

    @testset "non-commutative multiplication" begin
        # non-commutative multiplication
        Avals = Quaternion.(randn(10), randn(10), randn(10), randn(10))
        sA = sparse(rand(1:3, 10), rand(1:7, 10), Avals, 3, 7)
        sC = copy(sA)
        dA = Array(sA)

        b = Quaternion.(randn(7), randn(7), randn(7), randn(7))
        D = Diagonal(b)
        @test Array(sA * D) ≈ dA * D
        @test rmul!(copy(sA), D) ≈ dA * D
        @test mul!(sC, copy(sA), D) ≈ dA * D

        b = Quaternion.(randn(3), randn(3), randn(3), randn(3))
        D = Diagonal(b)
        @test Array(D * sA) ≈ D * dA
        @test lmul!(D, copy(sA)) ≈ D * dA
        @test mul!(sC, D, copy(sA)) ≈ D * dA
    end

    @testset "5-arg mul!" begin
        @testset "merge indices" begin
            # for zero arrays, merge and copy are identical
            A = spzeros(size(sA))
            SparseArrays.mergeinds!(A, sA)
            B = spzeros(size(sA))
            SparseArrays.copyinds!(B, sA)
            @test all(col -> nzrange(A, col) == nzrange(B, col), axes(A,2))
            # for arrays with different indices populated, merge should combine these
            A = spzeros(5,5)
            A[diagind(A,1)] .= 5
            B = spzeros(5,5)
            B[diagind(A,-1)] .= 10
            SparseArrays.mergeinds!(B, A)
            @test rowvals(B) == [2, 1,3, 2,4, 3,5, 4]
            @test [nzrange(B,col) for col in axes(B,2)] == [1:1, 2:3, 4:5, 6:7, 8:8]
            @test nonzeros(B) == [10, 0,10, 0,10, 0,10, 0]
            # for arrays with overlapping indices, merge should only add the extra ones
            A[diagind(A,2)] .= 5
            SparseArrays.mergeinds!(B, A)
            @test rowvals(B) == [2, 1,3, 1,2,4, 2,3,5, 3,4]
            @test [nzrange(B,col) for col in axes(B,2)] == [1:1, 2:3, 4:6, 7:9, 10:11]
            @test nonzeros(B) == [10, 0,10, 0,0,10, 0,0,10, 0,0]
        end
        for sA2 in (similar(sA), sprand(size(sA)..., 0.1))
            nonzeros(sA2) .= 1
            @testset for (alpha, beta) in [(true, false), (true, true), (2,3)]
                D = Diagonal(rand(size(sA,2)))
                @test mul!(copy(sA2), sA, D, alpha, beta) ≈ dA * D * alpha + sA2 * beta
                D = Diagonal(rand(size(sA,1)))
                @test mul!(copy(sA2), D, sA, alpha, beta) ≈ D * dA * alpha + sA2 * beta
            end
        end
    end
end

@testset "diagonal-sandwiched triple multiplication" begin
    S = sprand(4, 6, 0.2)
    D1 = Diagonal(axes(S,1))
    D2 = Diagonal(axes(S,2) .+ 4)
    A = Array(S)
    C = D1 * S * D2
    @test C isa SparseMatrixCSC
    @test C ≈ D1 * A * D2
    C = D2 * S' * D1
    @test C isa SparseMatrixCSC
    @test C ≈ D2 * A' * D1
    C = D1 * view(S, :, :) * D2
    @test C isa SparseMatrixCSC
    @test C ≈ D1 * A * D2

    @test_throws DimensionMismatch D2 * S * D2
    @test_throws DimensionMismatch D1 * S * D1
end

@testset "multiplication of sparse and dense matrices" begin
    function test_mul(A, B)
        expected = Matrix(A) * Matrix(B)
        @test A * B ≈ expected
        C = similar(expected)
        @test mul!(C, A, B) === C
        @test C ≈ expected
    end

    function test_mul_coefficients(A, B)
        expected = Matrix(A) * Matrix(B)
        C = similar(expected)
        ElType = eltype(C)
        general = ElType <: Complex ? ElType(2 + im) : ElType(2)
        vs = (false, true, zero(ElType), one(ElType), general)
        for α in vs, β in vs
            C .= rand.(ElType)
            expected′ = expected .* α .+ C .* β
            @test mul!(C, A, B, α, β) === C
            @test C ≈ expected′
        end
    end

    for ElType in (Int, Float64, ComplexF64, BigFloat)
        SP = sprand(ElType, 10, 10, 0.3)
        D = rand(ElType, 10, 10)
        fs = (identity, adjoint, transpose)
        for f1 in fs, f2 in fs
            test_mul(f1(SP), f2(D))
            test_mul(f1(D), f2(SP))
        end
        # Coefficients branch on the sparse transform and on plain/wrapped dense-left inputs.
        for f in fs
            test_mul_coefficients(f(SP), D)
            test_mul_coefficients(D, f(SP))
        end
        for f in (adjoint, transpose)
            test_mul_coefficients(f(D), SP)
        end
    end
end

# reads of the wrapped matrix are counted, to tell a kernel that copies each strided row
# once from one that rereads it for every stored entry
struct CountedReadsMatrix{T} <: AbstractMatrix{T}
    parent::Matrix{T}
    reads::Base.RefValue{Int}
end
Base.size(X::CountedReadsMatrix) = size(X.parent)
Base.getindex(X::CountedReadsMatrix, i::Int, j::Int) = (X.reads[] += 1; X.parent[i, j])

@testset "product kernels touch stored entries only" begin
    n = 8
    @testset "adjoint dense times adjoint sparse, $T" for T in (Float64, ComplexF64)
        A = sprandn(T, 6, n, 0.5); X = randn(T, n, 5); C0 = randn(T, 5, 6)
        for fx in (adjoint, transpose), fa in (adjoint, transpose)
            @test mul!(copy(C0), fx(X), fa(A), 2, 3) ≈ 2 * fx(X) * fa(Matrix(A)) + 3 * C0
        end
        Xc = CountedReadsMatrix(X, Ref(0))
        @test mul!(copy(C0), Xc', A', 2, 3) ≈ 2 * X' * Matrix(A)' + 3 * C0
        @test Xc.reads[] <= length(X)
    end
    P = mulcount_sparse(sparse(1.0I, n, n))
    one_, two = MulCount(1.0), MulCount(2.0)
    # sparse times sparse into a dense destination: one multiplication per pair of stored entries
    for f in (() -> mul!(fill(one_, n, n), P, P, true, false), () -> mul!(fill(one_, n, n), P', P, true, false),
              () -> mul!(fill(one_, n, n), Symmetric(P), P', true, false))
        @test mulcount(f) == n
    end
    # a symmetric sparse matrix times a sparse vector costs what the plain product does
    x = sparsevec(fill(one_, n)); y = fill(one_, n)
    @test mulcount(() -> mul!(copy(y), Symmetric(P), x, true, false)) == mulcount(() -> mul!(copy(y), P, x, true, false))
    # scaling a column-view destination by `β` stays sparse
    Q = mulcount_sparse(sparse(1.0I, n, n + 1))
    @test mulcount(() -> mul!(view(Q, :, 1:n), P, P, two, two)) <= 4n
    # the adjoint kernel for a sparse vector does not allocate per column
    A = sprandn(400, 400, 0.01); xs = sprandn(400, 0.1); ys = zeros(400)
    mul!(ys, A', xs, 2.0, 0.5)
    @test (@allocated mul!(ys, A', xs, 2.0, 0.5)) < 1000
end

@testset "dimension mismatch error" begin
    fs = [rand, (x, y)->adjoint(rand(y, x)), (x, y)->transpose(rand(y, x)),
          (x, y)->sprand(x, y, 0.5), (x, y)->adjoint(sprand(y, x, 0.5)),
          (x, y)->transpose(sprand(y, x, 0.5))]
    for fA in fs, fB in fs
        mul!(zeros(6, 10), fA(6, 8), fB(8, 10))
        @test_throws DimensionMismatch mul!(zeros(7, 10), fA(6, 8), fB(8, 10))
        @test_throws DimensionMismatch mul!(zeros(6, 11), fA(6, 8), fB(8, 10))
        @test_throws DimensionMismatch mul!(zeros(6, 10), fA(5, 8), fB(8, 10))
        @test_throws DimensionMismatch mul!(zeros(6, 10), fA(6, 9), fB(8, 10))
        @test_throws DimensionMismatch mul!(zeros(6, 10), fA(6, 8), fB(7, 10))
        @test_throws DimensionMismatch mul!(zeros(6, 10), fA(6, 8), fB(8, 9))
    end
end

end # module
