# This file is a part of Julia. License is MIT: https://julialang.org/license

# Long-tail regression tests for sparse products. Each testset names the issue it guards;
# they run only when the `torture` suite is selected.

module TortureMatmulTests
using Test
using SparseArrays
using LinearAlgebra
include("../testhelpers.jl")
using SparseArrays: AbstractSparseMatrixCSC, getcolptr, rowvals, nonzeros, fixed
using Random

@testset "Issue #33169" begin
    m21 = sparse([1, 2], [2, 2], SimpleSMatrix{2,1}.([rand(2, 1), rand(2, 1)]), 2, 2)
    m12 = sparse([1, 2], [2, 2], SimpleSMatrix{1,2}.([rand(1, 2), rand(1, 2)]), 2, 2)
    m22 = sparse([1, 2], [2, 2], SimpleSMatrix{2,2}.([rand(2, 2), rand(2, 2)]), 2, 2)
    m23 = sparse([1, 2], [2, 2], SimpleSMatrix{2,3}.([rand(2, 3), rand(2, 3)]), 2, 2)
    v12 = sparsevec([2], SimpleSMatrix{1,2}.([rand(1, 2)]))
    v21 = sparsevec([2], SimpleSMatrix{2,1}.([rand(2, 1)]))
    @test m22 * m21 ≈ Matrix(m22) * Matrix(m21)
    @test m22' * m21 ≈ Matrix(m22') * Matrix(m21)
    @test m21' * m22 ≈ Matrix(m21') * Matrix(m22)
    @test m23' * m22 * m21 ≈ Matrix(m23') * Matrix(m22) * Matrix(m21)
    @test m21 * v12 ≈ Matrix(m21) * Vector(v12)
    @test m12' * v12 ≈ Matrix(m12') * Vector(v12)
    @test v21' * m22 ≈ Vector(v21)' * Matrix(m22)
    @test v12' * m21' ≈ Vector(v12)' * Matrix(m21)'
    @test v21' * v21 ≈ Vector(v21)' * Vector(v21)
    @test v21' * m22 * v21 ≈ Vector(v21)' * Matrix(m22) * Vector(v21)
end

#PR #29045
@testset "Issue #28934" begin
    A = sprand(5,5,0.5)
    D = Diagonal(rand(5))
    C = copy(A)
    m1 = which(mul!, Base.typesof(C,A,D,true,false))
    m2 = which(mul!, Base.typesof(C,D,A,true,false))
    @test m1.module == SparseArrays
    @test m2.module == SparseArrays
end

# Five random draws; the core suite runs one.
@testset "matrix-vector multiplication (non-square)" begin
    for i = 1:5
        a = sprand(10, 5, 0.5)
        b = rand(5)
        @test maximum(abs.(a*b - Array(a)*b)) < 100*eps()
    end
end

# Ten random draws; the core suite runs one.
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

# The real eltype of the symmetric/Hermitian sparse product grid; the core suite keeps
# ComplexF64.
@testset "symmetric/Hermitian sparse times sparse, real eltype" begin
    n = 10
    @testset "$T" for T in (Float64,)
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
end

# The full 9 x 9 wrapper product; the core suite pairs each left wrapper with three.
@testset "destination array density in multiplication, all wrapper pairs" begin
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

# The real eltype of the dense times symmetric/Hermitian sparse grid; the core suite
# keeps ComplexF64.
@testset "Dense times symmetric/Hermitian sparse matrix multiplication, real eltype" begin
    A = [1 3; 2 4]
    As = sparse(A)
    B = [1 1; 1 1]
    @test mul!(copy(B), B, Hermitian(A), true, true) == mul!(copy(B), B, Hermitian(As), true, true)

    rng = Random.MersenneTwister(1)
    n = 20
    @testset "$T, $S($U)" for T in (Float64,), S in (Symmetric, Hermitian), U in (:U, :L)
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

# Both sizes and the full coefficient grid of the in-place sparse-sparse product; the
# core suite runs one size with zipped coefficients.
@testset "in-place sparse-sparse mul!, full grid" begin
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
end

# The real eltype of the adjoint/transpose times Diagonal loop; the core suite keeps
# ComplexF64.
@testset "scaling adjoints and transposes with a Diagonal, real eltype (issue #619)" begin
    # adjoint/transpose of a sparse matrix with a Diagonal (issue #619)
    for T in (Float64,), W in (adjoint, transpose)
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
end

end # module
