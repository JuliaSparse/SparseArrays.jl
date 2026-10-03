# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseMatmulTests
# `*`, `mul!`, `lmul!` and `rmul!`. Kept apart from linalg.jl so the two run on separate
# test workers.

using Test
using SparseArrays
using SparseArrays: AbstractSparseMatrixCSC, nonzeroinds, getcolptr, rowvals, nonzeros, fixed, _is_fixed
using LinearAlgebra
using Random
include("testhelpers.jl")

sA = sprandn(3, 7, 0.5)
sC = similar(sA)
dA = Array(sA)

# every transform appears once on each side
const TRANSFORM_PAIRS = ((identity, identity), (adjoint, transpose), (transpose, adjoint))

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
        @static if COMPREHENSIVE
        @test norm(A * b - Array(A) * b) <= 10eps()
        Ac = Diagonal(randn(Complex{Float64}, 10))
        @test norm(Ac * b - Ac * Vector(b)) <= 10eps()
        @test norm(Ac * b - Array(Ac) * b) <= 10eps()
        @test_throws DimensionMismatch A * [b; 1]
        end
        @test_throws DimensionMismatch A * b[1:end-1]
    end
end

@testset "sparse matrix * BitArray" begin
    A = sprand(5,5,0.3)
    MA = Array(A)
    B = trues(5)
    @test A*B ≈ MA*B
    B = trues(5,5)
    for (trA, trB) in (@static COMPREHENSIVE ? (TRANSFORM_PAIRS..., (identity, adjoint), (adjoint, identity)) : TRANSFORM_PAIRS[1:1])
        @test trA(A) * trB(B) ≈ trA(MA) * trB(B)
        @static if COMPREHENSIVE
        @test trB(B) * trA(A) ≈ trB(B) * trA(MA)
        end
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
        @static if COMPREHENSIVE
        c = sprandn(ComplexF32, n, n, q); cd = Array(c)
        @test (sA*c')::Matrix ≈ dA*cd'
        @test (c'*sA)::Matrix ≈ cd'*dA
        @test (sA'c)::Matrix ≈ dA'*cd
        @test (sA'c')::Matrix ≈ dA'*cd'
        end
    end
end

@testset "symmetric/Hermitian sparse times sparse" begin
    n = 10
    @testset "$T" for T in (Float64, ComplexF64)
        A = sprandn(T, n, n, 0.3); B = sprandn(T, n, n, 0.3)
        # standard: one pair for each eltype, as the two wrappers differ only for a complex one.
        # Comprehensive: every pair of wrappers as well, alternating between the eltypes.
        std = T <: Complex ? (Hermitian(A, :L), B') : (Symmetric(A), B)
        for S in (@static COMPREHENSIVE ? (Symmetric(A), Hermitian(A, :L), Symmetric(view(A, :, 1:n))) : std[1:1])
            alt = xor(S isa Hermitian, T <: Complex)
            for X in (@static COMPREHENSIVE ? ((S === std[1] && T <: Complex ? std[2:2] : ())..., (B, B', transpose(B), UpperTriangular(B),
                    view(B, :, 1:n), Hermitian(B), Symmetric(B, :L))[(alt ? 2 : 1):2:end]...) : std[2:2])
                @test (S * X)::SparseMatrixCSC ≈ Matrix(S) * Matrix(X)
                @test (X * S)::SparseMatrixCSC ≈ Matrix(X) * Matrix(S)
            end
            for x in (@static COMPREHENSIVE ? (sprandn(T, n, 0.5), view(B, :, 2))[1:(alt ? 0 : 2)] : T <: Complex ? (sprandn(T, n, 0.5),) : ())
                @test (S * x)::SparseVector ≈ Matrix(S) * Vector(x)
            end
        end
    end
    # one multiplication per pair of matching stored entries, not per element
    P = opcount_sparse(sparse(1.0I, n, n))
    for f in ((@static COMPREHENSIVE ? (() -> Symmetric(P) * P, () -> P * Symmetric(P), () -> UpperTriangular(P) * Symmetric(P)) : ())...,
              () -> P' * Symmetric(P), () -> Symmetric(P) * Symmetric(P, :L))
        @test mulcount(f) == n
    end
    x = sparsevec(fill(OpCount(1.0), n))
    @test mulcount(() -> Symmetric(P) * x) == mulcount(() -> P * x)
end

@testset "Sparse promotion in sparse matmul" begin
    # the index and element types are the point: both promote
    A = @static COMPREHENSIVE ? SparseMatrixCSC{Float32, Int8}(2, 2, Int8[1, 2, 3], Int8[1, 2], Float32[1., 2.]) :
        SparseMatrixCSC{Float64, Int16}(2, 2, Int16[1, 2, 3], Int16[1, 2], [1., 2.])
    MA = Array(A)
    B = @static COMPREHENSIVE ? SparseMatrixCSC{ComplexF32, Int32}(2, 2, Int32[1, 2, 3], Int32[1, 2], ComplexF32[1. + im, 2. - im]) :
        SparseMatrixCSC{ComplexF64, Int}(2, 2, [1, 2, 3], [1, 2], [1. + im, 2. - im])
    MB = Array(B)
    @static if COMPREHENSIVE
    @test A*transpose(B)                  ≈ MA * transpose(MB)
    end
    @test A*adjoint(B)                    ≈ MA * adjoint(MB)
    @static if COMPREHENSIVE
    @test transpose(A)*B                  ≈ transpose(MA) * MB
    @test transpose(A)*transpose(B)       ≈ transpose(MA) * transpose(MB)
    @test adjoint(B)*A                    ≈ adjoint(MB) * MA
    @test adjoint(B)*adjoint(complex.(A)) ≈ adjoint(MB) * adjoint(Array(complex.(A)))
    end
end

@testset "destination array density in multiplication" begin
    wrappers = (adjoint, transpose, Hermitian, Symmetric, UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular, UpperHessenberg)
    # standard: a transform, a symmetric and a triangular wrapper, and `Diagonal` and one of the
    # banded types, which share their methods; triangular.jl owns the triangular grid.
    # Comprehensive: each wrapper also meets itself, one other wrapper and one banded type.
    for tA in (@static COMPREHENSIVE ? wrappers : (adjoint, UpperTriangular))
        i = @static COMPREHENSIVE ? findfirst(==(tA), wrappers) : 1
        A = randn(5,5)
        At = tA(A)
        S = sprandn(5,5,0.3)
        St = tA(S)
        for tB in ((tA === adjoint ? (Symmetric,) : tA === UpperTriangular ? (transpose,) : ())...,
                   (@static COMPREHENSIVE ? (tA, wrappers[mod1(i + 4, end)]) : ())...)
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
                    SymTridiagonal(ones(5), ones(4)))[tA === UpperTriangular ? (2:2) : (mod1(i, 4):mod1(i, 4))]
            M = St*T
            @test M ≈ Matrix(St) * Matrix(T)
            @test issparse(M)
            N = T*St
            @test N ≈ Matrix(T) * Matrix(St)
            @test issparse(N)
        end
    end
    # two triangular factors of one kind give that kind; an implicit unit diagonal is made
    # explicit for the kernel
    S = sprandn(5, 5, 0.3); B = sprandn(5, 5, 0.3)
    for tA in (UnitUpperTriangular, LowerTriangular)
        D = tA(S) * tA(B)
        @test D isa tA && issparse(D)
        @test D ≈ Matrix(tA(S)) * Matrix(tA(B))
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
    # both triangles; the two wrappers differ only for a complex eltype
    @testset "symmetric/Hermitian sparse multiply with $S($U)" for (S, U, A, B) in
            ((Symmetric, :U, Areal, Breal), (Symmetric, :L, Areal, Breal),
             (Hermitian, :U, Acomplex, Bcomplex), (Hermitian, :L, Acomplex, Bcomplex),
             (@static COMPREHENSIVE ? ((Symmetric, :U, Acomplex, Bcomplex), (Hermitian, :L, Areal, Breal)) : ())...)
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
    std = (Symmetric, :L, (Acomplex, Bcomplex))
    @testset "symmetric/Hermitian sparseview multiply with $S($U)" for (S, U, (A, B)) in (std, (@static COMPREHENSIVE ?
            filter(t -> t !== std, pairwise((Symmetric, Hermitian), (:U, :L), ((Areal, Breal), (Acomplex, Bcomplex)))) : ())...)
        Asym = S(A, U)
        As = sparse(Asym) # takes most time
        # @test which(mul!, (typeof(B), typeof(Asym), typeof(B))).module == SparseArrays
        @test norm(Asym * B - As * B, Inf) <= eps() * n * p * 10
    end
end

@testset "Dense times symmetric/Hermitian sparse matrix multiplication" begin
    @static if COMPREHENSIVE
    A = [1 3; 2 4]
    As = sparse(A)
    B = [1 1; 1 1]
    @test mul!(copy(B), B, Hermitian(A), true, true) == mul!(copy(B), B, Hermitian(As), true, true)
    end

    rng = Random.MersenneTwister(1)
    n = 20
    # standard: the two wrappers differ only for a complex eltype
    @testset "$T, $S($U)" for (T, S, U) in (@static COMPREHENSIVE ? pairwise((Float64, ComplexF64), (Symmetric, Hermitian), (:U, :L)) :
                                            ((Float64, Symmetric, :U), (ComplexF64, Hermitian, :L)))
        P = sprandn(rng, T, n, n + 2, 0.2)
        nonzeros(P)[1] = 0
        C = randn(rng, T, 3, n)
        for (A, X) in ((S(P[:, 1:n], U), randn(rng, T, 3, n)),
                ((@static COMPREHENSIVE || S === Hermitian) ? ((S(view(P, :, 2:n+1), U), randn(rng, T, n, 3)'),) : ())...,
                (@static COMPREHENSIVE ? ((S(P[:, 1:n], U), transpose(randn(rng, T, n, 3))),) : ())...)
            @test X * A ≈ X * Matrix(A)
            @test mul!(copy(C), X, A, 2, 3) ≈ mul!(copy(C), X, Matrix(A), 2, 3)
        end
        X = S(randn(rng, T, n, n), U)
        C = randn(rng, T, n, n)
        Q = P[:, 1:n]
        for B in (@static COMPREHENSIVE ? (Q, Q', transpose(Q), view(P, :, 2:n+1), Symmetric(Q), Hermitian(Q, :L)) : S === Hermitian ? (Q',) : (Q,))
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
                          (view(S, [n:-1:1;], 1:n), D, c),   # no compressed storage of its own (#56)
                          (Symmetric(S), UpperHessenberg(D), view(D, [1:n;], 1)),
                          (Hermitian(S, :L), Hermitian(D, :L), 1.0:n))[
                # the kernels conjugate only a complex eltype; a real one takes the plain factor
                (@static COMPREHENSIVE ? (T <: Complex ? (1:6) : [1, 3, 5, 6]) : (T <: Complex ? [2, 4, 6] : [1]))]
            @test X * A ≈ Matrix(X) * Matrix(A)
            (@static COMPREHENSIVE || A isa Hermitian) && @test A * X ≈ Matrix(A) * Matrix(X)
            @static if COMPREHENSIVE
            @test mul!(copy(C), X, A, 2, 3) ≈ mul!(copy(C), Matrix(X), Matrix(A), 2, 3)
            end
            @test mul!(copy(C), A, X, 2, 3) ≈ mul!(copy(C), Matrix(A), Matrix(X), 2, 3)
            @test mul!(copy(c), A, x, 2, 3) ≈ mul!(copy(c), Matrix(A), Vector(x), 2, 3)
            @test mul!(copy(C), A, S, 2, 3) ≈ mul!(copy(C), Matrix(A), Matrix(S), 2, 3)
            @static if COMPREHENSIVE
            @test mul!(copy(C), A, S', 2, 3) ≈ mul!(copy(C), Matrix(A), Matrix(S'), 2, 3)
            end
        end
    end
    # a view that is not a column subset multiplies through its sparse copy (#56), and the
    # product with a dense factor is dense
    S = sprandn(rng, 10, 12, 0.3); G = view(S, [4, 1, 1, 9, 7], 2:11); M = Matrix(G)
    X, Y, x, y = randn(rng, 10, 3), randn(rng, 3, 5), randn(rng, 10), randn(rng, 5)
    @test which(mul!, Base.typesof(zeros(5, 3), 'N', 'N', G, X, true, false)).module == SparseArrays
    @test which(mul!, Base.typesof(zeros(3, 10), 'N', 'N', Y, G, true, false)).module == SparseArrays
    @test which(mul!, Base.typesof(zeros(5), 'N', G, x, true, false)).module == SparseArrays
    @test G * X ≈ M * X && (@static COMPREHENSIVE ? G * x ≈ M * x && G' * y ≈ M' * y && y' * G ≈ y' * M : true)
    @test Y * G isa Matrix && Y * G ≈ Y * M
    P = sprandn(rng, 10, 6, 0.3); Q = sprandn(rng, 8, 5, 0.3)
    @test G * P isa SparseMatrixCSC && G * P ≈ M * Matrix(P)
    @static if COMPREHENSIVE
    @test Q * G isa SparseMatrixCSC && Q * G ≈ Matrix(Q) * M
    @test mul!(zeros(5, 6), G, P) ≈ M * Matrix(P)
    end
    # the sparse kernels multiply by stored zeros, the generic fallbacks skip them
    Z = sparse([1, 2], [1, 2], [0.0, 1.0])
    @test isequal(view([Inf 1.0], [1], :) * Z, [NaN 1.0])
    @static if COMPREHENSIVE
    @test isequal(Z * view([Inf 1.0; 1.0 1.0], :, [1]), [NaN; 1.0;;])
    @test isequal(Z * view([Inf, 1.0], [1, 2]), [NaN, 1.0])
    @test isequal(Z * UpperHessenberg([Inf 1.0; 1.0 1.0]), [NaN 0.0; 1.0 1.0])
    @test isequal(Symmetric([Inf 1.0; 1.0 1.0]) * Hermitian(Z), [NaN 1.0; 0.0 1.0])
    end
    @test isequal(mul!(zeros(2, 2), Z, sparse([Inf 1.0; 1.0 1.0])), [NaN 0.0; 1.0 1.0])
    @static if COMPREHENSIVE
    @test isequal(mul!(zeros(2, 2), Z', sparse([Inf 1.0; 1.0 1.0])), [NaN 0.0; 1.0 1.0])
    end
end

@testset "in-place sparse-sparse mul!" begin
    for n in (20, (@static COMPREHENSIVE ? (30,) : ())...)
        sA = sprandn(ComplexF64, n, n, 0.1); A = Array(sA)
        sB = sprandn(ComplexF64, n, n, 0.1); B = Array(sB)
        sC = sprandn(ComplexF64, n, n, 0.1); C = Array(sC)
        a = randn(ComplexF64); b = randn(ComplexF64)
        vA = view(sA, :, 1:1:n)
        # the plain product, each transform once on each side with general coefficients, and
        # a view; the other size takes every pair of the factors, transforms and coefficients.
        # Vectors, so that destructuring the cases is compiled once.
        cases = n == 20 ? (Any[sA, identity, identity, true, false], Any[sA, identity, identity, a, b], Any[sA, adjoint, transpose, a, b],
                           Any[sA, transpose, adjoint, a, b], Any[vA, identity, adjoint, a, b]) :
            pairwise((sA, vA), (identity, adjoint, transpose), (identity, adjoint, transpose), (true, false, a), (true, false, b))
        for (sA, trA, trB, α, β) in cases
            # the three-argument form is the five-argument one with `true, false`
            (n == 30 || α === true) && @test mul!(copy(sC), trA(sA), trB(sB)) ≈ trA(A) * trB(B)
            @test mul!(copy(sC), trA(sA), trB(sB), α, β) ≈ C*β + trA(A) * trB(B) * α
        end
    end
    A = sprandn(ComplexF64, 8, 8, 0.3); B = sprandn(ComplexF64, 8, 8, 0.3); C = sprandn(ComplexF64, 8, 8, 0.3)
    for W in ((@static COMPREHENSIVE ? (Symmetric,) : ())..., Hermitian)
        @static if COMPREHENSIVE
        @test mul!(copy(C), W(A), B, 2, 3) ≈ 2 * W(Matrix(A)) * Matrix(B) + 3 * Matrix(C)
        end
        @test mul!(copy(C), A', W(B, :L), 2, 3) ≈ 2 * Matrix(A)' * W(Matrix(B), :L) + 3 * Matrix(C)
    end
    # a column-view destination is assigned through its parent
    P = sprandn(ComplexF64, 8, 10, 0.3); P0 = copy(P)
    @test mul!(view(P, :, 2:9), A, B', 2, 3) ≈ 2 * Matrix(A) * Matrix(B)' + 3 * Matrix(P0)[:, 2:9]
    @test P[:, [1, 10]] == P0[:, [1, 10]]
    @static if COMPREHENSIVE
    W = view(P, :, [5, 3, 9]); W0 = Matrix(W)
    @test mul!(W, A, B[:, 1:3], true, true) ≈ Matrix(A) * Matrix(B)[:, 1:3] + W0
    @test mul!(view(sparse(ones(2, 2)), :, 1:2), sparse([1.0 0; 0 0]), sparse([1.0 0; 0 0])) == [1 0; 0 0]
    end
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
    for (T, W) in ((ComplexF64, adjoint), (ComplexF64, transpose), (@static COMPREHENSIVE ? ((Float64, adjoint),) : ())...)
        S = sprand(T, 7, 3, 0.5); M = Matrix(S)
        Dl = Diagonal(randn(T, 3)); Dr = Diagonal(randn(T, 7))
        @test W(S) * Dr isa SparseMatrixCSC
        @test Dl * W(S) isa SparseMatrixCSC
        @test W(S) * Dr ≈ W(M) * Dr
        @test Dl * W(S) ≈ Dl * W(M)
        # the transpose shares the kernels, without the conjugation
        @static COMPREHENSIVE || W === adjoint || continue
        @test Dl * W(S) * Dr ≈ Dl * W(M) * Dr
        @test_throws DimensionMismatch W(S) * Dl
        @test_throws DimensionMismatch Dr * W(S)
        @static if COMPREHENSIVE
        # mixed eltypes promote
        Di = Diagonal(1:7)
        @test W(S) * Di ≈ W(M) * Di
        end
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
        @static if COMPREHENSIVE
        # a destination with another index type goes through a materialized copy
        C32 = SparseMatrixCSC{T,Int32}(spzeros(3, 7))
        @test mul!(C32, Dl, W(S)) ≈ Dl * W(M)
        end
        # so does a destination aliasing the parent
        Q = sprand(T, 5, 5, 0.5); MQ = Matrix(Q); Dq = Diagonal(randn(T, 5))
        @test mul!(Q, W(Q), Dq) ≈ W(MQ) * Dq
        @static if COMPREHENSIVE
        Q = sprand(T, 5, 5, 0.5); MQ = Matrix(Q)
        @test mul!(Q, Dq, W(Q)) ≈ Dq * W(MQ)
        # or sharing its storage
        Q = sprand(T, 5, 5, 0.5); MQ = Matrix(Q)
        Cs = SparseMatrixCSC(5, 5, copy(getcolptr(Q)), copy(rowvals(Q)), nonzeros(Q))
        @test mul!(Cs, W(Q), Dq) ≈ W(MQ) * Dq
        end
        # fixed operands are read, never written
        F = fixed(S)
        @test W(F) * Dr isa AbstractSparseMatrixCSC
        @test W(F) * Dr ≈ W(M) * Dr
        @static if COMPREHENSIVE
        @test Dl * W(F) isa AbstractSparseMatrixCSC
        @test Dl * W(F) ≈ Dl * W(M)
        end
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
    S = opcount_sparse(sprand(20, 30, 0.2))
    Dl = Diagonal(OpCount.(rand(30))); Dr = Diagonal(OpCount.(rand(20)))
    for W in (adjoint, (@static COMPREHENSIVE ? (transpose,) : ())...)
        @test mulcount(() -> W(S) * Dr) == nnz(S)
        @test mulcount(() -> Dl * W(S)) == nnz(S)
        @static if COMPREHENSIVE
        C = similar(W(S))
        @test mulcount(() -> mul!(C, W(S), Dr)) == nnz(S)
        @test mulcount(() -> mul!(C, Dl, W(S))) == nnz(S)
        end
    end
    @static if COMPREHENSIVE
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
    end
    # a fixed destination whose pattern contains the product's is filled in place; one whose
    # pattern lacks an entry throws and is left untouched
    S = sparse([1.0 0; 0 2]); D = Diagonal([2.0, 3.0])
    # standard: the plain matrix on the left and the adjoint on the right
    for W in (identity, adjoint, (@static COMPREHENSIVE ? (transpose,) : ())...),
            (f, x, y) in ((mul!, W(S), D), (mul!, D, W(S)))[(@static COMPREHENSIVE ? (1:2) : W === identity ? (1:1) : (2:2))]
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
end

@testset "scaling by a number, and inverse scaling" begin
    b = randn(3)
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
        Quaternion = quaternion_type()
        Avals = Quaternion.(randn(10), randn(10), randn(10), randn(10))
        sA = sparse(rand(1:3, 10), rand(1:7, 10), Avals, 3, 7)
        sC = copy(sA)
        dA = Array(sA)

        b = Quaternion.(randn(7), randn(7), randn(7), randn(7))
        D = Diagonal(b)
        @test Array(sA * D) ≈ dA * D
        @test rmul!(copy(sA), D) ≈ dA * D
        @static if COMPREHENSIVE
        @test mul!(sC, copy(sA), D) ≈ dA * D
        end

        b = Quaternion.(randn(3), randn(3), randn(3), randn(3))
        D = Diagonal(b)
        @test Array(D * sA) ≈ D * dA
        @test lmul!(D, copy(sA)) ≈ D * dA
        @static if COMPREHENSIVE
        @test mul!(sC, D, copy(sA)) ≈ D * dA
        end
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

    @testset "scale" begin
        x = sprand(16, 0.5)
        α = 2.5
        sx = SparseVector(length(x::SparseVector), nonzeroinds(x), nonzeros(x) * α)
        @test exact_equal(x * α, sx)
        @test exact_equal(x * (α + 0.0*im), complex(sx))
        @test exact_equal(α * x, sx)
        @test exact_equal((α + 0.0*im) * x, complex(sx))
        @static if COMPREHENSIVE
        @test exact_equal(x * α, sx)
        @test exact_equal(α * x, sx)
        @test exact_equal(x .* α, sx)
        @test exact_equal(α .* x, sx)
        end
        @test exact_equal(x / α, SparseVector(length(x::SparseVector), nonzeroinds(x), nonzeros(x) / α))

        xc = copy(x)
        @test rmul!(xc, α) === xc
        @test exact_equal(xc, sx)
        xc = copy(x)
        @test lmul!(α, xc) === xc
        @test exact_equal(xc, sx)
        xc = copy(x)
        @test rmul!(xc, complex(α, 0.0)) === xc
        @test exact_equal(xc, sx)
        xc = copy(x)
        @test lmul!(complex(α, 0.0), xc) === xc
        @test exact_equal(xc, sx)
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
    function test_mul(@nospecialize(A), @nospecialize(B))
        expected = Matrix(A) * Matrix(B)
        @test A * B ≈ expected
        C = similar(expected)
        @test mul!(C, A, B) === C
        @test C ≈ expected
    end

    function test_mul_coefficients(@nospecialize(A), @nospecialize(B))
        expected = Matrix(A) * Matrix(B)
        C = similar(expected)
        ElType = eltype(C)
        general = ElType <: Complex ? ElType(2 + im) : ElType(2)
        vs = (false, true, zero(ElType), one(ElType), general)
        # the Bool pair and the eltype pairs reach the zero, one and general branches; the
        # comprehensive ones mix the two kinds, or repeat a value
        for (α, β) in ((true, false), (false, true), (zero(ElType), one(ElType)), (one(ElType), general), (general, zero(ElType)),
                (@static COMPREHENSIVE ? (zip(vs, vs)..., (true, general), (general, false), (false, zero(ElType)), (one(ElType), true)) : ())...)
            C .= rand.(ElType)
            expected′ = expected .* α .+ C .* β
            @test mul!(C, A, B, α, β) === C
            @test C ≈ expected′
        end
    end

    # Int and BigFloat add the generic non-BLAS path to what the BLAS eltypes cover
    for ElType in (@static COMPREHENSIVE ? (Int, Float64, ComplexF64, BigFloat) : (Float64, ComplexF64))
        SP = sprand(ElType, 10, 10, 0.3)
        D = rand(ElType, 10, 10)
        fs = (identity, adjoint, transpose)
        # every transform once on each side; adjoint and transpose differ only for a complex eltype
        for (f1, f2) in (TRANSFORM_PAIRS[1:(ElType === Float64 ? 1 : end)]..., ((@static COMPREHENSIVE && ElType in STD_ELTYPES) ?
                ((identity, adjoint), (adjoint, identity), (transpose, transpose), (adjoint, adjoint)) : ())...)
            test_mul(f1(SP), f2(D))
            test_mul(f1(D), f2(SP))
        end
        # Coefficients branch on the sparse transform and on plain/wrapped dense-left inputs.
        for f in (@static COMPREHENSIVE ? (ElType in STD_ELTYPES ? fs : (adjoint,)) : ElType <: Complex ? (identity, adjoint) : ())
            test_mul_coefficients(f(SP), D)
            test_mul_coefficients(D, f(SP))
        end
        for f in (@static COMPREHENSIVE ? (adjoint, transpose) : ElType <: Complex ? (transpose,) : ())
            test_mul_coefficients(f(D), SP)
        end
    end
end

@testset "BLAS Level-2" begin
    @testset "dense A * sparse x -> dense y" begin
        # standard: a plain matrix with either eltype, and the transposed and wrapped factors
        # with the complex one, for which they differ. Comprehensive: every pair of the
        # eltypes and of the seven kinds of factor.
        cases = @static COMPREHENSIVE ? pairwise((Float64, ComplexF64), (Float64, ComplexF64), 1:7) : ()
        for TA in (Float64, ComplexF64), Tx in (Float64, ComplexF64)
            T = Base.promote_op(LinearAlgebra.matprod, TA, Tx)
            sel(k) = (TA == Tx && (k == 1 || TA <: Complex && k in (2, 3, 7))) || (TA, Tx, k) in cases
            sel(1) && let A = randn(TA, 9, 16), x = sprand(Tx, 16, 0.7)
                xf = Array(x)
                for α in [0.0, 1.0, 2.0], β in [0.0, 0.5, 1.0]
                    y = rand(T, 9)
                    rr = α*A*xf + β*y
                    @test mul!(y, A, x, α, β) === y
                    @test y ≈ rr
                end
                y = A*x
                @test isa(y, Vector{T})
                @test A*x ≈ A*xf
            end

            sel(2) && let A = randn(TA, 16, 9), x = sprand(Tx, 16, 0.7)
                xf = Array(x)
                for α in [0.0, 1.0, 2.0], β in [0.0, 0.5, 1.0]
                    y = rand(T, 9)
                    rr = α*transpose(A)*xf + β*y
                    @test mul!(y, transpose(A), x, α, β) === y
                    @test y ≈ rr
                end
                y = *(transpose(A), x)
                @test isa(y, Vector{T})
                @test y ≈ *(transpose(A), xf)
            end

            sel(3) && let A = randn(TA, 16, 9), x = sprand(Tx, 16, 0.7)
                xf = Array(x)
                for α in [0.0, 1.0, 2.0], β in [0.0, 0.5, 1.0]
                    y = rand(T, 9)
                    rr = α*A'xf + β*y
                    @test mul!(y, adjoint(A), x, α, β) === y
                    @test y ≈ rr
                end
                y = *(adjoint(A), x)
                @test isa(y, Vector{T})
                @test y ≈ *(adjoint(A), xf)
            end

            let A = randn(TA, 16, 16), x = sprand(Tx, 16, 0.7)
                xf = Array(x)
                for (k, wrap) in enumerate((M -> Symmetric(M, :U), M -> Symmetric(M, :L),
                        M -> Hermitian(M, :U), M -> Hermitian(M, :L)))
                    sel(k + 3) || continue
                    for α in (0.0, 1.0, 2.0), β in (0.0, 0.5, 1.0)
                        y = rand(T, 16)
                        rr = α*wrap(A)*xf + β*y
                        @test mul!(y, wrap(A), x, α, β) === y
                        @test y ≈ rr
                    end
                    y = *(wrap(A), x)
                    @test isa(y, Vector{T})
                    @test y ≈ *(wrap(A), xf)
                end
            end
        end
    end
end

@testset "BLAS Level-2: sparse factors" begin
    @testset "sparse A * sparse x -> dense y" begin
        let A = sprandn(9, 16, 0.5), x = sprand(16, 0.7)
            Af = Array(A)
            xf = Array(x)
            for α in [0.0, 1.0, 2.0], β in [0.0, 0.5, 1.0]
                y = rand(9)
                rr = α*Af*xf + β*y
                @test mul!(y, A, x, α, β) === y
                @test y ≈ rr
            end
            y = SparseArrays.densemv(A, x)
            @test isa(y, Vector{Float64})
            @test y ≈ Af*xf
        end

        let A = sprandn(16, 9, 0.5), x = sprand(16, 0.7)
            Af = Array(A)
            xf = Array(x)
            for α in [0.0, 1.0, 2.0], β in [0.0, 0.5, 1.0]
                y = rand(9)
                rr = α*Af'xf + β*y
                @test mul!(y, transpose(A), x, α, β) === y
                @test y ≈ rr
            end
            y = SparseArrays.densemv(A, x; trans='T')
            @test isa(y, Vector{Float64})
            @test y ≈ *(transpose(Af), xf)

            @static if COMPREHENSIVE
            A32 = SparseMatrixCSC{Float64,Int32}(A)
            @test mul!(zeros(9), transpose(A32), x) ≈ transpose(Af) * xf
            end
        end

        let A = sprandn(16, 16, 0.5), x = sprand(16, 0.7)
            Af = Array(A)
            xf = Array(x)
            for wrap in (M -> Symmetric(M, :U), M -> Symmetric(M, :L),
                M -> Hermitian(M, :U), M -> Hermitian(M, :L),
                M -> UpperTriangular(M), M -> UnitUpperTriangular(M),
                M -> LowerTriangular(M), M -> UnitLowerTriangular(M),
                M -> UpperTriangular(transpose(M)), M -> UnitUpperTriangular(transpose(M)),
                M -> LowerTriangular(transpose(M)), M -> UnitLowerTriangular(transpose(M)),
                M -> UpperTriangular(adjoint(M)), M -> UnitUpperTriangular(adjoint(M)),
                M -> LowerTriangular(adjoint(M)), M -> UnitLowerTriangular(adjoint(M)),
                # standard: both triangles of a symmetric wrapper, one triangle and one triangle
                # of an adjoint parent; triangular.jl owns the triangular grid
                M -> UpperTriangular(Symmetric(M)))[(@static COMPREHENSIVE ? (1:17) : [1, 2, 5, 15])]
                for α in (0.0, 1.0, 2.0), β in (0.0, 0.5, 1.0)
                    y = rand(16)
                    rr = α*wrap(Af)*xf + β*y
                    @test mul!(y, wrap(A), x, α, β) === y
                    @test y ≈ rr
                end
                y = wrap(A) * x
                @test y ≈ *(wrap(Af), xf)
            end
        end

        let A = complex.(sprandn(7, 8, 0.5), sprandn(7, 8, 0.5)),
            x = complex.(sprandn(8, 0.6), sprandn(8, 0.6)),
            x2 = complex.(sprandn(7, 0.75), sprandn(7, 0.75))
            Af = Array(A)
            xf = Array(x)
            x2f = Array(x2)
            @test SparseArrays.densemv(A, x; trans='N') ≈ Af * xf
            @test SparseArrays.densemv(A, x2; trans='T') ≈ transpose(Af) * x2f
            @test SparseArrays.densemv(A, x2; trans='C') ≈ Af'x2f
            @test_throws ArgumentError SparseArrays.densemv(A, x; trans='D')
        end

        @static if COMPREHENSIVE
        let A = sparse(bitrand(9, 16)), x = sparse(bitrand(16))
            Af = Array(A)
            xf = Array(x)
            y = SparseArrays.densemv(A, x)
            @test isa(y, Vector{Int})
            @test y == Af*xf
        end
        end
    end
    @testset "sparse A * sparse x -> sparse y" begin
        let A = sprandn(9, 16, 0.5), x = sprand(16, 0.7), x2 = sprand(9, 0.7)
            Af = Array(A)
            xf = Array(x)
            x2f = Array(x2)

            y = A*x
            @test isa(y, SparseVector{Float64,Int})
            @test all(nonzeros(y) .!= 0.0)
            @test Array(y) ≈ Af * xf
            @test (A * view(x, :))::SparseVector{Float64,Int} ≈ Af * xf

            y = *(transpose(A), x2)
            @test isa(y, SparseVector{Float64,Int})
            @test all(nonzeros(y) .!= 0.0)
            @test Array(y) ≈ Af'x2f
        end

        let A = complex.(sprandn(7, 8, 0.5), sprandn(7, 8, 0.5)),
            x = complex.(sprandn(8, 0.6), sprandn(8, 0.6)),
            x2 = complex.(sprandn(7, 0.75), sprandn(7, 0.75))
            Af = Array(A)
            xf = Array(x)
            x2f = Array(x2)

            y = A*x
            @test isa(y, SparseVector{ComplexF64,Int})
            @test Array(y) ≈ Af * xf

            y = *(transpose(A), x2)
            @test isa(y, SparseVector{ComplexF64,Int})
            @test Array(y) ≈ transpose(Af) * x2f

            y = *(adjoint(A), x2)
            @test isa(y, SparseVector{ComplexF64,Int})
            @test Array(y) ≈ Af'x2f

            @static if COMPREHENSIVE
            A32 = SparseMatrixCSC{ComplexF64,Int32}(A)
            # an index type of the vector that promotes with the matrix's, and one that matches
            for (x32, op) in ((x2, transpose), (SparseVector{ComplexF64,Int32}(x2), adjoint))
                y = op(A32) * x32
                @test isa(y, SparseVector{ComplexF64,promote_type(Int32, eltype(nonzeroinds(x32)))})
                @test Array(y) ≈ op(Af) * x2f
            end
            end
        end

        let A = sparse(bitrand(9, 16)), x = sparse(bitrand(16)), x2 = sparse(bitrand(9))
            Af = Array(A)
            xf = Array(x)
            x2f = Array(x2)

            y = A*x
            @test isa(y, SparseVector{Int, Int})
            @test Array(y) == Af*xf

            @static if COMPREHENSIVE
            y = A'*x2
            @test isa(y, SparseVector{Int, Int})
            @test Array(y) == Af'x2f
            end
        end
    end
    @static if COMPREHENSIVE
    @testset "sparse A * dense x -> dense y" begin
        let A = sparse(bitrand(9, 16)), x = Vector(bitrand(16)), x2 = Vector(bitrand(9))
            Af = Array(A)
            xf = Array(x)
            x2f = Array(x2)

            y = A*x
            @test isa(y, Vector{Int})
            @test y == Af*xf

            y = A'*x2
            @test isa(y, Vector{Int})
            @test y == Af'x2f
        end
    end
    end
end

@testset "products of LinearAlgebra's Q types with sparse operands" begin
    D = randn(7, 7)
    m = size(D, 1)
    # one operand of each kind gives the same dense result as its dense copy
    B, C, b = sprandn(m, 3, 0.5), sprandn(3, m, 0.5), sprandn(m, 0.5)
    # comprehensive: the other Q types, and every kind of sparse operand once, the Q types cycling
    @testset "$name" for (k, name, Q) in ((1, "qr", qr(D).Q), (@static COMPREHENSIVE ? ((2, "pivoted qr", qr(D, ColumnNorm()).Q),
                                       (3, "hessenberg", hessenberg(D).Q), (4, "lq", lq(D).Q)) : ())...)
        for X in (B, (@static COMPREHENSIVE ? ((sparse(B')', view(B, :, 1:2))[mod1(k, 2)],) : ())...)
            @test (Q * X)::Matrix ≈ Q * Matrix(X)
        end
        for X in (C, (@static COMPREHENSIVE ? ((transpose(sparse(transpose(C))), view(C, :, 1:m), view(B, :, 1:2)', transpose(b))[k],) : ())...)
            @test (X * Q')::Matrix ≈ Matrix(X) * Q'
        end
        @test (Q' * B)::Matrix ≈ Q' * Matrix(B)
        @test (C * Q)::Matrix ≈ Matrix(C) * Q
        for x in (b, (@static COMPREHENSIVE ? ((view(B, :, 1), view(b, 1:m))[mod1(k, 2)],) : ())...)
            @test (Q * x)::Vector ≈ Q * Vector(x)
        end
        @test (Q' * b)::Vector ≈ Q' * Vector(b)
        @test (b' * Q)::Adjoint ≈ Vector(b)' * Q
        @test_throws DimensionMismatch Q * sprandn(m + 1, 2, 0.5)
    end
    # one method serves the left Q types; the lq Q has its own
    let Q = lq(D).Q
        @test (Q * B)::Matrix ≈ Q * Matrix(B)
        @test (C * Q)::Matrix ≈ Matrix(C) * Q
        @test (Q' * b)::Vector ≈ Q' * Vector(b)
    end
end

@testset "product kernels touch stored entries only" begin
    n = 8
    # adjoint dense times adjoint sparse reads each entry of the dense factor at most once
    A = sprandn(ComplexF64, 6, n, 0.5); X = randn(ComplexF64, n, 5); C0 = randn(ComplexF64, 5, 6)
    Xc = CountedReads(X)
    @test mul!(copy(C0), Xc', A', 2, 3) ≈ 2 * X' * Matrix(A)' + 3 * C0
    @test Xc.reads[] <= length(X)
    @test mul!(copy(C0), transpose(X), transpose(A), 2, 3) ≈ 2 * transpose(X) * transpose(Matrix(A)) + 3 * C0
    P = opcount_sparse(sparse(1.0I, n, n))
    one_, two = OpCount(1.0), OpCount(2.0)
    # sparse times sparse into a dense destination: one multiplication per pair of stored entries
    @test mulcount(() -> mul!(fill(one_, n, n), P, P, true, false)) == n
    @test mulcount(() -> mul!(fill(one_, n, n), Symmetric(P), P', true, false)) == n
    # a symmetric sparse matrix times a sparse vector costs what the plain product does
    x = sparsevec(fill(one_, n)); y = fill(one_, n)
    @test mulcount(() -> mul!(copy(y), Symmetric(P), x, true, false)) == mulcount(() -> mul!(copy(y), P, x, true, false))
    # scaling a column-view destination by `β` stays sparse
    Q = opcount_sparse(sparse(1.0I, n, n + 1))
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
    # each dense factor with a sparse one on either side, and two sparse factors; comprehensive
    # adds a plain sparse factor times a dense one, and a plain dense one times a transformed sparse one
    for (i, j) in ((1, 4), (3, 6), (5, 2), (6, 3), (4, 4), (@static COMPREHENSIVE ? ((4, 1), (1, 5)) : ())...)
        fA, fB = fs[i], fs[j]
        mul!(zeros(6, 10), fA(6, 8), fB(8, 10))
        @test_throws DimensionMismatch mul!(zeros(7, 10), fA(6, 8), fB(8, 10))
        @test_throws DimensionMismatch mul!(zeros(6, 11), fA(6, 8), fB(8, 10))
        @test_throws DimensionMismatch mul!(zeros(6, 10), fA(5, 8), fB(8, 10))
        @test_throws DimensionMismatch mul!(zeros(6, 10), fA(6, 9), fB(8, 10))
        @test_throws DimensionMismatch mul!(zeros(6, 10), fA(6, 8), fB(7, 10))
        @test_throws DimensionMismatch mul!(zeros(6, 10), fA(6, 8), fB(8, 9))
    end
end

@static if COMPREHENSIVE
@testset "sparse right multiplication of Symmetric and Hermitian matrices #21431" begin
    S = sparse(1.0I, 2, 2)
    @test issparse(S*S*S)
    for T in (Symmetric, Hermitian)
        @test issparse(S*T(S)*S)
        @test issparse(S*(T(S)*S))
        @test issparse((S*T(S))*S)
    end
end
end

end # module
