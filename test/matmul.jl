# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseMatmulTests
# `*`, `mul!`, `lmul!` and `rmul!`. Kept apart from linalg.jl so the two run on separate
# test workers.

using Test
using SparseArrays
using SparseArrays: AbstractSparseMatrixCSC, nonzeroinds, getcolptr, rowvals, nonzeros, fixed, _is_fixed
using LinearAlgebra
include("testhelpers.jl")

# every transform appears once on each side
const TRANSFORM_PAIRS = ((identity, identity), (adjoint, transpose), (transpose, adjoint))

@static if COMPREHENSIVE
@testset "matrix-vector multiplication (non-square)" begin
    for (m, n) in ((10, 5), (5, 10))
        a = fixture(Float64, m, n)
        # integers, so that both products are exact
        b = Float64.(1:n)
        @test maximum(abs.(a*b - Array(a)*b)) < 100*eps()
    end
end
end

@testset "diagonal - sparse vector multiplication" begin
    for n in (10, (@static COMPREHENSIVE ? (7,) : ())...)
        b = fixturevec(Float64, n)
        A = Diagonal(fixturedense(Float64, n))
        @test mismatch(A * b, A * Vector(b)) === nothing
        # the eltype of the result is that of the product, not of the vector
        Ac = Diagonal(fixturedense(ComplexF64, n))
        @test mismatch(Ac * b, Ac * Vector(b)) === nothing
        @static if COMPREHENSIVE
        @test norm(A * b - Array(A) * b) <= 10eps()
        @test norm(Ac * b - Array(Ac) * b) <= 10eps()
        @test_throws DimensionMismatch A * [b; 1]
        end
        @test_throws DimensionMismatch A * b[1:end-1]
    end
end

@static if COMPREHENSIVE
@testset "sparse matrix * BitArray" begin
    A = fixture(Float64, 5, 5)
    MA = Array(A)
    B = trues(5)
    @test A*B ≈ MA*B
    B = trues(5,5)
    for (trA, trB) in TRANSFORM_PAIRS[1:2]
        @test trA(A) * trB(B) ≈ trA(MA) * trB(B)
        @static if COMPREHENSIVE
        @test trB(B) * trA(A) ≈ trB(B) * trA(MA)
        end
    end
end
end


@testset "matrix multiplication" begin
    for (a, b) in (
                            # standard: one product whose columns are gathered by a scan, one
                            # sparse enough that they are sorted, and a full column times a full
                            # row, whose product outgrows the size estimated for it
                            (fixture(Float64, 10, 5), fixture(Float64, 5, 15)),
                            (fixturestrided(Float64, 100, 100, 33), fixturestrided(Float64, 100, 60, 49)),
                            (sparse(1:10, fill(1, 10), 1.0:10.0, 10, 10), sparse(fill(1, 10), 1:10, 1.0:10.0, 10, 10)),
                            (@static COMPREHENSIVE ? ((fixturestrided(Float64, 100, 100, 9), fixturestrided(Float64, 100, 100, 7)),) : ())...,
                           )
        n = size(a, 2)
        ad = Array(a)
        bd = Array(b)
        as = sparse(a')
        bs = sparse(b')
        ab = a * b
        aab = ad * bd
        @test maximum(abs.(ab - aab)) < 100*eps()
        @test mismatch(ab, aab; approx=true) === nothing
        @test a*bs' == ab
        @test as'*b == ab
        @test as'*bs' == ab
        f = Diagonal(fixturedense(Float64, n))
        @test mismatch(a*f, ad*f) === nothing
        @test mismatch(f*b, f*bd) === nothing
        A = fixturedense(Float64, 2n, 2n)
        sA = view(A, 1:2:2n, 1:2:2n); dA = Array(sA)
        @test (sA*b)::Matrix ≈ dA*bd
        @test (a*sA)::Matrix ≈ ad*dA
        @test (sA'b)::Matrix ≈ dA'*bd
        @static if COMPREHENSIVE
        c = fixture(ComplexF32, n, n); cd = Array(c)
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
        A = fixture(T, n, n); B = permutedims(fixture(T, n, n))
        # standard: one pair for each eltype, as the two wrappers differ only for a complex one.
        # Comprehensive: every second factor once as well, shared out among the wrappers, for
        # the complex eltype; a wrapper is made sparse before the product, so the pairs add nothing.
        std = T <: Complex ? (Hermitian(A, :L), B') : (Symmetric(A), B)
        for S in (@static COMPREHENSIVE ? (Symmetric(A), Hermitian(A, :L), Symmetric(view(A, :, 1:n))) : std[1:1])
            alt = xor(S isa Hermitian, T <: Complex)
            for X in (@static COMPREHENSIVE ? ((S === std[1] ? std[2:2] : ())..., (T <: Complex ? (B, B', transpose(B), UpperTriangular(B),
                    view(B, :, 1:n), Hermitian(B), Symmetric(B, :L))[(S isa Hermitian ? 2 : parent(S) isa SubArray ? 3 : 1):3:end] : ())...) : std[2:2])
                @test mismatch((S * X)::SparseMatrixCSC, Matrix(S) * Matrix(X); approx=true) === nothing
                @test mismatch((X * S)::SparseMatrixCSC, Matrix(X) * Matrix(S); approx=true) === nothing
            end
            for x in (@static COMPREHENSIVE ? (fixturevec(T, n), view(B, :, 2))[1:(alt ? 0 : 2)] : T <: Complex ? (fixturevec(T, n),) : ())
                @test mismatch((S * x)::SparseVector, Matrix(S) * Vector(x); approx=true) === nothing
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

@static if COMPREHENSIVE
@testset "Sparse promotion in sparse matmul" begin
    # the index and element types are the point: both promote
    A = @static COMPREHENSIVE ? SparseMatrixCSC{Float32, Int8}(2, 2, Int8[1, 2, 3], Int8[1, 2], Float32[1., 2.]) :
        SparseMatrixCSC{Float64, Int16}(2, 2, Int16[1, 2, 3], Int16[1, 2], [1., 2.])
    MA = Array(A)
    B = @static COMPREHENSIVE ? SparseMatrixCSC{ComplexF32, Int32}(2, 2, Int32[1, 2, 3], Int32[1, 2], ComplexF32[1. + im, 2. - im]) :
        SparseMatrixCSC{ComplexF64, Int}(2, 2, [1, 2, 3], [1, 2], [1. + im, 2. - im])
    MB = Array(B)
    @test mismatch(A*adjoint(B), MA * adjoint(MB); Ti=Int32, approx=true) === nothing
    @static if COMPREHENSIVE
    @test mismatch(adjoint(B)*A, adjoint(MB) * MA; Ti=Int32, approx=true) === nothing
    end
end
end

@testset "destination array density in multiplication" begin
    # `transpose` and `Hermitian` would take the methods of `adjoint` and `Symmetric`, as the eltype is real
    wrappers = (adjoint, Symmetric, UpperTriangular, UnitUpperTriangular, UnitLowerTriangular, UpperHessenberg)
    # standard: a transform, a symmetric and a triangular wrapper, and `Diagonal` and one of the
    # banded types, which share their methods; triangular.jl owns the triangular grid.
    # Comprehensive: a triangular wrapper also meets itself, as the product keeps its kind,
    # another wrapper one other wrapper, and each one banded type.
    for tA in (@static COMPREHENSIVE ? wrappers : (adjoint, UpperTriangular))
        i = @static COMPREHENSIVE ? findfirst(==(tA), wrappers) : 1
        A = fixturedense(Float64, 5, 5)
        At = tA(A)
        S = fixture(Float64, 5, 5)
        St = tA(S)
        for tB in ((tA === adjoint ? (Symmetric,) : tA === UpperTriangular ? (transpose,) : ())...,
                   (@static COMPREHENSIVE ? (St isa LinearAlgebra.AbstractTriangular ? tA : wrappers[mod1(i + 2, end)],) : ())...)
            B = permutedims(fixture(Float64, 5, 5))
            Bt = tB(B)
            C = At*Bt
            @test C ≈ Matrix(At) * Matrix(Bt)
            @test !issparse(C)
            D = St*Bt
            @test D ≈ Matrix(St) * Matrix(Bt)
            @test issparse(D)
            @static COMPREHENSIVE && tA === tB && St isa LinearAlgebra.AbstractTriangular && @test D isa tA
        end
        b = fixturevec(Float64, 5)
        c = At * b
        @test c ≈ Matrix(At) * Vector(b)
        @test c isa DenseVector
        d = St*b
        @test mismatch(d, Matrix(St) * Vector(b); approx=true) === nothing
        @test d isa SparseVector
        for T in (Diagonal(fixturedense(Float64, 5)),
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
    @static if !COMPREHENSIVE
    S = fixture(Float64, 5, 5); B = permutedims(fixture(Float64, 5, 5))
    for tA in (UnitUpperTriangular, LowerTriangular)
        D = tA(S) * tA(B)
        @test D isa tA && issparse(D)
        @test D ≈ Matrix(tA(S)) * Matrix(tA(B))
    end
    end
end

@testset "multiplication of special sparse with dense matrix" begin
    # this results in a call of the most generic multiplication code in LinearAlgebra.jl
    A = [1.0 -2.0; 3.0 0.5]
    S = sparse(A)
    B = [0.25 -1.5]'
    @test Symmetric(S) * B ≈ Symmetric(A) * B
end

@static if COMPREHENSIVE
@testset "Symmetric of sparse matrix mul! dense vector" begin
    n = 1000
    p = 0.02
    # the strides give the density `p`; the two parts of the complex matrix share part of their pattern
    Areal = fixturestrided(Float64, n, n, 49)
    Breal = fixturedense(Float64, n)
    Acomplex = fixturestrided(Float64, n, n, 99) + fixturestrided(Float64, n, n, 101) * im
    Bcomplex = fixturedense(ComplexF64, n)
    # both triangles; the two wrappers differ only for a complex eltype
    @testset "symmetric/Hermitian sparse multiply with $S($U)" for (S, U, A, B) in
            ((Symmetric, :U, Areal, Breal), (Symmetric, :L, Areal, Breal),
             (Hermitian, :U, Acomplex, Bcomplex), (Hermitian, :L, Acomplex, Bcomplex),
             (@static COMPREHENSIVE ? ((Symmetric, :U, Acomplex, Bcomplex),) : ())...)
        Asym = S(A, U)
        As = sparse(Asym) # takes most time
        # @test which(mul!, (typeof(B), typeof(Asym), typeof(B))).module == SparseArrays
        @test norm(Asym * B - As * B, Inf) <= eps() * n * p * 10
    end
end
end

@static if COMPREHENSIVE
@testset "Symmetric of view of sparse matrix mul! dense vector" begin
    n = 1000
    p = 0.02
    # the strides give the density `p`; the two parts of the complex matrix share part of their pattern
    Areal = view(fixturestrided(Float64, n, n+10, 49), :, 6:n+5)
    Breal = fixturedense(Float64, n)
    Acomplex = view(fixturestrided(Float64, n, n+10, 99) + fixturestrided(Float64, n, n+10, 101) * im, :, 6:n+5)
    Bcomplex = fixturedense(ComplexF64, n)
    std = (Symmetric, :L, (Acomplex, Bcomplex))
    @testset "symmetric/Hermitian sparseview multiply with $S($U)" for (S, U, (A, B)) in (std, (@static COMPREHENSIVE ?
            eachvalue((Symmetric, Hermitian), (:U, :L), ((Areal, Breal), (Acomplex, Bcomplex))) : ())...)
        Asym = S(A, U)
        As = sparse(Asym) # takes most time
        # @test which(mul!, (typeof(B), typeof(Asym), typeof(B))).module == SparseArrays
        @test norm(Asym * B - As * B, Inf) <= eps() * n * p * 10
    end
end
end

@static if COMPREHENSIVE
@testset "Dense times symmetric/Hermitian sparse matrix multiplication" begin
    @static if COMPREHENSIVE
    A = [1 3; 2 4]
    As = sparse(A)
    B = [1 1; 1 1]
    @test mul!(copy(B), B, Hermitian(A), true, true) == mul!(copy(B), B, Hermitian(As), true, true)
    end

    n = 20
    # the two wrappers differ only for a complex eltype, which takes both, and both triangles
    std = ((Float64, Symmetric, :U), (ComplexF64, Hermitian, :L))
    @testset "$T, $S($U)" for (T, S, U) in (std..., (@static COMPREHENSIVE ?
            ((ComplexF64, Symmetric, :L), (ComplexF64, Hermitian, :U)) : ())...)
        P = fixture(T, n, n + 2)
        nonzeros(P)[1] = 0
        C = fixturedense(T, 3, n)
        for (A, X) in ((S(P[:, 1:n], U), permutedims(fixturedense(T, n, 3))),
                ((@static COMPREHENSIVE || S === Hermitian) ? ((S(view(P, :, 2:n+1), U), fixturedense(T, n, 3)'),) : ())...,
                ((@static COMPREHENSIVE && T <: Complex) ? ((S(P[:, 1:n], U), transpose(fixturedense(T, n, 3))),) : ())...)
            @test X * A ≈ X * Matrix(A)
            @test mul!(copy(C), X, A, 2, 3) ≈ mul!(copy(C), X, Matrix(A), 2, 3)
        end
        X = S(fixturedense(T, n, n), U)
        C = permutedims(fixturedense(T, n, n))
        Q = P[:, 1:n]
        # the transformed sparse factors meet the Hermitian dense one only
        for B in (@static COMPREHENSIVE ? (Q, view(P, :, 2:n+1), Q', transpose(Q))[1:(S === Hermitian ? 4 : 2)] : S === Hermitian ? (Q',) : (Q,))
            @test X * B ≈ Matrix(X) * Matrix(B)
            @test mul!(copy(C), X, B, 2, 3) ≈ mul!(copy(C), Matrix(X), Matrix(B), 2, 3)
        end
        @test_throws DimensionMismatch mul!(zeros(T, 3, n + 1), C, S(P[:, 1:n], U))
    end
    # the sparse kernel multiplies by stored zeros, the generic fallback skips them
    @test isequal([Inf 1.0] * Symmetric(sparse([1, 2], [1, 2], [0.0, 1.0])), [NaN 1.0])
    @test isequal(Symmetric([Inf 1.0; 1.0 1.0]) * sparse([1, 2], [1, 2], [0.0, 1.0]), [NaN 1.0; 0.0 1.0])
end
end

@testset "sparse-dense products take the same dense factors on either side" begin
    n = 12
    @testset "$T" for T in (Float64, ComplexF64)
        S = fixture(T, n, n)
        D = fixturedense(T, n, n)
        C = permutedims(D)
        c = D[:, 2]
        # one dense factor per sparse kernel rather than the full grid
        for (A, X, x) in ((S, view(D, [1:n;], :), 1.0:n),
                          (S', view(D, :, [1:n;])', view(D, [1:n;], 1)),
                          (view(S, :, [1:n;]), reshape(1.0:n^2, n, n), 1.0:n),
                          (view(S, [n:-1:1;], 1:n), D, c),   # no compressed storage of its own (#56)
                          (Symmetric(S), UpperHessenberg(D), view(D, [1:n;], 1)),
                          (Hermitian(S, :L), Hermitian(D, :L), 1.0:n))[
                # the kernels conjugate only a complex eltype; a real one takes the plain factor
                (T <: Complex ? (@static COMPREHENSIVE ? (2:6) : [2, 4, 6]) : [1])]
            @test X * A ≈ Matrix(X) * Matrix(A)
            (@static COMPREHENSIVE || A isa Hermitian) && @test A * X ≈ Matrix(A) * Matrix(X)
            # with general coefficients the Hermitian kernel conjugates the mirrored triangle itself
            (@static COMPREHENSIVE || A isa Hermitian) && @test mul!(copy(C), X, A, 2, 3) ≈ mul!(copy(C), Matrix(X), Matrix(A), 2, 3)
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
    S = fixture(Float64, 10, 12); G = view(S, [4, 1, 1, 9, 7], 2:11); M = Matrix(G)
    X, Y, x, y = fixturedense(Float64, 10, 3), fixturedense(Float64, 3, 5), fixturedense(Float64, 10), fixturedense(Float64, 5)
    @test which(mul!, Base.typesof(zeros(5, 3), 'N', 'N', G, X, true, false)).module == SparseArrays
    @test which(mul!, Base.typesof(zeros(3, 10), 'N', 'N', Y, G, true, false)).module == SparseArrays
    @test which(mul!, Base.typesof(zeros(5), 'N', G, x, true, false)).module == SparseArrays
    @test G * X ≈ M * X && (@static COMPREHENSIVE ? G * x ≈ M * x && G' * y ≈ M' * y && y' * G ≈ y' * M : true)
    @test Y * G isa Matrix && Y * G ≈ Y * M
    P = fixture(Float64, 10, 6); Q = fixture(Float64, 8, 5)
    @test G * P isa SparseMatrixCSC && mismatch(G * P, M * Matrix(P); approx=true) === nothing
    @static if COMPREHENSIVE
    @test Q * G isa SparseMatrixCSC && mismatch(Q * G, Matrix(Q) * M; approx=true) === nothing
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

@static if COMPREHENSIVE
@testset "symmetric/Hermitian sparse mul! into a dense destination" begin
    Quaternion = quaternion_type()
    q(i) = Quaternion.(eachcol(fixturedense(Float64, i, 4))...)
    # `alpha` multiplies from the right, as in `A*B*alpha`
    H = sparse(reshape(q(9), 3, 3))
    B = reshape(q(6), 3, 2)
    α = q(1)[1]
    for W in (Hermitian(H, :U), Symmetric(H, :L))
        @test mul!(zeros(eltype(B), 3, 2), W, B, α, false) ≈ Matrix(W) * B * α
        @test mul!(zeros(eltype(B), 3), W, B[:, 1], α, false) ≈ Matrix(W) * B[:, 1] * α
    end
    # `alpha === false` is a strong zero, so a NaN in `B` does not reach `C`
    S = Symmetric(sparse([1.0 2; 3 4]))
    @test mul!(ones(2, 2), S, [NaN 1; 1 1], false, 1.0) == ones(2, 2)
    @test mul!(ones(2), S, [NaN, 1], false, 2.0) == [2.0, 2.0]
    # the scalars keep their own type, so an integer destination takes any result it can hold
    Si = Symmetric(sparse([1 2; 3 4]))
    @test mul!(zeros(Int, 2, 2), Si, [2 2; 4 4], 0.5, 1) == [5 5; 10 10]
    @test mul!(ones(Int, 2, 2), Si, [2 2; 4 4], 0.5, 1) == mul!(ones(Int, 2, 2), Symmetric([1 2; 3 4]), [2 2; 4 4], 0.5, 1)
end

@testset "mul! into a dense destination that is an operand throws" begin
    S = sparse([1.0 2 0; 0 3 4; 5 0 6])
    D = [1.0 2 3; 4 5 6; 7 8 10]
    for f in (C -> mul!(C, S, C), C -> mul!(C, S', C), C -> mul!(C, transpose(S), C, 2.0, 3.0),
              C -> mul!(C, Symmetric(S), C), C -> mul!(C, S, C'), C -> mul!(C, C, S),
              C -> mul!(C, C, S'), C -> mul!(C, C', S), C -> mul!(C, transpose(C), transpose(S)),
              C -> mul!(C, C, Symmetric(S)), C -> mul!(C, C, view(S, :, [2, 1, 3])))
        C = copy(D)
        @test_throws ArgumentError f(C)
        @test C == D
    end
    x = [1.0, 2, 3]
    @test_throws ArgumentError mul!(x, S, x)
    @test_throws ArgumentError mul!(x, S', x)
    @test_throws ArgumentError mul!(x, Hermitian(S), x)
    @test x == [1.0, 2, 3]
    # a shape error is reported first, as for a dense product
    C = zeros(2, 3)
    @test_throws DimensionMismatch mul!(C, S, C)
end

@testset "Diagonal mul! into a sparse destination converts before it writes" begin
    S = sparse([1.0 2 0; 0 3 4; 5 0 6])
    C0 = sparse([1 0 0; 0 1 1; 1 0 0])
    for (D, α, β) in ((Diagonal([0.5, 1, 1]), true, false), (Diagonal([2, 1, 1]), 0.5, 1),
                      (Diagonal([2, 1, 1]), 2, 0.5), (Diagonal([2, 1, 1]), 0, 0.5)),
        f in ((C, A) -> mul!(C, A, D, α, β), (C, A) -> mul!(C, D, A, α, β),
              (C, A) -> mul!(C, A', D, α, β), (C, A) -> mul!(C, D, transpose(A), α, β))
        C = copy(C0)
        @test_throws InexactError f(C, S)
        @test same_pattern(C, C0) && nonzeros(C) == nonzeros(C0)
    end
    # a result the destination can hold is stored, whatever the eltypes of the operands
    for (D, α, β) in ((Diagonal([2.0, 1, 1]), true, false), (Diagonal([2, 4, 6]), 0.5, 1), (Diagonal([2, 1, 1]), 2.0, 3.0)),
        f in ((C, A) -> mul!(C, A, D, α, β), (C, A) -> mul!(C, D, A, α, β),
              (C, A) -> mul!(C, A', D, α, β), (C, A) -> mul!(C, D, transpose(A), α, β))
        @test mismatch(f(copy(C0), S), f(Matrix(C0), Matrix(S))) === nothing
    end
    Cf = fixed(copy(C0))
    @test_throws InexactError mul!(Cf, S, Diagonal([0.5, 1, 1]))
    @test Cf == C0
end

@testset "products with a wrapped operand follow the dense or banded factor" begin
    S = sparse(ComplexF64[1 2+3im 1im; 0 5 2-im; 3+im 0 7])
    D = Matrix(S) .+ 1
    H = Hermitian(S + S')
    x = sparsevec([1.0 + im, 0, 2])
    banded = (Bidiagonal([1.0, 2, 3], [4.0, 5], :U), Tridiagonal(real(D)), SymTridiagonal([1.0, 2, 3], [4.0, 5]))
    # sparse times banded stays sparse on either side
    for A in (UpperTriangular(S)', transpose(UnitLowerTriangular(S)), transpose(H), Symmetric(S)',
              view(S, [2, 1, 3], :)), B in banded
        @test issparse(A * B) && Matrix(A * B) ≈ Matrix(A) * B
        @test issparse(B * A) && Matrix(B * A) ≈ B * Matrix(A)
    end
    # sparse times dense is dense on either side
    for A in (transpose(H), Symmetric(S)'), B in (D, Symmetric(D), Hermitian(D + D'))
        @test (A * B)::Matrix{ComplexF64} ≈ Matrix(A) * B
        @test (B * A)::Matrix{ComplexF64} ≈ B * Matrix(A)
    end
    for A in (UpperTriangular(D)', transpose(UnitLowerTriangular(D)), Symmetric(D)', transpose(Hermitian(D + D')))
        @test (A * x)::Vector{ComplexF64} ≈ Matrix(A) * Vector(x)
    end
end
end

@testset "in-place sparse-sparse mul!" begin
    for n in (20, (@static COMPREHENSIVE ? (30,) : ())...)
        sA = fixture(ComplexF64, n, n); A = Array(sA)
        sB = permutedims(sA); B = Array(sB)
        sC = sA[n:-1:1, :]; C = Array(sC)
        a = 0.7 - 1.3im; b = -0.4 + 0.9im
        vA = view(sA, :, 1:1:n)
        # the plain product, each transform once on each side with general coefficients, and
        # a view; the other size takes every factor, transform and coefficient once more.
        # Vectors, so that destructuring the cases is compiled once.
        cases = n == 20 ? (Any[sA, identity, identity, true, false], Any[sA, identity, identity, a, b], Any[sA, adjoint, transpose, a, b],
                           Any[sA, transpose, adjoint, a, b], Any[vA, identity, adjoint, a, b]) :
            eachvalue((sA, vA), (identity, adjoint, transpose), (identity, adjoint, transpose), (true, false, a), (true, false, b))
        for (sA, trA, trB, α, β) in cases
            # the three-argument form is the five-argument one with `true, false`
            α === true && @test mismatch(mul!(copy(sC), trA(sA), trB(sB)), trA(A) * trB(B); approx=true) === nothing
            @test mismatch(mul!(copy(sC), trA(sA), trB(sB), α, β), C*β + trA(A) * trB(B) * α; approx=true) === nothing
        end
    end
    A = fixture(ComplexF64, 8, 8); B = permutedims(A); C = A[8:-1:1, :]
    for W in ((@static COMPREHENSIVE ? (Symmetric,) : ())..., Hermitian)
        @static if COMPREHENSIVE
        @test mismatch(mul!(copy(C), W(A), B, 2, 3), 2 * W(Matrix(A)) * Matrix(B) + 3 * Matrix(C); approx=true) === nothing
        end
        @test mismatch(mul!(copy(C), A', W(B, :L), 2, 3), 2 * Matrix(A)' * W(Matrix(B), :L) + 3 * Matrix(C); approx=true) === nothing
    end
    # a column-view destination is assigned through its parent
    P = fixture(ComplexF64, 8, 10); P0 = copy(P)
    @test mul!(view(P, :, 2:9), A, B', 2, 3) ≈ 2 * Matrix(A) * Matrix(B)' + 3 * Matrix(P0)[:, 2:9]
    @test P[:, [1, 10]] == P0[:, [1, 10]]
    @static if COMPREHENSIVE
    W = view(P, :, [5, 3, 9]); W0 = Matrix(W)
    @test mul!(W, A, B[:, 1:3], true, true) ≈ Matrix(A) * Matrix(B)[:, 1:3] + W0
    @test mul!(view(sparse(ones(2, 2)), :, 1:2), sparse([1.0 0; 0 0]), sparse([1.0 0; 0 0])) == [1 0; 0 0]
    end
    # the destination takes the pattern of the sparse product; the elementwise fallback keeps its own
    @test nnz(mul!(sparse(ones(2, 2)), sparse([1.0 0; 0 0]), sparse([1.0 0; 0 0]))) == 1
    @test mismatch(mul!(sparse(fill(complex(NaN), 8, 8)), A, B, true, false), Matrix(A) * Matrix(B); approx=true) === nothing
    X = copy(A); @test mismatch(mul!(X, X, B), Matrix(A) * Matrix(B); approx=true) === nothing
    X = copy(B); @test mismatch(mul!(X, A, X, 2, 3), 2 * Matrix(A) * Matrix(B) + 3 * Matrix(B); approx=true) === nothing
    @test_throws DimensionMismatch mul!(spzeros(8, 7), A, B)
    # nothing is written when the destination cannot hold the result
    P = sparse([1.0 2; 0 3]); Q = sparse([0.5 0; 1 1])
    Ci = sparse([1 1; 1 1]); @test_throws InexactError mul!(Ci, P, Q); @test Ci == [1 1; 1 1]
    F = SparseArrays.fixed(sparse([1.0 0; 0 1])); @test_throws ArgumentError mul!(F, P, Q); @test F == [1 0; 0 1]
    G = SparseArrays.fixed(sparse(ones(2, 2)))
    @test mismatch(mul!(G, P, Q, 2, 1), 2 * Matrix(P) * Matrix(Q) + ones(2, 2)) === nothing
end

@testset "scaling with * and mul!, rmul!, and lmul!" begin
    sA, dA = fixturepair(Float64, 3, 7)
    sC = similar(sA)
    b = fixturedense(Float64, 7)
    @test mismatch(sA * Diagonal(b), dA * Diagonal(b)) === nothing
    @test mismatch(mul!(sC, sA, Diagonal(b)), dA * Diagonal(b)) === nothing
    @test mismatch(rmul!(copy(sA), Diagonal(b)), dA * Diagonal(b)) === nothing
    b = fixturedense(Float64, 3)
    @test mismatch(Diagonal(b) * sA, Diagonal(b) * dA) === nothing
    @test mismatch(mul!(sC, Diagonal(b), sA), Diagonal(b) * dA) === nothing
    @test mismatch(lmul!(Diagonal(b), copy(sA)), Diagonal(b) * dA) === nothing

    # adjoint/transpose of a sparse matrix with a Diagonal (issue #619)
    for (T, W) in ((ComplexF64, adjoint), (ComplexF64, transpose))
        S = fixture(T, 7, 3); M = Matrix(S)
        Dl = Diagonal(fixturedense(T, 3)); Dr = Diagonal(fixturedense(T, 7))
        @test W(S) * Dr isa SparseMatrixCSC
        @test Dl * W(S) isa SparseMatrixCSC
        @test mismatch(W(S) * Dr, W(M) * Dr; approx=true) === nothing
        @test mismatch(Dl * W(S), Dl * W(M); approx=true) === nothing
        # the transpose shares the kernels, without the conjugation
        W === adjoint || continue
        @test mismatch(Dl * W(S) * Dr, Dl * W(M) * Dr; approx=true) === nothing
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
        @test mismatch(C, W(M) * Dr; approx=true) === nothing
        @test mul!(C, Dl, W(S)) === C
        @test mismatch(C, Dl * W(M); approx=true) === nothing
        C0 = fixture(T, 3, 7)
        @test mismatch(mul!(copy(C0), W(S), Dr, 2, 3), 2 * W(M) * Dr + 3 * Matrix(C0); approx=true) === nothing
        @test mismatch(mul!(copy(C0), Dl, W(S), 2, 3), 2 * Dl * W(M) + 3 * Matrix(C0); approx=true) === nothing
        @test mismatch(mul!(copy(C0), W(S), Dr, 2, 0), 2 * W(M) * Dr; approx=true) === nothing
        @test_throws DimensionMismatch mul!(C, W(S), Dl)
        @test_throws DimensionMismatch mul!(similar(S), Dl, W(S))
        @static if COMPREHENSIVE
        # a destination with another index type goes through a materialized copy
        C32 = SparseMatrixCSC{T,Int32}(spzeros(3, 7))
        @test mismatch(mul!(C32, Dl, W(S)), Dl * W(M); Ti=Int32, approx=true) === nothing
        end
        # so does a destination aliasing the parent
        Q = fixture(T, 5, 5); MQ = Matrix(Q); Dq = Diagonal(fixturedense(T, 5))
        @test mismatch(mul!(Q, W(Q), Dq), W(MQ) * Dq; approx=true) === nothing
        @static if COMPREHENSIVE
        Q = fixture(T, 5, 5); MQ = Matrix(Q)
        @test mismatch(mul!(Q, Dq, W(Q)), Dq * W(MQ); approx=true) === nothing
        # or sharing its storage
        Q = fixture(T, 5, 5); MQ = Matrix(Q)
        Cs = SparseMatrixCSC(5, 5, copy(getcolptr(Q)), copy(rowvals(Q)), nonzeros(Q))
        @test mismatch(mul!(Cs, W(Q), Dq), W(MQ) * Dq; approx=true) === nothing
        end
        # fixed operands are read, never written
        F = fixed(S)
        @test W(F) * Dr isa AbstractSparseMatrixCSC
        @test mismatch(W(F) * Dr, W(M) * Dr; approx=true) === nothing
        @static if COMPREHENSIVE
        @test Dl * W(F) isa AbstractSparseMatrixCSC
        @test mismatch(Dl * W(F), Dl * W(M); approx=true) === nothing
        end
        @test F == S
    end
    # a Diagonal times a fixed matrix keeps the structure, and the fixedness, of the input
    F = fixed(sA)
    let Dl = Diagonal(fixturedense(Float64, 3)), Dr = Diagonal(fixturedense(Float64, 7))
        @test mismatch(Dl * F, Dl * dA; approx=true) === nothing
        @test mismatch(F * Dr, dA * Dr; approx=true) === nothing
        @test _is_fixed(Dl * F) && _is_fixed(F * Dr)
    end
    # the kernels touch only the stored entries: exactly nnz(S) scalar multiplications,
    # whereas the generic Diagonal kernel visits every element of the result
    S = opcount_sparse(fixture(Float64, 20, 30))
    Dl = Diagonal(OpCount.(fixturedense(Float64, 30))); Dr = Diagonal(OpCount.(fixturedense(Float64, 20)))
    for W in (adjoint,)
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
    # `alpha == 0` ignores `A`, and `beta == 0` ignores `C`; the transpose shares the adjoint's kernels
    for W in (identity, adjoint)
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
    S = sparse([1.0 0; 0 2]); D = Diagonal([2.0, 3.0]); S1 = sparse([1.0 1; 0 1])
    # the plain matrix on the left and the adjoint on the right
    for W in (identity, adjoint),
            (f, x, y, x1, y1) in ((mul!, W(S), D, W(S1), D), (mul!, D, W(S), D, W(S1)))[W === identity ? (1:1) : (2:2)]
        F = fixed(sparse(ones(2, 2)))
        @test f(F, x, y) === F
        @test F == Matrix(x) * Matrix(y) && nnz(F) == 4 && _is_fixed(F)
        @test f(F, x, y, 2, 3) ≈ 5 * Matrix(x) * Matrix(y)
        G = fixed(sparse([1.0 0; 0 1]))
        @test_throws ArgumentError f(G, x1, y1)
        @test_throws ArgumentError f(G, x1, y1, 2, 3)
        @test G == [1 0; 0 1]
    end
end

@testset "scaling by a number, inverse scaling, non-commutative and 5-arg Diagonal mul!" begin
    sA, dA = fixturepair(Float64, 3, 7)
    sC = similar(sA)
    b = fixturedense(Float64, 3)
    @test mismatch(sA * 0.5, dA * 0.5) === nothing
    @test mismatch(mul!(sC, sA, 0.5), dA * 0.5) === nothing
    @test mismatch(rmul!(copy(sA), 0.5), dA * 0.5) === nothing
    @test mismatch(0.5 * sA, 0.5 * dA) === nothing
    @test mismatch(mul!(sC, sA, 0.5), 0.5 * dA) === nothing
    @test mismatch(lmul!(0.5, copy(sA)), 0.5 * dA) === nothing
    @test mul!(sC, 0.5, sA)   == mul!(sC, sA, 0.5)

    @testset "inverse scaling with mul!" begin
        bi = inv.(b)
        @test mismatch(ldiv!(Diagonal(b), copy(sA)), lmul!(Diagonal(bi), copy(dA)); approx=true) === nothing
        @test mismatch(ldiv!(transpose(Diagonal(b)), copy(sA)), lmul!(Diagonal(bi), copy(dA)); approx=true) === nothing
        @test mismatch(ldiv!(adjoint(Diagonal(b)), copy(sA)), lmul!(Diagonal(conj(bi)), copy(dA)); approx=true) === nothing
        Aob = Diagonal(b) \ sA
        @test Aob == ldiv!(Diagonal(b), copy(sA))
        @test issparse(Aob)
        @test_throws DimensionMismatch ldiv!(Diagonal(fill(1., length(b)+1)), copy(sA))
        @test_throws LinearAlgebra.SingularException ldiv!(Diagonal(zeros(length(b))), copy(sA))

        dAt = copy(transpose(dA))
        sAt = copy(transpose(sA))
        @test mismatch(rdiv!(copy(sAt), Diagonal(b)), rmul!(copy(dAt), Diagonal(bi)); approx=true) === nothing
        @test mismatch(rdiv!(copy(sAt), transpose(Diagonal(b))), rmul!(copy(dAt), Diagonal(bi)); approx=true) === nothing
        @test mismatch(rdiv!(copy(sAt), adjoint(Diagonal(b))), rmul!(copy(dAt), Diagonal(conj(bi))); approx=true) === nothing
        Atob = sAt / Diagonal(b)
        @test mismatch(Atob, rdiv!(copy(dAt), Diagonal(b))) === nothing
        @test issparse(Atob)
        @test_throws DimensionMismatch rdiv!(copy(sAt), Diagonal(fill(1., length(b)+1)))
        @test_throws LinearAlgebra.SingularException rdiv!(copy(sAt), Diagonal(zeros(length(b))))
    end

    @testset "non-commutative multiplication" begin
        Quaternion = quaternion_type()
        # the later testsets use the real `sA` and `dA` of the enclosing testset
        local sA, sC, dA
        Avals = Quaternion.(eachcol(fixturedense(Float64, 10, 4))...)
        # ten distinct positions, with the last two columns empty
        sA = sparse(mod1.(1:10, 3), mod1.(1:10, 5), Avals, 3, 7)
        sC = copy(sA)
        dA = Array(sA)

        b = Quaternion.(eachcol(fixturedense(Float64, 7, 4))...)
        D = Diagonal(b)
        @test mismatch(sA * D, dA * D; approx=true) === nothing
        @test rmul!(copy(sA), D) ≈ dA * D
        @static if COMPREHENSIVE
        @test mul!(sC, copy(sA), D) ≈ dA * D
        end

        b = Quaternion.(eachcol(fixturedense(Float64, 3, 4))...)
        D = Diagonal(b)
        @test mismatch(D * sA, D * dA; approx=true) === nothing
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
        # besides a destination with the pattern of `sA` and one with another pattern, one that
        # shares only the row indices (an empty column changes places) and one that shares only
        # the column pointers (the rows are reversed)
        for sA2 in (similar(sA), sparse([1, 3], [2, 6], [1.0, 1.0], size(sA)...), sA[:, [1, 3, 2, 4, 5, 6, 7]], sA[[3, 2, 1], :])
            nonzeros(sA2) .= 1
            @testset for (alpha, beta) in [(true, false), (true, true), (2,3)]
                D = Diagonal(fixturedense(Float64, size(sA,2)))
                @test mismatch(mul!(copy(sA2), sA, D, alpha, beta), dA * D * alpha + sA2 * beta; approx=true) === nothing
                D = Diagonal(fixturedense(Float64, size(sA,1)))
                @test mismatch(mul!(copy(sA2), D, sA, alpha, beta), D * dA * alpha + sA2 * beta; approx=true) === nothing
            end
        end
    end

    @testset "scale" begin
        x = fixturevec(Float64, 16)
        α = 2.5
        sx = SparseVector(length(x::SparseVector), nonzeroinds(x), nonzeros(x) * α)
        @test exact_equal(x * α, sx)
        @test exact_equal(x * (α + 0.0*im), complex(sx))
        @test exact_equal(α * x, sx)
        @test exact_equal((α + 0.0*im) * x, complex(sx))
        @static if COMPREHENSIVE
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
    S = fixture(ComplexF64, 3, 5)
    D1 = Diagonal(axes(S,1))
    D2 = Diagonal(axes(S,2) .+ 4)
    A = Array(S)
    C = D1 * S * D2
    @test C isa SparseMatrixCSC
    @test mismatch(C, D1 * A * D2; approx=true) === nothing
    C = D2 * S' * D1
    @test C isa SparseMatrixCSC
    @test mismatch(C, D2 * A' * D1; approx=true) === nothing
    C = D1 * view(S, :, :) * D2
    @test C isa SparseMatrixCSC
    @test mismatch(C, D1 * A * D2; approx=true) === nothing

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
        # comprehensive ones repeat a value
        for (α, β) in ((true, false), (false, true), (zero(ElType), one(ElType)), (one(ElType), general), (general, zero(ElType)),
                (@static COMPREHENSIVE ? zip(vs, vs) : ())...)
            C .= fixturedense(ElType, size(C)...)
            expected′ = expected .* α .+ C .* β
            @test mul!(C, A, B, α, β) === C
            @test C ≈ expected′
        end
    end

    # BigFloat adds the generic non-BLAS path to what the BLAS eltypes cover
    for ElType in (@static COMPREHENSIVE ? (Float64, ComplexF64, BigFloat) : (Float64, ComplexF64))
        SP = fixture(ElType, 10, 10)
        D = fixturedense(ElType, 10, 10)
        fs = (identity, adjoint, transpose)
        # every transform once on each side; adjoint and transpose differ only for a complex eltype
        for (f1, f2) in (TRANSFORM_PAIRS[1:(ElType <: Real ? 1 : end)]..., ((@static COMPREHENSIVE && ElType <: Complex) ?
                ((adjoint, adjoint),) : ())...)
            test_mul(f1(SP), f2(D))
            test_mul(f1(D), f2(SP))
        end
        # Coefficients branch on the sparse transform and on plain/wrapped dense-left inputs;
        # the real BLAS eltype would repeat the complex one's branches.
        ElType === Float64 && continue
        for f in (@static COMPREHENSIVE ? (ElType <: Complex ? fs : (adjoint,)) : (identity, adjoint))
            test_mul_coefficients(f(SP), D)
            test_mul_coefficients(D, f(SP))
        end
        for f in (@static COMPREHENSIVE ? (adjoint, transpose)[1:(ElType <: Complex ? 2 : 1)] : (transpose,))
            test_mul_coefficients(f(D), SP)
        end
    end
end

@testset "BLAS Level-2" begin
    @testset "dense A * sparse x -> dense y" begin
        # standard: a plain matrix with either eltype, and the transposed and wrapped factors
        # with the complex one, for which they differ. Comprehensive: a plain and an adjoint
        # factor with a vector of the other eltype, and the remaining complex wrapped factors.
        cases = @static COMPREHENSIVE ? ((Float64, ComplexF64, 1), (ComplexF64, Float64, 3),
            (ComplexF64, ComplexF64, 4), (ComplexF64, ComplexF64, 5), (ComplexF64, ComplexF64, 6)) : ()
        for TA in (Float64, ComplexF64), Tx in (Float64, ComplexF64)
            T = Base.promote_op(LinearAlgebra.matprod, TA, Tx)
            sel(k) = (TA == Tx && (k == 1 || TA <: Complex && k in (2, 3, 7))) || (TA, Tx, k) in cases
            sel(1) && let A = fixturedense(TA, 9, 16), x = fixturevec(Tx, 16)
                xf = Array(x)
                for α in [0.0, 1.0, 2.0], β in [0.0, 0.5, 1.0]
                    y = fixturedense(T, 9)
                    rr = α*A*xf + β*y
                    @test mul!(y, A, x, α, β) === y
                    @test y ≈ rr
                end
                y = A*x
                @test isa(y, Vector{T})
                @test A*x ≈ A*xf
            end

            sel(2) && let A = fixturedense(TA, 16, 9), x = fixturevec(Tx, 16)
                xf = Array(x)
                for α in [0.0, 1.0, 2.0], β in [0.0, 0.5, 1.0]
                    y = fixturedense(T, 9)
                    rr = α*transpose(A)*xf + β*y
                    @test mul!(y, transpose(A), x, α, β) === y
                    @test y ≈ rr
                end
                y = *(transpose(A), x)
                @test isa(y, Vector{T})
                @test y ≈ *(transpose(A), xf)
            end

            sel(3) && let A = fixturedense(TA, 16, 9), x = fixturevec(Tx, 16)
                xf = Array(x)
                for α in [0.0, 1.0, 2.0], β in [0.0, 0.5, 1.0]
                    y = fixturedense(T, 9)
                    rr = α*A'xf + β*y
                    @test mul!(y, adjoint(A), x, α, β) === y
                    @test y ≈ rr
                end
                y = *(adjoint(A), x)
                @test isa(y, Vector{T})
                @test y ≈ *(adjoint(A), xf)
            end

            let A = fixturedense(TA, 16, 16), x = fixturevec(Tx, 16)
                xf = Array(x)
                for (k, wrap) in enumerate((M -> Symmetric(M, :U), M -> Symmetric(M, :L),
                        M -> Hermitian(M, :U), M -> Hermitian(M, :L)))
                    sel(k + 3) || continue
                    for α in (0.0, 1.0, 2.0), β in (0.0, 0.5, 1.0)
                        y = fixturedense(T, 16)
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
        let A = fixture(Float64, 9, 16), x = fixturevec(Float64, 16)
            Af = Array(A)
            xf = Array(x)
            for α in [0.0, 1.0, 2.0], β in [0.0, 0.5, 1.0]
                y = fixturedense(Float64, 9)
                rr = α*Af*xf + β*y
                @test mul!(y, A, x, α, β) === y
                @test y ≈ rr
            end
            y = SparseArrays.densemv(A, x)
            @test isa(y, Vector{Float64})
            @test y ≈ Af*xf
        end

        let A = fixture(Float64, 16, 9), x = fixturevec(Float64, 16)
            Af = Array(A)
            xf = Array(x)
            for α in [0.0, 1.0, 2.0], β in [0.0, 0.5, 1.0]
                y = fixturedense(Float64, 9)
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

        let A = fixture(Float64, 16, 16), x = fixturevec(Float64, 16)
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
                # of an adjoint parent; triangular.jl owns the triangular grid. Comprehensive adds
                # the Hermitian wrapper, a unit triangle of a transposed and of an adjoint parent,
                # and a triangle of a symmetric wrapper.
                M -> UpperTriangular(Symmetric(M)))[(@static COMPREHENSIVE ? [1, 2, 3, 4, 5, 12, 14, 15, 17] : [1, 2, 5, 15])]
                for α in (0.0, 1.0, 2.0), β in (0.0, 0.5, 1.0)
                    y = fixturedense(Float64, 16)
                    rr = α*wrap(Af)*xf + β*y
                    @test mul!(y, wrap(A), x, α, β) === y
                    @test y ≈ rr
                end
                y = wrap(A) * x
                @test y ≈ *(wrap(Af), xf)
            end
        end

        let A = fixture(ComplexF64, 7, 8),
            x = fixturevec(ComplexF64, 8),
            x2 = fixturevec(ComplexF64, 7)
            Af = Array(A)
            xf = Array(x)
            x2f = Array(x2)
            @test SparseArrays.densemv(A, x; trans='N') ≈ Af * xf
            @test SparseArrays.densemv(A, x2; trans='T') ≈ transpose(Af) * x2f
            @test SparseArrays.densemv(A, x2; trans='C') ≈ Af'x2f
            @test_throws ArgumentError SparseArrays.densemv(A, x; trans='D')
        end

        @static if COMPREHENSIVE
        let A = map(!iszero, fixture(Float64, 9, 16)), x = map(!iszero, fixturevec(Float64, 16))
            Af = Array(A)
            xf = Array(x)
            y = SparseArrays.densemv(A, x)
            @test isa(y, Vector{Int})
            @test y == Af*xf
        end
        end
    end
    @testset "sparse A * sparse x -> sparse y" begin
        let A = fixture(Float64, 9, 16), x = fixturevec(Float64, 16), x2 = fixturevec(Float64, 9)
            Af = Array(A)
            xf = Array(x)
            x2f = Array(x2)

            y = A*x
            @test isa(y, SparseVector{Float64,Int})
            @test all(nonzeros(y) .!= 0.0)
            @test mismatch(y, Af * xf; Ti=Int, approx=true) === nothing
            @test mismatch((A * view(x, :))::SparseVector{Float64,Int}, Af * xf; approx=true) === nothing

            y = *(transpose(A), x2)
            @test isa(y, SparseVector{Float64,Int})
            @test all(nonzeros(y) .!= 0.0)
            @test mismatch(y, Af'x2f; Ti=Int, approx=true) === nothing
        end

        let A = fixture(ComplexF64, 7, 8),
            x = fixturevec(ComplexF64, 8),
            x2 = fixturevec(ComplexF64, 7)
            Af = Array(A)
            xf = Array(x)
            x2f = Array(x2)

            y = A*x
            @test isa(y, SparseVector{ComplexF64,Int})
            @test mismatch(y, Af * xf; Ti=Int, approx=true) === nothing

            y = *(transpose(A), x2)
            @test isa(y, SparseVector{ComplexF64,Int})
            @test mismatch(y, transpose(Af) * x2f; Ti=Int, approx=true) === nothing

            y = *(adjoint(A), x2)
            @test isa(y, SparseVector{ComplexF64,Int})
            @test mismatch(y, Af'x2f; Ti=Int, approx=true) === nothing

            @static if COMPREHENSIVE
            A32 = SparseMatrixCSC{ComplexF64,Int32}(A)
            # an index type of the vector that promotes with the matrix's, and one that matches
            for (x32, op) in ((x2, transpose), (SparseVector{ComplexF64,Int32}(x2), adjoint))
                y = op(A32) * x32
                @test isa(y, SparseVector{ComplexF64,promote_type(Int32, eltype(nonzeroinds(x32)))})
                @test mismatch(y, op(Af) * x2f; Ti=promote_type(Int32, eltype(nonzeroinds(x32))), approx=true) === nothing
            end
            end
        end

        let A = map(!iszero, fixture(Float64, 9, 16)), x = map(!iszero, fixturevec(Float64, 16)), x2 = map(!iszero, fixturevec(Float64, 9))
            Af = Array(A)
            xf = Array(x)
            x2f = Array(x2)

            y = A*x
            @test isa(y, SparseVector{Int, Int})
            @test mismatch(y, Af*xf; Ti=Int) === nothing

            @static if COMPREHENSIVE
            y = A'*x2
            @test isa(y, SparseVector{Int, Int})
            @test mismatch(y, Af'x2f; Ti=Int) === nothing
            end
        end
    end
    @static if COMPREHENSIVE
    @testset "sparse A * dense x -> dense y" begin
        let A = map(!iszero, fixture(Float64, 9, 16)), x = Vector(map(!iszero, fixturevec(Float64, 16))), x2 = Vector(map(!iszero, fixturevec(Float64, 9)))
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
    D = fixturedense(Float64, 7, 7)
    m = size(D, 1)
    # one operand of each kind gives the same dense result as its dense copy
    B, C, b = fixture(Float64, m, 3), fixture(Float64, 3, m), fixturevec(Float64, m)
    # comprehensive: the lq Q, which has a method of its own, and every kind of sparse operand
    # once, shared between the two
    @testset "$name" for (k, name, Q) in ((1, "qr", qr(D).Q), (@static COMPREHENSIVE ? ((2, "lq", lq(D).Q),) : ())...)
        for X in (B, (@static COMPREHENSIVE ? ((sparse(B')', view(B, :, 1:2))[k],) : ())...)
            @test (Q * X)::Matrix ≈ Q * Matrix(X)
        end
        for X in (C, (@static COMPREHENSIVE ? (transpose(sparse(transpose(C))), view(C, :, 1:m), view(B, :, 1:2)', transpose(b))[k:2:end] : ())...)
            @test (X * Q')::Matrix ≈ Matrix(X) * Q'
        end
        @test (Q' * B)::Matrix ≈ Q' * Matrix(B)
        @test (C * Q)::Matrix ≈ Matrix(C) * Q
        for x in (b, (@static COMPREHENSIVE ? ((view(B, :, 1), view(b, 1:m))[k],) : ())...)
            @test (Q * x)::Vector ≈ Q * Vector(x)
        end
        @test (Q' * b)::Vector ≈ Q' * Vector(b)
        @test (b' * Q)::Adjoint ≈ Vector(b)' * Q
        @test_throws DimensionMismatch Q * fixture(Float64, m + 1, 2)
    end
    # one method serves the left Q types; the lq Q has its own
    @static if !COMPREHENSIVE
    let Q = lq(D).Q
        @test (Q * B)::Matrix ≈ Q * Matrix(B)
        @test (C * Q)::Matrix ≈ Matrix(C) * Q
        @test (Q' * b)::Vector ≈ Q' * Vector(b)
    end
    end
end

@testset "product kernels touch stored entries only" begin
    n = 8
    # adjoint dense times adjoint sparse reads each entry of the dense factor at most once
    A = fixture(ComplexF64, 6, n); X = fixturedense(ComplexF64, n, 5); C0 = fixturedense(ComplexF64, 5, 6)
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
    A = fixturestrided(Float64, 400, 400, 99); xs = fixturevec(Float64, 400); ys = zeros(400)
    mul!(ys, A', xs, 2.0, 0.5)
    @test (@allocated mul!(ys, A', xs, 2.0, 0.5)) < 1000
end

@testset "dimension mismatch error" begin
    fs = [(x, y)->fixturedense(Float64, x, y), (x, y)->adjoint(fixturedense(Float64, y, x)), (x, y)->transpose(fixturedense(Float64, y, x)),
          (x, y)->fixture(Float64, x, y), (x, y)->adjoint(fixture(Float64, y, x)),
          (x, y)->transpose(fixture(Float64, y, x))]
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
