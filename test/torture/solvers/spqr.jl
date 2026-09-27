# This file is a part of Julia. License is MIT: https://julialang.org/license

# The remaining eltype and index-type combinations of the SPQR wrapper tests, and
# long-tail regressions; they run only when the `torture` suite is selected.

module TortureSPQRTests
using Test
using SparseArrays.SPQR
using LinearAlgebra: I, istril, istriu, lq, norm, qr, rank, rmul!, lmul!, factorize, Adjoint
using SparseArrays: SparseArrays, sparse, sprandn, SparseMatrixCSC
include("../../testhelpers.jl")

m, n = 100, 10
nn = 100

# The eltype/index-type combinations of the main grid that the core suite does not run.
@testset "element type of A: $eltyA" for (eltyA, iltyA) in ((Float64, Int32), (ComplexF64, Int))
    if eltyA <: Real
        A = sparse(iltyA[1:n; rand(1:m, nn - n)], iltyA[1:n; rand(1:n, nn - n)], randn(nn), m, n)
    else
        A = sparse(iltyA[1:n; rand(1:m, nn - n)], iltyA[1:n; rand(1:n, nn - n)], complex.(randn(nn), randn(nn)), m, n)
    end

    F = qr(A)
    @test size(F) == (m,n)
    # qr of an adjoint or transpose factorizes the sparse transpose with SPQR
    for X in (A', transpose(A))
        @test which(qr, Base.typesof(X)).module == SPQR
        G = qr(X; tol = 1e-3)
        @test G isa SPQR.QRSparse{eltyA, iltyA} && size(G) == (n, m)
        @test G.Q * G.R ≈ Matrix(X)[G.prow, G.pcol]
    end
    @test size(F, 1) == m
    @test size(F, 2) == n
    @test size(F, 3) == 1
    @test_throws ArgumentError size(F, 0)

    @testset "getindex" begin
        @test istriu(F.R)
        @test isperm(F.pcol)
        @test isperm(F.prow)
        @test @inferred((F -> F.pcol)(F)) isa Vector{iltyA}
        @test @inferred((F -> F.prow)(F)) isa Vector{iltyA}
        @test_throws isdefined(Base, :FieldError) ? FieldError : ErrorException F.T
    end

    @testset "apply Q" begin
        Q = F.Q
        Imm = Matrix{Float64}(I, m, m)
        @test Q' * (Q*Imm) ≈ Imm
        @test (Imm*Q) * Q' ≈ Imm
        @test ((Imm[:,1])' * Q')::Adjoint ≈ Q[:,1]'

        # test that Q'Pl*A*Pr = R
        R0 = Q'*Array(A[F.prow, F.pcol])
        @test R0[1:n, :] ≈ F.R
        @test norm(R0[n + 1:end, :], 1) < 1e-12

        offsizeA = Matrix{Float64}(I, m+1, m+1)
        @test_throws DimensionMismatch lmul!(Q, offsizeA)
        @test_throws DimensionMismatch lmul!(adjoint(Q), offsizeA)
        @test_throws DimensionMismatch rmul!(offsizeA, Q)
        @test_throws DimensionMismatch rmul!(offsizeA, adjoint(Q))

        # products with an operand of another element type convert Q
        Qd = Q * Matrix{eltyA}(I, m, m)
        b, B = complex.(randn(m), randn(m)), complex.(randn(3, m), randn(3, m))
        @test Q * b ≈ Qd * b
        @test Q' * b ≈ Qd' * b
        @test B * Q' ≈ B * Qd'
    end

    @testset "right-hand sides that are not strided arrays of the same element type" begin
        rhs(k) = (randn(2, k)', transpose(randn(2, k)), sprandn(k, 0.5), 1:k,
                  view(complex.(randn(k, 2), randn(k, 2)), :, 1),
                  complex.(randn(2, k), randn(2, k))', randn(ComplexF32, k))
        C = A[1:9, :]   # wide
        for X in rhs(m)
            @test A \ X ≈ Array(A) \ Array(X)
            @test F \ X ≈ Array(A) \ Array(X)
        end
        for X in rhs(9)
            @test C \ X ≈ Array(C) \ Array(X)
            @test lq(C) \ X ≈ Array(C) \ Array(X)
        end
        for X in rhs(n)
            @test F' \ X ≈ Array(A)' \ Array(X)
        end
    end

    @testset "element type of B: $eltyB" for eltyB in (Int, Float64, ComplexF64)
        if eltyB == Int
            B = rand(1:10, m, 2)
        elseif eltyB <: Real
            B = randn(m, 2)
        else
            B = complex.(randn(m, 2), randn(m, 2))
        end

        @inferred A\B
        @test A\B[:,1] ≈ Array(A)\B[:,1]
        @test A\B ≈ Array(A)\B
        @test_throws DimensionMismatch A\B[1:m-1,:]
        C, x = A[1:9, :], fill(eltyB(1), 9)
        @test C*(C\x) ≈ x # Underdetermined system
        # A \ b returns the minimum-norm solution for a wide A, like dense (#301)
        @test C\x ≈ Array(C)\x
        @test C\B[1:9, :] ≈ Array(C)\B[1:9, :]
        @test factorize(C)\x ≈ Array(C)\x

        # Minimum-norm solution of the underdetermined A'x = b (#656)
        D = B[1:n, :]
        @test F'\D ≈ Array(A)'\D
        @test F'\D[:,1] ≈ Array(A)'\D[:,1]
        @test transpose(F)\D ≈ transpose(Array(A))\D
        @test A'\D ≈ Array(A)'\D
        @test_throws DimensionMismatch F'\B
        # Least squares solve of the overdetermined C'y = x for the wide C
        y = B[1:n, 1]
        @test C'\y ≈ Array(C)'\y
        @test transpose(C)\y ≈ transpose(Array(C))\y
    end

    @testset "lq (#114)" begin
        W = A[1:9, :]   # wide
        F = lq(W)
        @test F isa SPQR.AdjointQRSparse{eltyA} && size(F) == size(W)
        @test F.L isa SparseMatrixCSC{eltyA, iltyA} && istril(F.L)
        @test F.L * F.Q ≈ Matrix(W)[F.prow, F.pcol]
        @test @inferred((F -> F.pcol)(F)) isa Vector{iltyA}
        @test @inferred((F -> F.prow)(F)) isa Vector{iltyA}
        @test rank(F) == 9 && propertynames(F) == (:L, :Q, :prow, :pcol)
        @test F' isa SPQR.QRSparse{eltyA, iltyA}
        @test occursin("L factor", sprint(show, MIME"text/plain"(), F))
        b = eltyA <: Real ? randn(9, 2) : complex.(randn(9, 2), randn(9, 2))
        @test F \ b ≈ Matrix(W) \ b   # the minimum-norm solution, as for dense lq
        @test F \ b[:, 1] ≈ Matrix(W) \ b[:, 1]
        @test lq(W; tol = 1e-3) \ b ≈ Matrix(W) \ b
        @test_throws DimensionMismatch lq(A) \ ones(eltyA, m)   # overdetermined, as for dense lq
        c = eltyA <: Real ? randn(n) : complex.(randn(n), randn(n))
        @test lq(A') \ c ≈ Matrix(A') \ c   # reuses qr(A)
        eltyA <: Real && @test lq(transpose(A)) \ c ≈ Matrix(A') \ c
    end

    # Make sure that conversion to Sparse doesn't use SuiteSparse's symmetric flag
    @test qr(SparseMatrixCSC{eltyA}(I, 5, 5)) \ fill(eltyA(1), 5) == fill(1, 5)
end

@testset "Issue 26368" begin
    A = sparse([0.0 1 0 0; 0 0 0 0])
    F = qr(A)
    @test (F.Q*F.R)::Matrix == A[F.prow,F.pcol]
end

@testset "Issue #585 for element type: $eltyA" for eltyA in (Float32, Float16, ComplexF32, ComplexF16)
    A = sparse(eltyA[1 0; 0 1])
    F = qr(A)
    @test eltype(F.Q) == eltype(F.R) == eltyA
end

@testset "ORDERING_FIXED with a dependent column, $Tv $Ti" for (Tv, Ti) in ((Float64, Int32), (ComplexF64, Int))
    # the second column is twice the first
    A = SparseMatrixCSC{Tv, Ti}([1 2 3; 4 8 6; 7 14 9; 1 2 5])
    F = qr(A; ordering=SPQR.ORDERING_FIXED)
    @test rank(F) == 2
    @test isperm(F.pcol) && istriu(F.R)
    @test F.Q * F.R ≈ A[F.prow, F.pcol]
    b = A * Tv[1, 2, 3]
    @test A * (F \ b) ≈ b
    c = A' * Tv[1, 2, 3, 4]
    @test A' * (F' \ c) ≈ c
    W = sparse(A')   # wide
    G = qr(W; ordering=SPQR.ORDERING_FIXED)
    @test G.Q * G.R ≈ W[G.prow, G.pcol]
    @test W * (G \ c) ≈ c
    # without dependent columns the ordering is the identity
    @test qr(A[:, [1, 3]]; ordering=SPQR.ORDERING_FIXED).pcol == 1:2
end

end # module
