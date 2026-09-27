# This file is a part of Julia. License is MIT: https://julialang.org/license

# Long-tail regression tests for the LinearAlgebra wrappers of sparse matrices. Each
# testset names the issue it guards; they run only when the `torture` suite is selected.

module TortureLinalgTests
using Test
using SparseArrays
using LinearAlgebra
include("../testhelpers.jl")

# Test temporary fix for issue #16548 in PR #16979. Somewhat brittle. Expect to remove with `\` revisions.
@testset "issue #16548" begin
    ms = methods(\, (SparseMatrixCSC, AbstractVecOrMat)).ms
    @test all(m -> m.module == SparseArrays, ms)
end

@testset "issue #19225" begin
    X = sparse([1 -1; -1 1])
    for T in (Symmetric, Hermitian)
        Y = T(copy(X))
        _Y = similar(Y)
        copyto!(_Y, Y)
        @test _Y == Y

        W = T(copy(X), :L)
        copyto!(W, Y)
        @test W.data == Y.data
        @test W.uplo != Y.uplo

        W[1,1] = 4
        @test W == T(sparse([4 -1; -1 1]))
        @test_throws ArgumentError (W[1,2] = 2)

        @test Y + I == T(sparse([2 -1; -1 2]))
        @test Y - I == T(sparse([0 -1; -1 0]))
        @test Y * I == Y

        @test Y .+ 1 == T(sparse([2 0; 0 2]))
        @test Y .- 1 == T(sparse([0 -2; -2 0]))
        @test Y * 2 == T(sparse([2 -2; -2 2]))
        @test Y / 1 == Y
    end
end

@testset "issue #29644" begin
    F = lu(Tridiagonal(sparse(1.0I, 3, 3)))
    @test F.L == Matrix(I, 3, 3)
    @test startswith(sprint(show, MIME("text/plain"), F),
                     "$(LinearAlgebra.LU){Float64, $(LinearAlgebra.Tridiagonal){Float64, $(SparseArrays.SparseVector)")
end

@testset "Symmetric and Hermitian #35325" begin
    A = sprandn(ComplexF64, 10, 10, 0.1)
    B = sprandn(ComplexF64, 10, 10, 0.1)

    @test Symmetric(real(A)) + Hermitian(B) isa Hermitian{ComplexF64, <:SparseMatrixCSC}
    @test Hermitian(A) + Symmetric(real(B)) isa Hermitian{ComplexF64, <:SparseMatrixCSC}
    @test Hermitian(A) + Symmetric(B) isa SparseMatrixCSC
    @testset "$Wrapper $op" for op ∈ (+, -), Wrapper ∈ (Hermitian, Symmetric)
        AWU = Wrapper(A, :U)
        AWL = Wrapper(A, :L)
        BWU = Wrapper(B, :U)
        BWL = Wrapper(B, :L)

        @test op(AWU, B) isa SparseMatrixCSC
        @test op(A, BWL) isa SparseMatrixCSC

        @test op(AWU, B) ≈ op(collect(AWU), B)
        @test op(AWL, B) ≈ op(collect(AWL), B)
        @test op(A, BWU) ≈ op(A, collect(BWU))
        @test op(A, BWL) ≈ op(A, collect(BWL))

        @test op(AWU, BWL) isa Wrapper{ComplexF64, <:SparseMatrixCSC}

        @test op(AWU, BWU) ≈ op(collect(AWU), collect(BWU))
        @test op(AWU, BWL) ≈ op(collect(AWU), collect(BWL))
        @test op(AWL, BWU) ≈ op(collect(AWL), collect(BWU))
        @test op(AWL, BWL) ≈ op(collect(AWL), collect(BWL))
    end
end

end # module
