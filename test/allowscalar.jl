using Test, SparseArrays

@testset "allowscalar" begin
    A = sprandn(10, 20, 0.9)
    A[1, 1] = 2
    @test A[1, 1] == 2
    SparseArrays.@allowscalar(false)
    @test_throws Any A[1, 1]
    @test_throws Any A[1, 1] = 2
    SparseArrays.@allowscalar(true)
    @test A[1, 1] == 2
    A[1, 1] = 3
    @test A[1, 1] == 3

    B = sprandn(10, 0.9)
    B[1] = 2
    @test B[1] == 2
    SparseArrays.@allowscalar(false)
    @test_throws Any B[1]
    SparseArrays.@allowscalar(true)
    @test B[1] == 2
    B[1] = 3
    @test B[1] == 3
end
