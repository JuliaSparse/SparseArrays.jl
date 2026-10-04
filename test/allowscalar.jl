# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseAllowScalarTests

using Test, SparseArrays
include("testhelpers.jl")

# the flag is process-wide, so it is restored even when an assertion throws
@testset "allowscalar" begin try
    A = fixture(Float64, 3, 5)
    A[1, 1] = 2
    @test A[1, 1] == 2
    SparseArrays.@allowscalar(false)
    @test_throws Any A[1, 1]
    @test_throws Any A[1, 1] = 2
    SparseArrays.@allowscalar(true)
    @test A[1, 1] == 2
    A[1, 1] = 3
    @test A[1, 1] == 3

    B = fixturevec(Float64, 10)
    B[1] = 2
    @test B[1] == 2
    SparseArrays.@allowscalar(false)
    @test_throws Any B[1]
    SparseArrays.@allowscalar(true)
    @test B[1] == 2
    B[1] = 3
    @test B[1] == 3
finally
    SparseArrays.@allowscalar(true)
end end

end # module
