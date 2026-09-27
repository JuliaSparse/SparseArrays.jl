# This file is a part of Julia. License is MIT: https://julialang.org/license

# Long-tail regression tests for sparse reductions. Each testset names the issue it
# guards; they run only when the `torture` suite is selected.

module TortureReductionsTests
using Test
using SparseArrays
using LinearAlgebra
include("../testhelpers.jl")

@testset "issue #6036" begin
    P = spzeros(Float64, 3, 3)
    for i = 1:3
        P[i,i] = i
    end

    @test minimum(P) === 0.0
    @test maximum(P) === 3.0
    @test minimum(-P) === -3.0
    @test maximum(-P) === 0.0

    @test maximum(P, dims=(1,)) == [1.0 2.0 3.0]
    @test maximum(P, dims=(2,)) == reshape([1.0,2.0,3.0],3,1)
    @test maximum(P, dims=(1,2)) == reshape([3.0],1,1)

    @test maximum(sparse(fill(-1,3,3))) == -1
    @test minimum(sparse(fill(1,3,3))) == 1
end

@testset "issue #10407" begin
    @test maximum(spzeros(5, 5)) == 0.0
    @test minimum(spzeros(5, 5)) == 0.0
end

@testset "issue #31453" for T in [UInt8, Int8, UInt16, Int16, UInt32, Int32]
    i = Int[1, 2]
    j = Int[2, 1]
    i2 = T.(i)
    j2 = T.(j)
    v = [500, 600]
    x1 = sparse(i, j, v)
    x2 = sparse(i2, j2, v)
    @test sum(x1) == sum(x2) == 1100
    @test sum(x1, dims=1) == sum(x2, dims=1)
    @test sum(x1, dims=2) == sum(x2, dims=2)
end

# #25943
@testset "operations on Integer subtypes" begin
    s = sparse(UInt8[1, 2, 3], UInt8[1, 2, 3], UInt8[1, 2, 3])
    @test sum(s, dims=2) == reshape([1, 2, 3], 3, 1)
end

end # module
