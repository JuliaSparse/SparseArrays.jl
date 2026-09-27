# This file is a part of Julia. License is MIT: https://julialang.org/license

# Long-tail regression tests for concatenation. Each testset names the issue it guards;
# they run only when the `torture` suite is selected.

module TortureConcatenationTests
using Test
using SparseArrays
using LinearAlgebra
include("../testhelpers.jl")

# From Base's abstractarray.jl
@testset "julia #17088" begin
    n = 10
    M = rand(n, n)
    @testset "vector of vectors" begin
        v = [[M]; [M]] # using vcat
        @test size(v) == (2,)
        @test !issparse(v)
    end
    @testset "matrix of vectors" begin
        m1 = [[M] [M]] # using hcat
        m2 = [[M] [M];] # using hvcat
        @test m1 == m2
        @test size(m1) == (1,2)
        @test !issparse(m1)
        @test !issparse(m2)
    end
end

end # module
