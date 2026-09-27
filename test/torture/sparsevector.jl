# This file is a part of Julia. License is MIT: https://julialang.org/license

# Long-tail regression tests for sparse vectors. Each testset names the issue it guards;
# they run only when the `torture` suite is selected.

module TortureSparseVectorTests
using Test
using SparseArrays
using LinearAlgebra
include("../testhelpers.jl")

@testset "issue #7507" begin
    @test (i7507=sparsevec(Dict{Int64, Float64}(), 10))==spzeros(10)
end

@testset "issue #8363" begin
    @test_throws ArgumentError sparsevec(Dict(-1=>1,1=>2))
end

@testset "issparse for sparse vectors #34253" begin
    v = sprand(10, 0.5)
    @test issparse(v)
    @test issparse(v')
    @test issparse(transpose(v))
end

@testset "reinterpret (issue #289, pr #296)" begin
    s = spzeros(3)
    r = reinterpret(Int64, s)
    @test r == s

    r[1] = Int64(12)
    @test r[1] === Int64(12)
    @test s[1] === reinterpret(Float64, Int64(12))
    @test r != s

    r[2] = Int64(0)
    @test r[2] === Int64(0)
    @test s[2] === 0.0

    z = reinterpret(Int64, -0.0)
    r[3] = z
    @test r[3] === z
    @test s[3] === -0.0
end

# From Base's arrayops.jl
@testset "copy!" begin
    @testset "AbstractVector" begin
        s = Vector([1, 2])
        for a = ([1], UInt[1], [3, 4, 5], UInt[3, 4, 5])
            @test s === copy!(s, SparseVector(a)) == Vector(a)
        end
    end
end

@testset "Issue #334" begin
    x = sprand(10, .3);
    @test issorted(sort!(x; alg=Base.DEFAULT_STABLE));
    @test_throws MethodError sort!(x; banana=:blue); # From discussion at #335
end

end # module
