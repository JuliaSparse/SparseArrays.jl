# This file is a part of Julia. License is MIT: https://julialang.org/license

# Random draws of the sparse solves that the core suite runs once; they run only when
# the `torture` suite is selected.

module TortureSparseLinalgSolversTests
using Test
using SparseArrays
using LinearAlgebra
include("../../testhelpers.jl")

# Five random draws; the core suite runs one.
@testset "complex left-division" begin
    for i = 1:5
        a = I + 0.1*sprandn(5, 5, 0.2)
        b = randn(5,3) + im*randn(5,3)
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())

        a = I + 0.1*sprandn(5, 5, 0.2) + 0.1*im*sprandn(5, 5, 0.2)
        b = randn(5,3)
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())
    end
end

end # module
