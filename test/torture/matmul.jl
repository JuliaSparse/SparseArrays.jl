# This file is a part of Julia. License is MIT: https://julialang.org/license

# Long-tail regression tests for sparse products. Each testset names the issue it guards;
# they run only when the `torture` suite is selected.

module TortureMatmulTests
using Test
using SparseArrays
using LinearAlgebra
include("../testhelpers.jl")

@testset "Issue #33169" begin
    m21 = sparse([1, 2], [2, 2], SimpleSMatrix{2,1}.([rand(2, 1), rand(2, 1)]), 2, 2)
    m12 = sparse([1, 2], [2, 2], SimpleSMatrix{1,2}.([rand(1, 2), rand(1, 2)]), 2, 2)
    m22 = sparse([1, 2], [2, 2], SimpleSMatrix{2,2}.([rand(2, 2), rand(2, 2)]), 2, 2)
    m23 = sparse([1, 2], [2, 2], SimpleSMatrix{2,3}.([rand(2, 3), rand(2, 3)]), 2, 2)
    v12 = sparsevec([2], SimpleSMatrix{1,2}.([rand(1, 2)]))
    v21 = sparsevec([2], SimpleSMatrix{2,1}.([rand(2, 1)]))
    @test m22 * m21 ≈ Matrix(m22) * Matrix(m21)
    @test m22' * m21 ≈ Matrix(m22') * Matrix(m21)
    @test m21' * m22 ≈ Matrix(m21') * Matrix(m22)
    @test m23' * m22 * m21 ≈ Matrix(m23') * Matrix(m22) * Matrix(m21)
    @test m21 * v12 ≈ Matrix(m21) * Vector(v12)
    @test m12' * v12 ≈ Matrix(m12') * Vector(v12)
    @test v21' * m22 ≈ Vector(v21)' * Matrix(m22)
    @test v12' * m21' ≈ Vector(v12)' * Matrix(m21)'
    @test v21' * v21 ≈ Vector(v21)' * Vector(v21)
    @test v21' * m22 * v21 ≈ Vector(v21)' * Matrix(m22) * Vector(v21)
end

#PR #29045
@testset "Issue #28934" begin
    A = sprand(5,5,0.5)
    D = Diagonal(rand(5))
    C = copy(A)
    m1 = which(mul!, Base.typesof(C,A,D,true,false))
    m2 = which(mul!, Base.typesof(C,D,A,true,false))
    @test m1.module == SparseArrays
    @test m2.module == SparseArrays
end

end # module
