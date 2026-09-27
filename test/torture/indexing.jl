# This file is a part of Julia. License is MIT: https://julialang.org/license

# Long-tail regression tests for sparse indexing. Each testset names the issue it guards;
# they run only when the `torture` suite is selected.

module TortureIndexingTests
using Test
using SparseArrays
using LinearAlgebra
using Test: guardseed
include("../testhelpers.jl")

@testset "Issue #30006" begin
    A = SparseMatrixCSC{Float64,Int32}(spzeros(3,3))
    A[:, 1] = [1, 2, 3]
    @test nnz(A) == 3
    @test nonzeros(A) == [1, 2, 3]
end

@testset "Issue #28963" begin
    @test_throws DimensionMismatch (spzeros(10,10)[:, :] = sprand(10,20,0.5))
end

@testset "issue #9917" begin
    @test sparse([]') == reshape(sparse([]), 1, 0)
    @test Array(sparse([])) == zeros(0)
    @test_throws BoundsError sparse([])[1]
    @test_throws BoundsError sparse([])[1] = 1
    x = sparse(1.0I, 100, 100)
    @test_throws BoundsError x[-10:10]
end

@testset "issue #14398" begin
    @test collect(view(sparse(I, 10, 10), 1:5, 1:5)') ≈ Matrix(I, 5, 5)
end

@testset "dropstored issue #20513" begin
    x = sparse(rand(3,3))
    SparseArrays.dropstored!(x, 1, 1)
    @test x[1, 1] == 0.0
    @test getcolptr(x) == [1, 3, 6, 9]
    SparseArrays.dropstored!(x, 2, 1)
    @test getcolptr(x) == [1, 2, 5, 8]
    @test x[2, 1] == 0.0
    SparseArrays.dropstored!(x, 2, 2)
    @test getcolptr(x) == [1, 2, 4, 7]
    @test x[2, 2] == 0.0
    SparseArrays.dropstored!(x, 2, 3)
    @test getcolptr(x) == [1, 2, 4, 6]
    @test x[2, 3] == 0.0
end

@testset "setindex issue #20657" begin
    local A = spzeros(3, 3)
    I = [1, 1, 1]; J = [1, 1, 1]
    A[I, 1] .= 1
    @test nnz(A) == 1
    A[1, J] .= 1
    @test nnz(A) == 1
    A[I, J] .= 1
    @test nnz(A) == 1
end

@testset "setindex with vector eltype (#29034)" begin
    A = sparse([1], [1], [Vector{Float64}(undef, 3)], 3, 3)
    A[1,1] = [1.0, 2.0, 3.0]
    @test A[1,1] == [1.0, 2.0, 3.0]
    @test_throws BoundsError setindex!(A, [4.0, 5.0, 6.0], 4, 3)
    @test_throws BoundsError setindex!(A, [4.0, 5.0, 6.0], 3, 4)
end

@testset "reverse search direction if step < 0 #21986" begin
    local A, B
    A = guardseed(1234) do
        sprand(5, 5, 1/5)
    end
    A = max.(A, copy(A'))
    LinearAlgebra.fillstored!(A, 1)
    B = A[5:-1:1, 5:-1:1]
    @test issymmetric(B)
end

@testset "Issue #29" begin
    s = sprand(6, 6, .2)
    li = LinearIndices(s)
    ci = CartesianIndices(s)
    @test s[li] == s[ci] == s[Matrix(li)] == s[Matrix(ci)]
end

# #20711
@testset "vec returns a view" begin
    local A = sparse(Matrix(1.0I, 3, 3))
    local v = vec(A)
    v[1] = 2
    @test A[1,1] == 2
end

@testset "-0.0 (issue #294, pr #296)" begin
    v = spzeros(1)
    v[1] = -0.0
    @test v[1] === -0.0

    m = spzeros(1, 1)
    m[1, 1] = -0.0
    @test m[1, 1] === -0.0
end

# From Base's arrayops.jl
@testset "CartesianIndex" begin
   a = spzeros(2,3)
    @test CartesianIndices(size(a)) == eachindex(a)
    a[CartesianIndex{2}(2,3)] = 5
    @test a[2,3] == 5
    b = view(a, 1:2, 2:3)
    b[CartesianIndex{2}(1,1)] = 7
    @test a[1,2] == 7
end

@testset "Assignment of singleton array to sparse array (julia #43644)" begin
    K = spzeros(3,3)
    b = zeros(3,3)
    b[3,:] = [1,2,3]
    K[3,1:3] += [1.0 2.0 3.0]'
    @test K == b
    K[3:3,1:3] += zeros(1, 3)
    @test K == b
    K[3,1:3] += zeros(3)
    @test K == b
    K[3,:] += zeros(3,1)
    @test K == b
    @test_throws DimensionMismatch K[3,1:2] += [1.0 2.0 3.0]'
end

# From Base's abstractarray.jl
@testset "itr, iterate" begin
    r = sparse(2:3:8)
    itr = eachindex(r)
    y = iterate(itr)
    @test y !== nothing
    y = iterate(itr, y[2])
    y = iterate(itr, y[2])
    @test y !== nothing
    val, state = y
    @test r[val] == 8
    @test iterate(itr, state) == nothing
end

end # module
