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

# The sizes and the full density of the reduction grid that the core suite does not run.
@testset "reductions along a dimension, remaining sizes and densities (#43, #377)" begin
    reductions = (   # (f, op); the last one has f(0) != 0
        (identity, +), (identity, *), (identity, max), (abs2, +), (x -> x > 0.5, |), (x -> x >= 0, &), (x -> x + 1, +),
    )
    @testset "size = ($m, $n), density = $d" for (m, n, d) in (((m, n, d) for (m, n) in ((1, 1), (30, 20)), d in (0.0, 0.2, 1.0))...,
                                                       ((m, n, 1.0) for (m, n) in ((6, 5), (1, 9), (9, 1)))...)
        A = sparse(sprand(m, n, d) .- 0.5)   # negative entries, so that max and min do not see 0 as a bound
        M = Matrix(A)
        V = view(A, :, (n + 1) ÷ 2:n)   # a view of a column range reduces like its copy (#377)
        C = A[:, (n + 1) ÷ 2:n]
        @test nnz(V) == nnz(C)
        for dims in (1, 2, (1, 2), 3), (f, op) in reductions
            rd = mapreduce(f, op, M; dims)
            r = mapreduce(f, op, A; dims)
            @test r isa Matrix && r ≈ rd
            rv, rc = mapreduce(f, op, V; dims), mapreduce(f, op, C; dims)
            @test typeof(rv) == typeof(rc) && isequal(rv, rc)
            # opt-in: the sparse result has the element type and values of the dense one
            T = eltype(rd)
            rs = mapreduce(f, op, A; dims, sparse = true)
            @test rs isa SparseMatrixCSC{T} && rs ≈ rd
            rvs = mapreduce(f, op, V; dims, sparse = true)
            @test rvs isa SparseMatrixCSC{T} && rvs ≈ mapreduce(f, op, Matrix(C); dims)
        end
        for dims in (1, 2)
            @test sum(A; dims, sparse = true) ≈ sum(M; dims)
            @test sum(abs, V; dims, sparse = true) ≈ sum(abs, Matrix(C); dims)
            @test prod(A; dims, sparse = true) ≈ prod(M; dims)
            @test maximum(A; dims, sparse = true) == maximum(M; dims)
            @test minimum(abs2, A; dims, sparse = true) == minimum(abs2, M; dims)
            @test sum(A; dims, init = 2.5, sparse = true) ≈ sum(M; dims, init = 2.5)
            @test mapreduce(abs, (x, y) -> x + y, A; dims, init = 1.5, sparse = true) ≈
                  mapreduce(abs, (x, y) -> x + y, M; dims, init = 1.5)
            @test count(>(0), A; dims, sparse = true) == count(>(0), M; dims)
            @test count(A .> 0; dims, sparse = true) == count(M .> 0; dims)
            @test count(A .> 0; dims, init = 3, sparse = true) == count(M .> 0; dims, init = 3)
            @test any(>(0), A; dims, sparse = true) == any(>(0), M; dims)
            @test any(A .> 0; dims, sparse = true) == any(M .> 0; dims)
            @test all(<(0.4), A; dims, sparse = true) == all(<(0.4), M; dims)
            @test all(A .< 0.4; dims, sparse = true) == all(M .< 0.4; dims)
            for r in (count(>(0), A; dims, sparse = true), any(A .> 0; dims, sparse = true), all(A .< 0.4; dims, sparse = true))
                @test r isa SparseMatrixCSC
            end
            # the default result and the scalar reductions are unchanged
            @test sum(A; dims) isa Matrix{Float64} && count(A .> 0; dims) isa Matrix{Int} && any(A .> 0; dims) isa Matrix{Bool}
        end
        @test sum(A) ≈ sum(M) && count(>(0), A) == count(>(0), M) && any(A .> 0) == any(M .> 0) && all(A .< 0.4) == all(M .< 0.4)
        @test_throws ArgumentError sum(A; sparse = true)
    end
end

# The dense reference of the hypersparse reduction is 10^6 x 3.
@testset "hypersparse maximum along a dimension against dense" begin
    A = sparse([5, 10^6, 5], [1, 2, 3], [1.0, 2.0, 3.0], 10^6, 3)
    @test maximum(A; dims = 2, sparse = true) == maximum(Matrix(A); dims = 2)
end

# The transposed and adjoint forms of the wrapper loop that the core suite does not run.
@testset "reductions of adjoints and transposes call `f` per stored entry, remaining forms" begin
    A, C, v, c = sprand(60, 50, 0.05), sprand(ComplexF64, 60, 50, 0.05), sprand(60, 0.1), sprand(ComplexF64, 60, 0.1)
    S = view(A, :, [7, 2, 2, 15])
    for X in (transpose(C), v', c'), dims in (1, 2, (1, 2)),
        (f, op) in ((abs2, +), (abs, max), (x -> abs(x) + 1, (x, y) -> x + y))   # LinearAlgebra does not forward the last
        calls = Ref(0)
        rd = mapreduce(f, op, Array(X); dims, init = 0.0)
        r = mapreduce(x -> (calls[] += 1; f(x)), op, X; dims, init = 0.0)
        @test r isa Array && r ≈ rd
        @test calls[] <= nnz(X) + sum(size(X)) + 1
        rs = mapreduce(f, op, X; dims, init = 0.0, sparse = true)
        @test rs isa (X isa AbstractVector ? SparseVector{Float64} : SparseMatrixCSC{Float64}) && rs ≈ rd
    end
end

end # module
