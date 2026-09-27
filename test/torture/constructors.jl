# This file is a part of Julia. License is MIT: https://julialang.org/license

# Long-tail regression tests for the sparse constructors. Each testset names the issue it
# guards; they run only when the `torture` suite is selected.

module TortureConstructorsTests
using Test
using SparseArrays
using LinearAlgebra
include("../testhelpers.jl")
using Random

@testset "Issue #15" begin
    s = sparse([1, 2], [1, 2], [10, missing])
    d = Matrix(s)

    s2 = sparse(d)

    @test s2[1, 1] == 10
    @test s2[2, 1] == 0
    @test s2[1, 2] == 0
    @test s2[2, 2] === missing
    @test typeof(s2) == typeof(s)

    x = spzeros(3)
    y = similar(x, Union{Int, Missing})
    y[1] = missing
    y[2] = 10
    @test y[1] === missing
    @test y[2] == 10
    @test y[3] == 0
end

@testset "Issue #30502" begin
    @test nnz(sprand(UInt8(16), UInt8(16), 1.0)) == 256
    @test nnz(sprand(UInt8(16), UInt8(16), 1.0, ones)) == 256
end

@testset "Issue #5190" begin
    @test_throws ArgumentError sparsevec([3,5,7],[0.1,0.0,3.2],4)
end

@testset "issue #5985" begin
    @test sprand(Bool, 4, 5, 0.0) == sparse(zeros(Bool, 4, 5))
    @test sprand(Bool, 4, 5, 1.00) == sparse(fill(true, 4, 5))
    sprb45nnzs = zeros(5)
    for i=1:5
        sprb45 = sprand(Bool, 4, 5, 0.5)
        @test length(sprb45) == 20
        sprb45nnzs[i] = sum(sprb45)[1]
    end
    @test 4 <= sum(sprb45nnzs)/length(sprb45nnzs) <= 16
end

@testset "issue #8225" begin
    @test_throws ArgumentError sparse([0],[-1],[1.0],2,2)
end

@testset "issue #9525" begin
    @test_throws ArgumentError sparse([3], [5], 1.0, 3, 3)
end

@testset "issue #10411" begin
    for (m,n) in ((2,-2),(-2,2),(-2,-2))
        @test_throws ArgumentError spzeros(m,n)
        @test_throws ArgumentError sparse(1.0I, m, n)
        @test_throws ArgumentError sprand(m,n,0.2)
    end
end

@testset "issues #10837 & #32466, sparse constructors from special matrices" begin
    T = Tridiagonal(randn(4),randn(5),randn(4))
    S = sparse(T)
    S2 = SparseMatrixCSC(T)
    @test Array(T) == Array(S) == Array(S2)
    @test S == S2
    T = SymTridiagonal(randn(5),rand(4))
    S = sparse(T)
    S2 = SparseMatrixCSC(T)
    @test Array(T) == Array(S) == Array(S2)
    @test S == S2
    B = Bidiagonal(randn(5),randn(4),:U)
    S = sparse(B)
    S2 = SparseMatrixCSC(B)
    @test Array(B) == Array(S) == Array(S2)
    @test S == S2
    B = Bidiagonal(randn(5),randn(4),:L)
    S = sparse(B)
    S2 = SparseMatrixCSC(B)
    @test Array(B) == Array(S) == Array(S2)
    @test S == S2
    D = Diagonal(randn(5))
    S = sparse(D)
    S2 = SparseMatrixCSC(D)
    @test Array(D) == Array(S) == Array(S2)
    @test S == S2

    # An issue discovered in #42574 where
    # SparseMatrixCSC{Tv, Ti}(::Diagonal) ignored Ti
    D = Diagonal(rand(3))
    S = SparseMatrixCSC{Float64, Int8}(D)
    @test S isa SparseMatrixCSC{Float64, Int8}
end

@testset "issue #12177, error path if triplet vectors are not all the same length" begin
    @test_throws ArgumentError sparse([1,2,3], [1,2], [1,2,3], 3, 3)
    @test_throws ArgumentError sparse([1,2,3], [1,2,3], [1,2], 3, 3)
end

@testset "issue #13008" begin
    @test_throws ArgumentError sparse(Vector(1:100), Vector(1:100), fill(5,100), 5, 5)
    @test_throws ArgumentError sparse(Int[], Vector(1:5), Vector(1:5))
end

@testset "issue described in https://groups.google.com/forum/#!topic/julia-dev/QT7qpIpgOaA" begin
    @test sparse([1,1], [1,1], [true, true]) == sparse([1,1], [1,1], [true, true], 1, 1) == fill(true, 1, 1)
    @test sparsevec([1,1], [true, true]) == sparsevec([1,1], [true, true], 1) == fill(true, 1)
end

@testset "issue #16073" begin
    @inferred sprand(1, 1, 1.0)
    @inferred sprand(1, 1, 1.0, rand, Float64)
    @inferred sprand(1, 1, 1.0, x -> round.(Int, rand(x) * 100))
end

@testset "Issue #28634" begin
    a = SparseMatrixCSC{Int8, Int16}([1 2; 3 4])
    na = SparseMatrixCSC(a)
    @test typeof(a) === typeof(na)
end

@testset "Ti cannot store all potential values #31024" begin
    # m * n >= typemax(Ti) but nnz < typemax(Ti)
    A = SparseMatrixCSC(12, 12, fill(Int8(1),13), Int8[], Int[])
    @test size(A) == (12,12) && nnz(A) == 0
    I1 = [Int8(i) for i in 1:20 for _ in 1:20]
    J1 = [Int8(i) for _ in 1:20 for i in 1:20]
    # m * n >= typemax(Ti) and nnz >= typemax(Ti)
    @test_throws ArgumentError sparse(I1, J1, ones(length(I1)))
    I1 = Int8.(rand(1:10, 500))
    J1 = Int8.(rand(1:10, 500))
    V1 = ones(500)
    # m * n < typemax(Ti) and length(I) >= typemax(Ti) - combining values
    @test_throws ArgumentError sparse(I1, J1, V1, 10, 10)
    # m * n >= typemax(Ti) and length(I) >= typemax(Ti)
    @test_throws ArgumentError sparse(I1, J1, V1, 12, 13)
    I1 = Int8.(rand(1:10, 126))
    J1 = Int8.(rand(1:10, 126))
    V1 = ones(126)
    # m * n >= typemax(Ti) and length(I) < typemax(Ti)
    @test size(sparse(I1, J1, V1, 100, 100)) == (100,100)
end

@testset "Typecheck too strict #31435" begin
    A = SparseMatrixCSC{Int,Int8}(70, 2, fill(Int8(1), 3), Int8[], Int[])
    A[5:67,1:2] .= ones(Int, 63, 2)
    @test nnz(A) == 126
    # nnz >= typemax
    @test_throws ArgumentError A[2,1] = 42
    # colptr short
    @test_throws ArgumentError SparseMatrixCSC(1, 1, Int[], Int[], Float64[])
    # colptr[1] must be 1
    @test_throws ArgumentError SparseMatrixCSC(10, 3, [0,1,1,1], Int[], Float64[])
    # colptr not ascending
    @test_throws ArgumentError SparseMatrixCSC(10, 3, [1,2,1,2], Int[], Float64[])
    # rowwal (and nzval) short
    @test_throws ArgumentError SparseMatrixCSC(10, 3, [1,2,2,4], [1,2], Float64[])
    # length(nzval) >= typemax
    @test_throws ArgumentError SparseMatrixCSC(5, 1, Int8[1,2], fill(Int8(1), 127), fill(7, 127))

    # length(I) >= typemax
    @test_throws ArgumentError sparse(UInt8.(1:255), fill(UInt8(1), 255), fill(1, 255))
    # m > typemax
    @test_throws ArgumentError sparse(UInt8.(1:254), fill(UInt8(1), 254), fill(1, 254), 256, 1)
    # n > typemax
    @test_throws ArgumentError sparse(UInt8.(1:254), fill(UInt8(1), 254), fill(1, 254), 255, 256)
    # n, m maximal
    @test sparse(UInt8.(1:254), fill(UInt8(1), 254), fill(1, 254), 255, 255) !== nothing
end

@testset "avoid aliasing of fields during constructing $T (issue #34630)" for T in
    (SparseMatrixCSC, SparseMatrixCSC{Float64}, SparseMatrixCSC{Float64,Int16})

    A = sparse([1 1; 1 0])
    B = T(A)
    @test A == B
    A[2,2] = 1
    @test A != B
    @test getcolptr(A) !== getcolptr(B)
    @test rowvals(A) !== rowvals(B)
    @test nonzeros(A) !== nonzeros(B)
end

@testset "Issue #574" begin
    a = spzeros(Float32, Int16, 2, 3)
    v = spzeros(Float32, Int16, 2)
    @test eltype(rowvals(zero(a))) <: Int16
    @test eltype(rowvals(zero(v))) <: Int16
end

# Ten seeds of the `sprand` against `randsubseq` comparison; the core suite runs one.
@testset "sprand" begin
    p=0.3; m=1000; n=2000;
    for s in 1:10
        # build a (dense) random matrix with randsubset + rand
        Random.seed!(s);
        v = randsubseq(1:m*n,p);
        x = zeros(m,n);
        x[v] .= rand(length(v));
        # redo the same with sprand
        Random.seed!(s);
        a = sprand(m,n,p);
        @test x == a
    end
end

end # module
