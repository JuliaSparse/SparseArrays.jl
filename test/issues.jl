# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseIssuesTests
using Test
using SparseArrays
using SparseArrays: nonzeroinds, getcolptr, move_fixed
using LinearAlgebra
using Random
include("testhelpers.jl")


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

@testset "Issue #33169" begin
    # distinct blocks, so that a product taking the wrong block or order differs
    blk(k, m, n) = reshape(Float64.(k .+ (1:m*n)), m, n)
    m21 = sparse([1, 2], [2, 2], SimpleSMatrix{2,1}.([blk(1, 2, 1), blk(3, 2, 1)]), 2, 2)
    m12 = sparse([1, 2], [2, 2], SimpleSMatrix{1,2}.([blk(5, 1, 2), blk(7, 1, 2)]), 2, 2)
    m22 = sparse([1, 2], [2, 2], SimpleSMatrix{2,2}.([blk(9, 2, 2), blk(13, 2, 2)]), 2, 2)
    m23 = sparse([1, 2], [2, 2], SimpleSMatrix{2,3}.([blk(17, 2, 3), blk(23, 2, 3)]), 2, 2)
    v12 = sparsevec([2], SimpleSMatrix{1,2}.([blk(29, 1, 2)]))
    v21 = sparsevec([2], SimpleSMatrix{2,1}.([blk(31, 2, 1)]))
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

@testset "Issue #30006" begin
    A = SparseMatrixCSC{Float64,Int32}(spzeros(3,3))
    A[:, 1] = [1, 2, 3]
    @test nnz(A) == 3
    @test nonzeros(A) == [1, 2, 3]
end

@testset "Issue #28963" begin
    @test_throws DimensionMismatch (spzeros(10,10)[:, :] = fixture(Float64, 10, 20))
end

@testset "Issue #30502" begin
    Random.seed!(30502)
    @test nnz(sprand(UInt8(16), UInt8(16), 1.0)) == 256
    @test nnz(sprand(UInt8(16), UInt8(16), 1.0, ones)) == 256
end

@testset "Issue #5190" begin
    @test_throws ArgumentError sparsevec([3,5,7],[0.1,0.0,3.2],4)
end

@testset "what used to be issue #5386" begin
    K,J,V = findnz(SparseMatrixCSC(2,1,[1,3],[1,2],[1.0,0.0]))
    @test length(K) == length(J) == length(V) == 2
end

@testset "issue #5824" begin
    @test fixture(Float64, 4, 5).^0 == sparse(fill(1,4,5))
end

@testset "issue #5985" begin
    Random.seed!(5985)
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

@testset "issue #5853, sparse diff" begin
    for i=1:2, a=Any[[1 2 3], reshape([1, 2, 3],(3,1)), Matrix(1.0I, 3, 3)]
        @test diff(sparse(a),dims=i) == diff(a,dims=i)
    end
end

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

@testset "issue #7507" begin
    @test (i7507=sparsevec(Dict{Int64, Float64}(), 10))==spzeros(10)
end

@testset "issue #7650" begin
    S = spzeros(3, 3)
    @test size(reshape(S, 9, 1)) == (9,1)
end

@testset "issue #7677" begin
    A = fixture(Float64, 5, 5)
    ACPY = copy(A)
    B = reshape(A,25,1)
    @test A == ACPY
end

@testset "issue #8225" begin
    @test_throws ArgumentError sparse([0],[-1],[1.0],2,2)
end

@testset "issue #8363" begin
    @test_throws ArgumentError sparsevec(Dict(-1=>1,1=>2))
end

@testset "issue #8976" begin
    @test conj.(sparse([1im])) == sparse(conj([1im]))
    @test conj!(sparse([1im])) == sparse(conj!([1im]))
end

@testset "issue #9525" begin
    @test_throws ArgumentError sparse([3], [5], 1.0, 3, 3)
end

@testset "issue #9917" begin
    @test sparse([]') == reshape(sparse([]), 1, 0)
    @test Array(sparse([])) == zeros(0)
    @test_throws BoundsError sparse([])[1]
    @test_throws BoundsError sparse([])[1] = 1
    x = sparse(1.0I, 100, 100)
    @test_throws BoundsError x[-10:10]
end

@testset "issue #10407" begin
    @test maximum(spzeros(5, 5)) == 0.0
    @test minimum(spzeros(5, 5)) == 0.0
end

@testset "issue #10411" begin
    for (m,n) in ((2,-2),(-2,2),(-2,-2))
        @test_throws ArgumentError spzeros(m,n)
        @test_throws ArgumentError sparse(1.0I, m, n)
        @test_throws ArgumentError sprand(m,n,0.2)
    end
end

@testset "issues #10837 & #32466, sparse constructors from special matrices" begin
    T = Tridiagonal([1.0, -2.0, 3.0, -4.0], [5.0, 6.0, -7.0, 8.0, 9.0], [-10.0, 11.0, 12.0, 13.0])
    S = sparse(T)
    S2 = SparseMatrixCSC(T)
    @test mismatch(S, Matrix(T)) === nothing
    @test mismatch(S2, Matrix(T)) === nothing
    T = SymTridiagonal([5.0, 6.0, -7.0, 8.0, 9.0], [1.0, 2.0, 3.0, 4.0])
    S = sparse(T)
    S2 = SparseMatrixCSC(T)
    @test mismatch(S, Matrix(T)) === nothing
    @test mismatch(S2, Matrix(T)) === nothing
    B = Bidiagonal([5.0, 6.0, -7.0, 8.0, 9.0], [1.0, -2.0, 3.0, -4.0], :U)
    S = sparse(B)
    S2 = SparseMatrixCSC(B)
    @test mismatch(S, Matrix(B)) === nothing
    @test mismatch(S2, Matrix(B)) === nothing
    B = Bidiagonal([5.0, 6.0, -7.0, 8.0, 9.0], [1.0, -2.0, 3.0, -4.0], :L)
    S = sparse(B)
    S2 = SparseMatrixCSC(B)
    @test mismatch(S, Matrix(B)) === nothing
    @test mismatch(S2, Matrix(B)) === nothing
    D = Diagonal([5.0, 6.0, -7.0, 8.0, 9.0])
    S = sparse(D)
    S2 = SparseMatrixCSC(D)
    @test mismatch(S, Matrix(D)) === nothing
    @test mismatch(S2, Matrix(D)) === nothing

    # An issue discovered in #42574 where
    # SparseMatrixCSC{Tv, Ti}(::Diagonal) ignored Ti
    D = Diagonal([1.0, 2.0, 3.0])
    S = SparseMatrixCSC{Float64, Int8}(D)
    @test S isa SparseMatrixCSC{Float64, Int8}
end

@testset "issue #12118: sparse matrices are closed under +, -, min, max" begin
    A12118 = sparse([1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5])
    B12118 = sparse([1,2,4,5],   [1,2,3,5],   [2,1,-1,-2])

    result = A12118 + B12118
    @test result == sparse([1,2,3,4,4,5], [1,2,3,3,4,5], [3,3,3,-1,4,3])
    @test typeof(result) == SparseMatrixCSC{Int,Int}

    result = A12118 - B12118
    @test result == sparse([1,2,3,4,4,5], [1,2,3,3,4,5], [-1,1,3,1,4,7])
    @test typeof(result) == SparseMatrixCSC{Int,Int}

    result = max.(A12118, B12118)
    @test result == sparse([1,2,3,4,5], [1,2,3,4,5], [2,2,3,4,5])
    @test typeof(result) == SparseMatrixCSC{Int,Int}

    result = min.(A12118, B12118)
    @test result == sparse([1,2,4,5], [1,2,3,5], [1,1,-1,-2])
    @test typeof(result) == SparseMatrixCSC{Int,Int}
end

@testset "issue #13008" begin
    @test_throws ArgumentError sparse(Vector(1:100), Vector(1:100), fill(5,100), 5, 5)
    @test_throws ArgumentError sparse(Int[], Vector(1:5), Vector(1:5))
end

@testset "issue #13024" begin
    A13024 = sparse([1,2,3,4,5], [1,2,3,4,5], fill(true,5))
    B13024 = sparse([1,2,4,5],   [1,2,3,5],   fill(true,4))

    result = broadcast(&, A13024, B13024)
    @test result == sparse([1,2,5], [1,2,5], fill(true,3))
    @test typeof(result) == SparseMatrixCSC{Bool,Int}

    result = broadcast(|, A13024, B13024)
    @test result == sparse([1,2,3,4,4,5], [1,2,3,3,4,5], fill(true,6))
    @test typeof(result) == SparseMatrixCSC{Bool,Int}

    result = broadcast(⊻, A13024, B13024)
    @test result == sparse([3,4,4], [3,3,4], fill(true,3), 5, 5)
    @test typeof(result) == SparseMatrixCSC{Bool,Int}

    result = broadcast(max, A13024, B13024)
    @test result == sparse([1,2,3,4,4,5], [1,2,3,3,4,5], fill(true,6))
    @test typeof(result) == SparseMatrixCSC{Bool,Int}

    result = broadcast(min, A13024, B13024)
    @test result == sparse([1,2,5], [1,2,5], fill(true,3))
    @test typeof(result) == SparseMatrixCSC{Bool,Int}

    for op in (+, -)
        @test op(A13024, B13024) == op(Array(A13024), Array(B13024))
    end
end

@testset "issue described in https://groups.google.com/forum/#!topic/julia-dev/QT7qpIpgOaA" begin
    @test sparse([1,1], [1,1], [true, true]) == sparse([1,1], [1,1], [true, true], 1, 1) == fill(true, 1, 1)
end

@testset "issparse for sparse vectors #34253" begin
    v = fixturevec(Float64, 10)
    @test issparse(v)
    @test issparse(v')
    @test issparse(transpose(v))
end

@testset "issue #16073" begin
    Random.seed!(16073)
    @inferred sprand(1, 1, 1.0)
    @inferred sprand(1, 1, 1.0, rand, Float64)
    @inferred sprand(1, 1, 1.0, x -> round.(Int, rand(x) * 100))
end

# Test temporary fix for issue #16548 in PR #16979. Somewhat brittle. Expect to remove with `\` revisions.
@testset "issue #16548" begin
    ms = methods(\, (SparseMatrixCSC, AbstractVecOrMat)).ms
    @test all(m -> m.module == SparseArrays, ms)
end

# Check that `broadcast` methods specialized for unary operations over `SparseMatrixCSC`s
# are called. (Issue #18705.) EDIT: #19239 unified broadcast over a single sparse matrix,
# eliminating the former operation classes.
@testset "issue #18705" begin
    S = sparse(Diagonal(1.0:5.0))
    @test isa(sin.(S), SparseMatrixCSC)
end

@testset "issue #19225" begin
    X = sparse([1 -1; -1 1])
    for T in (Symmetric, Hermitian)
        Y = T(copy(X))
        _Y = similar(Y)
        copyto!(_Y, Y)
        @test _Y == Y

        W = T(copy(X), :L)
        copyto!(W, Y)
        @test W.data == Y.data
        @test W.uplo != Y.uplo

        W[1,1] = 4
        @test W == T(sparse([4 -1; -1 1]))
        @test_throws ArgumentError (W[1,2] = 2)

        @test Y + I == T(sparse([2 -1; -1 2]))
        @test Y - I == T(sparse([0 -1; -1 0]))
        @test Y * I == Y

        @test Y .+ 1 == T(sparse([2 0; 0 2]))
        @test Y .- 1 == T(sparse([0 -2; -2 0]))
        @test Y * 2 == T(sparse([2 -2; -2 2]))
        @test Y / 1 == Y
    end
end

# Check that `broadcast` methods specialized for unary operations over
# `SparseMatrixCSC`s determine a reasonable return type.
@testset "issue #18974" begin
    S = sparse(Diagonal(Int64(1):Int64(4)))
    @test eltype(sin.(S)) == Float64
end

# Check calling of unary minus method specialized for SparseMatrixCSCs
@testset "issue #19503" begin
    @test which(-, (SparseMatrixCSC,)).module == SparseArrays
end

@testset "issue #14398" begin
    @test collect(view(sparse(I, 10, 10), 1:5, 1:5)') ≈ Matrix(I, 5, 5)
end

@testset "dropstored issue #20513" begin
    x = sparse(reshape(Float64.(1:9), 3, 3))
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

@testset "issue #29644" begin
    F = lu(Tridiagonal(sparse(1.0I, 3, 3)))
    @test F.L == Matrix(I, 3, 3)
    @test startswith(sprint(show, MIME("text/plain"), F),
                     "$(LinearAlgebra.LU){Float64, $(LinearAlgebra.Tridiagonal){Float64, $(SparseArrays.SparseVector)")
end

@testset "reverse search direction if step < 0 #21986" begin
    local A, B
    A = fixture(Float64, 5, 5)
    A = max.(A, copy(A'))
    LinearAlgebra.fillstored!(A, 1)
    B = A[5:-1:1, 5:-1:1]
    @test issymmetric(B)
end

@testset "Issue #28369" begin
    M = reshape([[1 2; 3 4], [9 10; 11 12], [5 6; 7 8], [13 14; 15 16]], (2,2))
    MP = reshape([[1 2; 3 4], [5 6; 7 8], [9 10; 11 12], [13 14; 15 16]], (2,2))
    S = sparse(M)
    SP = sparse(MP)
    @test isa(transpose(S), Transpose)
    @test transpose(S) == copy(transpose(S))
    @test Array(transpose(S)) == copy(transpose(M))
    @test permutedims(S) == SP
    @test permutedims(S, (2,1)) == SP
    MC = reshape([[(1+im) 2; 3 4], [9 10; 11 12], [(5 + 2im) 6; 7 8], [13 14; 15 16]], (2,2))
    SC = sparse(MC)
    @test isa(adjoint(SC), Adjoint)
    @test adjoint(SC) == copy(adjoint(SC))
    @test adjoint(MC) == copy(adjoint(SC))
end

@testset "Issue #28634" begin
    a = SparseMatrixCSC{Int8, Int16}([1 2; 3 4])
    na = SparseMatrixCSC(a)
    @test typeof(a) === typeof(na)
end

#PR #29045
@testset "Issue #28934" begin
    A = fixture(Float64, 5, 5)
    D = Diagonal(Float64.(1:5))
    C = copy(A)
    m1 = which(mul!, Base.typesof(C,A,D,true,false))
    m2 = which(mul!, Base.typesof(C,D,A,true,false))
    @test m1.module == SparseArrays
    @test m2.module == SparseArrays
end

@testset "issue #31453" for T in (UInt8, Int8, Int32)
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

@testset "Ti cannot store all potential values #31024" begin
    # m * n >= typemax(Ti) but nnz < typemax(Ti)
    A = SparseMatrixCSC(12, 12, fill(Int8(1),13), Int8[], Int[])
    @test size(A) == (12,12) && nnz(A) == 0
    I1 = [Int8(i) for i in 1:20 for _ in 1:20]
    J1 = [Int8(i) for _ in 1:20 for i in 1:20]
    # m * n >= typemax(Ti) and nnz >= typemax(Ti)
    @test_throws ArgumentError sparse(I1, J1, ones(length(I1)))
    I1 = Int8.(mod1.(1:500, 10))
    J1 = Int8.(mod1.(1:500, 7))
    V1 = ones(500)
    # m * n < typemax(Ti) and length(I) >= typemax(Ti) - combining values
    @test_throws ArgumentError sparse(I1, J1, V1, 10, 10)
    # m * n >= typemax(Ti) and length(I) >= typemax(Ti)
    @test_throws ArgumentError sparse(I1, J1, V1, 12, 13)
    I1 = Int8.(mod1.(1:126, 10))
    J1 = Int8.(mod1.(1:126, 7))
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

@testset "matrix construction does not alias fields, $T (issue #34630)" for T in
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

@testset "vector construction does not alias fields, $T (issue #34630)" for T in
    (SparseVector, SparseVector{Float64}, SparseVector{Float64,Int16})

    A = sparse([1; 0])
    B = T(A)
    @test A == B
    A[2] = 1
    @test A != B
    @test nonzeroinds(A) !== nonzeroinds(B)
    @test nonzeros(A) !== nonzeros(B)
end

@testset "Symmetric and Hermitian #35325" begin
    A = fixture(ComplexF64, 10, 10)
    B = permutedims(fixture(ComplexF64, 10, 10)) + sparse([9, 2], [2, 9], [3.0 - 2.0im, 1.0 + 4.0im], 10, 10)

    @testset "$Wrapper $op" for op ∈ (+, -), Wrapper ∈ (Hermitian, Symmetric)
        AWU = Wrapper(A, :U)
        AWL = Wrapper(A, :L)
        BWU = Wrapper(B, :U)
        BWL = Wrapper(B, :L)

        left = op(AWU, B)
        @test left isa SparseMatrixCSC
        right = op(A, BWL)
        @test right isa SparseMatrixCSC

        @test left ≈ op(collect(AWU), B)
        @test op(AWL, B) ≈ op(collect(AWL), B)
        @test op(A, BWU) ≈ op(A, collect(BWU))
        @test right ≈ op(A, collect(BWL))

        mixed = op(AWU, BWL)
        @test mixed isa Wrapper{ComplexF64, <:SparseMatrixCSC}

        @test op(AWU, BWU) ≈ op(collect(AWU), collect(BWU))
        @test mixed ≈ op(collect(AWU), collect(BWL))
        @test op(AWL, BWU) ≈ op(collect(AWL), collect(BWU))
        @test op(AWL, BWL) ≈ op(collect(AWL), collect(BWL))
    end
end

@testset "issue #41135" begin
    @test repr(SparseMatrixCSC([7;;])) == "sparse([1], [1], [7], 1, 1)"

    m = SparseMatrixCSC([0 3; 4 0])
    @test repr(m) == "sparse([2, 1], [1, 2], [4, 3], 2, 2)"
    @test eval(Meta.parse(repr(m))) == m
    @test summary(m) == "2×2 $SparseMatrixCSC{$Int, $Int} with 2 stored entries"

    m = fixture(Float64, 20, 30)
    @test occursin(r"^sparse\(\[.+\], \[.+\], \[.+\], \d+, \d+\)$", repr(m))
    @test eval(Meta.parse(repr(m))) == m

    m = sparse([85, 5, 38, 37, 59], [19, 72, 76, 98, 162], [0.8, 0.3, 0.2, 0.1, 0.5], 100, 200)
    @test repr(m) == "sparse([85, 5, 38, 37, 59], [19, 72, 76, 98, 162], [0.8, 0.3, 0.2, 0.1, 0.5], 100, 200)"
    @test eval(Meta.parse(repr(m))) == m
end

@testset "Issue #29" begin
    s = fixture(Float64, 5, 3)
    li = LinearIndices(s)
    ci = CartesianIndices(s)
    @test s[li] == s[ci] == s[Matrix(li)] == s[Matrix(ci)]
end

# #25943
@testset "operations on Integer subtypes" begin
    s = sparse(UInt8[1, 2, 3], UInt8[1, 2, 3], UInt8[1, 2, 3])
    @test sum(s, dims=2) == reshape([1, 2, 3], 3, 1)
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

@testset "uniform scaling should not change type #103" begin
    A = spzeros(Float32, Int8, 5, 5)
    B = I - A
    @test typeof(B) == typeof(A)
end

@testset "issue #731" begin
    x = MockTropical{Float64}(1.0)
    B = sparse([1, 2], [1,2], [x, x])
    C = [one(x) zero(x); zero(x) one(x)]
    @test one(B) == C
    @test B^0 == C
end

@testset "Issue #190" begin
    J = move_fixed(sparse(Diagonal(ones(4))))
    W = move_fixed(sparse(Diagonal(ones(4))))
    J[4, 4] = 0
    gamma = 1.0
    W .= gamma .* J
    @test W == J

    x = move_fixed(fixture(Float64, 5, 3))
    @test (x .= x .* 0; true)
    @test (x .= 0; true)
    @test (fill!(x, false); true)
end

@testset "Issue #27836" begin
    @test minimum(sparse([1, 2], [1, 2], ones(Int32, 2)), dims = 1) isa Matrix
end

@testset "issue #31758: out of bounds write in _map_zeropres!" begin
    y = sparsevec([2,7], [1., 2.], 10)
    x1 = sparsevec(fill(1.0, 10))
    x2 = sparsevec([2,7], [1., 2.], 10)
    x3 = sparsevec(fill(1.0, 10))
    f(x, y, z) = x == y == z == 0 ? 0.0 : NaN
    y .= f.(x1, x2, x3)
    @test all(isnan, y)
end

@testset "issue #42670 - error in sparsevec outer product" begin
    A = spzeros(Int, 4)
    B = copy(A)
    C = sparsevec([0 0 1 1 0 0])'
    A[2] = 1
    A[2] = 0
    @test A * C == B * C == spzeros(Int, 4, 6)
end

@testset "issue #46337 - error in sparsevec map" begin
    x = sparsevec([1], [1//1+0im])
    @test inv.(x) == [1//1+0im]
    y = spzeros(Int, 1)
    @test y ./ x == y
end

@testset "Issue 14013" begin
    s14013 = sparse([10.0 0.0 30.0; 0.0 1.0 0.0])
    a14013 = [10.0 0.0 30.0; 0.0 1.0 0.0]
    @test s14013 == a14013
    @test vec(s14013) == s14013[:] == a14013[:]
    @test Array(s14013)[1,:] == s14013[1,:] == a14013[1,:] == [10.0, 0.0, 30.0]
    @test Array(s14013)[2,:] == s14013[2,:] == a14013[2,:] == [0.0, 1.0, 0.0]
end

@testset "Issue 14046" begin
    s14046 = sparse([1.0, 2.5, -3.0, 4.0, 0.5])
    @test spzeros(5) + s14046 == s14046
    @test 2*s14046 == s14046 + s14046
end

@testset "13130 and 16661" begin
    xd = collect(1.0:10.0)
    @test issparse([fixture(Float64, 10, 3) fixturevec(Float64, 10)])
    @test issparse([fixture(Float64, 10, 1); fixturevec(Float64, 10)])

    @test issparse([fixture(Float64, 10, 3) xd])
    @test issparse([fixture(Float64, 10, 1)  xd])
    @test issparse([fixture(Float64, 10, 2) fixture(Float64, 10, 1) xd])
    @test issparse([fixture(Float64, 10, 1); xd])

    @test issparse([fixturevec(Float64, 10)  xd])
    @test issparse([fixturevec(Float64, 10); xd])
end

@testset "corner cases of broadcast arithmetic operations with scalars (#21515)" begin
    # test both scalar literals and variables
    areequal(a, b, c) = isequal(a, b) && isequal(b, c)
    inf, zeroh, zv, spzv = Inf, 0.0, zeros(1), spzeros(1)
    @test areequal(spzv .* Inf,  spzv .* inf,    sparsevec(zv .* Inf))
    @test areequal(Inf .* spzv,  inf .* spzv,    sparsevec(Inf .* zv))
    @test areequal(spzv ./ 0.0,  spzv ./ zeroh,  sparsevec(zv ./ 0.0))
    @test areequal(0.0 .\ spzv,  zeroh .\ spzv,  sparsevec(0.0 .\ zv))
end

@testset "issue #605" begin
    S = sparse([2, 3, 1], [1, 1, 3], [1, 1, 1], 3, 3)
    @test !issymmetric(S)
end

@testset "issue #748" begin
    for S in [
        sparse([3,3,4,1,2], [1,2,2,3,4], ones(Int,5), 4, 4),
        sparse([1,3,3,4,1,2], [1,1,2,2,3,4], ones(Int,6), 4, 4),
        sparse([2,3,1,3,4,1,2], [1,1,2,2,2,3,4], ones(Int,7), 4, 4),
        sparse([1,2,3,1,3,4,1,2], [1,1,1,2,2,2,3,4], ones(Int,8), 4, 4),
        sparse([3,2,3,4,1,2], [1,2,2,2,3,4], ones(Int,6), 4, 4),
        sparse([1,3,2,3,4,1,2], [1,1,2,2,2,3,4], ones(Int,7), 4, 4),
        sparse([2,3,1,2,3,4,1,2], [1,1,2,2,2,2,3,4], ones(Int,8), 4, 4),
        sparse([1,2,3,1,2,3,4,1,2], [1,1,1,2,2,2,2,3,4], ones(Int,9), 4, 4),
        sparse([3,3,4,1,2,4], [1,2,2,3,4,4], ones(Int,6), 4, 4),
        sparse([1,3,3,4,1,2,4], [1,1,2,2,3,4,4], ones(Int,7), 4, 4),
        sparse([2,3,1,3,4,1,2,4], [1,1,2,2,2,3,4,4], ones(Int,8), 4, 4),
        sparse([1,2,3,1,3,4,1,2,4], [1,1,1,2,2,2,3,4,4], ones(Int,9), 4, 4),
        sparse([3,2,3,4,1,2,4], [1,2,2,2,3,4,4], ones(Int,7), 4, 4),
        sparse([1,3,2,3,4,1,2,4], [1,1,2,2,2,3,4,4], ones(Int,8), 4, 4),
        sparse([2,3,1,2,3,4,1,2,4], [1,1,2,2,2,2,3,4,4], ones(Int,9), 4, 4),
        sparse([1,2,3,1,2,3,4,1,2,4], [1,1,1,2,2,2,2,3,4,4], ones(Int,10), 4, 4),
        SparseMatrixCSC(6, 6, [1,1,2,3,4,5,6], [4,4,3,6,1], [0,1,1,1,0]),
    ]
        @test !issymmetric(S)
    end
end

@testset "Adding sparse-backed SymTridiagonal (#46355)" begin
    a = SymTridiagonal(sparsevec(Int[1]), sparsevec(Int[]))
    @test a + a == Matrix(a) + Matrix(a)

    # symtridiagonal with non-empty off-diagonal
    b = SymTridiagonal(sparsevec(Int[1, 2, 3]), sparsevec(Int[1, 2]))
    @test b + b == Matrix(b) + Matrix(b)
end

@testset "Issue #246" begin
    a = OpCount.(fixturevec(Float64, 100))
    b = OpCount.(2 * fixturevec(Float64, 100))
    c = copy(a)
    nonzeros(c)[1] = 0
    d = dropzeros(c)

    for m in (identity, transpose)
        ma, mb, mc, md = m.((a, b, c, d))
        @test eqcount(() -> ma == mb) <= nnz(a) + nnz(b)
        @test (mc == md) == (Array(mc) == Array(md))
    end
end

@testset "sparse right multiplication of Symmetric and Hermitian matrices #21431" begin
    S = sparse(1.0I, 2, 2)
    @test issparse(S*S*S)
    for T in (Symmetric, Hermitian)
        @test issparse(S*T(S)*S)
        @test issparse(S*(T(S)*S))
        @test issparse((S*T(S))*S)
    end
end

@testset "vertical concatenation of SparseVectors with different el- and ind-type (#22225)" begin
    spv6464 = SparseVector(0, Int64[], Int64[])
    @test isa(vcat(spv6464, SparseVector(0, Int32[], Int32[])), SparseVector{Int64,Int64})
end

@testset "horizontal concatenation of SparseVectors with different el- and ind-type (#22225)" begin
    spv6464 = SparseVector(0, Int64[], Int64[])
    @test isa(hcat(spv6464, SparseVector(0, Int32[], Int32[])), SparseMatrixCSC{Int64,Int64})
end

# An array type from another package that owns the `vcat`/`hcat`/`hvcat` of its own
# arrays with anything; those methods must not become ambiguous when SparseArrays is
# loaded (#431)
@testset "no ambiguities with concatenation methods of other array types (#431)" begin
    A = ConcatArray([1 2; 3 4])
    v = ConcatArray([1, 2])
    S = sparse([1 0; 0 1])
    for x in (A, [5 6; 7 8], S)
        @test vcat(x, A)::ConcatArray == vcat(Array(x), A.data)
        @test hcat(x, A)::ConcatArray == hcat(Array(x), A.data)
        @test hvcat((2,), x, A)::ConcatArray == hvcat((2,), Array(x), A.data)
    end
    for x in (v, [5, 6], sparse([1, 0]))
        @test vcat(x, v)::ConcatArray == vcat(Array(x), v.data)
        @test hcat(x, v)::ConcatArray == hcat(Array(x), v.data)
    end
    # with the sparse array first, the generic fallback still yields a sparse result
    @test vcat(A, S)::SparseMatrixCSC == vcat(A.data, Array(S))
    @test hcat(A, S)::SparseMatrixCSC == hcat(A.data, Array(S))
    @test hvcat((2,), A, S)::SparseMatrixCSC == hvcat((2,), A.data, Array(S))
    @test vcat(A, [5 6; 7 8])::Matrix == vcat(A.data, [5 6; 7 8])
    @test vcat(v, sparse([1, 0]))::SparseVector == vcat(v.data, [1, 0])
end

@testset "issue #14816" begin
    m = 5
    intmat = fill(1, m, m)
    ltintmat = LowerTriangular([mod1(i + 2j, 5) for i in 1:m, j in 1:m])
    @test \(transpose(ltintmat), sparse(intmat)) ≈ \(transpose(ltintmat), intmat)
end

@testset "#16716" begin
    origmat = [-1.5 -0.7; 0.0 1.0]
    transmat = copy(origmat')
    utmat = UpperTriangular(origmat)
    ultmat = LinearAlgebra.UnitLowerTriangular(transmat)

    zerospvec = spzeros(Float64, 2)
    zerodvec = zeros(Float64, 2)

    for mat in (utmat, ultmat)
        @test isequal(\(mat, zerospvec), zerodvec)
        @test isequal(\(adjoint(mat), zerospvec), zerodvec)
        @test isequal(\(transpose(mat), zerospvec), zerodvec)
        @test isequal(ldiv!(mat, copy(zerospvec)), zerospvec)
        @test isequal(ldiv!(adjoint(mat), copy(zerospvec)), zerospvec)
        @test isequal(ldiv!(transpose(mat), copy(zerospvec)), zerospvec)
    end
end

@testset "Julia #37819: map preserves indices when length exceeds typemax(Ti)" begin
    Z = spzeros(Float64, Int32, 50000, 50000)
    @test widemul(size(Z)...) > typemax(Int32)
    @test isa(-Z, SparseMatrixCSC{Float64, Int32})
    @test isa(Z + Z, SparseMatrixCSC{Float64, Int32})
end

@testset "Julia #24534: indexing a sparse vector with UInt32 indices" begin
    # issue 24534
    let x = convert(SparseVector{Float64,UInt32},fixturevec(Float64, 100))
        I = [mod1(37k, 99) + 1 for k in [1:23; 1]]
        r = x[I]
        @test isa(r, SparseVector{Float64,UInt32})
        @test all(!iszero, nonzeros(r))
        @test Array(r) == Array(x)[I]
    end

    # issue 24534
    let x = convert(SparseVector{Float64,UInt32},fixturevec(Float64, 100))
        I = reshape([mod1(37k, 99) + 1 for k in [1:19; 1]], 20, 1)
        r = x[I]
        @test isa(r, SparseMatrixCSC{Float64,UInt32})
        @test all(!iszero, nonzeros(r))
        @test Array(r) == Array(x)[I]
    end
end

@testset "heap-allocated zero in sparse vectors (#389)" begin
    for T in (BigFloat,)
        x = spzeros(T, 4)
        x[1] = 0
        x[3] = zero(T)
        @test nnz(x) == 0
        x[2] = one(T)
        x[2] = 0
        @test nnz(x) == 1 && x[2] == 0
    end
    x = spzeros(BigFloat, 4)
    x[1] = -big(0.0)
    @test nnz(x) == 1 && signbit(x[1])
    x = spzeros(Union{Missing,Int}, 4)
    x[1] = missing
    @test nnz(x) == 1 && x[1] === missing
    A = sparse([1, 2], [1, 1], BigFloat[1, 2], 3, 2)
    copyto!(view(A, :, 1), BigFloat[0, 3, 0])   # as `setindex!` would store
    @test nnz(A) == 1 && A[2, 1] == 3
    copyto!(view(A, :, 1), [-big(0.0), big(3), big(0)])
    @test nnz(A) == 2 && signbit(A[1, 1])
end

@testset "heap-allocated zero in sparse matrices (#389)" begin
    for T in (BigFloat,)
        A = spzeros(T, 3, 3)
        A[1, 1] = 0
        A[2, 3] = zero(T)
        @test nnz(A) == 0
        A[2, 2] = one(T)
        A[2, 2] = 0         # an existing entry is overwritten in place
        @test nnz(A) == 1 && A[2, 2] == 0
    end
    A = spzeros(BigFloat, 2, 2)
    A[1, 1] = -big(0.0)     # the sign of a zero is not reconstructible
    @test nnz(A) == 1 && signbit(A[1, 1])
    A[2, 2] = -0.0
    @test nnz(A) == 2 && signbit(A[2, 2])
    M = Matrix(Diagonal(BigFloat[1, 2, 3, 4]))
    S = spzeros(BigFloat, 4, 4)
    S .= M                  # the generic broadcast assigns every element
    @test nnz(S) == 4 && S == M
end

@testset "Julia #44978: argmin and argmax with implicit zeros" begin
    let v = spzeros(3) #Julia #44978
        v[1] = 2
        @test argmin(v) == 2
        @test argmax(v) == 1
        v[2] = 2
        @test argmin(v) == 3
        v[1] = 0
        v[2] = 0
        v[3] = 2
        @test argmin(v) == 1
        @test argmax(v) == 3
    end

    let v = spzeros(3) #Julia #44978
        v[3] = 2
        @test argmax(v) == 3
    end
end

@testset "issue #714: findmin and findmax with NaN" begin
    let v = sparse([0, NaN]) #issue #714
        @test findmin(v) === (NaN, 2)
        @test findmax(v) === (NaN, 2)
    end
end

@testset "Julia #5169 and #5437: stored zeros (#15242/#14798)" begin
    # test for issue #5169, modified for new behavior following #15242/#14798
    @test nnz(sparse([1, 1], [1, 2], [0.0, -0.0])) == 2
    @test nnz(dropzeros!(sparse([1, 1], [1, 2], [0.0, -0.0]))) == 0
    # test for issue #5437, modified for new behavior following #15242/#14798
    @test nnz(sparse([1, 2, 3], [1, 2, 3], [0.0, 1.0, 2.0])) == 3
    @test nnz(dropzeros!(sparse([1, 2, 3],[1, 2, 3],[0.0, 1.0, 2.0]))) == 2
end

@testset "Julia #32568: search accepts non-Int start indices" begin
    x_sp = sparse([0,0,0,0,1,0,1,0,1,1,0])
    z_sp = sparsevec(Dict(1=>1, 5=>1, 8=>0, 10=>1))
    # issue 32568
    for T = (UInt, BigInt)
        @test findnext(!iszero, x_sp, T(4)) isa keytype(x_sp)
        @test findnext(!iszero, x_sp, T(5)) isa keytype(x_sp)
        @test findprev(!iszero, x_sp, T(5)) isa keytype(x_sp)
        @test findprev(!iszero, x_sp, T(6)) isa keytype(x_sp)
        @test findnext(iseven, x_sp, T(4)) isa keytype(x_sp)
        @test findnext(iseven, x_sp, T(5)) isa keytype(x_sp)
        @test findprev(iseven, x_sp, T(4)) isa keytype(x_sp)
        @test findprev(iseven, x_sp, T(5)) isa keytype(x_sp)
        @test findnext(!iszero, z_sp, T(4)) isa keytype(z_sp)
        @test findnext(!iszero, z_sp, T(5)) isa keytype(z_sp)
        @test findprev(!iszero, z_sp, T(4)) isa keytype(z_sp)
        @test findprev(!iszero, z_sp, T(5)) isa keytype(z_sp)
    end
end

@testset "Julia #30589: show Boolean sparse arrays" begin
    # Issue #30589
    @test sprint(show, "text/plain", sparse([true true]); context=:limit=>true) == "1×2 $SparseMatrixCSC{Bool, $Int} with 2 stored entries:\n 1  1"
    # issue #30589
    @test repr("text/plain", sparse([true])) == "1-element $(SparseArrays.SparseVector){Bool, $Int} with 1 stored entry:\n  [1]  =  1"
end

@testset "Julia #19304: hcat inference with identity matrices" begin
    @inferred hcat(fixture(Float64, 2, 1), 1.0I)
    @inferred hcat(fixture(Float64, 2, 1), Matrix(I, 2, 2))
    @inferred hcat(fixture(Float64, 2, 1), Matrix(1.0I, 2, 2))
end

@testset "a matrix of 2^64 entries does not overflow its length" begin
    B = spzeros(2^62, 4)
    @test any(iszero, B) && !all(isone, B)
    @test sum(x -> 1.0, B) == 2.0^64 && prod(x -> 2.0, B) == Inf
    @test count(iszero, B; init = big(0)) == big(2)^64
    @test findmax(B) == (0.0, CartesianIndex(1, 1))
    @test mapreduce(*, +, B, B) == 0.0
    @test_throws OverflowError B[[1]]
    @test_throws OverflowError B[[1]] = [1.0]
    @test_throws OverflowError B[1] = 1.0
    @test nnz(B) == 0
    @test_throws OverflowError sprand(Xoshiro(1), 2^62, 4, 1e-17)
end

if isdefined(Docs, :undocumented_names) # new in Julia 1.11
    @testset "docstrings (issue julia#52725)" begin
        @test isempty(Docs.undocumented_names(SparseArrays))
    end
end

# As part of the migration of SparseArrays.jl into its own repo,
# these tests have been moved from other files in julia tests to
# the SparseArrays.jl repo

module SparseTestsBase

using Test
using LinearAlgebra, SparseArrays
include("testhelpers.jl")

# From arrayops.jl

@testset "copy!" begin
    @testset "AbstractVector" begin
        s = Vector([1, 2])
        for a = ([1], UInt[1], [3, 4, 5], UInt[3, 4, 5])
            @test s === copy!(s, SparseVector(a)) == Vector(a)
        end
    end
end

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

# From abstractarray.jl

@testset "julia #17088" begin
    n = 10
    M = reshape(Float64.(1:n*n), n, n)
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

@testset "mapslices julia #21123" begin
    @test mapslices(nnz, sparse(1.0I, 3, 3), dims=1) == [1 1 1]
end

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

# From core.jl

# issue #12960
let
    A = sparse(1.0I, 3, 3)
    B = similar(A, UndefElt)
    @test repr(B) == "sparse([1, 2, 3], [1, 2, 3], $UndefElt[#undef, #undef, #undef], 3, 3)"
    @test occursin(
        " #undef     ⋅       ⋅\n    ⋅    #undef     ⋅\n    ⋅       ⋅    #undef",
        sprint(show, MIME("text/plain"), B; context=:limit=>true),
    )

    B[1,2] = UndefElt()
    @test repr(B)  == "sparse([1, 1, 2, 3], [1, 2, 2, 3], $UndefElt[#undef, $UndefElt(), #undef, #undef], 3, 3)"
    @test occursin(
        "\n #undef     UndefElt()     ⋅\n    ⋅    #undef            ⋅\n    ⋅          ⋅        #undef",
        sprint(show, MIME("text/plain"), B; context=:limit=>true),
    )
end

# julia issue #12063
# NOTE: should have > MAX_TUPLETYPE_LEN arguments
f12063(tt, g, p, c, b, v, cu::T, d::AbstractArray{T, 2}, ve) where {T} = 1
f12063(args...) = 2
g12063() = f12063(0, 0, 0, 0, 0, 0, 0.0, spzeros(0,0), Int[])
@test g12063() == 1

@testset "Issue #334" begin
    x = sparsevec([1, 2, 4, 7, 8], [3.0, -1.0, 0.0, 2.0, -4.0], 10);
    @test issorted(sort!(x; alg=Base.DEFAULT_STABLE));
    @test_throws MethodError sort!(x; banana=:blue); # From discussion at #335
end

@testset "Issue #512" begin
    # `sparsematrix.jl` checks the warning for the matrix itself
    x = sparse([1, 100, 3], [1, 4, 300], [20, 0, [2]])
    @test_warn "WARNING: could not find generic zero" begin
        @test contains(repr(MIME("text/plain"), transpose(x')), "⎡")
    end
    x = sparse([1, 3], [1, 2], Any[20, [2]])
    @test_warn "WARNING: could not find generic zero" begin
        @test contains(repr(MIME("text/plain"), transpose(x'); context=:limit=>true), "20   ⋅\n")
    end
end

@testset "Issue #574" begin
    a = spzeros(Float32, Int16, 2, 3)
    v = spzeros(Float32, Int16, 2)
    @test eltype(rowvals(zero(a))) <: Int16
    @test eltype(rowvals(zero(v))) <: Int16
end

@testset "Issue #618" begin
    # `sparsematrix.jl` runs the `Float64` case
    for v in (['a', 7.0, 1.0, 1.0], [similar(Any[1]); 1.0; 1.0; 1.0])
        # a repeated entry is invalid, so it is marked rather than shown
        x = SparseMatrixCSC(3, 3, [1, 3, 4, 5], [1, 1, 2, 3], v)
        @test_warn "WARNING: array contains duplicate entries" begin
            @test contains(sprint(show, MIME"text/plain"(), x; context=:limit=>true), "‼")
        end
    end
end

@testset "scaling by NaN reaches the entries that are not stored" begin
    A, Ad = fixturepair(Float64, 5, 3)
    x, xd = fixturepair(Float64, 7)
    @test mismatch(rmul!(copy(A), NaN), Ad * NaN) === nothing
    @test mismatch(lmul!(NaN, copy(A)), NaN * Ad) === nothing
    @test mismatch(mul!(similar(A), A, NaN), Ad * NaN) === nothing
    @test mismatch(mul!(similar(A), NaN, A), NaN * Ad) === nothing
    @test mismatch(rmul!(copy(x), NaN), xd * NaN) === nothing
    @test mismatch(lmul!(NaN, copy(x)), NaN * xd) === nothing
    @test mismatch(x * NaN, xd * NaN) === nothing
    @test mismatch(NaN * x, NaN * xd) === nothing
end

@testset "concatenation with a dense array keeps the stored zeros of a sparse one" begin
    A = sparse([1, 2], [1, 2], [1.0, 0.0])  # a stored zero
    D = [1.0 2.0; 0.0 4.0]
    Ad = Matrix(A)
    for (S, R) in ((hcat(A, D), hcat(Ad, D)), (vcat(A, D), vcat(Ad, D)),
                   (cat(A, D; dims=(1, 2)), cat(Ad, D; dims=(1, 2))),
                   (cat(A, 3.0; dims=(1, 2)), cat(Ad, 3.0; dims=(1, 2))),
                   (hcat(A, I), hcat(Ad, Matrix(I, 2, 2))))
        @test mismatch(S, R; Ti=Int) === nothing
        @test count(iszero, nonzeros(S)) == 1
    end
    # blocks of different types, as `I` gives, are not boxed once per column
    allocs(n) = (S = sparse(1.0I, n, n); hcat(S, I); @allocations(hcat(S, I)))
    @test allocs(1000) == allocs(2000)
end

end # SparseTestsBase

end # module
