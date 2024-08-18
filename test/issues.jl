module SparseIssuesTests
using Test
using SparseArrays
using SparseArrays: nonzeroinds, getcolptr
using LinearAlgebra
using Random
using Test: guardseed
using InteractiveUtils: @which
include("forbidproperties.jl")
include("simplesmatrix.jl")


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

@testset "Issue #30006" begin
    A = SparseMatrixCSC{Float64,Int32}(spzeros(3,3))
    A[:, 1] = [1, 2, 3]
    @test nnz(A) == 3
    @test nonzeros(A) == [1, 2, 3]
end

@testset "Issue #28963" begin
    @test_throws DimensionMismatch (spzeros(10,10)[:, :] = sprand(10,20,0.5))
end

@testset "Issue #30502" begin
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
    @test sprand(4,5,0.5).^0 == sparse(fill(1,4,5))
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
    A = sprand(5,5,0.5,(n)->rand(Float64,n))
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

@testset "issue #12118: sparse matrices are closed under +, -, min, max" begin
    A12118 = sparse([1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5])
    B12118 = sparse([1,2,4,5],   [1,2,3,5],   [2,1,-1,-2])

    @test A12118 + B12118 == sparse([1,2,3,4,4,5], [1,2,3,3,4,5], [3,3,3,-1,4,3])
    @test typeof(A12118 + B12118) == SparseMatrixCSC{Int,Int}

    @test A12118 - B12118 == sparse([1,2,3,4,4,5], [1,2,3,3,4,5], [-1,1,3,1,4,7])
    @test typeof(A12118 - B12118) == SparseMatrixCSC{Int,Int}

    @test max.(A12118, B12118) == sparse([1,2,3,4,5], [1,2,3,4,5], [2,2,3,4,5])
    @test typeof(max.(A12118, B12118)) == SparseMatrixCSC{Int,Int}

    @test min.(A12118, B12118) == sparse([1,2,4,5], [1,2,3,5], [1,1,-1,-2])
    @test typeof(min.(A12118, B12118)) == SparseMatrixCSC{Int,Int}
end

@testset "issue #13008" begin
    @test_throws ArgumentError sparse(Vector(1:100), Vector(1:100), fill(5,100), 5, 5)
    @test_throws ArgumentError sparse(Int[], Vector(1:5), Vector(1:5))
end

@testset "issue #13024" begin
    A13024 = sparse([1,2,3,4,5], [1,2,3,4,5], fill(true,5))
    B13024 = sparse([1,2,4,5],   [1,2,3,5],   fill(true,4))

    @test broadcast(&, A13024, B13024) == sparse([1,2,5], [1,2,5], fill(true,3))
    @test typeof(broadcast(&, A13024, B13024)) == SparseMatrixCSC{Bool,Int}

    @test broadcast(|, A13024, B13024) == sparse([1,2,3,4,4,5], [1,2,3,3,4,5], fill(true,6))
    @test typeof(broadcast(|, A13024, B13024)) == SparseMatrixCSC{Bool,Int}

    @test broadcast(⊻, A13024, B13024) == sparse([3,4,4], [3,3,4], fill(true,3), 5, 5)
    @test typeof(broadcast(⊻, A13024, B13024)) == SparseMatrixCSC{Bool,Int}

    @test broadcast(max, A13024, B13024) == sparse([1,2,3,4,4,5], [1,2,3,3,4,5], fill(true,6))
    @test typeof(broadcast(max, A13024, B13024)) == SparseMatrixCSC{Bool,Int}

    @test broadcast(min, A13024, B13024) == sparse([1,2,5], [1,2,5], fill(true,3))
    @test typeof(broadcast(min, A13024, B13024)) == SparseMatrixCSC{Bool,Int}

    for op in (+, -)
        @test op(A13024, B13024) == op(Array(A13024), Array(B13024))
    end
    for op in (max, min, &, |, xor)
        @test op.(A13024, B13024) == op.(Array(A13024), Array(B13024))
    end
end

@testset "issue #13792, use sparse triangular solvers for sparse triangular solves" begin
    local A, n, x
    n = 100
    A, b = sprandn(n, n, 0.5) + sqrt(n)*I, fill(1., n)
    @test LowerTriangular(A)\(LowerTriangular(A)*b) ≈ b
    @test UpperTriangular(A)\(UpperTriangular(A)*b) ≈ b
    A[2,2] = 0
    dropzeros!(A)
    @test_throws LinearAlgebra.SingularException LowerTriangular(A)\b
    @test_throws LinearAlgebra.SingularException UpperTriangular(A)\b
end

@testset "issue described in https://groups.google.com/forum/#!topic/julia-dev/QT7qpIpgOaA" begin
    @test sparse([1,1], [1,1], [true, true]) == sparse([1,1], [1,1], [true, true], 1, 1) == fill(true, 1, 1)
    @test sparsevec([1,1], [true, true]) == sparsevec([1,1], [true, true], 1) == fill(true, 1)
end

@testset "issparse for sparse vectors #34253" begin
    v = sprand(10, 0.5)
    @test issparse(v)
    @test issparse(v')
    @test issparse(transpose(v))
end

@testset "issue #16073" begin
    @inferred sprand(1, 1, 1.0)
    @inferred sprand(1, 1, 1.0, rand, Float64)
    @inferred sprand(1, 1, 1.0, x -> round.(Int, rand(x) * 100))
end

@testset "issue #14816" begin
    m = 5
    intmat = fill(1, m, m)
    ltintmat = LowerTriangular(rand(1:5, m, m))
    @test \(transpose(ltintmat), sparse(intmat)) ≈ \(transpose(ltintmat), intmat)
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

@testset "issue #19304" begin
    @inferred hcat(sparse(rand(2,1)), I)
    @inferred hcat(sparse(rand(2,1)), 1.0I)
    @inferred hcat(sparse(rand(2,1)), Matrix(I, 2, 2))
    @inferred hcat(sparse(rand(2,1)), Matrix(1.0I, 2, 2))
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

@testset "issue #29644" begin
    F = lu(Tridiagonal(sparse(1.0I, 3, 3)))
    @test F.L == Matrix(I, 3, 3)
    @test startswith(sprint(show, MIME("text/plain"), F),
                     "$(LinearAlgebra.LU){Float64, $(LinearAlgebra.Tridiagonal){Float64, $(SparseArrays.SparseVector)")
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
    @test permutedims(S, (1,2)) == S
    @test permutedims(S, (1,2)) !== S
    @test_throws ArgumentError permutedims(S, (1,3))
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
    A = sprand(5,5,0.5)
    D = Diagonal(rand(5))
    C = copy(A)
    m1 = @which mul!(C,A,D)
    m2 = @which mul!(C,D,A)
    @test m1.module == SparseArrays
    @test m2.module == SparseArrays
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

@testset "Symmetric and Hermitian #35325" begin
    A = sprandn(ComplexF64, 10, 10, 0.1)
    B = sprandn(ComplexF64, 10, 10, 0.1)

    @test Symmetric(real(A)) + Hermitian(B) isa Hermitian{ComplexF64, <:SparseMatrixCSC}
    @test Hermitian(A) + Symmetric(real(B)) isa Hermitian{ComplexF64, <:SparseMatrixCSC}
    @test Hermitian(A) + Symmetric(B) isa SparseMatrixCSC
    @testset "$Wrapper $op" for op ∈ (+, -), Wrapper ∈ (Hermitian, Symmetric)
        AWU = Wrapper(A, :U)
        AWL = Wrapper(A, :L)
        BWU = Wrapper(B, :U)
        BWL = Wrapper(B, :L)

        @test op(AWU, B) isa SparseMatrixCSC
        @test op(A, BWL) isa SparseMatrixCSC

        @test op(AWU, B) ≈ op(collect(AWU), B)
        @test op(AWL, B) ≈ op(collect(AWL), B)
        @test op(A, BWU) ≈ op(A, collect(BWU))
        @test op(A, BWL) ≈ op(A, collect(BWL))

        @test op(AWU, BWL) isa Wrapper{ComplexF64, <:SparseMatrixCSC}

        @test op(AWU, BWU) ≈ op(collect(AWU), collect(BWU))
        @test op(AWU, BWL) ≈ op(collect(AWU), collect(BWL))
        @test op(AWL, BWU) ≈ op(collect(AWL), collect(BWU))
        @test op(AWL, BWL) ≈ op(collect(AWL), collect(BWL))
    end
end

@testset "Multiplying with triangular sparse matrices #35609 #35610" begin
    n = 10
    A = sprand(n, n, 5/n)
    U = UpperTriangular(A)
    L = LowerTriangular(A)
    AM = Matrix(A)
    UM = Matrix(U)
    LM = Matrix(L)
    Y = A * U
    @test Y ≈ AM * UM
    @test typeof(Y) == typeof(A)
    Y = A * L
    @test Y ≈ AM * LM
    @test typeof(Y) == typeof(A)
    Y = U * A
    @test Y ≈ UM * AM
    @test typeof(Y) == typeof(A)
    Y = L * A
    @test Y ≈ LM * AM
    @test typeof(Y) == typeof(A)
    Y = U * U
    @test Y ≈ UM * UM
    @test typeof(Y) == typeof(U)
    Y = L * L
    @test Y ≈ LM * LM
    @test typeof(Y) == typeof(L)
    Y = L * U
    @test Y ≈ LM * UM
    @test typeof(Y) == typeof(A)
    Y = U * L
    @test Y ≈ UM * LM
    @test typeof(Y) == typeof(A)
end

@testset "issue #41135" begin
    @test repr(SparseMatrixCSC([7;;])) == "sparse([1], [1], [7], 1, 1)"

    m = SparseMatrixCSC([0 3; 4 0])
    @test repr(m) == "sparse([2, 1], [1, 2], [4, 3], 2, 2)"
    @test eval(Meta.parse(repr(m))) == m
    @test summary(m) == "2×2 $SparseMatrixCSC{$Int, $Int} with 2 stored entries"

    m = sprand(100, 100, .1)
    @test occursin(r"^sparse\(\[.+\], \[.+\], \[.+\], \d+, \d+\)$", repr(m))
    @test eval(Meta.parse(repr(m))) == m

    m = sparse([85, 5, 38, 37, 59], [19, 72, 76, 98, 162], [0.8, 0.3, 0.2, 0.1, 0.5], 100, 200)
    @test repr(m) == "sparse([85, 5, 38, 37, 59], [19, 72, 76, 98, 162], [0.8, 0.3, 0.2, 0.1, 0.5], 100, 200)"
    @test eval(Meta.parse(repr(m))) == m
end

@testset "Issue #29" begin
    s = sprand(6, 6, .2)
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
using Random, LinearAlgebra, SparseArrays

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
mutable struct T12960 end
import Base.zero
Base.zero(::Type{T12960}) = T12960()
Base.zero(x::T12960) = T12960()
let
    A = sparse(1.0I, 3, 3)
    B = similar(A, T12960)
    @test repr(B) == "sparse([1, 2, 3], [1, 2, 3], $T12960[#undef, #undef, #undef], 3, 3)"
    @test occursin(
        "\n #undef             ⋅            ⋅    \n       ⋅      #undef             ⋅    \n       ⋅            ⋅      #undef",
        repr(MIME("text/plain"), B),
    )

    B[1,2] = T12960()
    @test repr(B)  == "sparse([1, 1, 2, 3], [1, 2, 2, 3], $T12960[#undef, $T12960(), #undef, #undef], 3, 3)"
    @test occursin(
        "\n #undef          T12960()        ⋅    \n       ⋅      #undef             ⋅    \n       ⋅            ⋅      #undef",
        repr(MIME("text/plain"), B),
    )
end

# julia issue #12063
# NOTE: should have > MAX_TUPLETYPE_LEN arguments
f12063(tt, g, p, c, b, v, cu::T, d::AbstractArray{T, 2}, ve) where {T} = 1
f12063(args...) = 2
g12063() = f12063(0, 0, 0, 0, 0, 0, 0.0, spzeros(0,0), Int[])
@test g12063() == 1

@testset "Issue #210" begin
    io = IOBuffer()
    show(io, sparse([1 2; 3 4]))
    @test String(take!(io)) == "sparse([1, 2, 1, 2], [1, 1, 2, 2], [1, 3, 2, 4], 2, 2)"
    io = IOBuffer()
    show(io, sparse([1 2; 3 4])')
    @test String(take!(io)) == "adjoint(sparse([1, 2, 1, 2], [1, 1, 2, 2], [1, 3, 2, 4], 2, 2))"
    io = IOBuffer()
    show(io, transpose(sparse([1 2; 3 4])))
    @test String(take!(io)) == "transpose(sparse([1, 2, 1, 2], [1, 1, 2, 2], [1, 3, 2, 4], 2, 2))"
end

@testset "Issue #334" begin
    x = sprand(10, .3);
    @test issorted(sort!(x; alg=Base.DEFAULT_STABLE));
    @test_throws MethodError sort!(x; banana=:blue); # From discussion at #335
end

@testset "Issue #390" begin
    x = sparse([9 1 8
                0 3 72
                7 4 16])
    Base.swapcols!(x, 2, 3)
    @test x == sparse([9 8 1
                       0 72 3
                       7 16 4])
end

end # SparseTestsBase

end # module
