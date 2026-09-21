# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseLinalgTests

using Test
using SparseArrays
using SparseArrays: AbstractSparseMatrixCSC, nonzeroinds, getcolptr, rowvals, nonzeros, fixed, _is_fixed
using LinearAlgebra
using Random
include("forbidproperties.jl")
include("mulcount.jl")

sA = sprandn(3, 7, 0.5)
sC = similar(sA)
dA = Array(sA)

const BASE_TEST_PATH = joinpath(Sys.BINDIR, "..", "share", "julia", "test")
isdefined(Main, :Quaternions) || @eval Main include(joinpath($(BASE_TEST_PATH), "testhelpers", "Quaternions.jl"))
using .Main.Quaternions

@testset "circshift" begin
    m,n = 17,15
    A = sprand(m, n, 0.5)
    for rshift in (-1, 0, 1, 10), cshift in (-1, 0, 1, 10)
        shifts = (rshift, cshift)
        # using dense circshift to compare
        B = circshift(Matrix(A), shifts)
        # sparse circshift
        C = circshift(A, shifts)
        @test C == B
        # sparse circshift should not add structural zeros
        @test nnz(C) == nnz(A)
        # test circshift!
        D = similar(A)
        circshift!(D, A, shifts)
        @test D == B
        @test nnz(D) == nnz(A)
        # test different in/out types
        A2 = floor.(100A)
        E1 = spzeros(Int64, m, n)
        E2 = spzeros(Int64, m, n)
        circshift!(E1, A2, shifts)
        circshift!(E2, Matrix(A2), shifts)
        @test E1 == E2
    end
end

@testset "wrappers of sparse" begin
    m = n = 10
    A = spzeros(ComplexF64, m, n)
    A[:,1] = 1:m
    A[:,2] = [1 3 0 0 0 0 0 0 0 0]'
    A[:,3] = [2 4 0 0 0 0 0 0 0 0]'
    A[:,4] = [0 0 0 0 5 3 0 0 0 0]'
    A[:,5] = [0 0 0 0 6 2 0 0 0 0]'
    A[:,6] = [0 0 0 0 7 4 0 0 0 0]'
    A[:,7:n] = rand(ComplexF64, m, n-6)
    B = Matrix(A)
    dowrap(wr, A) = wr(A)
    dowrap(wr::Tuple, A) = (wr[1])(A, wr[2:end]...)

    @testset "sparse($wr(A))" for wr in (
                        Symmetric, (Symmetric, :L), Hermitian, (Hermitian, :L),
                        Transpose, Adjoint,
                        UpperTriangular, LowerTriangular,
                        UnitUpperTriangular, UnitLowerTriangular,
                        (view, 3:6, 2:5))

        @test SparseMatrixCSC(dowrap(wr, A)) == Matrix(dowrap(wr, B))
    end

    @testset "sparse($at($wr))" for at = (Transpose, Adjoint), wr =
        (UpperTriangular, LowerTriangular,
         UnitUpperTriangular, UnitLowerTriangular)

        @test SparseMatrixCSC(at(wr(A))) == Matrix(at(wr(B)))
    end

    @test sparse([1,2,3,4,5]') == SparseMatrixCSC([1 2 3 4 5])
    @test sparse(UpperTriangular(A')) == UpperTriangular(B')
    @test sparse(Adjoint(UpperTriangular(A'))) == Adjoint(UpperTriangular(B'))
    @test sparse(UnitUpperTriangular(spzeros(5,5))) == I
    deepwrap(A) = (Adjoint(LowerTriangular(view(Symmetric(A), 5:7, 4:6))))
    @test sparse(deepwrap(A)) == Matrix(deepwrap(B))
end

@testset "destination array density in solves" begin
    O = diagm(-1 => fill(-1, 9), 0 => fill(2, 10), 1 => fill(-1, 9))
    wrappers = (a -> Bidiagonal(a, :U),
                a -> Bidiagonal(a, :L),
                SymTridiagonal,
                Tridiagonal,
                LowerTriangular,
                UnitLowerTriangular,
                UpperTriangular,
                UnitUpperTriangular,
                # UpperHessenberg,
                a -> UpperHessenberg(float(a))
                )
    for T in wrappers
        A = T(O)
        bs = sprandn(10, 0.3)
        bd = Array(bs)
        x = A \ bs
        @test x ≈ A \ bd
        @test !issparse(x)
        Bs = sprandn(10, 3, 0.2)
        Bd = Matrix(Bs)
        X = A \ Bs
        @test X ≈ A \ Bd
        @test !issparse(X)
        Cs = copy(Bs')
        Cd = Matrix(Cs)
        Y = Cs / A
        @test Y ≈ Cd / A
        @test !issparse(Y)
    end
    b, B = ones(Int, 10), ones(Int, 10, 10)
    for T in (UnitLowerTriangular, UnitUpperTriangular)
        A = T(O)
        @test eltype(A \ b) == eltype(A \ B) == eltype(B / A) == Int
    end
end

@testset "sparse transpose adjoint" begin
    A = sprand(10, 10, 0.75)
    @test A' == SparseMatrixCSC(A')
    @test SparseMatrixCSC(A') isa SparseMatrixCSC
    @test transpose(A) == SparseMatrixCSC(transpose(A))
    @test SparseMatrixCSC(transpose(A)) isa SparseMatrixCSC
    @test SparseMatrixCSC{eltype(A)}(transpose(A)) == transpose(A)
    @test SparseMatrixCSC{eltype(A), Int}(transpose(A)) == transpose(A)
    @test SparseMatrixCSC{Float16}(transpose(A)) == transpose(SparseMatrixCSC{Float16}(A))
    @test SparseMatrixCSC{Float16, Int}(transpose(A)) == transpose(SparseMatrixCSC{Float16}(A))
    B = sprand(ComplexF64, 10, 10, 0.75)
    @test SparseMatrixCSC{eltype(B)}(adjoint(B)) == adjoint(B)
    @test SparseMatrixCSC{eltype(B), Int}(adjoint(B)) == adjoint(B)
    @test SparseMatrixCSC{ComplexF16}(adjoint(B)) == adjoint(SparseMatrixCSC{ComplexF16}(B))
    @test SparseMatrixCSC{ComplexF16, Int8}(adjoint(B)) == adjoint(SparseMatrixCSC{ComplexF16, Int8}(B))
end

@testset "Column view of sparse matrix " begin
    S = sparse(1:4, 1:4, 1:4)
    Sv = @view S[:,3:4]
    @test Sv * sparse(ones(2)) == Sv*ones(2) == Matrix(Sv) * ones(2)
    @test Sv * sparse(ones(2,2)) == Sv*ones(2,2) == Matrix(Sv) * ones(2,2)
end

@testset "UniformScaling" begin
    local A = sprandn(10, 10, 0.5)
    MA = Array(A)
    @test A + I == MA + I
    @test I + A == I + MA
    @test A - I == MA - I
    @test I - A == I - MA
end

@testset "unary minus for SparseMatrixCSC{Bool}" begin
    A = sparse([1,3], [1,3], [true, true])
    B = sparse([1,3], [1,3], [-1, -1])
    @test -A == B
end

@testset "sparse matrix norms" begin
    Ac = sprandn(10,10,.1) + im* sprandn(10,10,.1)
    MAc = Array(Ac)
    Ar = sprandn(10,10,.1)
    MAr = Array(Ar)
    Ai = ceil.(Int, Ar*100)
    MAi = Array(Ai)
    @test opnorm(Ac,1) ≈ opnorm(MAc,1)
    @test opnorm(Ac,Inf) ≈ opnorm(MAc,Inf)
    @test norm(Ac) ≈ norm(MAc)
    @test opnorm(Ar,1) ≈ opnorm(MAr,1)
    @test opnorm(Ar,Inf) ≈ opnorm(MAr,Inf)
    @test norm(Ar) ≈ norm(MAr)
    @test opnorm(Ai,1) ≈ opnorm(MAi,1)
    @test opnorm(Ai,Inf) ≈ opnorm(MAi,Inf)
    @test norm(Ai) ≈ norm(MAi)
    Ai = trunc.(Int, Ar*100)
    MAi = Array(Ai)
    @test opnorm(Ai,1) ≈ opnorm(MAi,1)
    @test opnorm(Ai,Inf) ≈ opnorm(MAi,Inf)
    @test norm(Ai) ≈ norm(MAi)
    Ai = round.(Int, Ar*100)
    MAi = Array(Ai)
    @test opnorm(Ai,1) ≈ opnorm(MAi,1)
    @test opnorm(Ai,Inf) ≈ opnorm(MAi,Inf)
    @test norm(Ai) ≈ norm(MAi)
    # make certain entries in nzval beyond
    # the range specified in colptr do not
    # impact norm of a sparse matrix
    foo = sparse(1.0I, 4, 4)
    resize!(nonzeros(foo), 5)
    setindex!(nonzeros(foo), NaN, 5)
    @test norm(foo) == 2.0

    # Test (m x 1) sparse matrix
    colM = sprandn(10, 1, 0.6)
    McolM = Array(colM)
    @test opnorm(colM, 1) ≈ opnorm(McolM, 1)
    @test opnorm(colM) ≈ opnorm(McolM)
    @test opnorm(colM, Inf) ≈ opnorm(McolM, Inf)
    @test_throws ArgumentError opnorm(colM, 3)

    # Test (1 x n) sparse matrix
    rowM = sprandn(1, 10, 0.6)
    MrowM = Array(rowM)
    @test opnorm(rowM, 1) ≈ opnorm(MrowM, 1)
    @test opnorm(rowM) ≈ opnorm(MrowM)
    @test opnorm(rowM, Inf) ≈ opnorm(MrowM, Inf)
    @test_throws ArgumentError opnorm(rowM, 3)
end

@testset "fillstored!" begin
    @test LinearAlgebra.fillstored!(sparse(2.0I, 5, 5), 1) == Matrix(I, 5, 5)
end

@testset "Diagonal linear solve" begin
    n = 12
    for relty in (Float32, Float64), elty in (relty, Complex{relty})
        dd=convert(Vector{elty}, randn(n))
        if elty <: Complex
            dd+=im*convert(Vector{elty}, randn(n))
        end
        D = Diagonal(dd); MD = Array(D)
        bd = rand(elty, n, n)
        b = sparse(bd)
        @test ldiv!(D, copy(b)) ≈ MD\bd
        @test_throws SingularException ldiv!(Diagonal(zeros(elty, n)), copy(b))
        b = rand(elty, n+1, n+1)
        b = sparse(b)
        @test_throws DimensionMismatch ldiv!(D, copy(b))
        b = view(rand(elty, n+1), Vector(1:n+1))
        @test_throws DimensionMismatch ldiv!(D, b)
        for b in (sparse(rand(elty,n,n)), sparse(rand(elty,n)))
            bd = Array(b)
            @test lmul!(copy(D), copy(b)) ≈ MD*bd
            @test lmul!(transpose(copy(D)), copy(b)) ≈ transpose(MD)*bd
            @test lmul!(adjoint(copy(D)), copy(b)) ≈ MD'*bd
        end

        v = sprand(eltype(D), size(D,1), 0.1)
        @test ldiv!(D, copy(v)) == D \ Array(v)
    end
end

@testset "triu/tril" begin
    n = 5
    local A = sprand(n, n, 0.2)
    AF = Array(A)
    @test Array(triu(A,1)) == triu(AF,1)
    @test Array(tril(A,1)) == tril(AF,1)
    @test Array(triu!(copy(A), 2)) == triu(AF,2)
    @test Array(tril!(copy(A), 2)) == tril(AF,2)
    @test tril(A, -n - 2) == zero(A)
    @test tril(A, n) == A
    @test triu(A, -n) == A
    @test triu(A, n + 2) == zero(A)

    # fkeep trim option
    @test isequal(length(rowvals(tril!(sparse([1,2,3], [1,2,3], [1,2,3], 3, 4), -1))), 0)
end

@testset "norm" begin
    local A
    A = sparse(Int[],Int[],Float64[],0,0)
    @test norm(A) == zero(eltype(A))
    A = sparse([1.0])
    @test norm(A) == 1.0
    @test_throws ArgumentError opnorm(sprand(5,5,0.2),3)
    @test_throws ArgumentError opnorm(sprand(5,5,0.2),2)
end

@testset "ishermitian/issymmetric" begin
    local A
    # real matrices
    A = sparse(1.0I, 5, 5)
    @test ishermitian(A) == true
    @test issymmetric(A) == true
    A[1,3] = 1.0
    @test ishermitian(A) == false
    @test issymmetric(A) == false
    A[3,1] = 1.0
    @test ishermitian(A) == true
    @test issymmetric(A) == true

    # complex matrices
    A = sparse((1.0 + 1.0im)I, 5, 5)
    @test ishermitian(A) == false
    @test issymmetric(A) == true
    A[1,4] = 1.0 + im
    @test ishermitian(A) == false
    @test issymmetric(A) == false

    A = sparse(ComplexF64(1)I, 5, 5)
    A[3,2] = 1.0 + im
    @test ishermitian(A) == false
    @test issymmetric(A) == false
    A[2,3] = 1.0 - im
    @test ishermitian(A) == true
    @test issymmetric(A) == false

    A = sparse(zeros(5,5))
    @test ishermitian(A) == true
    @test issymmetric(A) == true

    # explicit zeros
    A = sparse(ComplexF64(1)I, 5, 5)
    A[3,1] = 2
    nonzeros(A)[2] = 0.0
    @test ishermitian(A) == true
    @test issymmetric(A) == true

    # 15504
    m = n = 5
    colptr = [1, 5, 9, 13, 13, 17]
    rowval = [1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5]
    nzval = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    A = SparseMatrixCSC(m, n, colptr, rowval, nzval)
    @test issymmetric(A) == true
    nonzeros(A)[end - 3]  = 2.0
    @test issymmetric(A) == false

    # stored zeros must not cause out-of-bounds access when the
    # partner column runs out of stored entries
    A = sparse([3, 1], [2, 3], [1.0, 0.0], 3, 3)
    @test issymmetric(A) == false
    @test ishermitian(A) == false
    A = sparse([1, 2], [2, 1], [0.0, 0.0], 2, 2)
    @test issymmetric(A) == true
    @test ishermitian(A) == true

    # 16521
    @test issymmetric(sparse([0 0; 1 0])) == false
    @test issymmetric(sparse([0 1; 0 0])) == false
    @test issymmetric(sparse([0 0; 1 1])) == false
    @test issymmetric(sparse([1 0; 1 0])) == false
    @test issymmetric(sparse([0 1; 1 0])) == true
    @test issymmetric(sparse([1 1; 1 0])) == true

    # test some non-trivial cases
    local S
    @testset "random matrices" begin
        for sparsity in (0.1, 0.01, 0.0)
            S = sparse(Symmetric(sprand(20, 20, sparsity)))
            @test issymmetric(S)
            @test ishermitian(S)
            S = sparse(Symmetric(sprand(ComplexF64, 20, 20, sparsity)))
            @test issymmetric(S)
            @test !ishermitian(S) || isreal(S)
            S = sparse(Hermitian(sprand(ComplexF64, 20, 20, sparsity)))
            @test ishermitian(S)
            @test !issymmetric(S) || isreal(S)
        end
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
end

@testset "rotations" begin
    a = sparse( [1,1,2,3], [1,3,4,1], [1,2,3,4] )

    @test rot180(a,2) == a
    @test rot180(a,1) == sparse( [3,3,2,1], [4,2,1,4], [1,2,3,4] )
    @test rotr90(a,1) == sparse( [1,3,4,1], [3,3,2,1], [1,2,3,4] )
    @test rotl90(a,1) == sparse( [4,2,1,4], [1,1,2,3], [1,2,3,4] )
    @test rotl90(a,2) == rot180(a)
    @test rotr90(a,2) == rot180(a)
    @test rotl90(a,3) == rotr90(a)
    @test rotr90(a,3) == rotl90(a)

    #ensure we have preserved the correct dimensions!

    a = sparse(1.0I, 3, 5)
    @test size(rot180(a)) == (3,5)
    @test size(rotr90(a)) == (5,3)
    @test size(rotl90(a)) == (5,3)
end

@testset "istriu/istril" begin
    local A = fill(1, 5, 5)
    @test istriu(sparse(triu(A)))
    @test !istriu(sparse(A))
    @test istril(sparse(tril(A)))
    @test !istril(sparse(A))
end

@testset "trace" begin
    @test_throws DimensionMismatch tr(spzeros(5,6))
    @test tr(sparse(1.0I, 5, 5)) == 5
end

@testset "spdiagm" begin
    x = fill(1, 2)
    @test spdiagm(0 => x, -1 => x) == [1 0 0; 1 1 0; 0 1 0]
    @test spdiagm(0 => x,  1 => x) == [1 1 0; 0 1 1; 0 0 0]

    for (x, y) in ((rand(5), rand(4)),(sparse(rand(5)), sparse(rand(4))))
        @test spdiagm(-1 => x)::SparseMatrixCSC         == diagm(-1 => x)
        @test spdiagm( 0 => x)::SparseMatrixCSC         == diagm( 0 => x) == sparse(Diagonal(x))
        @test spdiagm(0 => x, -1 => y)::SparseMatrixCSC == diagm(0 => x, -1 => y)
        @test spdiagm(0 => x,  1 => y)::SparseMatrixCSC == diagm(0 => x,  1 => y)
    end
    # promotion
    @test spdiagm(0 => [1,2], 1 => [3.5], -1 => [4+5im]) == [1 3.5; 4+5im 2]

    # sparse eltypes should infer well, even for a `Vararg` tail of unknown length
    @test Base.infer_return_type(SparseArrays.spdiagm_eltype,
              Tuple{Vararg{Pair{Int,Vector{Float64}}}}) === Core.Typeof(Float64)

    # no diagonals
    @test spdiagm(3, 4)::SparseMatrixCSC{Bool,Int} == diagm(3, 4)
    @test spdiagm()::SparseMatrixCSC{Bool,Int} == diagm()

    # convenience constructor
    @test spdiagm(x)::SparseMatrixCSC == diagm(x)
    @test nnz(spdiagm(x)) == count(!iszero, x)
    @test nnz(spdiagm(sparse([x; 0]))) == 2
    @test spdiagm(3, 4, x)::SparseMatrixCSC == diagm(3, 4, x)
    @test nnz(spdiagm(3, 4, sparse([x; 0]))) == 2

    # non-square:
    for m=1:4, n=2:4
        if m < 2 || n < 3
            @test_throws DimensionMismatch spdiagm(m,n, 0 => x,  1 => x)
        else
            M = zeros(m,n)
            M[1:2,1:3] = [1 1 0; 0 1 1]
            @test spdiagm(m,n, 0 => x,  1 => x) == M
        end
    end

    # sparsity-preservation
    x = sprand(10, 0.2); y = ones(9)
    @test spdiagm(0 => x, 1 => y) == diagm(0 => x, 1 => y)
    @test nnz(spdiagm(0 => x, 1 => y)) == length(y) + nnz(x)
end

@testset "diag" begin
    for T in (Float64, ComplexF64)
        S1 = sprand(T,  5,  5, 0.5)
        S2 = sprand(T, 10,  5, 0.5)
        S3 = sprand(T,  5, 10, 0.5)
        for S in (S1, S2, S3)
            local A = Matrix(S)
            @test diag(S)::SparseVector{T,Int} == diag(A)
            for k in -size(S,1):size(S,2)
                @test diag(S, k)::SparseVector{T,Int} == diag(A, k)
            end
            @test_throws ArgumentError diag(S, -size(S,1)-1)
            @test_throws ArgumentError diag(S,  size(S,2)+1)
        end
    end
    # test that stored zeros are still stored zeros in the diagonal
    S = sparse([1,3],[1,3],[0.0,0.0]); V = diag(S)
    @test nonzeroinds(V) == [1,3]
    @test nonzeros(V) == [0.0,0.0]
end

@testset "conj" begin
    cA = sprandn(5,5,0.2) + im*sprandn(5,5,0.2)
    @test Array(conj.(cA)) == conj(Array(cA))
    @test Array(conj!(copy(cA))) == conj(Array(cA))
end

@testset "SparseMatrixCSC [c]transpose[!] and permute[!]" begin
    smalldim = 5
    largedim = 10
    nzprob = 0.4
    (m, n) = (smalldim, smalldim)
    A = sprand(m, n, nzprob)
    X = similar(A)
    C = copy(transpose(A))
    p = randperm(m)
    q = randperm(n)
    @testset "common error checking of [c]transpose! methods (ftranspose!)" begin
        @test_throws DimensionMismatch transpose!(A[:, 1:(smalldim - 1)], A)
        @test_throws DimensionMismatch transpose!(A[1:(smalldim - 1), 1], A)
        @test_throws ArgumentError transpose!(A, A) # #812
        @test_throws ArgumentError adjoint!(A, A)
    end
    @testset "common error checking of permute[!] methods / source-perm compat" begin
        @test_throws DimensionMismatch permute(A, p[1:(end - 1)], q)
        @test_throws DimensionMismatch permute(A, p, q[1:(end - 1)])
    end
    @testset "common error checking of permute[!] methods / source-dest compat" begin
        @test_throws DimensionMismatch permute!(A[1:(m - 1), :], A, p, q)
        @test_throws DimensionMismatch permute!(A[:, 1:(m - 1)], A, p, q)
        @test_throws ArgumentError permute!((Y = copy(X); resize!(rowvals(Y), nnz(A) - 1); Y), A, p, q)
        @test_throws ArgumentError permute!((Y = copy(X); resize!(nonzeros(Y), nnz(A) - 1); Y), A, p, q)
    end
    @testset "common error checking of permute[!] methods / source-workmat compat" begin
        @test_throws DimensionMismatch permute!(X, A, p, q, C[1:(m - 1), :])
        @test_throws DimensionMismatch permute!(X, A, p, q, C[:, 1:(m - 1)])
        @test_throws ArgumentError permute!(X, A, p, q, (D = copy(C); resize!(rowvals(D), nnz(A) - 1); D))
        @test_throws ArgumentError permute!(X, A, p, q, (D = copy(C); resize!(nonzeros(D), nnz(A) - 1); D))
    end
    @testset "common error checking of permute[!] methods / source-workcolptr compat" begin
        @test_throws DimensionMismatch permute!(A, p, q, C, Vector{eltype(rowvals(A))}(undef, length(getcolptr(A)) - 1))
    end
    @testset "common error checking of permute[!] methods / permutation validity" begin
        @test_throws ArgumentError permute!(A, (r = copy(p); r[2] = r[1]; r), q)
        @test_throws ArgumentError permute!(A, (r = copy(p); r[2] = m + 1; r), q)
        @test_throws ArgumentError permute!(A, p, (r = copy(q); r[2] = r[1]; r))
        @test_throws ArgumentError permute!(A, p, (r = copy(q); r[2] = n + 1; r))
    end
    @testset "overall functionality of [c]transpose[!] and permute[!]" begin
        for (m, n) in ((smalldim, smalldim), (smalldim, largedim), (largedim, smalldim))
            A = sprand(m, n, nzprob)
            At = copy(transpose(A))
            # transpose[!]
            fullAt = Array(transpose(A))
            @test copy(transpose(A)) == fullAt
            @test transpose!(similar(At), A) == fullAt
            # adjoint[!]
            C = A + im*A/2
            fullCh = Array(C')
            @test copy(C') == fullCh
            @test adjoint!(similar(sparse(fullCh)), C) == fullCh
            # permute[!]
            p = randperm(m)
            q = randperm(n)
            fullPAQ = Array(A)[p,q]
            @test permute(A, p, q) == sparse(Array(A[p,q]))
            @test permute!(similar(A), A, p, q) == fullPAQ
            @test permute!(similar(A), A, p, q, similar(At)) == fullPAQ
            @test permute!(copy(A), p, q) == fullPAQ
            @test permute!(copy(A), p, q, similar(At)) == fullPAQ
            @test permute!(copy(A), p, q, similar(At), similar(getcolptr(A))) == fullPAQ
        end
    end
end

@testset "transpose of SubArrays" begin
    A = view(sprandn(10, 10, 0.3), 1:4, 1:4)
    @test copy(transpose(Array(A))) == Array(transpose(A))
    @test copy(adjoint(Array(A))) == Array(adjoint(A))
end

@testset "exp" begin
    A = sprandn(5,5,0.2)
    @test ℯ.^A ≈ ℯ.^Array(A)
end

@testset "Adding sparse-backed SymTridiagonal (#46355)" begin
    a = SymTridiagonal(sparsevec(Int[1]), sparsevec(Int[]))
    @test a + a == Matrix(a) + Matrix(a)

    # symtridiagonal with non-empty off-diagonal
    b = SymTridiagonal(sparsevec(Int[1, 2, 3]), sparsevec(Int[1, 2]))
    @test b + b == Matrix(b) + Matrix(b)
end

@testset "kronecker product" begin
    for (m,n) in ((5,10),)
        a = sprand(m, 5, 0.4); a_d = Matrix(a)
        b = sprand(n, 6, 0.3); b_d = Matrix(b)
        v = view(a, :, 1); v_d = Vector(v)
        x = sprand(m, 0.4); x_d = Vector(x)
        y = sprand(n, 0.3); y_d = Vector(y)
        c_dis = Any[Bidiagonal(rand(m), rand(m-1), :U),
                    Bidiagonal(rand(m), rand(m-1), :L),
                    Diagonal(rand(m)),
                    SymTridiagonal(rand(m), rand(m-1)),
                    Tridiagonal(rand(m-1), rand(m), rand(m-1))]
        d_dis = Any[Bidiagonal(rand(n), rand(n-1), :U),
                    Bidiagonal(rand(n), rand(n-1), :L),
                    Diagonal(rand(n)),
                    SymTridiagonal(rand(n), rand(n-1)),
                    Tridiagonal(rand(n-1), rand(n), rand(n-1))]
        # mat ⊗ mat
        for t in (identity, adjoint, transpose)
            @test kron(t(a), b)::SparseMatrixCSC == kron(t(a_d), b_d)
            @test kron(a, t(b))::SparseMatrixCSC == kron(a_d, t(b_d))
            @test kron(t(a), t(b))::SparseMatrixCSC == kron(t(a_d), t(b_d))
            @test kron(t(a), b_d)::SparseMatrixCSC == kron(t(a_d), b_d)
            @test kron(a_d, t(b))::SparseMatrixCSC == kron(a_d, t(b_d))
        end
        for c_di in c_dis
            c_d = Array(c_di)
            for t in (identity, adjoint, transpose)
                @test kron(t(a), c_di)::SparseMatrixCSC == kron(t(a_d), c_d)
                @test kron(a, t(c_di))::SparseMatrixCSC == kron(a_d, t(c_d))
                @test kron(t(a), t(c_di))::SparseMatrixCSC == kron(t(a_d), t(c_d))
            end
            @test kron(c_di, y)::SparseMatrixCSC == kron(c_di, y_d)
        end
        for d_di in d_dis
            @test kron(x, d_di)::SparseMatrixCSC == kron(x_d, d_di)
        end
        # vec ⊗ vec
        @test Vector(kron(x, y)::SparseVector) == kron(x_d, y_d)
        @test Vector(kron(x_d, y)::SparseVector) == kron(x_d, y_d)
        @test Vector(kron(x, y_d)::SparseVector) == kron(x_d, y_d)
        for t in (identity, adjoint, transpose)
            # mat ⊗ vec
            @test kron(t(a), y)::SparseMatrixCSC == kron(t(a_d), y_d)
            @test kron(t(a_d), y)::SparseMatrixCSC == kron(t(a_d), y_d)
            @test kron(t(a), y_d)::SparseMatrixCSC == kron(t(a_d), y_d)
            # vec ⊗ mat
            @test kron(x, t(b))::SparseMatrixCSC == kron(x_d, t(b_d))
            @test kron(x_d, t(b))::SparseMatrixCSC == kron(x_d, t(b_d))
            @test kron(x, t(b_d))::SparseMatrixCSC == kron(x_d, t(b_d))
        end
        # vec ⊗ vec'
        @test kron(v, y')::SparseMatrixCSC == kron(v_d, y_d')
        @test kron(x, y')::SparseMatrixCSC == kron(x_d, y_d')
        # test different types
        z = convert(SparseVector{Float16, Int8}, y); z_d = Vector(z)
        @test Vector(kron(x, z)) == kron(x_d, z_d)
        @test kron(a, z) == kron(a_d, z_d)
        @test kron(z, b) == kron(z_d, b_d)
        # test bounds checks
        @test_throws DimensionMismatch kron!(copy(a), a, b)
        @test_throws DimensionMismatch kron!(copy(x), x, y)
        @test_throws DimensionMismatch kron!(spzeros(2,2), x, y')
    end
end

@testset "sparse Frobenius dot/inner product" begin
    full_view = M -> view(M, :, :)
    for i = 1:5
        A = sprand(ComplexF64,10,15,0.4); MA = Matrix(A)
        B = sprand(ComplexF64,10,15,0.5); MB = Matrix(B)
        C = rand(10,15) .> 0.3; MC = Matrix(C)
        @test dot(A,B) ≈ dot(MA, MB)
        @test dot(A,B) ≈ dot(A, MB)
        @test dot(A,B) ≈ dot(MA, B)
        @test dot(A,C) ≈ dot(MA, C)
        @test dot(C,A) ≈ dot(C, MA)
        # square matrices required by most linear algebra wrappers
        SA = A * A'; MSA = Matrix(SA)
        SB = B * B'; MSB = Matrix(SB)
        SC = C * C'; MSC = Matrix(SC)
        for W in (full_view, LowerTriangular, UpperTriangular, UpperHessenberg, Symmetric, Hermitian)
            WA = W(MSA)
            WB = W(MSB)
            WC = W(MSC)
            @test dot(WA,SB) ≈ dot(WA, MSB)
            @test dot(SA,WB) ≈ dot(MSA, WB)
            @test dot(SA,WC) ≈ dot(MSA, WC)
        end
        for W in (transpose, adjoint)
            WA = W(MA)
            WB = W(MB)
            WC = W(MC)
            TA = copy(W(A))
            TB = copy(W(B))
            @test dot(WA,TB) ≈ dot(WA, Matrix(TB))
            @test dot(TA,WB) ≈ dot(Matrix(TA), WB)
            @test dot(TA,WC) ≈ dot(Matrix(TA), WC)
            # lazy adjoint/transpose of a sparse matrix (issue #627)
            @test dot(W(A), TB) ≈ dot(WA, Matrix(TB))
            @test dot(TA, W(B)) ≈ dot(Matrix(TA), WB)
            @test dot(W(A), sparse(WC)) ≈ dot(WA, WC)
            @test_throws DimensionMismatch dot(W(A), B)
        end
        for M in (A, B, C)
            D = Diagonal(M * M')
            a = spzeros(Complex{Float64}, size(D, 1))
            a[1:3] = rand(Complex{Float64}, 3)
            b = spzeros(Complex{Float64}, size(D, 1))
            b[1:3] = rand(Complex{Float64}, 3)
            @test dot(a, D, b) ≈ dot(a, sparse(D), b)
            @test dot(b, D, a) ≈ dot(b, sparse(D), a)
            @test dot(b, D, a) ≈ dot(b, D, collect(a))
            @test dot(b, D, a) ≈ dot(collect(b), D, a)
            @test_throws DimensionMismatch dot(b, D, [a; 1])
            @test_throws DimensionMismatch dot([b; 1], D, a)
            @test_throws DimensionMismatch dot([b; 1], D, [a; 1])
        end
    end
    @test_throws DimensionMismatch dot(sprand(5,5,0.2),sprand(5,6,0.2))
    @test_throws DimensionMismatch dot(rand(5,5),sprand(5,6,0.2))
    @test_throws DimensionMismatch dot(sprand(5,5,0.2),rand(5,6))
    # stored zeros, empty columns, and non-square shapes with a lazy adjoint (issue #627)
    for W in (adjoint, transpose)
        A = sparse([1, 3, 3, 5], [1, 1, 4, 2], [1.0im, 0.0, 2.0, 3.0], 6, 4)
        B = sparse([1, 2, 4, 4], [3, 3, 1, 6], [1.0, 0.0, 4.0im, 5.0], 4, 6)
        @test dot(W(A), B) ≈ dot(W(Matrix(A)), Matrix(B))
        @test dot(B, W(A)) ≈ dot(Matrix(B), W(Matrix(A)))
        @test dot(W(spzeros(6, 4)), B) == 0
        @test dot(W(A), spzeros(4, 6)) == 0
        # Int eltype and small matrices with `Any`-free result type
        Ai = sparse([1, 2], [2, 1], [1, 2], 2, 2)
        @test dot(W(Ai), Ai) == dot(W(Matrix(Ai)), Matrix(Ai)) == 4
        @test dot(W(Ai), Ai) isa Int
    end
    # the kernel walks the sparser operand and multiplies only where both operands store
    # an entry, whereas the generic fallback multiplies every stored entry of the sparse
    # operand
    P = mulcount_sparse(sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 6, 4))
    for W in (adjoint, transpose)
        # disjoint patterns: `B[i, j]` is stored only where `P[j, i]` is not
        B = mulcount_sparse(sparse([1, 2, 4, 4], [2, 3, 1, 6], [1.0, 2.0, 3.0, 4.0], 4, 6))
        @test mulcount(() -> dot(W(P), B)) == 0
        @test mulcount(() -> dot(B, W(P))) == 0
        # two matching pairs, found from either side of the walk
        B = mulcount_sparse(sparse([1, 1, 2, 3, 4, 4], [1, 2, 3, 3, 1, 6], 1.0:6.0, 4, 6))
        @test nnz(B) + size(B, 2) > nnz(P) + size(P, 2)     # walks P
        @test mulcount(() -> dot(W(P), B)) == 2
        Pw = mulcount_sparse(sparse([1, 2, 3, 4, 5, 5, 5, 6, 6], [1, 2, 3, 4, 1, 2, 4, 1, 2], 1.0:9.0, 6, 4))
        @test nnz(Pw) + size(Pw, 2) > nnz(B) + size(B, 2)   # walks B
        @test mulcount(() -> dot(W(Pw), B)) == 2
    end
    # mixed adjoint and transpose wrappers walk the parents, multiplying only matching entries
    let A = sparse([1, 3, 3, 5], [1, 1, 4, 2], [1.0im, 0.0, 2.0, 3.0 + im], 6, 4),
        B = sparse([1, 3, 4, 5], [1, 4, 1, 2], [2.0 - im, 0.5im, 4.0im, 5.0], 6, 4)
        @test dot(A', transpose(B)) ≈ dot(Matrix(A)', transpose(Matrix(B)))
        @test dot(transpose(A), B') ≈ dot(transpose(Matrix(A)), Matrix(B)')
        @test_throws DimensionMismatch dot(A', transpose(sparse(B')))
        Ac, Bc = mulcount_sparse.(SparseMatrixCSC.(6, 4, getcolptr.((A, B)), rowvals.((A, B)), Ref(ones(4))))
        @test mulcount(() -> dot(Ac', transpose(Bc))) == 3
        @test mulcount(() -> dot(transpose(Ac), Bc')) == 3
    end
    # column views of sparse matrices reach the same kernels as their parents
    let Q = sparse([1, 3, 3, 4, 2, 4, 1], [1, 1, 2, 3, 4, 5, 6], [1.0im, 0.0, 2.0, 3.0 + im, 4.0, 5.0im, 6.0], 4, 6),
        x = [1.0im, 2.0, 3.0, 4.0 - im], y = [2.0, 1.0 - im, 3.0im, 1.0],
        sx = sparsevec([1, 3], [2.0im, 1.0], 4), sy = sparsevec([1, 2], [1.0 + im, 3.0], 4)
        # distinct operands, so that a misplaced conjugate shows
        V = view(Q, :, [6, 1, 3, 2]); M = Matrix(V)
        B = view(Q, :, 2:5); S = sparse(B); MB = Matrix(B)
        @test dot(V, B) ≈ dot(V, S) ≈ dot(V, MB) ≈ dot(M, MB)
        @test dot(S, V) ≈ dot(MB, V) ≈ dot(MB, M)
        for W in (adjoint, transpose)
            T = copy(W(S))
            @test dot(W(V), T) ≈ dot(W(M), W(MB))
            @test dot(T, W(V)) ≈ dot(W(MB), W(M))
        end
        @test dot(V', transpose(B)) ≈ dot(M', transpose(MB))
        @test dot(x, V, y) ≈ dot(x, M, y)
        @test dot(sx, V, sy) ≈ dot(Vector(sx), M, Vector(sy))
        for (H, uplo) in ((Symmetric, :U), (Hermitian, :L))
            @test dot(x, H(V, uplo), y) ≈ dot(x, H(M, uplo), y)
            @test dot(sx, H(V, uplo), sy) ≈ dot(Vector(sx), H(M, uplo), Vector(sy))
        end
        @test_throws DimensionMismatch dot(V, Q)
        A = mulcount_sparse(sparse(1.0I, 8, 10)); P = A[:, 1:8]; V = view(A, :, 1:8)
        u = fill(MulCount(1.0), 8); su = sparse(u); D = fill(MulCount(1.0), 8, 8)
        for f in (() -> dot(V, P), () -> dot(P, V), () -> dot(V, V), () -> dot(V', P), () -> dot(P', V),
                  () -> dot(V', V), () -> dot(V, V'), () -> dot(V', transpose(V)), () -> dot(D, V), () -> dot(V, D))
            @test mulcount(f) == 8
        end
        for f in (() -> dot(u, V, u), () -> dot(su, V, su), () -> dot(u, Symmetric(V), u), () -> dot(su, Symmetric(V), su))
            @test mulcount(f) == 16
        end
    end
    # far more columns than stored entries: a binary search per entry, no cursor array
    for W in (adjoint, transpose)
        P = sparse([1], [1], [1.0], 2, 10^5); B = sparse([1], [1], [2.0], 10^5, 2)
        @test dot(W(P), B) == 2
        dot(W(P), B)
        @test (@allocated dot(W(P), B)) < 1024
        # and a wide operand with few entries is not walked column by column
        P = sparse([1], [1], [1.0], 1, 10^6); B = sparse([1, 2], [1, 1], [2.0, 3.0], 10^6, 1)
        @test dot(W(P), B) == 2
        @test (@allocated dot(W(P), B)) < 1024
    end
    # fixed operands are read only
    @test dot(fixed(sprand(5, 4, 0.5))', sprand(4, 5, 0.5)) isa Float64
    # matrix-valued entries have no `zero`, but the result is a scalar
    Bm = sparse([1, 2, 2], [1, 1, 2], [rand(2, 2) for _ in 1:3], 2, 2)
    Mm = [zeros(2, 2) for _ in 1:2, _ in 1:2]
    for (i, j, v) in zip(findnz(Bm)...); Mm[i, j] = v; end
    @test dot(Bm, Bm) ≈ dot(Mm, Bm) ≈ dot(Bm, Mm) ≈ dot(Mm, Mm)
    @test dot(Bm', Bm) ≈ dot(Bm, Bm') ≈ dot(Mm', Mm)
    @test dot(spzeros(Matrix{Float64}, 2, 2), Bm) == 0
end

@testset "generalized dot product" begin
    A = sprand(ComplexF64, 10, 15, 1.0)
    A15 = sprand(ComplexF64, 15, 15, 1.0)
    Av = view(A, :, :)
    vx = sprand(ComplexF64, 10, 0.5)
    vy = sprand(ComplexF64, 15, 0.5)
    vy2 = sprand(ComplexF64, 15, 0.5)
    for (x, y, y2) in ((vx, vy, vy2), (Vector(vx), Vector(vy), Vector(vy2)))
        @test dot(x, A, y) ≈ dot(Vector(x), A, Vector(y)) ≈ (Vector(x)' * Matrix(A)) * Vector(y)
        @test dot(x, A, y) ≈ dot(x, Av, y)
        @test dot(x, SparseMatrixCSC{eltype(A),Int32}(A), y) ≈ dot(x, A, y)
        @test dot(x, collect(A), y) ≈ dot(x, A, y)
        @test dot(y, collect(A)', x) ≈ dot(y, A', x)
        @test dot(y, transpose(collect(A)), x) ≈ dot(y, transpose(A), x)
        @test dot(y, Hermitian(collect(A15)), y2) ≈ dot(y, Hermitian(A15), y2)
        @test dot(y, Symmetric(collect(A15)), y2) ≈ dot(y, Symmetric(A15), y2)
        B = BitMatrix(rand(Bool, 10, 15))
        @test dot(x, A, y) ≈ dot(x, Matrix(A), y)
        @test_throws DimensionMismatch dot([x, x], A, y)
        @test_throws DimensionMismatch dot(x, A, [y, y])
        @test iszero(dot(spzeros(length(x)), A, y))
    end
    # matrix-valued entries: `dot(x, A, y)` entrywise, not `dot(x, A) * y`
    Bm = sparse([1, 2, 2], [1, 1, 2], [rand(2, 2) for _ in 1:3], 2, 2)
    xm = [rand(2, 2) for _ in 1:2]; ym = [rand(2, 2) for _ in 1:2]
    r = sum(dot(xm[i], Bm[i, j], ym[j]) for (i, j) in zip(findnz(Bm)[1:2]...))
    @test dot(xm, Bm, ym) ≈ dot(sparsevec(xm), Bm, sparsevec(ym)) ≈ r

    for T in (Float64, ComplexF64, Quaternion{Float64}), trans in (Symmetric,  Hermitian), uplo in (:U, :L)
        B = sprandn(T, 10, 10, 0.2)
        x = sprandn(T, 10, 0.4)
        xd = Vector(x)
        S = trans(B, uplo)
        Sd = trans(Matrix(B), uplo)
        @test dot(x, S, x) ≈ dot(x, Sd, x) ≈ dot(xd, S, xd) ≈ dot(xd, Sd, xd)
    end
end

@testset "conversion to special LinearAlgebra types" begin
    # issue 40924
    @test convert(Diagonal, sparse(Diagonal(1:2))) isa Diagonal
    @test convert(Diagonal, sparse(Diagonal(1:2))) == Diagonal(1:2)
    @test convert(Tridiagonal, sparse(Tridiagonal(1:3, 4:7, 8:10))) isa Tridiagonal
    @test convert(Tridiagonal, sparse(Tridiagonal(1:3, 4:7, 8:10))) == Tridiagonal(1:3, 4:7, 8:10)
    @test convert(SymTridiagonal, sparse(SymTridiagonal(1:4, 5:7))) isa SymTridiagonal
    @test convert(SymTridiagonal, sparse(SymTridiagonal(1:4, 5:7))) == SymTridiagonal(1:4, 5:7)

    lt = LowerTriangular([1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0])
    @test convert(LowerTriangular, sparse(lt)) isa LowerTriangular
    @test convert(LowerTriangular, sparse(lt)) == lt

    ut = UpperTriangular([1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0])
    @test convert(UpperTriangular, sparse(ut)) isa UpperTriangular
    @test convert(UpperTriangular, sparse(ut)) == ut
end

@testset "SparseMatrixCSC construction from UniformScaling" begin
    @test_throws ArgumentError SparseMatrixCSC(I, -1, 3)
    @test_throws ArgumentError SparseMatrixCSC(I, 3, -1)
    @test SparseMatrixCSC(2I, 3, 3)::SparseMatrixCSC{Int,Int} == Matrix(2I, 3, 3)
    @test SparseMatrixCSC(2I, 3, 4)::SparseMatrixCSC{Int,Int} == Matrix(2I, 3, 4)
    @test SparseMatrixCSC(2I, 4, 3)::SparseMatrixCSC{Int,Int} == Matrix(2I, 4, 3)
    @test SparseMatrixCSC(2.0I, 3, 3)::SparseMatrixCSC{Float64,Int} == Matrix(2I, 3, 3)
    @test SparseMatrixCSC{Real}(2I, 3, 3)::SparseMatrixCSC{Real,Int} == Matrix(2I, 3, 3)
    @test SparseMatrixCSC{Float64}(2I, 3, 3)::SparseMatrixCSC{Float64,Int} == Matrix(2I, 3, 3)
    @test SparseMatrixCSC{Float64,Int32}(2I, 3, 3)::SparseMatrixCSC{Float64,Int32} == Matrix(2I, 3, 3)
    @test SparseMatrixCSC{Float64,Int32}(0I, 3, 3)::SparseMatrixCSC{Float64,Int32} == Matrix(0I, 3, 3)
end
@testset "sparse(S::UniformScaling, shape...) convenience constructors" begin
    # we exercise these methods only lightly as these methods call the SparseMatrixCSC
    # constructor methods well-exercised by the immediately preceding testset
    @test sparse(2I, 3, 4)::SparseMatrixCSC{Int,Int} == Matrix(2I, 3, 4)
    @test sparse(2I, (3, 4))::SparseMatrixCSC{Int,Int} == Matrix(2I, 3, 4)
    @test sparse(3I, 4, 5) == sparse(1:4, 1:4, 3, 4, 5)
    @test sparse(3I, 5, 4) == sparse(1:4, 1:4, 3, 5, 4)
end

end # module
