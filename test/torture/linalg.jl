# This file is a part of Julia. License is MIT: https://julialang.org/license

# Long-tail regression tests for the LinearAlgebra wrappers of sparse matrices. Each
# testset names the issue it guards; they run only when the `torture` suite is selected.

module TortureLinalgTests
using Test
using SparseArrays
using LinearAlgebra
include("../testhelpers.jl")
using SparseArrays: getcolptr

# Test temporary fix for issue #16548 in PR #16979. Somewhat brittle. Expect to remove with `\` revisions.
@testset "issue #16548" begin
    ms = methods(\, (SparseMatrixCSC, AbstractVecOrMat)).ms
    @test all(m -> m.module == SparseArrays, ms)
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

@testset "issue #29644" begin
    F = lu(Tridiagonal(sparse(1.0I, 3, 3)))
    @test F.L == Matrix(I, 3, 3)
    @test startswith(sprint(show, MIME("text/plain"), F),
                     "$(LinearAlgebra.LU){Float64, $(LinearAlgebra.Tridiagonal){Float64, $(SparseArrays.SparseVector)")
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

# The transform/wrapper pairs of `sparse(at(wr(A)))` that the core "wrappers of sparse"
# testset does not run.
@testset "wrappers of sparse, remaining transform/wrapper pairs" begin
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

    @testset "sparse($at($wr))" for (at, wr) in ((Transpose, LowerTriangular), (Transpose, UnitUpperTriangular),
                                                (Transpose, UnitLowerTriangular), (Adjoint, UpperTriangular),
                                                (Adjoint, LowerTriangular), (Adjoint, UnitUpperTriangular))
        @test SparseMatrixCSC(at(wr(A))) == Matrix(at(wr(B)))
    end
end

# The single-precision eltypes of the diagonal solve; the core suite runs Float64 and
# ComplexF64.
@testset "Diagonal linear solve" begin
    n = 12
    for relty in (Float32,), elty in (relty, Complex{relty})
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

# Five random draws of the Frobenius inner product comparison; the core suite runs one.
@testset "sparse Frobenius dot/inner product, random draws" begin
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
end

# The real eltype of the symmetric generalized dot product; the core suite keeps
# ComplexF64 and the quaternions.
@testset "generalized dot product, real eltype" begin
    Quaternion = quaternion_type()
    for T in (Float64,), trans in (Symmetric,  Hermitian), uplo in (:U, :L)
        B = sprandn(T, 10, 10, 0.2)
        x = sprandn(T, 10, 0.4)
        xd = Vector(x)
        S = trans(B, uplo)
        Sd = trans(Matrix(B), uplo)
        @test dot(x, S, x) ≈ dot(x, Sd, x) ≈ dot(xd, S, xd) ≈ dot(xd, Sd, xd)
    end
end

end # module
