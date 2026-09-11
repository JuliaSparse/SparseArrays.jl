module SparseLinalgProductTests
# Products, dot products, Kronecker products and conversions. Split off from linalg.jl
# so the two halves run on separate test workers.

using Test
using SparseArrays
using SparseArrays: nonzeroinds, getcolptr, rowvals, nonzeros, fixed
using LinearAlgebra
using Random
include("forbidproperties.jl")
include("util/mulcount.jl")

sA = sprandn(3, 7, 0.5)
sC = similar(sA)
dA = Array(sA)

const BASE_TEST_PATH = joinpath(Sys.BINDIR, "..", "share", "julia", "test")
isdefined(Main, :Quaternions) || @eval Main include(joinpath($(BASE_TEST_PATH), "testhelpers", "Quaternions.jl"))
using .Main.Quaternions

@testset "exp" begin
    A = sprandn(5,5,0.2)
    @test ℯ.^A ≈ ℯ.^Array(A)
end

@testset "matrix-vector multiplication (non-square)" begin
    for i = 1:5
        a = sprand(10, 5, 0.5)
        b = rand(5)
        @test maximum(abs.(a*b - Array(a)*b)) < 100*eps()
    end
end

@testset "diagonal - sparse vector mutliplication" begin
    for _ in 1:10
        b = spzeros(10)
        b[1:3] .= 1:3
        A = Diagonal(randn(10))
        @test norm(A * b - A * Vector(b)) <= 10eps()
        @test norm(A * b - Array(A) * b) <= 10eps()
        Ac = Diagonal(randn(Complex{Float64}, 10))
        @test norm(Ac * b - Ac * Vector(b)) <= 10eps()
        @test norm(Ac * b - Array(Ac) * b) <= 10eps()
        @test_throws DimensionMismatch A * [b; 1]
        @test_throws DimensionMismatch A * b[1:end-1]
    end
end

@testset "sparse matrix * BitArray" begin
    A = sprand(5,5,0.3)
    MA = Array(A)
    B = trues(5)
    @test A*B ≈ MA*B
    B = trues(5,5)
    for trA in (identity, adjoint, transpose), trB in (identity, adjoint, transpose)
        @test trA(A) * trB(B) ≈ trA(MA) * trB(B)
        @test trB(B) * trA(A) ≈ trB(B) * trA(MA)
    end
end


@testset "matrix multiplication" begin
    for (m, p, n, q, k) in (
                            (10, 0.7, 5, 0.3, 15),
                            (100, 0.01, 100, 0.01, 20),
                            (100, 0.1, 100, 0.2, 100),
                           )
        a = sprand(m, n, p); ad = Array(a)
        b = sprand(n, k, q); bd = Array(b)
        as = sparse(a')
        bs = sparse(b')
        ab = a * b
        aab = ad * bd
        @test maximum(abs.(ab - aab)) < 100*eps()
        @test a*bs' == ab
        @test as'*b == ab
        @test as'*bs' == ab
        f = Diagonal(rand(n))
        @test Array(a*f) == ad*f
        @test Array(f*b) == f*bd
        A = rand(2n, 2n)
        sA = view(A, 1:2:2n, 1:2:2n); dA = Array(sA)
        @test (sA*b)::Matrix ≈ dA*bd
        @test (a*sA)::Matrix ≈ ad*dA
        @test (sA'b)::Matrix ≈ dA'*bd
        c = sprandn(ComplexF32, n, n, q); cd = Array(c)
        @test (sA*c')::Matrix ≈ dA*cd'
        @test (c'*sA)::Matrix ≈ cd'*dA
        @test (sA'c)::Matrix ≈ dA'*cd
        @test (sA'c')::Matrix ≈ dA'*cd'
    end
end

@testset "multiplication of sparse matrix and triangular matrix" begin
    _sparse_test_matrix(n, T) =  T == Int ? sparse(rand(0:4, n, n)) : sprandn(T, n, n, 0.6)
    _triangular_test_matrix(n, TA, T) = T == Int ? TA(rand(0:9, n, n)) : TA(randn(T, n, n))

    n = 5
    for T1 in (Int, Float64, ComplexF32)
        S = _sparse_test_matrix(n, T1)
        MS = Matrix(S)
        for T2 in (Int, Float64, ComplexF32)
            for TM in (LowerTriangular, UnitLowerTriangular, UpperTriangular, UnitLowerTriangular)
                T = _triangular_test_matrix(n, TM, T2)
                MT = Matrix(T)
                @test isa(T * S, DenseMatrix)
                @test isa(S * T, DenseMatrix)
                for transT in (identity, adjoint, transpose), transS in (identity, adjoint, transpose)
                    @test transT(T) * transS(S) ≈ transT(MT) * transS(MS)
                    @test transS(S) * transT(T) ≈ transS(MS) * transT(MT)
                end
            end
        end
    end
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
        for (c_di, d_di) in Iterators.product(c_dis, d_dis)
            c = sparse(c_di); c_d = Array(c_di)
            d = sparse(d_di); d_d = Array(d_di)
            # mat ⊗ mat
            for t in (identity, adjoint, transpose)
                @test kron(t(a), b)::SparseMatrixCSC == kron(t(a_d), b_d)
                @test kron(a, t(b))::SparseMatrixCSC == kron(a_d, t(b_d))
                @test kron(t(a), t(b))::SparseMatrixCSC == kron(t(a_d), t(b_d))
                @test kron(t(a), b_d)::SparseMatrixCSC == kron(t(a_d), b_d)
                @test kron(a_d, t(b))::SparseMatrixCSC == kron(a_d, t(b_d))
                @test kron(t(a), c_di)::SparseMatrixCSC == kron(t(a_d), c_d)
                @test kron(a, t(c_di))::SparseMatrixCSC == kron(a_d, t(c_d))
                @test kron(t(a), t(c_di))::SparseMatrixCSC == kron(t(a_d), t(c_d))
                @test kron(c_di, y)::SparseMatrixCSC == kron(c_di, y_d)
                @test kron(x, d_di)::SparseMatrixCSC == kron(x_d, d_di)
            end
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
    # an entry (plus one multiplication seeding the accumulator), whereas the generic
    # fallback multiplies every stored entry of the sparse operand
    P = mulcount_sparse(sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 6, 4))
    for W in (adjoint, transpose)
        # disjoint patterns: `B[i, j]` is stored only where `P[j, i]` is not
        B = mulcount_sparse(sparse([1, 2, 4, 4], [2, 3, 1, 6], [1.0, 2.0, 3.0, 4.0], 4, 6))
        @test mulcount(() -> dot(W(P), B)) == 1
        @test mulcount(() -> dot(B, W(P))) == 1
        # two matching pairs, found from either side of the walk
        B = mulcount_sparse(sparse([1, 1, 2, 3, 4, 4], [1, 2, 3, 3, 1, 6], 1.0:6.0, 4, 6))
        @test nnz(B) > nnz(P)   # walks P
        @test mulcount(() -> dot(W(P), B)) == 1 + 2
        Pw = mulcount_sparse(sparse([1, 2, 3, 4, 5, 6, 6], [1, 2, 3, 4, 4, 1, 2], 1.0:7.0, 6, 4))
        @test nnz(Pw) > nnz(B)  # walks B
        @test mulcount(() -> dot(W(Pw), B)) == 1 + 2
    end
    # far more columns than stored entries: a binary search per entry, no cursor array
    for W in (adjoint, transpose)
        P = sparse([1], [1], [1.0], 2, 10^5); B = sparse([1], [1], [2.0], 10^5, 2)
        @test dot(W(P), B) == 2
        dot(W(P), B)
        @test (@allocated dot(W(P), B)) < 1024
    end
    # fixed operands are read only
    @test dot(fixed(sprand(5, 4, 0.5))', sprand(4, 5, 0.5)) isa Float64
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

@testset "diagonal-sandwiched triple multiplication" begin
    S = sprand(4, 6, 0.2)
    D1 = Diagonal(axes(S,1))
    D2 = Diagonal(axes(S,2) .+ 4)
    A = Array(S)
    C = D1 * S * D2
    @test C isa SparseMatrixCSC
    @test C ≈ D1 * A * D2
    C = D2 * S' * D1
    @test C isa SparseMatrixCSC
    @test C ≈ D2 * A' * D1
    C = D1 * view(S, :, :) * D2
    @test C isa SparseMatrixCSC
    @test C ≈ D1 * A * D2

    @test_throws DimensionMismatch D2 * S * D2
    @test_throws DimensionMismatch D1 * S * D1
end

@testset "multiplication of sparse and dense matrices" begin
    function test_mul(A, B)
        expected = Matrix(A) * Matrix(B)
        @test A * B ≈ expected
        C = similar(expected)
        @test mul!(C, A, B) === C
        @test C ≈ expected
        ElType = eltype(C)
        vs = Any[false, true, zero(ElType), one(ElType), one(ElType) + one(ElType)]
        for α in vs, β in vs
            C .= rand.(ElType)
            expected′ = expected .* α .+ C .* β
            @test mul!(C, A, B, α, β) === C
            @test C ≈ expected′
        end
    end

    for ElType in [Int, Float64, ComplexF64, BigFloat]
        SP = sprand(ElType, 10, 10, 0.3)
        D = rand(ElType, 10, 10)
        fs = [identity, adjoint, transpose]
        for f1 in fs, f2 in fs
            test_mul(f1(SP), f2(D))
            test_mul(f1(D), f2(SP))
        end
    end
end

@testset "dimension mismatch error" begin
    fs = [rand, (x, y)->adjoint(rand(y, x)), (x, y)->transpose(rand(y, x)),
          (x, y)->sprand(x, y, 0.5), (x, y)->adjoint(sprand(y, x, 0.5)),
          (x, y)->transpose(sprand(y, x, 0.5))]
    for fA in fs, fB in fs
        mul!(zeros(6, 10), fA(6, 8), fB(8, 10))
        @test_throws DimensionMismatch mul!(zeros(7, 10), fA(6, 8), fB(8, 10))
        @test_throws DimensionMismatch mul!(zeros(6, 11), fA(6, 8), fB(8, 10))
        @test_throws DimensionMismatch mul!(zeros(6, 10), fA(5, 8), fB(8, 10))
        @test_throws DimensionMismatch mul!(zeros(6, 10), fA(6, 9), fB(8, 10))
        @test_throws DimensionMismatch mul!(zeros(6, 10), fA(6, 8), fB(7, 10))
        @test_throws DimensionMismatch mul!(zeros(6, 10), fA(6, 8), fB(8, 9))
    end
end

@testset "type stability of linear solve" begin
    for relty in (Float16, Float32, Float64), elty in (relty, Complex{relty})
        A = sprand(elty, 2, 2, 1.0)
        B = randn(elty, 2, 2)
        b = randn(elty, 2)
        @inferred A \ b
        @inferred A \ B
    end
end

end
