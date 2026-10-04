# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseLinalgTests

using Test
using SparseArrays
using SparseArrays: AbstractSparseMatrixCSC, nonzeroinds, getcolptr, rowvals, nonzeros, fixed, _is_fixed
using LinearAlgebra
using Random
include("testhelpers.jl")

@static if COMPREHENSIVE
@testset "circshift" begin
    m,n = 17,15
    A = fixture(Float64, m, n)
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
        @static if COMPREHENSIVE
        # test different in/out types
        A2 = floor.(100A)
        E1 = spzeros(Int64, m, n)
        E2 = spzeros(Int64, m, n)
        circshift!(E1, A2, shifts)
        circshift!(E2, Matrix(A2), shifts)
        @test E1 == E2
        end
    end
    # the buffers of a source may be longer than its stored entries
    S = sparse([1 0 2; 0 3 0; 4 0 5.0])
    P = copy(S); push!(rowvals(P), 1); push!(nonzeros(P), 99.0)
    @test mismatch(circshift!(similar(S), P, (1, 1)), circshift(Matrix(S), (1, 1))) === nothing
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
    A[:,7:n] = [complex(i + j, i - 2j) for i in 1:m, j in 7:n]
    B = Matrix(A)
    dowrap(wr, A) = wr(A)
    dowrap(wr::Tuple, A) = (wr[1])(A, wr[2:end]...)

    @testset "sparse($wr(A))" for wr in (
                        Symmetric, (Hermitian, :L), UpperTriangular, UnitLowerTriangular,
                        (view, 3:6, 2:5), (@static COMPREHENSIVE ? (
                        (Symmetric, :L), Adjoint, UnitUpperTriangular) : ())...)

        @test mismatch(SparseMatrixCSC(dowrap(wr, A)), Matrix(dowrap(wr, B))) === nothing
    end

    @testset "sparse($at($wr))" for (at, wr) in ((Adjoint, LowerTriangular), (@static COMPREHENSIVE ? (
        (Transpose, UnitUpperTriangular),) : ())...)

        @test mismatch(SparseMatrixCSC(at(wr(A))), Matrix(at(wr(B)))) === nothing
    end

    @test mismatch(sparse([1,2,3,4,5]'), [1 2 3 4 5]) === nothing
    @static if COMPREHENSIVE
    @test sparse(Adjoint(UpperTriangular(A'))) == Adjoint(UpperTriangular(B'))
    @test sparse(UnitUpperTriangular(spzeros(5,5))) == I
    end
    deepwrap(A) = (Adjoint(LowerTriangular(view(Symmetric(A), 5:7, 4:6))))
    @test mismatch(sparse(deepwrap(A)), Matrix(deepwrap(B))) === nothing

    @testset "$wr of a non-CSC sparse matrix" for wr in (
                        Symmetric, (@static COMPREHENSIVE ? ((Hermitian, :L), (view, 3:6, 2:5)) : ())...)
        X = NonCSCSparse(A)
        @test mismatch(SparseMatrixCSC(dowrap(wr, X)), Matrix(dowrap(wr, B)); Tv=ComplexF64, Ti=Int) === nothing
        @static if COMPREHENSIVE
        @test mismatch(sparse(dowrap(wr, X)), Matrix(dowrap(wr, B)); Tv=ComplexF64, Ti=Int) === nothing
        end
    end
    @static if COMPREHENSIVE
    @test sparse(Adjoint(UnitUpperTriangular(NonCSCSparse(A))))::SparseMatrixCSC == Matrix(Adjoint(UnitUpperTriangular(B)))

    # the counter of the Symmetric/Hermitian copy kernel stays an `Int` over `Int32` indices
    A32 = SparseMatrixCSC{ComplexF64,Int32}(A)
    @test mismatch(SparseMatrixCSC(Symmetric(A32)), Matrix(Symmetric(B)); Tv=ComplexF64, Ti=Int32) === nothing
    @test mismatch(sparse(Hermitian(A32, :L)), Matrix(Hermitian(B, :L)); Tv=ComplexF64, Ti=Int32) === nothing
    for wr in (Symmetric{ComplexF64,typeof(A32)}, Hermitian{ComplexF64,typeof(A32)}),
            rangefun in (SparseArrays.nzrangeup, SparseArrays.nzrangelo)
        @test !hasunionlocal(SparseArrays._sparsem, (typeof(rangefun), wr), Int32, Int)
    end
    end
end

@testset "sums of Symmetric and Hermitian sparse matrices" begin
    A = fixture(ComplexF64, 4, 4)
    B = copy(A')
    # a real symmetric matrix is Hermitian, so the sum keeps the wrapper, and the triangle
    # the operand names: the two triangles of the fixture differ
    S = Symmetric(real(A), :L) + Hermitian(B)
    @test S isa Hermitian{ComplexF64, <:SparseMatrixCSC}
    @test S == Symmetric(real(Matrix(A)), :L) + collect(Hermitian(B))
    @test Hermitian(A) + Symmetric(real(B)) isa Hermitian{ComplexF64, <:SparseMatrixCSC}
    @test Hermitian(A) + Symmetric(B) isa SparseMatrixCSC
    # a wrapped and a plain sparse matrix; the #35325 testset in issues.jl has every
    # such combination, and it runs in comprehensive mode only
    @static if !COMPREHENSIVE
    @test mismatch(A + Hermitian(B), A + collect(Hermitian(B)); approx=true) === nothing
    @test mismatch(Hermitian(A) + B, collect(Hermitian(A)) + B; approx=true) === nothing
    end
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
    # the destination depends on the right-hand side and on a unit diagonal only, so the
    # triangular wrappers stand for the rest
    for T in (@static COMPREHENSIVE ? wrappers[5:7] : (LowerTriangular, UnitLowerTriangular))
        A = T(O)
        @static if COMPREHENSIVE
        bs = fixturevec(Float64, 10)
        bd = Array(bs)
        x = A \ bs
        @test x ≈ A \ bd
        @test !issparse(x)
        end
        Bs = fixture(Float64, 10, 3)
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
    @static if COMPREHENSIVE
    b, B = ones(Int, 10), ones(Int, 10, 10)
    for T in (UnitLowerTriangular,)
        A = T(O)
        @test eltype(A \ b) == eltype(A \ B) == eltype(B / A) == Int
    end
    end
end

@testset "Column view of sparse matrix " begin
    S = sparse(1:4, 1:4, 1:4)
    Sv = @view S[:,3:4]
    @test Sv * sparse(ones(2)) == Sv*ones(2) == Matrix(Sv) * ones(2)
    @test Sv * sparse(ones(2,2)) == Sv*ones(2,2) == Matrix(Sv) * ones(2,2)
end

@testset "UniformScaling" begin
    local A = fixture(Float64, 4, 4)
    MA = Array(A)
    @test mismatch(A + I, MA + I) === nothing
    @static if COMPREHENSIVE
    @test mismatch(I + A, I + MA) === nothing
    @test mismatch(A - I, MA - I) === nothing
    end
    @test mismatch(I - A, I - MA) === nothing
end

@static if COMPREHENSIVE
@testset "unary minus for SparseMatrixCSC{Bool}" begin
    A = sparse([1,3], [1,3], [true, true])
    B = sparse([1,3], [1,3], [-1, -1])
    @test -A == B
end
end

@testset "sparse matrix norms" begin
    # the largest column and the largest row are the last ones
    Ac = fixture(ComplexF64, 5, 3)
    MAc = Array(Ac)
    @static if COMPREHENSIVE
    Ar = imag(fixture(ComplexF64, 10, 8))   # entries of both signs
    Ai = ceil.(Int, Ar*100)
    MAi = Array(Ai)
    end
    @test opnorm(Ac,1) ≈ opnorm(MAc,1)
    @test opnorm(Ac,Inf) ≈ opnorm(MAc,Inf)
    @test norm(Ac) ≈ norm(MAc)
    @static if COMPREHENSIVE
    @test opnorm(Ai,1) ≈ opnorm(MAi,1)
    @test opnorm(Ai,Inf) ≈ opnorm(MAi,Inf)
    @test norm(Ai) ≈ norm(MAi)
    end
    # make certain entries in nzval beyond
    # the range specified in colptr do not
    # impact norm of a sparse matrix
    foo = sparse(1.0I, 4, 4)
    resize!(nonzeros(foo), 5)
    setindex!(nonzeros(foo), NaN, 5)
    @test norm(foo) == 2.0

    # a view of a column range sees the stored entries of those columns only, so
    # neither the other columns nor the entries beyond nnz contribute
    @test norm(view(foo, :, 2:3)) == sqrt(2.0)
    Az = sparse([1, 2, 3], [1, 2, 3], [1.0, 0.0, 3.0])   # stored zero at (2, 2)
    @test norm(view(Az, :, 2:3), 0) == norm(Matrix(Az)[:, 2:3], 0) == 1.0

    # Test (m x 1) sparse matrix
    colM = fixture(Float64, 10, 1)
    McolM = Array(colM)
    @test opnorm(colM, 1) ≈ opnorm(McolM, 1)
    @test opnorm(colM) ≈ opnorm(McolM)
    @test opnorm(colM, Inf) ≈ opnorm(McolM, Inf)
    @test_throws ArgumentError opnorm(colM, 3)

    # Test (1 x n) sparse matrix
    rowM = fixture(Float64, 1, 10)
    MrowM = Array(rowM)
    @test opnorm(rowM, 1) ≈ opnorm(MrowM, 1)
    @test opnorm(rowM) ≈ opnorm(MrowM)
    @test opnorm(rowM, Inf) ≈ opnorm(MrowM, Inf)
    @test_throws ArgumentError opnorm(rowM, 3)

    @testset "2-norm" begin
        rng = Random.Xoshiro(1)
        for T in (STD_ELTYPES..., (@static COMPREHENSIVE ? (ComplexF32,) : ())...), (m, n) in ((60, 40), (@static COMPREHENSIVE ? ((40, 60),) : ())...)
            A = sprandn(rng, T, m, n, 0.1)
            @test opnorm(A) ≈ opnorm(Array(A))
            @test opnorm(A) isa real(T)
        end
        @static if COMPREHENSIVE
        @test opnorm(sparse([1 2; 3 4])) ≈ opnorm([1 2; 3 4])
        end
        # the vector of ones is in the null space
        @test opnorm(sparse([1.0 -1.0; -1.0 1.0])) ≈ 2
        @static if COMPREHENSIVE
        # clustered singular values
        @test opnorm(spdiagm([fill(1.0, 9999); 1 + 1e-6])) ≈ 1 + 1e-6 rtol=1e-12
        # slow convergence, with about as many iterations as columns
        L = spdiagm(-1 => -ones(1999), 0 => 2ones(2000), 1 => -ones(1999))
        @test opnorm(L) ≈ 4cos(pi/4002)^2 rtol=1e-12
        end
        # magnitudes beyond the range of `Float64`, at `Float64` accuracy
        @test opnorm(spdiagm([1e-307, 5e-308])) ≈ 1e-307
        @static if COMPREHENSIVE
        @test opnorm(spdiagm([big"1e400", big"5e399"])) ≈ big"1e400" rtol=1e-10
        end
        @test isnan(opnorm(sparse([1.0 NaN; 2.0 3.0])))
        Z = spzeros(4, 5)
        Z[2, 3] = 1
        nonzeros(Z)[1] = 0
        @test opnorm(Z) === 0.0
    end
end

@testset "fillstored!" begin
    @test mismatch(LinearAlgebra.fillstored!(sparse(2.0I, 5, 5), 1), Matrix(1.0I, 5, 5)) === nothing
end

@testset "Diagonal linear solve" begin
    n = 12
    for elty in (ComplexF64,)
        dd=convert(Vector{elty}, 1:n)
        if elty <: Complex
            dd+=im*convert(Vector{elty}, n:-1:1)
        end
        D = Diagonal(dd); MD = Array(D)
        b = fixture(elty, n, 5)
        bd = Array(b)
        @test mismatch(ldiv!(D, copy(b)), MD\bd; approx=true) === nothing
        @test_throws SingularException ldiv!(Diagonal(zeros(elty, n)), copy(b))
        b = Matrix(fixture(elty, n+1, n+1))
        b = sparse(b)
        @test_throws DimensionMismatch ldiv!(D, copy(b))
        @static if COMPREHENSIVE
        b = view(Vector(fixturevec(elty, n+1)), Vector(1:n+1))
        @test_throws DimensionMismatch ldiv!(D, b)
        end
        for b in (fixture(elty, n, 5), fixturevec(elty, n))
            bd = Array(b)
            @test mismatch(lmul!(copy(D), copy(b)), MD*bd; approx=true) === nothing
            @test mismatch(lmul!(adjoint(copy(D)), copy(b)), MD'*bd; approx=true) === nothing
        end

        v = fixturevec(eltype(D), size(D,1))
        @test mismatch(ldiv!(D, copy(v)), D \ Array(v)) === nothing
    end

    # `D \ A` and `A / D` returned `Inf` for a singular `D`; the zero is not the first entry,
    # so a kernel that checked while scaling would leave the destination modified
    @testset "D \\ A and A / D" begin
        A = sparse([1.0+im 0 2; 0 3 0]); MA = Matrix(A); v = sparsevec([2], [1.0+im], 2)
        Dl = Diagonal([2.0, 1+im]); Dr = Diagonal([1.0, 2im, 3])
        @test mismatch(Dl \ A, Dl \ MA; approx=true) === nothing
        @test mismatch(A / Dr, MA / Dr; approx=true) === nothing
        @test mismatch(ldiv!(Dl, copy(v)), Dl \ Vector(v); approx=true) === nothing
        z = 0.0im; Dl0 = Diagonal([1.0, z]); Dr0 = Diagonal([1.0, 2, z]); B = copy(A)
        @test_throws SingularException(2) Dl0 \ A
        @test_throws SingularException(2) ldiv!(Dl0, B)
        @test B == A
        @test_throws SingularException(3) A / Dr0
        @test_throws SingularException(3) rdiv!(B, Dr0)
        @test B == A
        @test_throws DimensionMismatch Dr \ A
        # a destination with the columns of `A` but not its rows
        C = spzeros(ComplexF64, 3, 3)
        @test_throws DimensionMismatch ldiv!(C, Dl, A)
        @test nnz(C) == 0
    end
end

@testset "\\ and factorize substitute for diagonal and triangular matrices" begin
    # a `Rational` result proves substitution: a factorization would work in Float64
    T = Rational{Int}
    L = sparse(T[2 0 0; 1 3 0; 0 1 4])
    D, U = sparse(Diagonal(T[2, 3, 4])), sparse(transpose(L))
    b = T[1, 2, 3]
    @test factorize(D) isa Diagonal{T, <:SparseVector{T}}
    @test factorize(L) isa LowerTriangular{T, <:SparseMatrixCSC{T}}
    @test factorize(U) isa UpperTriangular{T, <:SparseMatrixCSC{T}}
    for A in (D, L, U)
        x = A \ b
        @test x isa Vector{T} && A * x == b
    end
    # the adjoint solve substitutes with the transformed choice
    x = L' \ b
    @test x isa Vector{T} && L' * x == b
    @static if COMPREHENSIVE
    # substitution keeps the result eltype of the right-hand side
    D = sparse([2.0 0; 0 3])
    for S in (D, sparse([2.0 0; 1 3]))
        @test (S \ Any[1.0, 2.0])::Vector{Any} ≈ Matrix(S) \ [1.0, 2.0]
    end
    Db = sparse([1, 2], [1, 2], [[2.0 0; 0 2], [3.0 0; 0 3]])
    @test Db \ [[1.0, 1.0], [3.0, 3.0]] == [[0.5, 0.5], [1.0, 1.0]]
    end
end

@testset "triu/tril" begin
    n = 5
    local A = fixture(Float64, n, 3)
    AF = Array(A)
    @test mismatch(triu(A,1), triu(AF,1)) === nothing
    @test mismatch(tril(A,1), tril(AF,1)) === nothing
    @test mismatch(triu!(copy(A), 2), triu(AF,2)) === nothing
    @test mismatch(tril!(copy(A), 2), tril(AF,2)) === nothing
    @test tril(A, -n - 2) == zero(A)
    @test tril(A, n) == A
    @test triu(A, -n) == A
    @test triu(A, n + 2) == zero(A)

    @static if COMPREHENSIVE
    # the copy pointer of `triu` stays an `Int` over `Int32` indices, like `tril`'s
    A32 = SparseMatrixCSC{Float64,Int32}(A)
    @test triu(A32, 1)::SparseMatrixCSC{Float64,Int32} == triu(AF, 1)
    @test !hasunionlocal(triu, (typeof(A32), Int), Int32, Int)
    @test !hasunionlocal(tril, (typeof(A32), Int), Int32, Int)
    end

    # fkeep trim option
    @test isequal(length(rowvals(tril!(sparse([1,2,3], [1,2,3], [1,2,3], 3, 4), -1))), 0)
end

@testset "norm" begin
    local A
    A = sparse(Int[],Int[],Float64[],0,0)
    @test norm(A) == zero(eltype(A))
    A = sparse([1.0])
    @test norm(A) == 1.0
    @test norm(sparse([1.0 0; 0 2]), -1) == norm(view(sparse([1.0 0; 0 2]), :, 1:2), -Inf) == 0.0
    @test isnan(norm(sparse([NaN 0; 0 2]), -1)) && isnan(norm(view(sparse([NaN 0; 0 2]), :, 1:2), -Inf))
    # with every entry stored there is no implicit zero to take the minimum
    @test norm(sparse([1.0 2; 3 4]), -1) ≈ 0.48 && norm(view(sparse([1.0 2; 3 4]), :, 1:2), -Inf) == 1.0
    @test_throws ArgumentError opnorm(fixture(Float64, 5, 3), 3)
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

    # a view of a column range is checked through the stored entries of those columns;
    # a stored zero without a stored counterpart does not break symmetry
    Cv = sparse([1, 2, 3, 1, 2, 1], [2, 4, 3, 5, 1, 6], [2.0, 2.0 + im, 2.0 - im, 0.0, 7.0, 8.0], 4, 6)
    V = view(Cv, :, 2:5)   # V[1, 4] is the stored zero
    @test ishermitian(V) == ishermitian(Matrix(V)) == true
    @test issymmetric(V) == issymmetric(Matrix(V)) == false
    @static if COMPREHENSIVE
    # an empty view may name columns outside its parent
    @test issymmetric(view(spzeros(0, 2), :, 100:99)) && ishermitian(view(spzeros(0, 2), :, 100:99))
    end
    @test issymmetric(A) == true

    # explicit zeros
    A = sparse(ComplexF64(1)I, 5, 5)
    A[3,1] = 2
    nonzeros(A)[2] = 0.0
    @test ishermitian(A) == true
    @test issymmetric(A) == true

    @static if COMPREHENSIVE
    # 15504
    m = n = 5
    colptr = [1, 5, 9, 13, 13, 17]
    rowval = [1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5]
    nzval = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    A = SparseMatrixCSC(m, n, colptr, rowval, nzval)
    @test issymmetric(A) == true
    nonzeros(A)[end - 3]  = 2.0
    @test issymmetric(A) == false
    end

    # stored zeros must not cause out-of-bounds access when the
    # partner column runs out of stored entries
    A = sparse([3, 1], [2, 3], [1.0, 0.0], 3, 3)
    @test issymmetric(A) == false
    @test ishermitian(A) == false
    A = sparse([1, 2], [2, 1], [0.0, 0.0], 2, 2)
    @test issymmetric(A) == true
    @test ishermitian(A) == true

    @static if COMPREHENSIVE
    # the partner-column offset stays an `Int` over `Int32` indices
    A32 = SparseMatrixCSC{ComplexF64,Int32}(sparse([1, 3, 2, 3, 1], [1, 1, 2, 2, 3], [1.0, 0.0, 2.0, 1.0 + im, 0.0], 3, 3))
    @test issymmetric(A32) == false
    @test ishermitian(A32) == false
    @test issymmetric(SparseMatrixCSC{ComplexF64,Int32}(sparse([1, 2, 1], [1, 1, 2], [1.0, 1.0 + im, 1.0 + im], 2, 2))) == true
    @test ishermitian(SparseMatrixCSC{ComplexF64,Int32}(sparse([1, 2, 1], [1, 1, 2], [1.0, 1.0 + im, 1.0 - im], 2, 2))) == true
    for check in (transpose, adjoint)
        @test !hasunionlocal(SparseArrays.is_hermsym, (typeof(A32), typeof(check)), Int32, Int)
    end

    # 16521
    @test issymmetric(sparse([0 0; 1 0])) == false
    @test issymmetric(sparse([0 1; 0 0])) == false
    @test issymmetric(sparse([0 0; 1 1])) == false
    @test issymmetric(sparse([1 0; 1 0])) == false
    @test issymmetric(sparse([0 1; 1 0])) == true
    @test issymmetric(sparse([1 1; 1 0])) == true
    end

    # test some non-trivial cases
    local S
    @testset "symmetrized matrices" begin
        # the two triangles of each differ; the last two have one stored entry and none
        for C in (fixture(ComplexF64, 20, 20), (@static COMPREHENSIVE ? (sparse([3], [17], [2.0 + im], 20, 20), spzeros(ComplexF64, 20, 20)) : ())...)
            @static if COMPREHENSIVE
            S = sparse(Symmetric(real(C)))
            @test issymmetric(S)
            @test ishermitian(S)
            end
            S = sparse(Symmetric(C))
            @test issymmetric(S)
            @test !ishermitian(S) || isreal(S)
            S = sparse(Hermitian(C))
            @test ishermitian(S)
            @test !issymmetric(S) || isreal(S)
        end
    end

    @static if COMPREHENSIVE
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
end

@testset "diff" begin
    @testset "$T" for T in (Float64,)
        A = fixture(T, 7, 5)
        A[2, 2] = zero(T); A[3, 2] = one(T); A[4, 2] = one(T) # stored zero and a cancelling pair
        A[4, 1] = one(T)                                       # and a pair cancelling across columns
        A[:, 4] .= zero(T)                                     # a column of stored zeros
        M = Array(A)
        for dims in (1, 2)
            D = diff(A; dims)
            @test D isa SparseMatrixCSC{T,Int}
            @test mismatch(D, diff(M; dims)) === nothing
            # the difference of two equal stored entries is not stored
            @test (dims == 1 ? 3 : 4) ∉ view(rowvals(D), nzrange(D, dims == 1 ? 2 : 1))
            @static if COMPREHENSIVE
            @test diff(sparse(Int32.(1:7), Int32.(1:7), one(T)); dims) isa SparseMatrixCSC{T,Int32}
            end
        end
        for dims in (0, 3, 7)
            @test_throws ArgumentError diff(M; dims)
            @test_throws ArgumentError diff(A; dims)
        end
    end
    @static if COMPREHENSIVE
    @testset "empty and unit sizes" begin
        for (m, n) in ((0, 0), (0, 3), (3, 0), (1, 3), (3, 1), (1, 1)), dims in (1, 2)
            A = spzeros(m, n)
            D = diff(A; dims)
            @test D isa SparseMatrixCSC{Float64,Int}
            @test size(D) == size(diff(Array(A); dims))
            @test mismatch(D, diff(Array(A); dims)) === nothing
        end
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

    a = sparse(1I, 3, 5)
    @test size(rot180(a)) == (3,5)
    @test size(rotr90(a)) == (5,3)
    @test size(rotl90(a)) == (5,3)

    # the index type and stored zeros survive, and a fixed input rotates into a plain copy
    a = (@static COMPREHENSIVE ? SparseMatrixCSC{Int,Int32} : identity)(sparse([1,1,2,3], [1,3,4,1], [1,2,3,4]))
    a[2,4] = 0
    for rot in (rot180, rotr90, rotl90)
        R = rot(a)
        @test mismatch(R, rot(collect(a))) === nothing && R isa typeof(a) && nnz(R) == 4
    end
    @static if COMPREHENSIVE
    @test rot180(fixed(a)) == rot180(collect(a)) && rot180(fixed(a)) isa SparseMatrixCSC
    b = sparse([1, 2, 1], [1, 2, 3], [1.0, 2.0, 3.0], 3, 3)
    push!(nonzeros(b), NaN)
    @test rot180(b) == rot180(Matrix(sparse([1, 2, 1], [1, 2, 3], [1.0, 2.0, 3.0], 3, 3)))
    end
end

@testset "istriu/istril" begin
    @static if COMPREHENSIVE
    local A = fill(1, 5, 5)
    @test istriu(sparse(triu(A)))
    @test !istriu(sparse(A))
    @test istril(sparse(tril(A)))
    @test !istril(sparse(A))

    @testset "extreme band offsets" begin
        S = sparse([1], [1], [1.0], 3, 3)
        for k in (typemax(Int)-1, typemax(Int))
            @test !istriu(S, k)
            @test istril(S, k)
        end
        @test istriu(S, typemin(Int))
        @test !istril(S, typemin(Int))
        nonzeros(S)[1] = 0
        for k in (typemin(Int), typemax(Int))
            @test istriu(S, k) && istril(S, k)
        end
        S = SparseMatrixCSC(typemax(Int), 3, [1, 2, 2, 2], [1], [1.0])
        @test !istriu(S, 2)
    end
    end

    @testset "band offset k, $T $(m)x$(n)" for T in (ComplexF64,), (m, n) in ((@static COMPREHENSIVE ? ((1, 2), (2, 1)) : ())..., (3, 5), (5, 3), (@static COMPREHENSIVE ? ((0, 3), (3, 0)) : ())...)
        @test which(istriu, (SparseMatrixCSC{T,Int}, Int)).module === SparseArrays
        @test which(istril, (SparseMatrixCSC{T,Int}, Int)).module === SparseArrays
        v = T <: Complex ? T(2 + im) : T(2)
        full = sparse(fill(v, m, n))
        for k in -3:3
            @test istriu(full, k) == istriu(Matrix(full), k)
            @test istril(full, k) == istril(Matrix(full), k)
            for i in 1:m, j in 1:n
                S = sparse([i], [j], [v], m, n)
                for X in (S, adjoint(S), transpose(S))
                    @test istriu(X, k) == istriu(Matrix(X), k)
                    @test istril(X, k) == istril(Matrix(X), k)
                end
                # a stored zero is not a nonzero
                nonzeros(S)[1] = zero(T)
                @test istriu(S, k) && istril(S, k)
                @test istriu(adjoint(S), k) && istril(transpose(S), k)
            end
        end
    end
end

@testset "trace" begin
    @test_throws DimensionMismatch tr(spzeros(5,6))
    @test tr(sparse(1.0I, 5, 5)) == 5
end

@testset "spdiagm" begin
    x = fill(1, 2)
    @test mismatch(spdiagm(0 => x, -1 => x), [1 0 0; 1 1 0; 0 1 0]) === nothing
    @test mismatch(spdiagm(0 => x,  1 => x), [1 1 0; 0 1 1; 0 0 0]) === nothing

    @static if COMPREHENSIVE
    v = Vector(fixturevec(Float64, 5))
    @test spdiagm(-1 => v)::SparseMatrixCSC         == diagm(-1 => v)
    @test spdiagm( 0 => v)::SparseMatrixCSC         == diagm( 0 => v) == sparse(Diagonal(v))
    # promotion
    @test spdiagm(0 => [1,2], 1 => [3.5], -1 => [4+5im]) == [1 3.5; 4+5im 2]

    # sparse eltypes should infer well, even for a `Vararg` tail of unknown length
    @test Base.infer_return_type(SparseArrays.spdiagm_eltype,
              Tuple{Vararg{Pair{Int,Vector{Float64}}}}) === Core.Typeof(Float64)

    # no diagonals
    @test spdiagm(3, 4)::SparseMatrixCSC{Bool,Int} == zeros(Bool, 3, 4)
    @test spdiagm()::SparseMatrixCSC{Bool,Int} == zeros(Bool, 0, 0)
    end

    # convenience constructor
    @test mismatch(spdiagm(x), diagm(x)) === nothing
    @test nnz(spdiagm(x)) == count(!iszero, x)
    @test nnz(spdiagm(sparse([x; 0]))) == 2
    @test mismatch(spdiagm(3, 4, x), diagm(3, 4, x)) === nothing
    @static if COMPREHENSIVE
    @test nnz(spdiagm(3, 4, sparse([x; 0]))) == 2
    end

    # non-square:
    for m=(@static COMPREHENSIVE ? (1:4) : (1, 3)), n=(@static COMPREHENSIVE ? (2:4) : (2, 4))
        if m < 2 || n < 3
            @test_throws DimensionMismatch spdiagm(m,n, 0 => x,  1 => x)
        else
            M = zeros(m,n)
            M[1:2,1:3] = [1 1 0; 0 1 1]
            @test mismatch(spdiagm(m,n, 0 => x,  1 => x), M; Tv=Int, Ti=Int) === nothing
        end
    end

    # sparsity-preservation
    x = fixturevec(Float64, 10); y = ones(Int, 9)
    @test spdiagm(0 => x, 1 => y)::SparseMatrixCSC{Float64,Int} == Bidiagonal(Vector(x), ones(9), :U)
    @test nnz(spdiagm(0 => x, 1 => y)) == length(y) + nnz(x)
end

@static if COMPREHENSIVE
@testset "diag" begin
    for T in (Float64,)
        S1 = fixture(T,  5,  5)
        S2 = fixture(T, 10,  5)
        S3 = fixture(T,  5, 10)
        for S in (S2, (@static COMPREHENSIVE ? (S1, S3) : ())...)
            local A = Matrix(S)
            @test diag(S)::SparseVector{T,Int} == diag(A)
            for k in -size(S,1):size(S,2)
                @test diag(S, k)::SparseVector{T,Int} == diag(A, k)
            end
            @test diag(S, -size(S,1)-1) == diag(A, -size(S,1)-1) == T[]
            @test diag(S,  size(S,2)+1) == diag(A,  size(S,2)+1) == T[]
        end
    end
    # test that stored zeros are still stored zeros in the diagonal
    S = sparse([1,3],[1,3],[0.0,0.0]); V = diag(S)
    @test nonzeroinds(V) == [1,3]
    @test nonzeros(V) == [0.0,0.0]

    # a view of a column range walks the stored entries of those columns and keeps its
    # stored zeros, like the parent
    S = sparse([1, 3, 2, 4], [2, 4, 4, 5], [1.0, 0.0, 2.0, 3.0], 4, 6)
    V = view(S, :, 2:5)
    @test diag(V)::SparseVector{Float64,Int} == diag(Matrix(V))
    @test nonzeros(diag(V)) == [1.0, 0.0, 3.0]
    @test isempty(diag(view(spzeros(2, 3), :, 1:2), 3))
end
end

@testset "conj" begin
    cA = fixture(ComplexF64, 5, 3)
    @test mismatch(conj.(cA), conj(Array(cA))) === nothing
    @test mismatch(conj!(copy(cA)), conj(Array(cA))) === nothing
end

@static if COMPREHENSIVE
@testset "exp" begin
    A = fixture(Float64, 5, 3)
    @test ℯ.^A ≈ ℯ.^Array(A)
end

@testset "Adding sparse-backed SymTridiagonal (#46355)" begin
    a = SymTridiagonal(sparsevec(Int[1]), sparsevec(Int[]))
    @test a + a == Matrix(a) + Matrix(a)

    # symtridiagonal with non-empty off-diagonal
    b = SymTridiagonal(sparsevec(Int[1, 2, 3]), sparsevec(Int[1, 2]))
    @test b + b == Matrix(b) + Matrix(b)
end
end

@testset "kronecker product" begin
    for (m,n) in ((5,10),)
        a = fixture(ComplexF64, m, 3); a_d = Matrix(a)
        b = fixture(ComplexF64, 3, 5); b_d = Matrix(b)
        v = view(a, :, 1); v_d = Vector(v)
        x = fixturevec(ComplexF64, m); x_d = Vector(x)
        y = fixturevec(Float64, n); y_d = Vector(y)
        c_dis = Any[(@static COMPREHENSIVE ? (Diagonal(Vector{Float64}(1:m)),) : ())...,
                    Tridiagonal(Vector{Float64}(1:m-1), Vector{Float64}(11:10+m), Vector{Float64}(21:19+m))]
        @static if COMPREHENSIVE
        d_dis = Any[Bidiagonal(Vector{Float64}(1:n), Vector{Float64}(11:9+n), :U),
                    SymTridiagonal(Vector{Float64}(1:n), Vector{Float64}(11:9+n))]
        end
        # mat ⊗ mat
        for t in (identity, adjoint, (@static COMPREHENSIVE ? (transpose,) : ())...)
            @test mismatch(kron(t(a), b), kron(t(a_d), b_d)) === nothing
            @test mismatch(kron(a, t(b)), kron(a_d, t(b_d))) === nothing
            @test mismatch(kron(t(a), t(b)), kron(t(a_d), t(b_d))) === nothing
            @test mismatch(kron(t(a), b_d), kron(t(a_d), b_d)) === nothing
            @test mismatch(kron(a_d, t(b)), kron(a_d, t(b_d))) === nothing
        end
        Kd = kron(a_d, y_d)
        @test mismatch(kron!(spzeros(eltype(Kd), size(Kd)...), a, y_d), Kd) === nothing
        @test mismatch(kron!(spzeros(eltype(Kd), size(Kd)...), y_d, a), kron(y_d, a_d)) === nothing
        @static if COMPREHENSIVE
        # complex operands, which an adjoint conjugates
        ac = fixture(ComplexF64, m, 5); ac_d = Matrix(ac)
        bc = fixture(ComplexF64, n, 6); bc_d = Matrix(bc)
        for (ta, tb) in ((adjoint, transpose),)
            @test mismatch(kron(ta(ac), tb(bc)), kron(ta(ac_d), tb(bc_d))) === nothing
        end
        end
        for c_di in c_dis
            c_d = Array(c_di)
            for t in (identity, (@static COMPREHENSIVE ? (adjoint,) : ())...)
                @test mismatch(kron(t(a), c_di), kron(t(a_d), c_d)) === nothing
                @test mismatch(kron(a, t(c_di)), kron(a_d, t(c_d))) === nothing
                @test mismatch(kron(t(a), t(c_di)), kron(t(a_d), t(c_d))) === nothing
            end
            @test mismatch(kron(c_di, y), kron(c_di, y_d)) === nothing
        end
        @static if COMPREHENSIVE
        for d_di in d_dis
            @test mismatch(kron(x, d_di), kron(x_d, d_di)) === nothing
        end
        end
        # vec ⊗ vec
        @test mismatch(kron(x, y), kron(x_d, y_d)) === nothing
        @test mismatch(kron(x_d, y), kron(x_d, y_d)) === nothing
        @static if COMPREHENSIVE
        @test mismatch(kron(x, y_d), kron(x_d, y_d)) === nothing
        end
        for t in (identity, (@static COMPREHENSIVE ? (transpose,) : ())...)
            # mat ⊗ vec
            @test mismatch(kron(t(a), y), kron(t(a_d), y_d)) === nothing
            @static if COMPREHENSIVE
            @test mismatch(kron(t(a_d), y), kron(t(a_d), y_d)) === nothing
            @test mismatch(kron(t(a), y_d), kron(t(a_d), y_d)) === nothing
            end
            # vec ⊗ mat
            @test mismatch(kron(x, t(b)), kron(x_d, t(b_d))) === nothing
            @static if COMPREHENSIVE
            @test mismatch(kron(x_d, t(b)), kron(x_d, t(b_d))) === nothing
            @test mismatch(kron(x, t(b_d)), kron(x_d, t(b_d))) === nothing
            end
        end
        # vec ⊗ vec'
        @test mismatch(kron(v, y'), kron(v_d, y_d')) === nothing
        @test mismatch(kron(x, y'), kron(x_d, y_d')) === nothing
        @static if COMPREHENSIVE
        # test different types
        z = convert(SparseVector{Float16, Int8}, y); z_d = Vector(z)
        @test mismatch(kron(x, z), kron(x_d, z_d); Ti=Int) === nothing
        @test mismatch(kron(a, z), kron(a_d, z_d); Ti=Int) === nothing
        @test mismatch(kron(z, b), kron(z_d, b_d); Ti=Int) === nothing
        # the index types promote, whichever operand has the wider one
        a32 = SparseMatrixCSC{ComplexF64,Int32}(a)
        @test mismatch(kron(a32, b), kron(a_d, b_d); Ti=Int) === nothing
        @test mismatch(kron(b, a32), kron(b_d, a_d); Ti=Int) === nothing
        end
        # test bounds checks
        @test_throws DimensionMismatch kron!(copy(a), a, b)
        @test_throws DimensionMismatch kron!(copy(x), x, y)
        @test_throws DimensionMismatch kron!(spzeros(ComplexF64, length(x)*length(y) + 1), x, y)
        @test_throws DimensionMismatch kron!(spzeros(2,2), x, y')
    end
end

@testset "sparse Frobenius dot/inner product" begin
    full_view = M -> view(M, :, :)
    for (m, n) in ((10, 15), (@static COMPREHENSIVE ? ((15, 10),) : ())...)
        A = fixture(ComplexF64, m, n); MA = Matrix(A)
        # another pattern and other values, with entries only one of the two stores
        B = permutedims(fixture(ComplexF64, n, m)); MB = Matrix(B)
        @static if COMPREHENSIVE
        C = [(i + 2j) % 4 != 0 for i in 1:m, j in 1:n]
        end
        @test dot(A,B) ≈ dot(MA, MB)
        @test dot(A,B) ≈ dot(A, MB)
        @test dot(A,B) ≈ dot(MA, B)
        @static if COMPREHENSIVE
        @test dot(A,C) ≈ dot(MA, C)
        @test dot(C,A) ≈ dot(C, MA)
        end
        # square matrices required by most linear algebra wrappers
        SA = A * A'; MSA = Matrix(SA)
        SB = B * B'; MSB = Matrix(SB)
        # the dense operand is read through `getindex` whatever wraps it, so a view and a
        # wrapper stand for the rest
        for W in ((@static COMPREHENSIVE ? (full_view,) : ())..., Hermitian)
            WA = W(MSA)
            WB = W(MSB)
            @test dot(WA,SB) ≈ dot(WA, MSB)
            @test dot(SA,WB) ≈ dot(MSA, WB)
        end
        for W in ((@static COMPREHENSIVE ? (transpose,) : ())..., adjoint)
            WA = W(MA)
            WB = W(MB)
            TA = copy(W(A))
            TB = copy(W(B))
            @test dot(WA,TB) ≈ dot(WA, Matrix(TB))
            @test dot(TA,WB) ≈ dot(Matrix(TA), WB)
            # lazy adjoint/transpose of a sparse matrix (issue #627)
            @test dot(W(A), TB) ≈ dot(WA, Matrix(TB))
            @test dot(TA, W(B)) ≈ dot(Matrix(TA), WB)
            @test_throws DimensionMismatch dot(W(A), B)
        end
        for M in (A,)
            # a diagonal that is not real, which the adjoint of `D` conjugates
            D = Diagonal(M * transpose(M))
            a = spzeros(Complex{Float64}, size(D, 1))
            a[1:3] = [complex(1.0, 2.0), complex(0.0, -0.5), complex(3.0, 0.0)]
            b = spzeros(Complex{Float64}, size(D, 1))
            b[1:3] = [complex(2.0, -1.0), complex(1.0, 1.0), complex(0.0, -1.5)]
            @test dot(a, D, b) ≈ dot(a, sparse(D), b)
            @test dot(b, D, a) ≈ dot(b, sparse(D), a)
            @test dot(b, D, a) ≈ dot(b, D, collect(a))
            @test dot(b, D, a) ≈ dot(collect(b), D, a)
            @test_throws DimensionMismatch dot(b, D, [a; 1])
            @test_throws DimensionMismatch dot([b; 1], D, a)
            @test_throws DimensionMismatch dot([b; 1], D, [a; 1])
        end
    end
    @test_throws DimensionMismatch dot(fixture(Float64, 5, 5), fixture(Float64, 5, 6))
    @test_throws DimensionMismatch dot(ones(5, 5), fixture(Float64, 5, 6))
    @test_throws DimensionMismatch dot(fixture(Float64, 5, 5), ones(5, 6))
    # stored zeros, empty columns, and non-square shapes with a lazy adjoint (issue #627)
    for W in (adjoint, transpose)
        A = sparse([1, 3, 3, 5], [1, 1, 4, 2], [1.0im, 0.0, 2.0, 3.0], 6, 4)
        B = sparse([1, 2, 4, 4], [3, 3, 1, 6], [1.0, 0.0, 4.0im, 5.0], 4, 6)
        @test dot(W(A), B) ≈ dot(W(Matrix(A)), Matrix(B))
        @test dot(B, W(A)) ≈ dot(Matrix(B), W(Matrix(A)))
        @test dot(W(spzeros(ComplexF64, 6, 4)), B) == 0
        @test dot(W(A), spzeros(ComplexF64, 4, 6)) == 0
        @static if COMPREHENSIVE
        # Int eltype and small matrices with `Any`-free result type
        Ai = sparse([1, 2], [2, 1], [1, 2], 2, 2)
        @test dot(W(Ai), Ai) == dot(W(Matrix(Ai)), Matrix(Ai)) == 4
        @test dot(W(Ai), Ai) isa Int
        end
    end
    # the kernel walks the sparser operand and multiplies only where both operands store
    # an entry, whereas the generic fallback multiplies every stored entry of the sparse
    # operand
    P = opcount_sparse(sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 6, 4))
    for W in (adjoint,)
        # disjoint patterns: `B[i, j]` is stored only where `P[j, i]` is not
        B = opcount_sparse(sparse([1, 2, 4, 4], [2, 3, 1, 6], [1.0, 2.0, 3.0, 4.0], 4, 6))
        @test mulcount(() -> dot(W(P), B)) == 0
        @test mulcount(() -> dot(B, W(P))) == 0
        # two matching pairs, found from either side of the walk
        B = opcount_sparse(sparse([1, 1, 2, 3, 4, 4], [1, 2, 3, 3, 1, 6], 1.0:6.0, 4, 6))
        @test nnz(B) + size(B, 2) > nnz(P) + size(P, 2)     # walks P
        @test mulcount(() -> dot(W(P), B)) == 2
        Pw = opcount_sparse(sparse([1, 2, 3, 4, 5, 5, 5, 6, 6], [1, 2, 3, 4, 1, 2, 4, 1, 2], 1.0:9.0, 6, 4))
        @test nnz(Pw) + size(Pw, 2) > nnz(B) + size(B, 2)   # walks B
        @test mulcount(() -> dot(W(Pw), B)) == 2
    end
    # mixed adjoint and transpose wrappers walk the parents, multiplying only matching entries
    let A = sparse([1, 3, 3, 5], [1, 1, 4, 2], [1.0im, 0.0, 2.0, 3.0 + im], 6, 4),
        B = sparse([1, 3, 4, 5], [1, 4, 1, 2], [2.0 - im, 0.5im, 4.0im, 5.0], 6, 4)
        @test dot(A', transpose(B)) ≈ dot(Matrix(A)', transpose(Matrix(B)))
        @test dot(transpose(A), B') ≈ dot(transpose(Matrix(A)), Matrix(B)')
        @test_throws DimensionMismatch dot(A', transpose(sparse(B')))
        @static if COMPREHENSIVE
        Ac = opcount_sparse(SparseMatrixCSC(6, 4, getcolptr(A), rowvals(A), ones(4)))
        Bc = opcount_sparse(SparseMatrixCSC(6, 4, getcolptr(B), rowvals(B), ones(4)))
        @test mulcount(() -> dot(Ac', transpose(Bc))) == 3
        @test mulcount(() -> dot(transpose(Ac), Bc')) == 3
        end
    end
    # column views of sparse matrices reach the same kernels as their parents
    let Q = sparse([1, 3, 3, 4, 2, 4, 1], [1, 1, 2, 3, 4, 5, 6], [1.0im, 0.0, 2.0, 3.0 + im, 4.0, 5.0im, 6.0], 4, 6),
        x = [1.0im, 2.0, 3.0, 4.0 - im], y = [2.0, 1.0 - im, 3.0im, 1.0],
        sx = sparsevec([1, 3], [2.0im, 1.0], 4), sy = sparsevec([1, 2], [1.0 + im, 3.0], 4)
        # distinct operands, so that a misplaced conjugate shows
        V = view(Q, :, [1, 5, 6, 2]); M = Matrix(V)
        B = view(Q, :, 2:5); S = sparse(B); MB = Matrix(B)
        @test dot(V, B) ≈ dot(V, S) ≈ dot(V, MB) ≈ dot(M, MB) && !isreal(dot(M, MB))
        @test dot(S, V) ≈ dot(MB, V) ≈ dot(MB, M)
        for W in (adjoint, (@static COMPREHENSIVE ? (transpose,) : ())...)
            T = copy(W(S))
            @test dot(W(V), T) ≈ dot(W(M), W(MB))
            @test dot(T, W(V)) ≈ dot(W(MB), W(M))
        end
        @test dot(V', B) ≈ dot(M', MB)
        @test dot(S', B) ≈ dot(MB', MB)
        @static if COMPREHENSIVE
        @test dot(V', transpose(B)) ≈ dot(M', transpose(MB))
        end
        @test dot(x, V, y) ≈ dot(x, M, y)
        @test dot(sx, V, sy) ≈ dot(Vector(sx), M, Vector(sy))
        for (H, uplo) in ((@static COMPREHENSIVE ? ((Symmetric, :U),) : ())..., (Hermitian, :L))
            @test dot(x, H(V, uplo), y) ≈ dot(x, H(M, uplo), y)
            @test dot(sx, H(V, uplo), sy) ≈ dot(Vector(sx), H(M, uplo), Vector(sy))
        end
        @test_throws DimensionMismatch dot(V, Q)
        @static if COMPREHENSIVE
        A = opcount_sparse(sparse(1.0I, 8, 10)); P = A[:, 1:8]; V = view(A, :, 1:8)
        u = fill(OpCount(1.0), 8); su = sparse(u); D = fill(OpCount(1.0), 8, 8)
        # each kernel once with a view operand
        for f in (() -> dot(V, P), () -> dot(V', P), () -> dot(V, V'), () -> dot(V', transpose(V)), () -> dot(D, V))
            @test mulcount(f) == 8
        end
        for f in (() -> dot(u, V, u), () -> dot(su, V, su), () -> dot(u, Symmetric(V), u), () -> dot(su, Symmetric(V), su))
            @test mulcount(f) == 16
        end
        end
    end
    # far more columns than stored entries: a binary search per entry, no cursor array
    for W in (adjoint,)
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
    @test dot(fixed(fixture(Float64, 5, 4))', fixture(Float64, 4, 5)) isa Float64
    @static if COMPREHENSIVE
    # matrix-valued entries have no `zero`, but the result is a scalar
    Bm = sparse([1, 2, 2], [1, 1, 2], [[1.0 2; 3 4] .+ k for k in 1:3], 2, 2)
    Mm = [zeros(2, 2) for _ in 1:2, _ in 1:2]
    for (i, j, v) in zip(findnz(Bm)...); Mm[i, j] = v; end
    @test dot(Bm, Bm) ≈ dot(Mm, Bm) ≈ dot(Bm, Mm) ≈ dot(Mm, Mm)
    @test dot(Bm', Bm) ≈ dot(Bm, Bm') ≈ dot(Mm', Mm)
    @test dot(spzeros(Matrix{Float64}, 2, 2), Bm) == 0
    end
end

@testset "generalized dot product" begin
    A = fixture(ComplexF64, 10, 15)
    A15 = fixture(ComplexF64, 15, 15)
    Av = view(A, :, :)
    vx = fixturevec(ComplexF64, 10)
    vy = fixturevec(ComplexF64, 15)
    vy2 = sparsevec([2, 3, 9, 15], [1.0 - im, 2.0im, -3.0, 4.0 + 2im], 15)
    for (x, y, y2) in ((vx, vy, vy2), (Vector(vx), Vector(vy), Vector(vy2)))
        @test dot(x, A, y) ≈ dot(Vector(x), A, Vector(y)) ≈ (Vector(x)' * Matrix(A)) * Vector(y)
        @test dot(x, A, y) ≈ dot(x, Av, y)
        @static if COMPREHENSIVE
        @test dot(x, SparseMatrixCSC{eltype(A),Int32}(A), y) ≈ dot(x, A, y)
        end
        @test dot(x, collect(A), y) ≈ dot(x, A, y)
        @test dot(y, collect(A)', x) ≈ dot(y, A', x)
        @static if COMPREHENSIVE
        @test dot(y, Hermitian(collect(A15)), y2) ≈ dot(y, Hermitian(A15), y2)
        @test dot(y, Symmetric(collect(A15)), y2) ≈ dot(y, Symmetric(A15), y2)
        end
        @test dot(x, A, y) ≈ dot(x, Matrix(A), y)
        @test_throws DimensionMismatch dot([x, x], A, y)
        @test_throws DimensionMismatch dot(x, A, [y, y])
        @test iszero(dot(spzeros(length(x)), A, y))
    end
    @test iszero(dot(spzeros(ComplexF64, 10), collect(A), vy))
    @static if COMPREHENSIVE
    # matrix-valued entries: `dot(x, A, y)` entrywise, not `dot(x, A) * y`
    Bm = sparse([1, 2, 2], [1, 1, 2], [[1.0 2; 3 4] .+ k for k in 1:3], 2, 2)
    xm = [[1.0 -1; 2 0.5] .* k for k in 1:2]; ym = [[0.5 3; -2 1] .+ k for k in 1:2]
    r = sum(dot(xm[i], Bm[i, j], ym[j]) for (i, j) in zip(findnz(Bm)[1:2]...))
    @test dot(xm, Bm, ym) ≈ dot(sparsevec(xm), Bm, sparsevec(ym)) ≈ r
    end

    for (T, trans, uplo) in ((ComplexF64, Symmetric, :U), (ComplexF64, Hermitian, :L), (@static COMPREHENSIVE ?
            ((Float64, Symmetric, :L), (ComplexF64, Symmetric, :L), (ComplexF64, Hermitian, :U),
             (quaternion_type(){Float64}, Hermitian, :U)) : ())...)
        B = fixture(T, 10, 10)
        x = fixturevec(T, 10)
        if !(T <: Union{Real,Complex})
            # the fixture of a quaternion type is real, and real entries commute
            nonzeros(B) .= [T(k, -k, 2, k % 3) for k in 1:nnz(B)]
            nonzeros(x) .= [T(1, k, -k, 3) for k in 1:nnz(x)]
        end
        xd = Vector(x)
        S = trans(B, uplo)
        Sd = trans(Matrix(B), uplo)
        @test dot(x, S, x) ≈ dot(x, Sd, x) ≈ dot(xd, S, xd) ≈ dot(xd, Sd, xd)
    end
    # a real Hermitian dense matrix between sparse vectors has a method of its own
    xr = real(vx); Sr = Hermitian(real(Matrix(A))[:, 1:10])
    @test dot(xr, Sr, xr) ≈ dot(Vector(xr), Sr, Vector(xr))
end

@static if COMPREHENSIVE
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
end

@testset "SparseMatrixCSC construction from UniformScaling" begin
    @test_throws ArgumentError SparseMatrixCSC(I, -1, 3)
    @test_throws ArgumentError SparseMatrixCSC(I, 3, -1)
    @test mismatch(SparseMatrixCSC(2I, 3, 3), Matrix(2I, 3, 3); Tv=Int, Ti=Int) === nothing
    @test mismatch(SparseMatrixCSC(2I, 3, 4), Matrix(2I, 3, 4); Tv=Int, Ti=Int) === nothing
    @test mismatch(SparseMatrixCSC(2I, 4, 3), Matrix(2I, 4, 3); Tv=Int, Ti=Int) === nothing
    @test mismatch(SparseMatrixCSC(2.0I, 3, 3), Matrix(2I, 3, 3); Tv=Float64, Ti=Int) === nothing
    @static if COMPREHENSIVE
    @test mismatch(SparseMatrixCSC{Real}(2I, 3, 3), Matrix(2I, 3, 3); Tv=Real, Ti=Int) === nothing
    end
    @test mismatch(SparseMatrixCSC{Float64}(2I, 3, 3), Matrix(2I, 3, 3); Tv=Float64, Ti=Int) === nothing
    @test mismatch(SparseMatrixCSC{Float64}(0I, 3, 3), Matrix(0I, 3, 3); Tv=Float64, Ti=Int) === nothing
    @static if COMPREHENSIVE
    @test mismatch(SparseMatrixCSC{Float64,Int32}(2I, 3, 3), Matrix(2I, 3, 3); Tv=Float64, Ti=Int32) === nothing
    @test mismatch(SparseMatrixCSC{Float64,Int32}(0I, 3, 3), Matrix(0I, 3, 3); Tv=Float64, Ti=Int32) === nothing
    end
end
@static if COMPREHENSIVE
@testset "sparse(S::UniformScaling, shape...) convenience constructors" begin
    # we exercise these methods only lightly as these methods call the SparseMatrixCSC
    # constructor methods well-exercised by the immediately preceding testset
    @test mismatch(sparse(2I, 3, 4), Matrix(2I, 3, 4); Tv=Int, Ti=Int) === nothing
    @test mismatch(sparse(2I, (3, 4)), Matrix(2I, 3, 4); Tv=Int, Ti=Int) === nothing
    @test sparse(3I, 4, 5) == sparse(1:4, 1:4, 3, 4, 5)
    @test sparse(3I, 5, 4) == sparse(1:4, 1:4, 3, 5, 4)
end
end

end # module
