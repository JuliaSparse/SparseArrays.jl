
# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseTriangularTests

using Test
using SparseArrays
using SparseArrays: nonzeroinds, getcolptr, rowvals, nonzeros, fixed, FixedSparseVector
using LinearAlgebra
using Random
include("testhelpers.jl")

const TRIANGLES = (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)
const TRANSFORMS = (identity, adjoint, transpose)
# The kernels are compiled per transform and triangle, so a standard run gives each transform
# one triangle and a comprehensive run every pair. A tuple holding a transform has a type of
# its own for each transform, so the standard cases are vectors and `iscase`, which tells
# whether a nested loop runs the case `c`, is not specialized.
const TRICASES = @static COMPREHENSIVE ? Iterators.product(TRANSFORMS, TRIANGLES) :
    (Any[identity, UpperTriangular], Any[adjoint, LowerTriangular], Any[transpose, UnitUpperTriangular], Any[identity, UnitLowerTriangular])
iscase(@nospecialize(c), cases) = any(x -> x === c, cases)

@testset "multiplication of sparse matrix and triangular matrix" begin
    _sparse_test_matrix(n, T) =  T == Int ? sparse(rand(0:4, n, n)) : sprandn(T, n, n, 0.6)
    _triangular_test_matrix(n, TA, T) = T == Int ? TA(rand(0:9, n, n)) : TA(randn(T, n, n))

    function test_triangular_product(S, T)
        @test (T * S)::DenseMatrix ≈ Matrix(T) * Matrix(S)
        @test (S * T)::DenseMatrix ≈ Matrix(S) * Matrix(T)
    end

    n = 5
    wrappers = Any[(Float64, LowerTriangular, identity, transpose), (ComplexF64, UnitLowerTriangular, adjoint, identity),
                   (Float64, UpperTriangular, transpose, adjoint), (ComplexF64, UnitUpperTriangular, identity, transpose)]
    @static COMPREHENSIVE && append!(wrappers, pairwise(STD_ELTYPES, TRIANGLES, TRANSFORMS, TRANSFORMS))
    @testset "wrappers" begin
        for ElType in STD_ELTYPES
            S = _sparse_test_matrix(n, ElType)
            for TM in (LowerTriangular, UnitLowerTriangular, UpperTriangular, UnitUpperTriangular)
                T = _triangular_test_matrix(n, TM, ElType)
                for transT in (identity, adjoint, transpose), transS in (identity, adjoint, transpose)
                    iscase((ElType, TM, transT, transS), wrappers) || continue
                    test_triangular_product(transS(S), transT(T))
                end
            end
        end
    end
    types = (Int, Float64, (@static COMPREHENSIVE ? (ComplexF32,) : ())...)
    promotions = @static COMPREHENSIVE ? pairwise(types, types, (LowerTriangular, UpperTriangular)) : Any[(Int, Float64, LowerTriangular)]
    @testset "promotion" begin
        for T1 in types, T2 in types
            S = _sparse_test_matrix(n, T1)
            for TM in (LowerTriangular, UpperTriangular)
                iscase((T1, T2, TM), promotions) || continue
                test_triangular_product(S, _triangular_test_matrix(n, TM, T2))
            end
        end
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
    @static if COMPREHENSIVE
    Y = A * L
    @test Y ≈ AM * LM
    @test typeof(Y) == typeof(A)
    Y = U * A
    @test Y ≈ UM * AM
    @test typeof(Y) == typeof(A)
    end
    Y = L * A
    @test Y ≈ LM * AM
    @test typeof(Y) == typeof(A)
    Y = U * U
    @test Y ≈ UM * UM
    @test typeof(Y) == typeof(U)
    @static if COMPREHENSIVE
    Y = L * L
    @test Y ≈ LM * LM
    @test typeof(Y) == typeof(L)
    end
    Y = L * U
    @test Y ≈ LM * UM
    @test typeof(Y) == typeof(A)
    @static if COMPREHENSIVE
    Y = U * L
    @test Y ≈ UM * LM
    @test typeof(Y) == typeof(A)
    end
end


begin
    rng = Random.MersenneTwister(0)
    n = 100
    B = ones(n)
    X = reshape(1.0:3n, 3, n) ./ n
    s = sprandn(rng, n, 0.05)
    sd = Vector(s)
    A = sprand(rng, n, n, 0.01)
    MA = Matrix(A)
    lA = sprand(rng, n, n+10, 0.01)
    @test nnz(lA[:, n+1:n+10]) == nnz(view(lA, :, n+1:n+10))
    @testset "triangular multiply with $tr($wr)" for (tr, wr) in TRICASES
        AW = tr(wr(A))
        MAW = (@static COMPREHENSIVE ? identity : Matrix)(tr(wr(MA)))
        @test AW * B ≈ MAW * B
        @test AW * s ≈ MAW * s ≈ MAW * sd
        @test AW * A ≈ MAW * MA
        @test X * AW ≈ rmul!(copy(X), AW) ≈ X * MAW
        @test mul!(similar(X), view(X, [1, 2, 3], :), AW) ≈ X * MAW
        @test X * AW isa Matrix
        tr === identity && @test AW * AW isa wr
        # and for SparseMatrixCSCView - a view of all rows and unit range of cols
        COMPREHENSIVE || wr in (UpperTriangular, LowerTriangular) || continue # a view-backed triangle is a kernel of its own
        vAW = tr(wr(view([zero(A)+I A], :, (n+1):2n)))
        @test vAW * B ≈ AW * B
        @test vAW * A ≈ AW * A
        @test X * vAW ≈ X * MAW
    end
    a = sprand(rng, ComplexF64, n, n, 0.01)
    a[1, 1] = 2 + im # Exercise conjugation of a stored nonunit diagonal.
    ma = Matrix(a)
    ct, tc = x -> adjoint(transpose(x)), x -> transpose(adjoint(x))
    @testset "triangular multiply with conjugate matrices" for (tr, wr) in (@static COMPREHENSIVE ?
        Iterators.product((ct, tc), TRIANGLES) : (Any[ct, UpperTriangular], Any[tc, UnitLowerTriangular]))
        AW = tr(wr(a))
        MAW = (@static COMPREHENSIVE ? identity : Matrix)(tr(wr(ma)))
        @test AW * B ≈ MAW * B
        @test AW * s ≈ MAW * s ≈ MAW * sd
        @test X * AW ≈ rmul!(complex(X), AW) ≈ X * MAW
        @test X * AW isa Matrix
        # and for SparseMatrixCSCView - a view of all rows and unit range of cols
        COMPREHENSIVE || wr in (UpperTriangular, LowerTriangular) || continue # a view-backed triangle is a kernel of its own
        vAW = tr(wr(view([zero(a)+I a], :, (n+1):2n)))
        @test vAW * B ≈ AW * B
        @test X * vAW ≈ X * MAW
    end
    @static if COMPREHENSIVE
    # the implicit unit diagonal may not fit the index type (#816)
    A8 = sparse(Int8[1, 2, 5], Int8[2, 1, 7], [2.0, 3.0, 4.0], 127, 127)
    @test UnitUpperTriangular(A8) * A8 isa SparseMatrixCSC{Float64,Int8}
    @test UnitLowerTriangular(A8) * A8 ≈ Matrix(UnitLowerTriangular(A8)) * Matrix(A8)
    @test UnitUpperTriangular(A8) * spzeros(127) == zeros(127)
    end
    # vectors outside the kernel's types take the generic product
    T2 = sparse([1.0 2.0; 0.0 1.0])
    w = WrappedSparseVector(sparsevec([0.0, 3.0]))
    @test UnitUpperTriangular(T2) * w == UpperTriangular(T2) * w == [6.0, 3.0]
    A = A - Diagonal(diag(A)) + 2I # avoid rounding errors by division
    MA = Matrix(A)
    @testset "triangular solver for $tr($wr)" for (tr, wr) in TRICASES
        AW = tr(wr(A))
        MAW = (@static COMPREHENSIVE ? identity : Matrix)(tr(wr(MA)))
        @test AW \ B ≈ MAW \ B
        @test !issparse(AW \ B)
        # and for SparseMatrixCSCView - a view of all rows and unit range of cols
        COMPREHENSIVE || wr in (UpperTriangular, LowerTriangular) || continue # a view-backed triangle is a kernel of its own
        vAW = tr(wr(view([zero(A)+I A], :, (n+1):2n)))
        @test vAW \ B ≈ AW \ B
    end
    @testset "triangular singular exceptions" begin
        A = LowerTriangular(sparse([0 2.0;0 1]))
        @test_throws SingularException(1) A \ ones(2)
        A = UpperTriangular(sparse([1.0 0;0 0]))
        @test_throws SingularException(2) A \ ones(2)
    end
end

@testset "triangular sparse structural cases" begin
    structural = Any[(ComplexF64, UpperTriangular, identity), (ComplexF64, UnitLowerTriangular, identity),
                     (ComplexF64, LowerTriangular, adjoint), (ComplexF64, UnitUpperTriangular, adjoint),
                     (ComplexF64, UpperTriangular, transpose), (ComplexF64, UnitLowerTriangular, transpose)]
    @static COMPREHENSIVE && append!(structural, pairwise(STD_ELTYPES, TRIANGLES, TRANSFORMS))
    for T in (Float64, ComplexF64)
        A = sparse([1, 2, 5, 1], [1, 1, 5, 6], T[2, 3, 0, 4], 6, 6)
        T <: Complex && (nonzeros(A)[2] += im)
        b = T[1, -2, 0, 3, 0, 4]
        for W in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular),
            op in (identity, transpose, adjoint)
            iscase((T, W, op), structural) || continue
            S = op(W(A))
            D = Matrix(S)
            @static if COMPREHENSIVE
            @test S * b ≈ D * b
            end
            @test S * sparse(b) ≈ D * b
            @test S * A ≈ D * Matrix(A)
            @test S * hcat(b, 2b) ≈ D * hcat(b, 2b)
            @test transpose(hcat(b, 2b)) * S ≈ transpose(hcat(b, 2b)) * D
            @test S * spzeros(T, 6) == zeros(T, 6)
        end
    end
end

@testset "triangular kernel dispatch and shape checks" begin
    A = sparse([1, 2, 2, 3], [1, 1, 2, 3], [2.0, 1.0, 3.0, 4.0], 3, 3)
    B = ones(3, 2)
    # a sparse matrix and its adjoint reach the SparseArrays hooks
    for hook in (LinearAlgebra.generic_trimatmul!, LinearAlgebra.generic_trimatdiv!), S in (A, A')
        @test which(hook, Tuple{typeof(B), Char, Char, typeof(identity), typeof(S), typeof(B)}).module === SparseArrays
    end
    # a bad destination shape throws before anything is written
    C = fill(-1.0, 3, 3)
    @test_throws DimensionMismatch LinearAlgebra.generic_trimatmul!(C, 'U', 'N', identity, A, B)
    @test all(==(-1.0), C)
    # a vector product may write to a one-column matrix, as with a dense triangle
    for W in (LowerTriangular, UpperTriangular)
        @test mul!(zeros(3, 1), W(A), ones(3)) == mul!(zeros(3, 1), W(Matrix(A)), ones(3))
    end
end

@testset "triangular products visit stored entries" begin
    @static if COMPREHENSIVE
    for n in (8, 16), W in (UpperTriangular, LowerTriangular)
        A = opcount_sparse(sparse(1:n, 1:n, ones(n), n, n))
        @test mulcount(() -> W(A) * A) == n
        # These wrappers currently select generic triangular multiplication.
        for op in (transpose, adjoint)
            count = mulcount(() -> op(W(A)) * A)
            @test_broken count <= 2n
        end
    end
    end
    n = 1000
    A = opcount_sparse(sparse(1:n, 1:n, ones(n), n, n))
    for W in (@static COMPREHENSIVE ? TRIANGLES : (UpperTriangular, UnitLowerTriangular))
        @test mulcount(() -> W(A) * A) == n
        @test mulcount(() -> W(view(A, :, :)) * A) == n
        X = OpCount.(ones(2, n))
        for op in (identity, transpose, adjoint)
            COMPREHENSIVE || iscase((W, op), ((UpperTriangular, identity), (UnitLowerTriangular, adjoint))) || continue
            @test mulcount(() -> X * op(W(A))) == 2n
        end
    end
    @test_throws DimensionMismatch ones(2, 3) * UpperTriangular(sparse(1.0I, 4, 4))
end


@testset "multiplication of triangular sparse and dense matrices" begin
    n = 7
    B = rand(n, 3)
    _triangular_sparse_matrix(n, ULT, T) = T == Int ? ULT(sparse(rand(0:10, n, n))) : ULT(sprandn(T, n, n, 0.4))
    eltypecases = Any[(Int, adjoint, UpperTriangular), (ComplexF64, transpose, UnitLowerTriangular)]
    @static COMPREHENSIVE && append!(eltypecases, eachvalue((Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64),
        (transpose, adjoint), (LowerTriangular, UnitUpperTriangular, UnitLowerTriangular, UpperTriangular)))
    for T in (Int, Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64)
        for AT in (adjoint, transpose)
            for TR in (UpperTriangular, UnitUpperTriangular, LowerTriangular, UnitLowerTriangular)
                iscase((T, AT, TR), eltypecases) || continue
                TS = AT(_triangular_sparse_matrix(n, TR, T))
                @test isa(TS * B, DenseMatrix)
                @test TS * B ≈ Matrix(TS)*B
            end
        end
    end
end


@testset "multiplication of Triangular sparse matrices with sparse vectors #35642" begin
    n = 10
    A = sprand(n, n, 5/n)
    U = UpperTriangular(A)
    L = LowerTriangular(A)
    x = sprand(n, 5/n)
    y = view(A, :, 6)
    z = view(x, :)
    ty = typeof
    @testset "matvec multiplication $(ty(X)) * $(ty(v))" for (X, v) in (@static COMPREHENSIVE ?
        Iterators.product((U, L), (x, y, z)) : (Any[U, y], Any[L, z]))
        @test X * v ≈ Matrix(X) * Vector(v)
        @test typeof(X * v) == typeof(x)
    end
end


@testset "Triangular and SparseVector multiplications" begin
    n = 10
    types = (Int, Float64, ComplexF64)
    tritypes = (LowerTriangular, UnitUpperTriangular)
    vectorcases = Any[(Float64, Float64, LowerTriangular), (ComplexF64, ComplexF64, UnitUpperTriangular),
                      (Float64, ComplexF64, UnitUpperTriangular)]
    @static COMPREHENSIVE && append!(vectorcases, pairwise(types, types, tritypes))
    for ta in types
        for tri in tritypes
            if ta == Int
                T = tri(rand(1:9, n, n))
            else
                T = tri(randn(ta, n, n))
            end
            for tb in types
                iscase((ta, tb, tri), vectorcases) || continue
                if tb == Int
                    x = sparse(rand(0:4, n))
                else
                    x = sprandn(tb, n, 0.6)
                end
                @test T * x ≈ Array(T) * Array(x)
                COMPREHENSIVE || ta == tb || continue # promotion does not depend on the transform
                @test T' * x ≈ Array(T)' * Array(x)
                @test transpose(T) * x ≈ transpose(Array(T)) * Array(x)
                @test x' * T ≈ Array(x)' * Array(T)
                @test x' * T' ≈ Array(x)' * Array(T)'
                @test x' * transpose(T) ≈ Array(x)' * transpose(Array(T))
            end
        end
    end

    # 0-dimensional case
    x = sparse(zeros(0))
    for tri in (@static COMPREHENSIVE ? tritypes : (LowerTriangular,))
        T = tri(zeros(0, 0))
        @test T*x == Array(T) * Array(x)
        @test T' * x == Array(T)' * Array(x)
        @test transpose(T) * x == transpose(Array(T)) * Array(x)
        @test x' * T == Array(x)' * Array(T)
        @test x' * T' == Array(x)' * Array(T)'
        @test x' * transpose(T) == Array(x)' * transpose(Array(T))
    end
end


@static if COMPREHENSIVE
@testset "issue #14816" begin
    m = 5
    intmat = fill(1, m, m)
    ltintmat = LowerTriangular(rand(1:5, m, m))
    @test \(transpose(ltintmat), sparse(intmat)) ≈ \(transpose(ltintmat), intmat)
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
end

@testset "complex matrix-vector multiplication and triangular or diagonal left-division" begin
    for i = 1:(@static COMPREHENSIVE ? 5 : 1)
        @static if COMPREHENSIVE
        a = I + 0.1*sprandn(5, 5, 0.2)
        b = randn(5,3) + im*randn(5,3)
        c = randn(5) + im*randn(5)
        d = randn(5) + im*randn(5)
        α = rand(ComplexF64)
        β = rand(ComplexF64)
        @test (maximum(abs.(a*b - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(mul!(similar(b), a, b) - Array(a)*b)) < 100*eps()) # for compatibility with present matmul API. Should go away eventually.
        @test (maximum(abs.(mul!(similar(c), a, c) - Array(a)*c)) < 100*eps()) # for compatibility with present matmul API. Should go away eventually.
        @test (maximum(abs.(mul!(similar(b), transpose(a), b) - transpose(Array(a))*b)) < 100*eps()) # for compatibility with present matmul API. Should go away eventually.
        @test (maximum(abs.(mul!(similar(c), transpose(a), c) - transpose(Array(a))*c)) < 100*eps()) # for compatibility with present matmul API. Should go away eventually.
        @test (maximum(abs.(a'b - Array(a)'b)) < 100*eps())
        @test (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps())
        @test (maximum(abs.((a'*c + d) - (Array(a)'*c + d))) < 1000*eps())
        @test (maximum(abs.((α*transpose(a)*c + β*d) - (α*transpose(Array(a))*c + β*d))) < 1000*eps())
        @test (maximum(abs.((transpose(a)*c + d) - (transpose(Array(a))*c + d))) < 1000*eps())
        c = randn(6) + im*randn(6)
        @test_throws DimensionMismatch α*transpose(a)*c + β*c
        @test_throws DimensionMismatch α*transpose(a)*fill(1.,5) + β*c

        a = I + 0.1*sprandn(5, 5, 0.2) + 0.1*im*sprandn(5, 5, 0.2)
        b = randn(5,3)
        @test (maximum(abs.(a*b - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(a'b - Array(a)'b)) < 100*eps())
        @test (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps())
        end

        a = I + tril(0.1*sprandn(5, 5, 0.2))
        b = randn(5,3) + im*randn(5,3)
        @static if COMPREHENSIVE
        @test (maximum(abs.(a*b - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(a'b - Array(a)'b)) < 100*eps())
        @test (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps())
        end
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @static if COMPREHENSIVE
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        end
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())

        @static if COMPREHENSIVE
        a = I + tril(0.1*sprandn(5, 5, 0.2) + 0.1*im*sprandn(5, 5, 0.2))
        b = randn(5,3)
        @test (maximum(abs.(a*b - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(a'b - Array(a)'b)) < 100*eps())
        @test (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps())
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())

        a = I + triu(0.1*sprandn(5, 5, 0.2))
        b = randn(5,3) + im*randn(5,3)
        @test (maximum(abs.(a*b - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(a'b - Array(a)'b)) < 100*eps())
        @test (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps())
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())
        end

        a = I + triu(0.1*sprandn(5, 5, 0.2) + 0.1*im*sprandn(5, 5, 0.2))
        b = randn(5,3)
        @static if COMPREHENSIVE
        @test (maximum(abs.(a*b - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(a'b - Array(a)'b)) < 100*eps())
        @test (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps())
        end
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @static if COMPREHENSIVE
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())
        end
        # UpperTriangular/LowerTriangular solve
        a = UpperTriangular(I + triu(0.1*sprandn(5, 5, 0.2)))
        b = sprandn(5, 5, 0.2)
        @test (maximum(abs.(a\b - Array(a)\Array(b))) < 1000*eps())
        # test error throwing for bwdTrisolve
        @test_throws DimensionMismatch a\Matrix{Float64}(I, 6, 6)
        a = LowerTriangular(I + tril(0.1*sprandn(5, 5, 0.2)))
        b = sprandn(5, 5, 0.2)
        @test (maximum(abs.(a\b - Array(a)\Array(b))) < 1000*eps())
        # test error throwing for fwdTrisolve
        @test_throws DimensionMismatch a\Matrix{Float64}(I, 6, 6)

        a = sparse(Diagonal(randn(5) + im*randn(5)))
        b = randn(5,3)
        @static if COMPREHENSIVE
        @test (maximum(abs.(a*b - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(a'b - Array(a)'b)) < 100*eps())
        @test (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps())
        end
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @static if COMPREHENSIVE
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())

        b = randn(5,3) + im*randn(5,3)
        @test (maximum(abs.(a*b - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(a'b - Array(a)'b)) < 100*eps())
        @test (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps())
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())
        end
    end
end

@testset "factorize of a triangular matrix, and unsupported eigen and inv" begin
    D = sparse(Diagonal(1.0:5.0))
    A = D + sparse([1, 4], [3, 2], [0.5, -1.5], 5, 5)
    A = A*transpose(A)
    @test !isdiag(A)
    @test factorize(triu(A)) == triu(A)
    @test isa(factorize(triu(A)), UpperTriangular{Float64, SparseMatrixCSC{Float64, Int}})
    @test factorize(tril(A)) == tril(A)
    @test isa(factorize(tril(A)), LowerTriangular{Float64, SparseMatrixCSC{Float64, Int}})
    @test factorize(D) == D
    @test isa(factorize(D), Diagonal{Float64})
    @test_throws ErrorException eigen(A)
    @test_throws ErrorException inv(A)
end


# PR 28242
@testset "forward and backward solving of transpose/adjoint triangular matrices" begin
    rng = MersenneTwister(20180730)
    n = 10
    A = sprandn(rng, n, n, 0.8); A += Diagonal((1:n) - diag(A))
    B = ones(n, 2)
    for (Ttri, triul ) in ((UpperTriangular, triu), (LowerTriangular, tril))
        for trop in (adjoint, transpose)
            COMPREHENSIVE || Ttri === LowerTriangular && trop === adjoint || continue
            AT = Ttri(A)           # ...Triangular wrapped
            AC = triul(A)          # copied part of A
            ATa = trop(AT)         # wrapped Adjoint
            ACa = sparse(trop(AC)) # copied and adjoint
            @test AT \ B ≈ AC \ B
            @test ATa \ B ≈ ACa \ B
            @test ATa \ sparse(B) ≈ ATa \ B
            @test Matrix(ATa) \ B ≈ ATa \ B
            @test ATa * ( ATa \ B ) ≈ B
        end
    end
    # the accumulator of the transposed solves is seeded from the converted destination,
    # so an Int right-hand side or a real one against a complex matrix does not widen it
    Ai = sparse(1.0I, n, n) + triu(A, 1); bi = ones(Int, n)
    Ac = sparse(((1 + im) * Ai)'); bf = ones(Float64, n)
    for trop in (adjoint, transpose)
        COMPREHENSIVE || trop === adjoint || continue
        @test trop(UpperTriangular(Ai)) \ bi ≈ Matrix(trop(UpperTriangular(Ai))) \ bi
        @static if COMPREHENSIVE
        @test trop(LowerTriangular(Ac)) \ bf ≈ Matrix(trop(LowerTriangular(Ac))) \ bf
        end
        for (S, rhs, upper, T1, T2) in ((Ai, bi, Val{true}, Float64, Int), (Ac, bf, Val{false}, Float64, ComplexF64))
            COMPREHENSIVE || S === Ai || continue
            C = similar(rhs, eltype(S))
            @test !hasunionlocal(SparseArrays._trimatdiv!,
                (typeof(C), upper, Bool, typeof(trop), typeof(S), typeof(rhs)), T1, T2)
        end
    end
    @static if COMPREHENSIVE
    # Int32 indices keep the kernels type-stable
    A32 = SparseMatrixCSC{Float64,Int32}(Ai); b32 = ones(n); C32 = similar(b32)
    for kernel in (SparseArrays._trimatdiv!, SparseArrays._trimatmul!)
        @test !hasunionlocal(kernel, (typeof(C32), Val{true}, Bool, typeof(transpose), typeof(A32), typeof(b32)), Int32, Int)
    end
    end
end


@static if COMPREHENSIVE
@testset "ldiv with different element types (#40171)" begin
    sA = sparse(Int16.(1:4), Int16.(1:4), ones(4))
    @test all(ldiv!(LowerTriangular(sA), ones(4)) .≈ 1.)
end
end
@testset "ldiv ops with triangular matrices and sparse vecs (#14005)" begin
    m = 10
    sprmat = sprand(m, m, 0.2)
    sparsefloatmat = I + sprmat/(2m)
    sparsecomplexmat = I + SparseMatrixCSC(m, m, getcolptr(sprmat), rowvals(sprmat), complex.(nonzeros(sprmat), nonzeros(sprmat))/(4m))
    sparseintmat = 10m*I + SparseMatrixCSC(m, m, getcolptr(sprmat), rowvals(sprmat), round.(Int, nonzeros(sprmat)*10))

    denseintmat = I*10m + rand(1:m, m, m)
    densefloatmat = I + randn(m, m)/(2m)
    densecomplexmat = I + randn(ComplexF64, m, m)/(4m)

    inttypes = (Int64, (@static COMPREHENSIVE ? (Int32, BigInt) : ())...)
    floattypes = (Float32, Float64, BigFloat)
    complextypes = (ComplexF32, ComplexF64)
    eltypes = (inttypes..., floattypes..., complextypes...)
    coretypes = (Int64, Float64, ComplexF64)
    @static COMPREHENSIVE || (eltypes = coretypes)

    # A strided backing takes the block solve of this package and a sparse one the sparse
    # kernels; each backing meets an upper and a lower, a unit and a nonunit triangle and
    # each transform, mostly complex since real solves do not conjugate.
    boundaries = Any[(ComplexF64, true, LowerTriangular, identity), (ComplexF64, false, UpperTriangular, adjoint),
                     (ComplexF64, true, UnitUpperTriangular, transpose), (ComplexF64, false, UnitLowerTriangular, transpose),
                     (Float64, true, UpperTriangular, adjoint), (Float64, false, LowerTriangular, identity)]
    @static COMPREHENSIVE && append!(boundaries, pairwise(STD_ELTYPES, (true, false), TRIANGLES, TRANSFORMS))
    # an integer quotient, integer to float, and real to complex in both directions
    promotions = Any[(Int64, Int64, true, LowerTriangular), (Int64, Float64, false, UnitLowerTriangular),
                     (Float64, ComplexF64, false, LowerTriangular), (ComplexF64, Float64, true, UnitLowerTriangular)]
    @static COMPREHENSIVE && append!(promotions, pairwise(eltypes, eltypes, (true, false), (LowerTriangular, UnitLowerTriangular)))

    @testset "wrapper dispatch and active-index boundaries" for T in (Float64, ComplexF64)
        densemat, sparsemat = T == Float64 ? (densefloatmat, sparsefloatmat) :
                                            (densecomplexmat, sparsecomplexmat)
        z = T == Float64 ? T(2) : T(2 + 3im)
        spvecs = (spzeros(T, m),
                  SparseVector(m, [1], [z]),
                  SparseVector(m, [m], [z]),
                  SparseVector(m, [3, 7], [z, -z]),
                  SparseVector(m, [1, 3, m], [zero(T), z, zero(T)]))
        for backing in (densemat, sparsemat), tri in (LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular),
            transform in (identity, adjoint, transpose)
            iscase((T, backing isa Matrix, tri, transform), boundaries) || continue
            mat = transform(tri(backing))
            for spvec in spvecs
                check_trisolve(mat, spvec)
            end
            if backing isa Matrix
                @test which(\, Tuple{typeof(mat), typeof(first(spvecs))}).module === SparseArrays
                @test which(ldiv!, Tuple{typeof(mat), typeof(first(spvecs))}).module === SparseArrays
            end
        end
    end

    @testset "fixed right-hand side" begin
        mat = LowerTriangular(densefloatmat)
        # a pattern covering the active range 4:m is kept as it is, stored zero included
        b = fixed(SparseVector(m, collect(4:m), [4.0, 0.0, 6.0, 7.0, 8.0, 9.0, 10.0]))
        x = ldiv!(mat, copy(b))
        @test x isa FixedSparseVector && nonzeroinds(x) == 4:m && x ≈ mat \ Array(b)
        @test (mat \ b)::Vector{Float64} ≈ mat \ Array(b)
        # a gap in the active range is rejected before anything is written
        g = fixed(SparseVector(m, [3, 7], [2.0, -2.0]))
        @test_throws ArgumentError ldiv!(mat, g)
        @test nonzeroinds(g) == [3, 7] && nonzeros(g) == [2.0, -2.0]
        @test (mat \ g)::Vector{Float64} ≈ mat \ Array(g)
    end

    @testset "index type and eltype of the right-hand side" begin
        L = LowerTriangular([2.0 1 1; 1 2 1; 1 1 2])
        for Ti in (UInt64, (@static COMPREHENSIVE ? (Int128,) : ())...)
            b = SparseVector(3, Ti[2], [1.0])
            x = ldiv!(L, copy(b))
            @test nonzeroinds(x) == 2:3 && x ≈ L \ Array(b)
        end
        @static if COMPREHENSIVE
        b = SparseVector(3, [1, 3], Any[1.0, 2.0])
        x = ldiv!(L, copy(b))
        @test nonzeroinds(x) == 1:3 && x ≈ L \ [1.0, 0.0, 2.0]
        end
    end

    @testset "scalar promotion" for eltypemat in eltypes
        (densemat, sparsemat) = eltypemat in inttypes ? (denseintmat, sparseintmat) :
                                eltypemat in floattypes ? (densefloatmat, sparsefloatmat) :
                                eltypemat in complextypes && (densecomplexmat, sparsecomplexmat)
        densemat = convert(Matrix{eltypemat}, densemat)
        sparsemat = convert(SparseMatrixCSC{eltypemat}, sparsemat)
        for eltypevec in eltypes
            (eltypemat in coretypes && eltypevec in coretypes) ||
                eltypemat == Float64 || eltypevec == Float64 || continue
            vals = eltypevec <: Complex ? eltypevec[2 + 3im, -1 + 2im, 3 - im] : eltypevec[2, -1, 3]
            spvec = SparseVector(m, [2, 5, 8], vals)
            for backing in (densemat, sparsemat), tri in (LowerTriangular, UnitLowerTriangular)
                iscase((eltypemat, eltypevec, backing isa Matrix, tri), promotions) || continue
                check_trisolve(tri(backing), spvec)
            end
        end
    end
end
@static if COMPREHENSIVE
@testset "#16716" begin
    origmat = [-1.5 -0.7; 0.0 1.0]
    transmat = copy(origmat')
    utmat = UpperTriangular(origmat)
    ltmat = LowerTriangular(transmat)
    uutmat = LinearAlgebra.UnitUpperTriangular(origmat)
    ultmat = LinearAlgebra.UnitLowerTriangular(transmat)

    zerospvec = spzeros(Float64, 2)
    zerodvec = zeros(Float64, 2)

    for mat in (utmat, ltmat, uutmat, ultmat)
        @test isequal(\(mat, zerospvec), zerodvec)
        @test isequal(\(adjoint(mat), zerospvec), zerodvec)
        @test isequal(\(transpose(mat), zerospvec), zerodvec)
        @test isequal(ldiv!(mat, copy(zerospvec)), zerospvec)
        @test isequal(ldiv!(adjoint(mat), copy(zerospvec)), zerospvec)
        @test isequal(ldiv!(transpose(mat), copy(zerospvec)), zerospvec)
    end
end
end

end # module SparseTriangularTests
