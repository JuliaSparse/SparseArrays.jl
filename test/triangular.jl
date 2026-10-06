
# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseTriangularTests

using Test
using SparseArrays
using SparseArrays: nonzeroinds, getcolptr, rowvals, nonzeros, fixed, FixedSparseVector
using LinearAlgebra
include("testhelpers.jl")

const TRIANGLES = (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)
const TRANSFORMS = (identity, adjoint, transpose)
# The kernels are compiled per transform and triangle, so a standard run takes the fewest
# cases that reach every branch of them. Here those are the unit triangles; the conjugate,
# dense-product and dispatch tests below have the nonunit ones. The kernels specialize on
# the transform and on upper or lower, and take the unit diagonal as a runtime flag, so a
# comprehensive run adds the cases that give every transform both an upper and a lower
# triangle and each of the four triangles. A tuple holding a transform has a type of its
# own for each transform, so the cases are vectors and `iscase`, which tells whether a
# nested loop runs the case `c`, is not specialized.
const TRICASES = (Any[transpose, UnitUpperTriangular], Any[identity, UnitLowerTriangular],
    (@static COMPREHENSIVE ? (Any[identity, UpperTriangular], Any[adjoint, LowerTriangular],
    Any[adjoint, UnitUpperTriangular], Any[transpose, LowerTriangular]) : ())...)
iscase(@nospecialize(c), cases) = any(x -> x === c, cases)

@testset "multiplication of sparse matrix and triangular matrix" begin
    _sparse_test_matrix(n, T) = fixture(T, n, n)
    _triangular_test_matrix(n, TA, T) = TA(T[T <: Complex ? complex(i - 2j, i + j) : i - 2j for i in 1:n, j in 1:n])

    function test_triangular_product(S, T)
        @test (T * S)::DenseMatrix ≈ Matrix(T) * Matrix(S)
        @test (S * T)::DenseMatrix ≈ Matrix(S) * Matrix(T)
    end

    n = 5
    wrappers = Any[(Float64, LowerTriangular, identity, transpose)]
    @static COMPREHENSIVE && append!(wrappers, Any[(ComplexF64, UnitLowerTriangular, adjoint, identity),
                   (Float64, UpperTriangular, transpose, adjoint), (ComplexF64, UnitUpperTriangular, identity, transpose)],
                   eachvalue(STD_ELTYPES, TRIANGLES, TRANSFORMS, (adjoint, transpose, identity)))
    @testset "wrappers" begin
        for ElType in (Float64, (@static COMPREHENSIVE ? (ComplexF64,) : ())...)
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
    @static if COMPREHENSIVE
    types = (Int, Float64, ComplexF32)
    # each type on either side, never with itself
    promotions = eachvalue(types, (Float64, ComplexF32, Int), (LowerTriangular, UpperTriangular))
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
end


@testset "Multiplying with triangular sparse matrices #35609 #35610" begin
    # most of the diagonal is not stored, and one diagonal entry is a stored zero
    A = fixture(Float64, 4, 4)
    U = UpperTriangular(A)
    L = LowerTriangular(A)
    AM = Matrix(A)
    UM = Matrix(U)
    LM = Matrix(L)
    Y = A * U
    @test mismatch(Y, AM * UM; approx=true) === nothing
    @test typeof(Y) == typeof(A)
    @static if COMPREHENSIVE
    Y = A * L
    @test mismatch(Y, AM * LM; approx=true) === nothing
    @test typeof(Y) == typeof(A)
    Y = U * A
    @test mismatch(Y, UM * AM; approx=true) === nothing
    @test typeof(Y) == typeof(A)
    end
    Y = L * A
    @test mismatch(Y, LM * AM; approx=true) === nothing
    @test typeof(Y) == typeof(A)
    Y = U * U
    @test mismatch(parent(Y), UM * UM; approx=true) === nothing
    @test typeof(Y) == typeof(U)
    @static if COMPREHENSIVE
    Y = L * L
    @test mismatch(parent(Y), LM * LM; approx=true) === nothing
    @test typeof(Y) == typeof(L)
    end
    Y = L * U
    @test mismatch(Y, LM * UM; approx=true) === nothing
    @test typeof(Y) == typeof(A)
    @static if COMPREHENSIVE
    Y = U * L
    @test mismatch(Y, UM * LM; approx=true) === nothing
    @test typeof(Y) == typeof(A)
    end
end


begin
    n = 10
    B = ones(n)
    X = reshape(1.0:3n, 3, n) ./ n
    s = fixturevec(Float64, n)
    sd = Vector(s)
    # scaled so that the unit triangles, whose diagonal is not stored, stay well conditioned
    A = fixture(Float64, n, n) / 4n
    MA = Matrix(A)
    lA = fixture(Float64, n, n+10)
    @test nnz(lA[:, n+1:n+10]) == nnz(view(lA, :, n+1:n+10))
    @testset "triangular multiply with $tr($wr)" for (tr, wr) in TRICASES
        AW = tr(wr(A))
        MAW = Matrix(tr(wr(MA)))
        @test AW * B ≈ MAW * B
        @static if COMPREHENSIVE
        @test AW * s ≈ MAW * s ≈ MAW * sd
        @test AW * A ≈ MAW * MA
        end
        @test X * AW ≈ rmul!(copy(X), AW) ≈ X * MAW
        @static if COMPREHENSIVE
        @test mul!(similar(X), view(X, [1, 2, 3], :), AW) ≈ X * MAW
        end
        @test X * AW isa Matrix
        tr === identity && @test AW * AW isa wr
        # and for SparseMatrixCSCView - a view of all rows and unit range of cols
        COMPREHENSIVE || continue # a view-backed triangle is a kernel of its own
        vAW = tr(wr(view([zero(A)+I A], :, (n+1):2n)))
        @test vAW * B ≈ AW * B
        @test vAW * A ≈ AW * A
        @test X * vAW ≈ X * MAW
    end
    a = fixture(ComplexF64, n, n) / 4n
    a[1, 1] = 2 + im # Exercise conjugation of a stored nonunit diagonal.
    ma = Matrix(a)
    ct, tc = x -> adjoint(transpose(x)), x -> transpose(adjoint(x))
    # both orders of the wrappers reach the same conjugating kernel, one for an upper triangle
    # and one for a lower
    @testset "triangular multiply with conjugate matrices" for (tr, wr) in (Any[ct, UpperTriangular],
        (@static COMPREHENSIVE ? (Any[tc, UnitLowerTriangular],) : ())...)
        AW = tr(wr(a))
        MAW = (@static COMPREHENSIVE ? identity : Matrix)(tr(wr(ma)))
        @test AW * B ≈ MAW * B
        @static if COMPREHENSIVE
        @test AW * s ≈ MAW * s ≈ MAW * sd
        end
        @test X * AW ≈ rmul!(complex(X), AW) ≈ X * MAW
        @test X * AW isa Matrix
        # and for SparseMatrixCSCView - a view of all rows and unit range of cols
        COMPREHENSIVE || continue # a view-backed triangle is a kernel of its own
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
        MAW = Matrix(tr(wr(MA)))
        @test AW \ B ≈ MAW \ B
        @test !issparse(AW \ B)
        # and for SparseMatrixCSCView - a view of all rows and unit range of cols
        COMPREHENSIVE || continue # a view-backed triangle is a kernel of its own
        vAW = tr(wr(view([zero(A)+I A], :, (n+1):2n)))
        @test vAW \ B ≈ AW \ B
    end
    @testset "triangular singular exceptions" begin
        A = LowerTriangular(sparse([0 2.0;0 1]))
        @test_throws SingularException(1) A \ ones(2)
        A = UpperTriangular(sparse([1.0 0;0 0]))
        @test_throws SingularException(2) A \ ones(2)
        # a column with stored entries but no stored diagonal: singular for a nonunit
        # triangle, and for a unit one the last off-diagonal entry is not the diagonal
        Au, Al = sparse([1, 1], [1, 2], [1.0, 2.0], 2, 2), sparse([2, 2], [1, 2], [2.0, 1.0], 2, 2)
        @test_throws SingularException(2) UpperTriangular(Au) \ ones(2)
        @test_throws SingularException(1) LowerTriangular(Al) \ ones(2)
        @test UnitUpperTriangular(Au) \ ones(2) == [-1.0, 1.0]
        @test UnitLowerTriangular(Al) \ ones(2) == [1.0, -1.0]
    end
    # a transpose of an adjoint is a conjugate, which the solve applies to the stored values
    ad = copy(a)
    for i in 1:n
        ad[i, i] = 2 + im
    end
    @testset "triangular solver for conjugate matrices" for (tr, wr) in (Any[ct, UpperTriangular],
        (@static COMPREHENSIVE ? (Any[tc, UnitLowerTriangular],) : ())...)
        AW = tr(wr(ad))
        @test AW \ B ≈ Matrix(AW) \ B
    end
    @static if COMPREHENSIVE
    # A wrapper applied twice by its constructor is the matrix itself, not its conjugate.
    for (W, wr) in eachvalue((Adjoint, Transpose), (UpperTriangular, LowerTriangular))
        AW = wr(W(W(ad)))
        @test AW \ B ≈ Matrix(AW) \ B
        @test AW * B ≈ Matrix(AW) * B
        @test X * AW ≈ X * Matrix(AW)
    end
    end
end

@static if COMPREHENSIVE
@testset "triangular sparse structural cases" begin
    # every transform with an upper and a lower triangle; a real matrix takes the same branches
    structural = Any[(ComplexF64, UpperTriangular, identity), (ComplexF64, UnitLowerTriangular, identity),
                     (ComplexF64, LowerTriangular, adjoint), (ComplexF64, UnitUpperTriangular, adjoint),
                     (ComplexF64, UpperTriangular, transpose), (ComplexF64, UnitLowerTriangular, transpose),
                     (Float64, LowerTriangular, adjoint)]
    for T in (Float64, ComplexF64)
        A = sparse([1, 2, 5, 1], [1, 1, 5, 6], T[2, 3, 0, 4], 6, 6)
        T <: Complex && (nonzeros(A)[2] += im)
        b = T[1, -2, 0, 3, 0, 4]
        for W in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular),
            op in (identity, transpose, adjoint)
            iscase((T, W, op), structural) || continue
            S = op(W(A))
            D = Matrix(S)
            @test S * b ≈ D * b
            @test S * sparse(b) ≈ D * b
            @test S * A ≈ D * Matrix(A)
            @test S * hcat(b, 2b) ≈ D * hcat(b, 2b)
            @test transpose(hcat(b, 2b)) * S ≈ transpose(hcat(b, 2b)) * D
            @test S * spzeros(T, 6) == zeros(T, 6)
        end
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
        for op in (transpose, adjoint)
            @test mulcount(() -> op(W(A)) * A) <= 2n
            @test mulcount(() -> A * op(W(A))) <= 2n
        end
    end
    end
    n = 1000
    A = opcount_sparse(sparse(1:n, 1:n, ones(n), n, n))
    for W in (UpperTriangular, (@static COMPREHENSIVE ? (UnitLowerTriangular,) : ())...)
        @test mulcount(() -> W(A) * A) == n
        @static if COMPREHENSIVE
        @test mulcount(() -> W(view(A, :, :)) * A) == n
        end
        X = OpCount.(ones(2, n))
        for op in (identity, transpose, adjoint)
            COMPREHENSIVE || op === identity || continue
            @test mulcount(() -> X * op(W(A))) == 2n
        end
    end
    @test_throws DimensionMismatch ones(2, 3) * UpperTriangular(sparse(1.0I, 4, 4))
end

@static if COMPREHENSIVE
@testset "products of an adjoint or transpose sparse triangular matrix" begin
    for T in (Float64, ComplexF64)
        S = sparse(T[1 2 0 1; 3 4 5 0; 0 6 7 8; 2 0 9 3])
        B = sparse(T[0 1 2 0; 1 0 0 3; 4 0 1 0; 0 2 0 1])
        T <: Complex && (S += im * B; B = B + im * S)
        H = Hermitian(B + B')
        x = B[:, 2]
        for W in TRIANGLES, op in (transpose, adjoint, a -> transpose(adjoint(a))), M in (S, view(S, :, 1:4))
            L, D = op(W(M)), op(W(Matrix(M)))
            for (R, DR) in ((B, Matrix(B)), (B', Matrix(B)'), (H, Matrix(H)), (W(B), W(Matrix(B))), (L, D))
                @test L * R ≈ D * DR
                @test R * L ≈ DR * D
                @test L * R isa Union{SparseMatrixCSC,SparseArrays.SparseTriangular}
                @test R * L isa Union{SparseMatrixCSC,SparseArrays.SparseTriangular}
            end
            @test L * x ≈ D * Vector(x)
            @test L * x isa SparseVector
        end
    end
end
end


@testset "multiplication of triangular sparse and dense matrices" begin
    n = 7
    B = reshape(1.0:3n, n, 3) ./ n
    # `+ I` stores the diagonal, which the nonunit branch of the product kernel needs
    _triangular_sparse_matrix(n, ULT, T) = ULT(fixture(T, n, n) + I)
    eltypecases = Any[(ComplexF64, adjoint, LowerTriangular)]
    @static COMPREHENSIVE && append!(eltypecases,
        Any[(Int, adjoint, UpperTriangular)],
        eachvalue((Float32, Float64, ComplexF64),
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


@static if COMPREHENSIVE
@testset "multiplication of Triangular sparse matrices with sparse vectors #35642" begin
    n = 10
    A = fixture(Float64, n, n)
    U = UpperTriangular(A)
    L = LowerTriangular(A)
    x = fixturevec(Float64, n)
    y = view(A, :, 6)
    z = view(x, :)
    ty = typeof
    @testset "matvec multiplication $(ty(X)) * $(ty(v))" for X in (U, L), v in (x, y, z)
        @test X * v ≈ Matrix(X) * Vector(v)
        @test typeof(X * v) == typeof(x)
    end
end
end


@testset "Triangular and SparseVector multiplications" begin
    n = 10
    types = (Int, Float64, ComplexF64)
    tritypes = (LowerTriangular, UnitUpperTriangular)
    vectorcases = Any[(ComplexF64, ComplexF64, UnitUpperTriangular)]
    @static COMPREHENSIVE && append!(vectorcases, Any[(Float64, Float64, LowerTriangular)],
                      eachvalue(types, (Float64, ComplexF64, Int), tritypes))
    for ta in (@static COMPREHENSIVE ? types : (ComplexF64,))
        for tri in (@static COMPREHENSIVE ? tritypes : (UnitUpperTriangular,))
            if ta == Int
                T = tri([mod1(i + 2j, 9) for i in 1:n, j in 1:n])
            else
                T = tri(ta[ta <: Complex ? complex(i - 2j, i + j) : i - 2j for i in 1:n, j in 1:n])
            end
            for tb in types
                iscase((ta, tb, tri), vectorcases) || continue
                if tb == Int
                    x = fixturevec(Int, n)
                else
                    x = fixturevec(tb, n)
                end
                @test T * x ≈ Array(T) * Array(x)
                COMPREHENSIVE || ta == tb || continue # promotion does not depend on the transform
                @test T' * x ≈ Array(T)' * Array(x)
                @test transpose(T) * x ≈ transpose(Array(T)) * Array(x)
                @test mismatch((x' * T)', (Array(x)' * Array(T))'; approx=true) === nothing
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
        @static if COMPREHENSIVE
        @test transpose(T) * x == transpose(Array(T)) * Array(x)
        @test x' * T == Array(x)' * Array(T)
        @test x' * T' == Array(x)' * Array(T)'
        @test x' * transpose(T) == Array(x)' * transpose(Array(T))
        end
    end
end


@static if COMPREHENSIVE
@testset "issue #14816" begin
    m = 5
    intmat = fill(1, m, m)
    ltintmat = LowerTriangular([mod1(i + 2j, 5) for i in 1:m, j in 1:m])
    @test \(transpose(ltintmat), sparse(intmat)) ≈ \(transpose(ltintmat), intmat)
end


@testset "issue #13792, use sparse triangular solvers for sparse triangular solves" begin
    local A, n, x
    n = 100
    # diagonally dominant, so both triangles are well conditioned
    A, b = fixture(Float64, n, n) / 4n^2 + sqrt(n)*I, fill(1., n)
    @test LowerTriangular(A)\(LowerTriangular(A)*b) ≈ b
    @test UpperTriangular(A)\(UpperTriangular(A)*b) ≈ b
    A[2,2] = 0
    dropzeros!(A)
    @test_throws LinearAlgebra.SingularException LowerTriangular(A)\b
    @test_throws LinearAlgebra.SingularException UpperTriangular(A)\b
end

@testset "complex matrix-vector multiplication and triangular or diagonal left-division" begin
    # the comparisons have absolute tolerances, so the operands are scaled to order one
    let
        a = I + 0.02*fixture(Float64, 5, 5)
        b = reshape(complex.(1:15, 15:-1:1), 5, 3) / 15
        c = complex.(1:5, 5:-1:1) / 5
        d = complex.(5:-1:1, 1:5) / 5
        α = 0.3 + 0.7im
        β = 0.6 - 0.2im
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
        c = complex.(1:6, 6:-1:1) / 6
        @test_throws DimensionMismatch α*transpose(a)*c + β*c
        @test_throws DimensionMismatch α*transpose(a)*fill(1.,5) + β*c

        a = I + 0.02*fixture(ComplexF64, 5, 5)
        b = reshape(1.0:15.0, 5, 3) / 15
        @test (maximum(abs.(a*b - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(a'b - Array(a)'b)) < 100*eps())
        @test (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps())

        a = I + tril(0.02*fixture(Float64, 5, 5))
        b = reshape(complex.(1:15, 15:-1:1), 5, 3) / 15
        @test (maximum(abs.(a*b - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(a'b - Array(a)'b)) < 100*eps())
        @test (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps())
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())

        a = I + tril(0.02*fixture(ComplexF64, 5, 5))
        b = reshape(1.0:15.0, 5, 3) / 15
        @test (maximum(abs.(a*b - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(a'b - Array(a)'b)) < 100*eps())
        @test (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps())
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())

        a = I + triu(0.02*fixture(Float64, 5, 5))
        b = reshape(complex.(1:15, 15:-1:1), 5, 3) / 15
        @test (maximum(abs.(a*b - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(a'b - Array(a)'b)) < 100*eps())
        @test (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps())
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())

        a = I + triu(0.02*fixture(ComplexF64, 5, 5))
        b = reshape(1.0:15.0, 5, 3) / 15
        @test (maximum(abs.(a*b - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(a'b - Array(a)'b)) < 100*eps())
        @test (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps())
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())
        # UpperTriangular/LowerTriangular solve
        a = UpperTriangular(I + triu(0.02*fixture(Float64, 5, 5)))
        b = 0.05*permutedims(fixture(Float64, 5, 5))
        @test (maximum(abs.(a\b - Array(a)\Array(b))) < 1000*eps())
        # test error throwing for bwdTrisolve
        @test_throws DimensionMismatch a\Matrix{Float64}(I, 6, 6)
        a = LowerTriangular(I + tril(0.02*fixture(Float64, 5, 5)))
        b = 0.05*permutedims(fixture(Float64, 5, 5))
        @test (maximum(abs.(a\b - Array(a)\Array(b))) < 1000*eps())
        # test error throwing for fwdTrisolve
        @test_throws DimensionMismatch a\Matrix{Float64}(I, 6, 6)

        a = sparse(Diagonal(complex.(1:5, 5:-1:1) / 5))
        b = reshape(1.0:15.0, 5, 3) / 15
        @test (maximum(abs.(a*b - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(a'b - Array(a)'b)) < 100*eps())
        @test (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps())
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())

        b = reshape(complex.(1:15, 15:-1:1), 5, 3) / 15
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
    @static if COMPREHENSIVE
    @test !isdiag(A)
    @test factorize(triu(A)) == triu(A)
    @test isa(factorize(triu(A)), UpperTriangular{Float64, SparseMatrixCSC{Float64, Int}})
    @test factorize(tril(A)) == tril(A)
    @test isa(factorize(tril(A)), LowerTriangular{Float64, SparseMatrixCSC{Float64, Int}})
    @test factorize(D) == D
    @test isa(factorize(D), Diagonal{Float64})
    end
    @test_throws ErrorException eigen(A)
    @test_throws ErrorException inv(A)
end


# PR 28242
@testset "forward and backward solving of transpose/adjoint triangular matrices" begin
    n = 10
    A = fixture(Float64, n, n) / 4n; A += Diagonal((1:n) - diag(A))
    B = ones(n, 2)
    @static if COMPREHENSIVE
    for (Ttri, triul ) in ((UpperTriangular, triu), (LowerTriangular, tril))
        for trop in (adjoint, transpose)
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
    sprmat = fixture(Float64, m, m) / 4m
    sparsefloatmat = I + sprmat/(2m)
    sparsecomplexmat = I + SparseMatrixCSC(m, m, getcolptr(sprmat), rowvals(sprmat), complex.(nonzeros(sprmat), nonzeros(sprmat))/(4m))
    @static if COMPREHENSIVE
    sparseintmat = 10m*I + SparseMatrixCSC(m, m, getcolptr(sprmat), rowvals(sprmat), round.(Int, nonzeros(sprmat)*10))
    end

    denseintmat = I*10m + [mod1(i + 3j, m) for i in 1:m, j in 1:m]
    densefloatmat = I + [(i - 2j) / m for i in 1:m, j in 1:m]/(2m)
    densecomplexmat = I + [complex(i - 2j, i + j) / m for i in 1:m, j in 1:m]/(4m)

    inttypes = (Int64, (@static COMPREHENSIVE ? (Int32,) : ())...)
    floattypes = (Float32, Float64, BigFloat)
    complextypes = (ComplexF32, ComplexF64)
    eltypes = (inttypes..., floattypes..., complextypes...)

    # A strided backing takes the block solve of this package and a sparse one the sparse
    # kernels and the generic destination, which differs for a unit triangle. The standard
    # cases are complex, since real solves do not conjugate, and the strided one is a lower
    # triangle of a transpose, which the block solve unwraps.
    boundaries = Any[(ComplexF64, false, UpperTriangular, adjoint),
                     (ComplexF64, true, UnitUpperTriangular, transpose), (ComplexF64, false, UnitLowerTriangular, transpose)]
    @static COMPREHENSIVE && append!(boundaries, Any[(ComplexF64, true, LowerTriangular, identity),
                     (Float64, true, UpperTriangular, adjoint), (Float64, false, LowerTriangular, identity),
                     (ComplexF64, true, UnitLowerTriangular, adjoint), (ComplexF64, false, UnitUpperTriangular, identity),
                     (ComplexF64, false, UpperTriangular, transpose), (ComplexF64, false, LowerTriangular, adjoint)])
    # an integer quotient, integer to float, and real to complex in both directions
    promotions = Any[(Int64, Int64, true, LowerTriangular), (Int64, Float64, false, UnitLowerTriangular),
                     (Float64, ComplexF64, false, LowerTriangular), (ComplexF64, Float64, true, UnitLowerTriangular)]
    # the remaining pairs of standard types, an integer unit solve across integer types, and
    # Float64 on either side of an Int32, a single-precision and a non-IEEE type
    @static COMPREHENSIVE && append!(promotions, Any[
        (Float64, Int64, true, UnitLowerTriangular), (Int64, ComplexF64, false, LowerTriangular),
        (ComplexF64, Int64, true, LowerTriangular), (Int32, Int64, false, UnitLowerTriangular),
        (Float64, Int32, true, LowerTriangular), (Float32, Float64, true, LowerTriangular),
        (Float64, Float32, false, UnitLowerTriangular), (BigFloat, Float64, false, LowerTriangular),
        (Float64, BigFloat, true, UnitLowerTriangular)])

    @testset "wrapper dispatch and active-index boundaries" for T in ((@static COMPREHENSIVE ? (Float64,) : ())..., ComplexF64)
        densemat, sparsemat = T == Float64 ? (densefloatmat, sparsefloatmat) :
                                            (densecomplexmat, sparsecomplexmat)
        z = T == Float64 ? T(2) : T(2 + 3im)
        spvecs = (spzeros(T, m),
                  (@static COMPREHENSIVE ? (SparseVector(m, [1], [z]),
                  SparseVector(m, [m], [z])) : ())...,
                  SparseVector(m, [3, 7], [z, -z]),
                  (@static COMPREHENSIVE ? (SparseVector(m, [1, 3, m], [zero(T), z, zero(T)]),) : ())...)
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

    # a unit triangle does not divide, so an integer solve stays integer
    @test (UnitUpperTriangular(sparse([1 2; 0 1])) \ sparsevec([1, 2]))::SparseVector{Int,Int} == [-3, 2]

    @static if COMPREHENSIVE
    @testset "index type and eltype of the right-hand side" begin
        L = LowerTriangular([2.0 1 1; 1 2 1; 1 1 2])
        for Ti in (UInt64,)
            b = SparseVector(3, Ti[2], [1.0])
            x = ldiv!(L, copy(b))
            @test nonzeroinds(x) == 2:3 && x ≈ L \ Array(b)
        end
        b = SparseVector(3, [1, 3], Any[1.0, 2.0])
        x = ldiv!(L, copy(b))
        @test nonzeroinds(x) == 1:3 && x ≈ L \ [1.0, 0.0, 2.0]
    end

    @testset "scalar promotion" for eltypemat in eltypes
        (densemat, sparsemat) = eltypemat in inttypes ? (denseintmat, sparseintmat) :
                                eltypemat in floattypes ? (densefloatmat, sparsefloatmat) :
                                eltypemat in complextypes && (densecomplexmat, sparsecomplexmat)
        densemat = convert(Matrix{eltypemat}, densemat)
        sparsemat = convert(SparseMatrixCSC{eltypemat}, sparsemat)
        for eltypevec in eltypes
            vals = eltypevec <: Complex ? eltypevec[2 + 3im, -1 + 2im, 3 - im] : eltypevec[2, -1, 3]
            spvec = SparseVector(m, [2, 5, 8], vals)
            for backing in (densemat, sparsemat), tri in (LowerTriangular, UnitLowerTriangular)
                iscase((eltypemat, eltypevec, backing isa Matrix, tri), promotions) || continue
                check_trisolve(tri(backing), spvec)
            end
        end
    end
    end
end
@static if COMPREHENSIVE
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
end

end # module SparseTriangularTests
