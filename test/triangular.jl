
# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseTriangularTests

using Test
using SparseArrays
using SparseArrays: nonzeroinds, getcolptr, rowvals, nonzeros
using LinearAlgebra
using Random
include("mulcount.jl")

@testset "multiplication of sparse matrix and triangular matrix" begin
    _sparse_test_matrix(n, T) =  T == Int ? sparse(rand(0:4, n, n)) : sprandn(T, n, n, 0.6)
    _triangular_test_matrix(n, TA, T) = T == Int ? TA(rand(0:9, n, n)) : TA(randn(T, n, n))

    function test_triangular_product(S, T)
        @test (T * S)::DenseMatrix ≈ Matrix(T) * Matrix(S)
        @test (S * T)::DenseMatrix ≈ Matrix(S) * Matrix(T)
    end

    n = 5
    @testset "wrappers" begin
        for ElType in (Float64, ComplexF32)
            S = _sparse_test_matrix(n, ElType)
            for TM in (LowerTriangular, UnitLowerTriangular, UpperTriangular, UnitUpperTriangular)
                T = _triangular_test_matrix(n, TM, ElType)
                for transT in (identity, adjoint, transpose), transS in (identity, adjoint, transpose)
                    test_triangular_product(transS(S), transT(T))
                end
            end
        end
    end
    @testset "promotion" begin
        for T1 in (Int, Float64, ComplexF32), T2 in (Int, Float64, ComplexF32)
            S = _sparse_test_matrix(n, T1)
            for TM in (LowerTriangular, UpperTriangular)
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


# an AbstractSparseVector outside the types the sparse product kernel handles
struct WrappedSparseVector <: AbstractSparseVector{Float64,Int}
    x::SparseVector{Float64,Int}
end
Base.size(v::WrappedSparseVector) = size(v.x)
Base.getindex(v::WrappedSparseVector, i::Int) = v.x[i]
SparseArrays.nonzeros(v::WrappedSparseVector) = nonzeros(v.x)
SparseArrays.nonzeroinds(v::WrappedSparseVector) = nonzeroinds(v.x)

begin
    rng = Random.MersenneTwister(0)
    n = 100
    B = ones(n)
    s = sprandn(rng, n, 0.05)
    sd = Vector(s)
    A = sprand(rng, n, n, 0.01)
    MA = Matrix(A)
    lA = sprand(rng, n, n+10, 0.01)
    @test nnz(lA[:, n+1:n+10]) == nnz(view(lA, :, n+1:n+10))
    @testset "triangular multiply with $tr($wr)" for tr in (identity, adjoint, transpose),
    wr in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)
        AW = tr(wr(A))
        MAW = tr(wr(MA))
        @test AW * B ≈ MAW * B
        @test AW * s ≈ MAW * s ≈ MAW * sd
        @test AW * A ≈ MAW * MA
        tr === identity && @test AW * AW isa wr
        # and for SparseMatrixCSCView - a view of all rows and unit range of cols
        vAW = tr(wr(view([zero(A)+I A], :, (n+1):2n)))
        @test vAW * B ≈ AW * B
        @test vAW * A ≈ AW * A
    end
    a = sprand(rng, ComplexF64, n, n, 0.01)
    a[1, 1] = 2 + im # Exercise conjugation of a stored nonunit diagonal.
    ma = Matrix(a)
    @testset "triangular multiply with conjugate matrices" for tr in (x -> adjoint(transpose(x)), x -> transpose(adjoint(x))),
        wr in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)
        AW = tr(wr(a))
        MAW = tr(wr(ma))
        @test AW * B ≈ MAW * B
        @test AW * s ≈ MAW * s ≈ MAW * sd
        # and for SparseMatrixCSCView - a view of all rows and unit range of cols
        vAW = tr(wr(view([zero(a)+I a], :, (n+1):2n)))
        @test vAW * B ≈ AW * B
    end
    # the implicit unit diagonal may not fit the index type (#816)
    A8 = sparse(Int8[1, 2, 5], Int8[2, 1, 7], [2.0, 3.0, 4.0], 127, 127)
    @test UnitUpperTriangular(A8) * A8 isa SparseMatrixCSC{Float64,Int8}
    @test UnitLowerTriangular(A8) * A8 ≈ Matrix(UnitLowerTriangular(A8)) * Matrix(A8)
    @test UnitUpperTriangular(A8) * spzeros(127) == zeros(127)
    # vectors outside the kernel's types take the generic product
    T2 = sparse([1.0 2.0; 0.0 1.0])
    w = WrappedSparseVector(sparsevec([0.0, 3.0]))
    @test UnitUpperTriangular(T2) * w == UpperTriangular(T2) * w == [6.0, 3.0]
    A = A - Diagonal(diag(A)) + 2I # avoid rounding errors by division
    MA = Matrix(A)
    @testset "triangular solver for $tr($wr)" for tr in (identity, adjoint, transpose),
    wr in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)
        AW = tr(wr(A))
        MAW = tr(wr(MA))
        @test AW \ B ≈ MAW \ B
        @test !issparse(AW \ B)
        # and for SparseMatrixCSCView - a view of all rows and unit range of cols
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
    for T in (Float64, ComplexF64)
        A = sparse([1, 2, 5, 1], [1, 1, 5, 6], T[2, 3, 0, 4], 6, 6)
        T <: Complex && (nonzeros(A)[2] += im)
        b = T[1, -2, 0, 3, 0, 4]
        for W in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular),
            op in (identity, transpose, adjoint)
            S = op(W(A))
            D = Matrix(S)
            @test S * b ≈ D * b
            @test S * sparse(b) ≈ D * b
            @test S * A ≈ D * Matrix(A)
            @test S * hcat(b, 2b) ≈ D * hcat(b, 2b)
            @test S * spzeros(T, 6) == zeros(T, 6)
        end
    end
end

@testset "triangular products visit stored entries" begin
    for n in (8, 16), W in (UpperTriangular, LowerTriangular)
        A = mulcount_sparse(sparse(1:n, 1:n, ones(n), n, n))
        @test mulcount(() -> W(A) * A) == n
        # These wrappers currently select generic triangular multiplication.
        for op in (transpose, adjoint)
            count = mulcount(() -> op(W(A)) * A)
            @test_broken count <= 2n
        end
    end
    n = 1000
    A = mulcount_sparse(sparse(1:n, 1:n, ones(n), n, n))
    for W in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)
        @test mulcount(() -> W(A) * A) == n
        @test mulcount(() -> W(view(A, :, :)) * A) == n
    end
end


@testset "multiplication of triangular sparse and dense matrices" begin
    n = 7
    B = rand(n, 3)
    _triangular_sparse_matrix(n, ULT, T) = T == Int ? ULT(sparse(rand(0:10, n, n))) : ULT(sprandn(T, n, n, 0.4))
    for T in (Int, Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64)
        for AT in (adjoint, transpose)
            for TR in (UpperTriangular, UnitUpperTriangular, LowerTriangular, UnitLowerTriangular)
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
    @testset "matvec multiplication $(ty(X)) * $(ty(v))" for X in (U, L), v in (x, y, z)
        @test X * v ≈ Matrix(X) * Vector(v)
        @test typeof(X * v) == typeof(x)
    end
end


@testset "Triangular and SparseVector multiplications" begin
    n = 10
    types = (Int, Float64, ComplexF64)
    tritypes = (LowerTriangular, UnitUpperTriangular)
    for ta in types
        for tri in tritypes
            if ta == Int
                T = tri(rand(1:9, n, n))
            else
                T = tri(randn(ta, n, n))
            end
            for tb in types
                if tb == Int
                    x = sparse(rand(0:4, n))
                else
                    x = sprandn(tb, n, 0.6)
                end
                @test T * x ≈ Array(T) * Array(x)
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
    for tri in tritypes
        T = tri(zeros(0, 0))
        @test T*x == Array(T) * Array(x)
        @test T' * x == Array(T)' * Array(x)
        @test transpose(T) * x == transpose(Array(T)) * Array(x)
        @test x' * T == Array(x)' * Array(T)
        @test x' * T' == Array(x)' * Array(T)'
        @test x' * transpose(T) == Array(x)' * transpose(Array(T))
    end
end


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


# PR 28242
@testset "forward and backward solving of transpose/adjoint triangular matrices" begin
    rng = MersenneTwister(20180730)
    n = 10
    A = sprandn(rng, n, n, 0.8); A += Diagonal((1:n) - diag(A))
    B = ones(n, 2)
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


@testset "ldiv with different element types (#40171)" begin
    sA = sparse(Int16.(1:4), Int16.(1:4), ones(4))
    @test all(ldiv!(LowerTriangular(sA), ones(4)) .≈ 1.)
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

    inttypes = (Int32, Int64, BigInt)
    floattypes = (Float32, Float64, BigFloat)
    complextypes = (ComplexF32, ComplexF64)
    eltypes = (inttypes..., floattypes..., complextypes...)
    coretypes = (Int64, Float64, ComplexF64)

    function check_solve(mat, spvec)
        fspvec = Array(spvec)
        T = typeof(zero(eltype(mat))*zero(eltype(spvec)) + zero(eltype(mat))*zero(eltype(spvec)))
        if !(mat isa Union{UnitLowerTriangular,UnitUpperTriangular})
            T = typeof(zero(T)/one(eltype(mat)))
        end
        @test (mat \ spvec)::Vector{T} ≈ mat \ fspvec
        if eltype(spvec) == T
            @test ldiv!(mat, copy(spvec)) ≈ ldiv!(mat, copy(fspvec))
        end
    end

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
            mat = transform(tri(backing))
            for spvec in spvecs
                check_solve(mat, spvec)
            end
            if backing isa Matrix
                @test which(\, Tuple{typeof(mat), typeof(first(spvecs))}).module === SparseArrays
                @test which(ldiv!, Tuple{typeof(mat), typeof(first(spvecs))}).module === SparseArrays
            end
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
                check_solve(tri(backing), spvec)
            end
        end
    end
end
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

end # module SparseTriangularTests
