# This file is a part of Julia. License is MIT: https://julialang.org/license

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
