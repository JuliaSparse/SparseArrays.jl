# This file is a part of Julia. License is MIT: https://julialang.org/license

# Long-tail regression tests for sparse `map` and `broadcast`. Each testset names the
# issue it guards; they run only when the `torture` suite is selected.

module TortureHigherOrderFnsTests
using Test
using SparseArrays
using LinearAlgebra
include("../testhelpers.jl")

@testset "issue #5824" begin
    @test sprand(4,5,0.5).^0 == sparse(fill(1,4,5))
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

# Check that `broadcast` methods specialized for unary operations over `SparseMatrixCSC`s
# are called. (Issue #18705.) EDIT: #19239 unified broadcast over a single sparse matrix,
# eliminating the former operation classes.
@testset "issue #18705" begin
    S = sparse(Diagonal(1.0:5.0))
    @test isa(sin.(S), SparseMatrixCSC)
end

# Check that `broadcast` methods specialized for unary operations over
# `SparseMatrixCSC`s determine a reasonable return type.
@testset "issue #18974" begin
    S = sparse(Diagonal(Int64(1):Int64(4)))
    @test eltype(sin.(S)) == Float64
end

# The eltypes of the dense-divisor (#551) and row-scaling (#543) checks that the core
# pair-broadcast testset does not run.
@testset "broadcast[!] over pairs with a dense argument, remaining eltypes (#551, #543)" begin
    # A dense argument has no structural zeros, so `f(0, 0)` (`NaN` for `/`) must not
    # densify the result, and zero quotients, `-0.0` included, are not stored (#551)
    for T in (Float64,)
        A = sparse(T[0 0; 0.5 0; 0 0])
        x = T[1, 2, -3]
        y = T[-1 2]
        for (C, R) in ((A ./ x, Array(A) ./ x), (A ./ y, Array(A) ./ y), (x .\ A, x .\ Array(A)),
                       (A ./ x[1:2]', Array(A) ./ x[1:2]'), (A ./ view(x, 1:3), Array(A) ./ x),
                       (A ./ x ./ y, Array(A) ./ x ./ y))
            @test C isa SparseMatrixCSC{T}
            @test C == R
            @test nnz(C) == 1
        end
        # zeros of the dense argument still give `Inf` and `NaN`, and only those rows fill
        x0 = T[1, 0, 2]
        C = (A .+ sparse(T[0 0; 0 0; 0 1])) ./ x0
        @test isequal(C, sparse(Array(A .+ sparse(T[0 0; 0 0; 0 1])) ./ x0))
        @test nnz(C) == 3
        D = sprand(T, 3, 2, 0.5)
        @test broadcast!(/, D, A, x) === D
        @test D == Array(A) ./ x
        @test nnz(D) == 1
    end

    # scaling rows by a vector scans the matrix's stored entries instead of merging the
    # vector against every column, so `f` is called O(nnz + m) times, not O(m * n) (#543)
    for T in (ComplexF64,)
        m, n = 40, 30
        A = sprand(T, m, n, 0.05)
        v = rand(T, m) .+ 1
        ncalls = Ref(0)
        for (f, args) in ((*, (v, A)), (*, (A, v)), (/, (A, v)), (\, (v, A)),
                           (*, (A, sparse(v))), (*, (sparse(v), A)))
            ncalls[] = 0
            counted(x, y) = (ncalls[] += 1; f(x, y))
            C = broadcast(counted, args...)
            @test ncalls[] <= nnz(A) + 2m
            @test C == broadcast(f, map(Array, args)...)
            @test nnz(C) == nnz(A)
            ncalls[] = 0
            D = sprand(T, m, n, 0.5)
            @test broadcast!(counted, D, args...) == C
            @test ncalls[] <= nnz(A) + 2m
        end
        # a zero in the vector fills its row with `NaN`, which needs the merge
        v0 = copy(v); v0[3] = 0
        @test isequal(A ./ v0, sparse(Array(A) ./ v0))
        @test isequal(v0 .\ A, sparse(v0 .\ Array(A)))
        @test isequal(view(A, :, 2:n) ./ view(v0, :), sparse(Array(A)[:, 2:n] ./ v0))
        @test isequal(view(v0, :) .\ view(A, :, 2:n), sparse(v0 .\ Array(A)[:, 2:n]))
        @test A .+ v == Array(A) .+ v
        @test v .+ A == v .+ Array(A)
        # `f` is not probed against a zero for a row the matrix stores in full
        @test sqrt.(sparse(T[2 3; 0 0]) .- T[1, 0]) == sqrt.(T[2 3; 0 0] .- T[1, 0])
        @test sqrt.(T[-1, 0] .- sparse(T[-2 -3; 0 0])) == sqrt.(T[-1, 0] .- T[-2 -3; 0 0])
    end
end

# The full `tens` x `tens` product of the second and third arguments; the core suite
# restricts the third to three representatives.
@testset "broadcast[!] implementation capable of handling >2 (input) sparse vectors/matrices" begin
    N, M, p = 10, 12, 0.3
    f(x, y, z) = x + y + z + 1
    mats = (sprand(N, M, p), sprand(N, 1, p), sprand(1, M, p), sprand(1, 1, 1.0), spzeros(1, 1))
    vecs = (sprand(N, p), sprand(1, 1.0), spzeros(1))
    tens = (mats..., vecs...)
    for Xo in tens
        X = ndims(Xo) == 1 ? SparseVector{Float32,Int32}(Xo) : SparseMatrixCSC{Float32,Int32}(Xo)
        # use different types to check internal type stability via allocation tests below
        shapeX, fX = size(X), Array(X)
        for Y in tens, Z in tens
            fY, fZ = Array(Y), Array(Z)
            # --> test broadcast entry point
            @test broadcast(+, X, Y, Z) == sparse(broadcast(+, fX, fY, fZ))
            @test broadcast(*, X, Y, Z) == sparse(broadcast(*, fX, fY, fZ))
            @test broadcast(f, X, Y, Z) == sparse(broadcast(f, fX, fY, fZ))
            # TODO strengthen this test, avoiding dependence on checking whether
            # check_broadcast_axes throws to determine whether sparse broadcast should throw
            try
                Base.Broadcast.combine_axes(spzeros((shapeX .- 1)...), Y, Z)
            catch
                @test_throws DimensionMismatch broadcast(+, spzeros((shapeX .- 1)...), Y, Z)
            end
            # --> test broadcast! entry point / +-like zero-preserving op
            fQ = broadcast(+, fX, fY, fZ); Q = sparse(fQ)
            broadcast!(+, Q, X, Y, Z); Q = sparse(fQ) # warmup for @allocated
            @test (@allocated broadcast!(+, Q, X, Y, Z)) < 500
            @test broadcast!(+, Q, X, Y, Z) == sparse(broadcast!(+, fQ, fX, fY, fZ))
            # --> test broadcast! entry point / *-like zero-preserving op
            fQ = broadcast(*, fX, fY, fZ); Q = sparse(fQ)
            broadcast!(*, Q, X, Y, Z); Q = sparse(fQ) # warmup for @allocated
            @test (@allocated broadcast!(*, Q, X, Y, Z)) < 500
            @test broadcast!(*, Q, X, Y, Z) == sparse(broadcast!(*, fQ, fX, fY, fZ))
            # --> test broadcast! entry point / not zero-preserving op
            fQ = broadcast(f, fX, fY, fZ); Q = sparse(fQ)
            broadcast!(f, Q, X, Y, Z); Q = sparse(fQ) # warmup for @allocated
            @test (@allocated broadcast!(f, Q, X, Y, Z)) < 500
            @test broadcast!(f, Q, X, Y, Z) == sparse(broadcast!(f, fQ, fX, fY, fZ))
            # --> test shape checks for both broadcast and broadcast! entry points
            # TODO strengthen this test, avoiding dependence on checking whether
            # check_broadcast_axes throws to determine whether sparse broadcast should throw
            try
                Base.Broadcast.check_broadcast_axes(axes(Q), spzeros((shapeX .- 1)...), Y, Z)
            catch
                @test_throws DimensionMismatch broadcast!(f, Q, spzeros((shapeX .- 1)...), Y, Z)
            end
        end
    end
end

# The dense-comparison bulk of the assorted two-argument broadcast tests; largely covered
# by the pair-broadcast grids.
@testset "assorted tests of sparse broadcast over two input arguments, against dense" begin
    N, p = 10, 0.3
    A, B, CF = sprand(N, N, p), sprand(N, N, p), rand(N, N)
    AF, BF, C = Array(A), Array(B), sparse(CF)

    @test A .* B == AF .* BF
    @test A[1,:] .* B == AF[1,:] .* BF
    @test A[:,1] .* B == AF[:,1] .* BF
    @test A .* B[1,:] == AF .*  BF[1,:]
    @test A .* B[:,1] == AF .*  BF[:,1]

    @test A[1,:] .* BF == AF[1,:] .* BF
    @test A[:,1] .* BF == AF[:,1] .* BF
    @test A .* BF[1,:] == AF .*  BF[1,:]
    @test A .* BF[:,1] == AF .*  BF[:,1]

    @test AF[1,:] .* B == AF[1,:] .* BF
    @test AF[:,1] .* B == AF[:,1] .* BF
    @test AF .* B[1,:] == AF .*  BF[1,:]
    @test AF .* B[:,1] == AF .*  BF[:,1]

    @test A .* 3 == AF .* 3
    @test 3 .* A == 3 .* AF
    @test A[1,:] .* 3 == AF[1,:] .* 3
    @test A[:,1] .* 3 == AF[:,1] .* 3

    @test A .- 3 == AF .- 3
    @test 3 .- A == 3 .- AF
    @test A .- B == AF .- BF
    @test A - AF == zeros(size(AF))
    @test AF - A == zeros(size(AF))
    @test A[1,:] .- B == AF[1,:] .- BF
    @test A[:,1] .- B == AF[:,1] .- BF
    @test A .- B[1,:] == AF .-  BF[1,:]
    @test A .- B[:,1] == AF .-  BF[:,1]

    @test A .+ 3 == AF .+ 3
    @test 3 .+ A == 3 .+ AF
    @test A .+ B == AF .+ BF
    @test A + AF == AF + A
    @test (A .< B) == (AF .< BF)
    @test (A .!= B) == (AF .!= BF)

    @test A ./ 3 == AF ./ 3
    @test A .\ 3 == AF .\ 3
    @test 3 ./ A == 3 ./ AF
    @test 3 .\ A == 3 .\ AF
    @test A .\ C == AF .\ CF
    @test A ./ C == AF ./ CF
    @test A ./ CF[:,1] == AF ./ CF[:,1]
    @test A .\ CF[:,1] == AF .\ CF[:,1]
    @test BF ./ C == BF ./ CF
    @test BF .\ C == BF .\ CF

    @test A .^ 3 == AF .^ 3
    @test 3 .^ A == 3 .^ AF
    @test A .^ BF[:,1] == AF .^ BF[:,1]
    @test BF[:,1] .^ A == BF[:,1] .^ AF
end

@testset "Issue #27836" begin
    @test minimum(sparse([1, 2], [1, 2], ones(Int32, 2)), dims = 1) isa Matrix
end

@testset "Sparse outer product, for type $T and vector $op" for
         op in (transpose, adjoint),
         T in (Float64,)
    m, n, p = 100, 250, 0.1
    A = sprand(T, m, n, p)
    a, b = view(A, :, 1), sprand(T, m, p)
    av, bv = Vector(a), Vector(b)
    v = @inferred a .* op(b)
    w = @inferred b .* op(a)
    @test issparse(v)
    @test issparse(w)
    @test v == av .* op(bv)
    @test w == bv .* op(av)
end

end # module
