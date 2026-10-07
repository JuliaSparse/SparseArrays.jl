# This file is a part of Julia. License is MIT: https://julialang.org/license

# These tests cover the higher order functions specialized for sparse arrays defined in
# base/sparse/higherorderfns.jl, particularly map[!]/broadcast[!] for SparseVectors and
# SparseMatrixCSCs at present.

module HigherOrderFnsTests

using Test
using SparseArrays
using SparseArrays: getcolptr, nonzeroinds
using LinearAlgebra
include("testhelpers.jl")
# A standard run maps and broadcasts over `Float64`/`Int` arrays only. A comprehensive run
# repeats one case of each kernel with an argument of another element and index type: the
# kernels are generic in both, and every further retyped shape compiles them again.
const retypes = @static COMPREHENSIVE ? (identity,
    X -> ndims(X) == 1 ? SparseVector{Float32,Int32}(X) : SparseMatrixCSC{Float32,Int32}(X)) : (identity,)
function test_map_and_map!(A, alloc_tests)
    # --> test map entry point
    fA = Array(A)
    shapeA = size(A)
    @test mismatch(map(sin, A), map(sin, fA)) === nothing
    @test mismatch(map(cos, A), map(cos, fA)) === nothing
    # --> test map! entry point
    fX = copy(fA); X = similar(A)
    map!(sin, X, A); X = similar(A) # warmup for @allocated
    map!(sin, X, A); X = similar(A) # warmup for @allocated
    if @allocated(map!(sin, X, A)) != 0 && alloc_tests
        println(stderr, "A in test_map_and_map! is ", A)
    end
    X = similar(A)
    @test @allocated(map!(sin, X, A)) == 0 || !alloc_tests
    @test mismatch(map!(sin, X, A), map!(sin, fX, fA)) === nothing
    @test mismatch(map!(cos, X, A), map!(cos, fX, fA)) === nothing
    @test_throws DimensionMismatch map!(sin, X, spzeros((shapeA .- 1)...))
end
@testset "map[!] implementation specialized for a single (input) sparse vector/matrix" begin
    N, M = 10, 12
    for shapeA in ((N,), (N, M))
        A = length(shapeA) == 1 ? fixturevec(Float64, N) : fixture(Float64, N, M)
        nonzeros(A)[sin.(nonzeros(A)) .== 0] .= .0
        nonzeros(A)[cos.(nonzeros(A)) .== 0] .= .0
        A = dropzeros(A)
        test_map_and_map!(A, true)
    end
    @static if COMPREHENSIVE
    # https://github.com/JuliaLang/julia/issues/37819
    Z = spzeros(Float64, Int32, 50000, 50000)
    @test isa(-Z, SparseMatrixCSC{Float64, Int32})
    end
end

@testset "map[!] implementation specialized for a pair of (input) sparse vectors/matrices" begin
    N, M = 10, 12
    f(x, y) = x + y + 1
    for shapeA in ((N,), (N, M)), retype in retypes
        # `+` of two vectors has a method of its own, so the matrices take the other type
        retype === identity || length(shapeA) == 2 || continue
        # the two patterns differ, and the second stored entry of the vectors cancels
        A, Bo = length(shapeA) == 1 ? (sparsevec([2, 3, 6, 10], [-5.0, 2.0, 0.5, 4.0], N), fixturevec(Float64, N)) :
                                      (permutedims(fixture(Float64, M, N)), fixture(Float64, N, M))
        B = retype(Bo)
        # use different types to check internal type stability via allocation tests below
        fA, fB = map(Array, (A, B))
        # --> test map entry point
        @test mismatch(map(+, A, B), map(+, fA, fB); Ti=Int) === nothing
        @static if COMPREHENSIVE
        # the index type is promoted over all the arguments, not taken from the first
        @test mismatch(map(+, B, A), map(+, fB, fA); Ti=Int) === nothing
        end
        @static if COMPREHENSIVE
        @test mismatch(map(*, A, B), map(*, fA, fB)) === nothing
        end
        @test mismatch(map(f, A, B), map(f, fA, fB)) === nothing
        retype === identity && @test_throws DimensionMismatch map(+, A, spzeros((shapeA .- 1)...))
        # --> test map! entry point
        fX = map(+, fA, fB); X = sparse(fX)
        map!(+, X, A, B); X = sparse(fX) # warmup for @allocated
        @test (@allocated map!(+, X, A, B)) < 500
        @test mismatch(map!(+, X, A, B), map!(+, fX, fA, fB)) === nothing
        @static if COMPREHENSIVE
        fX = map(*, fA, fB); X = sparse(fX)
        map!(*, X, A, B); X = sparse(fX) # warmup for @allocated
        @test (@allocated map!(*, X, A, B)) < 500
        @test mismatch(map!(*, X, A, B), map!(*, fX, fA, fB)) === nothing
        end
        @test mismatch(map!(f, X, A, B), map!(f, fX, fA, fB)) === nothing
        retype === identity && @test_throws DimensionMismatch map!(f, X, A, spzeros((shapeA .- 1)...))
    end
    @static if COMPREHENSIVE
    # https://github.com/JuliaLang/julia/issues/37819
    Z = spzeros(Float64, Int32, 50000, 50000)
    @test isa(Z + Z, SparseMatrixCSC{Float64, Int32})
    end
end

@static if COMPREHENSIVE
@testset "map[!] implementation capable of handling >2 (input) sparse vectors/matrices" begin
    N, M = 10, 12
    f(x, y, z) = x + y + z + 1
    for (shapeA, retype) in eachvalue(((N, M), (N,)), retypes)
        # three different patterns, with entries that only one, only two and all three store
        A, B, Co = length(shapeA) == 1 ?
            (fixturevec(Float64, N), sparsevec([2, 3, 6, 10], [-5.0, 2.0, 0.5, 4.0], N), sparsevec([1, 2, 6, 9], [1.0, 3.0, -0.5, 2.0], N)) :
            (fixture(Float64, N, M), permutedims(fixture(Float64, M, N)), sparse([1, 3, 3, 9, 10], [2, 2, 7, 5, 12], [1.5, -2.0, 0.5, 3.0, 4.0], N, M))
        C = retype(Co)
        # use different types to check internal type stability via allocation tests below
        fA, fB, fC = map(Array, (A, B, C))
        # --> test map entry point
        @test mismatch(map(+, A, B, C), map(+, fA, fB, fC); Ti=Int) === nothing
        @test mismatch(map(+, C, A, B), map(+, fC, fA, fB); Ti=Int) === nothing
        @static if COMPREHENSIVE
        @test mismatch(map(*, A, B, C), map(*, fA, fB, fC)) === nothing
        end
        @test mismatch(map(f, A, B, C), map(f, fA, fB, fC)) === nothing
        retype === identity && @test_throws DimensionMismatch map(+, A, B, spzeros(N, M - 1))
        # --> test map! entry point
        fX = map(+, fA, fB, fC); X = sparse(fX)
        map!(+, X, A, B, C); X = sparse(fX) # warmup for @allocated
        @test (@allocated map!(+, X, A, B, C)) < 500
        @test mismatch(map!(+, X, A, B, C), map!(+, fX, fA, fB, fC)) === nothing
        @static if COMPREHENSIVE
        fX = map(*, fA, fB, fC); X = sparse(fX)
        map!(*, X, A, B, C); X = sparse(fX) # warmup for @allocated
        @test (@allocated map!(*, X, A, B, C)) < 500
        @test mismatch(map!(*, X, A, B, C), map!(*, fX, fA, fB, fC)) === nothing
        end
        @test mismatch(map!(f, X, A, B, C), map!(f, fX, fA, fB, fC)) === nothing
        retype === identity && @test_throws DimensionMismatch map!(f, X, A, B, spzeros((shapeA .- 1)...))
    end
end
end

@testset "broadcast! implementation specialized for solely an output sparse vector/matrix (no inputs)" begin
    N, M = 10, 12
    V, C = fixturevec(Float64, N), fixture(Float64, N, M)
    fV, fC = Array(V), Array(C)
    @test mismatch(broadcast!(() -> 0, V), broadcast!(() -> 0, fV)) === nothing
    @test mismatch(broadcast!(() -> 0, C), broadcast!(() -> 0, fC)) === nothing
    @static if COMPREHENSIVE
    @test let z = 0, fz = 0; broadcast!(() -> z += 1, V) == broadcast!(() -> fz += 1, fV); end
    end
    @test let z = 0, fz = 0; broadcast!(() -> z += 1, C) == broadcast!(() -> fz += 1, fC); end
end

@testset "broadcast implementation specialized for a single (input) sparse vector/matrix" begin
    # broadcast for a single (input) sparse vector/matrix falls back to map, tested
    # extensively above. here we simply lightly exercise the relevant broadcast entry
    # point.
    N, M = 10, 12
    a, A = fixturevec(Float64, N), fixture(Float64, N, M)
    fa, fA = Array(a), Array(A)
    @test mismatch(broadcast(sin, a), broadcast(sin, fa)) === nothing
    @test mismatch(broadcast(sin, A), broadcast(sin, fA)) === nothing
    # also test the typed broadcast
    @test mismatch(broadcast(convert, Float32, A), broadcast(convert, Float32, fA)) === nothing
end

@testset "broadcast! implementation specialized for a single (input) sparse vector/matrix" begin
    N, M = 10, 12
    f(x, y) = x + y + 1
    # A stored zero of the one input stays stored, and the destinations below are sized from
    # the dense result, so the allocation bounds need inputs without one.
    mats = map(dropzeros, (fixture(Float64, N, M), (@static COMPREHENSIVE ? (fixture(Float64, N, 1),) : ())..., fixture(Float64, 1, M), (@static COMPREHENSIVE ? (sparse(fill(0.5, 1, 1)), spzeros(1, 1)) : ())...))
    vecs = map(dropzeros, (fixturevec(Float64, N), sparse([0.5]), (@static COMPREHENSIVE ? (spzeros(1),) : ())...))
    # --> test with matrix destination (Z/fZ)
    fZ = Array(first(mats))
    for Xo in (mats..., vecs...), retype in retypes
        retype === identity || Xo === first(mats) || continue
        X = retype(Xo)
        shapeX, fX = size(X), Array(X)
        # --> test broadcast! entry point / zero-preserving op
        broadcast!(sin, fZ, fX); Z = sparse(fZ)
        broadcast!(sin, Z, X); Z = sparse(fZ) # warmup for @allocated
        @test (@allocated broadcast!(sin, Z, X)) < 500
        @test mismatch(broadcast!(sin, Z, X), broadcast!(sin, fZ, fX)) === nothing
        # --> test broadcast! entry point / not-zero-preserving op
        broadcast!(cos, fZ, fX); Z = sparse(fZ)
        broadcast!(cos, Z, X); Z = sparse(fZ) # warmup for @allocated
        @test (@allocated broadcast!(cos, Z, X)) < 500
        @test mismatch(broadcast!(cos, Z, X), broadcast!(cos, fZ, fX)) === nothing
        # --> test shape checks for broadcast! entry point
        # TODO strengthen this test, avoiding dependence on checking whether
        # check_broadcast_axes throws to determine whether sparse broadcast should throw
        retype === identity && try
            Base.Broadcast.check_broadcast_axes(axes(Z), spzeros((shapeX .- 1)...))
        catch
            @test_throws DimensionMismatch broadcast!(sin, Z, spzeros((shapeX .- 1)...))
        end
    end
    # --> test with vector destination (V/fV)
    fV = Array(first(vecs))
    for Xo in vecs, retype in retypes # vector target
        retype === identity || Xo === first(vecs) || continue
        X = retype(Xo)
        shapeX, fX = size(X), Array(X)
        # --> test broadcast! entry point / zero-preserving op
        broadcast!(sin, fV, fX); V = sparse(fV)
        broadcast!(sin, V, X); V = sparse(fV) # warmup for @allocated
        @test (@allocated broadcast!(sin, V, X)) < 500
        @test mismatch(broadcast!(sin, V, X), broadcast!(sin, fV, fX)) === nothing
        # --> test broadcast! entry point / not-zero-preserving
        broadcast!(cos, fV, fX); V = sparse(fV)
        broadcast!(cos, V, X); V = sparse(fV) # warmup for @allocated
        @test (@allocated broadcast!(cos, V, X)) < 500
        @test mismatch(broadcast!(cos, V, X), broadcast!(cos, fV, fX)) === nothing
        # --> test shape checks for broadcast! entry point
        # TODO strengthen this test, avoiding dependence on checking whether
        # check_broadcast_axes throws to determine whether sparse broadcast should throw
        retype === identity && try
            Base.Broadcast.check_broadcast_axes(axes(V), spzeros((shapeX .- 1)...))
        catch
            @test_throws DimensionMismatch broadcast!(sin, V, spzeros((shapeX .- 1)...))
        end
    end
    # Tests specific to #19895, i.e. for broadcast!(identity, C, A) specializations
    Z = copy(first(mats)); fZ = Array(Z)
    V = copy(first(vecs)); fV = Array(V)
    for X in (mats..., vecs...)
        @test mismatch(broadcast!(identity, Z, X), broadcast!(identity, fZ, Array(X))) === nothing
        X isa SparseVector && @test mismatch(broadcast!(identity, V, X), broadcast!(identity, fV, Array(X))) === nothing
    end
end

@testset "map[!] and broadcast[!] over one sparse array keep its pattern (issue #454)" begin
    for A in (sparse([1, 1, 2, 3], [1, 2, 3, 2], [0, 2, 3, 4], 3, 3),                      # stored zero
              sparse([1, 1, 2, 3], [1, 2, 3, 2], [0.0im, 2.0, 3.0im, 0.5], 3, 3),
              sparsevec([1, 3, 5], [0, 2, 3], 6),
              sparsevec([1, 3, 5], [0.0im, 2.0, 3.0im], 6))[@static COMPREHENSIVE ? [1, 2, 4] : [2, 4]]
        fA = Array(A)
        for f in (eltype(A) <: Complex ? (identity, x -> 0 * x) :
                  (x -> 2x, x -> abs(x) > 1))                                             # every f has f(0) == 0
            C = f.(A)
            @test C == f.(fA) && same_pattern(C, A)
            @test same_pattern(map(f, A), A)
            @test same_pattern(map!(f, similar(A, Base.promote_op(f, eltype(A))), A), A)
        end
        @static if COMPREHENSIVE
        @test same_pattern(Float64.(real(A)), A) && same_pattern(2 .* A, A) && same_pattern(A .* 0, A)
        end
        @test nnz(A .- A) == 0                  # cancellation between two arrays is still dropped
    end
    # a stored zero in a row expands into a densely stored column, an empty column stays empty
    r = sparse([1, 1], [1, 3], [0, 2], 1, 3)
    C = broadcast!(x -> 2x, spzeros(Int, 2, 3), r)
    @test C == 2 .* repeat(Array(r), 2, 1) && getcolptr(C) == [1, 3, 3, 5]
    # an n×1 matrix broadcast into an empty vector grows the destination
    c = sparse([1, 3], [1, 1], [0.0, 2.0], 4, 1)
    for f in (x -> 2x, (@static COMPREHENSIVE ? (zero,) : ())...)
        y = broadcast!(f, spzeros(4), c)
        @test y == f.(vec(Array(c))) && nonzeroinds(y) == [1, 3]
    end
    @test broadcast!(+, spzeros(4), c, c) == [0, 0, 4, 0]
    # map over wrappers and views keeps the stored entries, as over the parent
    M = sparse([1, 1, 2], [1, 2, 2], [1.0im, 0.0, 2.0], 3, 3)
    x = sparsevec([1, 2], [0.0, 3.0], 4)
    for (W, P) in ((transpose(M), copy(transpose(M))), (M', copy(M')), (view(x, :), x),
                   (view(x, 1:3), x[1:3]), (view(M, :, 2), M[:, 2]))[@static COMPREHENSIVE ? [1, 2, 4, 5] : [2, 4]]
        # doubling shows a wrapper's conjugation in the values
        for f in (x -> 2x,)
            C = map(f, W)
            @test mismatch(C, map(f, Array(W))) === nothing && same_pattern(C, P)
        end
    end
    # Mapping a wrapper must not copy its elements or break their aliases (#892).
    for op in (transpose, adjoint), D in ([1 2; 3 4], [1im 2; 3 4im])
        (op === transpose && eltype(D) <: Real) || (@static COMPREHENSIVE && op === adjoint && eltype(D) <: Complex) || continue
        B = CountedReads(D)
        S = SparseMatrixCSC(1, 1, [1, 2], [1], [B])
        @test only(nonzeros(map(parent, op(S)))) === B
        @test B.reads[] == 0
        @test map(sum, op(S)) == map(sum, op(Array(S)))
    end
end

@testset "broadcast[!] implementation specialized for pairs of (input) sparse vectors/matrices" begin
    N, M = 10, 12
    f(x, y) = x + y + 1
    # the matrix, the row and the vector each have a stored zero and unstored entries
    mats = (fixture(Float64, N, M), (@static COMPREHENSIVE ? (fixture(Float64, N, 1),) : ())..., fixture(Float64, 1, M), (@static COMPREHENSIVE ? (sparse(fill(0.5, 1, 1)), spzeros(1, 1)) : ())...)
    vecs = (fixturevec(Float64, N), sparse([0.5]), (@static COMPREHENSIVE ? (spzeros(1),) : ())...)
    tens = (mats..., vecs...)
    fZ = Array(first(mats))
    for Xo in tens, retype in retypes
        X = retype(Xo)
        # use different types to check internal type stability via allocation tests below
        shapeX, fX = size(X), Array(X)
        for Y in tens
            # an argument of another type meets one matrix and vector pair, in each order
            retype === identity || (Xo, Y) === (mats[1], vecs[1]) || (Xo, Y) === (vecs[1], mats[1]) || continue
            fY = Array(Y)
            # --> test broadcast entry point
            @test mismatch(broadcast(+, X, Y), broadcast(+, fX, fY)) === nothing
            @static if COMPREHENSIVE
            @test mismatch(broadcast(*, X, Y), broadcast(*, fX, fY)) === nothing
            end
            # not zero-preserving: dense, like the dense result
            @test broadcast(f, X, Y)::typeof(broadcast(f, fX, fY)) == broadcast(f, fX, fY)
            # TODO strengthen this test, avoiding dependence on checking whether
            # check_broadcast_axes throws to determine whether sparse broadcast should throw
            retype === identity && try
                Base.Broadcast.combine_axes(spzeros((shapeX .- 1)...), Y)
            catch
                @test_throws DimensionMismatch broadcast(+, spzeros((shapeX .- 1)...), Y)
            end
            # --> test broadcast! entry point / +-like zero-preserving op
            broadcast!(+, fZ, fX, fY); Z = sparse(fZ)
            broadcast!(+, Z, X, Y); Z = sparse(fZ) # warmup for @allocated
            @test (@allocated broadcast!(+, Z, X, Y)) < 500
            @test mismatch(broadcast!(+, Z, X, Y), broadcast!(+, fZ, fX, fY)) === nothing
            # a zero sum of stored entries is not stored
            @test nnz(broadcast(+, X, Y)) == count(!iszero, broadcast(+, fX, fY)) && nnz(Z) == count(!iszero, fZ)
            @static if COMPREHENSIVE
            # --> test broadcast! entry point / *-like zero-preserving op
            broadcast!(*, fZ, fX, fY); Z = sparse(fZ)
            broadcast!(*, Z, X, Y); Z = sparse(fZ) # warmup for @allocated
            @test (@allocated broadcast!(*, Z, X, Y)) < 500
            @test mismatch(broadcast!(*, Z, X, Y), broadcast!(*, fZ, fX, fY)) === nothing
            end
            # --> test broadcast! entry point / not zero-preserving op
            broadcast!(f, fZ, fX, fY); Z = sparse(fZ)
            broadcast!(f, Z, X, Y); Z = sparse(fZ) # warmup for @allocated
            @test (@allocated broadcast!(f, Z, X, Y)) < 500
            @test mismatch(broadcast!(f, Z, X, Y), broadcast!(f, fZ, fX, fY)) === nothing
            # --> test shape checks for both broadcast and broadcast! entry points
            # TODO strengthen this test, avoiding dependence on checking whether
            # check_broadcast_axes throws to determine whether sparse broadcast should throw
            retype === identity && try
                Base.Broadcast.check_broadcast_axes(axes(Z), spzeros((shapeX .- 1)...), Y)
            catch
                @test_throws DimensionMismatch broadcast!(f, Z, spzeros((shapeX .- 1)...), Y)
            end
        end
    end

    @static if COMPREHENSIVE
    # fix#23857
    @test mismatch(sparse([1; 0]) ./ [1], [1.0; 0.0]) === nothing
    @test isequal(sparse([1 2; 1 0]) ./ [1; 0], sparse([1.0 2; Inf NaN]))
    @test mismatch(sparse([1  0]) ./ [1], [1.0 0.0]) === nothing
    @test isequal(sparse([1 2; 1 0]) ./ [1 0], sparse([1.0 Inf; 1 NaN]))

    # 0 \ 0 is NaN, so the quotient of two sparse arrays is dense
    @test (sparse([1]) .\ sparse([1; 0]))::Vector{Float64} == [1.0; 0.0]
    @test isequal((sparse([1; 0]) .\ sparse([1 2; 1 0]))::Matrix{Float64}, [1.0 2; Inf NaN])
    @test (sparse([1]) .\ sparse([1  0]))::Matrix{Float64} == [1.0 0.0]
    @test isequal((sparse([1 0]) .\ sparse([1 2; 1 0]))::Matrix{Float64}, [1.0 Inf; 1 NaN])
    end

    # A dense argument has no structural zeros, so `f(0, 0)` (`NaN` for `/`) must not
    # densify the result, and zero quotients, `-0.0` included, are not stored (#551)
    for T in (Float64,)
        A = sparse(T[0 0; 0.5 0; 0 0])
        x = T[1, 2, -3]
        y = T[-1 2]
        for (C, R) in ((A ./ x, Array(A) ./ x), (A ./ y, Array(A) ./ y), (x .\ A, x .\ Array(A)),
                       (@static COMPREHENSIVE ? (
                       (A ./ x[1:2]', Array(A) ./ x[1:2]'), (A ./ view(x, 1:3), Array(A) ./ x),
                       (A ./ x ./ y, Array(A) ./ x ./ y)) : ())...)
            @test C isa SparseMatrixCSC{T}
            @test C == R
            @test nnz(C) == 1
        end
        # zeros of the dense argument still give `Inf` and `NaN`, and only those rows fill
        x0 = T[1, 0, 2]
        C = (A .+ sparse(T[0 0; 0 0; 0 1])) ./ x0
        @test isequal(C, sparse(Array(A .+ sparse(T[0 0; 0 0; 0 1])) ./ x0))
        @test nnz(C) == 3
        D = fixture(T, 3, 2)
        @test broadcast!(/, D, A, x) === D
        @test D == Array(A) ./ x
        @test nnz(D) == 1
    end
    # `f(0, 0)` is not evaluated when it is not needed
    fthrows(a, b) = iszero(a) && iszero(b) ? error("f(0, 0) evaluated") : a * b
    @test broadcast(fthrows, sparse([0 1.0; 2.0 0]), [1.0, 2.0]) == [0 1.0; 4.0 0]

    # scaling rows by a vector scans the matrix's stored entries instead of merging the
    # vector against every column, so `f` is called O(nnz + m) times, not O(m * n) (#543)
    for T in (Float64,)
        m, n = 40, 30
        # 60 entries in the odd rows: the bound below needs a matrix much sparser than m * n
        A = sparse([mod1(2k + 1, m) for k in 1:60], [mod1(7k, n) for k in 1:60], collect(T, 1:60), m, n)
        v = collect(T, 2:m+1)
        ncalls = Ref(0)
        for (f, args) in ((*, (v, A)), (*, (A, v)), (/, (A, v)), (\, (v, A)),
                           (*, (A, sparse(v))), (*, (sparse(v), A)))[@static COMPREHENSIVE ? (1:6) : [1, 3, 5]]
            ncalls[] = 0
            counted(x, y) = (ncalls[] += 1; f(x, y))
            C = broadcast(counted, args...)
            @test ncalls[] <= nnz(A) + 2m
            @test C == broadcast(f, map(Array, args)...)
            @test nnz(C) == nnz(A)
            ncalls[] = 0
            D = fixture(T, m, n)
            @test broadcast!(counted, D, args...) == C
            @test ncalls[] <= nnz(A) + 2m
        end
        # a zero in the vector fills its row with `NaN`, which needs the merge
        v0 = copy(v); v0[3] = 0
        @test isequal(A ./ v0, sparse(Array(A) ./ v0))
        # as does a zero in a view of a dense vector, and the other rows are not filled
        @test isequal(A ./ view(v0, 1:m), sparse(Array(A) ./ v0)) && nnz(A ./ view(v0, 1:m)) == count(!iszero, Array(A) ./ v0)
        @static if COMPREHENSIVE
        @test isequal(v0 .\ A, sparse(v0 .\ Array(A)))
        @test isequal(view(A, :, 2:n) ./ view(v0, :), sparse(Array(A)[:, 2:n] ./ v0))
        @test isequal(view(v0, :) .\ view(A, :, 2:n), sparse(v0 .\ Array(A)[:, 2:n]))
        end
        @test A .+ v == Array(A) .+ v
        @static if COMPREHENSIVE
        @test v .+ A == v .+ Array(A)
        end
        # `f` is not probed against a zero for a row the matrix stores in full
        @test sqrt.(sparse(T[2 3; 0 0]) .- T[1, 0]) == sqrt.(T[2 3; 0 0] .- T[1, 0])
        @static if COMPREHENSIVE
        @test sqrt.(T[-1, 0] .- sparse(T[-2 -3; 0 0])) == sqrt.(T[-1, 0] .- T[-2 -3; 0 0])
        end
    end
    # nor is a zero constructed, which does not exist for `Any`
    @test sparse(Any[1 3; 2 4]) .* [2, 3] == [2 6; 6 12]
    @static if COMPREHENSIVE
    @test [2, 3] .* sparse(Any[1 3; 2 4]) == [2 6; 6 12]
    end

end


@testset "broadcast[!] implementation capable of handling >2 (input) sparse vectors/matrices" begin
    N, M = 10, 12
    f(x, y, z) = x + y + z + 1
    mats = (fixture(Float64, N, M), (@static COMPREHENSIVE ? (fixture(Float64, N, 1),) : ())..., fixture(Float64, 1, M), (@static COMPREHENSIVE ? (sparse(fill(0.5, 1, 1)), spzeros(1, 1)) : ())...)
    vecs = (fixturevec(Float64, N), sparse([0.5]), (@static COMPREHENSIVE ? (spzeros(1),) : ())...)
    tens = (mats..., vecs...)
    # Each vector/matrix mix of the three arguments is its own specialization of one generic
    # kernel. A standard run takes the all-vector mix, the only one with a vector result, and
    # a vector between two matrices. An argument of another type takes two more mixes, with a
    # matrix and a vector in each position: the shapes do not depend on the type.
    mixes = ((1, 1, 1), (2, 1, 2))
    triples = @static COMPREHENSIVE ? Any[(mats[1], vecs[1], vecs[1]), (vecs[1], mats[1], mats[1])] : ()
    for Xo in tens, retype in retypes
        X = retype(Xo)
        # use different types to check internal type stability via allocation tests below
        shapeX, fX = size(X), Array(X)
        for Y in tens, Z in tens
            (retype === identity ? (ndims(X), ndims(Y), ndims(Z)) in mixes : any(t -> t === (Xo, Y, Z), triples)) || continue
            # the shape checks do not depend on the type of X, so a mix runs them once
            shapecheck = retype === identity || !((ndims(X), ndims(Y), ndims(Z)) in mixes)
            fY, fZ = Array(Y), Array(Z)
            # --> test broadcast entry point
            @test mismatch(broadcast(+, X, Y, Z), broadcast(+, fX, fY, fZ)) === nothing
            @static if COMPREHENSIVE
            @test mismatch(broadcast(*, X, Y, Z), broadcast(*, fX, fY, fZ)) === nothing
            end
            @test broadcast(f, X, Y, Z)::typeof(broadcast(f, fX, fY, fZ)) == broadcast(f, fX, fY, fZ)
            # TODO strengthen this test, avoiding dependence on checking whether
            # check_broadcast_axes throws to determine whether sparse broadcast should throw
            shapecheck && try
                Base.Broadcast.combine_axes(spzeros((shapeX .- 1)...), Y, Z)
            catch
                @test_throws DimensionMismatch broadcast(+, spzeros((shapeX .- 1)...), Y, Z)
            end
            # --> test broadcast! entry point / +-like zero-preserving op
            fQ = broadcast(+, fX, fY, fZ); Q = sparse(fQ)
            broadcast!(+, Q, X, Y, Z); Q = sparse(fQ) # warmup for @allocated
            @test (@allocated broadcast!(+, Q, X, Y, Z)) < 500
            @test mismatch(broadcast!(+, Q, X, Y, Z), broadcast!(+, fQ, fX, fY, fZ)) === nothing
            @static if COMPREHENSIVE
            # --> test broadcast! entry point / *-like zero-preserving op
            fQ = broadcast(*, fX, fY, fZ); Q = sparse(fQ)
            broadcast!(*, Q, X, Y, Z); Q = sparse(fQ) # warmup for @allocated
            @test (@allocated broadcast!(*, Q, X, Y, Z)) < 500
            @test mismatch(broadcast!(*, Q, X, Y, Z), broadcast!(*, fQ, fX, fY, fZ)) === nothing
            end
            # --> test broadcast! entry point / not zero-preserving op
            fQ = broadcast(f, fX, fY, fZ); Q = sparse(fQ)
            broadcast!(f, Q, X, Y, Z); Q = sparse(fQ) # warmup for @allocated
            @test (@allocated broadcast!(f, Q, X, Y, Z)) < 500
            @test mismatch(broadcast!(f, Q, X, Y, Z), broadcast!(f, fQ, fX, fY, fZ)) === nothing
            # --> test shape checks for both broadcast and broadcast! entry points
            # TODO strengthen this test, avoiding dependence on checking whether
            # check_broadcast_axes throws to determine whether sparse broadcast should throw
            shapecheck && try
                Base.Broadcast.check_broadcast_axes(axes(Q), spzeros((shapeX .- 1)...), Y, Z)
            catch
                @test_throws DimensionMismatch broadcast!(f, Q, spzeros((shapeX .- 1)...), Y, Z)
            end
        end
    end
end

@static if COMPREHENSIVE
@testset "sparse map/broadcast with result eltype not a concrete subtype of Number (#19561/#19589)" begin
    N = 4
    A, fA = sparse(1.0I, N, N), Matrix(1.0I, N, N)
    B, fB = spzeros(1, N), zeros(1, N)
    intorfloat_zeropres(xs...) = all(iszero, xs) ? zero(Float64) : Int(1)
    intorfloat_notzeropres(xs...) = all(iszero, xs) ? Int(1) : zero(Float64)
    for fn in (intorfloat_zeropres, intorfloat_notzeropres)
        @test mismatch(map(fn, A), map(fn, fA); Tv=Real) === nothing
        # the broadcast that is not zero-preserving is dense, and takes the dense eltype
        check = fn === intorfloat_zeropres ? (S, D) -> mismatch(S, D; Tv=Real) === nothing :
            (S, D) -> typeof(S) === typeof(D) && S == D
        @test check(broadcast(fn, A), broadcast(fn, fA))
        @test check(broadcast(fn, A, B), broadcast(fn, fA, fB))
        @test check(broadcast(fn, B, A), broadcast(fn, fB, fA))
    end
    for fn in (intorfloat_zeropres,)
        @test mismatch(broadcast(fn, A, B, A), broadcast(fn, fA, fB, fA); Tv=Real) === nothing
    end
end
end

@static if COMPREHENSIVE
@testset "broadcast[!] over combinations of scalars and sparse vectors/matrices" begin
    N, M = 10, 12
    elT = Float64
    s = Float32(2.0)
    # A stored zero times a scalar stays stored, and `check_scalar_broadcast` sizes its
    # destination from the dense result, so its allocation bound needs inputs without one.
    V = dropzeros(fixturevec(elT, N))
    Vᵀ = transpose(dropzeros(fixture(elT, 1, N)))
    A = dropzeros(fixture(elT, N, M))
    Aᵀ = transpose(dropzeros(fixture(elT, M, N)))
    ordered(xs...) = foldl((x, y) -> 2x + y, xs)

    @testset "array forms and argument counts" begin
        # every argument tuple is a specialization of its own, so each array form and each
        # argument count appears once, with one of the two functions
        for (f, args) in ((*, (s, A)), (ordered, (s, V)), (*, (s, Aᵀ)), (*, (s, Vᵀ)),
                          (ordered, (s, A, V)), (*, (s, A, V, Aᵀ)))
            check_scalar_broadcast(f, args)
        end
    end
    @testset "scalar positions" begin
        t, u = Float32(3), Float32(5)
        # a scalar after the arrays, two between them, and three apart
        for args in ((A, V, s), (A, s, t, V), (s, A, t, V, u))
            check_scalar_broadcast(ordered, args)
        end
    end
    # test combinations at the limit of inference (eight arguments net)
    for args in ((s, V, s, A, s, V, s, A),
                 (V, A, V, A, s, V, A, V))
        check_scalar_broadcast(*, args, 900)
    end
end
end

@testset "broadcast[!] over combinations of scalars, sparse arrays, structured matrices, and dense vectors/matrices" begin
    N = 10
    s = 0.3
    V = fixturevec(Float64, N)
    A = fixture(Float64, N, N)
    Z = copy(A)
    sparsearrays = (V, A)
    fV, fA = map(Array, sparsearrays)
    D = Diagonal(collect(Float64, 1:N))
    B = Bidiagonal(collect(Float64, 1:N), collect(Float64, 2:N), :U)
    T = Tridiagonal(collect(Float64, 2:N), collect(Float64, 1:N), -collect(Float64, 2:N))
    S = SymTridiagonal(collect(Float64, 1:N), collect(Float64, 2:N))
    structuredarrays = (D, B, T, S)
    fstructuredarrays = map(Array, structuredarrays)
    # each structured type once on each side, rather than every pair of them
    partner = @static COMPREHENSIVE ? IdDict{Any,Any}(D => B, B => T, T => S, S => D) : nothing
    for (X, fX) in zip(structuredarrays, fstructuredarrays)
        (@static COMPREHENSIVE || X === D) && @test (Q = broadcast(+, V, A, X); Q isa SparseMatrixCSC && Q == sparse(broadcast(+, fV, fA, fX)))
        (@static COMPREHENSIVE || X === B) && @test mismatch(broadcast!(+, Z, V, A, X), broadcast(+, fV, fA, fX)) === nothing
        (@static COMPREHENSIVE || X === T) && @test (Q = broadcast(*, s, V, A, X); Q isa SparseMatrixCSC && Q == sparse(broadcast(*, s, fV, fA, fX)))
        (@static COMPREHENSIVE || X === S) && @test mismatch(broadcast!(*, Z, s, V, A, X), broadcast(*, s, fV, fA, fX)) === nothing
        for (Y, fY) in zip(structuredarrays, fstructuredarrays)
            (@static COMPREHENSIVE ? partner[X] === Y : (X === D && Y === B)) && @test mismatch(broadcast!(+, Z, X, Y), broadcast(+, fX, fY)) === nothing
            (@static COMPREHENSIVE ? partner[X] === Y : (X === T && Y === S)) && @test mismatch(broadcast!(*, Z, X, Y), broadcast(*, fX, fY)) === nothing
        end
    end
    C = reverse(Vector(fixturevec(Float64, N)))
    M = Matrix(permutedims(fixture(Float64, N, N)))
    densearrays = (C, M)
    fD, fB = Array(D), Array(B)
    for X in densearrays
        (@static COMPREHENSIVE || X === C) && @test mismatch(broadcast!(+, Z, D, X), broadcast(+, fD, X)) === nothing
        (@static COMPREHENSIVE || X === M) && @test mismatch(broadcast!(*, Z, s, B, X), broadcast(*, s, fB, X)) === nothing
        @static if COMPREHENSIVE
        @test broadcast(+, V, B, X)::Matrix == broadcast(+, fV, fB, X)
        @test mismatch(broadcast!(+, Z, V, B, X), broadcast(+, fV, fB, X)) === nothing
        end
        (@static COMPREHENSIVE || X === C) && @test broadcast(+, V, A, X)::Matrix == broadcast(+, fV, fA, X)
        @static if COMPREHENSIVE
        @test mismatch(broadcast!(+, Z, V, A, X), broadcast(+, fV, fA, X)) === nothing
        end
        (@static COMPREHENSIVE || X === M) && @test mismatch(broadcast(*, s, V, A, X)::SparseMatrixCSC, broadcast(*, s, fV, fA, X)) === nothing
        @static if COMPREHENSIVE
        @test mismatch(broadcast!(*, Z, s, V, A, X), broadcast(*, s, fV, fA, X)) === nothing
        end
        # Issue #20954 combinations of sparse arrays and Adjoint/Transpose vectors
        if X isa Vector
            @test broadcast(+, A, X')::Matrix == broadcast(+, fA, X')
            @static if COMPREHENSIVE
            @test mismatch(broadcast(*, V, X')::SparseMatrixCSC, broadcast(*, fV, X')) === nothing
            end
        end
    end
    @test V .+ ntuple(identity, N) isa Vector
    @static if COMPREHENSIVE
    @test A .+ ntuple(identity, N) isa Matrix
    end
end

@testset "broadcast[!] over views of dense and sparse arrays (#508)" begin
    N = 10
    V = fixturevec(Float64, N)
    A = fixture(Float64, N, N)
    Z = copy(A)
    C = collect(Float64, 1:N)
    M = reshape(collect(Float64, 1:N*N), N, N)
    # views of dense arrays count as dense: a sum with one is dense, any other broadcast sparse
    for X in (view(M, :, :), view(M, 1:N, 1:N), view(M, collect(1:N), :), view(M, :, :)')[@static COMPREHENSIVE ? (2:4) : [2, 4]]
        fX = Array(X)
        (@static COMPREHENSIVE || !(X isa Adjoint)) && @test broadcast(+, A, X)::Matrix == broadcast(+, Array(A), fX)
        (@static COMPREHENSIVE || X isa Adjoint) && @test mismatch(broadcast(*, A, X)::SparseMatrixCSC, broadcast(*, Array(A), fX)) === nothing
        (@static COMPREHENSIVE || !(X isa Adjoint)) && @test mismatch(broadcast!(*, Z, A, X), broadcast(*, Array(A), fX)) === nothing
        # the structural zeros of A must be preserved by a zero-preserving op
        (@static COMPREHENSIVE || X isa Adjoint) && @test nnz(broadcast(*, A, X)) <= nnz(A)
    end
    for x in (view(C, :), view(C, 1:N), view(C, collect(1:N)))[@static COMPREHENSIVE ? (2:3) : (2:2)]
        @test mismatch(broadcast(*, V, x)::SparseVector, broadcast(*, Array(V), Array(x))) === nothing
        @test nnz(broadcast(*, V, x)) <= nnz(V)
    end
    # views of sparse arrays likewise
    S = permutedims(fixture(Float64, 2N, N))   # its second column has stored entries
    @test broadcast(*, A, view(S, :, 1:N))::SparseMatrixCSC ==
        sparse(broadcast(*, Array(A), Array(S[:, 1:N])))
    # and on their own, sparse views of whole columns give a sparse result
    x = sparsevec([1, 3, 5], [0.0im, 2.0, 3.0im], 6)
    for (X, T) in ((view(S, :, :), SparseMatrixCSC), (view(S, :, 2:N), SparseMatrixCSC),
                   (view(S, :, [1, 3]), SparseMatrixCSC), (view(S, :, 2), SparseVector),
                   (view(x, :), SparseVector), (view(x, 2:5), SparseVector))[@static COMPREHENSIVE ? [2, 3, 4, 6] : [2, 6]]
        fX = Array(X)
        @test (2 .* X)::T == 2 .* fX && nnz(2 .* X) <= nnz(copy(X))
        @test (X .^ 0)::T == fX .^ 0
        @test cos.(X)::Array == cos.(fX)
        @test (X .+ 1)::Array == fX .+ 1
        @static if COMPREHENSIVE
        @test (X .* fX)::T == fX .* fX
        end
        @test (X .+ copy(X))::T == 2 .* fX
        @static if COMPREHENSIVE
        @test isequal((X ./ fX)::T, sparse(fX ./ fX))
        end
    end
    # adjoints and transposes of sparse vector views convert through a sparse vector
    for v in (view(x, :), view(x, 2:5), view(S, :, 2))[@static COMPREHENSIVE ? (2:3) : (2:2)], f in (adjoint, transpose)
        # the column of a matrix takes one of the two wrappers: both convert it the same way
        (@static COMPREHENSIVE && parent(v) === S && f === transpose) && continue
        fv = Array(v)
        (@static COMPREHENSIVE || f === adjoint) && @test (v .+ f(v))::SparseMatrixCSC == fv .+ f(fv)
        @static if COMPREHENSIVE
        @test (f(v) .+ v)::SparseMatrixCSC == f(fv) .+ fv
        @test (v .* f(v))::SparseMatrixCSC == fv .* f(fv)
        end
        (@static COMPREHENSIVE || f === transpose) && @test (f(v) .* v)::SparseMatrixCSC == f(fv) .* fv
        @static if COMPREHENSIVE
        @test (f(v) .+ sparse(fv))::SparseMatrixCSC == f(fv) .+ fv
        end
        (@static COMPREHENSIVE || f === adjoint) && @test SparseMatrixCSC(f(v))::SparseMatrixCSC == f(fv)
    end
    # sparse views are converted with `copy`, which keeps their stored entries in O(nnz)
    @test nnz(SparseArrays.HigherOrderFns._sparsifystructured(view(x, :))) == 3
    @test nnz(SparseArrays.HigherOrderFns._sparsifystructured(view(sparse([1, 1], [1, 2], [0.0, 1.0]), :, 1:2))) == 2
    # a view of an unsupported array still diverts to generic dense broadcast
    @test broadcast(*, A, view(PermutedDimsArray(M, (2, 1)), :, :)) isa Matrix
end

@testset "map[!] over combinations of sparse and structured matrices" begin
    N = 10
    A = fixture(Float64, N, N)
    Z, fA = copy(A), Array(A)
    D = Diagonal(collect(Float64, 1:N))
    B = Bidiagonal(collect(Float64, 1:N), collect(Float64, 2:N), :U)
    T = Tridiagonal(collect(Float64, 2:N), collect(Float64, 1:N), -collect(Float64, 2:N))
    S = SymTridiagonal(collect(Float64, 1:N), collect(Float64, 2:N))
    structuredarrays = (D, B, T, S)
    fstructuredarrays = map(Array, structuredarrays)
    # each structured type once on each side, rather than every pair of them
    partner = @static COMPREHENSIVE ? IdDict{Any,Any}(D => T, T => S, S => B, B => D) : nothing
    for (X, fX) in zip(structuredarrays, fstructuredarrays)
        (@static COMPREHENSIVE || X === D) && @test mismatch(map!(sin, Z, X), map(sin, fX)) === nothing
        (@static COMPREHENSIVE || X === B) && @test mismatch(map!(cos, Z, X), map(cos, fX)) === nothing
        (@static COMPREHENSIVE || X === T) && @test (Q = map(+, A, X); Q isa SparseMatrixCSC && Q == sparse(map(+, fA, fX)))
        (@static COMPREHENSIVE || X === S) && @test mismatch(map!(+, Z, A, X), map(+, fA, fX)) === nothing
        for (Y, fY) in zip(structuredarrays, fstructuredarrays)
            @static if COMPREHENSIVE
            partner[X] === Y || continue
            @test mismatch(map!(+, Z, X, Y), map(+, fX, fY)) === nothing
            end
            (@static COMPREHENSIVE || (X === D && Y === T)) && @test mismatch(map!(*, Z, X, Y), map(*, fX, fY)) === nothing
            (@static COMPREHENSIVE || (X === S && Y === B)) && @test (Q = map(+, X, A, Y); Q isa SparseMatrixCSC && Q == sparse(map(+, fX, fA, fY)))
            (@static COMPREHENSIVE || (X === B && Y === D)) && @test mismatch(map!(+, Z, X, A, Y), map(+, fX, fA, fY)) === nothing
        end
    end
end

# Older tests of sparse broadcast, now largely covered by the tests above
@testset "assorted tests of sparse broadcast over two input arguments" begin
    N = 10
    # `CF` has no zero, so that no quotient below is `NaN`
    A, B, CF = fixture(Float64, N, N), permutedims(fixture(Float64, N, N)), reshape(collect(Float64, 1:N*N), N, N)
    AF, BF, C = Array(A), Array(B), sparse(CF)

    @test A .* B == AF .* BF
    @test A[1,:] .* B == AF[1,:] .* BF
    @static if COMPREHENSIVE
    @test A[1,:] .* BF == AF[1,:] .* BF
    end
    @test A .* BF[:,1] == AF .*  BF[:,1]

    @static if COMPREHENSIVE
    @test AF .* B[1,:] == AF .*  BF[1,:]

    @test A .* 3 == AF .* 3
    @test A[1,:] .* 3 == AF[1,:] .* 3
    end
    @test 3 .- A == 3 .- AF
    @static if COMPREHENSIVE
    @test A .- B == AF .- BF
    end
    @test A - AF == zeros(size(AF))
    @static if COMPREHENSIVE
    @test AF - A == zeros(size(AF))
    @test A + AF == AF + A
    end
    @test (A .< B) == (AF .< BF)
    @static if COMPREHENSIVE
    @test (A .!= B) == (AF .!= BF)

    @test A .\ 3 == AF .\ 3
    end
    @test 3 ./ A == 3 ./ AF
    @static if COMPREHENSIVE
    @test A .\ C == AF .\ CF
    end
    @test A ./ C == AF ./ CF
    @static if COMPREHENSIVE
    @test BF ./ C == BF ./ CF
    end

    @test A .^ 3 == AF .^ 3
    @static if COMPREHENSIVE
    @test 3 .^ A == 3 .^ AF
    end

    # broadcasting against a dense-ish vector grows storage on demand instead of
    # preallocating the bound, which for these shapes is the dense size (#47)
    # the two diagonals: two entries in each row, 1% of the matrix
    M = sparse([1:200; 1:200], [1:200; 200:-1:1], collect(Float64, 1:400), 200, 200)
    v = collect(Float64, 1:200)
    @test M .* v == Array(M) .* v   # sparse result
    @static if COMPREHENSIVE
    @test M .* v' == Array(M) .* v'
    end
    @test M .+ v .* 1 == Array(M) .+ v   # full sparse result: does grow to the bound
    M .* v; @static COMPREHENSIVE && M .* v' # warmup for @allocated
    # the bound would be 200 * 200 * (8 + 8) bytes = 640 KB
    @test @allocated(M .* v) < 2^16
    @static if COMPREHENSIVE
    @test @allocated(M .* v') < 2^16
    end

    @test spzeros(0,0)  + spzeros(0,0) == zeros(0,0)
    @static if COMPREHENSIVE
    @test spzeros(0,0)  * spzeros(0,0) == zeros(0,0)
    end
    @test spzeros(1,0) .+ spzeros(2,1) == zeros(2,0)
    @test spzeros(1,0) .* spzeros(2,1) == zeros(2,0)
    @test spzeros(1,2) .+ spzeros(0,1) == zeros(0,2)
    @test spzeros(1,2) .* spzeros(0,1) == zeros(0,2)
    # a result with no rows must not be densified, even when f(0, ...) != 0: zero colptr step
    @test ((x, y) -> x + y + 1).(spzeros(1,2), spzeros(0,1)) == fill(1.0, 0, 2)
    @test broadcast!(x -> x + 1, spzeros(0,2), spzeros(0,1)) == fill(1.0, 0, 2)
end

@testset "sparse vector broadcast of two arguments" begin
    sv1, sv5 = sparse([2.5]), sparse(collect(Float64, 1:5))
    for (sa, sb) in ((sv1, sv1), (sv1, sv5), (sv5, sv1), (sv5, sv5))[@static COMPREHENSIVE ? (1:4) : [2, 4]]
        fa, fb = Vector(sa), Vector(sb)
        for f in (max, (@static COMPREHENSIVE ? (*,) : ())...)
            @test @inferred(broadcast(f, sa, sb))::SparseVector == broadcast(f, fa, fb)
            @test @inferred(broadcast(f, Vector(sa), sb))::SparseVector == broadcast(f, fa, fb)
            @test @inferred(broadcast(f, sa, Vector(sb)))::SparseVector == broadcast(f, fa, fb)
            @test @inferred(broadcast(f, SparseMatrixCSC(sa), sb))::SparseMatrixCSC == broadcast(f, reshape(fa, Val(2)), fb)
            @test @inferred(broadcast(f, sa, SparseMatrixCSC(sb)))::SparseMatrixCSC == broadcast(f, fa, reshape(fb, Val(2)))
            if length(fa) == length(fb)
                @test @inferred(map(f, sa, sb))::SparseVector == broadcast(f, fa, fb)
            end
        end
        if length(fa) == length(fb)
            for f in (+, -)
                @test @inferred(f(sa, sb))::SparseVector == f(fa, fb)
                @test @inferred(f(Vector(sa), sb))::Vector == f(fa, fb)
                @test @inferred(f(sa, Vector(sb)))::Vector == f(fa, fb)
            end
        end
    end
end

# kept out of the testsets so that `@allocated` measures the call alone
densesum(x, y) = x .+ y

@testset "broadcast of + and - with a dense array or a scalar is dense (#516)" begin
    s, S = sparsevec([1, 3], [1.5, -2.0], 4), sparse([1, 3, 4], [1, 2, 2], [1.0, 2.0, -3.0], 4, 2)
    d, D = [1.0, 0.0, 2.0, 0.0], [1.0 0.0; 0.0 2.0; 3.0 0.0; 0.0 0.0]
    fs, fS = Array(s), Array(S)
    @test @inferred(broadcast(+, s, d))::Vector{Float64} == fs + d
    @test @inferred(broadcast(-, D, S))::Matrix{Float64} == D - fS
    @test @inferred(broadcast(+, s, 1))::Vector{Float64} == fs .+ 1
    # the other broadcasts with a dense array or a scalar, and sums without one, stay sparse
    @test (S .* D)::SparseMatrixCSC == fS .* D
    @test (S .- S)::SparseMatrixCSC == fS .- fS
    @static if COMPREHENSIVE
    # the shapes broadcast expands, and views and wrappers on either side
    @test @inferred(broadcast(-, s, d'))::Matrix{Float64} == fs .- d'
    @test (view(s, 2:4) .+ view(d, 2:4))::Vector{Float64} == fs[2:4] .+ d[2:4]
    @test (D' .- S')::Matrix{Float64} == D' .- fS'
    # a view of columns picked by a vector is densified from its stored entries, not by
    # indexing the view, which reads the column index once for each of its entries
    T, J = sparse([1, 30, 50], [1, 2, 2], [1.0, 2.0, -3.0], 50, 2), CountedReads([2, 1])
    @test (view(T, :, J) .+ ones(50, 2))::Matrix{Float64} == Array(T)[:, [2, 1]] .+ 1
    @test J.reads[] < 50
    # a fused expression is dense only when it is made of + and - alone
    fused(s, d) = s .+ d .- 1 .+ s .- (.-d)
    @test @inferred(fused(s, d))::Vector{Float64} == fs .+ d .- 1 .+ fs .+ d
    scaled(s, d) = 2 .* s .+ d
    @test @inferred(scaled(s, d))::SparseVector == 2 .* fs .+ d
    # a scalar on either side, wrapped or not, and in a sum of sparse arrays alone
    @test @inferred(broadcast(-, 1, S))::Matrix{Float64} == 1 .- fS
    @test (s .+ fill(1.0))::Vector{Float64} == fs .+ 1
    @test (S .+ s .- Ref(2))::Matrix{Float64} == fS .+ fs .- 2
    @test (.-s)::SparseVector == .-fs
    # a sparse destination keeps the result sparse
    x = copy(s); x .+= d; x .-= 1
    @test x::SparseVector == fs + d .- 1
    # an empty result does not densify the sparse arguments, whatever their size
    e = zeros(1, 0)
    @test @inferred(densesum(s, e))::Matrix{Float64} == fs .+ e
    @test (spzeros(Int, 0, 2) .- D[1:0, :])::Matrix{Float64} == zeros(0, 2)
    long = spzeros(10^6)
    densesum(long, e)
    @test @allocated(densesum(long, e)) == @allocated(densesum(s, e))
    end
end

@testset "broadcast over sparse arrays of a function that is not zero at zero is dense" begin
    s, S = sparsevec([1, 3], [1.5, -2.0], 4), sparse([1, 3, 4], [1, 2, 2], [1.0, 2.0, -3.0], 4, 2)
    fs, fS = Array(s), Array(S)
    @test @inferred(broadcast(cos, S))::Matrix{Float64} == cos.(fS)
    @test isequal(@inferred(broadcast(/, s, s))::Vector{Float64}, fs ./ fs)
    # a zero-preserving function stays sparse, and so does any function of a scalar
    @test @inferred(broadcast(sin, S))::SparseMatrixCSC == sin.(fS)
    @test @inferred(broadcast(^, S, 0))::SparseMatrixCSC == fS .^ 0
    plus(c) = x -> x + c
    @test @inferred(broadcast(plus(1.0), S))::SparseMatrixCSC == fS .+ 1
    @static if COMPREHENSIVE
    # the result is the dense one, whatever its type
    @test (S .== S)::BitMatrix == (fS .== fS)
    fused(S, s) = exp.(S .* s)
    @test @inferred(fused(S, s))::Matrix{Float64} == exp.(fS .* fs)
    # wrappers and banded matrices count as sparse arguments
    @test cos.(S')::Matrix{Float64} == cos.(fS')
    @test isequal((S[1:2, :] ./ Diagonal([1.0, 2.0]))::Matrix{Float64}, fS[1:2, :] ./ Diagonal([1.0, 2.0]))
    # `map` and a sparse destination keep the result sparse
    @test map(cos, S)::SparseMatrixCSC == cos.(fS)
    @test broadcast!(cos, similar(S), S)::SparseMatrixCSC == cos.(fS)
    # sums, differences and products are sparse and inferable for an element type whose
    # zero the compiler cannot evaluate
    B = sparse(BigFloat[1 0; 0 2])
    @test @inferred(broadcast(*, B, B))::SparseMatrixCSC == Array(B) .* Array(B)
    @test cos.(B)::Matrix{BigFloat} == cos.(Array(B))
    # a function that throws at zero is left to the sparse kernels: only a matrix with a
    # structural zero evaluates it there
    shifted(x) = sqrt(x - 1)
    @test shifted.(sparse([1.0 2.0; 5.0 10.0]))::SparseMatrixCSC == [0.0 1.0; 2.0 3.0]
    @test_throws DomainError shifted.(S)
    end
end

@testset "aliasing and indexed assignment or broadcast!" begin
    A = sparsevec([0, 0, 1, 1])
    B = sparsevec([1, 1, 0, 0])
    A .+= B
    @test A == sparse([1,1,1,1])

    A = fixture(Float64, 10, 8)
    fA = Array(A)
    b = collect(Float64, 1:10);
    broadcast!(/, A, A, b)
    @test A == fA ./ Array(b)

    a = sparse([1,3,5])
    b = sparse([3,1,2])
    a[b] = a
    @test a == [3,5,1]
    a = sparse([3,2,1])
    a[a] = [4,5,6]
    @test a == [6,5,4]

    A = sparse([1,2,3,4])
    V = view(A, A)
    @test V == A
    V[1] = 2
    @test V == A == [2,2,3,4]
    V[1] = 2^30
    @test V == A == [2^30, 2, 3, 4]

    A = sparse([2,1,4,3])
    V = view(A, :)
    A[V] = (1:4) .+ 2^30
    @test A == [2,1,4,3] .+ 2^30

    @static if COMPREHENSIVE
    A = sparse([2,1,4,3])
    R = reshape(view(A, :), 2, 2)
    A[R] = (1:4) .+ 2^30
    @test A == [2,1,4,3] .+ 2^30
    end

    A = sparse([2,1,4,3])
    R = reshape(A, 2, 2)
    A[R] = (1:4) .+ 2^30
    @test A == [2,1,4,3] .+ 2^30

    # And broadcasting
    a = sparse([1,3,5])
    b = sparse([3,1,2])
    a[b] .= a
    @test a == [3,5,1]
    a = sparse([3,2,1])
    a[a] .= [4,5,6]
    @test a == [6,5,4]

    @static if COMPREHENSIVE
    A = sparse([2,1,4,3])
    V = view(A, :)
    A[V] .= (1:4) .+ 2^30
    @test A == [2,1,4,3] .+ 2^30
    end

    A = sparse([2,1,4,3])
    R = reshape(view(A, :), 2, 2)
    A[R] .= reshape((1:4) .+ 2^30, 2, 2)
    @test A == [2,1,4,3] .+ 2^30

    @static if COMPREHENSIVE
    A = sparse([2,1,4,3])
    R = reshape(A, 2, 2)
    A[R] .= reshape((1:4) .+ 2^30, 2, 2)
    @test A == [2,1,4,3] .+ 2^30
    end

    # map! with the destination among the inputs (issue #26)
    A = sparse([100 0; 300 400])
    @test map!(x -> x + 1, A) == [101 1; 301 401]
    v = sparsevec([1, 0, 2])
    @test map!(x -> x + 1, v) == [2, 1, 3]
    S0 = fixture(Float64, 10, 10); S1 = permutedims(S0); S2 = sparse([1, 4, 4, 9], [2, 2, 7, 10], [1.5, -2.0, 0.5, 4.0], 10, 10); C = copy(S0)
    @test map!(+, S0, S0, S1) == C + S1
    S0 = copy(C)
    @test map!(+, S0, S1, S2, S0) == S1 + S2 + C
    S0 = copy(C); D = Diagonal(collect(Float64, 1:10))
    @test map!(+, S0, D, S0) == D + C
end

@testset "1-dimensional 'opt-out' (non) sparse broadcasting" begin
    # SparseArrays intentionally only promotes to sparse for limited array types
    # More support may be added in the future, but for now let's make sure that
    # broadcast still performs as expected (issue #26977)
    A = spzeros(5)
    @test A .+ (1:5) == 1:5
    @test A .* 2 .+ view(collect(1:10), 1:5) == 1:5
    @static if COMPREHENSIVE
    @test 2 .* A .+ view(1:10, 1:5) == 1:5
    @test (A .+ (1:5)) .* 2 == 2:2:10
    @test ((1:5) .+ A) .* 2 == 2:2:10
    end
    @test 2 .* ((1:5) .+ A) == 2:2:10
    @static if COMPREHENSIVE
    @test 2 .* (A .+ (1:5)) == 2:2:10
    end
    # in-place with an unsupported (Tuple) argument used to recurse, see #573
    B = sparsevec([2], [3.0], 5)
    @test (B .= .*(B, A .+ 1, (2,))) == [0, 6, 0, 0, 0]

    @static if COMPREHENSIVE
    # lu(zeros(5,5)) throw SingularException, see #42343
    @test_throws SingularException Diagonal(spzeros(5)) \ view(ones(10), 1:5)
    end
end

@static if COMPREHENSIVE
@testset "Issue #27836" begin
    @test minimum(sparse([1, 2], [1, 2], ones(Int32, 2)), dims = 1) isa Matrix
end
end

@testset "Issue #30118" begin
    @static if COMPREHENSIVE
    @test ((_, x) -> x).(Int, spzeros(3)) == spzeros(3)
    end
    @test ((_, _, x) -> x).(Int, Int, spzeros(3)) == spzeros(3)
    @static if COMPREHENSIVE
    @test ((_, _, _, x) -> x).(Int, Int, Int, spzeros(3)) == spzeros(3)
    @test ((_, _, _, _, x) -> x).(Int, Int, Int, Int, spzeros(3)) == spzeros(3)
    @test typeof(((_, _, _, _, x) -> x).(Int, Int, Int, Int, spzeros(3))) == typeof(spzeros(3))
    @test typeof(((x, _, _, _, _, _, y) -> x + y).(spzeros(3), Int, Float32, Int, 1, Int, spzeros(3, 3))) == typeof(spzeros(3, 3))
    end
end

using SparseArrays.HigherOrderFns: SparseVecStyle, SparseMatStyle

@testset "Issue #30120: method ambiguity" begin
    # HigherOrderFns._copy(f) was ambiguous.  It may be impossible to
    # invoke this from dot notation and it is an error anyway.  But
    # when someone invokes it by accident, we want it to produce a
    # meaningful error.
    err = try
        copy(Broadcast.Broadcasted{SparseVecStyle}(rand, ()))
    catch err
        err
    end
    @test err isa MethodError
    @test !occursin("is ambiguous", sprint(showerror, err))
    @test err.f === SparseArrays.HigherOrderFns._copy
end

@testset "Sparse outer product, for type $T and vector $op" for
         (op, T) in (@static COMPREHENSIVE ? pairwise : eachvalue)((transpose, adjoint), (Float64, ComplexF64))
    m, n = 100, 250
    A = fixture(T, m, n)
    a, b = view(A, :, 1), fixturevec(T, n)   # of different lengths, so the product is not square
    av, bv = Vector(a), Vector(b)
    v = @inferred a .* op(b)
    w = @inferred b .* op(a)
    @test issparse(v)
    @test issparse(w)
    @test mismatch(v, av .* op(bv); Ti=Int) === nothing
    @test mismatch(w, bv .* op(av); Ti=Int) === nothing
    @static if COMPREHENSIVE
    # the index type is promoted over both vectors, and the wider one holds the result;
    # the promotion depends on neither `T` nor `op`
    if T === Float64 && op === transpose
        c = SparseVector{T,Int8}(fixturevec(T, m))
        @test mismatch(c .* op(b), Vector(c) .* op(bv); Ti=Int) === nothing
        @test mismatch(b .* op(c), bv .* op(Vector(c)); Ti=Int) === nothing
    end
    end
end

@testset "Sparse outer product drops products with stored zeros" begin
    # the result is sized for nnz(x) * nnz(y) entries and must shrink for each stored zero
    # of either vector (an oversized result errored, issue #42670)
    A = SparseMatrixCSC(4, 1, [1, 4], [1, 2, 4], [1.0, 0.0, 3.0])
    a, b = view(A, :, 1), SparseVector(3, [1, 2, 3], [0.0, 2.0, 5.0])
    C = a .* transpose(b)
    @test mismatch(C, Vector(a) .* transpose(Vector(b))) === nothing
    @test nnz(C) == 4
end

@static if COMPREHENSIVE
@testset "issue #31758: out of bounds write in _map_zeropres!" begin
    y = sparsevec([2,7], [1., 2.], 10)
    x1 = sparsevec(fill(1.0, 10))
    x2 = sparsevec([2,7], [1., 2.], 10)
    x3 = sparsevec(fill(1.0, 10))
    f(x, y, z) = x == y == z == 0 ? 0.0 : NaN
    y .= f.(x1, x2, x3)
    @test all(isnan, y)
end
end

@testset "Vec/Mat Style" begin
    @test SparseVecStyle(Val(0)) == SparseVecStyle()
    @test SparseVecStyle(Val(1)) == SparseVecStyle()
    @test SparseVecStyle(Val(2)) == SparseMatStyle()
    @test SparseVecStyle(Val(3)) == Broadcast.DefaultArrayStyle{3}()
    @test SparseMatStyle(Val(0)) == SparseMatStyle()
    @test SparseMatStyle(Val(1)) == SparseMatStyle()
    @test SparseMatStyle(Val(2)) == SparseMatStyle()
    @test SparseMatStyle(Val(3)) == Broadcast.DefaultArrayStyle{3}()
end

@testset "extrema" begin
    n = 10
    # entries of both signs, so that a structural zero is not always the minimum
    A = fixture(Float64, n, n + 2) - permutedims(fixture(Float64, n + 2, n))
    B = Array(A)
    C = Array{Real}(undef, 0, 0)
    x = fixturevec(Float64, n) - sparsevec([2, 5, 6], [7.0, 1.0, 2.5], n)
    y = Array(x)
    z = Array{Real}(undef, 0)
    f(x) = x^3
    @test extrema(A) == extrema(B)
    @test extrema(x) == extrema(y)
    @test extrema(f, A) == extrema(f, B)
    @test extrema(f, x) == extrema(f, y)
    @test extrema(spzeros(n, n)) == (0.0, 0.0)
    @test extrema(spzeros(n)) == (0.0, 0.0)
    @test_throws "reducing over an empty" extrema(spzeros(0, 0))
    @test_throws "reducing over an empty" extrema(spzeros(0))
    @test extrema(sparse(ones(n, n))) == (1.0, 1.0)
    @test extrema(sparse(ones(n))) == (1.0, 1.0)
    @static if COMPREHENSIVE
    @test extrema(A; dims=:) == extrema(B; dims=:)
    end
    @test extrema(A; dims=1) == extrema(B; dims=1)
    @test extrema(A; dims=2) == extrema(B; dims=2)
    @static if COMPREHENSIVE
    @test extrema(A; dims=(1,2)) == extrema(B; dims=(1,2))
    end
    @test extrema(f, A; dims=1) == extrema(f, B; dims=1)
    @static if COMPREHENSIVE
    @test_throws "reducing over an empty" extrema(sparse(C); dims=1) == extrema(C; dims=1)
    @test extrema(A; dims=[]) == extrema(B; dims=[])
    @test extrema(x; dims=:) == extrema(y; dims=:)
    end
    @test extrema(x; dims=1) == extrema(y; dims=1)
    @static if COMPREHENSIVE
    @test extrema(f, x; dims=1) == extrema(f, y; dims=1)
    @test_throws "reducing over an empty" extrema(sparse(z); dims=1)
    @test extrema(x; dims=[]) == extrema(y; dims=[])
    end
end

function test_extrema(a; dims_test = @static COMPREHENSIVE ? ((), 1, 2, (1,2), 3) : (1, (1,2)))
    for dims in dims_test
        vext = extrema(a; dims)
        vmin, vmax = minimum(a; dims), maximum(a; dims)
        @test all(x -> isequal(x[1], x[2:3]), zip(vext,vmin,vmax))
    end
end
@testset "NaN test for sparse extrema" begin
    for sz = (10, (@static COMPREHENSIVE ? (3, 100) : ())...)
        # `NaN`s at stored and at unstored positions, and columns without one
        A = fixture(Float64, sz, sz)
        A[1:sz+3:sz^2] .= NaN
        test_extrema(A)
        A = fixturevec(Float64, sz*sz)
        A[1:sz+3:sz^2] .= NaN
        test_extrema(A; dims_test = @static COMPREHENSIVE ? ((), 1, 2) : (1,))
    end
end

@static if COMPREHENSIVE
@testset "issue #42670 - error in sparsevec outer product" begin
    A = spzeros(Int, 4)
    B = copy(A)
    C = sparsevec([0 0 1 1 0 0])'
    A[2] = 1
    A[2] = 0
    @test A * C == B * C == spzeros(Int, 4, 6)
end

@testset "issue #46337 - error in sparsevec map" begin
    x = sparsevec([1], [1//1+0im])
    @test inv.(x) == [1//1+0im]
    y = spzeros(Int, 1)
    @test y ./ x == y
end
end

@testset "map and broadcast kernels with the row count at typemax of the index type" begin
    # Int16 because a narrow index type is the point
    for Ti in (Int16,)
        n = Int(typemax(Ti))
        A = SparseMatrixCSC(n, 1, Ti[1, 2], Ti[1], [1.0])
        x = SparseVector(n, Ti[1], [1.0])
        for S in (x, (@static COMPREHENSIVE ? (A,) : ())...)
            @test map(+, S, S) == 2Array(S) && SparseArrays.indtype(map(+, S, S)) == Ti
            @test S .+ S == 2Array(S) && SparseArrays.indtype(S .+ S) == Ti
            @static if COMPREHENSIVE
            @test map(+, S, S, S) == 3Array(S) && SparseArrays.indtype(map(+, S, S, S)) == Ti
            @test broadcast(+, S, S, S) == 3Array(S) && SparseArrays.indtype(broadcast(+, S, S, S)) == Ti
            end
        end
        # a dense-structured result needs a column pointer of n + 1, so only the vector fits
        @test map((a, b) -> a + b + 1, x, x) == 2Array(x) .+ 1
        y = SparseVector(n, Ti.(1:n), ones(n))   # nnz + 1 does not fit the index type either
        @test y .+ y == 2Array(y) && SparseArrays.indtype(y .+ y) == Ti
        @static if COMPREHENSIVE
        @test x .+ x .+ 1 == 2Array(x) .+ 1 && map(+, y, y) == 2Array(y)
        end
    end
    for Ti in (Int16, (@static COMPREHENSIVE ? (Int32,) : ())...)
        M = SparseMatrixCSC{Float64,Ti}
        @test !hasunionlocal(SparseArrays.HigherOrderFns._map_zeropres!, (typeof(+), M, M, M), Ti, Int)
        @test !hasunionlocal(SparseArrays.HigherOrderFns._broadcast_zeropres!, (typeof(+), M, M, M), Ti, Int)
    end
    @static if COMPREHENSIVE
    n = Int128(typemax(Int)) + 1; w = SparseVector(n, [n], [1.0])   # indices wider than Int stay
    @test nonzeroinds(map!(+, SparseVector(n, [n], [0.0]), w, w)) == [n]
    end
end

end # module
