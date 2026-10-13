# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseReductionTests

using Test
using SparseArrays
using SparseArrays: getcolptr, nonzeroinds, _show_with_braille_patterns, _isnotzero, fixed, _is_fixed
using LinearAlgebra
using Random
include("testhelpers.jl")

# `@inferred` on a call with keywords compiles a wrapper per signature, so those go through this
along(f::F, X, dims) where {F} = f(X; dims)

@testset "reductions" begin
    sA = sparse([1, 3, 2, 3, 1, 2], [1, 1, 3, 4, 6, 7], [-1.5, 2.0, 0.5, -3.0, 4.0, -0.25], 3, 7)
    pA = sparse(reshape((1:21) ./ 32, 3, 7))   # every entry stored and in (0, 1)
    @static if COMPREHENSIVE
    se33 = SparseMatrixCSC{Float64}(I, 3, 3)
    p28227 = sparse(Real[0 0.5])
    end

    for arr in (sA, pA, spzeros(3, 3), fixture(Float64, 5, 3), (@static COMPREHENSIVE ? (se33, p28227) : ())...)
        farr = Array(arr)
        for f in (sum, prod, minimum, maximum)
            @test f(arr) ≈ f(farr)
            @test f(arr, dims=1) ≈ f(farr, dims=1)
            @test f(arr, dims=2) ≈ f(farr, dims=2)
            @test f(arr, dims=(1, 2)) ≈ fill(f(farr), 1, 1)
            @test isequal(f(arr, dims=3), f(farr, dims=3))
        end
        for f in (+, (@static COMPREHENSIVE ? (*,) : ())...)
            @static if COMPREHENSIVE
            @test mapreduce(identity, f, arr) ≈ mapreduce(identity, f, farr)
            end
            @test mapreduce(x -> x + 1, f, arr) ≈ mapreduce(x -> x + 1, f, farr)
        end
    end

    for s0 in (spzeros(3, 7), (@static COMPREHENSIVE ? (spzeros(1, 3), spzeros(3, 1)) : ())...), d in (1, 2, 3, (1,2))
        @test all(isone, sum(s0, dims=d, init=1.0))
    end

    for f in (sum, prod, maximum)
        # Test with a map function that maps to non-zero
        for arr in (sA, (@static COMPREHENSIVE ? (se33, pA) : ())...)
            @test f(x->x+1, arr) ≈ f(arr .+ 1)
        end

        # case where f(0) would throw: a sparse matrix that stores every entry
        pA1 = map(x -> x + 1, pA)
        @test f(x->sqrt(x-1), pA1) ≈ f(sqrt.(pA))
        # `sum` still evaluates the map at the structural zero and throws here
        if (@static COMPREHENSIVE ? f !== sum : f === maximum)
            @test f(x->sqrt(x-1), pA1, dims=1) ≈ f(sqrt.(pA), dims=1)
            @test f(x->sqrt(x-1), pA1, dims=2) ≈ f(sqrt.(pA), dims=2)
            @static if COMPREHENSIVE
            @test f(x->sqrt(x-1), pA1, dims=3) ≈ f(sqrt.(pA), dims=3)
            end
            @test f(x->sqrt(x-1), pA1; dims=1, sparse=true) ≈ f(sqrt.(pA), dims=1)
            @test f(x->sqrt(x-1), pA1; dims=2, sparse=true) ≈ f(sqrt.(pA), dims=2)
        end
    end

    @testset "small integers: the entries not stored widen as for dense" begin
        # `sum` and `prod` of small integers give an `Int`, `mapreduce` with `+` or `*` keeps
        # the element type and wraps, and `Bool` products stay `Bool`, as for dense input
        # the wrappers and the maps go with one integer type: `Bool` differs in `sum` and `prod` only
        for T in (Int16, (@static COMPREHENSIVE ? (Bool,) : ())...),
            A in (sparse(T[1 1; 1 0]), (@static COMPREHENSIVE ? (T === Int16 ? (sparse(T[1 0 0; 0 0 1])', spzeros(T, 2, 2), sparsevec(T[1, 0, 1])) : ()) : ())...)
            A isa AbstractMatrix && (A[1, 2] = zero(T))   # a stored zero
            M = Array(A)
            for f in (T === Bool ? () : (x -> x + one(T), (@static COMPREHENSIVE ? (abs2,) : ())...)), op in (+, *)
                @test @inferred(mapreduce(f, op, A)) === mapreduce(f, op, M)
                A isa SparseMatrixCSC || continue   # the `dims` kernels are the matrix ones
                for dims in (1, 2)
                    r, rd = mapreduce(f, op, A; dims), mapreduce(f, op, M; dims)
                    @test typeof(r) == typeof(rd) && r == rd
                end
            end
            @test @inferred(sum(A)) === sum(M) && @inferred(prod(A)) === prod(M)
        end
        # the unstored entries wrap with `+` and `*` as the stored ones do, widen for `sum`,
        # and take the type of `init` or of another mapped value
        for (T, n) in ((Int16, 11000),)   # 6n is more than a `T` holds
            f = x -> x + T(3)
            A = spzeros(T, 2, n)
            @test mapreduce(f, +, A) === mapreduce(f, +, zeros(T, 2n)) === (6n) % T
            @test mapreduce(f, *, A) === mapreduce(f, *, zeros(T, 2n))
            @test sum(f, A) === 6n && sum(f, A; dims = 2) == [3n; 3n;;]
            @test mapreduce(f, +, A; init = 0) === 6n
            @static if COMPREHENSIVE
            @test mapreduce(f, +, A; dims = 2) == mapreduce(f, +, Array(A); dims = 2)
            @test mapreduce(f, +, A; dims = 2, init = 0) == mapreduce(f, +, A; dims = 2, init = 0, sparse = true) == [3n; 3n;;]
            @test mapreduce(f, *, A; dims = 2, init = 1) == mapreduce(f, *, Array(A); dims = 2, init = 1)
            end
        end
        @static if COMPREHENSIVE
        @test mapreduce(x -> iszero(x) ? Int8(100) : 1000.0, +, sparse([1, 0, 0])) === 1200.0
        # a stored `Inf` or `NaN` is not absorbed by an unstored zero
        @test isnan(prod(sparse([0.0 Inf; 1.0 2.0])))
        # nor is an unstored zero lost when the product of the stored entries overflows
        @test prod(sparse([0.0 1e200; 1e200 1.0])) === 0.0
        end
    end

    @testset "logical reductions" begin
        v = spzeros(Bool, 5, 2)
        @test !any(v)
        @test !all(v)
        @test iszero(v)
        @test count(v) == 0
        v = SparseMatrixCSC(5, 2, [1, 2, 2], [1], [false])
        @test !any(v)
        @test !all(v)
        @test iszero(v)
        @test count(v) == 0
        v = SparseMatrixCSC(5, 2, [1, 2, 2], [1], [true])
        @test any(v)
        @test !all(v)
        @test !iszero(v)
        @test count(v) == 1
        v[2,1] = true
        @test any(v)
        @test !all(v)
        @test !iszero(v)
        @test count(v) == 2
        v .= true
        @test any(v)
        @test all(v)
        @test !iszero(v)
        @test count(v) == length(v)
        @test all(!iszero, spzeros(0, 0))
        @test !any(iszero, spzeros(0, 0))
        @static if COMPREHENSIVE
            # a predicate that is not Boolean is rejected, as for dense
            for S in (sparse([1 0; 0 0]), sparsevec([1, 0]))
                @test_throws TypeError any(S)
                @test_throws TypeError all(S)
            end
            # also next to a `missing`, which `|` and `&` would let absorb the other value
            S = sparse(reshape(Union{Missing,Int}[missing, 1], 2, 1))
            @test_throws TypeError any(S)
            @test_throws TypeError all(S)
        end
    end

    @testset "seed without copying the first slice" begin
        # the `Array` seed is built only for an empty slice; a non-empty reduction
        # allocates the result and Base's own temporary, not a dense copy of the slice
        A = sparse([i + 7j for j in 1:4 for i in 1:100:10^5], repeat(1:4; inner=1000), collect(1.0:4000.0), 10^5, 4)
        for g in (A -> maximum(A; dims=2), (@static COMPREHENSIVE ? (A -> minimum(A'; dims=1), A -> extrema(A; dims=2),
                  A -> maximum(view(A, :, 1:2); dims=2)) : ())...)
            r = g(A)
            @test r isa Array
            @test (@allocated g(A)) < 2.5 * sizeof(r)
        end
    end
    @testset "empty cases" begin
        errchecker(str) = occursin(": reducing over an empty collection is not allowed", str) ||
                          occursin(": reducing with ", str) ||
                          occursin("collection slices must be non-empty", str) ||
                          occursin("array slices must be non-empty", str)
        @test sum(sparse(Int[])) === 0
        @test prod(sparse(Int[])) === 1
        @test_throws errchecker minimum(sparse(Int[]))
        @test_throws errchecker maximum(sparse(Int[]))

        for f in (sum, (@static COMPREHENSIVE ? (prod,) : ())...)
            @test isequal(f(spzeros(0, 1), dims=1), f(Matrix{Int}(I, 0, 1), dims=1))
            @test isequal(f(spzeros(0, 1), dims=2), f(Matrix{Int}(I, 0, 1), dims=2))
            @test isequal(f(spzeros(0, 1), dims=(1, 2)), f(Matrix{Int}(I, 0, 1), dims=(1, 2)))
            @test isequal(f(spzeros(0, 1), dims=3), f(Matrix{Int}(I, 0, 1), dims=3))
        end
        # an empty array has no value to seed even an empty result with: these throw along
        # every dimension, as for dense input
        for f in (maximum, findmax, (@static COMPREHENSIVE ? (minimum, findmin) : ())...)
            for dims in (1, 2, (1, 2), 3)
                @test_throws errchecker f(Matrix{Int}(I, 0, 1); dims)
                @test_throws errchecker f(spzeros(0, 1); dims)
            end
        end
        E = spzeros(3, 0)
        for (X, dims, f) in (Any[E, 1, minimum], Any[E', 2, maximum], Any[spzeros(0), 2, extrema], Any[spzeros(0)', 1, minimum],
            (@static COMPREHENSIVE ? (Any[E, 3, maximum], Any[view(E, :, 1:0), 1, extrema], Any[transpose(spzeros(0)), 3, maximum]) : ())...)
            @test_throws errchecker f(Array(X); dims)
            @test_throws errchecker f(X; dims)
        end

        some_exception(op) = try return (Some(op()), nothing); catch ex; return (nothing, ex); end
        reduced_shape(sz, dims) = ntuple(d -> d in dims ? 1 : sz[d], length(sz))
        # every reduction of an empty array gives the value, or the error, of the dense one
        @testset "$r(spzeros($T, $sz); dims=$dims)" for
            r in (minimum, maximum, findmin, findmax, extrema, sum, prod, mapreduce, all, any, count),
            T in (Int, Union{Missing, Int}, Number, Union{Missing, Number}, Bool, Union{Missing, Bool}),
            sz in ((0,), (0,1), (1,0), (0,0),),
            dims in (1, 2, (1,2))

            A = spzeros(T, sz...)
            rsz = reduced_shape(sz, dims)

            v, ex = some_exception() do; r(A); end
            if isnothing(v)
                @test_throws typeof(ex) r(A; dims)
                @test_throws typeof(ex) r(zeros(T, sz...))
                @test_throws typeof(ex) r(zeros(T, sz...); dims)
            else
                actual = fill(something(v), rsz)
                @test something(v) === r(zeros(T, sz...))
                @test isequal(r(A; dims), actual)
                @test eltype(r(A; dims)) === eltype(actual)
            end

            for f in (identity, abs, abs2)
                v, ex = some_exception() do; r(f, A); end
                if isnothing(v)
                    @test_throws typeof(ex) r(f, A; dims)
                    @test_throws typeof(ex) r(f, zeros(T, sz...))
                    @test_throws typeof(ex) r(f, zeros(T, sz...); dims)
                else
                    actual = fill(something(v), rsz)
                    @test something(v) === r(f, zeros(T, sz...))
                    @test isequal(r(f, A; dims), actual)
                    @test eltype(r(f, A; dims)) === eltype(actual)
                end
            end
        end
    end
    @testset "seeds of minimum, maximum and extrema along a dimension" begin
        # NaN, missing and the `abs`/`abs2` zero seed are handled as for dense input
        N = sparse([NaN 1.0; 2.0 3.0])
        @static if COMPREHENSIVE
        M = sparse(Union{Missing,Float64}[missing 1.0; 2.0 3.0])
        end
        C = sparse(ComplexF64[1+im 0; 0 -2])
        # each argument type and each of `missing` and NaN once, not their product with the reductions
        for (X, f) in (Any[N, minimum], Any[N, extrema], Any[N', maximum], Any[sparsevec([NaN, 1.0]), maximum],
            (@static COMPREHENSIVE ? (Any[M, maximum], Any[view(N, :, 1:2), extrema], Any[sparsevec([NaN, 1.0])', minimum],
                                      Any[transpose(sparsevec(Union{Missing,Float64}[missing, 1.0])), extrema]) : ())...), dims in (1, 2)
            r, rd = f(X; dims), f(Array(X); dims)
            @test typeof(r) == typeof(rd) && isequal(r, rd)
        end
        for X in (N, C, spzeros(0, 3), (@static COMPREHENSIVE ? (N', view(C, :, 1:2), sparsevec(ComplexF64[1 + im, 0, -2])') : ())...), dims in (1, 2), g in (abs, (@static COMPREHENSIVE ? (X === C ? (abs2,) : ()) : ())...)
            r, rd = maximum(g, X; dims), maximum(g, Array(X); dims)   # no throw for the empty axis
            @test typeof(r) == typeof(rd) && isequal(r, rd)
        end
        @test maximum(N; dims = 1, sparse = true) isa SparseMatrixCSC && mismatch(maximum(N; dims = 1, sparse = true), maximum(Array(N); dims = 1)) === nothing
    end
end

@testset "reduction kernels: chunked ranges and implicit zeros" begin
    @testset "all-zero arrays with f(0) != 0, straddling pairwise blocksize" begin
        # The mapreduce kernels may be called with sub-ranges of the full index
        # space; zero counts must be relative to the requested chunk (sizes
        # straddle Base's pairwise blocksizes of 1024/4096)
        for (m, n) in ((60, 60), (1, 5000), (5000, 1), (33, 129), (3, 3))
            @test sum(x->x+1, spzeros(m, n)) == m*n
            @test mapreduce(x->x+1, +, spzeros(m, n)) == m*n
            @test count(iszero, spzeros(m, n)) == m*n
            @test mapreduce(x->x+1, (a,b)->a+b, spzeros(m, n)) == m*n
        end
    end
    @testset "non-commutative kernels: interior and trailing implicit zeros" begin
        # prod is dispatched as a non-commutative op; its kernel must not
        # overcount trailing zeros (these all have f(0) = 2, so any zero
        # miscounting changes the result)
        @test prod(x->x+2, sparse([2],[1],[1.0],4,1)) == 3*2^3
        @test prod(x->x+2, sparse([1,3],[1,1],[1.0,1.0],4,2)) == 3*3*2^6
        @test prod(x->x+2, sparse([4],[2],[1.0],4,3)) == 3*2^11
        @test mapreduce(x->x+2, (a,b)->a+b, sparse([2],[1],[1.0],2,2)) == 3+3*2
    end
    @testset "randomized parity with dense" begin
        Random.seed!(1234)
        f0 = x -> 1 + x/100
        for (m, n, d) in ((5000, 1, 0.001), (5000, 1, 0.2), (3000, 3, 0.01),
                          (50, 200, 0.05), (2049, 5, 0.3), (1024, 1024, 0.001))
            A = sprand(m, n, d)
            D = Array(A)
            for f in (identity, abs2, x->x+1), op in (+, max, (a,b)->a+b)
                @test mapreduce(f, op, A) ≈ mapreduce(f, op, D)
            end
            @test prod(f0, A) ≈ prod(f0, D)
            @test extrema(A) == extrema(D)
        end
    end
    @testset "row-wise (dims=2) reductions: single-pass rowmajor" begin
        Random.seed!(2468)
        for (m, n, d) in ((30, 40, 0.1), (2000, 100, 0.01), (7, 7, 0.9), (5, 1, 0.5),
                          (3, 2000, 0.001), (1500, 1500, 0.002))
            A = sprand(m, n, d)
            D = Array(A)
            @test sum(A; dims=2) ≈ sum(D; dims=2)
            @test isequal(maximum(A; dims=2), maximum(D; dims=2))
            @test sum(x->x+1, A; dims=2) ≈ sum(x->x+1, D; dims=2)
            @test prod(x->1+x/10, A; dims=2) ≈ prod(x->1+x/10, D; dims=2)
            @test count(!iszero, A; dims=2) == count(!iszero, D; dims=2)
            # in-place, both update semantics
            R = fill(1.0, m, 1)
            @test sum!(R, A) ≈ sum(D; dims=2)
            R = fill(1.0, m, 1)
            @test sum!(R, A; init=false) ≈ sum(D; dims=2) .+ 1
            R = fill(-Inf, m, 1)
            @test isequal(maximum!(R, A), maximum(D; dims=2))
            Rv = zeros(m)
            @test sum!(Rv, A) ≈ vec(sum(D; dims=2))
            S = A .> 0.5
            Rb = fill(false, m, 1)
            @test Base.mapreduce!(identity, |, Rb, S) == any(Array(S); dims=2)
            # an op not known to commute folds each row's chain in order, in the same pass
            @test mapreduce(x->2x, (a,b)->a+b, A; dims=2) ≈ mapreduce(x->2x, (a,b)->a+b, D; dims=2)
            @test mapreduce(string, *, A; dims=2) == mapreduce(string, *, D; dims=2)
        end
        @test sum(x->x+1, spzeros(3, 500); dims=2) == fill(500.0, 3, 1)
    end
    @testset "prod kernels: zero powers must not mask NaN/Inf" begin
        @test isequal(prod(SparseVector(3, [1], [NaN])), NaN)
        @test isequal(prod(SparseVector(3, [1], [Inf])), NaN)
        @test isequal(prod(sparse([1],[2],[NaN],2,2)), NaN)
        @test prod(SparseVector(2, [1], [-1.0])) === -0.0
        @test prod(spzeros(3, 3)) === 0.0
        @test prod(sparse([1],[1],[3],2,2)) === 0
        Random.seed!(8642)
        f0 = x -> 1 + x/100
        for (m, n, d) in ((4, 1, 0.5), (30, 40, 0.1), (2000, 10, 0.01), (100, 100, 0.9), (5000, 1, 0.001))
            A = sprand(m, n, d)
            D = Array(A)
            @test prod(A) == prod(D)
            @test prod(f0, A) ≈ prod(f0, D)
            @test prod(f0, A; dims=1) ≈ prod(f0, D; dims=1)
            v = sprand(m, d)
            dv = Vector(v)
            @test prod(v) == prod(dv)
            @test prod(f0, v) ≈ prod(f0, dv)
            @test prod(v; init=2.0) == prod(dv; init=2.0)
        end
    end
    @testset "init is honored on every reduction path" begin
        # Note: `init` must be a neutral element and may be used any number of
        # times, so only neutral inits can be compared against dense exactly
        Random.seed!(97531)
        for (m, n, d) in ((40, 40, 0.2), (2049, 3, 0.1))
            A = sprand(m, n, d)
            D = Array(A)
            @test mapreduce(x->x+1, +, A; init=0.0) ≈ mapreduce(x->x+1, +, D; init=0.0)
            @test mapreduce(abs2, max, A; init=-Inf) == mapreduce(abs2, max, D; init=-Inf)
            @test mapreduce(x->1+x/100, *, A; init=1.0) ≈ mapreduce(x->1+x/100, *, D; init=1.0)
            for dims in (1, 2, (1,2))
                @test mapreduce(abs2, +, A; dims, init=0.0) ≈ mapreduce(abs2, +, D; dims, init=0.0)
                @test mapreduce(identity, max, A; dims, init=-Inf) == mapreduce(identity, max, D; dims, init=-Inf)
            end
            v = sprand(m, d)
            dv = Vector(v)
            @test mapreduce(x->x+1, +, v; init=0.0) ≈ mapreduce(x->x+1, +, dv; init=0.0)
            @test sum(v; init=0.0) ≈ sum(dv; init=0.0)
            @test maximum(v; init=-Inf) == maximum(dv; init=-Inf)
        end
    end
    @testset "reduction cost scales with nnz, not logical length" begin
        # hypersparse: 10^10 logical elements, ~100 stored values. Any strategy
        # that walks — or even chunks over — the logical index space takes
        # seconds to minutes here, while the nnz-aware kernels take microseconds.
        # The generous bound keeps CI stable on slow machines.
        H = sprand(100_000, 100_000, 100/1e10)
        for r in (sum, maximum, minimum, prod, A->count(!iszero, A), extrema, findmin, findmax,
                  A->any(!iszero, A), A->all(iszero, A))
            r(H)  # compile & warm up
            t = @elapsed r(H)
            @test t < 0.5
        end
        @test sum(H) == sum(nonzeros(H))
        @test count(!iszero, H) == count(!iszero, nonzeros(H))
        # with f(0) != 0 the zero count shows up in the value, so this would
        # catch a zero count that overflowed a 32-bit Int
        @test sum(x->x+1, H) ≈ 1.0e10 + sum(nonzeros(H))
    end
    @testset "whole-array any/all parity with dense" begin
        Random.seed!(24680)
        for (m, n, d) in ((10, 10, 0.3), (100, 100, 0.01), (5, 1, 0.5), (3, 3, 1.0))
            A = sprand(Bool, m, n, d)
            D = Array(A)
            @test any(A) == any(D)
            @test all(A) == all(D)
            @test any(!, A) == any(!, D)
            @test all(!, A) == all(!, D)
            F = sprand(m, n, d)
            DF = Array(F)
            @test any(x -> x > 0.5, F) == any(x -> x > 0.5, DF)
            @test all(x -> x < 0.5, F) == all(x -> x < 0.5, DF)
        end
        # documented empty results, which mapreduce alone would error on
        @test any(spzeros(Bool, 0, 0)) == false
        @test all(spzeros(Bool, 0, 0)) == true
        @test any(spzeros(Bool, 0, 3)) == false
        @test all(spzeros(Bool, 3, 0)) == true
        # missing keeps its documented three-valued behavior
        for vals in ([missing, true], [missing, false])
            M = sparse([1,2], [1,2], Vector{Union{Missing,Bool}}(vals), 3, 3)
            DM = Array(M)
            @test isequal(any(M), any(DM))
            @test isequal(all(M), all(DM))
        end
    end
end

@testset "argmax, argmin, findmax, findmin" begin
    S = fixture(Float64, 9, 7)
    nonzeros(S)[2:3:end] .*= -1   # both extremes are stored entries
    A = Array(S)
    @test @inferred(argmax(S)) == argmax(A)
    @test @inferred(argmin(S)) == argmin(A)
    @test @inferred(findmin(S)) == findmin(A)
    @test @inferred(findmax(S)) == findmax(A)
    # a stored zero is a zero like those not stored: the first of either kind is the one found
    F = fixture(Float64, 5, 3)
    G = sparse([1, 3, 1, 2], [1, 1, 3, 3], [0.0, -2.0, -1.0, -3.0], 5, 3)   # the maximum is a zero
    for region in [(1,), (2,), (1,2)], m in [findmax, findmin]
        @test m(S, dims=region) == m(A, dims=region)
        @test m(F, dims=region) == m(Array(F), dims=region) && m(G, dims=region) == m(Array(G), dims=region)
    end
    @test argmin(F) == argmin(Array(F)) && argmax(G) == argmax(Array(G))
    @static if COMPREHENSIVE
    # a stored -0.0 is less than the zero of an entry that is not stored, as for dense input
    for Z in (sparse([1], [1], [-0.0], 2, 1), sparse([1, 2], [1, 2], [-0.0, -0.0], 2, 2),
              sparse([1, 2, 1, 2], [1, 1, 2, 2], [-0.0, -0.0, -0.0, -0.0], 2, 2),
              sparse([1, 2, 3], [1, 1, 2], [-0.0, 0.0, -0.0], 3, 2), sparse([1, 2, 1], [1, 1, 2], [0.0, -0.0, NaN], 3, 2)),
        m in (findmax, findmin)
        @test isequal(m(Z), m(Array(Z)))
        for region in (1, 2, (1,2))
            @test isequal(m(Z, dims=region), m(Array(Z), dims=region))
        end
    end
    # `missing`, which dense ranks above `NaN`
    D = Union{Missing,Float64}[NaN missing; 0.0 2.0]
    @test isequal(findmin(sparse(D)), findmin(D))
    @test isequal(findmax(sparse(D), dims=1), findmax(D, dims=1))
    end
    for m in [findmax, findmin]
        @test_throws MethodError m(S, (4, 3))   # no positional `dims`, as for dense input
    end
    S = spzeros(10,8)
    A = Array(S)
    @test argmax(S) == argmax(A) == CartesianIndex(1,1)
    @test argmin(S) == argmin(A) == CartesianIndex(1,1)
    @static if COMPREHENSIVE
    # along a dimension, each slice of a matrix that stores nothing reports its own first index
    @test all(m(S, dims=d) == m(A, dims=d) for m in (findmax, findmin), d in (1, 2))
    end

    A = @static COMPREHENSIVE ? Matrix{Int}(I, 0, 0) : zeros(0, 0)
    S = sparse(A)
    iA = try argmax(A); catch; end
    iS = try argmax(S); catch; end
    @test iA === iS === nothing
    iA = try argmin(A); catch; end
    iS = try argmin(S); catch; end
    @test iA === iS === nothing

    @testset "findmin/findmax parity with dense" begin
        Random.seed!(1357)
        cases = Any[
            sparse([1.0 2; 3 4]),                       # fully stored
            spzeros(3, 4),                              # all implicit zeros
            sparse([2],[2],[5.0],3,3),                  # single stored value
            sparse([2],[2],[-5.0],3,3),
            sparse([1,2],[1,2],[0.0, -0.0],2,2),        # stored (explicit) zeros
            sparse([1.0 NaN 6.0; NaN 2.0 4.0]),         # NaN, fully stored
            sparse([2,1],[1,2],[NaN, NaN],2,3),         # NaN among implicit zeros
            sparse([1],[3],[NaN],2,3),
            sparse([1,2],[1,1],[missing, 1.0],2,2),     # missing
            sparse([1],[1],[missing],2,2),
        ]
        for (m, n, d) in ((10, 10, 0.3), (50, 20, 0.1), (7, 3, 0.9),
                          (100, 100, 0.01), (1, 5, 0.5), (5, 1, 0.5))
            push!(cases, sprand(m, n, d))
            push!(cases, -sprand(m, n, d))
        end
        for A in cases
            D = Array(A)
            for r in (findmin, findmax)
                @test isequal(r(A), r(D))
                for dims in (1, 2, (1,2), (2,), 3, (1,3))
                    @test isequal(r(A; dims), r(D; dims))
                end
                if !(eltype(A) >: Missing)
                    for f in (abs2, x->-x)
                        @test isequal(r(f, A), r(f, D))
                        @test isequal(r(f, A; dims=1), r(f, D; dims=1))
                    end
                end
            end
            @test isequal(argmin(A), argmin(D))
            @test isequal(argmax(A), argmax(D))
            @test isequal(argmin(A; dims=2), argmin(D; dims=2))
        end
        # eltype must be preserved exactly like dense
        B = sparse([1],[1],Union{Missing,Float64}[2.0],3,3)
        DB = Array(B)
        @test typeof(findmin(B; dims=1)) == typeof(findmin(DB; dims=1))
        @test typeof(findmin(B)) == typeof(findmin(DB))
    end
end

@testset "findmin/findmax/minimum/maximum" begin
    A = sparse([1.0 5.0 6.0;
                5.0 2.0 4.0])
    for (tup, rval, rind) in [((1,), [1.0 2.0 4.0], [CartesianIndex(1,1) CartesianIndex(2,2) CartesianIndex(2,3)]),
                              ((2,), reshape([1.0,2.0], 2, 1), reshape([CartesianIndex(1,1),CartesianIndex(2,2)], 2, 1)),
                              ((1,2), fill(1.0,1,1),fill(CartesianIndex(1,1),1,1))]
        @test findmin(A, dims=tup) == (rval, rind)
    end

    for (tup, rval, rind) in [((1,), [5.0 5.0 6.0], [CartesianIndex(2,1) CartesianIndex(1,2) CartesianIndex(1,3)]),
                              ((2,), reshape([6.0,5.0], 2, 1), reshape([CartesianIndex(1,3),CartesianIndex(2,1)], 2, 1)),
                              ((1,2), fill(6.0,1,1),fill(CartesianIndex(1,3),1,1))]
        @test findmax(A, dims=tup) == (rval, rind)
    end

    #issue 23209

    A = sparse([1.0 5.0 6.0;
                NaN 2.0 4.0])
    for (tup, rval, rind) in [((1,), [NaN 2.0 4.0], [CartesianIndex(2,1) CartesianIndex(2,2) CartesianIndex(2,3)]),
                              ((2,), reshape([1.0, NaN], 2, 1), reshape([CartesianIndex(1,1),CartesianIndex(2,1)], 2, 1)),
                              ((1,2), fill(NaN,1,1),fill(CartesianIndex(2,1),1,1))]
        @test isequal(findmin(A, dims=tup), (rval, rind))
    end

    for (tup, rval, rind) in [((1,), [NaN 5.0 6.0], [CartesianIndex(2,1) CartesianIndex(1,2) CartesianIndex(1,3)]),
                              ((2,), reshape([6.0, NaN], 2, 1), reshape([CartesianIndex(1,3),CartesianIndex(2,1)], 2, 1)),
                              ((1,2), fill(NaN,1,1),fill(CartesianIndex(2,1),1,1))]
        @test isequal(findmax(A, dims=tup), (rval, rind))
    end

    A = sparse([1.0 NaN 6.0;
                NaN 2.0 4.0])
    for (tup, rval, rind) in [((1,), [NaN NaN 4.0], [CartesianIndex(2,1) CartesianIndex(1,2) CartesianIndex(2,3)]),
                              ((2,), reshape([NaN, NaN], 2, 1), reshape([CartesianIndex(1,2),CartesianIndex(2,1)], 2, 1)),
                              ((1,2), fill(NaN,1,1),fill(CartesianIndex(2,1),1,1))]
        @test isequal(findmin(A, dims=tup), (rval, rind))
    end

    for (tup, rval, rind) in [((1,), [NaN NaN 6.0], [CartesianIndex(2,1) CartesianIndex(1,2) CartesianIndex(1,3)]),
                              ((2,), reshape([NaN, NaN], 2, 1), reshape([CartesianIndex(1,2),CartesianIndex(2,1)], 2, 1)),
                              ((1,2), fill(NaN,1,1),fill(CartesianIndex(2,1),1,1))]
        @test isequal(findmax(A, dims=tup), (rval, rind))
    end

    A = sparse([Inf -Inf Inf  -Inf;
                Inf  Inf -Inf -Inf])
    for (tup, rval, rind) in [((1,), [Inf -Inf -Inf -Inf], [CartesianIndex(1,1) CartesianIndex(1,2) CartesianIndex(2,3) CartesianIndex(1,4)]),
                              ((2,), reshape([-Inf -Inf], 2, 1), reshape([CartesianIndex(1,2),CartesianIndex(2,3)], 2, 1)),
                              ((1,2), fill(-Inf,1,1),fill(CartesianIndex(1,2),1,1))]
        @test isequal(findmin(A, dims=tup), (rval, rind))
    end

    for (tup, rval, rind) in [((1,), [Inf Inf Inf -Inf], [CartesianIndex(1,1) CartesianIndex(2,2) CartesianIndex(1,3) CartesianIndex(1,4)]),
                              ((2,), reshape([Inf Inf], 2, 1), reshape([CartesianIndex(1,1),CartesianIndex(2,1)], 2, 1)),
                              ((1,2), fill(Inf,1,1),fill(CartesianIndex(1,1),1,1))]
        @test isequal(findmax(A, dims=tup), (rval, rind))
    end

    @static if COMPREHENSIVE
    A = sparse([BigInt(10)])
    for (tup, rval, rind) in [((2,), [BigInt(10)], [1])]
        @test isequal(findmin(A, dims=tup), (rval, rind))
    end

    for (tup, rval, rind) in [((2,), [BigInt(10)], [1])]
        @test isequal(findmax(A, dims=tup), (rval, rind))
    end

    A = sparse([BigInt(-10)])
    for (tup, rval, rind) in [((2,), [BigInt(-10)], [1])]
        @test isequal(findmin(A, dims=tup), (rval, rind))
    end

    for (tup, rval, rind) in [((2,), [BigInt(-10)], [1])]
        @test isequal(findmax(A, dims=tup), (rval, rind))
    end

    A = sparse([BigInt(10) BigInt(-10)])
    for (tup, rval, rind) in [((2,), reshape([BigInt(-10)], 1, 1), reshape([CartesianIndex(1,2)], 1, 1))]
        @test isequal(findmin(A, dims=tup), (rval, rind))
    end

    for (tup, rval, rind) in [((2,), reshape([BigInt(10)], 1, 1), reshape([CartesianIndex(1,1)], 1, 1))]
        @test isequal(findmax(A, dims=tup), (rval, rind))
    end

    # sparse arrays of types without zero(T) are forbidden
    @test_throws MethodError sparse(["a", "b"])
    end
end

# Support the case when user defined `zero` and `isless` for non-numerical type
@static if COMPREHENSIVE
@testset "findmin/findmax for non-numerical type" begin
    A = sparse([CustomType("a"), CustomType("b")])

    for (tup, rval, rind) in [((1,), [CustomType("a")], [1])]
        @test isequal(findmin(A, dims=tup), (rval, rind))
    end

    for (tup, rval, rind) in [((1,), [CustomType("b")], [2])]
        @test isequal(findmax(A, dims=tup), (rval, rind))
    end
end
end

@static if COMPREHENSIVE
@testset "any/all predicates over dims = 1" begin
    As = sparse([2, 3], [2, 3], [0.0, 1.0]) # empty, structural zero, non-zero
    Ad = Matrix(As)
    Bs = copy(As) # like As, but full column
    Bs[:,3] .= 1.0
    Bd = Matrix(Bs)
    Cs = copy(Bs) # like Bs, but full column is all structural zeros
    Cs[:,3] .= 0.0
    Cd = Matrix(Cs)

    @testset "any($(repr(pred)))" for pred in (iszero, !iszero, (@static COMPREHENSIVE ? (>(-1.0),) : ())...)
        @test any(pred, As, dims = 1) == any(pred, Ad, dims = 1)
        @test any(pred, Bs, dims = 1) == any(pred, Bd, dims = 1)
        @test any(pred, Cs, dims = 1) == any(pred, Cd, dims = 1)
    end
    @testset "all($(repr(pred)))" for pred in (iszero, !iszero, (@static COMPREHENSIVE ? (>(-1.0),) : ())...)
        @test all(pred, As, dims = 1) == all(pred, Ad, dims = 1)
        @test all(pred, Bs, dims = 1) == all(pred, Bd, dims = 1)
        @test all(pred, Cs, dims = 1) == all(pred, Cd, dims = 1)
    end
end
end

@testset "mapreducecols" begin
    n = 20
    m = 10
    A = fixture(Float64, n, m)
    B = mapreduce(identity, +, A, dims=2)
    for row in 1:n
        @test B[row] ≈ sum(A[row, :])
    end
    @test B ≈ mapreduce(identity, +, Matrix(A), dims=2)
    # case when f(0) =\= 0
    B = mapreduce(x->x+1, +, A, dims=2)
    for row in 1:n
        @test B[row] ≈ sum(A[row, :] .+ 1)
    end
    @test B ≈ mapreduce(x->x+1, +, Matrix(A), dims=2)
    @static if COMPREHENSIVE
    # case when there are no zeros in the sparse matrix
    A = sparse(reshape(Float64.(1:n*m), n, m))
    B = mapreduce(identity, +, A, dims=2)
    for row in 1:n
        @test B[row] ≈ sum(A[row, :])
    end
    @test B ≈ mapreduce(identity, +, Matrix(A), dims=2)
    # `f(0)` is not added to a row that stores every entry
    f = x -> x == 0 ? Inf : x
    D = [1.0 2.0; 0.0 3.0]
    @test isequal(mapreduce(f, +, sparse(D); dims=2), mapreduce(f, +, D; dims=2))
    end
end

# kept out of the testsets so that `@allocated` measures the call alone
mapreduce2(f::F, op::G, X, Y) where {F,G} = mapreduce(f, op, X, Y)

@testset "mapreduce of a binary function over two sparse arrays (#33), $T" for T in STD_ELTYPES
    f, g = (x, y) -> abs2(x - 2y), (x, y) -> x - 2y + 1   # g(0, 0) != 0
    x, xd = fixturepair(T, 6)
    yd = reverse(xd); yd[3] = 2; y = sparse(yd)
    A, Ad = fixturepair(T, 5, 3)
    Bd = reverse(Ad, dims=1); Bd[2, 1] = 2; B = sparse(Bd)
    for (X, Y) in ((x, y), (A, B))
        @test which(mapreduce, (typeof(f), typeof(+), typeof(X), typeof(Y))).module === SparseArrays
    end
    cases = Any[(x, xd, y, yd), (A, Ad, B, Bd)]
    @static if COMPREHENSIVE
    Z, F = spzeros(T, 5, 3), sparse(fixturedense(T, 5, 3))
    append!(cases, [
        (view(A, :, 2), Ad[:, 2], view(B, :, 3), Bd[:, 3]),             # column views
        (view(x, :), xd, y, yd),
        (view(A, :, [3, 1, 1]), Ad[:, [3, 1, 1]], B, Bd),               # a column subset
        (Z, Matrix(Z), Z, Matrix(Z)), (Z, Matrix(Z), B, Bd),            # nothing stored
        (F, Matrix(F), F, Matrix(F)), (A, Ad, F, Matrix(F)),            # every entry stored
        (x, xd, sparse([0, 3, 0, 0, 1, 0]), [0, 3, 0, 0, 1, 0]),        # two element types
        (spzeros(T, 1), zeros(T, 1), spzeros(T, 1), zeros(T, 1)),
    ])
    end
    # the anonymous `op` is not known to commute, and is folded in the order of the elements
    ops = (+, (@static COMPREHENSIVE ? (*, (a, b) -> a + b) : ())...)
    for (X, Xd, Y, Yd) in cases, h in (f, g), op in ops
        @test mapreduce(h, op, X, Y) ≈ mapreduce(h, op, Xd, Yd)
        @static if COMPREHENSIVE
        h === g && @test mapreduce(h, op, X, Y; init = one(T)) ≈ mapreduce(h, op, Xd, Yd; init = one(T))
        end
    end
    @static if COMPREHENSIVE
    if T <: Real
        for (X, Xd, Y, Yd) in cases, (h, op) in ((f, max), (g, min))
            @test mapreduce(h, op, X, Y) === mapreduce(h, op, Xd, Yd)
        end
    end
    # a product of matrices does not commute: the factors keep the order of the elements,
    # those at the entries neither array stores too, as for dense input
    block = (x, y) -> [1 x; y 2]
    @test !SparseArrays._commutes(block, *, T, T) && SparseArrays._commutes(g, *, T, T)
    @test mapreduce(block, *, A, B) ≈ mapreduce(block, *, Ad, Bd)
    @test mapreduce(block, *, x, y; init = T[1 2; 3 4]) ≈ mapreduce(block, *, xd, yd; init = T[1 2; 3 4])
    @test @inferred(mapreduce(f, +, x, y)) isa real(T)
    @test @inferred(mapreduce(g, +, A, B)) isa T
    # `f` is not evaluated at an entry that is not stored when every entry is
    full = sparse(fixturedense(T, 5, 3))
    @test mapreduce((x, y) -> iszero(x) ? error() : x * y, +, full, full) ≈ sum(Matrix(full) .^ 2)
    # no array of the mapped values, whatever the size
    big1, big2 = sprand(T, 10^4, 0.1), sprand(T, 10^4, 0.1)
    mapreduce2(f, +, x, y); mapreduce2(f, +, big1, big2)
    @test @allocated(mapreduce2(f, +, big1, big2)) == @allocated(mapreduce2(f, +, x, y))
    # left to Base's method
    @test mapreduce(g, +, spzeros(T, 0, 3), spzeros(T, 0, 3)) === mapreduce(g, +, zeros(T, 0, 3), zeros(T, 0, 3))
    @test_throws DimensionMismatch mapreduce(f, +, x, sparse(yd[1:5]))
    @test_throws DimensionMismatch mapreduce(f, +, A, sparse(Bd[:, 1:2]))
    @test mapreduce(g, +, A, B; dims = 1) ≈ mapreduce(g, +, Ad, Bd; dims = 1)
    @test mapreduce(g, +, A, B; dims = 2, sparse = true) ≈ mapreduce(g, +, Ad, Bd; dims = 2)
    end
end

@static if COMPREHENSIVE
@testset "mapreduce of a binary function: small integers as for dense" begin
    A = sparse(Int8[0 100 0; 3 0 -7])
    B = sparse(Int8[0 100 5; 0 0 -7])
    for h in (+, (x, y) -> x - y + Int8(1)), op in (+, *, Base.add_sum, Base.mul_prod)
        @test mapreduce(h, op, A, B) === mapreduce(h, op, Matrix(A), Matrix(B))
        @test mapreduce(h, op, A, B; init = 1) === mapreduce(h, op, Matrix(A), Matrix(B); init = 1)
    end
    # `init` widens the whole reduction, not one of two accumulators
    a, b = sparse(fill(Int8(100), 4)), spzeros(Int8, 4)
    for (X, Y) in ((a, b), (b, a), (a, a)), op in (+, *)
        @test mapreduce(+, op, X, Y; init = 1) === mapreduce(+, op, Vector(X), Vector(Y); init = 1)
    end
    # an element type without a zero, for arrays that store every entry
    c = SparseVector(2, [1, 2], Any[1, 2])
    C = SparseMatrixCSC(2, 1, [1, 3], [1, 2], Any[1, 2])
    @test mapreduce(+, +, c, c) === mapreduce(+, +, C, C) === 6
    @test mapreduce(+, +, c, sparse([3, 4]); init = 0.5) === 10.5
end
end

@testset "reductions along a dimension: dense by default, sparse with `sparse = true` (#43), column views (#377)" begin
    # (f, op); the last one has f(0) != 0, and `x != 0` under `&` holds for a full column only,
    # which is where the predicate kernel may not stop at the first entry not stored
    reductions = (
        (@static COMPREHENSIVE ? ((identity, *),) : ())...,
        (identity, +), (identity, max), (x -> x > 0, |), (x -> x != 0, &), (x -> x + 1, +),
    )
    viewed = reductions[@static COMPREHENSIVE ? [3, 4, 6] : [2, 5]]   # a view shares the kernels: one more `op` is enough
    @testset "size = ($m, $n), density = $d" for (m, n) in ((6, 5), (@static COMPREHENSIVE ? ((1, 1), (1, 9), (9, 1), (30, 20)) : ())...),
                                                 d in (0.0, 0.2, 1.0)
        # entries of both signs in about a fraction `d` of the positions; the first column is
        # full and negative, so that 0 is not a bound for its maximum
        val(i, j) = ((3i + 7j) % 11 - 5.25) / 11
        A = sparse([d > 0 && j == 1 ? -abs(val(i, j)) : d == 1 || (d > 0 && (i + 2j) % 5 == 0) ? val(i, j) : 0.0 for i in 1:m, j in 1:n])
        M = Matrix(A)
        V = view(A, :, (n + 1) ÷ 2:n)   # a view of a column range reduces like its copy (#377)
        C = A[:, (n + 1) ÷ 2:n]
        @test nnz(V) == nnz(C)
        for dims in (1, 2, (1, 2), 3), (f, op) in reductions
            rd = mapreduce(f, op, M; dims)
            r = mapreduce(f, op, A; dims)
            @test r isa Matrix && r ≈ rd
            if (f, op) in viewed
            rv, rc = mapreduce(f, op, V; dims), mapreduce(f, op, C; dims)
            @test typeof(rv) == typeof(rc) && isequal(rv, rc)
            end
            # opt-in: the sparse result has the element type and values of the dense one
            T = eltype(rd)
            rs = mapreduce(f, op, A; dims, sparse = true)
            @test rs isa SparseMatrixCSC{T} && mismatch(rs, rd; approx = true) === nothing
            if (f, op) in viewed
            rvs = mapreduce(f, op, V; dims, sparse = true)
            @test rvs isa SparseMatrixCSC{T} && mismatch(rvs, mapreduce(f, op, Matrix(C); dims); approx = true) === nothing
            end
        end
        for dims in (1, 2)
            @test mismatch(sum(A; dims, sparse = true), sum(M; dims); approx = true) === nothing
            @static if COMPREHENSIVE
            @test mismatch(sum(abs, V; dims, sparse = true), sum(abs, Matrix(C); dims); approx = true) === nothing
            end
            @test mismatch(prod(A; dims, sparse = true), prod(M; dims); approx = true) === nothing
            @test mismatch(maximum(A; dims, sparse = true), maximum(M; dims)) === nothing
            @static if COMPREHENSIVE
            @test mismatch(minimum(abs2, A; dims, sparse = true), minimum(abs2, M; dims)) === nothing
            @test mismatch(sum(A; dims, init = 2.5, sparse = true), sum(M; dims, init = 2.5); approx = true) === nothing
            @test mapreduce(abs, (x, y) -> x + y, A; dims, init = 1.5, sparse = true) ≈
                  mapreduce(abs, (x, y) -> x + y, M; dims, init = 1.5)
            end
            @test mismatch(count(>(0), A; dims, sparse = true), count(>(0), M; dims)) === nothing
            @static if COMPREHENSIVE
            @test mismatch(count(A .> 0; dims, sparse = true), count(M .> 0; dims)) === nothing
            end
            @test mismatch(count(A .> 0; dims, init = 3, sparse = true), count(M .> 0; dims, init = 3)) === nothing
            @static if COMPREHENSIVE
            @test mismatch(any(>(0), A; dims, sparse = true), any(>(0), M; dims)) === nothing
            end
            @test mismatch(any(A .> 0; dims, sparse = true), any(M .> 0; dims)) === nothing
            @static if COMPREHENSIVE
            @test mismatch(all(<(0.4), A; dims, sparse = true), all(<(0.4), M; dims)) === nothing
            end
            @test mismatch(all(A .< 0.4; dims, sparse = true), all(M .< 0.4; dims)) === nothing
            for r in (count(>(0), A; dims, sparse = true), any(A .> 0; dims, sparse = true), all(A .< 0.4; dims, sparse = true))
                @test r isa SparseMatrixCSC
            end
            # the default result and the scalar reductions are unchanged
            @test sum(A; dims) isa Matrix{Float64} && count(A .> 0; dims) isa Matrix{Int} && any(A .> 0; dims) isa Matrix{Bool}
        end
        @test sum(A) ≈ sum(M) && count(>(0), A) == count(>(0), M) && any(A .> 0) == any(M .> 0) && all(A .< 0.4) == all(M .< 0.4)
        @test_throws ArgumentError sum(A; sparse = true)
    end
    C = fixture(ComplexF64, 5, 3)
    MC, VC = Matrix(C), view(C, :, 2:3)
    for dims in (1, 2)
        @test sum(C; dims, sparse = true) isa SparseMatrixCSC{ComplexF64} && mismatch(sum(C; dims, sparse = true), sum(MC; dims); approx = true) === nothing
        # no entry of `C` equals its conjugate, so the adjoint and the transpose reduce differently
        @test mismatch(sum(C'; dims, sparse = true), sum(Array(C'); dims); approx = true) === nothing
        @static if COMPREHENSIVE
        @test mismatch(prod(abs2, C; dims, sparse = true), prod(abs2, MC; dims); approx = true) === nothing
        @test sum(VC; dims) isa Matrix{ComplexF64} && sum(VC; dims) == sum(Matrix(VC); dims)
        end
    end
    @static if COMPREHENSIVE
    @test any(Positive(), C .|> real; dims = 1, sparse = true) == any(Positive(), real.(MC); dims = 1)
    end
    @test_throws ArgumentError extrema(C; dims = 1, sparse = true)   # a tuple has no zero
    # only rows and columns that store something get an entry, unless a slice that stores
    # nothing reduces to something nonzero
    A = sparse([1, 2], [1, 1], [-1.0, 1.0], 4, 3)
    @test nnz(sum(A; dims = 1, sparse = true)) == 1   # a stored, cancelled zero
    @test nnz(sum(A; dims = 2, sparse = true)) == 2
    @test nnz(sum(x -> x + 1, A; dims = 2, sparse = true)) == 4
    @test nnz(sum(A; dims = 2, init = 1.0, sparse = true)) == 4
    @test nnz(prod(A; dims = 1, sparse = true)) == 1   # the product of an unstored column is 0
    @test mismatch(sum(A; dims = 2, sparse = true), sum(Matrix(A); dims = 2)) === nothing
    # the element type is that of the dense result
    @static if COMPREHENSIVE
    @test sum(sparse(Int8[1 2; 3 4]); dims = 1, sparse = true) isa SparseMatrixCSC{Int}
    end
    @test sum(sparse([true false]); dims = 2, sparse = true) isa SparseMatrixCSC{Int}
    @static if COMPREHENSIVE
    @test maximum(sparse(Int8[1 2; 3 4]); dims = 1, sparse = true) isa SparseMatrixCSC{Int8}
    @test sum(sparse(Int8[1 2; 3 4]); dims = 1, init = Int8(1), sparse = true) isa SparseMatrixCSC{Int8}
    end
    # empty dimensions
    for (m, n) in ((0, 4), (4, 0), (@static COMPREHENSIVE ? ((0, 0),) : ())...), dims in (1, 2, (1, 2))
        A = spzeros(m, n)
        @test sum(A; dims) == sum(Matrix(A); dims)
        for (g, gs) in ((X -> sum(X; dims), X -> sum(X; dims, sparse = true)),
                        (X -> prod(X; dims), X -> prod(X; dims, sparse = true)),
                        (X -> sum(x -> x + 1, X; dims), X -> sum(x -> x + 1, X; dims, sparse = true)),
                        (X -> all(X .> 0; dims), X -> all(X .> 0; dims, sparse = true)),
                        (X -> maximum(X; dims), X -> maximum(X; dims, sparse = true)))
            md = try g(Matrix(A)) catch err; err end   # may throw over an empty axis
            if md isa ArgumentError
                @test_throws ArgumentError gs(A)
            else
                @test mismatch(gs(A), md) === nothing
            end
        end
    end
    @test_throws ArgumentError sum(spzeros(3, 3); dims = 0, sparse = true)
    # the `sparse` keyword is folded away, so the result type is inferred for every argument
    # type that has the keyword, adjoints and transposes included (`@inferred` cannot pass
    # the keyword as a constant, so the opt-in goes through a function)
    A, v = fixture(Float64, 4, 3), fixturevec(Float64, 4)
    B = sparse(A .> 0)
    sparsesum(X, dims) = sum(X; dims, sparse = true)
    sparsecount(X, dims) = count(iszero, X; dims, sparse = true)
    sparseany(X, dims) = any(X; dims, sparse = true)
    b = sparse(v .> 0)
    @static if COMPREHENSIVE
    # every reduction for the argument types that the loop below leaves out
    for (X, P) in ((transpose(A), transpose(B)), (v', b')),
        dims in (1, 2)
        @test @inferred(sum(X; dims)) isa Array{Float64}
        @test @inferred(maximum(abs, X; dims)) isa Array{Float64}
        @test @inferred(count(iszero, X; dims)) isa Array{Int}
        @test @inferred(count(P; dims)) isa Array{Int}
        @test @inferred(any(iszero, X; dims)) isa Array{Bool}
        @test @inferred(all(P; dims)) isa Array{Bool}
        @test @inferred(sparsesum(X, dims)) isa AbstractSparseArray{Float64}
        @test @inferred(sparsecount(X, dims)) isa AbstractSparseArray{Int}
        @test @inferred(sparseany(P, dims)) isa AbstractSparseArray{Bool}
    end
    end
    for dims in (1, 2)   # one argument type per method with the keyword
        @test @inferred(along(sum, A, dims)) isa Array{Float64}
        @test @inferred(sparsesum(A, dims)) isa AbstractSparseArray{Float64}
        @test @inferred(along(all, B', dims)) isa Array{Bool}
        @test @inferred(sparsecount(A', dims)) isa AbstractSparseArray{Int}
        @test @inferred(along(sum, v, dims)) isa Array{Float64}
        @test @inferred(sparsesum(v, dims)) isa AbstractSparseArray{Float64}
        @test @inferred(along(count, transpose(b), dims)) isa Array{Int}
        @test @inferred(sparseany(transpose(b), dims)) isa AbstractSparseArray{Bool}
    end
    # hypersparse: only the rows that store something are visited, and each of them, the last
    # one included, folds two entries into one
    A = sparse([5, 10^6, 5, 10^6], [1, 2, 3, 3], [1.0, 2.0, 3.0, 4.0], 10^6, 3)
    r = sum(A; dims = 2, sparse = true)
    @test nnz(r) == 2 && r[5] == 4.0 && r[10^6] == 6.0
    @test mismatch(r, sum(Matrix(A); dims = 2)) === nothing
    @test mismatch(maximum(A; dims = 2, sparse = true), maximum(Matrix(A); dims = 2)) === nothing
    @test nnz(sum(A; dims = 1, sparse = true)) == 3
    sum(A; dims = 2, sparse = true)
    @test (@allocated sum(A; dims = 2, sparse = true)) < 2^12
    # a column-range view goes through the sparse kernels, not the element-wise fallback (#377)
    V = view(A, :, 2:3)
    @test which(Base.mapreduce_kernel, Base.typesof(identity, +, V, Base._InitialValue(), CartesianIndices(V))).module == SparseArrays
    @test which(Base.mapreduce_pairwise, Base.typesof(identity, +, V, Base._InitialValue(), CartesianIndices(V))).module == SparseArrays
    @test nnz(sum(V; dims = 2, sparse = true)) == 2
    # the adjoint of a sparse vector is a row that reduces through its parent
    w = sparsevec([2, 4], [1.5, -2.0], 6)
    @test which(Base.mapreducedim, Base.typesof(identity, +, w', Base._InitialValue(), 2)).module == SparseArrays
    @test which(Base.mapreducedim, Base.typesof(identity, +, transpose(w), Base._InitialValue(), 1)).module == SparseArrays
    @test nnz(sum(w'; dims = 1, sparse = true)) == 2 && nnz(sum(w'; dims = 2, sparse = true)) == 1
    @test nnz(sum(x -> x + 1, w'; dims = 1, sparse = true)) == 6 && nnz(sum(spzeros(5)'; dims = 2, sparse = true)) == 0
    # adjoints, views of a column subset and sparse vectors reduce like their copy, calling `f`
    # for the stored entries and once per slice rather than per element
    # 150 entries at distinct positions, a few in every row and column
    rows, cols = [mod1(7k, 60) for k in 1:150], [mod1(11k, 50) for k in 1:150]
    A = sparse(rows, cols, [(-1.0)^k * k for k in 1:150], 60, 50)
    C = sparse(rows, cols, [complex(k, 151.0 - 2k) for k in 1:150], 60, 50)
    v = sparsevec([2, 17, 43, 60], [1.0, -2.0, 0.5, 3.0], 60)
    c = sparsevec([2, 17, 43, 60], [1.0+2im, -2.0+im, 0.5-im, 3.0-2im], 60)
    S = view(A, :, [7, 2, 2, 15])
    # views that are not a column subset reduce through their copy (#56)
    G, R = view(C, [9, 2, 2, 40, 17], [3, 8, 8, 31]), view(A, 5:40, 2:49)
    maps = ((abs2, +), (abs, max), (x -> abs(x) + 1, (x, y) -> x + y))   # LinearAlgebra does not forward the last
    for (X, (f, op)) in (Any[A', maps[3]], Any[transpose(C), maps[3]], Any[c', maps[3]], Any[S, maps[2]], Any[v, maps[2]],
        (@static COMPREHENSIVE ? (Any[C', maps[1]], Any[G, maps[1]], Any[v', maps[3]]) : ())...), dims in (1, 2, (1, 2))
        calls = Ref(0)
        rd = mapreduce(f, op, Array(X); dims, init = 0.0)
        r = mapreduce(x -> (calls[] += 1; f(x)), op, X; dims, init = 0.0)
        @test r isa Array && r ≈ rd
        @test calls[] <= nnz(X isa SubArray ? copy(X) : X) + sum(size(X)) + 1
        rs = mapreduce(f, op, X; dims, init = 0.0, sparse = true)
        @test rs isa (X isa AbstractVector ? SparseVector{Float64} : SparseMatrixCSC{Float64}) && mismatch(rs, rd; approx = true) === nothing
    end
    @static if COMPREHENSIVE
    # every reduction for the kinds of argument that the loop below reduces once
    for X in (S, R, v, transpose(c)), dims in (1, 2)
        M = Array(X)
        @test sum(X; dims) isa Array && sum(X; dims) ≈ sum(M; dims)
        @test mismatch(prod(X; dims, sparse = true), prod(M; dims); approx = true) === nothing
        @test mismatch(count(!iszero, X; dims, sparse = true), count(!iszero, M; dims)) === nothing
        @test mismatch(any(!iszero, X; dims, sparse = true), any(!iszero, M; dims)) === nothing
        @test mismatch(all(iszero, X; dims, sparse = true), all(iszero, M; dims)) === nothing
    end
    end
    for dims in (1, 2)   # one of these reductions per argument type
        @test mismatch(prod(A'; dims, sparse = true), prod(Array(A'); dims); approx = true) === nothing
        @test mismatch(all(iszero, A'; dims, sparse = true), all(iszero, Array(A'); dims)) === nothing
        for X in (c', transpose(c))   # complex, so that the two differ and neither may conjugate for the other
            @test sum(X; dims) isa Array && sum(X; dims) ≈ sum(Array(X); dims)
        end
        @test mismatch(count(!iszero, v'; dims, sparse = true), count(!iszero, Array(v'); dims)) === nothing
        @test mismatch(any(!iszero, v'; dims, sparse = true), any(!iszero, Array(v'); dims)) === nothing
        @test sum(S; dims) isa Array && sum(S; dims) ≈ sum(Matrix(S); dims)
        @test mismatch(count(!iszero, R; dims, sparse = true), count(!iszero, Matrix(R); dims)) === nothing
    end
    @test sum(S) ≈ sum(Matrix(S)) && prod(x -> x + 1, S) ≈ prod(x -> x + 1, Matrix(S))
    @static if COMPREHENSIVE
    @test sum(G) ≈ sum(Matrix(G)) && maximum(abs, G) == maximum(abs, Matrix(G))
    end
    @test count(!iszero, R) == count(!iszero, Matrix(R))
    # a vector that stores something keeps an entry for a sum that cancels, and one that stores
    # nothing gets an entry when its sum is nonzero
    @test nnz(sum(sparsevec([1.0, -1.0]); dims = 1, sparse = true)) == 1 && nnz(sum(spzeros(5); dims = 1, sparse = true)) == 0
    @test mismatch(sum(x -> x + 1, spzeros(5); dims = 1, sparse = true), [5.0]) === nothing
    # reducing both dimensions of an adjoint keeps its element order for a non-commutative `op`
    firstnz(x, y) = iszero(x) ? y : x
    B = sparse([0.0 1.0; 2.0 0.0])
    @test mapreduce(identity, firstnz, B'; dims = (1, 2), init = 0) == [1;;] == mapreduce(identity, firstnz, B'; dims = (1, 2), init = 0, sparse = true)
    u = sparsevec([0.0, 2.0, 1.0])'
    @static if COMPREHENSIVE
    @test mapreduce(identity, firstnz, u; dims = (1, 2), init = 0) == [2;;] == mapreduce(identity, firstnz, u; dims = (1, 2), init = 0, sparse = true)
    end
    @test mapreduce(identity, firstnz, u; dims = 2, init = 0) == [2;;] == mapreduce(identity, firstnz, u; dims = 2, init = 0, sparse = true)
    @static if COMPREHENSIVE
    # the element type of the dense result for a `Union`, and no f(0) for a full matrix
    @test sum(sparse(Union{Int,Float64}[1.5 2; 3 4]); dims = 1, sparse = true) == [4.5 6.0]
    @test maximum(x -> 1 ÷ x, sparse([1 2; 3 4]); dims = 1, sparse = true) == [1 0]
    end
    # an empty column range outside the parent
    V = view(spzeros(4, 5), :, 10:9)
    @test nnz(V) == 0 && sum(V) == 0 && size(sum(V; dims = 1, sparse = true)) == (1, 0)
    @static if COMPREHENSIVE
    # a dimension beyond 2 maps the stored entries of a view only
    calls = Ref(0)
    @test mapreduce(x -> (calls[] += 1; x), +, view(A, :, [7, 2]); dims = 3, sparse = true) == A[:, [7, 2]]
    @test calls[] <= nnz(A) + 1
    end
end

end # module
