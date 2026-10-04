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

se33 = SparseMatrixCSC{Float64}(I, 3, 3)

sA = sprandn(3, 7, 0.5)

@testset "reductions" begin
    pA = sparse(rand(3, 7))
    @static if COMPREHENSIVE
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

        # case where f(0) would throw
        @test f(x->sqrt(x-1), pA .+ 1) ≈ f(sqrt.(pA))
        # `sum` still evaluates the map at the structural zero and throws here
        if (@static COMPREHENSIVE ? f !== sum : f === maximum)
            @test f(x->sqrt(x-1), pA .+ 1, dims=1) ≈ f(sqrt.(pA), dims=1)
            @test f(x->sqrt(x-1), pA .+ 1, dims=2) ≈ f(sqrt.(pA), dims=2)
            @static if COMPREHENSIVE
            @test f(x->sqrt(x-1), pA .+ 1, dims=3) ≈ f(sqrt.(pA), dims=3)
            end
            @test f(x->sqrt(x-1), pA .+ 1; dims=1, sparse=true) ≈ f(sqrt.(pA), dims=1)
            @test f(x->sqrt(x-1), pA .+ 1; dims=2, sparse=true) ≈ f(sqrt.(pA), dims=2)
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
    end

    @testset "seed without copying the first slice" begin
        # the `Array` seed is built only for an empty slice; a non-empty reduction
        # allocates the result and Base's own temporary, not a dense copy of the slice
        A = sprand(10^5, 4, 0.01)
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
        for f in (maximum, findmax, (@static COMPREHENSIVE ? (minimum, findmin) : ())...)
            @test_throws errchecker f(spzeros(0, 1), dims=1)
            @test isequal(f(spzeros(0, 1), dims=2), f(Matrix{Int}(I, 0, 1), dims=2))
            @test_throws errchecker f(spzeros(0, 1), dims=(1, 2))
            @test isequal(f(spzeros(0, 1), dims=3), f(Matrix{Int}(I, 0, 1), dims=3))
        end
        # the result along a dimension of an empty array is dense, as for dense input, and is
        # inferred as such: Base takes `map` of the empty first slice, which would be sparse
        E = spzeros(3, 0)
        for (X, dims, f) in (Any[E, 1, minimum], Any[E', 2, maximum], Any[spzeros(0), 2, extrema], Any[spzeros(0)', 1, minimum],
            (@static COMPREHENSIVE ? (Any[E, 3, maximum], Any[view(E, :, 1:0), 1, extrema], Any[transpose(spzeros(0)), 3, maximum]) : ())...)
            r, rd = f(X; dims), f(Array(X); dims)
            @test typeof(r) == typeof(rd) && size(r) == size(rd)
            @test typeof(@inferred along(f, X, dims)) == typeof(rd)
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

@testset "argmax, argmin, findmax, findmin" begin
    S = sprand(100,80, 0.5)
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
    end
    for m in [findmax, findmin]
        @test_throws ArgumentError m(S, (4, 3))
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
end

@testset "findmin/findmax/minimum/maximum" begin
    A = sparse([1.0 5.0 6.0;
                5.0 2.0 4.0])
    for (tup, rval, rind) in [((1,), [1.0 2.0 4.0], [CartesianIndex(1,1) CartesianIndex(2,2) CartesianIndex(2,3)]),
                              ((2,), reshape([1.0,2.0], 2, 1), reshape([CartesianIndex(1,1),CartesianIndex(2,2)], 2, 1)),
                              ((1,2), fill(1.0,1,1),fill(CartesianIndex(1,1),1,1))]
        @test findmin(A, tup) == (rval, rind)
    end

    for (tup, rval, rind) in [((1,), [5.0 5.0 6.0], [CartesianIndex(2,1) CartesianIndex(1,2) CartesianIndex(1,3)]),
                              ((2,), reshape([6.0,5.0], 2, 1), reshape([CartesianIndex(1,3),CartesianIndex(2,1)], 2, 1)),
                              ((1,2), fill(6.0,1,1),fill(CartesianIndex(1,3),1,1))]
        @test findmax(A, tup) == (rval, rind)
    end

    #issue 23209

    A = sparse([1.0 5.0 6.0;
                NaN 2.0 4.0])
    for (tup, rval, rind) in [((1,), [NaN 2.0 4.0], [CartesianIndex(2,1) CartesianIndex(2,2) CartesianIndex(2,3)]),
                              ((2,), reshape([1.0, NaN], 2, 1), reshape([CartesianIndex(1,1),CartesianIndex(2,1)], 2, 1)),
                              ((1,2), fill(NaN,1,1),fill(CartesianIndex(2,1),1,1))]
        @test isequal(findmin(A, tup), (rval, rind))
    end

    for (tup, rval, rind) in [((1,), [NaN 5.0 6.0], [CartesianIndex(2,1) CartesianIndex(1,2) CartesianIndex(1,3)]),
                              ((2,), reshape([6.0, NaN], 2, 1), reshape([CartesianIndex(1,3),CartesianIndex(2,1)], 2, 1)),
                              ((1,2), fill(NaN,1,1),fill(CartesianIndex(2,1),1,1))]
        @test isequal(findmax(A, tup), (rval, rind))
    end

    A = sparse([1.0 NaN 6.0;
                NaN 2.0 4.0])
    for (tup, rval, rind) in [((1,), [NaN NaN 4.0], [CartesianIndex(2,1) CartesianIndex(1,2) CartesianIndex(2,3)]),
                              ((2,), reshape([NaN, NaN], 2, 1), reshape([CartesianIndex(1,2),CartesianIndex(2,1)], 2, 1)),
                              ((1,2), fill(NaN,1,1),fill(CartesianIndex(2,1),1,1))]
        @test isequal(findmin(A, tup), (rval, rind))
    end

    for (tup, rval, rind) in [((1,), [NaN NaN 6.0], [CartesianIndex(2,1) CartesianIndex(1,2) CartesianIndex(1,3)]),
                              ((2,), reshape([NaN, NaN], 2, 1), reshape([CartesianIndex(1,2),CartesianIndex(2,1)], 2, 1)),
                              ((1,2), fill(NaN,1,1),fill(CartesianIndex(2,1),1,1))]
        @test isequal(findmax(A, tup), (rval, rind))
    end

    A = sparse([Inf -Inf Inf  -Inf;
                Inf  Inf -Inf -Inf])
    for (tup, rval, rind) in [((1,), [Inf -Inf -Inf -Inf], [CartesianIndex(1,1) CartesianIndex(1,2) CartesianIndex(2,3) CartesianIndex(1,4)]),
                              ((2,), reshape([-Inf -Inf], 2, 1), reshape([CartesianIndex(1,2),CartesianIndex(2,3)], 2, 1)),
                              ((1,2), fill(-Inf,1,1),fill(CartesianIndex(1,2),1,1))]
        @test isequal(findmin(A, tup), (rval, rind))
    end

    for (tup, rval, rind) in [((1,), [Inf Inf Inf -Inf], [CartesianIndex(1,1) CartesianIndex(2,2) CartesianIndex(1,3) CartesianIndex(1,4)]),
                              ((2,), reshape([Inf Inf], 2, 1), reshape([CartesianIndex(1,1),CartesianIndex(2,1)], 2, 1)),
                              ((1,2), fill(Inf,1,1),fill(CartesianIndex(1,1),1,1))]
        @test isequal(findmax(A, tup), (rval, rind))
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
    A = sprand(n, m, 0.2)
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
    A = sparse(rand(n, m))
    B = mapreduce(identity, +, A, dims=2)
    for row in 1:n
        @test B[row] ≈ sum(A[row, :])
    end
    @test B ≈ mapreduce(identity, +, Matrix(A), dims=2)
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
        A = sparse(sprand(m, n, d) .- 0.5)   # negative entries, so that max and min do not see 0 as a bound
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
        @test mismatch(sum(A; dims, sparse = true), sum(Matrix(A); dims)) === nothing
        @test mismatch(prod(A; dims, sparse = true), prod(Matrix(A); dims)) === nothing
        @test mismatch(sum(x -> x + 1, A; dims, sparse = true), sum(x -> x + 1, Matrix(A); dims)) === nothing
        @test mismatch(all(A .> 0; dims, sparse = true), all(Matrix(A) .> 0; dims)) === nothing
        md = try maximum(Matrix(A); dims) catch err; err end   # throws over an empty axis
        if md isa ArgumentError
            @test_throws ArgumentError maximum(A; dims, sparse = true)
        else
            @test mismatch(maximum(A; dims, sparse = true), md) === nothing
        end
    end
    @test_throws ArgumentError sum(spzeros(3, 3); dims = 0, sparse = true)
    # the `sparse` keyword is folded away, so the result type is inferred for every argument
    # type that has the keyword, adjoints and transposes included (`@inferred` cannot pass
    # the keyword as a constant, so the opt-in goes through a function)
    A, v = sprand(4, 3, 0.5), sprand(4, 0.5)
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
    @test which(Base._mapreducedim!, Base.typesof(identity, +, zeros(1, 1), V)).module == SparseArrays
    @test which(Base._mapreduce, Base.typesof(identity, +, IndexCartesian(), V)).module == SparseArrays
    @test nnz(sum(V; dims = 2, sparse = true)) == 2
    # the adjoint of a sparse vector is a row that reduces through its parent
    w = sparsevec([2, 4], [1.5, -2.0], 6)
    @test which(Base._mapreducedim!, Base.typesof(identity, +, zeros(1, 1), w')).module == SparseArrays
    @test which(Base._mapreducedim!, Base.typesof(identity, +, zeros(1, 6), transpose(w))).module == SparseArrays
    @test nnz(sum(w'; dims = 1, sparse = true)) == 2 && nnz(sum(w'; dims = 2, sparse = true)) == 1
    @test nnz(sum(x -> x + 1, w'; dims = 1, sparse = true)) == 6 && nnz(sum(spzeros(5)'; dims = 2, sparse = true)) == 0
    # adjoints, views of a column subset and sparse vectors reduce like their copy, calling `f`
    # for the stored entries and once per slice rather than per element
    A, C = sprand(60, 50, 0.05), sprand(ComplexF64, 60, 50, 0.05)
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
