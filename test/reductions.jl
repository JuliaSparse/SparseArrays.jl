# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseReductionTests

using Test
using SparseArrays
using SparseArrays: getcolptr, nonzeroinds, _show_with_braille_patterns, _isnotzero, fixed, _is_fixed
using LinearAlgebra
using Random
using Test: guardseed
using InteractiveUtils: @which
include("forbidproperties.jl")

se33 = SparseMatrixCSC{Float64}(I, 3, 3)

sA = sprandn(3, 7, 0.5)
sC = similar(sA)
dA = Array(sA)

@testset "reductions" begin
    pA = sparse(rand(3, 7))
    p28227 = sparse(Real[0 0.5])

    for arr in (se33, sA, pA, p28227, spzeros(3, 3))
        farr = Array(arr)
        for f in (sum, prod, minimum, maximum)
            @test f(arr) ≈ f(farr)
            @test f(arr, dims=1) ≈ f(farr, dims=1)
            @test f(arr, dims=2) ≈ f(farr, dims=2)
            @test f(arr, dims=(1, 2)) ≈ [f(farr)]
            @test isequal(f(arr, dims=3), f(farr, dims=3))
        end
        for f in (+, *, min, max)
            @test mapreduce(identity, f, arr) ≈ mapreduce(identity, f, farr)
            @test mapreduce(x -> x + 1, f, arr) ≈ mapreduce(x -> x + 1, f, farr)
        end
    end

    for s0 in (spzeros(3, 7), spzeros(1, 3), spzeros(3, 1)), d in (1, 2, 3, (1,2))
        @test all(isone, sum(s0, dims=d, init=1.0))
    end

    for f in (sum, prod, minimum, maximum)
        # Test with a map function that maps to non-zero
        for arr in (se33, sA, pA)
            @test f(x->x+1, arr) ≈ f(arr .+ 1)
        end

        # case where f(0) would throw
        @test f(x->sqrt(x-1), pA .+ 1) ≈ f(sqrt.(pA))
        # `sum` still evaluates the map at the structural zero and throws here
        if f !== sum
            @test f(x->sqrt(x-1), pA .+ 1, dims=1) ≈ f(sqrt.(pA), dims=1)
            @test f(x->sqrt(x-1), pA .+ 1, dims=2) ≈ f(sqrt.(pA), dims=2)
            @test f(x->sqrt(x-1), pA .+ 1, dims=3) ≈ f(sqrt.(pA), dims=3)
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

    @testset "empty cases" begin
        errchecker(str) = occursin(": reducing over an empty collection is not allowed", str) ||
                          occursin(": reducing with ", str) ||
                          occursin("collection slices must be non-empty", str) ||
                          occursin("array slices must be non-empty", str)
        @test sum(sparse(Int[])) === 0
        @test prod(sparse(Int[])) === 1
        @test_throws errchecker minimum(sparse(Int[]))
        @test_throws errchecker maximum(sparse(Int[]))

        for f in (sum, prod)
            @test isequal(f(spzeros(0, 1), dims=1), f(Matrix{Int}(I, 0, 1), dims=1))
            @test isequal(f(spzeros(0, 1), dims=2), f(Matrix{Int}(I, 0, 1), dims=2))
            @test isequal(f(spzeros(0, 1), dims=(1, 2)), f(Matrix{Int}(I, 0, 1), dims=(1, 2)))
            @test isequal(f(spzeros(0, 1), dims=3), f(Matrix{Int}(I, 0, 1), dims=3))
        end
        for f in (minimum, maximum, findmin, findmax)
            @test_throws errchecker f(spzeros(0, 1), dims=1)
            @test isequal(f(spzeros(0, 1), dims=2), f(Matrix{Int}(I, 0, 1), dims=2))
            @test_throws errchecker f(spzeros(0, 1), dims=(1, 2))
            @test isequal(f(spzeros(0, 1), dims=3), f(Matrix{Int}(I, 0, 1), dims=3))
        end
    end
end

@testset "argmax, argmin, findmax, findmin" begin
    S = sprand(100,80, 0.5)
    A = Array(S)
    @test @inferred(argmax(S)) == argmax(A)
    @test @inferred(argmin(S)) == argmin(A)
    @test @inferred(findmin(S)) == findmin(A)
    @test @inferred(findmax(S)) == findmax(A)
    for region in [(1,), (2,), (1,2)], m in [findmax, findmin]
        @test m(S, dims=region) == m(A, dims=region)
    end
    for m in [findmax, findmin]
        @test_throws ArgumentError m(S, (4, 3))
    end
    S = spzeros(10,8)
    A = Array(S)
    @test argmax(S) == argmax(A) == CartesianIndex(1,1)
    @test argmin(S) == argmin(A) == CartesianIndex(1,1)

    A = Matrix{Int}(I, 0, 0)
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

# Support the case when user defined `zero` and `isless` for non-numerical type
struct CustomType
    x::String
end
Base.zero(::Type{CustomType}) = CustomType("")
Base.zero(x::CustomType) = zero(CustomType)
Base.isless(x::CustomType, y::CustomType) = isless(x.x, y.x)

@testset "findmin/findmax for non-numerical type" begin
    A = sparse([CustomType("a"), CustomType("b")])

    for (tup, rval, rind) in [((1,), [CustomType("a")], [1])]
        @test isequal(findmin(A, dims=tup), (rval, rind))
    end

    for (tup, rval, rind) in [((1,), [CustomType("b")], [2])]
        @test isequal(findmax(A, dims=tup), (rval, rind))
    end
end

@testset "any/all predicates over dims = 1" begin
    As = sparse([2, 3], [2, 3], [0.0, 1.0]) # empty, structural zero, non-zero
    Ad = Matrix(As)
    Bs = copy(As) # like As, but full column
    Bs[:,3] .= 1.0
    Bd = Matrix(Bs)
    Cs = copy(Bs) # like Bs, but full column is all structural zeros
    Cs[:,3] .= 0.0
    Cd = Matrix(Cs)

    @testset "any($(repr(pred)))" for pred in (iszero, !iszero, >(-1.0), !=(1.0))
        @test any(pred, As, dims = 1) == any(pred, Ad, dims = 1)
        @test any(pred, Bs, dims = 1) == any(pred, Bd, dims = 1)
        @test any(pred, Cs, dims = 1) == any(pred, Cd, dims = 1)
    end
    @testset "all($(repr(pred)))" for pred in (iszero, !iszero, >(-1.0), !=(1.0))
        @test all(pred, As, dims = 1) == all(pred, Ad, dims = 1)
        @test all(pred, Bs, dims = 1) == all(pred, Bd, dims = 1)
        @test all(pred, Cs, dims = 1) == all(pred, Cd, dims = 1)
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
    # case when there are no zeros in the sparse matrix
    A = sparse(rand(n, m))
    B = mapreduce(identity, +, A, dims=2)
    for row in 1:n
        @test B[row] ≈ sum(A[row, :])
    end
    @test B ≈ mapreduce(identity, +, Matrix(A), dims=2)
end

@testset "reductions along a dimension: dense by default, sparse with `sparse = true` (#43), column views (#377)" begin
    reductions = (   # (f, op); the last one has f(0) != 0
        (identity, +), (identity, *), (identity, max), (abs2, +), (x -> x > 0.5, |), (x -> x >= 0, &), (x -> x + 1, +),
    )
    @testset "size = ($m, $n), density = $d" for (m, n) in ((6, 5), (1, 1), (1, 9), (9, 1), (30, 20)),
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
            rv, rc = mapreduce(f, op, V; dims), mapreduce(f, op, C; dims)
            @test typeof(rv) == typeof(rc) && isequal(rv, rc)
            # opt-in: the sparse result has the element type and values of the dense one
            T = eltype(rd)
            rs = mapreduce(f, op, A; dims, sparse = true)
            @test rs isa SparseMatrixCSC{T} && rs ≈ rd
            rvs = mapreduce(f, op, V; dims, sparse = true)
            @test rvs isa SparseMatrixCSC{T} && rvs ≈ mapreduce(f, op, Matrix(C); dims)
        end
        for dims in (1, 2)
            @test sum(A; dims, sparse = true) ≈ sum(M; dims)
            @test sum(abs, V; dims, sparse = true) ≈ sum(abs, Matrix(C); dims)
            @test prod(A; dims, sparse = true) ≈ prod(M; dims)
            @test maximum(A; dims, sparse = true) == maximum(M; dims)
            @test minimum(abs2, A; dims, sparse = true) == minimum(abs2, M; dims)
            @test sum(A; dims, init = 2.5, sparse = true) ≈ sum(M; dims, init = 2.5)
            @test mapreduce(abs, (x, y) -> x + y, A; dims, init = 1.5, sparse = true) ≈
                  mapreduce(abs, (x, y) -> x + y, M; dims, init = 1.5)
            @test count(>(0), A; dims, sparse = true) == count(>(0), M; dims)
            @test count(A .> 0; dims, sparse = true) == count(M .> 0; dims)
            @test count(A .> 0; dims, init = 3, sparse = true) == count(M .> 0; dims, init = 3)
            @test any(>(0), A; dims, sparse = true) == any(>(0), M; dims)
            @test any(A .> 0; dims, sparse = true) == any(M .> 0; dims)
            @test all(<(0.4), A; dims, sparse = true) == all(<(0.4), M; dims)
            @test all(A .< 0.4; dims, sparse = true) == all(M .< 0.4; dims)
            for r in (count(>(0), A; dims, sparse = true), any(A .> 0; dims, sparse = true), all(A .< 0.4; dims, sparse = true))
                @test r isa SparseMatrixCSC
            end
            # the default result and the scalar reductions are unchanged
            @test sum(A; dims) isa Matrix{Float64} && count(A .> 0; dims) isa Matrix{Int} && any(A .> 0; dims) isa Matrix{Bool}
        end
        @test sum(A) ≈ sum(M) && count(>(0), A) == count(>(0), M) && any(A .> 0) == any(M .> 0) && all(A .< 0.4) == all(M .< 0.4)
        @test_throws ArgumentError sum(A; sparse = true)
    end
    C = sprand(ComplexF64, 6, 5, 0.3)
    MC, VC = Matrix(C), view(C, :, 2:5)
    for dims in (1, 2)
        @test sum(C; dims, sparse = true) isa SparseMatrixCSC{ComplexF64} && sum(C; dims, sparse = true) ≈ sum(MC; dims)
        @test prod(abs2, C; dims, sparse = true) ≈ prod(abs2, MC; dims)
        @test sum(VC; dims) isa Matrix{ComplexF64} && sum(VC; dims) == sum(Matrix(VC); dims)
    end
    struct Positive end   # a callable that is not a `Function`
    (::Positive)(x) = x > 0
    @test any(Positive(), C .|> real; dims = 1, sparse = true) == any(Positive(), real.(MC); dims = 1)
    @test_throws ArgumentError extrema(C; dims = 1, sparse = true)   # a tuple has no zero
    # only rows and columns that store something get an entry, unless a slice that stores
    # nothing reduces to something nonzero
    A = sparse([1, 2], [1, 1], [-1.0, 1.0], 4, 3)
    @test nnz(sum(A; dims = 1, sparse = true)) == 1   # a stored, cancelled zero
    @test nnz(sum(A; dims = 2, sparse = true)) == 2
    @test nnz(sum(x -> x + 1, A; dims = 2, sparse = true)) == 4
    @test nnz(sum(A; dims = 2, init = 1.0, sparse = true)) == 4
    @test nnz(prod(A; dims = 1, sparse = true)) == 1   # the product of an unstored column is 0
    @test sum(A; dims = 2, sparse = true) == sum(Matrix(A); dims = 2)
    # the element type is that of the dense result
    @test sum(sparse(Int8[1 2; 3 4]); dims = 1, sparse = true) isa SparseMatrixCSC{Int}
    @test sum(sparse([true false]); dims = 2, sparse = true) isa SparseMatrixCSC{Int}
    @test maximum(sparse(Int8[1 2; 3 4]); dims = 1, sparse = true) isa SparseMatrixCSC{Int8}
    @test sum(sparse(Int8[1 2; 3 4]); dims = 1, init = Int8(1), sparse = true) isa SparseMatrixCSC{Int8}
    # empty dimensions
    for (m, n) in ((0, 4), (4, 0), (0, 0)), dims in (1, 2, (1, 2))
        A = spzeros(m, n)
        @test sum(A; dims) == sum(Matrix(A); dims)
        @test sum(A; dims, sparse = true) == sum(Matrix(A); dims)
        @test prod(A; dims, sparse = true) == prod(Matrix(A); dims)
        @test sum(x -> x + 1, A; dims, sparse = true) == sum(x -> x + 1, Matrix(A); dims)
        @test all(A .> 0; dims, sparse = true) == all(Matrix(A) .> 0; dims)
        md = try maximum(Matrix(A); dims) catch err; err end   # throws over an empty axis
        if md isa ArgumentError
            @test_throws ArgumentError maximum(A; dims, sparse = true)
        else
            @test maximum(A; dims, sparse = true) == md
        end
    end
    @test_throws ArgumentError sum(spzeros(3, 3); dims = 0, sparse = true)
    # hypersparse: only the rows that store something are visited
    A = sparse([5, 10^6, 5], [1, 2, 3], [1.0, 2.0, 3.0], 10^6, 3)
    r = sum(A; dims = 2, sparse = true)
    @test nnz(r) == 2 && r[5] == 4.0 && r[10^6] == 2.0
    @test maximum(A; dims = 2, sparse = true) == maximum(Matrix(A); dims = 2)
    @test nnz(sum(A; dims = 1, sparse = true)) == 3
    sum(A; dims = 2, sparse = true)
    @test (@allocated sum(A; dims = 2, sparse = true)) < 2^12
    # a column-range view goes through the sparse kernels, not the element-wise fallback (#377)
    V = view(A, :, 2:3)
    @test (@which Base._mapreducedim!(identity, +, zeros(10^6, 1), V)).module == SparseArrays
    @test (@which Base._mapreduce(identity, +, IndexCartesian(), V)).module == SparseArrays
    @test nnz(sum(V; dims = 2, sparse = true)) == 2
    # adjoints, views of a column subset and sparse vectors reduce like their copy, calling `f`
    # for the stored entries and once per slice rather than per element
    A, C, v = sprand(60, 50, 0.05), sprand(ComplexF64, 60, 50, 0.05), sprand(60, 0.1)
    S = view(A, :, [7, 2, 2, 15])
    for X in (A', transpose(C), C', S, v), dims in (1, 2, (1, 2)),
        (f, op) in ((abs2, +), (abs, max), (x -> abs(x) + 1, (x, y) -> x + y))   # LinearAlgebra does not forward the last
        calls = Ref(0)
        rd = mapreduce(f, op, Array(X); dims, init = 0.0)
        r = mapreduce(x -> (calls[] += 1; f(x)), op, X; dims, init = 0.0)
        @test r isa Array && r ≈ rd
        @test calls[] <= nnz(X) + sum(size(X)) + 1
        rs = mapreduce(f, op, X; dims, init = 0.0, sparse = true)
        @test rs isa (X isa AbstractVector ? SparseVector{Float64} : SparseMatrixCSC{Float64}) && rs ≈ rd
    end
    for X in (A', S, v), dims in (1, 2)
        M = Array(X)
        @test sum(X; dims) isa Array && sum(X; dims) ≈ sum(M; dims)
        @test prod(X; dims, sparse = true) ≈ prod(M; dims)
        @test count(!iszero, X; dims, sparse = true) == count(!iszero, M; dims)
        @test any(!iszero, X; dims, sparse = true) == any(!iszero, M; dims)
        @test all(iszero, X; dims, sparse = true) == all(iszero, M; dims)
    end
    @test sum(S) ≈ sum(Matrix(S)) && prod(x -> x + 1, S) ≈ prod(x -> x + 1, Matrix(S))
    @test nnz(sum(v; dims = 1, sparse = true)) == 1 && nnz(sum(spzeros(5); dims = 1, sparse = true)) == 0
    # reducing both dimensions of an adjoint keeps its element order for a non-commutative `op`
    firstnz(x, y) = iszero(x) ? y : x
    B = sparse([0 1; 2 0])
    @test mapreduce(identity, firstnz, B'; dims = (1, 2), init = 0) == [1;;] == mapreduce(identity, firstnz, B'; dims = (1, 2), init = 0, sparse = true)
    # the element type of the dense result for a `Union`, and no f(0) for a full matrix
    @test sum(sparse(Union{Int,Float64}[1.5 2; 3 4]); dims = 1, sparse = true) == [4.5 6.0]
    @test maximum(x -> 1 ÷ x, sparse([1 2; 3 4]); dims = 1, sparse = true) == [1 0]
    # an empty column range outside the parent
    V = view(spzeros(4, 5), :, 10:9)
    @test nnz(V) == 0 && sum(V) == 0 && size(sum(V; dims = 1, sparse = true)) == (1, 0)
    # a dimension beyond 2 maps the stored entries of a view only
    calls = Ref(0)
    @test mapreduce(x -> (calls[] += 1; x), +, view(A, :, [7, 2]); dims = 3, sparse = true) == A[:, [7, 2]]
    @test calls[] <= nnz(A) + 1
end

end # module
