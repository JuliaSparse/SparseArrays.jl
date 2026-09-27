# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseFixedTests

using Test, SparseArrays, LinearAlgebra
using SparseArrays: AbstractSparseVector, AbstractSparseMatrixCSC, FixedSparseCSC, FixedSparseVector, ReadOnly,
    getcolptr, rowvals, nonzeros, nonzeroinds, _is_fixed, fixed, move_fixed, fkeep!, indtype
include("testhelpers.jl")

@testset "ReadOnly" begin
    v = randn(100)
    r = ReadOnly(v)
    @test length(r) == length(v)
    @test r == v
    @test v == r
    @test r == r
    @test ReadOnly([v v]) == ReadOnly([v r])
    @test copy(r)::ReadOnly == r
    @test ReadOnly(r) === r
    @test (resize!(r, length(r)); true)
    @test_throws ArgumentError resize!(r, length(r) - 1)
    @test_throws ArgumentError resize!(r, length(r) + 1)
    @test_throws ArgumentError r[1] = r[1] + 1
    @test_throws ArgumentError r[1] = r[1] - 1
    @test (r[1] = r[1]; true)
end

@testset "SparseMatrixCSC from readonly" begin

    # test that SparseMatrixCSC from readonly does copy
    A = sprandn(12, 11, 0.3)
    B = SparseMatrixCSC(size(A)..., ReadOnly(getcolptr(A)), ReadOnly(rowvals(A)), nonzeros(A))

    @test typeof(B) == typeof(A)
    @test A == B

    @test getcolptr(A) == getcolptr(B)
    @test getcolptr(A) !== getcolptr(B)
    @test rowvals(A) == rowvals(B)
    @test rowvals(A) !== rowvals(B)
    @test nonzeros(A) == nonzeros(B)
    @test nonzeros(A) === nonzeros(B)

    # a ReadOnly wrapping a sparse array forwards the sparse array interface
    for X in (A, A[:, 1], SparseMatrixCSC{Float64,Int32}(A))
        R = ReadOnly(X)
        @test issparse(R)
        @test nnz(R) == nnz(X)
        @test indtype(R) === indtype(X)
    end
end

@testset "FixedSparseCSC" begin
    A = sprandn(10, 10, 0.3)

    F = FixedSparseCSC(copy(A))
    Ft = FixedSparseCSC{eltype(A),eltype(rowvals(A))}(A)
    @test typeof(Ft) == typeof(F)
    @test Ft == F

    @test same_pattern(F, A)
    nonzeros(F) .= 0
    @test same_pattern(F, A)
    dropzeros!(F)
    @test same_pattern(F, A)
    H = F ./ 1
    @test typeof(H) == typeof(F)
    @test same_pattern(F, H, A)
    H = map!(zero, copy(F), F)
    @test same_pattern(F, H, A)
    @test_throws ArgumentError map!(x -> x + 1, H, F)
    @test_throws ArgumentError H .= F .+ 1
    G = sprandn(10, 10, 0.3)
    @test_throws ArgumentError map!(identity, H, G)
    @test_throws ArgumentError map!(+, H, A, G)
    @test_throws ArgumentError H .= A .+ A .+ G
    @test same_pattern(F, H, A)
    @test map!((x, y, z) -> x - y + z - z, H, A, A, A) == 0 .* A   # zeros stay stored
    @test same_pattern(F, H, A)
    F .= false
    @test same_pattern(F, H, A)
    F .= A .+ A
    @test F == A .+ A
    @test same_pattern(F, H, A)
    F .= A .- A
    @test F == A .- A
    @test same_pattern(F, H, A)
    F .= H .* A
    @test F == H .* A
    @test same_pattern(F, H, A)

    f1(F, A) = @allocated(F .= A .+ A)
    f1(F, A)
    @test f1(F, A) == 0

    f2(F, A) = @allocated(F .= A .- A)
    f2(F, A)
    @test f2(F, A) == 0

    f3(F, A, H) = @allocated(F .= H .* A)
    f3(F, A, H)
    f3(F, A, H)
    @test f3(F, A, H) == 0

    B = similar(F)
    @test typeof(B) == typeof(F)
    @test same_pattern(B, F)
    @test similar(F, 3, 3) isa SparseMatrixCSC
    @test typeof(FixedSparseCSC{Float32,Int32}(F)) == FixedSparseCSC{Float32,Int32}
    G = fixed(sparse([1, 2], [1, 2], [1.0, 2.0], 2, 2))
    @test_throws ArgumentError G[2, 1] = 1.0
    @test_throws ArgumentError G[:, 1] .= 1.0
    @test_throws ArgumentError G[1:2, 1:2] = ones(2, 2)
    @test_throws ArgumentError copyto!(G, sparse(ones(2, 2)))
    @test same_pattern(G, sparse(Diagonal([1.0, 2.0]))) && G == Diagonal([1.0, 2.0])
    G[1:2, 1:2] = [3 0; 0 4]
    G[:, 2] .= 0
    @test G == [3 0; 0 0] && nnz(G) == 2
    G .= sparse([2], [2], [6.0], 2, 2)   # a subset pattern zero-fills the rest
    @test G == [0 0; 0 6] && nnz(G) == 2
    @test circshift(G, (1, 0)) == circshift(Matrix(G), (1, 0))
    @test Diagonal([2.0, 3.0]) * G == [0 0; 0 18] && Symmetric(G) * G == [0 0; 0 36]
end
@testset "SparseMatrixCSC conversions" begin
    A = sprandn(10, 10, 0.3)
    F = fixed(copy(A))
    B = SparseMatrixCSC(F)
    @test A == B

    # fixed(x...)
    @test sparse(2I, 3, 3) == sparse(fixed(2I, 3, 3))
    @test SparseArrays._unsafe_unfix(A) == A
end
@testset "FixedSparseVector" begin
    y = sparsevec([2, 5, 7], [1.5, -2.0, 0.25], 10)
    x = FixedSparseVector(copy(y))
    @test same_pattern(x, y)
    @test_throws ArgumentError map!(v -> v + 1, x, y)
    @test_throws ArgumentError map!(identity, x, sparsevec([1, 5], [3.0, 4.0], 10))
    @test same_pattern(x, y)
    nonzeros(x) .= 0
    @test same_pattern(x, y)
    dropzeros!(x)
    @test same_pattern(x, y)
    z = x ./ 2
    @test same_pattern(x, y, z)
    f(x, y, z) = @allocated(x .= y .+ y) +
        @allocated(x .= y .- y) +
        @allocated(x .= z .* y)
    f(x, y, z)
    f(x, y, z)
    @test f(x, y, z) == 0
    t = similar(x)
    @test typeof(t) == typeof(x)
    @test same_pattern(t, x)
    @test similar(x, 5) isa SparseVector
    @test typeof(FixedSparseVector{Float32,Int32}(x)) == FixedSparseVector{Float32,Int32}
    w = fixed(sparsevec([1, 3], [1.0, 2.0], 4))
    @test_throws ArgumentError w[2] = 1.0
    @test_throws ArgumentError copyto!(w, sparsevec([2], [1.0], 4))
    @test same_pattern(w, sparsevec([1, 3], [1.0, 2.0], 4))
    w .= sparsevec([3], [5.0], 4)
    @test w == [0, 0, 5, 0] && nnz(w) == 2
end

@testset "Issue #190" begin
    J = move_fixed(sparse(Diagonal(ones(4))))
    W = move_fixed(sparse(Diagonal(ones(4))))
    J[4, 4] = 0
    gamma = 1.0
    W .= gamma .* J
    @test W == J

    x = move_fixed(sprandn(10, 10, 0.1))
    @test (x .= x .* 0; true)
    @test (x .= 0; true)
    @test (fill!(x, false); true)
end

@testset "`getindex`` should return type with same `_is_fixed`" begin
    for A in [sprandn(10, 10, 0.1), fixed(sprandn(10, 10, 0.1))]
        @test _is_fixed(A) == _is_fixed(A[:, :])
        @test _is_fixed(A) == _is_fixed(A[:, 1])
        @test _is_fixed(A) == _is_fixed(A[1, :])
        @test _is_fixed(A) == _is_fixed(A[1:2, 1:2])
        @test _is_fixed(A) == _is_fixed(A[2:4, 2:3])
    end
    for A in [sprandn(10, 0.1), fixed(sprandn(10, 0.1))]
        @test _is_fixed(A) == _is_fixed(A[:])
        @test _is_fixed(A) == _is_fixed(A[1:3])
    end
end

@testset "getindex with unsorted indices keeps the pattern read-only" begin
    S = sparse([1, 2, 3, 1], [1, 2, 3, 3], [1.0, 2.0, 3.0, 4.0])
    F = fixed(S)
    pattern = (copy(parent(getcolptr(F))), copy(parent(rowvals(F))), copy(nonzeros(F)))
    mask = [true, false, true]
    for (I, J) in (([2, 1], [1, 2]), ([3, 1, 2], :), (:, [3, 1]), ([2, 1], mask), (mask, [3, 1]),
                   ([3, 1, 2], 3))
        R = F[I, J]
        @test R == S[I, J]
        @test _is_fixed(R)
        @test size(R) == size(S[I, J])
    end
    @test (parent(getcolptr(F)), parent(rowvals(F)), nonzeros(F)) == pattern
end

@testset "cumsum, cumprod and accumulate return a writable copy" begin
    for T in (Float64, ComplexF64)
        S = sparse([1, 2, 3, 1], [1, 2, 3, 3], T[1, 2, 3, 4])
        F = fixed(S)
        pattern = (copy(parent(getcolptr(F))), copy(parent(rowvals(F))), copy(nonzeros(F)))
        @test which(cumsum, (typeof(F),)).module === SparseArrays
        @test which(cumprod, (typeof(F),)).module === SparseArrays
        @test which(accumulate, (typeof(+), typeof(F))).module === SparseArrays
        for d in (1, 2)
            for f in (cumsum, cumprod)
                R = f(F, dims=d)
                @test R == f(S, dims=d) == f(Array(S), dims=d)
                @test R isa SparseMatrixCSC{T} && !_is_fixed(R)
            end
            R = accumulate(+, F, dims=d, init=one(T))
            @test R == accumulate(+, S, dims=d, init=one(T)) == accumulate(+, Array(S), dims=d, init=one(T))
            @test R isa SparseMatrixCSC{T} && !_is_fixed(R)
        end
        @test accumulate(-, F) == accumulate(-, S)
        @test (parent(getcolptr(F)), parent(rowvals(F)), nonzeros(F)) == pattern

        s = sparsevec([1, 3], T[1, 2], 4)
        v = fixed(s)
        vpattern = (copy(parent(nonzeroinds(v))), copy(nonzeros(v)))
        @test which(cumsum, (typeof(v),)).module === SparseArrays
        for f in (cumsum, cumprod)
            r = f(v)
            @test r == f(s) == f(Array(s)) == f(v, dims=1)
            @test r isa SparseVector{T} && !_is_fixed(r)
        end
        r = accumulate(+, v, init=one(T))
        @test r == accumulate(+, s, init=one(T)) == accumulate(+, Array(s), init=one(T))
        @test r isa SparseVector{T} && !_is_fixed(r)
        @test (parent(nonzeroinds(v)), nonzeros(v)) == vpattern
    end
end

always_false(x...) = false
@testset "Test fkeep!" begin
    for a in [sprandn(10, 10, 0.99) + I, sprandn(10, 0.1) .+ 1]
        a = fixed(a)
        b = copy(a)
        fkeep!(always_false, b)
        @test nnz(a) == nnz(b)
        @test all(iszero, nonzeros(b))

    end
end

end # module
