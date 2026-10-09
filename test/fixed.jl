# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseFixedTests

using Test, SparseArrays, LinearAlgebra
using SparseArrays: AbstractSparseVector, AbstractSparseMatrixCSC, FixedSparseCSC, FixedSparseVector, ReadOnly,
    getcolptr, rowvals, nonzeros, nonzeroinds, _is_fixed, fixed, move_fixed, fkeep!, indtype
include("testhelpers.jl")

@testset "ReadOnly" begin
    v = Float64.(1:100)
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
    A = fixture(Float64, 5, 3)
    B = SparseMatrixCSC(size(A)..., ReadOnly(getcolptr(A)), ReadOnly(rowvals(A)), nonzeros(A))

    @test typeof(B) == typeof(A)
    @test mismatch(B, Array(A)) === nothing

    @test getcolptr(A) == getcolptr(B)
    @test getcolptr(A) !== getcolptr(B)
    @test rowvals(A) == rowvals(B)
    @test rowvals(A) !== rowvals(B)
    @test nonzeros(A) == nonzeros(B)
    @test nonzeros(A) === nonzeros(B)

    # a ReadOnly wrapping a sparse array forwards the sparse array interface
    for X in (A, A[:, 1], (@static COMPREHENSIVE ? (SparseMatrixCSC{Float64,Int32}(A),) : ())...)
        R = ReadOnly(X)
        @test issparse(R)
        @test nnz(R) == nnz(X)
        @test indtype(R) === indtype(X)
    end
end

@testset "FixedSparseCSC" begin
    # the stored zero makes a kernel that takes the destination for an ordinary matrix
    # drop an entry
    A = fixture(Float64, 5, 3)

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
    @static if COMPREHENSIVE
    @test_throws ArgumentError H .= F .+ 1
    end
    G = sparse(ones(5, 3))
    @test_throws ArgumentError map!(identity, H, G)
    @test_throws ArgumentError H .= G .+ G
    @static if COMPREHENSIVE
    @test_throws ArgumentError map!(+, H, A, G)
    @test_throws ArgumentError H .= A .+ A .+ G
    @test same_pattern(F, H, A)
    @test map!((x, y, z) -> x - y + z - z, H, A, A, A) == 0 .* A   # zeros stay stored
    end
    @test same_pattern(F, H, A)
    F .= false
    @test same_pattern(F, H, A)
    F .= A .+ A
    @test mismatch(F, Array(A .+ A)) === nothing
    @test same_pattern(F, H, A)
    @static if COMPREHENSIVE
    F .= H .* A
    @test mismatch(F, Array(H .* A)) === nothing
    @test same_pattern(F, H, A)
    end

    f1(F, A) = @allocated(F .= A .+ A)
    f1(F, A)
    @test f1(F, A) == 0

    @static if COMPREHENSIVE
    f3(F, A, H) = @allocated(F .= H .* A)
    f3(F, A, H)
    f3(F, A, H)
    @test f3(F, A, H) == 0
    end

    B = similar(F)
    @test typeof(B) == typeof(F)
    @test same_pattern(B, F)
    @test similar(F, 3, 3) isa SparseMatrixCSC
    @test typeof(FixedSparseCSC{ComplexF64,Int}(F)) == FixedSparseCSC{ComplexF64,Int}
    @static if COMPREHENSIVE
    @test typeof(FixedSparseCSC{Float32,Int32}(F)) == FixedSparseCSC{Float32,Int32}
    end
    G = fixed(sparse([1, 2], [1, 2], [1.0, 2.0], 2, 2))
    @test_throws ArgumentError G[2, 1] = 1.0
    @test_throws ArgumentError G[:, 1] .= 1.0
    @test_throws ArgumentError G[1:2, 1:2] = ones(2, 2)
    # the entry outside the pattern lies above a stored one in its column, and is not to
    # be written there
    @test_throws ArgumentError copyto!(G, sparse([1], [2], [5.0], 2, 2))
    @test same_pattern(G, sparse(Diagonal([1.0, 2.0]))) && G == Diagonal([1.0, 2.0])
    G[1:2, 1:2] = [3 0; 0 4]
    G[:, 2] .= 0
    @test G == [3 0; 0 0] && nnz(G) == 2
    G .= sparse([2], [2], [6.0], 2, 2)   # a subset pattern zero-fills the rest
    @test G == [0 0; 0 6] && nnz(G) == 2
    # a fixed destination of circshift! must already hold the shifted pattern
    @test_throws ArgumentError circshift!(G, G, (1, 0))
    @test G == [0 0; 0 6] && nnz(G) == 2
    H = fixed(circshift(G, (1, 1)))
    @test circshift!(H, G, (1, 1)) === H && H == Diagonal([6.0, 0.0]) && nnz(H) == 2
    @test Diagonal([2.0, 3.0]) * G == [0 0; 0 18] && (@static COMPREHENSIVE ? Symmetric(G) * G == [0 0; 0 36] : true)
end
@testset "SparseMatrixCSC conversions" begin
    A = fixture(Float64, 3, 5)
    F = fixed(copy(A))
    B = SparseMatrixCSC(F)
    @test mismatch(B, Array(A)) === nothing && same_pattern(B, A) && !_is_fixed(B)

    # fixed(x...)
    @test sparse(2I, 3, 5) == sparse(fixed(2I, 3, 5))
    @test SparseArrays._unsafe_unfix(A) == A

    @static if COMPREHENSIVE
    # every constructor of a plain matrix returns one, which shares nothing with `F`
    for T in (SparseMatrixCSC, SparseMatrixCSC{Float64}, SparseMatrixCSC{Float64,Int}, SparseMatrixCSC{ComplexF64})
        B = T(F)
        @test B isa SparseMatrixCSC && !_is_fixed(B) && B == A
        B[3, 1] = 1; nonzeros(B) .= 0
        @test exact_equal(SparseMatrixCSC(F), A)
        @test convert(T, F) isa SparseMatrixCSC
    end
    # the structure-preserving forms of `similar` stay fixed, the others are writable
    @test similar(F, Float32, Int) isa FixedSparseCSC{Float32,Int} && same_pattern(similar(F, Float32, Int), F)
    @test similar(F, Float32, Int, 3, 5) isa SparseMatrixCSC{Float32,Int}
    end
end

@testset "fixed and FixedSparseCSC copy a writable array" begin
    A = fixture(Float64, 3, 5); A0 = copy(A)
    x = sparsevec([1, 3], [1.0, 2.0], 4); x0 = copy(x)
    cases = (@static COMPREHENSIVE ? (fixed, FixedSparseCSC, FixedSparseCSC{Float64,Int}) : (fixed,))
    for T in cases
        F = T(A)
        @test _is_fixed(F) && F == A0 && same_pattern(F, A0)
        A[2, 1] = 7.0; SparseArrays.dropstored!(A, 1, 1); nonzeros(A) .= 3.0
        @test F == A0 && same_pattern(F, A0)
        nonzeros(F) .= 4.0
        @test all(==(3.0), nonzeros(A))
        A = copy(A0)
    end
    f = fixed(x)
    x[2] = 7.0
    @test f == x0 && nonzeroinds(f) == [1, 3]
    # `move_fixed` reuses the buffers, and a fixed array shares its pattern with its copy
    M = move_fixed(A)
    @test getcolptr(M) == getcolptr(A) && parent(getcolptr(M)) === getcolptr(A) && nonzeros(M) === nonzeros(A)
    @static if COMPREHENSIVE
    G = fixed(M)
    @test getcolptr(G) === getcolptr(M) && rowvals(G) === rowvals(M) && nonzeros(G) !== nonzeros(M)
    end
end

@static if COMPREHENSIVE
@testset "copying a fixed array onto itself" begin
    A = sparse([1, 2], [1, 2], [1.0, 2.0]); x = sparsevec([1, 3], [1.0, 2.0], 4)
    for (F, expected) in ((fixed(A), copy(A)), (fixed(x), copy(x)))
        @test copyto!(F, F) === F && F == expected
        @test copy!(F, F) === F && F == expected
        @test (F .= F) === F && F == expected
        # a writable array on the same buffers
        @test copyto!(F, SparseArrays._unsafe_unfix(F)) === F && F == expected
    end
end
end
@testset "FixedSparseVector" begin
    y = fixturevec(Float64, 10)
    yvals = copy(nonzeros(y))
    x = FixedSparseVector(y)
    @test same_pattern(x, y)
    @test typeof(FixedSparseVector{Float64,Int}(y)) == typeof(x)
    @test_throws ArgumentError map!(v -> v + 1, x, y)
    @test_throws ArgumentError map!(identity, x, sparsevec([3, 5], [3.0, 4.0], 10))
    @test same_pattern(x, y)
    nonzeros(x) .= 0
    # a fixed vector made from `y` owns its buffers
    @test same_pattern(x, y) && nonzeros(y) == yvals
    dropzeros!(x)
    @test same_pattern(x, y)
    z = x ./ 2
    @test same_pattern(x, y, z)
    f(x, y, z) = @allocated(x .= y .+ y) +
        (@static COMPREHENSIVE ? @allocated(x .= y .- y) : 0)
    f(x, y, z)
    f(x, y, z)
    @test f(x, y, z) == 0
    t = similar(x)
    @test typeof(t) == typeof(x)
    @test same_pattern(t, x)
    @test similar(x, 5) isa SparseVector
    @static if COMPREHENSIVE
    @test typeof(FixedSparseVector{Float32,Int32}(x)) == FixedSparseVector{Float32,Int32}
    end
    w = fixed(sparsevec([1, 3], [1.0, 2.0], 4))
    @test_throws ArgumentError w[2] = 1.0
    @test_throws ArgumentError copyto!(w, sparsevec([2], [1.0], 4))
    @test same_pattern(w, sparsevec([1, 3], [1.0, 2.0], 4)) && nonzeros(w) == [1.0, 2.0]
    w .= sparsevec([3], [5.0], 4)
    @test w == [0, 0, 5, 0] && nnz(w) == 2
    r = real(fixed(sparsevec([1, 3], ComplexF64[1 + 2im, 3], 4)))
    @test r isa SparseVector{Float64,Int}
    @test r == [1, 0, 3, 0] && nonzeroinds(r) == [1, 3]
    @static if COMPREHENSIVE
    r = imag(fixed(sparsevec([1, 3], ComplexF64[1 + 2im, 3], 4)))
    @test r isa SparseVector{Float64,Int}
    @test r == [2, 0, 0, 0] && nonzeroinds(r) == [1, 3]
    end
end

@testset "structure queries and unary maps keep the fixed pattern" begin
    # `issymmetric`/`ishermitian` advance a copy of the column pointers, which are
    # read-only in a fixed matrix
    S = fixed(sparse([4.0 1 0; 1 4 1; 0 1 4]))
    @test issymmetric(S) && ishermitian(S)
    @test !issymmetric(fixed(sparse([4.0 1 0; 2 4 1; 0 1 4])))
    @static if COMPREHENSIVE
    @test ishermitian(fixed(sparse(ComplexF64[4 1+im 0; 1-im 4 1; 0 1 4])))
    @test !issymmetric(fixed(sparse(ComplexF64[4 1+im 0; 1-im 4 1; 0 1 4])))
    end
    A = sparse([1, 2, 2], [1, 1, 2], (@static COMPREHENSIVE ? [1, 2, 0] : [1.0, 2.0, 0.0]), 3, 3)
    x = sparsevec([1, 3], (@static COMPREHENSIVE ? [1, 0] : [1.0, 0.0]), 4)
    F, y = fixed(A), fixed(x)
    @test which(-, (typeof(F),)).module === SparseArrays
    for f in (-, complex, (@static COMPREHENSIVE ? (float,) : ())...)
        R, r = f(F), f(y)
        @test R isa FixedSparseCSC && r isa FixedSparseVector
        @test R == f(A) && same_pattern(R, A) && r == f(x) && same_pattern(r, x)
    end
    @test getcolptr(-F) === getcolptr(F) && nonzeroinds(-y) === nonzeroinds(y)
end

@static if COMPREHENSIVE
@testset "Issue #190" begin
    J = move_fixed(sparse(Diagonal(ones(4))))
    W = move_fixed(sparse(Diagonal(ones(4))))
    J[4, 4] = 0
    gamma = 1.0
    W .= gamma .* J
    @test W == J

    x = move_fixed(fixture(Float64, 5, 3))
    @test (x .= x .* 0; true)
    @test (x .= 0; true)
    @test (fill!(x, false); true)
end
end

@testset "fill! keeps a fixed pattern" begin
    P = sparse([1, 3], [1, 2], [1.0, 2.0], 3, 3)
    A = fixed(copy(P))
    @test_throws ArgumentError fill!(A, 2.0)
    @test same_pattern(A, P) && nonzeros(A) == [1.0, 2.0] && _is_fixed(A)
    @test fill!(A, 0.0) === A && same_pattern(A, P) && iszero(nonzeros(A)) && _is_fixed(A)
    F = fixed(sparse(ones(2, 3)))
    @test fill!(F, 2.5) === F && F == fill(2.5, 2, 3) && nnz(F) == 6 && _is_fixed(F)
end

@static if COMPREHENSIVE
@testset "`getindex`` should return type with same `_is_fixed`" begin
    for A in [fixed(fixture(Float64, 5, 3))]
        @test _is_fixed(A) == _is_fixed(A[:, :])
        @test _is_fixed(A) == _is_fixed(A[:, 1])
        @test _is_fixed(A) == _is_fixed(A[1, :])
        @test _is_fixed(A) == _is_fixed(A[1:2, 1:2])
        @test _is_fixed(A) == _is_fixed(A[2:4, 2:3])
    end
    for A in [fixturevec(Float64, 10), fixed(fixturevec(Float64, 10))]
        @test _is_fixed(A) == _is_fixed(A[:])
        @test _is_fixed(A) == _is_fixed(A[1:3])
    end
end
end

@testset "getindex with unsorted indices keeps the pattern read-only" begin
    S = sparse([1, 2, 3, 1], [1, 2, 3, 3], [1.0, 2.0, 3.0, 4.0])
    F = fixed(S)
    pattern = (copy(parent(getcolptr(F))), copy(parent(rowvals(F))), copy(nonzeros(F)))
    mask = [true, false, true]
    for (I, J) in (([2, 1], [1, 2]), (@static COMPREHENSIVE ? (([3, 1, 2], :), (:, [3, 1]), ([2, 1], mask), (mask, [3, 1]),
                   ([3, 1, 2], 3)) : ())...)
        R = F[I, J]
        @test R == S[I, J]
        @test _is_fixed(R)
        @test size(R) == size(S[I, J])
    end
    @test (parent(getcolptr(F)), parent(rowvals(F)), nonzeros(F)) == pattern
end

@testset "cumsum, cumprod and accumulate return a writable copy" begin
    for T in (Float64,)
        S = sparse([1, 2, 3, 1], [1, 2, 3, 3], T[1, 2, 3, 4])
        F = fixed(S)
        pattern = (copy(parent(getcolptr(F))), copy(parent(rowvals(F))), copy(nonzeros(F)))
        @test which(cumsum, (typeof(F),)).module === SparseArrays
        @test which(cumprod, (typeof(F),)).module === SparseArrays
        @test which(accumulate, (typeof(+), typeof(F))).module === SparseArrays
        for d in (1, 2)
            for f in (cumsum, (@static COMPREHENSIVE ? (cumprod,) : ())...)
                R = f(F, dims=d)
                @test R == f(S, dims=d) == f(Array(S), dims=d)
                @test R isa SparseMatrixCSC{T} && !_is_fixed(R)
            end
            R = accumulate(+, F, dims=d, init=one(T))
            @test R == accumulate(+, S, dims=d, init=one(T)) == accumulate(+, Array(S), dims=d, init=one(T))
            @test R isa SparseMatrixCSC{T} && !_is_fixed(R)
        end
        @test (parent(getcolptr(F)), parent(rowvals(F)), nonzeros(F)) == pattern

        s = sparsevec([1, 3], T[1, 2], 4)
        v = fixed(s)
        vpattern = (copy(parent(nonzeroinds(v))), copy(nonzeros(v)))
        @test which(cumsum, (typeof(v),)).module === SparseArrays
        for f in (cumsum,)
            r = f(v)
            @test r == f(s) == f(Array(s)) == f(v, dims=1)
            @test r isa SparseVector{T} && !_is_fixed(r)
        end
        @static if COMPREHENSIVE
        r = accumulate(+, v, init=one(T))
        @test r == accumulate(+, s, init=one(T)) == accumulate(+, Array(s), init=one(T))
        @test r isa SparseVector{T} && !_is_fixed(r)
        end
        @test (parent(nonzeroinds(v)), nonzeros(v)) == vpattern
    end
end

@static if COMPREHENSIVE
@testset "FixedSparseVector getindex with index vectors" begin
    x = sparsevec([1, 3, 6], [1.0, 2.0, 3.0], 8)
    v = fixed(x)
    for I in ([3, 1], Int[], (@static COMPREHENSIVE ? ([true, false, true, false, false, true, false, true],) : ())...)
        r = v[I]
        @test r isa FixedSparseVector{Float64,Int} && r == x[I]
    end
    @test nonzeroinds(v) == [1, 3, 6] && nonzeros(v) == [1.0, 2.0, 3.0]
end
end

@static if COMPREHENSIVE
@testset "fixed arrays do not leak a MethodError from the read-only pattern" begin
    A = sparse([4.0 1 0; 1 3 1; 0 1 5]); F = fixed(copy(A))
    x = sparsevec([1, 3], [1.0, 2.0], 3); f = fixed(copy(x))
    @test (f * 2)::FixedSparseVector == 2x && (2 * f)::FixedSparseVector == 2x && (f / 2)::FixedSparseVector == x / 2
    @test SparseVector(fixed(A[:, 1:1]))::SparseVector == A[:, 1]
    @test copy(reshape(F, 1, 9)) == reshape(A, 1, 9)
    @test SparseArrays.dropstored!(F, 1, 1) === F && F[1, 1] == 0 && same_pattern(F, A)
    @test copy!(f, [5.0, 0.0, 6.0]) === f && f == [5.0, 0.0, 6.0] && nonzeroinds(f) == [1, 3]
    @test_throws ArgumentError permute!(fixed(copy(A)), A, [2, 1, 3], [1, 3, 2])
end
end

always_false(x...) = false
below_diagonal(i, j, v) = i > j
@testset "Test fkeep!" begin
    for a in [fixture(Float64, 4, 4) + I, map(x -> x + 1, fixturevec(Float64, 10))]
        a = fixed(a)
        b = copy(a)
        fkeep!(always_false, b)
        @test nnz(a) == nnz(b)
        @test all(iszero, nonzeros(b))

    end
    # a predicate that tells a row from a column, on a matrix that is not square
    A = fixture(Float64, 5, 3)
    F = fkeep!(below_diagonal, fixed(copy(A)))
    @test mismatch(F, tril(Array(A), -1)) === nothing && same_pattern(F, A)
end

end # module
