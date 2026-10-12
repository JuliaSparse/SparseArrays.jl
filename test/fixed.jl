# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseFixedTests

using Test, SparseArrays, LinearAlgebra
using SparseArrays: AbstractSparseVector, AbstractSparseMatrixCSC, FixedSparseCSC, FixedSparseVector, ReadOnly,
    getcolptr, rowvals, nonzeros, nonzeroinds, _is_fixed, fixed, move_fixed, fkeep!, dropstored!, indtype
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
    H = F ./ 1
    @test H isa SparseMatrixCSC && same_pattern(F, H, A)
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
    @test B isa SparseMatrixCSC{Float64,Int} && same_pattern(B, F)
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
    @static if COMPREHENSIVE
    # a throw leaves the values alone too: the entry outside the pattern or the value that
    # does not convert is found before the block is zeroed
    Gi = fixed(sparse([1, 2], [1, 2], [1, 2], 2, 2))
    R = CartesianIndices((1:2, 1:2))
    @test_throws InexactError copyto!(Gi, sparse([1, 2], [1, 2], [1.0, 2.5]))
    @test_throws ArgumentError copyto!(Gi, R, sparse([5 6; 7 8]), R)
    @test_throws InexactError copyto!(Gi, R, sparse([5.5 0; 0 8]), R)
    @test nonzeros(Gi) == [1, 2]
    @test copyto!(Gi, R, sparse([5 0; 0 8]), R) === Gi && nonzeros(Gi) == [5, 8]
    end
    G[1:2, 1:2] = [3 0; 0 4]
    G[:, 2] .= 0
    @test G == [3 0; 0 0] && nnz(G) == 2
    G .= sparse([2], [2], [6.0], 2, 2)   # a subset pattern zero-fills the rest
    @test G == [0 0; 0 6] && nnz(G) == 2
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
    # every form of `similar` is writable; the structure-preserving ones copy the pattern
    @test similar(F, Float32, Int) isa SparseMatrixCSC{Float32,Int} && same_pattern(similar(F, Float32, Int), F)
    @test parent(rowvals(F)) !== rowvals(similar(F))
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
    z = x ./ 2
    @test z isa SparseVector && same_pattern(x, y, z)
    f(x, y, z) = @allocated(x .= y .+ y) +
        (@static COMPREHENSIVE ? @allocated(x .= y .- y) : 0)
    f(x, y, z)
    f(x, y, z)
    @test f(x, y, z) == 0
    t = similar(x)
    @test t isa SparseVector{Float64,Int} && same_pattern(t, x)
    @test similar(x, 5) isa SparseVector
    @static if COMPREHENSIVE
    @test typeof(FixedSparseVector{Float32,Int32}(x)) == FixedSparseVector{Float32,Int32}
    end
    w = fixed(sparsevec([1, 3], [1.0, 2.0], 4))
    @test_throws ArgumentError w[2] = 1.0
    @test_throws ArgumentError copyto!(w, sparsevec([2], [1.0], 4))
    @test same_pattern(w, sparsevec([1, 3], [1.0, 2.0], 4)) && nonzeros(w) == [1.0, 2.0]
    @static if COMPREHENSIVE
    wi = fixed(sparsevec([1, 3], [1, 2], 4))
    @test_throws InexactError copyto!(wi, sparsevec([1, 3], [1.0, 2.5], 4))
    @test nonzeros(wi) == [1, 2]
    end
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

@testset "structure queries read a fixed matrix" begin
    # `issymmetric`/`ishermitian` advance a copy of the column pointers, which are
    # read-only in a fixed matrix
    S = fixed(sparse([4.0 1 0; 1 4 1; 0 1 4]))
    @test issymmetric(S) && ishermitian(S)
    @test !issymmetric(fixed(sparse([4.0 1 0; 2 4 1; 0 1 4])))
    @static if COMPREHENSIVE
    @test ishermitian(fixed(sparse(ComplexF64[4 1+im 0; 1-im 4 1; 0 1 4])))
    @test !issymmetric(fixed(sparse(ComplexF64[4 1+im 0; 1-im 4 1; 0 1 4])))
    end
end

# As an operand a fixed array is the plain array with the same entries: the result has the
# type, pattern and values that the plain array gives, and shares no pattern buffer with
# the fixed one.
@testset "reading a fixed array gives the result of the plain array" begin
    A = sparse([1, 3, 2, 4, 1, 3, 4], [1, 1, 2, 2, 3, 4, 4], [1.0, 2, 3, 0, 5, 6, 7], 4, 4)
    B = sparse([1, 2, 4, 2], [1, 2, 3, 4], [2.0, 1, 1, 3], 4, 4)
    x = sparsevec([1, 3, 4], [1.0, 0.0, 3.0], 4); y = sparsevec([2, 3], [1.0, 2.0], 4)
    D = Diagonal([1.0, 2, 3, 4])
    F, f = fixed(A), fixed(x)
    function sameresult(op, P, Q)
        want, got = op(P), op(Q)
        typeof(want) == typeof(got) || return false
        want isa Union{AbstractSparseMatrixCSC,AbstractSparseVector} || return isequal(want, got)
        return exact_equal(want, got) && !_is_fixed(got) &&
            !any(b -> b === parent(getcolptr(F)) || b === parent(rowvals(F)) || b === parent(nonzeroinds(f)),
                 SparseArrays._storagebuffers(got))
    end
    matops = (S -> 2S, (@static COMPREHENSIVE ? (S -> -S, S -> S / 2, complex, S -> conj(complex(S)), S -> abs.(S), S -> map(sin, S),
        S -> S + B, S -> B .* S, S -> S .* y, S -> S * B, S -> D * S, S -> S * D, S -> S + I, S -> copy(S'), S -> kron(S, B),
        S -> S[:, :], S -> S[:, 2], S -> S[2:3, 1:2], S -> S[[3, 1], :], S -> S[:], S -> [S B], S -> [S; B], S -> [S B; B S],
        S -> fill!(similar(S), 1), S -> fill!(similar(S, Int), 1), sparse, dropzeros, tril, triu, reverse, rot180, S -> circshift(S, (1, 1)),
        S -> sort(S, dims=1), S -> permute(S, [2, 1, 3, 4], [1, 3, 2, 4]), S -> cumsum(S, dims=1), S -> sum(S, dims=1),
        S -> S * x, S -> S * ones(4), norm, sum, nnz, findnz, issymmetric) : ())...)
    for op in matops
        @test sameresult(op, A, F)
    end
    vecops = (v -> v + y, (@static COMPREHENSIVE ? (v -> -v, v -> 2v, v -> v / 2, complex, v -> conj(complex(v)), v -> abs.(v),
        v -> y - v, v -> v .* y, v -> kron(v, y), v -> v[:], v -> v[1:3], v -> v[[3, 1]], v -> [v; y], v -> [v y],
        v -> fill!(similar(v), 1), dropzeros, reverse, v -> circshift(v, 1), sort, cumsum, v -> A * v, v -> dot(v, y), sum, findnz) : ())...)
    for op in vecops
        @test sameresult(op, x, f)
    end
    # only `copy` stays fixed, on the same pattern; an operation that returns its argument
    # returns the fixed array
    C, c = copy(F), copy(f)
    @test C isa FixedSparseCSC && getcolptr(C) === getcolptr(F) && rowvals(C) === rowvals(F) && nonzeros(C) !== nonzeros(F)
    @test c isa FixedSparseVector && nonzeroinds(c) === nonzeroinds(f) && nonzeros(c) !== nonzeros(f)
    @static if COMPREHENSIVE
    @test float(F) === F && float(f) === f
    # `float` of a complex array and `complex` of a real one, where one of the two is the identity
    Fc, fc = fixed(complex(A)), fixed(complex(x))
    @test float(Fc) === Fc && float(fc) === fc && complex(Fc) === Fc && complex(fc) === fc
    # the ways to ask for a copy that do not go through `copy`
    for op in (S -> permutedims(S, (1, 2)), S -> rot180(S, 2), S -> rotl90(S, 4), S -> rotr90(S, 0), S -> rotl90(S, 3), S -> rotr90(S, 2))
        @test sameresult(op, A, F)
    end
    end
    @test exact_equal(F, A) && exact_equal(f, x)
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

@testset "getindex with unsorted indices keeps the pattern read-only" begin
    S = sparse([1, 2, 3, 1], [1, 2, 3, 3], [1.0, 2.0, 3.0, 4.0])
    F = fixed(S)
    pattern = (copy(parent(getcolptr(F))), copy(parent(rowvals(F))), copy(nonzeros(F)))
    mask = [true, false, true]
    for (I, J) in (([2, 1], [1, 2]), (@static COMPREHENSIVE ? (([3, 1, 2], :), (:, [3, 1]), ([2, 1], mask), (mask, [3, 1]),
                   ([3, 1, 2], 3)) : ())...)
        R = F[I, J]
        @test R == S[I, J]
        @test !_is_fixed(R)
        @test size(R) == size(S[I, J])
    end
    @test (parent(getcolptr(F)), parent(rowvals(F)), nonzeros(F)) == pattern
end

@testset "cumsum, cumprod and accumulate of a fixed array" begin
    for T in (Float64,)
        S = sparse([1, 2, 3, 1], [1, 2, 3, 3], T[1, 2, 3, 4])
        F = fixed(S)
        pattern = (copy(parent(getcolptr(F))), copy(parent(rowvals(F))), copy(nonzeros(F)))
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
        @test r isa SparseVector{Float64,Int} && r == x[I]
    end
    @test nonzeroinds(v) == [1, 3, 6] && nonzeros(v) == [1.0, 2.0, 3.0]
end
end

@static if COMPREHENSIVE
@testset "fixed arrays do not leak a MethodError from the read-only pattern" begin
    A = sparse([4.0 1 0; 1 3 1; 0 1 5]); F = fixed(copy(A))
    x = sparsevec([1, 3], [1.0, 2.0], 3); f = fixed(copy(x))
    @test (f * 2)::SparseVector == 2x && (2 * f)::SparseVector == 2x && (f / 2)::SparseVector == x / 2
    @test SparseVector(fixed(A[:, 1:1]))::SparseVector == A[:, 1]
    @test copy(reshape(F, 1, 9)) == reshape(A, 1, 9)
    @test copy!(f, [5.0, 0.0, 6.0]) === f && f == [5.0, 0.0, 6.0] && nonzeroinds(f) == [1, 3]
    @test_throws ArgumentError permute!(fixed(copy(A)), A, [2, 1, 3], [1, 3, 2])
end
end

# A function whose purpose is to change the pattern throws, whatever the pattern and the
# values are, and leaves the array as it was.
always_true(x...) = true
@testset "functions that change the pattern throw on a fixed array" begin
    A = sparse([1, 3, 2, 3, 1, 3], [1, 1, 2, 2, 3, 3], [1.0, 2, 3, 0, 5, 6], 3, 3)
    x = sparsevec([1, 3, 4], [1.0, 0.0, 3.0], 5)
    F, f = fixed(A), fixed(x)
    matfs = (dropzeros!, (@static COMPREHENSIVE ? (S -> droptol!(S, 0.5), S -> fkeep!(always_true, S), tril!, triu!,
        S -> dropstored!(S, 1, 1), S -> dropstored!(S, 1:2, 1:2), reverse!, S -> reverse!(S, dims=1),
        S -> sort!(S, dims=1),
        S -> permute!(S, [1, 2, 3], [1, 2, 3]), S -> dropstored!(S, [2], [1]), S -> dropstored!(S, 1:0, 1:0), S -> dropstored!(S, :, 1),
        S -> Base.swaprows!(S, 1, 3), S -> Base.swapcols!(S, 1, 3)) : ())...)
    for f! in matfs
        @test_throws ArgumentError f!(F)
        @test exact_equal(F, A) && _is_fixed(F)
    end
    vecfs = (dropzeros!, (@static COMPREHENSIVE ? (v -> droptol!(v, 0.5), v -> fkeep!(always_true, v), v -> dropstored!(v, 1),
        reverse!, v -> circshift!(v, 1), v -> permute!(v, collect(1:5)), v -> invpermute!(v, collect(1:5)), sort!) : ())...)
    for f! in vecfs
        @test_throws ArgumentError f!(f)
        @test exact_equal(f, x) && _is_fixed(f)
    end
    @static if COMPREHENSIVE
    # also where the function would have nothing to change
    Z = fixed(spzeros(2, 2))
    @test_throws ArgumentError dropzeros!(Z)
    @test_throws ArgumentError tril!(fixed(sparse([1.0 0; 2 3])))
    @test_throws "SparseMatrixCSC" dropzeros!(F)
    @test_throws "SparseVector" dropzeros!(f)
    # the error names the function that was called
    @test_throws "droptol!" droptol!(f, 0.5)
    @test_throws "circshift!" circshift!(f, 1)
    @test_throws "sort!" sort!(f)
    @test_throws "sort!" sort!(view(F, :, 1))
    @test_throws "permute!" permute!(F, [1, 2, 3], [1, 2, 3])
    end
end

# The kernels for arguments of different shapes write the pattern of the destination as
# they go, so a fixed destination is checked first.
@static if COMPREHENSIVE
@testset "broadcast of different shapes into a fixed destination" begin
    A = sparse([1, 3, 2, 3, 1, 3], [1, 1, 2, 2, 3, 3], [1.0, 2, 3, 4, 5, 6], 3, 3)
    v = sparsevec([1, 3], [2.0, 3.0], 3)
    F = fixed(A)
    @test (F .= A .* v) === F && F == A .* Vector(v) && same_pattern(F, A)   # row 2 is zeroed, and stays stored
    F = fixed(A)
    @test_throws ArgumentError F .= A .+ v'
    @test exact_equal(F, A)
    # the destination's shape extends the arguments': each column of `W` takes `v .+ v`
    W = fixed(sparse(ones(3, 2)))
    @test (W .= v .+ v) === W && W == [4.0 4.0; 0.0 0.0; 6.0 6.0] && nnz(W) == 6
    w = fixed(sparse(ones(3)))
    V = sparse(reshape([2.0, 0.0, 3.0], 3, 1))
    @test (w .= V .+ V) === w && w == [4.0, 0.0, 6.0] && nnz(w) == 3
    Wi = fixed(sparse(ones(Int, 3, 2)))
    @test_throws InexactError Wi .= v ./ 4 .+ v
    @test nonzeros(Wi) == ones(Int, 6)
    # a one-column destination takes the vector result
    M = fixed(sparse(reshape([1.0, 0.0, 2.0], 3, 1)))
    @test (M .= v .* v) === M && M == reshape([4.0, 0.0, 9.0], 3, 1) && nnz(M) == 2
end
end

@static if COMPREHENSIVE
@testset "block assignment into a fixed array checks the pattern before writing" begin
    A = sparse([1.0 0 0; 0 0 0; 4 0 5]); A[1, 2] = 0
    x = sparsevec([1, 3], [1.0, 0.0], 3)
    F = fixed(A)
    @test (F[:, 1] = [7.0, 0, 8]; F[[1, 3], [1, 3]] = [6.0 0; 9 3]; F[:, 1] .= 0; F == [0.0 0 0; 0 0 0; 0 0 3]) && same_pattern(F, A)
    @test (F[:, 3] = view(F, :, 3); F[3, 3] == 3)
    f = fixed(x)
    @test (f[:] = [7.0, 0, 8]; f == [7.0, 0, 8]) && nonzeroinds(f) == [1, 3]
    # the first entries of each of these lie inside the pattern, and a later one does not
    for write! in (S -> (S[:, 1] = [7.0, 8, 9]), S -> (S[[1, 3], [1, 3]] = [7.0 6; 8 9]), S -> (S[1, :] = [7.0, 8, 9]),
                   S -> (S[:, 1] .= 7), S -> (S[:, 1] = [7.0, 0, 1im]))
        F = fixed(A)
        @test_throws Union{ArgumentError,InexactError} write!(F)
        @test exact_equal(F, A)
    end
    f = fixed(x)
    @test_throws ArgumentError f[:] = [7.0, 8, 9]
    @test exact_equal(f, x)
end
end

@static if COMPREHENSIVE
@testset "a result within the pattern of a fixed destination is stored, the rest zeroed" begin
    J = sparse([1.0 2 0; 0 3 0; 4 0 5]); M = sparse([1.0 0 0; 0 1 0; 0 0 1])
    P = M + J
    for (write!, R) in ((W -> W .= 2 .* M, 2M), (W -> W .= .-M, -M), (W -> W .= M .+ M, 2M), (W -> W .= M .+ M .+ M, 3M),
                        (W -> map!(identity, W, M), M), (W -> map!(+, W, M, M), 2M), (W -> W .= M, M), (W -> copyto!(W, M), M))
        W = fixed(P)
        @test write!(W) === W && W == R && same_pattern(W, P)
    end
    # a zero of the result may lie outside the pattern, stored or not
    D = fixed(M)
    @test (D .= J .* M) === D && D == J .* M && same_pattern(D, M)
    Z = sparse([1, 3, 2, 3], [1, 1, 2, 3], [1.0, 0.0, 3.0, 5.0], 3, 3)   # a stored zero at (3, 1)
    @test copyto!(D, Z) === D && D == [1.0 0 0; 0 3 0; 0 0 5] && same_pattern(D, M)
    d = fixed(sparsevec([1], [1.0], 3))
    @test copyto!(d, sparsevec([1, 3], [2.0, 0.0], 3)) === d && d == [2.0, 0, 0]
    # a nonzero may not
    D = fixed(M)
    @test_throws ArgumentError D .= J .+ M
    @test_throws ArgumentError D .= 2 .* J
    @test_throws ArgumentError map!(identity, D, J)
    @test exact_equal(D, M)
end

# These write a result into a destination apart from their operands, so the pattern of a
# fixed destination decides, as it does for a broadcast.
@testset "a result written into a separate fixed destination fits the pattern or throws" begin
    A = sparse([1.0 0 2; 0 3 0; 4 0 5]); B = sparse([1.0 0 2; 1 3 0; 4 0 5])
    x = sparsevec([1, 3], [1.0, 2.0], 3); y = sparsevec([1, 2], [1.0, 2.0], 3)
    p = [3, 2, 1]
    for (write!, R, bad!) in (
            (W -> mul!(W, A, 2), 2A, W -> mul!(W, B, 2)), (W -> mul!(W, 2, A), 2A, W -> mul!(W, 2, B)),
            (W -> transpose!(W, sparse(A')), A, W -> transpose!(W, sparse(B'))),
            (W -> adjoint!(W, sparse(A')), A, W -> adjoint!(W, sparse(B'))),
            (W -> circshift!(W, A, (0, 0)), A, W -> circshift!(W, A, (1, 0))),
            (W -> permute!(W, A, p, p), A[p, p], W -> permute!(W, A, [2, 1, 3], [1, 2, 3])),
            (W -> mul!(W, B, Diagonal([0.0, 1.0, 1.0])), B * Diagonal([0.0, 1, 1]), W -> mul!(W, B, Diagonal([1.0, 1.0, 1.0]))),
            (W -> mul!(W, Diagonal([0.0, 1.0, 1.0]), sparse(B')), Diagonal([0.0, 1, 1]) * B', W -> mul!(W, Diagonal([1.0, 1.0, 1.0]), sparse(B'))))
        W = fixed(A)
        @test write!(W) === W && W == R && same_pattern(W, A)
        W = fixed(A)
        @test_throws ArgumentError bad!(W)
        @test exact_equal(W, A)
    end
    for (write!, r, bad!) in (
            (w -> mul!(w, A, x), A * x, w -> mul!(w, B, x)), (w -> mul!(w, A', x), A' * x, w -> mul!(w, B', y)),
            (w -> mul!(w, Matrix(A), x), A * x, w -> mul!(w, Matrix(B), x)),
            (w -> mul!(w, UpperTriangular(Matrix(A)), x), UpperTriangular(Matrix(A)) * x, w -> mul!(w, LowerTriangular(Matrix(B)), x)),
            (w -> mul!(w, A, x, 2, 1), 2 * (A * x) + x, w -> mul!(w, B, x, 2, 1)),
            (w -> axpy!(2, x, w), 3x, w -> axpy!(2, y, w)),
            (w -> circshift!(w, x, 0), x, w -> circshift!(w, x, 1)),
            (w -> (w[[3, 1]] = [7.0, 8.0]; w), [8.0, 0, 7], w -> (w[[1, 2]] = [7.0, 8.0])),
            (w -> (w[[true, false, true]] = [7.0, 8.0]; w), [7.0, 0, 8], w -> (w[[true, true, false]] = [7.0, 8.0])))
        w = fixed(x)
        @test write!(w) === w && w == r && nonzeroinds(w) == [1, 3]
        w = fixed(x)
        @test_throws ArgumentError bad!(w)
        @test exact_equal(w, x)
    end
    # an index vector assigns into a writable vector as before
    s = copy(x)
    @test (s[[2, 1]] = [7.0, 8.0]; s == [8.0, 7, 2]) && (s[[true, false, true]] = [0.0, 1.0]; s == [0.0, 7, 1])
end

# One test decides what may lie outside the pattern, whatever the kernel: a value that
# `setindex!` leaves unstored.
@testset "a value outside a fixed pattern is accepted when setindex! would not store it" begin
    M = sparse([1.0 0 0; 0 1 0; 0 0 1]); x = sparsevec([1], [1.0], 3)
    Z = sparse([1, 3], [1, 1], [2.0, 0.0], 3, 3); z = sparsevec([1, 3], [2.0, 0.0], 3)
    N = sparse([1, 3], [1, 1], [2.0, -0.0], 3, 3); n = sparsevec([1, 3], [2.0, -0.0], 3)
    for write! in (W -> copyto!(W, Z), W -> (W[:, :] = Z), W -> (W[:, 1] = z), W -> copyto!(view(W, :, 1), z),
                   W -> (W[3, 1] = 0.0; W[1, 1] = 2.0), W -> (W[[1, 3]] = [2.0, 0.0]))
        W = fixed(M)
        write!(W)
        @test W[:, 1] == [2.0, 0, 0] && same_pattern(W, M)
    end
    for write! in (W -> copyto!(W, N), W -> (W[:, :] = N), W -> (W[:, 1] = n), W -> copyto!(view(W, :, 1), n), W -> (W[3, 1] = -0.0),
                   W -> (W[[1, 3]] = [2.0, -0.0]))
        W = fixed(M)
        @test_throws ArgumentError write!(W)
        @test exact_equal(W, M)
    end
    for write! in (w -> copyto!(w, z), w -> (w[:] = z), w -> (w[1:3] = z), w -> (w[[1, 3]] = [2.0, 0.0]), w -> (w[3] = 0.0; w[1] = 2.0))
        w = fixed(x)
        write!(w)
        @test w == [2.0, 0, 0] && nonzeroinds(w) == [1]
    end
    for write! in (w -> copyto!(w, n), w -> (w[:] = n), w -> (w[1:3] = n), w -> (w[[1, 3]] = [2.0, -0.0]), w -> (w[3] = -0.0),
                   w -> copyto!(w, Vector(n)))
        w = fixed(x)
        @test_throws ArgumentError write!(w)
        @test exact_equal(w, x)
    end
end

@testset "rdiv! and ldiv! by a number write the stored values of a fixed array" begin
    A = sparse([1.0 2 0; 0 3 0; 4 0 5]); x = sparsevec([1, 3], [1.0, 2.0], 3)
    F, f = fixed(A), fixed(x)
    @test rdiv!(F, 2) === F && F == A / 2 && same_pattern(F, A)
    @test ldiv!(2, F) === F && F == A / 4
    @test rdiv!(f, 2) === f && ldiv!(2, f) === f && f == x / 4
    # `0 / 0` is not zero, so the result needs every entry
    for b in (0, NaN)
        F, f = fixed(A), fixed(x)
        @test_throws ArgumentError rdiv!(F, b)
        @test_throws ArgumentError ldiv!(b, F)
        @test_throws ArgumentError rdiv!(f, b)
        @test exact_equal(F, A) && exact_equal(f, x)
    end
    S = copy(A)
    @test rdiv!(S, 2) === S && mismatch(S, Matrix(A) / 2) === nothing && nnz(S) == 5
    @test isequal(rdiv!(copy(A), 0), rdiv!(Matrix(A), 0)) && isequal(ldiv!(0, copy(x)), ldiv!(0, Vector(x)))
end

@testset "sparse and fixed of a fixed vector follow the matrix" begin
    x = sparsevec([1, 3], [1.0, 2.0], 3); f = fixed(x)
    for g in (sparse, sparsevec)
        s = g(f)
        @test s isa SparseVector && s == x && nonzeros(s) !== nonzeros(f) && (s[2] = 1; nnz(s) == 3 && nnz(f) == 2)
    end
    for g in (fixed, FixedSparseVector, FixedSparseVector{Float64,Int})
        h = g(f)
        @test _is_fixed(h) && nonzeroinds(h) === nonzeroinds(f) && nonzeros(h) !== nonzeros(f) && h == x
    end
end
end

end # module
