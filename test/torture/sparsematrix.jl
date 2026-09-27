# This file is a part of Julia. License is MIT: https://julialang.org/license

# Long-tail regression tests for `SparseMatrixCSC` outside indexing and the constructors.
# Each testset names the issue it guards; they run only when the `torture` suite is
# selected.

module TortureSparseMatrixTests
using Test
using SparseArrays
using LinearAlgebra
include("../testhelpers.jl")

@testset "what used to be issue #5386" begin
    K,J,V = findnz(SparseMatrixCSC(2,1,[1,3],[1,2],[1.0,0.0]))
    @test length(K) == length(J) == length(V) == 2
end

@testset "issue #5853, sparse diff" begin
    for i=1:2, a=Any[[1 2 3], reshape([1, 2, 3],(3,1)), Matrix(1.0I, 3, 3)]
        @test diff(sparse(a),dims=i) == diff(a,dims=i)
    end
end

@testset "issue #7650" begin
    S = spzeros(3, 3)
    @test size(reshape(S, 9, 1)) == (9,1)
end

@testset "issue #7677" begin
    A = sprand(5,5,0.5,(n)->rand(Float64,n))
    ACPY = copy(A)
    B = reshape(A,25,1)
    @test A == ACPY
end

@testset "issue #8976" begin
    @test conj.(sparse([1im])) == sparse(conj([1im]))
    @test conj!(sparse([1im])) == sparse(conj!([1im]))
end

# Check calling of unary minus method specialized for SparseMatrixCSCs
@testset "issue #19503" begin
    @test which(-, (SparseMatrixCSC,)).module == SparseArrays
end

@testset "Issue #28369" begin
    M = reshape([[1 2; 3 4], [9 10; 11 12], [5 6; 7 8], [13 14; 15 16]], (2,2))
    MP = reshape([[1 2; 3 4], [5 6; 7 8], [9 10; 11 12], [13 14; 15 16]], (2,2))
    S = sparse(M)
    SP = sparse(MP)
    @test isa(transpose(S), Transpose)
    @test transpose(S) == copy(transpose(S))
    @test Array(transpose(S)) == copy(transpose(M))
    @test permutedims(S) == SP
    @test permutedims(S, (2,1)) == SP
    @test permutedims(S, (1,2)) == S
    @test permutedims(S, (1,2)) !== S
    @test_throws ArgumentError permutedims(S, (1,3))
    MC = reshape([[(1+im) 2; 3 4], [9 10; 11 12], [(5 + 2im) 6; 7 8], [13 14; 15 16]], (2,2))
    SC = sparse(MC)
    @test isa(adjoint(SC), Adjoint)
    @test adjoint(SC) == copy(adjoint(SC))
    @test adjoint(MC) == copy(adjoint(SC))
end

@testset "issue #41135" begin
    @test repr(SparseMatrixCSC([7;;])) == "sparse([1], [1], [7], 1, 1)"

    m = SparseMatrixCSC([0 3; 4 0])
    @test repr(m) == "sparse([2, 1], [1, 2], [4, 3], 2, 2)"
    @test eval(Meta.parse(repr(m))) == m
    @test summary(m) == "2×2 $SparseMatrixCSC{$Int, $Int} with 2 stored entries"

    m = sprand(100, 100, .1)
    @test occursin(r"^sparse\(\[.+\], \[.+\], \[.+\], \d+, \d+\)$", repr(m))
    @test eval(Meta.parse(repr(m))) == m

    m = sparse([85, 5, 38, 37, 59], [19, 72, 76, 98, 162], [0.8, 0.3, 0.2, 0.1, 0.5], 100, 200)
    @test repr(m) == "sparse([85, 5, 38, 37, 59], [19, 72, 76, 98, 162], [0.8, 0.3, 0.2, 0.1, 0.5], 100, 200)"
    @test eval(Meta.parse(repr(m))) == m
end

# From Base's abstractarray.jl
@testset "mapslices julia #21123" begin
    @test mapslices(nnz, sparse(1.0I, 3, 3), dims=1) == [1 1 1]
end

# From Base's core.jl: an eltype with `zero` but no `show`-able stored values, and a
# function with more arguments than `MAX_TUPLETYPE_LEN` that dispatches on a sparse
# argument. Both are defined at module top level so that the testsets below can use them.
mutable struct T12960 end
Base.zero(::Type{T12960}) = T12960()
Base.zero(x::T12960) = T12960()

f12063(tt, g, p, c, b, v, cu::T, d::AbstractArray{T, 2}, ve) where {T} = 1
f12063(args...) = 2
g12063() = f12063(0, 0, 0, 0, 0, 0, 0.0, spzeros(0,0), Int[])

@testset "show of #undef stored entries (julia #12960)" begin
    A = sparse(1.0I, 3, 3)
    B = similar(A, T12960)
    @test repr(B) == "sparse([1, 2, 3], [1, 2, 3], $T12960[#undef, #undef, #undef], 3, 3)"
    @test occursin(
        "\n #undef             ⋅            ⋅    \n       ⋅      #undef             ⋅    \n       ⋅            ⋅      #undef",
        repr(MIME("text/plain"), B),
    )

    B[1,2] = T12960()
    @test repr(B)  == "sparse([1, 1, 2, 3], [1, 2, 2, 3], $T12960[#undef, $T12960(), #undef, #undef], 3, 3)"
    @test occursin(
        "\n #undef          T12960()        ⋅    \n       ⋅      #undef             ⋅    \n       ⋅            ⋅      #undef",
        repr(MIME("text/plain"), B),
    )
end

@testset "dispatch on a sparse argument past MAX_TUPLETYPE_LEN (julia #12063)" begin
    @test g12063() == 1
end

@testset "Issue #210" begin
    io = IOBuffer()
    show(io, sparse([1 2; 3 4]))
    @test String(take!(io)) == "sparse([1, 2, 1, 2], [1, 1, 2, 2], [1, 3, 2, 4], 2, 2)"
    io = IOBuffer()
    show(io, sparse([1 2; 3 4])')
    @test String(take!(io)) == "adjoint(sparse([1, 2, 1, 2], [1, 1, 2, 2], [1, 3, 2, 4], 2, 2))"
    io = IOBuffer()
    show(io, transpose(sparse([1 2; 3 4])))
    @test String(take!(io)) == "transpose(sparse([1, 2, 1, 2], [1, 1, 2, 2], [1, 3, 2, 4], 2, 2))"
end

@testset "Issue #390" begin
    x = sparse([9 1 8
                0 3 72
                7 4 16])
    Base.swapcols!(x, 2, 3)
    @test x == sparse([9 8 1
                       0 72 3
                       7 16 4])
end

# Repeats the core `dropdims` check on five random draws.
@testset "dropdims" begin
    for i = 1:5
        am = sprand(20, 1, 0.2)
        av = dropdims(am, dims=2)
        @test ndims(av) == 1
        @test all(av.==am)
        am = sprand(1, 20, 0.2)
        av = dropdims(am, dims=1)
        @test ndims(av) == 1
        @test all(av' .== am)
    end
end

# The eltypes of the real/imag/abs/abs2 loop that the core "unary functions" testset
# does not run.
@testset "unary functions, remaining eltypes" begin
    for T in (Float16, Float32, BigInt, BigFloat)
        R = rand(T[1:100;], 2, 2)
        I = rand(T[1:100;], 2, 2)
        D = R + I*im
        S = sparse(D)
        spR = sparse(R)

        @test R == real.(S) == real(S)
        @test I == imag.(S) == imag(S)
        @test conj(Array(S)) == conj.(S) == conj(S)
        @test real.(spR) == R
        @test nnz(imag.(spR)) == nnz(imag(spR)) == 0
        @test abs.(S) == abs.(D)
        @test abs2.(S) == abs2.(D)

        # test aliasing of real and conj of real valued matrix
        @test real(spR) === spR
        @test conj(spR) === spR
    end
end

# The full 3 x 3 grid of matrix pairs; the core suite keeps the complex pair.
@testset "Comparisons to adjoints are efficient" for
    # The counting guard below distinguishes stored-entry traversal from the generic
    # length(A) fallback, so these do not need to be large matrices.
    A in Any[sparse(1*I(100)), sprandn(100, 100, 0.1), sprandn(ComplexF64, 100, 100, 0.9)],
    B in Any[sparse(1*I(100)), sprandn(100, 100, 0.1), sprandn(ComplexF64, 100, 100, 0.9)]
    if size(A) == size(B)
        A = OpCount.(A)
        B = OpCount.(B)
        As = Any[A, A', transpose(A)]
        Bs = Any[B, B', transpose(B)]
        for A′ in As, B′ in Bs
            # skip adjoints of transposes; these are not really supported
            ((A′ isa Adjoint && B′ isa Transpose) || (A′ isa Transpose && B′ isa Adjoint)) && continue
            @test eqcount(() -> A′ == B′) ≤ 1 + (nnz(A′) + nnz(B′))
        end
    end
end

@testset "Issue #246" begin
    for t in [Int, UInt8]
        a = OpCount.(sprand(t, 100, 0.5))
        b = OpCount.(sprand(t, 100, 0.5))

        c = if nnz(a) != 0
            c = copy(a)
            nonzeros(c)[1] = 0
            c
        else
            c = copy(a)
            push!(nonzeros(c), zero(t))
            push!(nonzerosinds(c), 1)
            c
        end
        d = dropzeros(c)

        for m in [identity, transpose, adjoint]
            ma, mb, mc, md = m.([a, b, c, d])

            @test eqcount(() -> ma == mb) <= nnz(a) + nnz(b)

            @test (mc == md) == (Array(mc) == Array(md))
        end
    end
end

# The sizes of the sort grid that the core "sort/sort! of a sparse matrix" testset
# does not run.
@testset "sort/sort! of a sparse matrix, remaining sizes" begin
    @testset "size = ($m, $n), density = $d" for (m, n) in ((1, 1), (20, 13)), d in (0.3, 1.0)
        A = sprand(m, n, d)
        M = Matrix(A)
        for dims in (1, 2), kws in ((;), (; rev=true), (; by=abs), (; alg=Base.DEFAULT_STABLE))
            expected = size(M, dims) == 0 ? M : sort(M; dims, kws...)
            B = copy(A)
            @test sort!(B; dims, kws...) === B
            @test B isa SparseMatrixCSC
            @test Matrix(B) == expected
            # sorting only moves the stored entries around
            @test nnz(B) == nnz(A)
            S = sort(A; dims, kws...)
            @test S isa SparseMatrixCSC
            @test Matrix(S) == expected
            @test A == sparse(M) # `sort` leaves its argument alone
        end
    end
end

# Every size relation of destination and source in `copyto!`; the core suite keeps one
# fitting and one too-small pair.
@testset "copyto! size combinations" begin
    # Test various size(A) / size(B) combinations
    for mA in [5, 10, 20], nA in [5, 10, 20], mB in [5, 10, 20], nB in [5, 10, 20]
        A = sprand(mA,nA,0.4)
        Aorig = copy(A)
        B = sprand(mB,nB,0.4)
        if mA*nA >= mB*nB
            copyto!(A,B)
            @assert(A[1:length(B)] == B[:])
            @assert(A[length(B)+1:end] == Aorig[length(B)+1:end])
        else
            @test_throws BoundsError copyto!(A,B)
        end
    end
end

end # module
