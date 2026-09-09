# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseTests

using Test
using SparseArrays
using SparseArrays: getcolptr, nonzeroinds, _show_with_braille_patterns, _isnotzero, fixed, _is_fixed
using LinearAlgebra
using Printf: @printf # for debug
using Random
using Test: guardseed
using InteractiveUtils: @which
using Dates
include("forbidproperties.jl")
include("simplesmatrix.jl")

@testset "_isnotzero" begin
    @test !_isnotzero(0::Int)
    @test _isnotzero(1::Int)
    @test _isnotzero(missing)
    @test !_isnotzero(0.0)
    @test _isnotzero(1.0)
end


@testset "issparse" begin
    @test issparse(sparse(fill(1,5,5)))
    @test !issparse(fill(1,5,5))
    @test nnz(zero(sparse(fill(1,5,5)))) == 0
end

@testset "findnz for adjoint/transpose (issue #632)" begin
    A = sparse([1, 1, 2, 3], [1, 2, 3, 2], [1.0+2.0im, 3.0, 4.0-1.0im, 0.0], 3, 4)
    for T in (Float64, ComplexF64), op in (adjoint, transpose)
        B = op(T == Float64 ? real(A) : A)
        I, J, V = findnz(B)
        @test (I, J, V) == findnz(SparseMatrixCSC(B))
        @test issorted(collect(zip(J, I)))  # column-major order of the wrapper
        @test all(B[i, j] == v for (i, j, v) in zip(I, J, V))
        @test length(I) == nnz(B)
        @test typeof(I) == typeof(J) == Vector{Int} && eltype(V) == T
        @test all(isempty, findnz(op(spzeros(T, 2, 3))))
    end
    x = sparsevec([2, 4], [1.0+im, 0.0], 5)
    for op in (adjoint, transpose)
        I, J, V = findnz(op(x))
        @test I == [1, 1] && J == [2, 4] && V == op.([1.0+im, 0.0])
        @test (I, J, V) == findnz(sparse(op(x)))
        @test all(isempty, findnz(op(spzeros(3))))
    end
end

@testset "isequal semantics match dense (issue #561)" begin
    # The stored-entries-only complexity guarantee is checked with an operation-counting
    # eltype below ("== and isequal walk stored entries only").
    n = 100
    A = spzeros(n, n); A[1, 1] = 1
    B = spzeros(n, n); B[1, 1] = 1
    @test isequal(A, B) && A == B
    B[n, n] = 2
    @test !isequal(A, B) && A != B
    @test !isequal(spzeros(2, 3), spzeros(3, 2))
    # isequal semantics for NaN and signed zeros must match dense arrays
    X = sparse([1, 2, 3], [1, 1, 2], [NaN, -0.0, 2.0], 3, 3)
    for Y in (sparse([1, 2, 3], [1, 1, 2], [NaN, -0.0, 2.0], 3, 3),
              sparse([1, 3], [1, 2], [NaN, 2.0], 3, 3),
              sparse([1, 2, 3], [1, 1, 2], [NaN, 0.0, 2.0], 3, 3),
              sparse([1, 2, 3, 3], [1, 1, 2, 3], [NaN, -0.0, 2.0, 0.0], 3, 3),
              sparse([1, 2, 3], [1, 1, 2], [1.0, -0.0, 2.0], 3, 3),
              sparse([2, 2, 3], [1, 2, 2], [NaN, -0.0, 2.0], 3, 3),
              sparse([1, 2, 3], [1, 1, 2], [NaN, -0.0, 2], 3, 3))
        @test isequal(X, Y) == isequal(Matrix(X), Matrix(Y))
        @test isequal(Y, X) == isequal(Matrix(Y), Matrix(X))
        @test (X == Y) == (Matrix(X) == Matrix(Y))
    end
end

@testset "isequal for adjoint/transpose of sparse matrices" begin
    n = 100
    A = spzeros(n, n); A[1, 1] = 1
    B = copy(A)
    for (L, R) in ((A', B'), (transpose(A), transpose(B)), (A, B'), (A', B),
                   (A, transpose(B)), (transpose(A), B), (A', transpose(B)))
        @test isequal(L, R)
    end
    A[1, 2] = 1; B[2, 1] = 1
    @test isequal(A, B') && isequal(A', B) && !isequal(A', B') && !isequal(A, B)
    @test !isequal(spzeros(2, 3)', spzeros(2, 3))
    # adjoint vs transpose of a complex matrix nests wrappers (`Adjoint{<:Any,<:Transpose}`)
    C = sparse([1, 2], [2, 3], [1.0im, 2.0], 3, 3)
    @test C' == transpose(conj(C)) && isequal(C', transpose(conj(C)))
    @test C' != transpose(C) && !isequal(C', transpose(C))
    # isequal semantics for NaN, signed zeros and conjugation must match dense arrays
    X = sparse([1, 2, 3, 1], [1, 1, 2, 3], [NaN, -0.0, 2.0, 1.0im], 3, 3)
    for Y in (sparse([1, 2, 3, 1], [1, 1, 2, 3], [NaN, -0.0, 2.0, 1.0im], 3, 3),
              sparse([1, 2, 3, 1], [1, 1, 2, 3], [NaN, -0.0, 2.0, -1.0im], 3, 3),
              sparse([1, 1, 2, 3], [1, 2, 3, 1], [NaN, -0.0, 2.0, 1.0im], 3, 3),
              sparse([1, 1, 2, 3], [1, 2, 3, 1], [NaN, -0.0, 2.0, -1.0im], 3, 3),
              sparse([1, 3, 1], [1, 2, 3], [NaN, 2.0, 1.0im], 3, 3),
              sparse([1, 2, 3, 1], [1, 1, 2, 3], [NaN, 0.0, 2.0, 1.0im], 3, 3),
              sparse([1, 2, 3, 1, 3], [1, 1, 2, 3, 3], [NaN, -0.0, 2.0, 1.0im, 0.0], 3, 3),
              sparse([1, 2, 3, 1], [1, 1, 2, 3], [1.0, -0.0, 2.0, 1.0im], 3, 3))
        for (L, R) in ((X', Y'), (transpose(X), transpose(Y)), (X, Y'), (X', Y),
                       (X, transpose(Y)), (transpose(X), Y), (X', transpose(Y)))
            @test isequal(L, R) == isequal(Matrix(L), Matrix(R))
            @test isequal(R, L) == isequal(Matrix(R), Matrix(L))
            @test (L == R) == (Matrix(L) == Matrix(R))
        end
    end
end

@testset "iszero specialization for SparseMatrixCSC" begin
    @test !iszero(sparse(I, 3, 3))                  # test failure
    @test iszero(spzeros(3, 3))                     # test success with no stored entries
    S = sparse(I, 3, 3)
    S[:] .= 0
    @test iszero(S)  # test success with stored zeros via broadcasting
    S = sparse(I, 3, 3)
    fill!(S, 0)
    @test iszero(S)  # test success with stored zeros via fill!
    @test_throws ArgumentError iszero(SparseMatrixCSC(2, 2, [1,2,3], [1,2], [0,0,1])) # test failure with nonzeros beyond data range
end
@testset "isone specialization for SparseMatrixCSC" begin
    @test isone(sparse(I, 3, 3))    # test success
    @test !isone(sparse(I, 3, 4))   # test failure for non-square matrix
    @test !isone(spzeros(3, 3))     # test failure for too few stored entries
    @test !isone(sparse(2I, 3, 3))  # test failure for non-one diagonal entries
    @test !isone(sparse(Bidiagonal(fill(1, 3), fill(1, 2), :U))) # test failure for non-zero off-diag entries
    # issue #763: stored zeros must not be counted towards the diagonal
    M = sparse([1 0; 1 1]) * sparse([1 0; -1 0])
    @test nnz(M) == 2 && !isone(M) && !isone(Matrix(M))
    @test !isone(SparseMatrixCSC(2, 2, [1, 3, 3], [1, 2], [1, 0]))
    @test !isone(SparseMatrixCSC(2, 2, [1, 2, 3], [1, 1], [1, 0]))
    @test isone(SparseMatrixCSC(2, 2, [1, 3, 4], [1, 2, 2], [1, 0, 1]))  # stored zero off-diagonal is fine
end

@testset "indtype" begin
    @test SparseArrays.indtype(sparse(Int8[1,1],Int8[1,1],[1,1])) == Int8
end

se33 = SparseMatrixCSC{Float64}(I, 3, 3)
do33 = fill(1.,3)

@testset "sparse binary operations" begin
    @test isequal(se33 * se33, se33)

    @test Array(se33 + convert(SparseMatrixCSC{Float32,Int32}, se33)) == Matrix(2I, 3, 3)
    @test Array(se33 * convert(SparseMatrixCSC{Float32,Int32}, se33)) == Matrix(I, 3, 3)

    @testset "shape checks for sparse elementwise binary operations equivalent to map" begin
        sqrfloatmat, colfloatmat = sprand(4, 4, 0.5), sprand(4, 1, 0.5)
        @test_throws DimensionMismatch (+)(sqrfloatmat, colfloatmat)
        @test_throws DimensionMismatch (-)(sqrfloatmat, colfloatmat)
        @test_throws DimensionMismatch map(min, sqrfloatmat, colfloatmat)
        @test_throws DimensionMismatch map(max, sqrfloatmat, colfloatmat)
        sqrboolmat, colboolmat = sprand(Bool, 4, 4, 0.5), sprand(Bool, 4, 1, 0.5)
        @test_throws DimensionMismatch map(&, sqrboolmat, colboolmat)
        @test_throws DimensionMismatch map(|, sqrboolmat, colboolmat)
        @test_throws DimensionMismatch map(xor, sqrboolmat, colboolmat)
    end

    # ascertain inference friendliness, ref. https://github.com/JuliaLang/julia/pull/25083#issuecomment-353031641
    sparsevec = SparseVector([1.0, 2.0, 3.0])
    @test map(-, Adjoint(sparsevec), Adjoint(sparsevec)) isa Adjoint{Float64,SparseVector{Float64,Int}}
    @test map(-, Transpose(sparsevec), Transpose(sparsevec)) isa Transpose{Float64,SparseVector{Float64,Int}}
    @test broadcast(-, Adjoint(sparsevec), Adjoint(sparsevec)) isa Adjoint{Float64,SparseVector{Float64,Int}}
    @test broadcast(-, Transpose(sparsevec), Transpose(sparsevec)) isa Transpose{Float64,SparseVector{Float64,Int}}
    @test broadcast(+, Adjoint(sparsevec), 1.0, Adjoint(sparsevec)) isa Adjoint{Float64,SparseVector{Float64,Int}}
    @test broadcast(+, Transpose(sparsevec), 1.0, Transpose(sparsevec)) isa Transpose{Float64,SparseVector{Float64,Int}}

    @testset "binary ops with matrices" begin
        λ = complex(randn(),randn())
        J = UniformScaling(λ)
        B = bitrand(2, 2)
        @test B + I == B + Matrix(I, size(B))
        @test I + B == B + Matrix(I, size(B))
        AA = randn(2, 2)
        for SS in (sprandn(3,3, 0.5), sparse(Int(1)I, 3, 3))
            for S in (SS, view(SS, 1:3, 1:3))
                @test @inferred(I*S) !== S # Don't alias
                @test @inferred(S*I) !== S # Don't alias

                @test @inferred(S*J) == S*λ
                @test @inferred(J*S) == S*λ
            end
        end
    end
    @testset "binary operations on sparse matrices with union eltype" begin
        A = sparse([1,2,1], [1,1,2], Union{Int, Missing}[1, missing, 0])
        MA = Array(A)
        for fun in (+, -, *, min, max)
            if fun in (+, -)
                @test collect(skipmissing(Array(fun(A, A)))) == collect(skipmissing(Array(fun(MA, MA))))
            end
            @test collect(skipmissing(Array(map(fun, A, A)))) == collect(skipmissing(map(fun, MA, MA)))
            @test collect(skipmissing(Array(broadcast(fun, A, A)))) == collect(skipmissing(broadcast(fun, MA, MA)))
        end
        b = convert(SparseMatrixCSC{Union{Float64, Missing}}, sprandn(Float64, 20, 10, 0.2)); b[rand(1:200, 3)] .= missing
        C = convert(SparseMatrixCSC{Union{Float64, Missing}}, sprandn(Float64, 20, 10, 0.9)); C[rand(1:200, 3)] .= missing
        CA = Array(C)
        D = convert(SparseMatrixCSC{Union{Float64, Missing}}, spzeros(Float64, 20, 10)); D[rand(1:200, 3)] .= missing
        E = convert(SparseMatrixCSC{Union{Float64, Missing}}, spzeros(Float64, 20, 10))
        for B in (b, C, D, E), fun in (+, -, *, min, max)
            BA = Array(B)
            # reverse order for opposite nonzeroinds-structure
            if fun in (+, -)
                @test collect(skipmissing(Array(fun(B, C)))) == collect(skipmissing(Array(fun(BA, CA))))
                @test collect(skipmissing(Array(fun(C, B)))) == collect(skipmissing(Array(fun(CA, BA))))
            end
            @test collect(skipmissing(Array(map(fun, B, C)))) == collect(skipmissing(map(fun, BA, CA)))
            @test collect(skipmissing(Array(map(fun, C, B)))) == collect(skipmissing(map(fun, CA, BA)))
            @test collect(skipmissing(Array(broadcast(fun, B, C)))) == collect(skipmissing(broadcast(fun, BA, CA)))
            @test collect(skipmissing(Array(broadcast(fun, C, B)))) == collect(skipmissing(broadcast(fun, CA, BA)))
        end
    end

end

let
    a116 = copy(reshape(1:16, 4, 4))
    s116 = sparse(a116)

    @testset "sparse ref" begin
        p = [4, 1, 2, 3, 2]
        @test Array(s116[p,:]) == a116[p,:]
        @test Array(s116[:,p]) == a116[:,p]
        @test Array(s116[p,p]) == a116[p,p]
    end

    @testset "sparse assignment" begin
        p = [4, 1, 3]
        a116[p, p] .= -1
        s116[p, p] .= -1
        @test a116 == s116

        p = [2, 1, 4]
        a116[p, p] = reshape(1:9, 3, 3)
        s116[p, p] = reshape(1:9, 3, 3)
        @test a116 == s116
    end
end

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
        # these actually throw due to #10533
        # @test f(x->sqrt(x-1), pA .+ 1, dims=1) ≈ f(sqrt(pA), dims=1)
        # @test f(x->sqrt(x-1), pA .+ 1, dims=2) ≈ f(sqrt(pA), dims=2)
        # @test f(x->sqrt(x-1), pA .+ 1, dims=3) ≈ f(pA)
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

@testset "findall" begin
    # issue described in https://groups.google.com/d/msg/julia-users/Yq4dh8NOWBQ/GU57L90FZ3EJ
    A = sparse(I, 5, 5); MA = Array(A)
    @test findall(A) == findall(x -> x == true, A) == findall(MA)
    # Non-stored entries are true
    @test findall(x -> x == false, A) == findall(x -> x == false, MA)

    # Not all stored entries are true
    @test findall(sparse([true false])) == [CartesianIndex(1, 1)]
    @test findall(x -> x > 1, sparse([1 2])) == [CartesianIndex(1, 2)]
end

@testset "access to undefined error types that initially allocate elements as #undef" begin
    @test sparse(1:2, 1:2, Number[1,2])^2 == sparse(1:2, 1:2, [1,4])
    sd1 = diff(sparse([1,1,1], [1,2,3], Number[1,2,3]), dims=1)
end

@testset "unary functions" begin
    A = sprand(5, 15, 0.5)
    C = A + im*A
    Afull = Array(A)
    Cfull = Array(C)
    # Test representatives of [unary functions that map zeros to zeros and may map nonzeros to zeros]
    @test sin.(Afull) == Array(sin.(A))
    @test tan.(Afull) == Array(tan.(A)) # should be redundant with sin test
    @test ceil.(Afull) == Array(ceil.(A))
    @test floor.(Afull) == Array(floor.(A)) # should be redundant with ceil test
    @test real.(Afull) == Array(real.(A)) == Array(real(A))
    @test imag.(Afull) == Array(imag.(A)) == Array(imag(A))
    @test conj.(Afull) == Array(conj.(A)) == Array(conj(A))
    @test real.(Cfull) == Array(real.(C)) == Array(real(C))
    @test imag.(Cfull) == Array(imag.(C)) == Array(imag(C))
    @test conj.(Cfull) == Array(conj.(C)) == Array(conj(C))
    # Test representatives of [unary functions that map zeros to zeros and nonzeros to nonzeros]
    @test expm1.(Afull) == Array(expm1.(A))
    @test abs.(Afull) == Array(abs.(A))
    @test abs2.(Afull) == Array(abs2.(A))
    @test abs.(Cfull) == Array(abs.(C))
    @test abs2.(Cfull) == Array(abs2.(C))
    # Test representatives of [unary functions that map both zeros and nonzeros to nonzeros]
    @test cos.(Afull) == Array(cos.(A))
    # Test representatives of remaining vectorized-nonbroadcast unary functions
    @test ceil.(Int, Afull) == Array(ceil.(Int, A))
    @test floor.(Int, Afull) == Array(floor.(Int, A))
    # Tests of real, imag, abs, and abs2 for SparseMatrixCSC{Int,X}s previously elsewhere
    for T in (Int, Float16, Float32, Float64, BigInt, BigFloat)
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

@testset "oneunit of sparse matrix" begin
    A = sparse([Second(0) Second(0); Second(0) Second(0)])
    @test oneunit(sprand(2, 2, 0.5)) isa SparseMatrixCSC{Float64}
    @test oneunit(A) isa SparseMatrixCSC{Second}
    @test one(sprand(2, 2, 0.5)) isa SparseMatrixCSC{Float64}
    @test one(A) isa SparseMatrixCSC{Int}
end

@testset "transpose! does not allocate" begin
    function f()
        A = sprandn(10, 10, 0.1)
        X = copy(A)
        return @allocated transpose!(X, A)
    end
    #precompile
    f()
    f()
    @test f() == 0
end

struct Counting{T} <: Number
    elt::T
end
@static if VERSION ≥ v"1.8"
    counter::Int = 0
    resetcounter() = (global counter; counter=0)
    stepcounter() = (global counter; counter+=1)
    getcounter() = (global counter; counter)
else
    const counter = Ref(0)
    resetcounter() = (global counter; counter[]=0)
    stepcounter() = (global counter; counter[]+=1)
    getcounter() = (global counter; counter[])
end
Base.:(==)(x::Counting, y::Counting) = (stepcounter(); x.elt==y.elt)
Base.promote_rule(::Type{Counting{T}}, ::Type{Counting{U}}) where {T,U} = Counting{promote_rule(T, U)}
Base.iszero(x::Counting) = iszero(x.elt)
Base.zero(::Type{Counting{T}}) where {T} = Counting(zero(T))
Base.zero(x::Counting) = Counting(zero(x.elt))
Base.adjoint(x::Counting) = Counting(adjoint(x.elt))
Base.transpose(x::Counting) = Counting(transpose(x.elt))
Base.isequal(x::Counting, y::Counting) = (stepcounter(); isequal(x.elt, y.elt))

# Deterministic replacement for wall-clock guards: with a counting eltype, a comparison
# that walks only stored entries performs at most nnz(A) + nnz(B) element comparisons,
# whereas the generic AbstractArray fallback performs length(A) of them.
@testset "== and isequal walk stored entries only (issues #561, #766, #768)" begin
    n = 1000
    v = sparsevec([1, n ÷ 2], Counting.([1.0, 2.0]), n)
    w = sparsevec([1, n ÷ 2, n], Counting.([1.0, 0.0, 3.0]), n)
    A = sparse([1, n ÷ 2], [1, n], Counting.([1.0, 2.0]), n, n)
    B = sparse([1, n ÷ 2, 7], [1, n, 7], Counting.([1.0, 2.0, 0.0]), n, n)
    for (x, y) in ((v, v), (v, w), (w, v), (v', w'), (transpose(v), transpose(w)),
                   (A, A), (A, B), (B, A), (A', B'), (transpose(A), transpose(B)),
                   (A, B'), (A', B), (A, transpose(B)), (transpose(A), B), (A', transpose(B)))
        budget = nnz(parent(x isa Union{Adjoint,Transpose} ? x : x') ) +
                 nnz(parent(y isa Union{Adjoint,Transpose} ? y : y'))
        for eq in (==, isequal)
            resetcounter()
            eq(x, y)
            @test getcounter() <= budget
        end
    end
end

@testset "Comparisons to adjoints are efficient" for
    A in Any[sparse(1*I(10000)), sprandn(10000, 10000, 0.00001), sprandn(ComplexF64, 100, 100, 0.9)],
    B in Any[sparse(1*I(10000)), sprandn(10000, 10000, 0.00001), sprandn(ComplexF64, 100, 100, 0.9)]
    if size(A) == size(B)
        A = Counting.(A)
        B = Counting.(B)
        As = Any[A, A', transpose(A)]
        Bs = Any[B, B', transpose(B)]
        for A′ in As, B′ in Bs
            # skip adjoints of transposes; these are not really supported
            ((A′ isa Adjoint && B′ isa Transpose) || (A′ isa Transpose && B′ isa Adjoint)) && continue
            c = (resetcounter(); A′ == B′; getcounter())
            @test c ≤ 1 + (nnz(A′) + nnz(B′))
        end
    end
end


@testset "Issue #246" begin
    for t in [Int, UInt8, Float64]
        a = Counting.(sprand(t, 100, 0.5))
        b = Counting.(sprand(t, 100, 0.5))

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

            resetcounter()
            ma == mb
            @test getcounter() <= nnz(a) + nnz(b)

            @test (mc == md) == (Array(mc) == Array(md))
        end
    end
end

@testset "copytrito!" begin
    S = sparse([1,2,2,2,3], [1,1,2,2,4], [5, -19, 73, 12, -7])
    M = fill(Inf, size(S))
    copytrito!(M, S, 'U')
    for col in axes(S, 2)
        for row in 1:min(col, size(S,1))
            @test M[row, col] == S[row, col]
        end
        for row in min(col, size(S,1))+1:size(S,1)
            @test isinf(M[row, col])
        end
    end
    M .= Inf
    copytrito!(M, S, 'L')
    for col in axes(S, 2)
        for row in 1:col-1
            @test isinf(M[row, col])
        end
        for row in col:size(S, 1)
            @test M[row, col] == S[row, col]
        end
    end
    @test_throws ArgumentError copytrito!(M, S, 'M')
end

@testset "istriu/istril" begin
    for T in Any[Tridiagonal(1:3, 1:4, 1:3),
                    Bidiagonal(1:4, 1:3, :U), Bidiagonal(1:4, 1:3, :L),
                    Diagonal(1:4),
                    diagm(-2=>1:2, 2=>1:2)]
        S = sparse(T)
        for k in -5:5
            @test istriu(S, k) == istriu(T, k)
            @test istril(S, k) == istril(T, k)
        end
    end
end

@testset "isdiag" begin
    # Diagonal matrices should return true
    @test isdiag(sparse(Diagonal(1:4)))
    @test isdiag(sparse(Diagonal([1.0, 2.0, 3.0])))
    @test isdiag(spzeros(5, 5))  # Empty matrix is diagonal

    # Non-diagonal matrices should return false
    @test !isdiag(sparse(Tridiagonal(1:3, 1:4, 1:3)))
    @test !isdiag(sparse(Bidiagonal(1:4, 1:3, :U)))
    @test !isdiag(sparse(Bidiagonal(1:4, 1:3, :L)))
    @test !isdiag(sparse([1 2; 3 4]))

    # Non-square diagonal matrices should return true (consistent with generic isdiag)
    @test isdiag(sparse([1 0 0; 0 2 0]))  # 2x3 diagonal
    @test isdiag(sparse([1 0; 0 2; 0 0]))  # 3x2 diagonal
    @test isdiag(spzeros(3, 5))  # Empty non-square matrix is diagonal
    @test isdiag(spzeros(5, 3))

    # Non-square non-diagonal matrices should return false
    @test !isdiag(sparse([1 1 0; 0 2 0]))  # Off-diagonal element
    @test !isdiag(sparse([1 0; 0 2; 1 0]))  # Off-diagonal element

    # Consistency with dense isdiag
    for T in Any[Diagonal(1:4), Tridiagonal(1:3, 1:4, 1:3),
                 Bidiagonal(1:4, 1:3, :U), diagm(-1=>1:3, 1=>1:3)]
        S = sparse(T)
        @test isdiag(S) == isdiag(T)
    end

    # Explicit zeros on off-diagonal should still be diagonal
    S = sparse([1, 2, 1], [1, 2, 2], [1.0, 2.0, 0.0])
    @test isdiag(S)
end

@testset "sort/sort! of a sparse matrix" begin
    # `sort` of a dense matrix with `size(M, dims) == 0` errors in Base, so those cases are
    # compared against the input itself rather than against a dense reference
    # `dims = 2` covers the transposed shapes, so only one orientation of each is listed;
    # fully structural matrices are covered by the "empty and zero-size matrices" testset
    @testset "size = ($m, $n), density = $d" for (m, n) in ((6, 5), (1, 1), (0, 3), (1, 9),
                                                            (20, 13)),
                                                 d in (0.3, 1.0)
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

    @testset "index type $Ti" for Ti in (Int32, Int64)
        A = SparseMatrixCSC{Float64,Ti}(sprand(11, 7, 0.4))
        for dims in (1, 2)
            @test sort(A; dims) isa SparseMatrixCSC{Float64,Ti}
            @test Matrix(sort(A; dims)) == sort(Matrix(A); dims)
        end
    end

    @testset "keyword arguments" begin
        A = sprand(50, 50, 0.1)
        # `scratch` is forwarded to the underlying `sort!` and ignored by the search for
        # where the structural zeros belong (see #335)
        @test Matrix(sort!(copy(A); dims=1, scratch=Vector{Float64}(undef, 50))) ==
            sort(Matrix(A); dims=1)
        @test_throws MethodError sort!(copy(A); dims=1, banana=:blue)
        @test_throws ArgumentError sort!(copy(A); dims=3)
        @test_throws ArgumentError sort!(copy(A); dims=0)
        @test_throws UndefKeywordError sort!(copy(A))
        # keywords are validated even when every column is structurally empty, or when
        # there are no columns (or rows) at all
        for Z in (spzeros(3, 3), spzeros(3, 0), spzeros(0, 3)), dims in (1, 2)
            @test_throws MethodError sort!(copy(Z); dims, banana=:blue)
            @test_throws TypeError sort!(copy(Z); dims, rev=1)
        end
        # the ordering is only evaluated at zero when there are structural zeros to place,
        # so a `by` that is undefined at zero works on fully stored columns as it does for
        # dense matrices
        F = sparse([1 2; 3 4])
        for dims in (1, 2)
            @test Matrix(sort(F; dims, by = x -> 1 ÷ x)) == sort(Matrix(F); dims, by = x -> 1 ÷ x)
        end
        @test_throws DivideError sort(sparse([1 0; 3 4]); dims=1, by = x -> 1 ÷ x)
    end

    @testset "shared scratch buffer" begin
        # each column is sorted with one shared scratch buffer rather than a fresh one
        # per column, so the allocation count does not grow with the number of columns;
        # the columns are long enough for Base to want a scratch buffer, but short enough
        # to stay below its radix sort, which allocates a counts vector per call
        A = sprand(400, 400, 0.5)
        B = copy(A)
        sort!(B; dims=1) # compile
        B = copy(A)
        nallocs = @allocations sort!(B; dims=1)
        @test nallocs < size(A, 2)
        @test Matrix(B) == sort(Matrix(A); dims=1)
    end

    @testset "column views" begin
        A = sprand(7, 4, 0.5)
        M = Matrix(A)
        for j in axes(A, 2), kws in ((;), (; rev=true), (; by=abs))
            B = copy(A)
            c = view(B, :, j)
            @test sort!(c; kws...) === c
            @test nnz(B) == nnz(A)
            expected = copy(M)
            sort!(view(expected, :, j); kws...)
            @test Matrix(B) == expected
        end
    end

    @testset "fixed matrices" begin
        A = sprand(6, 5, 0.4)
        F = fixed(A)
        for dims in (1, 2)
            # `sort!` refuses to touch the read-only structure and leaves `F` intact
            @test_throws ArgumentError sort!(F; dims)
            @test F == A
            # `sort` returns a writable copy
            S = sort(F; dims)
            @test S isa SparseMatrixCSC
            @test !_is_fixed(S)
            @test Matrix(S) == sort(Matrix(A); dims)
            @test F == A
        end
        Z = fixed(spzeros(3, 3))
        for dims in (1, 2)
            @test_throws ArgumentError sort!(Z; dims)
            @test sort(Z; dims) == Z
        end
    end

    @testset "empty and zero-size matrices" begin
        # `Base.sort` on a *dense* matrix with `size(M, dims) == 0` throws
        # `ArgumentError: step cannot be zero`, so there is no dense reference to compare
        # against for every `dims` here; the sparse methods just return the (empty) matrix
        # unchanged
        @testset "size = ($m, $n)" for (m, n) in ((0, 3), (3, 0), (0, 0))
            A = spzeros(m, n)
            for dims in (1, 2)
                B = copy(A)
                @test sort!(B; dims) === B
                @test size(B) == (m, n)
                @test nnz(B) == 0
                @test B == A
                S = sort(A; dims)
                @test S isa SparseMatrixCSC{Float64,Int}
                @test size(S) == (m, n)
                @test nnz(S) == 0
            end
        end

        # structurally empty, but not zero-size: here dense does give a reference
        @testset "all structural zeros, size = ($m, $n)" for (m, n) in ((1, 1), (5, 4))
            A = spzeros(m, n)
            for dims in (1, 2)
                B = sort!(copy(A); dims)
                @test Matrix(B) == sort(Matrix(A); dims)
                @test nnz(B) == 0
                @test getcolptr(B) == getcolptr(A)
            end
        end

        # a single column/row that is entirely structural next to a populated one
        A = SparseMatrixCSC(4, 3, [1, 1, 5, 5], [1, 2, 3, 4], [1.0, -2.0, 0.0, 3.0])
        for dims in (1, 2)
            @test Matrix(sort(A; dims)) == sort(Matrix(A); dims)
            @test nnz(sort(A; dims)) == nnz(A)
        end
    end

    @testset "stored zeros" begin
        # column 1 stores an explicit zero next to structural zeros
        A = SparseMatrixCSC(4, 2, [1, 3, 4], [1, 3, 2], [0.0, -1.0, 2.0])
        for dims in (1, 2)
            @test Matrix(sort(A; dims)) == sort(Matrix(A); dims)
            @test nnz(sort(A; dims)) == nnz(A)
        end
    end
end

end # module
