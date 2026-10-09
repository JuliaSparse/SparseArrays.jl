# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseMatrixTests

using Test
using SparseArrays
using SparseArrays: getcolptr, nonzeroinds, _show_with_braille_patterns, _isnotzero, _isimplicitzero, fixed, _is_fixed
using LinearAlgebra
using Random
include("testhelpers.jl")

@testset "_isnotzero" begin
    @test !_isnotzero(0::Int)
    @test _isnotzero(1::Int)
    @test _isnotzero(missing)
    @test !_isnotzero(0.0)
    @test _isnotzero(1.0)
    @test _isnotzero([1.0]) && !_isnotzero([0.0])
    @test !SparseArrays._iszero(missing)   # `iszero(missing)` is not a `Bool`
end

@testset "_isimplicitzero" begin
    @test _isimplicitzero(0, Int)
    @test !_isimplicitzero(1, Int)
    @test _isimplicitzero(0.0, Float64)
    @test !_isimplicitzero(-0.0, Float64)   # egality keeps the sign of a zero
    @test !_isimplicitzero(missing, Union{Missing,Int})
    @test _isimplicitzero(big(0.0), BigFloat)
    @test !_isimplicitzero(-big(0.0), BigFloat)
    @static if COMPREHENSIVE
    @test _isimplicitzero(zero(Complex{BigFloat}), Complex{BigFloat})
    @test !_isimplicitzero(Complex{BigFloat}(0, 1), Complex{BigFloat})
    @test !_isimplicitzero([0.0], Vector{Float64})   # array elements are always stored
    end
end

@static if COMPREHENSIVE
@testset "issparse" begin
    @test issparse(sparse(fill(1,5,5)))
    @test !issparse(fill(1,5,5))
    @test nnz(zero(sparse(fill(1,5,5)))) == 0
end
end

@testset "iteratenz" begin
    for i in 1:20
        A = sprandn(100, 100, 1 / i)
        @test collect(SparseArrays.iternz(A)) == collect(zip(findnz(A)...))
    end
    A = sprandn(20, 30, 0.2)
    V = view(A, :, 4:17)
    @test collect(SparseArrays.iternz(V)) == collect(zip(findnz(sparse(V))...))
    @test length(SparseArrays.iternz(V)) == nnz(V)
    @static if COMPREHENSIVE
        for cols in (5:4, 31:30, [9, 2, 2, 30], Int[], 2:3:29, :)
            V = view(A, :, cols)
            @test collect(SparseArrays.iternz(V)) == collect(zip(findnz(sparse(V))...))
            @test length(SparseArrays.iternz(V)) == nnz(V)
        end
        A = sparse(Int32[1, 3], Int32[2, 2], [1.0, 2.0], 3, 4)
        @test typeof(first(SparseArrays.iternz(view(A, :, 2:3)))) == Tuple{Int32, Int32, Float64}
        @test typeof(first(SparseArrays.iternz(A))) == eltype(SparseArrays.iternz(A)) == Tuple{Int32, Int32, Float64}
        x = sparsevec(Int32[2, 5], [1.0, 2.0], 6)
        @test typeof(first(SparseArrays.iternz(x))) == eltype(SparseArrays.iternz(x)) == Tuple{Int32, Float64}
    end
end

@testset "findnz for adjoint/transpose (issue #632)" begin
    A = sparse([1, 1, 2, 3], [1, 2, 3, 2], [1.0+2.0im, 3.0, 4.0-1.0im, 0.0], 3, 4)
    for T in (ComplexF64,), op in (@static COMPREHENSIVE ? (adjoint, transpose) : (adjoint,))
        B = op(A)
        I, J, V = findnz(B)
        @test (I, J, V) == findnz(SparseMatrixCSC(B))
        @static if COMPREHENSIVE
        @test issorted(collect(zip(J, I)))  # column-major order of the wrapper
        @test all(B[i, j] == v for (i, j, v) in zip(I, J, V))
        end
        @test length(I) == nnz(B)
        @test typeof(I) == typeof(J) == Vector{Int} && eltype(V) == T
        @test all(isempty, findnz(op(spzeros(T, 2, 3))))
    end
    x = sparsevec([2, 4], [1.0+im, 0.0], 5)
    for op in (@static COMPREHENSIVE ? (adjoint, transpose) : (adjoint,))
        I, J, V = findnz(op(x))
        @test I == [1, 1] && J == [2, 4] && V == op.([1.0+im, 0.0])
        @test (I, J, V) == findnz(sparse(op(x)))
        @static if COMPREHENSIVE
        @test all(isempty, findnz(op(spzeros(3))))
        end
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
    @static if COMPREHENSIVE
    # column-range and column-subset views compare through the stored-entry merge on
    # either side instead of the elementwise AbstractArray fallback
    P = sparse([1, 2, 3, 1], [1, 2, 3, 4], [1.0, 0.0, 2.0, 5.0], 3, 4)
    Q = sparse([1, 3], [1, 3], [1.0, 2.0], 3, 3)    # P's stored zero is implicit here
    @test view(P, :, 1:3) == Q && isequal(Q, view(P, :, [1, 2, 3]))
    @test view(P, :, 1:3) != sparse([1, 2, 3], [1, 2, 3], [1.0, 1.0, 2.0], 3, 3) &&
          !isequal(view(P, :, [3, 2, 1]), view(P, :, 1:3))
    end
end

@testset "hash matches dense" begin
    # The stored-entries-only complexity guarantee is checked with an operation-counting
    # eltype below ("hash walks stored entries only").
    n = 1000
    A = spzeros(n, n); A[1, 1] = 1
    B = copy(A); B[2, 2] = 0.0   # explicitly stored zero must not change the hash
    @test hash(B) == hash(A) && isequal(B, A)
    # entries scattered over the matrix, so that the runs of zeros between them vary in length
    scattered(m, k) = sparse([mod1(i * i, m) for i in 1:k], [mod1(3i, m) for i in 1:k], [1.5i for i in 1:k], m, m)
    for m in ((@static COMPREHENSIVE ? (2,) : ())..., 10, 200), X in (scattered(m, 2m), (@static COMPREHENSIVE ? (-scattered(m, m * m ÷ 3),) : ())..., spzeros(m, m))
        k = min(3, nnz(X)); nonzeros(X)[1:k] .= [NaN, -0.0, 0.0][1:k]
        @test hash(X) == hash(Matrix(X))
        @test hash(X, UInt(7)) == hash(Matrix(X), UInt(7))
    end
end

@testset "isequal for adjoint/transpose of sparse matrices" begin
    n = 100
    A = spzeros(n, n); A[1, 1] = 1
    B = copy(A)
    for (L, R) in ((A', B'), (A, B'), (transpose(A), B),
                   (@static COMPREHENSIVE ? ((A', B),) : ())...)
        @test isequal(L, R)
    end
    A[1, 2] = 1; B[2, 1] = 1
    @test isequal(A, B') && isequal(A', B) && !isequal(A', B') && !isequal(A, B)
    @test !isequal(spzeros(2, 3)', spzeros(2, 3))
    # the wrapped operand stores entries that the other one does not
    @test spzeros(n, n) != B' && !isequal(spzeros(n, n), B') && B' != spzeros(n, n)
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
        for (L, R) in ((X', Y'), (X, Y'), (@static COMPREHENSIVE ? ((transpose(X), transpose(Y)),
                       (X', Y), (X, transpose(Y)), (X', transpose(Y))) : ())...)
            @test isequal(L, R) == isequal(Matrix(L), Matrix(R))
            @test isequal(R, L) == isequal(Matrix(R), Matrix(L))
            @test (L == R) == (Matrix(L) == Matrix(R))
        end
    end
end

@static if COMPREHENSIVE
@testset "iszero specialization for SparseMatrixCSC" begin
    @test !iszero(sparse(I, 3, 3))                  # test failure
    @test iszero(spzeros(3, 3))                     # test success with no stored entries
    @static if COMPREHENSIVE
    S = sparse(I, 3, 3)
    S[:] .= 0
    @test iszero(S)  # test success with stored zeros via broadcasting
    end
    S = sparse(I, 3, 3)
    fill!(S, 0)
    @test iszero(S)  # test success with stored zeros via fill!
    @test_throws ArgumentError iszero(SparseMatrixCSC(2, 2, [1,2,3], [1,2], [0,0,1])) # test failure with nonzeros beyond data range
end
end

@testset "isone specialization for SparseMatrixCSC" begin
    @test isone(sparse(I, 3, 3))    # test success
    @test !isone(sparse(I, 3, 4))   # test failure for non-square matrix
    @test !isone(spzeros(3, 3))     # test failure for too few stored entries
    @test !isone(sparse(2I, 3, 3))  # test failure for non-one diagonal entries
    @test !isone(sparse(Bidiagonal(fill(1, 3), fill(1, 2), :U))) # test failure for non-zero off-diag entries
    @static if COMPREHENSIVE
    # issue #763: stored zeros must not be counted towards the diagonal
    M = sparse([1 0; 1 1]) * sparse([1 0; -1 0])
    @test nnz(M) == 2 && !isone(M) && !isone(Matrix(M))
    @test !isone(SparseMatrixCSC(2, 2, [1, 3, 3], [1, 2], [1, 0]))
    end
    @test !isone(SparseMatrixCSC(2, 2, [1, 2, 3], [1, 1], [1, 0]))
    @test isone(SparseMatrixCSC(2, 2, [1, 3, 4], [1, 2, 2], [1, 0, 1]))  # stored zero off-diagonal is fine
end

@testset "a subtype implementing rowvals and nonzeros but not the get* accessors" begin
    S = sparse([1.0 0 2; 0 3 0; 4 0 5])
    A = LegacyCSC(S)
    @test SparseArrays.getrowval(A) === rowvals(S) && SparseArrays.getnzval(A) === nonzeros(S)
    @test nonzeros(view(A, :, 2:3)) == nonzeros(view(S, :, 2:3))
    @test A[2, 2] == 3
    @test copy(A) == S
    x = [1.0, 2, 3]
    @test A * x == S * x
    @test A' * x == S' * x
    @test A * S == S * A == A * A == S * S
    @test A * Matrix(S) == S * Matrix(S)
end

@testset "indtype" begin
    Ti = @static COMPREHENSIVE ? Int8 : Int
    A = sparse(Ti[1,1],Ti[1,1],[1,1])
    @test SparseArrays.indtype(A) == Ti
    for W in (A', transpose(A), Symmetric(A), Hermitian(A), UpperTriangular(A),
              view(A, :, 1:1), view(A, :, [1]), view(A, :, 1))
        @test SparseArrays.indtype(W) == Ti
    end
    @test SparseArrays.indtype(sparsevec(Ti[1], [1.0])') == Ti
end

@testset "exported CSC accessors" begin
    for name in (:AbstractSparseMatrixCSC, :getcolptr, :getrowval, :getnzval, :indtype)
        @test Base.isexported(SparseArrays, name)
    end
end

@testset "sparse binary operations" begin
    se33 = SparseMatrixCSC{Float64}(I, 3, 3)
    @test isequal(se33 * se33, se33)

    @static if COMPREHENSIVE
    @test Array(se33 + convert(SparseMatrixCSC{Float32,Int32}, se33)) == Matrix(2I, 3, 3)
    end

    @testset "shape checks for sparse elementwise binary operations equivalent to map" begin
        sqrfloatmat, colfloatmat = fixture(Float64, 4, 4), fixture(Float64, 4, 1)
        @test_throws DimensionMismatch (+)(sqrfloatmat, colfloatmat)
        @test_throws DimensionMismatch map(min, sqrfloatmat, colfloatmat)
    end

    @static if COMPREHENSIVE
    # ascertain inference friendliness, ref. https://github.com/JuliaLang/julia/pull/25083#issuecomment-353031641
    sparsevec = SparseVector([1.0, 2.0, 3.0])
    @test map(-, Adjoint(sparsevec), Adjoint(sparsevec)) isa Adjoint{Float64,SparseVector{Float64,Int}}
    @test broadcast(-, Transpose(sparsevec), Transpose(sparsevec)) isa Transpose{Float64,SparseVector{Float64,Int}}
    @test broadcast(+, Adjoint(sparsevec), 1.0, Adjoint(sparsevec)) isa Adjoint{Float64,SparseVector{Float64,Int}}
    end

    @testset "binary ops with matrices" begin
        λ = complex(0.5, -1.5)
        J = UniformScaling(λ)
        @static if COMPREHENSIVE
        for R in (fixture(Float64, 2, 3), spzeros(3, 0))
            @test_throws DimensionMismatch R + I
            @test_throws DimensionMismatch I + R
            @test_throws DimensionMismatch R - J
            @test_throws DimensionMismatch J - R
        end
        end
        for SS in (fixture(Float64, 3, 3), (@static COMPREHENSIVE ? (sparse(Int(1)I, 3, 3),) : ())...)
            for S in (SS,)
                @test @inferred(I*S) !== S # Don't alias
                @test @inferred(S*I) !== S # Don't alias

                @test @inferred(S*J) == S*λ
                @test @inferred(J*S) == S*λ
            end
        end
    end
    @testset "sparse with strided dense is dense" begin
        # The result type proves the strided methods are dispatched to: the AbstractArray
        # fallback broadcasts, which makes a sparse result out of a dense operand.
        @static if COMPREHENSIVE
        # the kinds of strided operand that the lines after this block leave out: a view
        # with a step, a transpose and the adjoint of a view, and a sparse view with a matrix
        for (T, fun, k) in ((Float64, +, 1), (ComplexF64, -, 2), (Float64, -, 3), (ComplexF64, +, 4))
            S = fixture(T, 5, 4)
            S[2, 3] = 0   # stored zero
            M = T[T <: Real ? i + 2j : complex(i + 2j, i - 3j) for i in 1:5, j in 1:4]
            X, Y = k == 1 ? (S, view(repeat(M, 1, 2), :, 1:2:7)) :
                   k == 2 ? (S, transpose(Matrix(transpose(M)))) :
                   k == 3 ? (S, view(Matrix(M'), 1:4, :)') : (view(S, :, 2:4), M[:, 2:4])
            A, B = Array(X), Array(Y)
            @test @inferred(fun(X, Y))::Matrix{T} == fun(A, B)
            @test @inferred(fun(Y, X))::Matrix{T} == fun(B, A)
            k == 4 || continue
            @test UpperTriangular(M[1:4, :]) - S[1:4, :] isa Matrix{T}
            @test S[1:4, :] + Symmetric(M[1:4, :]) isa Matrix{T}
        end
        end
        # The kernel is the same for every eltype and operator, so each kind of operand pair
        # (plain, views on both sides, adjoint dense) runs once. These are written out because
        # a loop over operand pairs compiles tuple iteration for every pair of types.
        S = fixture(Float64, 5, 4)
        S[2, 3] = 0   # stored zero
        M = reshape(Float64.(1:20), 5, 4)
        @test @inferred(S + M)::Matrix{Float64} == Array(S) + M
        @test @inferred(M + S)::Matrix{Float64} == M + Array(S)
        SV, MV = view(S, :, 2:4), view(M, :, 2:4)
        @test @inferred(SV - MV)::Matrix{Float64} == Array(SV) - Array(MV)
        @test @inferred(MV - SV)::Matrix{Float64} == Array(MV) - Array(SV)
        C = fixture(ComplexF64, 5, 4)
        N = (reshape(Float64.(1:20), 4, 5) * (1.0 - 2.0im))'
        @test @inferred(C + N)::Matrix{ComplexF64} == Array(C) + Array(N)
        @test @inferred(N + C)::Matrix{ComplexF64} == Array(N) + Array(C)
        @test S[1:4, :] + Symmetric(M[1:4, :]) isa Matrix{Float64}
        @test_throws DimensionMismatch S + M[:, 1:3]
        @test_throws DimensionMismatch M[:, 1:3]' - S
        # promotion follows the dense method
        @test sparse([1, 2], [1, 2], [1, 2]) + [1.5 0; 0 1.5] isa Matrix{Float64}
        @test sparse([1], [1], Real[1.5], 2, 2) + [1 2; 3 4]' == [2.5 3; 2 4]
    end
    @testset "binary operations on sparse matrices with union eltype" begin
        @static if COMPREHENSIVE
        A = sparse([1,2,1], [1,1,2], Union{Int, Missing}[1, missing, 0])
        MA = Array(A)
        for fun in (-, max)
            if fun in (+, -)
                @test collect(skipmissing(Array(fun(A, A)))) == collect(skipmissing(Array(fun(MA, MA))))
            end
            @test collect(skipmissing(Array(map(fun, A, A)))) == collect(skipmissing(map(fun, MA, MA)))
            @test collect(skipmissing(Array(broadcast(fun, A, A)))) == collect(skipmissing(broadcast(fun, MA, MA)))
        end
        end
        # `b` is sparse and `C` nearly full, and each gets a `missing` on a stored entry and
        # on an unstored one
        b = convert(SparseMatrixCSC{Union{Float64, Missing}}, sparse([2, 7, 7, 13, 20, 1, 9], [1, 1, 4, 4, 6, 9, 10], [1.5, -2.0, 0.0, 4.0, -0.5, 3.0, 7.0], 20, 10)); b[[2, 64, 150]] .= missing
        C = convert(SparseMatrixCSC{Union{Float64, Missing}}, sparse([(i + 2j) % 7 == 0 ? 0.0 : i - 2.5j for i in 1:20, j in 1:10])); C[[5, 7, 150]] .= missing
        CA = Array(C)
        D = convert(SparseMatrixCSC{Union{Float64, Missing}}, spzeros(Float64, 20, 10)); D[[3, 64, 199]] .= missing
        E = convert(SparseMatrixCSC{Union{Float64, Missing}}, spzeros(Float64, 20, 10))
        for B in (b, (@static COMPREHENSIVE ? (C, D) : ())..., E), fun in (+, (@static COMPREHENSIVE ? (*, min) : ())...)
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

@static if COMPREHENSIVE
@testset "dropdims" begin
    for n in (20, 7, 1)
        am = fixture(Float64, n, 1)
        av = dropdims(am, dims=2)
        @test ndims(av) == 1
        @test all(av.==am)
        am = fixture(Float64, 1, n)
        av = dropdims(am, dims=1)
        @test ndims(av) == 1
        @test all(av' .== am)
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
    @static if COMPREHENSIVE
        # matches at irregular positions, none, and every stored entry
        B = sparse([1, 3, 2, 3, 1, 4], [1, 1, 2, 3, 5, 5], [3.0, -1.0, 2.0, -4.0, 7.0, 5.0], 4, 5)
        for p in (>(0), >(9), !iszero)
            @test findall(p, B) == findall(p, Array(B))
        end
        @test_throws TypeError findall(x -> 1, B)
    end
end

@testset "access to undefined error types that initially allocate elements as #undef" begin
    @test sparse(1:2, 1:2, Number[1,2])^2 == sparse(1:2, 1:2, [1,4])
    sd1 = diff(sparse([1,1,1], [1,2,3], Number[1,2,3]), dims=1)
end

@testset "unary functions" begin
    A = fixture(Float64, 3, 5)
    C = fixture(ComplexF64, 3, 5)
    Afull = Array(A)
    Cfull = Array(C)
    # Test representatives of [unary functions that map zeros to zeros and may map nonzeros to zeros]
    @test mismatch(sin.(A), sin.(Afull)) === nothing
    @static if COMPREHENSIVE
    @test mismatch(ceil.(A), ceil.(Afull)) === nothing
    @test iszero(imag.(A)) && nnz(imag.(A)) == nnz(A)   # broadcast keeps the pattern
    @test mismatch(conj.(C), conj.(Cfull)) === nothing
    # Test representatives of [unary functions that map zeros to zeros and nonzeros to nonzeros]
    @test mismatch(expm1.(A), expm1.(Afull)) === nothing
    @test mismatch(abs.(C), abs.(Cfull)) === nothing
    end
    @test mismatch(real(C), real(Cfull)) === nothing
    @test mismatch(imag(C), imag(Cfull)) === nothing
    @test mismatch(conj(C), conj(Cfull)) === nothing
    # `real` and `conj` of a real matrix alias it, and its `imag` stores nothing
    @test real(A) === A
    @test conj(A) === A
    @test nnz(imag(A)) == 0 && size(imag(A)) == size(A)
    # the negation owns its buffers
    @test mismatch(-C, -Cfull) === nothing
    @test getcolptr(-C) !== getcolptr(C) && rowvals(-C) !== rowvals(C)
    # Test representatives of [unary functions that map both zeros and nonzeros to nonzeros]
    @test cos.(A)::Matrix{Float64} == cos.(Afull)
    # Test representatives of remaining vectorized-nonbroadcast unary functions
    @test mismatch(ceil.(Int, A), ceil.(Int, Afull)) === nothing
end

@testset "dense copies and sums when the zero is of another type than the eltype" begin
    S = sparse([1, 1, 2], [1, 2, 2], Variable.(1:3))
    M = [10 20; 30 40]
    E = Expression
    for B in (M, (@static COMPREHENSIVE ? (Matrix(M')',) : ())...)
        @test (S + B)::Matrix{E} == [E(11) E(22); E(30) E(43)]
        @test (B + S)::Matrix{E} == [E(11) E(22); E(30) E(43)]
        @test (S - B)::Matrix{E} == [E(-9) E(-18); E(-30) E(-37)]
        @test (B - S)::Matrix{E} == [E(9) E(18); E(30) E(37)]
        @test (view(S, :, 1:2) + B)::Matrix{E} == [E(11) E(22); E(30) E(43)]
    end
    D = [E(1) E(2); E(0) E(3)]
    v = sparsevec([2], [Variable(1)], 2)
    for f in (Array, (@static COMPREHENSIVE ? (Matrix,) : ())...)
        @test f(S)::Matrix{E} == D
        @test f(transpose(S))::Matrix{E} == permutedims(D)
        @static if COMPREHENSIVE
        @test f(view(S, :, 1:2))::Matrix{E} == D
        @test f(transpose(v))::Matrix{E} == [E(0) E(1)]
        end
    end
    for f in (Array, (@static COMPREHENSIVE ? (Vector,) : ())...)
        @test f(v)::Vector{E} == [E(0), E(1)]
        @static if COMPREHENSIVE
        @test f(view(S, :, 1))::Vector{E} == [E(1), E(0)]
        @test f(view(v, 1:2))::Vector{E} == [E(0), E(1)]
        end
    end
    @test collect(S)::Matrix{E} == D
    @test collect(v)::Vector{E} == [E(0), E(1)]
    @test (v + [10, 20])::Vector{E} == [E(10), E(21)]
    @static if COMPREHENSIVE
    @test ([10, 20] - v)::Vector{E} == [E(10), E(19)]
    @test hcat(S, M) == hcat(D, M)
    @test vcat(S, M) == vcat(D, M)
    end
    # a mutable zero is not shared between positions
    Z = Array(sparse([1], [1], [Variable(1)], 2, 2))
    @test Z[1, 2] !== Z[2, 2]
    Z = sparse([1], [1], [Variable(1)], 2, 2) + M
    @test Z[1, 2] !== Z[2, 2]
    # an eltype named by the caller is kept
    @test_throws MethodError Matrix{Variable}(S)
    @static if COMPREHENSIVE
    # a structurally dense array needs no zero
    @test Array(sparse([1, 2, 1, 2], [1, 1, 2, 2], Variable.(1:4)))::Matrix{Variable} == Variable.([1 3; 2 4])
    F = sparse(fill([1.0 2.0], 2, 2))
    @test Array(F)::Matrix{Matrix{Float64}} == fill([1.0 2.0], 2, 2)
    @test (F + fill([1.0 1.0], 2, 2))::Matrix{Matrix{Float64}} == fill([2.0 3.0], 2, 2)
    end
end

@static if COMPREHENSIVE
@testset "oneunit of sparse matrix" begin
    A = sparse([Meters(0) Meters(0); Meters(0) Meters(0)])
    @test oneunit(fixture(Float64, 2, 2)) isa SparseMatrixCSC{Float64}
    @test oneunit(A) isa SparseMatrixCSC{Meters}
    @test oneunit(A) == [Meters(1) Meters(0); Meters(0) Meters(1)]
    @test one(fixture(Float64, 2, 2)) isa SparseMatrixCSC{Float64}
    @test one(A) isa SparseMatrixCSC{Int}
end
end

@testset "transpose! does not allocate" begin
    function f()
        A = fixture(Float64, 10, 10)
        X = copy(A)
        return @allocated transpose!(X, A)
    end
    #precompile
    f()
    f()
    @test f() == 0
end

@testset "sparse transpose adjoint" begin
    A = fixture(Float64, 10, 10)
    @test A' == SparseMatrixCSC(A')
    @test SparseMatrixCSC(A') isa SparseMatrixCSC
    @test transpose(A) == SparseMatrixCSC(transpose(A))
    @test SparseMatrixCSC(transpose(A)) isa SparseMatrixCSC
    @test SparseMatrixCSC{eltype(A)}(transpose(A)) == transpose(A)
    @test SparseMatrixCSC{eltype(A), Int}(transpose(A)) == transpose(A)
    @static if COMPREHENSIVE
    @test SparseMatrixCSC{Float16}(transpose(A)) == transpose(SparseMatrixCSC{Float16}(A))
    end
    B = fixture(ComplexF64, 4, 4)
    @test mismatch(SparseMatrixCSC{eltype(B)}(adjoint(B)), Matrix(B)') === nothing
    @test mismatch(SparseMatrixCSC{eltype(B), Int}(adjoint(B)), Matrix(B)'; Ti=Int) === nothing
    # complex values that differ from their conjugates tell transpose from adjoint
    @test mismatch(copy(transpose(B)), transpose(Matrix(B))) === nothing
    @static if COMPREHENSIVE
    @test SparseMatrixCSC{ComplexF16, Int8}(adjoint(B)) == adjoint(SparseMatrixCSC{ComplexF16, Int8}(B))
    end
    @test SparseMatrixCSC{ComplexF64}(view(A, :, 1:2)) == A[:, 1:2]
    @test permutedims(A, (2, 1)) == transpose(A)
    @test permutedims(A, (1, 2)) == A
    @test permutedims(A, (1, 2)) !== A
    @test_throws ArgumentError permutedims(A, (1, 3))
    # nested wrappers convert from the inside out
    @test Matrix(sparse(Hermitian(B'))) == Matrix(Hermitian(Matrix(B')))
    # `Adjoint(...)`, since `adjoint` of a triangular matrix rewraps its parent
    @test Matrix(sparse(Adjoint(UnitUpperTriangular(A)))) == Matrix(UnitUpperTriangular(Matrix(A))')
end

@testset "SparseMatrixCSC [c]transpose[!] and permute[!]" begin
    smalldim = 5
    largedim = 10
    Random.seed!(20240817)   # for the permutations
    (m, n) = (smalldim, smalldim)
    A = fixture(Float64, m, n)
    X = similar(A)
    C = copy(transpose(A))
    p = randperm(m)
    q = randperm(n)
    @testset "common error checking of [c]transpose! methods (ftranspose!)" begin
        @test_throws DimensionMismatch transpose!(A[:, 1:(smalldim - 1)], A)
        @test_throws DimensionMismatch transpose!(A[1:(smalldim - 1), 1], A)
        # the column count matches, and there are too many rows
        @test_throws DimensionMismatch transpose!(spzeros(n + 1, m), A)
        @test_throws ArgumentError transpose!(A, A) # #812
        @test_throws ArgumentError adjoint!(A, A)
    end
    @testset "common error checking of permute[!] methods / source-perm compat" begin
        @test_throws DimensionMismatch permute(A, p[1:(end - 1)], q)
        @test_throws DimensionMismatch permute(A, p, q[1:(end - 1)])
    end
    @testset "common error checking of permute[!] methods / source-dest compat" begin
        @test_throws DimensionMismatch permute!(A[1:(m - 1), :], A, p, q)
        @test_throws DimensionMismatch permute!(A[:, 1:(m - 1)], A, p, q)
        @test_throws ArgumentError permute!((Y = copy(X); resize!(rowvals(Y), nnz(A) - 1); Y), A, p, q)
        @test_throws ArgumentError permute!((Y = copy(X); resize!(nonzeros(Y), nnz(A) - 1); Y), A, p, q)
    end
    @testset "common error checking of permute[!] methods / source-workmat compat" begin
        @test_throws DimensionMismatch permute!(X, A, p, q, C[1:(m - 1), :])
        @test_throws DimensionMismatch permute!(X, A, p, q, C[:, 1:(m - 1)])
        @test_throws ArgumentError permute!(X, A, p, q, (D = copy(C); resize!(rowvals(D), nnz(A) - 1); D))
        @test_throws ArgumentError permute!(X, A, p, q, (D = copy(C); resize!(nonzeros(D), nnz(A) - 1); D))
    end
    @testset "common error checking of permute[!] methods / source-workcolptr compat" begin
        @test_throws DimensionMismatch permute!(A, p, q, C, Vector{eltype(rowvals(A))}(undef, length(getcolptr(A)) - 1))
    end
    @testset "common error checking of permute[!] methods / permutation validity" begin
        @test_throws ArgumentError permute!(A, (r = copy(p); r[2] = r[1]; r), q)
        @test_throws ArgumentError permute!(A, (r = copy(p); r[2] = m + 1; r), q)
        @test_throws ArgumentError permute!(A, p, (r = copy(q); r[2] = r[1]; r))
        @test_throws ArgumentError permute!(A, p, (r = copy(q); r[2] = n + 1; r))
        @static if COMPREHENSIVE
        # an invalid permutation leaves the destination as it was, whether or not `q` is
        # longer than the column pointers of the workspace
        for B in (A, A[1:2, :])
            Y = copy(B); Y0 = copy(Y); pB = randperm(size(B, 1)); D = similar(copy(transpose(B)))
            for (pbad, qbad) in ((pB, (r = copy(q); r[2] = r[1]; r)), ((r = copy(pB); r[2] = r[1]; r), q))
                @test_throws ArgumentError permute!(Y, B, pbad, qbad)
                @test exact_equal(Y, Y0)
                @test_throws ArgumentError permute!(Y, B, pbad, qbad, D)
                @test exact_equal(Y, Y0)
            end
            @test mismatch(permute!(Y, B, pB, q, D), Array(B)[pB, q]) === nothing
        end
        end
    end
    @static if COMPREHENSIVE
    @testset "common error checking of permute[!] methods / aliasing" begin
        B = copy(A); B0 = copy(B)
        @test_throws ArgumentError permute!(B, B, p, q)
        @test_throws ArgumentError permute!(B, B, p, q, C)
        @test_throws ArgumentError permute!(X, B, p, q, X)
        @test_throws ArgumentError permute!(X, B, p, q, B)
        @test_throws ArgumentError permute!(B, p, q, B)
        @test_throws ArgumentError permute!(B, p, q, B, similar(getcolptr(B)))
        @test_throws ArgumentError permute!(B, p, q, C, getcolptr(B))
        # sharing one buffer is enough
        @test_throws ArgumentError permute!(SparseMatrixCSC(m, n, copy(getcolptr(B)), copy(rowvals(B)), nonzeros(B)), B, p, q)
        @test exact_equal(B, B0)
        # matrices without stored entries have empty buffers, which do not alias
        E = spzeros(m, n)
        @test permute!(spzeros(m, n), E, p, q) == E
        @test permute!(E, p, q, spzeros(n, m)) == spzeros(m, n)
    end
    end
    @testset "overall functionality of [c]transpose[!] and permute[!]" begin
        for (m, n) in ((@static COMPREHENSIVE ? ((smalldim, smalldim),) : ())..., (smalldim, largedim), (@static COMPREHENSIVE ? ((largedim, smalldim),) : ())...)
            A = fixture(Float64, m, n)
            At = copy(transpose(A))
            # transpose[!]
            fullAt = Array(transpose(A))
            @test mismatch(copy(transpose(A)), fullAt) === nothing
            @test mismatch(transpose!(similar(At), A), fullAt) === nothing
            # adjoint[!]
            C = A + im*A/2
            fullCh = Array(C')
            @test mismatch(copy(C'), fullCh) === nothing
            @test mismatch(adjoint!(similar(sparse(fullCh)), C), fullCh) === nothing
            # permute[!]
            p = randperm(m)
            q = randperm(n)
            fullPAQ = Array(A)[p,q]
            @test mismatch(permute(A, p, q), Array(A[p,q])) === nothing
            @test mismatch(permute!(similar(A), A, p, q), fullPAQ) === nothing
            @test mismatch(permute!(similar(A), A, p, q, similar(At)), fullPAQ) === nothing
            @test mismatch(permute!(copy(A), p, q), fullPAQ) === nothing
            @test mismatch(permute!(copy(A), p, q, similar(At)), fullPAQ) === nothing
            @test mismatch(permute!(copy(A), p, q, similar(At), similar(getcolptr(A))), fullPAQ) === nothing
        end
    end
end

@static if COMPREHENSIVE
@testset "transpose of SubArrays" begin
    A = view(fixture(Float64, 10, 10), 1:4, 1:4)
    @test copy(transpose(Array(A))) == Array(transpose(A))
    @test copy(adjoint(Array(A))) == Array(adjoint(A))
end
end

# Deterministic replacement for wall-clock guards: with a counting eltype, a comparison
# that walks only stored entries performs at most nnz(A) + nnz(B) element comparisons,
# whereas the generic AbstractArray fallback performs length(A) of them.
@testset "== and isequal walk stored entries only (issues #561, #766, #768)" begin
    n = 1000
    v = sparsevec([1, n ÷ 2], OpCount.([1.0, 2.0]), n)
    w = sparsevec([1, n ÷ 2, n], OpCount.([1.0, 0.0, 3.0]), n)
    A = sparse([1, n ÷ 2], [1, n], OpCount.([1.0, 2.0]), n, n)
    B = sparse([1, n ÷ 2, 7], [1, n, 7], OpCount.([1.0, 2.0, 0.0]), n, n)
    for (x, y) in ((v, w), (A, B), (A', B'), (A, transpose(B)),
                   (@static COMPREHENSIVE ? ((v, v), (w, v), (v', w'), (view(v, 1:n), w),
                   (A, A), (B, A), (transpose(A), B), (A', transpose(B)),
                   (A, view(B, :, [1:n;]))) : ())...)
        budget = nnz(parent(x isa Union{Adjoint,Transpose} ? x : x') ) +
                 nnz(parent(y isa Union{Adjoint,Transpose} ? y : y'))
        for eq in (==, isequal)
            @test eqcount(() -> eq(x, y)) <= budget
        end
    end
end

# A sparse array is compared with a dense one without indexing the sparse one, which an
# eltype without a `zero` cannot do at an unstored position: the generic fallback throws.
@testset "== and isequal of a sparse and a dense array" begin
    Dm = [[1;;] [0;;]; [0;;] [2;;]]
    Bm = sparse(Dm)
    A = sparse([1, 3, 2, 3], [1, 1, 3, 4], [1.0, NaN, -0.0, 2.0], 3, 4)
    x = sparsevec([2, 5], [NaN, 3.0], 6)
    C = sparse([1, 2], [2, 3], [1.0im, 2.0], 3, 3)
    for eq in (==, isequal)
        @test eq(Bm, Dm) && eq(Dm, Bm) && !eq(Bm, [[1;;] [0;;]; [0;;] [3;;]])
        # as the dense comparison, for NaN, signed zeros and conjugation: the wrappers
        # conjugate the sparse entries and their implicit zero, never the dense ones
        for (L, R) in ((A, Matrix(A)), (C', Matrix(C')), (x, Vector(x)),
                       (@static COMPREHENSIVE ? ((spzeros(3, 4), -zeros(3, 4)),
                       (view(A, :, 2:3), Matrix(A)[:, 2:3]), (x', Matrix(x')),
                       (sparse([true false; false true]), BitMatrix([true false; false true]))) : ())...)
            @test eq(L, R) === eq(Array(L), R)
            @test eq(R, L) === eq(R, Array(L))
        end
    end
    @static if COMPREHENSIVE
        @test sparsevec(Dm[:, 1]) == Dm[:, 1] && sparse(Any[1 0; 0 2]) == [1 0; 0 2]
        @test spzeros(Matrix{Int}, 2, 2) != zeros(Int, 2, 2)
    end
end

@static if COMPREHENSIVE
@testset "== and isequal compare the implicit zeros of the two eltypes (issue #234)" begin
    Zi, Zm, Zv = spzeros(Int, 2, 2), spzeros(Matrix{Int}, 2, 2), spzeros(Variable, 2, 2)
    Si = sparse([1, 2, 1, 2], [1, 1, 2, 2], [0, 0, 0, 0])   # stored zeros in every position
    S = sparse([1, 1, 2], [1, 2, 2], Variable.(1:3))
    T = sparse([1, 2], [1, 2], Variable.([1, 3]))
    G = sparse([1, 2, 1, 2], [1, 1, 2, 2], Tagged.(1:4, :m))
    # a number is not an array, the zero of a `Variable` is not the number zero, and a
    # stored entry without a counterpart meets the zero of the other eltype
    for (L, R) in ((Zi, Zm), (Si, Zm), (Zi, Zv), (Zi', Zm), (spzeros(Int, 2), spzeros(Matrix{Int}, 2)),
                   (S, T), (view(S, :, 1), T[:, 2]))
        @test L != R && R != L
    end
    @test !isequal(Zi, Zm)
    # equal zeros, no `zero` to compare, or no position to compare
    @test S == copy(S) && Zi == spzeros(2, 2) && Zm == spzeros(Matrix{Float64}, 2, 2)
    @test sparse(Any[1 0; 0 2]) == sparse([1, 2, 2], [1, 1, 2], [1.0, 0.0, 2.0])
    @test spzeros(Int, 0, 2) == spzeros(Matrix{Int}, 0, 2) && spzeros(Union{}, 0, 0) == spzeros(0, 0)
    # a number type without a zero of the type
    @test G == copy(G) && G == Tagged.([1 3; 2 4], :m) && G != sparse([1], [1], [Tagged(1, :m)], 2, 2)
    # the arguments keep their order
    Zo = spzeros(OneSided, 2, 2)
    for R in (spzeros(2, 2), sparse([1], [1], [0.0], 2, 2), transpose(spzeros(2, 2)), zeros(2, 2))
        @test Zo == R && R != Zo
    end
    # `missing` propagates, and a later difference still decides
    M = sparse([1, 2], [1, 2], [missing, 1.0])
    N = sparse([1, 2], [1, 2], [missing, 2.0])
    for R in (Matrix(M), copy(M), transpose(M))
        @test (M == R) === missing && (R == M) === missing && isequal(M, R)
    end
    @test (view(M, :, 1) == M[:, 1]) === missing
    @test (M == N) === false && (M == Matrix(N)) === false
end
end

# `Base.hash` on a large array skips runs of equal values with `findprev(!isequal(elt), A, i)`.
# Each such call on a sparse array costs at most nnz(A) + 1 element comparisons and `hash`
# makes only a handful of them, whereas the generic `findprev` performs up to length(A).
@testset "hash walks stored entries only (issue #570)" begin
    n = 10^5
    v = sparsevec([1, n ÷ 2], OpCount.([1.0, 2.0]), n)
    w = sparsevec([1, n ÷ 2, n], OpCount.([1.0, 0.0, 3.0]), n)
    A = sparse([1, n ÷ 2], [1, n], OpCount.([1.0, 2.0]), n, n)
    B = sparse([1, n ÷ 2, 7], [1, n, 7], OpCount.([1.0, 2.0, 0.0]), n, n)
    for x in (v, w, A, B)
        @test eqcount(() -> hash(x)) <= 8 * (nnz(x) + 1)
    end
    @test hash(v) == hash(Vector(v)) && hash(w) == hash(Vector(w))
end

@static if COMPREHENSIVE
@testset "Issue #246" begin
    for t in [Float64]
        a = OpCount.(fixturevec(t, 100))
        b = OpCount.(2 * fixturevec(t, 100))

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

        for m in [identity, transpose]
            ma, mb, mc, md = m.([a, b, c, d])

            @test eqcount(() -> ma == mb) <= nnz(a) + nnz(b)

            @test (mc == md) == (Array(mc) == Array(md))
        end
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
                    (@static COMPREHENSIVE ? (
                    Bidiagonal(1:4, 1:3, :U), Bidiagonal(1:4, 1:3, :L),
                    Diagonal(1:4),
                    ) : ())...,
                    diagm(-2=>1:2, 2=>1:2)]
        S = sparse(T)
        for k in -5:5
            @test istriu(S, k) == istriu(T, k)
            @test istril(S, k) == istril(T, k)
        end
    end

    # a view of a column range walks the stored entries of those columns, ignoring
    # stored zeros, instead of the generic O(mn) element scan
    S = sparse([1, 2, 4, 2, 4, 1], [1, 2, 3, 4, 5, 6], [1.0, 2.0, 0.0, 3.0, 4.0, 5.0], 4, 6)
    V = view(S, :, 2:5)   # V[4, 2] is the stored zero
    @test istriu(V, -1) == istriu(Matrix(V), -1) == true
    @test istril(V) == istril(Matrix(V)) == false
end

@testset "isdiag" begin
    # Diagonal matrices should return true
    @test isdiag(sparse(Diagonal(1:4)))
    @test isdiag(sparse(Diagonal([1.0, 2.0, 3.0])))
    @test isdiag(spzeros(5, 5))  # Empty matrix is diagonal

    # Non-diagonal matrices should return false
    @test !isdiag(sparse(Tridiagonal(1:3, 1:4, 1:3)))
    @static if COMPREHENSIVE
    @test !isdiag(sparse(Bidiagonal(1:4, 1:3, :U)))
    @test !isdiag(sparse(Bidiagonal(1:4, 1:3, :L)))
    end
    @test !isdiag(sparse([1 2; 3 4]))

    # Non-square diagonal matrices should return true (consistent with generic isdiag)
    @test isdiag(sparse([1 0 0; 0 2 0]))  # 2x3 diagonal
    @test isdiag(sparse([1 0; 0 2; 0 0]))  # 3x2 diagonal
    @test isdiag(spzeros(3, 5))  # Empty non-square matrix is diagonal
    @test isdiag(spzeros(5, 3))

    # Non-square non-diagonal matrices should return false
    @test !isdiag(sparse([1 1 0; 0 2 0]))  # Off-diagonal element
    @test !isdiag(sparse([1 0; 0 2; 1 0]))  # Off-diagonal element

    @static if COMPREHENSIVE
    # Consistency with dense isdiag
    for T in Any[Diagonal(1:4), Tridiagonal(1:3, 1:4, 1:3),
                 Bidiagonal(1:4, 1:3, :U), diagm(-1=>1:3, 1=>1:3)]
        S = sparse(T)
        @test isdiag(S) == isdiag(T)
    end
    end

    # Explicit zeros on off-diagonal should still be diagonal
    S = sparse([1, 2, 1], [1, 2, 2], [1.0, 2.0, 0.0])
    @test isdiag(S)

    # views of a column range walk their stored entries
    S = sparse([1, 1, 2, 1, 3], [1, 2, 3, 4, 4], [1.0, 0.0, 2.0, 0.0, 3.0], 3, 4)
    V = view(S, :, 2:4)   # V[1, 3] is the stored zero
    @test isdiag(V) == isdiag(Matrix(V)) == true
end

@testset "sort/sort! of a sparse matrix" begin
    # `sort` of a dense matrix with `size(M, dims) == 0` errors in Base, so those cases are
    # compared against the input itself rather than against a dense reference
    # `dims = 2` covers the transposed shapes, so only one orientation of each is listed;
    # fully structural matrices are covered by the "empty and zero-size matrices" testset
    # `rev` and `alg` are only forwarded to the sort of each column, so they run in
    # comprehensive mode alone, once for each shape, density and dimension
    extra = @static COMPREHENSIVE ? eachvalue(((6, 5), (1, 1), (0, 3), (1, 9), (20, 13)),
                                              (0.3, 1.0), (1, 2),
                                              ((; rev=true), (; alg=Base.DEFAULT_STABLE))) : ()
    Random.seed!(20240818)   # sorting needs values in no particular order
    @testset "size = ($m, $n), density = $d" for (m, n) in ((6, 5), (@static COMPREHENSIVE ? ((1, 1), (0, 3), (1, 9),
                                                            (20, 13)) : ())...),
                                                 d in (0.3, 1.0)
        A = sprand(m, n, d)
        M = Matrix(A)
        for dims in (1, 2), kws in ((;), (; rev=true), (; by=abs), (; alg=Base.DEFAULT_STABLE))
            kws in ((;), (; by=abs)) || ((m, n), d, dims, kws) in extra || continue
            expected = size(M, dims) == 0 ? M : sort(M; dims, kws...)
            B = copy(A)
            @test sort!(B; dims, kws...) === B
            @test B isa SparseMatrixCSC
            @test mismatch(B, expected) === nothing
            # sorting only moves the stored entries around
            @test nnz(B) == nnz(A)
            S = sort(A; dims, kws...)
            @test S isa SparseMatrixCSC
            @test mismatch(S, expected) === nothing
            @test A == sparse(M) # `sort` leaves its argument alone
        end
    end

    @static if COMPREHENSIVE
    @testset "index type $Ti" for Ti in (Int32,)
        A = SparseMatrixCSC{Float64,Ti}(sprand(11, 7, 0.4))
        for dims in (1, 2)
            @test sort(A; dims) isa SparseMatrixCSC{Float64,Ti}
            @test mismatch(sort(A; dims), sort(Matrix(A); dims); Ti) === nothing
        end
    end
    end

    @testset "keyword arguments" begin
        A = sprand(50, 50, 0.1)
        # `scratch` is forwarded to the underlying `sort!` and ignored by the search for
        # where the structural zeros belong (see #335)
        @test mismatch(sort!(copy(A); dims=1, scratch=Vector{Float64}(undef, 50)),
                       sort(Matrix(A); dims=1)) === nothing
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
        @static if COMPREHENSIVE
        # the ordering is only evaluated at zero when there are structural zeros to place,
        # so a `by` that is undefined at zero works on fully stored columns as it does for
        # dense matrices
        F = sparse([1 2; 3 4])
        by = x -> 1 ÷ x
        for dims in (1, 2)
            @test Matrix(sort(F; dims, by)) == sort(Matrix(F); dims, by)
        end
        @test_throws DivideError sort(sparse([1 0; 3 4]); dims=1, by)
        end
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
        @test mismatch(B, sort(Matrix(A); dims=1)) === nothing
    end

    @testset "column views" begin
        A = sprand(7, 4, 0.5)
        M = Matrix(A)
        for j in axes(A, 2), kws in ((;), (@static COMPREHENSIVE ? ((; rev=true),) : ())...)
            B = copy(A)
            c = view(B, :, j)
            @test sort!(c; kws...) === c
            @test nnz(B) == nnz(A)
            expected = copy(M)
            sort!(view(expected, :, j); kws...)
            @test mismatch(B, expected) === nothing
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
            @test mismatch(S, sort(Matrix(A); dims)) === nothing
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
        @testset "all structural zeros, size = ($m, $n)" for (m, n) in ((@static COMPREHENSIVE ? ((1, 1),) : ())..., (5, 4))
            A = spzeros(m, n)
            for dims in (1, 2)
                B = sort!(copy(A); dims)
                @test mismatch(B, sort(Matrix(A); dims)) === nothing
                @test nnz(B) == 0
                @test getcolptr(B) == getcolptr(A)
            end
        end

        # a single column/row that is entirely structural next to a populated one
        A = SparseMatrixCSC(4, 3, [1, 1, 5, 5], [1, 2, 3, 4], [1.0, -2.0, 0.0, 3.0])
        for dims in (1, 2)
            @test mismatch(sort(A; dims), sort(Matrix(A); dims)) === nothing
            @test nnz(sort(A; dims)) == nnz(A)
        end
    end

    @testset "stored zeros" begin
        # column 1 stores an explicit zero next to structural zeros
        A = SparseMatrixCSC(4, 2, [1, 3, 4], [1, 3, 2], [0.0, -1.0, 2.0])
        for dims in (1, 2)
            @test mismatch(sort(A; dims), sort(Matrix(A); dims)) === nothing
            @test nnz(sort(A; dims)) == nnz(A)
        end
    end
end

@testset "repeat tests" begin
    A = fixture(Float64, 5, 3)
    A_full = Matrix(A)
    for m = 0:3
        @test issparse(repeat(A, m))
        @test mismatch(repeat(A, m), repeat(A_full, m)) === nothing
        for n = 0:3
            @test issparse(repeat(A, m, n))
            @test mismatch(repeat(A, m, n), repeat(A_full, m, n)) === nothing
        end
    end
    @static if COMPREHENSIVE
    # a non-Int index type is kept, including in the column pointers
    A32 = SparseMatrixCSC{ComplexF64,Int32}(fixture(ComplexF64, 5, 3))
    A32_full = Matrix(A32)
    for m = 0:2, n = 0:3
        R = repeat(A32, m, n)
        @test R isa SparseMatrixCSC{ComplexF64,Int32}
        @test mismatch(R, repeat(A32_full, m, n); Ti=Int32) === nothing
        @test repeat(A32, m) isa SparseMatrixCSC{ComplexF64,Int32}
    end
    end
end

@testset "copyto!" begin
    A = fixture(Float64, 5, 5)
    B = 2 * permutedims(fixture(Float64, 5, 5))
    Ar = copyto!(A, B)
    @test Ar === A
    @test A == B
    @test pointer(nonzeros(A)) != pointer(nonzeros(B))
    @test pointer(rowvals(A)) != pointer(rowvals(B))
    @test pointer(getcolptr(A)) != pointer(getcolptr(B))
    # Test size(A) != size(B), but length(A) == length(B)
    B = fixture(Float64, 25, 1)
    copyto!(A, B)
    @test A[:] == B[:]
    # Test various size(A) / size(B) combinations
    sizes = @static COMPREHENSIVE ? [5, 10, 20] : [5, 20]
    for mA in sizes, nA in sizes, mB in sizes, nB in sizes
        A = fixture(Float64, mA, nA)
        Aorig = copy(A)
        B = 2 * fixture(Float64, mB, nB)
        if mA*nA >= mB*nB
            copyto!(A,B)
            @assert(A[1:length(B)] == B[:])
            @assert(A[length(B)+1:end] == Aorig[length(B)+1:end])
        else
            @test_throws BoundsError copyto!(A,B)
        end
    end
    # Test eltype(A) != eltype(B), size(A) != size(B)
    A = fixture(Float64, 5, 5)
    Aorig = copy(A)
    B = sparse(Float32[1 0 7; 2 5 0; 0 6 9])
    copyto!(A, B)
    @test A[1:9] == B[:]
    @test A[10:end] == Aorig[10:end]
    @static if COMPREHENSIVE
    # Test eltype(A) != eltype(B), size(A) == size(B)
    A = fixture(Float64, 3, 3)
    B = sparse(Float32[1 0 7; 2 5 0; 0 6 9])
    copyto!(A, B)
    @test A == B
    end
    # an empty source leaves the destination untouched, as for dense
    A = sparse([3, 4, 2, 1], [1, 1, 2, 4], [1.0, 2.0, 3.0, 4.0], 4, 4)
    Aorig = copy(A)
    for B in (spzeros(0, 0), (@static COMPREHENSIVE ? (spzeros(0, 3), spzeros(3, 0)) : ())...)
        @test copyto!(A, B) === A
        @test A == Aorig
    end
    @static if COMPREHENSIVE
    # indtype(A) != indtype(B), for every size relation
    A = SparseMatrixCSC{Float64,Int32}(fixture(Float64, 5, 5))
    Aorig = copy(A)
    for B in (2 * permutedims(fixture(Float64, 5, 5)), fixture(Float64, 25, 1), 2 * fixture(Float64, 3, 3))
        copyto!(A, B)
        @test A isa SparseMatrixCSC{Float64,Int32}
        @test A[1:length(B)] == B[:]
        @test A[length(B)+1:end] == Aorig[length(B)+1:end]
        copyto!(A, Aorig)
    end
    end
    # Test copyto!(dense, sparse)
    B = permutedims(fixture(Float64, 5, 5))   # has stored entries in `Rsrc` below
    A = reshape(Float64.(1:25), 5, 5)
    A´ = similar(A)
    Ac = copyto!(A, B)
    @test Ac === A
    @test A == copyto!(A´, Matrix(B))
    # Test copyto!(dense, Rdest, sparse, Rsrc)
    A = reshape(Float64.(1:25), 5, 5)
    A´ = similar(A)
    Rsrc = CartesianIndices((3:4, 2:3))
    Rdest = CartesianIndices((2:3, 1:2))
    copyto!(A, Rdest, B, Rsrc)
    copyto!(A´, Rdest, Matrix(B), Rsrc)
    @test A[Rdest] == A´[Rdest] == Matrix(B)[Rsrc]
    # Test unaliasing of B´
    B´ = copy(B)
    copyto!(B´, Rdest, B´, Rsrc)
    @test Matrix(B´)[Rdest] == Matrix(B)[Rsrc]
    # Test that only elements at overlapping linear indices are overwritten
    A = sparse([2.0 5.0 8.0; 3.0 6.0 9.0; 4.0 7.0 10.0]); B = ones(4, 4)
    Bc = copyto!(B, A)
    @test B[4, :] != B[:, 4] == ones(4)
    @test Bc === B
    @static if COMPREHENSIVE
    # Allow no-op copyto! with empty source even for incompatible eltypes
    A = sparse(fill("", 0, 0))
    @test copyto!(B, A) == B
    end

    @static if COMPREHENSIVE
    # a value that does not convert, or more entries than the index type holds, is
    # found before the pattern of the destination is rewritten
    A = sparse([1, 2], [1, 2], [1, 2]); A0 = copy(A)
    @test_throws InexactError copyto!(A, sparse([1, 2, 2], [1, 1, 2], [1.0, 2.5, 3.0]))
    @test same_pattern(A, A0) && A == A0
    A = sparse([1, 2], [1, 2], [1, 2], 2, 3)   # the source covers part of the destination
    A0 = copy(A)
    @test_throws InexactError copyto!(A, sparse([1.0 2.5; 0.0 3.0]))
    @test same_pattern(A, A0) && A == A0
    A = spzeros(Float64, Int8, 12, 12)
    @test_throws ArgumentError copyto!(A, sparse(ones(12, 12)))
    @test nnz(A) == 0 && mismatch(A, zeros(12, 12); Ti=Int8) === nothing
    @test mismatch(copyto!(A, sparse(ones(12, 10))), [ones(12, 10) zeros(12, 2)]; Ti=Int8) === nothing
    end

    # Test correct error for too small destination array
    @test_throws BoundsError copyto!(zeros(2,2), fixture(Float64, 3, 3))
end

@testset "copyto! into dense arrays of any shape" begin
    S = sparse([1, 3, 1], [1, 2, 3], [1.0, 3.0, 4.0], 3, 3)
    D = Matrix(S)
    @test which(copyto!, Tuple{Matrix{Float64}, typeof(S)}).module === SparseArrays
    for dest in (fill(-1.0, 3, 3), fill(-1.0, 9), fill(-1.0, 3, 3, 1))
        @test copyto!(dest, S) === dest && vec(dest) == vec(D)
    end
    @test copyto!(fill(-1.0, 3, 2), view(S, :, 2:3)) == D[:, 2:3]
    @test copyto!(fill(-1.0, 3, 3), S') == D'
    @static if COMPREHENSIVE
    # a full pattern never asks for a zero, so an eltype without one copies (issue #28369)
    M = reshape([fill(k, 1, 2) for k in 1:4], 2, 2)
    @test Array(sparse(M)) == M
    end
    # a destination that is one of the source's own buffers still receives the source's values
    for wrap in (identity, adjoint, transpose, A -> view(A, :, 1:2))
        A = sparse([1.0 2.0; 3.0 4.0]); B = copy(A)
        @test copyto!(nonzeros(A), wrap(A)) == vec(Matrix(wrap(B)))
        @test rowvals(A) == rowvals(B) && getcolptr(A) == getcolptr(B)
        A = sparse([1.0 2.0; 3.0 4.0])
        @test copyto!(rowvals(A), wrap(A)) == vec(Matrix(wrap(B)))
    end
    for wrap in (identity, (@static COMPREHENSIVE ? (adjoint, x -> view(x, 1:3)) : ())...)
        x = sparsevec([1.0, 2.0, 3.0]); y = copy(x)
        @test vec(copyto!(nonzeros(x), wrap(x))) == vec(collect(wrap(y)))
        @test nonzeroinds(x) == nonzeroinds(y)
    end
    @static if COMPREHENSIVE
    Sc = sparse([1 2; 3 4])
    @test copyto!(nonzeros(Sc), view(Sc, :, 2)) == [2, 4, 2, 4]
    # `ReadOnly` reports no data ids; the fixed pattern still shares `Sf`'s buffers. The
    # destination is the row-index buffer itself, so only the result and `colptr` are checked.
    Sf = sparse([1 2; 3 4]); Ff = fixed(Sf)
    @test copyto!(rowvals(Sf), adjoint(Ff)) == [1, 2, 3, 4] && getcolptr(Ff) == [1, 3, 5]
    end
    # a sparse vector type that only implements the `nonzeroinds`/`nonzeros` interface
    w = WrappedSparseVector(sparsevec([2], [5.0], 4))
    @test copyto!(fill(-1.0, 4), w) == [0, 5, 0, 0]
    small = fill(-1.0, 2, 2)
    @test_throws BoundsError copyto!(small, S)
    @test all(small .== -1)
end

@testset "error conditions for reshape, and dropdims" begin
    local A = sparse([1, 4, 4], [2, 2, 5], [true, false, true], 5, 5)
    @test_throws DimensionMismatch reshape(A,(20, 2))
    @test_throws ArgumentError dropdims(A,dims=(1, 1))
end

@testset "droptol" begin
    # the entries are (i + 3j)/2000, so one of them equals the tolerance and is dropped
    A = triu(fixture(Float64, 10, 10)) / 2000
    kept = map(x -> abs(x) > 0.01 ? x : 0.0, Matrix(A))
    @test getcolptr(SparseArrays.droptol!(A, 0.01)) == getcolptr(sparse(kept))
    @test mismatch(A, kept) === nothing
    @test 0 < nnz(A) < nnz(triu(fixture(Float64, 10, 10)))
    @test isequal(SparseArrays.droptol!(sparse([1], [1], [1]), 1), SparseMatrixCSC(1, 1, Int[1, 1], Int[], Int[]))
end

@static if COMPREHENSIVE
@testset "dropzeros[!]" begin
    smalldim = 5
    largedim = 10
    targetnumposzeros = 5
    targetnumnegzeros = 5
    for (m, n) in ((@static COMPREHENSIVE ? ((largedim, largedim),) : ())..., (smalldim, largedim), (@static COMPREHENSIVE ? ((largedim, smalldim),) : ())...)
        local A = fixture(Float64, m, n)
        A[1, 1] = 1   # no stored zero in the reference
        struczerosA = findall(x -> x == 0, A)
        # the two sets share positions 1 and 13
        poszerosinds = struczerosA[range(1; step=3, length=targetnumposzeros)]
        negzerosinds = struczerosA[range(1; step=4, length=targetnumnegzeros)]
        Aposzeros = copy(A)
        Aposzeros[poszerosinds] .= 2
        Anegzeros = copy(A)
        Anegzeros[negzerosinds] .= -2
        Abothsigns = copy(Aposzeros)
        Abothsigns[negzerosinds] .= -2
        map!(x -> x == 2 ? 0.0 : x, nonzeros(Aposzeros), nonzeros(Aposzeros))
        map!(x -> x == -2 ? -0.0 : x, nonzeros(Anegzeros), nonzeros(Anegzeros))
        map!(x -> x == 2 ? 0.0 : x == -2 ? -0.0 : x, nonzeros(Abothsigns), nonzeros(Abothsigns))
        for Awithzeros in ((@static COMPREHENSIVE ? (Aposzeros, Anegzeros) : ())..., Abothsigns)
            # Basic functionality / dropzeros!
            @test dropzeros!(copy(Awithzeros)) == A
            # Basic functionality / dropzeros
            @test dropzeros(Awithzeros) == A
            # Check trimming works as expected
            @test length(nonzeros(dropzeros!(copy(Awithzeros)))) == length(nonzeros(A))
            @test length(rowvals(dropzeros!(copy(Awithzeros)))) == length(rowvals(A))
        end
    end
    # original lone dropzeros test
    local A = sparse([1 2 3; 4 5 6; 7 8 9])
    nonzeros(A)[2] = nonzeros(A)[6] = nonzeros(A)[7] = 0
    @test getcolptr(dropzeros!(A)) == [1, 3, 5, 7]
    @static if COMPREHENSIVE
    # test for issue #5169, modified for new behavior following #15242/#14798
    @test nnz(sparse([1, 1], [1, 2], [0.0, -0.0])) == 2
    @test nnz(dropzeros!(sparse([1, 1], [1, 2], [0.0, -0.0]))) == 0
    # test for issue #5437, modified for new behavior following #15242/#14798
    @test nnz(sparse([1, 2, 3], [1, 2, 3], [0.0, 1.0, 2.0])) == 3
    @test nnz(dropzeros!(sparse([1, 2, 3],[1, 2, 3],[0.0, 1.0, 2.0]))) == 2
    end
end
end

@static if COMPREHENSIVE
@testset "fkeep! with a predicate that throws" begin
    D = collect(reshape(1.0:36.0, 6, 6))
    A = sparse(D)
    @test_throws ErrorException SparseArrays.fkeep!((i, j, x) -> (i, j) == (3, 4) ? error("stop") : isodd(i), A)
    # the entries visited before the throw are filtered and the others are kept
    for j in 1:6, i in 1:6
        (j < 4 || (j == 4 && i < 3)) && iseven(i) && (D[i, j] = 0)
    end
    @test mismatch(A, D) === nothing && nnz(A) == 26
    A = sparse(D)
    @test_throws ErrorException SparseArrays.fkeep!((i, j, x) -> error("stop"), A)
    @test mismatch(A, D) === nothing
end

@testset "tril! and triu! with a diagonal out of range" begin
    D = ones(3, 4)
    for k in (typemax(Int), typemax(Int) - 1, typemin(Int), typemin(Int) + 1, big(2)^70, -big(2)^70)
        kd = clamp(k, -5, 5)
        @test mismatch(tril!(sparse(D), k), tril(D, kd)) === nothing
        @test mismatch(triu!(sparse(D), k), triu(D, kd)) === nothing
    end
end
end

@testset "similar should not alias the input sparse array" begin
    a = sparse([1.0 4.0 7.0; 2.0 5.0 8.0; 3.0 6.0 9.0])
    Ti = @static COMPREHENSIVE ? Int32 : Int
    b = similar(a, Float32, Ti)
    c = similar(b, Float32, Ti)
    SparseArrays.dropstored!(b, 1, 1)
    @test length(rowvals(c)) == 9
    @test length(nonzeros(c)) == 9
end

@static if COMPREHENSIVE
@testset "similar with type conversion" begin
    local A = sparse(1.0I, 5, 5)
    @test size(similar(A, ComplexF64, Int)) == (5, 5)
    @test typeof(similar(A, ComplexF64, Int)) == SparseMatrixCSC{ComplexF64, Int}
    @test size(similar(A, ComplexF64, Int8)) == (5, 5)
    @test typeof(similar(A, ComplexF64, Int8)) == SparseMatrixCSC{ComplexF64, Int8}
    @test similar(A, ComplexF64,(6, 6)) == spzeros(ComplexF64, 6, 6)
    @test convert(Matrix, A) == Array(A) # lolwut, are you lost, test?
end
end

@testset "similar for SparseMatrixCSC" begin
    local A = sparse(1.0I, 5, 5)
    Ti = @static COMPREHENSIVE ? Int8 : Int
    # test similar without specifications (preserves stored-entry structure)
    simA = similar(A)
    @test typeof(simA) == typeof(A)
    @test size(simA) == size(A)
    @test getcolptr(simA) == getcolptr(A)
    @test rowvals(simA) == rowvals(A)
    @test length(nonzeros(simA)) == length(nonzeros(A))
    # test similar with entry type specification (preserves stored-entry structure)
    simA = similar(A, Float32)
    @test typeof(simA) == SparseMatrixCSC{Float32,eltype(getcolptr(A))}
    @test size(simA) == size(A)
    @test getcolptr(simA) == getcolptr(A)
    @test rowvals(simA) == rowvals(A)
    @test length(nonzeros(simA)) == length(nonzeros(A))
    # test similar with entry and index type specification (preserves stored-entry structure)
    simA = similar(A, Float32, Ti)
    @test typeof(simA) == SparseMatrixCSC{Float32,Ti}
    @test size(simA) == size(A)
    @test getcolptr(simA) == getcolptr(A)
    @test rowvals(simA) == rowvals(A)
    @test length(nonzeros(simA)) == length(nonzeros(A))
    # test similar with Dims{2} specification (preserves storage space only, not stored-entry structure)
    simA = similar(A, (6,6))
    @test typeof(simA) == typeof(A)
    @test size(simA) == (6,6)
    @test getcolptr(simA) == fill(1, 6+1)
    @test length(rowvals(simA)) == 0
    @test length(nonzeros(simA)) == 0
    # test similar with entry type and Dims{2} specification (empty storage space)
    simA = similar(A, Float32, (6,6))
    @test typeof(simA) == SparseMatrixCSC{Float32,eltype(getcolptr(A))}
    @test size(simA) == (6,6)
    @test getcolptr(simA) == fill(1, 6+1)
    @test length(rowvals(simA)) == 0
    @test length(nonzeros(simA)) == 0
    # test similar with entry type, index type, and Dims{2} specification (preserves storage space only)
    simA = similar(A, Float32, Ti, (6,6))
    @test typeof(simA) == SparseMatrixCSC{Float32, Ti}
    @test size(simA) == (6,6)
    @test getcolptr(simA) == fill(1, 6+1)
    @test length(rowvals(simA)) == 0
    @test length(nonzeros(simA)) == 0
    # test similar with Dims{1} specification (preserves nothing)
    simA = similar(A, (6,))
    @test typeof(simA) == SparseVector{eltype(nonzeros(A)),eltype(getcolptr(A))}
    @test size(simA) == (6,)
    @test length(nonzeroinds(simA)) == 0
    @test length(nonzeros(simA)) == 0
    @static if COMPREHENSIVE
    # test similar with entry type and Dims{1} specification (preserves nothing)
    simA = similar(A, Float32, (6,))
    @test typeof(simA) == SparseVector{Float32,eltype(getcolptr(A))}
    @test size(simA) == (6,)
    @test length(nonzeroinds(simA)) == 0
    @test length(nonzeros(simA)) == 0
    # test similar with entry type, index type, and Dims{1} specification (preserves nothing)
    simA = similar(A, Float32, Int8, (6,))
    @test typeof(simA) == SparseVector{Float32,Int8}
    @test size(simA) == (6,)
    @test length(nonzeroinds(simA)) == 0
    @test length(nonzeros(simA)) == 0
    end
    # test entry points to similar with entry type, index type, and non-Dims shape specification
    @test similar(A, Float32, Ti, 6, 6) == similar(A, Float32, Ti, (6, 6))
    @test similar(A, Float32, Ti, 6) == similar(A, Float32, Ti, (6,))
end

@testset "similar should preserve underlying storage type and uplo flag" begin
    m, n = 4, 3
    sparsemat = fixture(Float64, m, m)
    for SymType in (Symmetric, (@static COMPREHENSIVE ? (Hermitian,) : ())...)
        symsparsemat = SymType(sparsemat)
        @test isa(similar(symsparsemat), typeof(symsparsemat))
        @test similar(symsparsemat).uplo == symsparsemat.uplo
        @test isa(similar(symsparsemat, Float32), SymType{Float32,<:SparseMatrixCSC{Float32}})
        @test similar(symsparsemat, Float32).uplo == symsparsemat.uplo
        @test isa(similar(symsparsemat, (n, n)), typeof(sparsemat))
        @test isa(similar(symsparsemat, Float32, (n, n)), SparseMatrixCSC{Float32})
    end
end

@testset "similar should preserve underlying storage type" begin
    local m, n = 4, 3
    sparsemat = fixture(Float64, m, m)
    for TriType in (UpperTriangular, (@static COMPREHENSIVE ? (UnitLowerTriangular,) : ())...)
        trisparsemat = TriType(sparsemat)
        @test isa(similar(trisparsemat), typeof(trisparsemat))
        @test isa(similar(trisparsemat, Float32), TriType{Float32,<:SparseMatrixCSC{Float32}})
        @test isa(similar(trisparsemat, (n, n)), typeof(sparsemat))
        @test isa(similar(trisparsemat, Float32, (n, n)), SparseMatrixCSC{Float32})
    end
end

@testset "sparse findprev/findnext operations" begin

    x = [0,0,0,0,1,0,1,0,1,1,0]
    x_sp = sparse(x)

    for i=1:length(x)
        @test findnext(!iszero, x,i) == findnext(!iszero, x_sp,i)
        @test findprev(!iszero, x,i) == findprev(!iszero, x_sp,i)
    end

    y = [7 0 0 0 0;
         1 0 1 0 0;
         1 7 0 7 1;
         0 0 1 0 0;
         1 0 1 1 0.0]
    y_sp = [x == 7 ? -0.0 : x for x in sparse(y)]
    y = Array(y_sp)
    @test isequal(y_sp[1,1], -0.0)

    for i in keys(y)
        @test findnext(!iszero, y,i) == findnext(!iszero, y_sp,i)
        @test findprev(!iszero, y,i) == findprev(!iszero, y_sp,i)
        @test findnext(iszero, y,i) == findnext(iszero, y_sp,i)
        @test findprev(iszero, y,i) == findprev(iszero, y_sp,i)
    end

    z_sp = sparsevec(Dict(1=>1, 5=>1, 8=>0, 10=>1))
    z = collect(z_sp)

    for i in keys(z)
        @test findnext(!iszero, z,i) == findnext(!iszero, z_sp,i)
        @test findprev(!iszero, z,i) == findprev(!iszero, z_sp,i)
    end

    @static if COMPREHENSIVE
    # issue 32568
    for T = (UInt, BigInt)
        @test findnext(!iszero, x_sp, T(4)) isa keytype(x_sp)
        @test findnext(!iszero, x_sp, T(5)) isa keytype(x_sp)
        @test findprev(!iszero, x_sp, T(5)) isa keytype(x_sp)
        @test findprev(!iszero, x_sp, T(6)) isa keytype(x_sp)
        @test findnext(iseven, x_sp, T(4)) isa keytype(x_sp)
        @test findnext(iseven, x_sp, T(5)) isa keytype(x_sp)
        @test findprev(iseven, x_sp, T(4)) isa keytype(x_sp)
        @test findprev(iseven, x_sp, T(5)) isa keytype(x_sp)
        @test findnext(!iszero, z_sp, T(4)) isa keytype(z_sp)
        @test findnext(!iszero, z_sp, T(5)) isa keytype(z_sp)
        @test findprev(!iszero, z_sp, T(4)) isa keytype(z_sp)
        @test findprev(!iszero, z_sp, T(5)) isa keytype(z_sp)
    end
    end

    # The sparse methods must actually extend `Base.findnext`/`Base.findprev` and skip
    # implicit zeros for predicates other than `!iszero`, e.g. the `!isequal(elt)` that
    # `Base.hash` uses to skip runs of equal values.
    @test SparseArrays.findnext === Base.findnext && SparseArrays.findprev === Base.findprev
    n = 10^9
    big = spzeros(n); big[1] = 1; big[n ÷ 2] = -0.0
    @test findprev(!isequal(0.0), big, n) == n ÷ 2
    @test findprev(!isequal(-0.0), big, n ÷ 2) == n ÷ 2 - 1   # implicit 0.0 is not isequal(-0.0)
    @test findnext(!isequal(0.0), big, 2) == n ÷ 2
    @test findnext(!isequal(0.0), big, n ÷ 2 + 1) === nothing
    # the predicate is evaluated once on the implicit zero and then on stored entries only
    calls = Ref(0)
    counted = x -> (calls[] += 1; !isequal(x, 0.0))
    @test findprev(counted, big, n) == n ÷ 2 && calls[] <= nnz(big) + 1
    calls[] = 0
    @test findnext(counted, big, 2) == n ÷ 2 && calls[] <= nnz(big) + 1
    for i in keys(y), f in (!isequal(0.0), !isequal(-0.0), !isequal(7.0), !isequal(NaN))
        @test findnext(f, y, i) == findnext(f, y_sp, i)
        @test findprev(f, y, i) == findprev(f, y_sp, i)
    end

    # views of a column range, of a column and of a whole vector search the stored
    # entries of the viewed columns by bisection instead of scanning every element
    V = view(y_sp, :, 2:4); Y = y[:, 2:4]
    for i in keys(Y), f in (!iszero, !isequal(0.0))
        @test findnext(f, V, i) == findnext(f, Y, i)
        @test findprev(f, V, i) == findprev(f, Y, i)
    end
    for i in keys(z)
        @test findnext(!iszero, view(z_sp, :), i) == findnext(!iszero, z, i)
        @static if COMPREHENSIVE
        @test findnext(!iszero, view(z_sp, :), CartesianIndex(i)) == findnext(!iszero, z, CartesianIndex(i))
        @test findprev(!iszero, view(z_sp, :), CartesianIndex(i)) == findprev(!iszero, z, CartesianIndex(i))
        end
    end
    B = spzeros(100, 100); B[2, 3] = 1.0; B[100, 99] = -0.0
    VB = view(B, :, 2:100)
    calls[] = 0
    @test findnext(counted, VB, CartesianIndex(3, 2)) == CartesianIndex(100, 98) && calls[] <= nnz(VB) + 1
    calls[] = 0
    @test findprev(counted, view(B, :, 99), 100) == 100 && calls[] <= nnz(view(B, :, 99)) + 1
end

#testing the sparse matrix/vector access functions nnz, nzrange, rowvals, nonzeros
@testset "generic sparse matrix access functions" begin
    I = [1,3,4,5, 1,3,4,5, 1,3,4,5];
    J = [4,4,4,4, 5,5,5,5, 6,6,6,6];
    V = [14,34,44,54, 15,35,45,55, 16,36,46,56];
    A = sparse(I, J, V, 9, 9);
    AU = UpperTriangular(A)
    AL = LowerTriangular(A)
    b = SparseVector(9, I[1:4], V[1:4])
    c = view(A, :, 5)
    @static if COMPREHENSIVE
    d = view(b, :)
    @test nnz(d) == 4
    @test nzrange(d, 1) == 1:4
    @test rowvals(d) == I[1:4]
    @test nonzeros(d) == V[1:4]
    end

    @test (nnz(A), nnz(AU), nnz(AL), nnz(b), nnz(c)) == (12, 11, 3, 4, 4)
    for M in (A, AU, AL, b, c, (@static COMPREHENSIVE ? (d,) : ())...)
        @test_throws BoundsError nzrange(M, 0)
        @test_throws BoundsError nzrange(M, size(M, 2) + 1)
    end
    @testset "nzrange(A, $i)" for (i, nzr) in ((1,1:0),(4,1:4),(5,5:8),(6,9:12),(9,13:12))
        @test nzrange(A, i) == nzr
    end
    # the wrappers' accessors describe the entries of their triangle only
    @testset "nzrange(AU, $i)" for (i, nzr) in ((2,1:0),(4,1:3),(5,4:7),(6,8:11),(8,12:11))
        @test nzrange(AU, i) == nzr
    end
    @testset "nzrange(AL, $i)" for (i, nzr) in ((3,1:0),(4,1:2),(5,3:3),(6,4:3),(7,4:3))
        @test nzrange(AL, i) == nzr
    end
    @test nzrange(b, 1) == 1:4
    @test nzrange(c, 1) == 1:4

    @test rowvals(A) == I
    @test rowvals(AU) == I[[1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]]
    @test rowvals(AL) == I[[3, 4, 8]]
    @test rowvals(b) == I[1:4]
    @test rowvals(c) == I[5:8]

    @test nonzeros(A) == V
    @test nonzeros(AU) == V[[1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]]
    @test nonzeros(AL) == V[[3, 4, 8]]
    @test nonzeros(b) == V[1:4]
    @test nonzeros(c) == V[5:8]
    # the storage tier addresses the parent's vectors
    for W in (AU, AL)
        @test SparseArrays.getrowval(W) === rowvals(A) && SparseArrays.getnzval(W) === nonzeros(A)
        @test all(j -> SparseArrays.getnzval(W)[SparseArrays.getnzrange(W, j)] == nonzeros(W)[nzrange(W, j)], 1:9)
    end
    # a view of some columns describes its own entries (#376)
    e = view(A, :, 5:6)
    @test nonzeros(e) == V[5:12] && rowvals(e) == I[5:12] && nzrange(e, 2) == 5:8
end

@static if COMPREHENSIVE
@testset "nonzeros, rowvals and nzrange of triangular and symmetric wrappers (#64)" begin
    @testset "$(nameof(T)) $(nameof(W)) $uplo" for T in (Float64, ComplexF64),
            (W, uplo) in ((UpperTriangular, :U), (LowerTriangular, :L), (Symmetric, :U),
                          (Symmetric, :L), (Hermitian, :U), (Hermitian, :L))
        A = fixture(T, 9, 9)                     # with a stored zero on the diagonal
        S = W <: Union{Symmetric,Hermitian} ? W(A, uplo) : W(A)
        C = uplo == :U ? triu(A) : tril(A)       # the stored triangle keeps its pattern
        @test length(nonzeros(S)) == length(rowvals(S)) == nnz(S) == nnz(C)
        @test nonzeros(S) == nonzeros(C)         # the values as stored, before any conjugation
        @test rowvals(S) == rowvals(C)
        @test all(j -> nzrange(S, j) == nzrange(C, j), axes(S, 2))
        @test_throws BoundsError nzrange(S, 0)
        @test_throws BoundsError nzrange(S, 10)
        # the documented loop visits exactly the stored triangle, and the wrapper reads it
        seen = spzeros(T, 9, 9)
        rows, vals = rowvals(S), nonzeros(S)
        for j in axes(S, 2), k in nzrange(S, j)
            seen[rows[k], j] = vals[k]
        end
        @test seen == C
        @test all(S[i, j] == (W <: Hermitian && i == j ? real(C[i, j]) : C[i, j]) for j in 1:9, i in 1:9 if !iszero(C[i, j]))
        # the storage tier and the public accessors walk the same entries, and writes go through
        for j in axes(S, 2)
            @test SparseArrays.getnzval(S)[SparseArrays.getnzrange(S, j)] == nonzeros(S)[nzrange(S, j)]
        end
        other = uplo == :U ? tril(A, -1) : triu(A, 1)
        fill!(nonzeros(S), T(7))
        @test all(==(T(7)), nonzeros(uplo == :U ? triu(A) : tril(A)))
        @test (uplo == :U ? tril(A, -1) : triu(A, 1)) == other
        @test nnz(A) == nnz(C) + nnz(other)
    end
    # a wrapper of a column-range view gathers within the view's columns
    A = fixture(Float64, 12, 14)
    for (S, R) in ((UpperTriangular(view(A, :, 2:13)), UpperTriangular(A[:, 2:13])),
                   (Symmetric(view(A, :, 2:13), :L), Symmetric(A[:, 2:13], :L)))
        @test nonzeros(S) == nonzeros(R) && rowvals(S) == rowvals(R) && nnz(S) == nnz(R)
        @test all(j -> nzrange(S, j) == nzrange(R, j), 1:12)
    end
end
end

@testset "copy a ReshapedArray of SparseMatrixCSC" begin
    A = fixture(Float64, 20, 10)
    rA = reshape(A, 10, 20)
    crA = copy(rA)
    @test reshape(crA, 20, 10) == A
    # shapes that gather many source columns into one destination column, split one source
    # column across many, and leave trailing empty destination columns
    Ti = @static COMPREHENSIVE ? Int32 : Int
    A32 = SparseMatrixCSC{Float64,Ti}(sparse([1, 2, 4, 3, 4], [1, 1, 2, 3, 3], 1.0:5.0, 4, 3))
    for (m, n) in ((12, 1), (1, 12), (2, 6), (6, 2), (3, 4))
        rA = copy(reshape(A32, m, n))
        @test rA isa SparseMatrixCSC{Float64,Ti}
        @test mismatch(rA, reshape(Matrix(A32), m, n); Ti) === nothing
    end
    @test mismatch(copy(reshape(spzeros(4, 3), 6, 2)), zeros(6, 2)) === nothing
    # column boundaries past half of `typemax(Int)`, and a source whose last linear index
    # is `typemax(Int)` itself
    m = typemax(Int) ÷ 2 + 2
    A = spzeros(m, 5)
    copyto!(A, SparseMatrixCSC(m + 1, 1, [1, 2], [m + 1], [1.0]))
    @test getcolptr(A) == [1, 1, 2, 2, 2, 2] && rowvals(A) == [1]
    m = typemax(Int) ÷ 2 + 1
    A = spzeros(m ÷ 2, 4)
    copyto!(A, SparseMatrixCSC(m, 2, [1, 1, 2], [m], [1.0]))
    @test getcolptr(A) == [1, 1, 1, 1, 2] && rowvals(A) == [m ÷ 2]
end

@static if COMPREHENSIVE
@testset "SparseMatrixCSCView" begin
    A  = fixture(Float64, 10, 10)
    vA = view(A, :, 1:5) # a CSCView contains all rows and a UnitRange of the columns
    # the storage tier addresses the parent's vectors
    @test SparseArrays.getnzval(vA)  === SparseArrays.getnzval(A)
    @test SparseArrays.getrowval(vA) === SparseArrays.getrowval(A)
    @test SparseArrays.getcolptr(vA) == SparseArrays.getcolptr(A[:, 1:5])
    @test all(j -> SparseArrays.getnzrange(vA, j) == nzrange(A, j), 1:5)
    sA = view(A, :, 1:10)   # a square view can be wrapped
    for W in (UpperTriangular(sA), LowerTriangular(sA))
        @test SparseArrays.getnzval(W) === SparseArrays.getnzval(A)
        @test SparseArrays.getrowval(W) === SparseArrays.getrowval(A)
        @test all(j -> SparseArrays.getnzrange(W, j) ⊆ nzrange(A, j), 1:10)
    end
    # the public accessors describe the view's own entries
    @test nonzeros(vA) == nonzeros(A[:, 1:5])
    @test rowvals(vA) == rowvals(A[:, 1:5])
    @test all(j -> nzrange(vA, j) == nzrange(A[:, 1:5], j), 1:5)
end

@testset "nonzeros, rowvals and nzrange of column views (#376)" begin
    a = sparse([1 0 2; 0 3 0])
    b = view(a, :, 2:3)
    @test nonzeros(b) == [3, 2]
    @test rowvals(b) == [2, 1]
    @test nzrange(b, 1) == 1:1
    @test nzrange(b, 2) == 2:2
    @test_throws BoundsError nzrange(b, 0)
    @test_throws BoundsError nzrange(b, 3)
    nonzeros(b) .*= 10   # writes go through to the parent
    @test a == [1 0 20; 0 30 0]

    e = view(spzeros(4, 5), :, 10:9)   # an empty range need not lie within the parent
    @test nnz(e) == 0 && isempty(nonzeros(e)) && isempty(rowvals(e))

    @testset "$(nameof(T)) $(name)" for T in (Float64, ComplexF64), (name, cols) in
            (("range", 1:5), ("empty range", 4:3), ("permuted subset", [8, 2, 1]),
             ("repeated column", [4, 6, 4]))
        A = fixture(T, 6, 9)           # with a stored zero in column 1 and an empty column 2
        S = view(A, :, cols)
        C = A[:, cols]                 # the copy has the same stored pattern
        @test length(nonzeros(S)) == length(rowvals(S)) == nnz(S) == nnz(C)
        @test nonzeros(S) == nonzeros(C)
        @test rowvals(S) == rowvals(C)
        @test all(j -> nzrange(S, j) == nzrange(C, j), axes(S, 2))
        @test_throws BoundsError nzrange(S, 0)
        @test_throws BoundsError nzrange(S, length(cols) + 1)
        # the storage tier and the public accessors walk the same entries
        for j in axes(S, 2)
            @test SparseArrays.getnzval(S)[SparseArrays.getnzrange(S, j)] == nonzeros(S)[nzrange(S, j)]
            @test SparseArrays.getrowval(S)[SparseArrays.getnzrange(S, j)] == rowvals(S)[nzrange(S, j)]
        end
        # writes go through to the parent and touch nothing else
        others = setdiff(1:9, cols)
        before = A[:, others]
        fill!(nonzeros(S), T(7))
        @test all(==(T(7)), nonzeros(A[:, cols]))
        @test nnz(A[:, cols]) == nnz(C)
        @test A[:, others] == before
        if cols isa UnitRange && !isempty(cols)
            @test nzrange(S, 1) isa UnitRange
            @test @allocated(nzrange(S, 1)) == 0
        end
    end
end
end

@static if COMPREHENSIVE
@testset "fill! for SubArrays" begin
    a = fixture(Float64, 10, 10)
    b = copy(a)
    sa = view(a, 1:10, 2:3)
    sa_filled = fill!(sa, 0.0)
    # `fill!` should return the sub array instead of its parent.
    @test sa_filled === sa
    b[1:10, 2:3] .= 0.0
    @test a == b
    sb = view(a, 1:2, 1:2)
    @test (@inferred fill!(sb, 1.0)) === sb
    for empty in (view(a, 1:0, 1:2), view(a, 1:2, 1:0))
        @test (@inferred fill!(empty, 3.0)) === empty
    end
    @test a[1:2, 1:2] == fill(1.0, 2, 2)
    @static if COMPREHENSIVE
    A = sparse([1], [1], [Vector{Float64}(undef, 3)], 3, 3)
    A[1,1] = [1.0, 2.0, 3.0]
    B = deepcopy(A)
    sA = view(A, 1:1, 1:2)
    fill!(sA, [4.0, 5.0, 6.0])
    for jj in 1:2
        B[1, jj] = [4.0, 5.0, 6.0]
    end
    @test A == B

    # https://github.com/JuliaSparse/SparseArrays.jl/pull/433
    C = sparse([1], [1], [CustomType("a")], 3, 3)
    sC = view(C, 1:1, 1:2)
    fill!(sC, zero(CustomType))
    @test C[1:1, 1:2] == zeros(CustomType, 1, 2)
    end
end
end

using Base: swaprows!, swapcols!
@testset "swaprows!, swapcols!" begin
    S = sparse(
        [ 0.  0  0  0  0   0
          0  -1  1  1  0   0
          0   0  0  1  1   0
          0   0  1  1  1  -1])

    for (f!, i, j) in
            ((swaprows!, 1, 2), # Test swapping rows where one row is fully sparse
             (swaprows!, 2, 3), # Test swapping rows of unequal length
             (swaprows!, 2, 4), # Test swapping non-adjacent rows
             (swapcols!, 1, 2), # Test swapping columns where one column is fully sparse
             (swapcols!, 2, 3), # Test swapping columns of unequal length
             (swapcols!, 2, 4)) # Test swapping non-adjacent columns
        Scopy = copy(S)
        Sdense = Array(S)
        f!(Scopy, i, j); f!(Sdense, i, j)
        @test mismatch(Scopy, Sdense) === nothing
    end

    for (A, i, j) in (
            (sparse([1.0  2.0  3.0;
                     0.0  0.0  0.0;
                     4.0  5.0  6.0]), 1, 2),
            (sparse([1.0  0.0  5.0;
                     0.0  2.0  0.0;
                     0.0  3.0  6.0;
                     7.0  4.0  0.0]), 1, 2),
            (sparse(reshape([1.0, 2.0, 3.0, 4.0, 0.0, 0.0], 6, 1)), 2, 6))
        Scopy = copy(A)
        Sdense = Array(A)
        swaprows!(Scopy, i, j); swaprows!(Sdense, i, j)
        @test mismatch(Scopy, Sdense) === nothing
    end

    # columns with the same number of stored entries (issue #390)
    x = sparse([9.0 1 8
                0 3 72
                7 4 16])
    swapcols!(x, 2, 3)
    @test x == sparse([9.0 8 1
                       0 72 3
                       7 16 4])

    @static if COMPREHENSIVE
    x0 = copy(x)
    @test_throws BoundsError swaprows!(x, 1, 4)
    @test_throws BoundsError swaprows!(x, 0, 2)
    @test same_pattern(x, x0) && x == x0
    end
end

@testset "issymmetric with stored zeros and missing partner entries" begin
    # column 3 is exhausted by A[1, 3] before the partner of A[3, 2] is looked up
    @test !issymmetric(sparse([3, 3, 1], [1, 2, 3], [1.0, 1.0, 1.0], 3, 3))
    # A[1, 3] has no partner, which the search for the partner of A[3, 2] walks into
    @test !issymmetric(sparse([3, 1], [2, 3], [1.0, 1.0], 3, 3))
    # a stored zero ahead of the partner entry is skipped
    @test issymmetric(SparseMatrixCSC(3, 3, [1, 1, 2, 4], [3, 1, 2], [1.0, 0.0, 1.0]))
end

@testset "count specializations" begin
    # count should throw for sparse arrays for which zero(eltype) does not exist
    @test_throws MethodError count(SparseMatrixCSC(2, 2, Int[1, 2, 3], Int[1, 2], Any[true, true]))
    @test_throws MethodError count(SparseVector(2, Int[1], Any[true]))
    # a stored zero is counted once by a predicate that holds at zero
    A = fixture(Float64, 5, 3)
    @test count(iszero, A) == count(iszero, Matrix(A))
    @test count(iszero, view(A, :, 1:2)) == count(iszero, Matrix(A)[:, 1:2])
end

@testset "show" begin
    io = IOBuffer()
    repl = IOContext(io, :limit=>true)

    A = spzeros(Float64, Int64, 0, 0)
    for (transform, showstring) in zip(
        (identity, adjoint, transpose), (
        "0×0 $SparseMatrixCSC{Float64, Int64} with 0 stored entries",
        "0×0 $Adjoint{Float64, $SparseMatrixCSC{Float64, Int64}} with 0 stored entries",
        "0×0 $Transpose{Float64, $SparseMatrixCSC{Float64, Int64}} with 0 stored entries"
        ))
        show(repl , MIME"text/plain"(), transform(A))
        @test String(take!(io)) == showstring
    end

    A = sparse(Int64[1], Int64[1], [1.0])
    for (transform, showstring) in zip(
        (identity, adjoint, transpose), (
        "1×1 $SparseMatrixCSC{Float64, Int64} with 1 stored entry:\n 1.0",
        "1×1 $Adjoint{Float64, $SparseMatrixCSC{Float64, Int64}} with 1 stored entry:\n 1.0",
        "1×1 $Transpose{Float64, $SparseMatrixCSC{Float64, Int64}} with 1 stored entry:\n 1.0",
        ))
        show(repl , MIME"text/plain"(), transform(A))
        @test String(take!(io)) == showstring
    end

    @static if COMPREHENSIVE
    A = spzeros(Float32, Int64, 2, 2)
    for (transform, showstring) in zip(
        (identity, adjoint, transpose), (
        "2×2 $SparseMatrixCSC{Float32, Int64} with 0 stored entries:\n ⋅  ⋅\n ⋅  ⋅",
        "2×2 $Adjoint{Float32, $SparseMatrixCSC{Float32, Int64}} with 0 stored entries:\n ⋅  ⋅\n ⋅  ⋅",
        "2×2 $Transpose{Float32, $SparseMatrixCSC{Float32, Int64}} with 0 stored entries:\n ⋅  ⋅\n ⋅  ⋅",
        ))
        show(repl , MIME"text/plain"(), transform(A))
        @test String(take!(io)) == showstring
    end
    end

    A = sparse(Int64[1, 1], Int64[1, 2], [1.0, 2.0])
    for (transform, showstring, braille) in zip(
        (identity, adjoint, transpose), (
        "1×2 $SparseMatrixCSC{Float64, Int64} with 2 stored entries:\n 1.0  2.0",
        "2×1 $Adjoint{Float64, $SparseMatrixCSC{Float64, Int64}} with 2 stored entries:\n 1.0\n 2.0",
        "2×1 $Transpose{Float64, $SparseMatrixCSC{Float64, Int64}} with 2 stored entries:\n 1.0\n 2.0",
        ),
        ("\n[⠉]", "\n[⠃]", "\n[⠃]"))
        show(repl , MIME"text/plain"(), transform(A))
        @test String(take!(io)) == showstring
        _show_with_braille_patterns(convert(IOContext, io), transform(A))
        @test contains(String(take!(io)), braille)
    end

    # every 1-dot braille pattern
    for (i, b) in enumerate(split("⠁⠂⠄⡀⠈⠐⠠⢀", ""))
        A = spzeros(8, 4)
        A[mod1(i, 4), (i - 1) ÷ 4 + 1] = 1
        _show_with_braille_patterns(convert(IOContext, io), A)
        out = String(take!(io))
        @test occursin(b, out) == true
        for c in split("⠁⠂⠄⡀⠈⠐⠠⢀", "")
            b == c && continue
            @test occursin(c, out) == false
        end
    end

    # empty braille pattern Char(10240)
    A = spzeros(2, 2)
    for transform in (identity, adjoint, transpose)
        expected = ":\n[" * Char(10240) * "]"
        _show_with_braille_patterns(convert(IOContext, io), transform(A))
        @test contains(String(take!(io)), expected)
    end

    A = sparse(Int64[1, 2, 4, 2, 3], Int64[1, 1, 1, 2, 2], fill(1.0, 5), 4, 2)
    for (transform, showstring, braille) in zip(
        (identity, adjoint, transpose), (
        "4×2 $SparseMatrixCSC{Float64, Int64} with 5 stored entries:\n 1.0   ⋅\n 1.0  1.0\n  ⋅   1.0\n 1.0   ⋅",
        "2×4 $Adjoint{Float64, $SparseMatrixCSC{Float64, Int64}} with 5 stored entries:\n 1.0  1.0   ⋅   1.0\n  ⋅   1.0  1.0   ⋅",
        "2×4 $Transpose{Float64, $SparseMatrixCSC{Float64, Int64}} with 5 stored entries:\n 1.0  1.0   ⋅   1.0\n  ⋅   1.0  1.0   ⋅",
        ),
        ("\n[⡳]", "\n[⠙⠊]", "\n[⠙⠊]"))
        show(repl , MIME"text/plain"(), transform(A))
        @test String(take!(io)) == showstring
        _show_with_braille_patterns(convert(IOContext, io), transform(A))
        @test contains(String(take!(io)), braille)
    end

    @static if COMPREHENSIVE
    A = sparse(Int64[1, 3, 2, 4], Int64[1, 1, 2, 2], Int64[1, 1, 1, 1], 7, 3)
    for (transform, showstring, braille) in zip(
        (identity, adjoint, transpose), (
        "7×3 $SparseMatrixCSC{Int64, Int64} with 4 stored entries:\n 1  ⋅  ⋅\n ⋅  1  ⋅\n 1  ⋅  ⋅\n ⋅  1  ⋅\n ⋅  ⋅  ⋅\n ⋅  ⋅  ⋅\n ⋅  ⋅  ⋅",
        "3×7 $Adjoint{Int64, $SparseMatrixCSC{Int64, Int64}} with 4 stored entries:\n 1  ⋅  1  ⋅  ⋅  ⋅  ⋅\n ⋅  1  ⋅  1  ⋅  ⋅  ⋅\n ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅",
        "3×7 $Transpose{Int64, $SparseMatrixCSC{Int64, Int64}} with 4 stored entries:\n 1  ⋅  1  ⋅  ⋅  ⋅  ⋅\n ⋅  1  ⋅  1  ⋅  ⋅  ⋅\n ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅",
        ),
        ("⎡⢕⠀⎤\n" *
         "⎣⠀⠀⎦",
         "[⠑⠑⠀⠀]",
         "[⠑⠑⠀⠀]"))
        show(repl , MIME"text/plain"(), transform(A))
        @test String(take!(io)) == showstring
        _show_with_braille_patterns(convert(IOContext, io), transform(A))
        @test contains(String(take!(io)), braille)
    end
    end

    A = sparse(Int64[1:10;], Int64[1:10;], fill(Float64(1), 10))
    brailleString = "⎡⠑⢄⠀⠀⠀⎤\n" *
                    "⎢⠀⠀⠑⢄⠀⎥\n" *
                    "⎣⠀⠀⠀⠀⠑⎦"
    for transform in (identity, adjoint, transpose)
        _show_with_braille_patterns(convert(IOContext, io), transform(A))
        @test contains(String(take!(io)), brailleString)
    end

    # Issue #657: wrappers and views display as the sparse matrix they are equal to, which
    # leaves out entries outside the wrapper's triangle, like A[1, 3] and B[40, 1]
    shown, contents = show_plain, show_contents
    A = sparse([1, 2, 1, 3], [1, 1, 3, 3], [1.0, 2.0, 9.0, 3.0])
    C = sparse([1, 1, 2], [1, 2, 2], [1.0+1im, 2im, 3.0+0im])
    for X in (Symmetric(A, :L), view(A, :, 2:3), UpperTriangular(A), Diagonal(sparsevec([1, 3], [1.0, 2.0])),
            (@static COMPREHENSIVE ? (Hermitian(C), UnitLowerTriangular(A), Diagonal(view(A, :, 1))) : ())...)
        @test shown(X) == sprint(summary, X) * ":\n" * contents(sparse(X))
    end
    B = sparse(1:40, [2:40; 1], 1.0, 40, 40)
    for X in (view(B, :, 2:40), (@static COMPREHENSIVE ? (Hermitian(B, :U), LowerTriangular(B)) : ())...)
        @test shown(X; displaysize=(10, 80)) == sprint(summary, X) * ", displaying at 1/2 scale:\n" * contents(sparse(X); displaysize=(10, 80))
    end
    E = Symmetric(spzeros(0, 0))
    @test shown(E) == sprint(summary, E)
    @static if COMPREHENSIVE
    Z = sparse([1im 2im])
    @test contents(Z') == contents(copy(Z'))
    # a stored `#undef` prints as it does for the parent
    U = SparseMatrixCSC(2, 2, [1, 2, 3], [1, 2], Vector{BigFloat}(undef, 2))
    for X in (Hermitian(U),)
        @test shown(X) == sprint(summary, X) * ":\n" * contents(U)
    end
    # matrix-valued entries print as the wrapper's `getindex` returns them
    b = [1 2; 3 4]
    redirect_stderr(devnull) do
        H = Hermitian(sparse([1, 2, 1], [1, 2, 2], [b, b, b]))
        @test contents(H) == contents(sparse([1, 2, 1, 2], [1, 1, 2, 2], [H[1, 1], H[2, 1], H[1, 2], H[2, 2]]))
        @test contents(Diagonal(sparsevec([1], [b], 2))) == contents(sparse([1], [1], [b], 2, 2))
    end
    end
    # the braille pattern of a wrapper is drawn without copying its entries
    shown_bytes(X) = @allocated sprint(show, "text/plain", X; context=(:limit=>true, :displaysize=>(24, 80)))
    D = spdiagm(ones(100_000))
    for X in (D, Symmetric(D), (@static COMPREHENSIVE ? (D', Hermitian(D), UnitLowerTriangular(D), view(D, :, 1:100_000), Diagonal(sparsevec(1:100_000, 1.0))) : ())...)
        shown_bytes(X)
        @test shown_bytes(X) < nnz(D)
    end

    @static if COMPREHENSIVE
    # Issue #30589
    @test sprint(show, "text/plain", sparse([true true]); context=:limit=>true) == "1×2 $SparseMatrixCSC{Bool, $Int} with 2 stored entries:\n 1  1"
    end

    function _filled_sparse(m::Integer, n::Integer)
        C = CartesianIndices((m, n))[:]
        Is = [Int64(x[1]) for x in C]
        Js = [Int64(x[2]) for x in C]
        return sparse(Is, Js, (@static COMPREHENSIVE ? true : 1.0), m, n)
    end

    # vertical scaling
    ioc = IOContext(io, :displaysize => (5, 80), :limit => true)
    _show_with_braille_patterns(ioc, _filled_sparse(10, 10))
    @test contains(String(take!(io)), "\n[⣿⣿]")

    _show_with_braille_patterns(ioc, _filled_sparse(20, 10))
    @test contains(String(take!(io)), "\n[⣿]")

    # horizontal scaling
    ioc = IOContext(io, :displaysize => (80, 4), :limit => true)
    _show_with_braille_patterns(ioc, _filled_sparse(8, 8))
    @test contains(String(take!(io)), "\n[⣿⣿]")

    _show_with_braille_patterns(ioc, _filled_sparse(8, 16))
    @test contains(String(take!(io)), "\n[⠛⠛]")

    # respect IOContext while displaying J
    I, J, V = shuffle(1:50), shuffle(1:50), [1:50;]
    S = sparse(I, J, V)
    I, J, V = I[sortperm(J)], sort(J), V[sortperm(J)]
    @test repr(S) == "sparse($I, $J, $V, $(size(S,1)), $(size(S,2)))"
    limctxt(x) = repr(x, context=:limit=>true)
    expstr = "sparse($(limctxt(I)), $(limctxt(J)), $(limctxt(V)), $(size(S,1)), $(size(S,2)))"
    @test limctxt(S) == expstr

    # `show` of an adjoint or transpose wraps the constructor call (issue #210)
    @test repr(sparse([1 2; 3 4])') == "adjoint(sparse([1, 2, 1, 2], [1, 1, 2, 2], [1, 3, 2, 4], 2, 2))"
    @test repr(transpose(sparse([1 2; 3 4]))) == "transpose(sparse([1, 2, 1, 2], [1, 1, 2, 2], [1, 3, 2, 4], 2, 2))"
end

@testset "show warns about an eltype without a zero and about duplicate entries" begin
    # issue #512: suppresses but does not fix the error
    # entry by entry when it fits a limited display, as a braille pattern otherwise
    x = sparse([1, 1, 3], [1, 4, 3], [20, 0, [2]])
    @test_warn "WARNING: could not find generic zero" repr(MIME("text/plain"), x; context=:limit=>true)
    x = sparse([1, 100, 3], [1, 4, 300], [20, 0, [2]])
    @test_warn "WARNING: could not find generic zero" repr(MIME("text/plain"), x)
    # issue #618: a repeated entry is invalid, so it is marked rather than shown
    x = SparseMatrixCSC(3, 3, [1, 3, 4, 5], [1, 1, 2, 3], [1.0, 1.0, 1.0, 1.0])
    @test_warn "WARNING: array contains duplicate entries" begin
        @test contains(sprint(show, MIME"text/plain"(), x; context=:limit=>true), "‼")
    end
end

@static if COMPREHENSIVE
@testset "issparse for specialized matrix types" begin
    m = fixture(Float64, 10, 10)
    @test issparse(Symmetric(m))
    @test issparse(Hermitian(m))
    @test issparse(LowerTriangular(m))
    @test issparse(LinearAlgebra.UnitLowerTriangular(m))
    @test issparse(UpperTriangular(m))
    @test issparse(LinearAlgebra.UnitUpperTriangular(m))
    @test issparse(adjoint(m))
    @test issparse(transpose(m))
    @test issparse(Symmetric(Array(m))) == false
    @test issparse(Hermitian(Array(m))) == false
    @test issparse(LowerTriangular(Array(m))) == false
    @test issparse(LinearAlgebra.UnitLowerTriangular(Array(m))) == false
    @test issparse(UpperTriangular(Array(m))) == false
    @test issparse(LinearAlgebra.UnitUpperTriangular(Array(m))) == false
    @test issparse(Base.ReshapedArray(m, (20, 5), ()))
    @test issparse(@view m[1:3, :])

    # greater nesting
    @test issparse(Symmetric(UpperTriangular(m)))
    @test issparse(Symmetric(UpperTriangular(Array(m)))) == false
end
end

@testset "equality ==" begin
    A1 = sparse(1.0I, 10, 10)
    A2 = sparse(1.0I, 10, 10)
    nonzeros(A1)[end]=0
    @test A1!=A2
    nonzeros(A1)[end]=1
    @test A1==A2
    A1[1:4,end] .= 1
    @test A1!=A2
    nonzeros(A1)[end-4:end-1].=0
    @test A1==A2
    A2[1:4,end-1] .= 1
    @test A1!=A2
    nonzeros(A2)[end-5:end-2].=0
    @test A1==A2
    A2[2:3,1] .= 1
    @test A1!=A2
    nonzeros(A2)[2:3].=0
    @test A1==A2
    A1[2:5,1] .= 1
    @test A1!=A2
    nonzeros(A1)[2:5].=0
    @test A1==A2
    @test sparse([1,1,0])!=sparse([0,1,1])
end

@testset "reverse" begin
    @testset "$name" for (name, S) in (("standard", sparse([2,2,4], [1,2,5], [-19.0, 73, -7])),
                            (@static COMPREHENSIVE ? (
                            ("fixture", fixture(Float64, 15, 18)),
                            ("zeros", spzeros(20, 40)),
                            ) : ())...,
                            ("fixed", SparseArrays.fixed(sparse([2,2,4], [1,2,5], [-19.0, 73, -7]))))
        w = collect(S)
        revS = reverse(S)
        @test mismatch(revS, reverse(w)) === nothing
        @test nnz(revS) == nnz(S)
        if S isa SparseMatrixCSC
            S2 = copy(S)
            reverse!(S2)
            @test S2 == revS
            @test nnz(S2) == nnz(S)
        end
        for dims in 1:2
            revS = reverse(S; dims)
            @test mismatch(revS, reverse(w; dims)) === nothing
            @test nnz(revS) == nnz(S)
            if S isa SparseMatrixCSC
                S2 = copy(S)
                reverse!(S2; dims)
                @test S2 == revS
                @test nnz(S2) == nnz(S)
            end
        end
        revS = reverse(S, dims=(1,2))
        @test mismatch(revS, reverse(w, dims=(1,2))) === nothing
        @test nnz(revS) == nnz(S)
        if S isa SparseMatrixCSC
            S2 = copy(S)
            reverse!(S2, dims=(1,2))
            @test S2 == revS
            @test nnz(S2) == nnz(S)
        end
    end
    @testset "stored zero, index type and dims validation" begin
        S = (@static COMPREHENSIVE ? SparseMatrixCSC{ComplexF64,Int32} : SparseMatrixCSC{Float64,Int})(sparse([2,2,4], [1,2,5], [-19, 73, -7]))
        S[2,2] = 0
        w = collect(S)
        for dims in (:, 1, 2)
            R = reverse(S; dims)
            @test R == reverse(w; dims) && R isa typeof(S) && nnz(R) == 3
        end
        @test reverse(S, dims=(1,)) == reverse(w, dims=1)
        @test_throws ArgumentError reverse(S, dims=3)
        @test_throws ArgumentError reverse(S, dims=(1,1))
    end
    @testset "reverse! of a fixed pattern" begin
        # a pattern that is its own mirror along both axes: only the values move
        S = fixed(SparseMatrixCSC{Float64,(@static COMPREHENSIVE ? Int32 : Int)}(sparse([1,3,2,1,3], [1,1,2,3,3], [1.0,2,3,0,5])))
        for dims in (:, 1, 2)
            F = copy(S)
            @test reverse!(F; dims) === F && F == reverse(collect(S); dims) && same_pattern(F, S)
        end
        # mirrored along rows only, so reversing the columns would move the pattern
        F = fixed(sparse([1,3], [1,1], [1.0,2], 3, 3))
        @test_throws ArgumentError reverse!(F; dims=2)
        @test_throws ArgumentError reverse!(F)
        @test F == [1 0 0; 0 0 0; 2 0 0] && rowvals(F) == [1,3]
        for dims in (:, 1, 2)
            @test iszero(reverse!(fixed(spzeros(Float64, (@static COMPREHENSIVE ? UInt64 : Int), 3, 3)); dims))
        end
    end
end

@static if COMPREHENSIVE
@testset "hash of complex sparse arrays matches dense" begin
    C = fixture(ComplexF64, 10, 10); c = fixturevec(ComplexF64, 10)
    @test hash(C) == hash(Matrix(C)) && hash(C') == hash(Matrix(C'))
    @test hash(c) == hash(Vector(c))
end
end

@static if COMPREHENSIVE
@testset "views and lazy adjoints go through the sparse kernels" begin
    insparse(f, args...) = which(f, typeof.(args)).module === SparseArrays
    for T in (Float64, ComplexF64)
        A = sparse(T[1 2 0 0; 2 0 3 0; 0 3 0 0; 0 0 0 4]); T <: Complex && (A[1, 2] = 2 + im; A[2, 1] = 2 - im)
        nonzeros(A)[end] = 0   # a stored zero
        B = sparse(T[1 0 2; 0 0 3; 4 5 0; 0 6 0])
        F = fixed(copy(A))
        wrapped(S, D) = ((view(S, :, 2:size(S, 2)), view(D, :, 2:size(D, 2))), (view(S, :, [3, 1, 2]), view(D, :, [3, 1, 2])),
                         (view(S, [3, 1], :), view(D, [3, 1], :)), (S', D'), (transpose(S), transpose(D)))
        for (S, D) in ((A, Array(A)), (B, Array(B)), (F, Array(A))), (X, Y) in wrapped(S, D)
            for f in (reverse, rot180, rotl90, rotr90, permutedims, x -> circshift(x, (1, -1)), x -> circshift(x, 2),
                      x -> reverse(x; dims=1), x -> reverse(x; dims=2), x -> permutedims(x, (1, 2)),
                      x -> copy(reshape(x, size(x, 2), size(x, 1))), x -> copy(reshape(x, 1, length(x))))
                R = f(X)
                @test R isa SparseMatrixCSC{T,Int} && R == f(Y)
            end
            @test insparse(reverse, X) && insparse(rot180, X) && insparse(rotl90, X) && insparse(rotr90, X)
            @test insparse(circshift, X, (1, 1)) && insparse(circshift, X, 1) && insparse(permutedims, X, (2, 1))
            @test insparse(sort, X) && insparse(copy, reshape(X, 1, length(X)))
            @test insparse(issymmetric, X) && insparse(ishermitian, X)
            @test issymmetric(X) == issymmetric(Y) && ishermitian(X) == ishermitian(Y)
            @test count(iszero, X) == count(iszero, Y) && count(x -> imag(x) > 0, X) == count(x -> imag(x) > 0, Y)
            if T <: Real
                for dims in 1:2
                    R = sort(X; dims)
                    @test R isa SparseMatrixCSC{T,Int} && R == sort(Y; dims) && sort(X; dims, rev=true) == sort(Y; dims, rev=true)
                end
            end
            if X isa SubArray
                @test insparse(==, X, X') && insparse(==, X', X) && insparse(isequal, X, transpose(X)) && insparse(==, S, X') && insparse(==, X', S)
                @test (X == X') == (Y == Y') && (X' == X) == (Y' == Y) && (X == transpose(X)) == (Y == transpose(Y))
                @test (copy(X) == X') == (Y == Y') && (transpose(X) == copy(X)) == (Y == Y') || T <: Complex
                @test insparse(+, X, I) && insparse(-, I, X)
                if size(X, 1) == size(X, 2)
                    @test X + I == Y + I && I + X == I + Y && X - 2I == Y - 2I && 2I - X == 2I - Y
                    @test X + I isa SparseMatrixCSC{T,Int}
                else
                    @test_throws DimensionMismatch X + I
                    @test_throws DimensionMismatch I - X
                end
            else
                @test which(Base._simple_count, (typeof(iszero), typeof(X), Int)).module === SparseArrays
            end
        end
        for S in (A + transpose(A), A + A'), X in (view(S, :, :), view(S, :, [1, 2, 3, 4]), view(S, 1:4, 1:4), S', transpose(S), view(S, :, 1:4)')
            Y = collect(X)
            @test issymmetric(X) == issymmetric(Y) && ishermitian(X) == ishermitian(Y)
            @test (X == X') == (Y == Y') && (X == transpose(X)) == (Y == transpose(Y))
        end
    end
    # the parent is counted in place: the allocation does not grow with the matrix
    countzeros(X) = count(iszero, X)
    A = sprand(50, 50, 0.1); B = sprand(500, 500, 0.1)
    countzeros(A'); countzeros(transpose(B))
    @test (@allocated countzeros(A')) == (@allocated countzeros(B'))
    @test countzeros(B') == countzeros(B) && countzeros(transpose(B)) == countzeros(B)
end
end

end # module
