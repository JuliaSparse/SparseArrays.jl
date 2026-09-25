# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseMatrixTests

using Test
using SparseArrays
using SparseArrays: getcolptr, nonzeroinds, _show_with_braille_patterns, _isnotzero, _isimplicitzero, fixed, _is_fixed
using LinearAlgebra
using Random
using Test: guardseed
include("forbidproperties.jl")

@testset "_isnotzero" begin
    @test !_isnotzero(0::Int)
    @test _isnotzero(1::Int)
    @test _isnotzero(missing)
    @test !_isnotzero(0.0)
    @test _isnotzero(1.0)
end

@testset "_isimplicitzero" begin
    @test _isimplicitzero(0, Int)
    @test !_isimplicitzero(1, Int)
    @test _isimplicitzero(0.0, Float64)
    @test !_isimplicitzero(-0.0, Float64)   # egality keeps the sign of a zero
    @test !_isimplicitzero(missing, Union{Missing,Int})
    @test _isimplicitzero(big(0.0), BigFloat)
    @test !_isimplicitzero(-big(0.0), BigFloat)
    @test _isimplicitzero(zero(Complex{BigFloat}), Complex{BigFloat})
    @test !_isimplicitzero(Complex{BigFloat}(0, 1), Complex{BigFloat})
    @test !_isimplicitzero([0.0], Vector{Float64})   # array elements are always stored
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

@testset "hash matches dense" begin
    # The stored-entries-only complexity guarantee is checked with an operation-counting
    # eltype below ("hash walks stored entries only").
    n = 1000
    A = spzeros(n, n); A[1, 1] = 1
    B = copy(A); B[2, 2] = 0.0   # explicitly stored zero must not change the hash
    @test hash(B) == hash(A) && isequal(B, A)
    for m in (2, 10, 200), X in (sprand(m, m, 0.1), sprandn(m, m, 0.3), spzeros(m, m))
        k = min(3, nnz(X)); nonzeros(X)[1:k] .= [NaN, -0.0, 0.0][1:k]
        @test hash(X) == hash(Matrix(X))
        @test hash(X, UInt(7)) == hash(Matrix(X), UInt(7))
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
    A = sparse(Int8[1,1],Int8[1,1],[1,1])
    @test SparseArrays.indtype(A) == Int8
    for W in (A', transpose(A), Symmetric(A), Hermitian(A), UpperTriangular(A),
              view(A, :, 1:1), view(A, :, [1]), view(A, :, 1))
        @test SparseArrays.indtype(W) == Int8
    end
    @test SparseArrays.indtype(sparsevec(Int8[1], [1.0])') == Int8
end

@testset "exported CSC accessors" begin
    for name in (:AbstractSparseMatrixCSC, :getcolptr, :getrowval, :getnzval, :indtype)
        @test Base.isexported(SparseArrays, name)
    end
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

# A quantity with a unit: `one` is the dimensionless identity, `oneunit` keeps the unit
struct Meters <: Number
    x::Int
    Meters(x::Int) = new(x)
end
Base.zero(::Type{Meters}) = Meters(0)
Base.one(::Type{Meters}) = 1

@testset "oneunit of sparse matrix" begin
    A = sparse([Meters(0) Meters(0); Meters(0) Meters(0)])
    @test oneunit(sprand(2, 2, 0.5)) isa SparseMatrixCSC{Float64}
    @test oneunit(A) isa SparseMatrixCSC{Meters}
    @test oneunit(A) == [Meters(1) Meters(0); Meters(0) Meters(1)]
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
counter::Int = 0
resetcounter() = (global counter; counter=0)
stepcounter() = (global counter; counter+=1)
getcounter() = (global counter; counter)
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

# `Base.hash` on a large array skips runs of equal values with `findprev(!isequal(elt), A, i)`.
# Each such call on a sparse array costs at most nnz(A) + 1 element comparisons and `hash`
# makes only a handful of them, whereas the generic `findprev` performs up to length(A).
@testset "hash walks stored entries only (issue #570)" begin
    n = 10^5
    v = sparsevec([1, n ÷ 2], Counting.([1.0, 2.0]), n)
    w = sparsevec([1, n ÷ 2, n], Counting.([1.0, 0.0, 3.0]), n)
    A = sparse([1, n ÷ 2], [1, n], Counting.([1.0, 2.0]), n, n)
    B = sparse([1, n ÷ 2, 7], [1, n, 7], Counting.([1.0, 2.0, 0.0]), n, n)
    for x in (v, w, A, B)
        resetcounter()
        hash(x)
        @test getcounter() <= 8 * (nnz(x) + 1)
    end
    @test hash(v) == hash(Vector(v)) && hash(w) == hash(Vector(w))
end

@testset "Comparisons to adjoints are efficient" for
    # The counting guard below distinguishes stored-entry traversal from the generic
    # length(A) fallback, so these do not need to be large matrices.
    A in Any[sparse(1*I(100)), sprandn(100, 100, 0.1), sprandn(ComplexF64, 100, 100, 0.9)],
    B in Any[sparse(1*I(100)), sprandn(100, 100, 0.1), sprandn(ComplexF64, 100, 100, 0.9)]
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

@testset "products of LinearAlgebra's Q types with sparse operands" begin
    D = randn(7, 7)
    m = size(D, 1)
    # one operand of each kind gives the same dense result as its dense copy
    B, C, b = sprandn(m, 3, 0.5), sprandn(3, m, 0.5), sprandn(m, 0.5)
    @testset "$name" for (name, Q) in (("qr", qr(D).Q), ("pivoted qr", qr(D, ColumnNorm()).Q),
                                       ("hessenberg", hessenberg(D).Q), ("lq", lq(D).Q))
        for X in (B, sparse(B')', view(B, :, 1:2))
            @test (Q * X)::Matrix ≈ Q * Matrix(X)
        end
        for X in (C, transpose(sparse(transpose(C))), view(C, :, 1:m), view(B, :, 1:2)', transpose(b))
            @test (X * Q')::Matrix ≈ Matrix(X) * Q'
        end
        @test (Q' * B)::Matrix ≈ Q' * Matrix(B)
        @test (C * Q)::Matrix ≈ Matrix(C) * Q
        for x in (b, view(B, :, 1), view(b, 1:m))
            @test (Q * x)::Vector ≈ Q * Vector(x)
        end
        @test (Q' * b)::Vector ≈ Q' * Vector(b)
        @test (b' * Q)::Adjoint ≈ Vector(b)' * Q
        @test_throws DimensionMismatch Q * sprandn(m + 1, 2, 0.5)
    end
end

@testset "repeat tests" begin
    A = sprand(6, 4, 0.5)
    A_full = Matrix(A)
    for m = 0:3
        @test issparse(repeat(A, m))
        @test repeat(A, m) == repeat(A_full, m)
        for n = 0:3
            @test issparse(repeat(A, m, n))
            @test repeat(A, m, n) == repeat(A_full, m, n)
        end
    end
    # a non-Int index type is kept, including in the column pointers
    A32 = SparseMatrixCSC{ComplexF64,Int32}(sprand(ComplexF64, 5, 3, 0.5))
    A32_full = Matrix(A32)
    for m = 0:2, n = 0:3
        R = repeat(A32, m, n)
        @test R isa SparseMatrixCSC{ComplexF64,Int32}
        @test R == repeat(A32_full, m, n)
        @test repeat(A32, m) isa SparseMatrixCSC{ComplexF64,Int32}
    end
end

@testset "copyto!" begin
    A = sprand(5, 5, 0.2)
    B = sprand(5, 5, 0.2)
    Ar = copyto!(A, B)
    @test Ar === A
    @test A == B
    @test pointer(nonzeros(A)) != pointer(nonzeros(B))
    @test pointer(rowvals(A)) != pointer(rowvals(B))
    @test pointer(getcolptr(A)) != pointer(getcolptr(B))
    # Test size(A) != size(B), but length(A) == length(B)
    B = sprand(25, 1, 0.2)
    copyto!(A, B)
    @test A[:] == B[:]
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
    # Test eltype(A) != eltype(B), size(A) != size(B)
    A = sprand(5, 5, 0.2)
    Aorig = copy(A)
    B = sparse(rand(Float32, 3, 3))
    copyto!(A, B)
    @test A[1:9] == B[:]
    @test A[10:end] == Aorig[10:end]
    # Test eltype(A) != eltype(B), size(A) == size(B)
    A = sparse(rand(Float64, 3, 3))
    B = sparse(rand(Float32, 3, 3))
    copyto!(A, B)
    @test A == B
    # Test copyto!(dense, sparse)
    B = sprand(5, 5, 1.0)
    A = rand(5,5)
    A´ = similar(A)
    Ac = copyto!(A, B)
    @test Ac === A
    @test A == copyto!(A´, Matrix(B))
    # Test copyto!(dense, Rdest, sparse, Rsrc)
    A = rand(5,5)
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
    A = sprand(3, 3, 1.0); B = ones(4, 4)
    Bc = copyto!(B, A)
    @test B[4, :] != B[:, 4] == ones(4)
    @test Bc === B
    # Allow no-op copyto! with empty source even for incompatible eltypes
    A = sparse(fill("", 0, 0))
    @test copyto!(B, A) == B

    # Test correct error for too small destination array
    @test_throws BoundsError copyto!(rand(2,2), sprand(3,3,0.2))
end

@testset "error conditions for reshape, and dropdims" begin
    local A = sprand(Bool, 5, 5, 0.2)
    @test_throws DimensionMismatch reshape(A,(20, 2))
    @test_throws ArgumentError dropdims(A,dims=(1, 1))
end

@testset "droptol" begin
    A = guardseed(1234321) do
        triu(sprand(10, 10, 0.2))
    end
    @test getcolptr(SparseArrays.droptol!(A, 0.01)) == [1, 1, 1, 1, 2, 2, 2, 4, 4, 5, 5]
    @test isequal(SparseArrays.droptol!(sparse([1], [1], [1]), 1), SparseMatrixCSC(1, 1, Int[1, 1], Int[], Int[]))
end

@testset "dropzeros[!]" begin
    smalldim = 5
    largedim = 10
    nzprob = 0.4
    targetnumposzeros = 5
    targetnumnegzeros = 5
    for (m, n) in ((largedim, largedim), (smalldim, largedim), (largedim, smalldim))
        local A = sprand(m, n, nzprob)
        struczerosA = findall(x -> x == 0, A)
        poszerosinds = unique(rand(struczerosA, targetnumposzeros))
        negzerosinds = unique(rand(struczerosA, targetnumnegzeros))
        Aposzeros = copy(A)
        Aposzeros[poszerosinds] .= 2
        Anegzeros = copy(A)
        Anegzeros[negzerosinds] .= -2
        Abothsigns = copy(Aposzeros)
        Abothsigns[negzerosinds] .= -2
        map!(x -> x == 2 ? 0.0 : x, nonzeros(Aposzeros), nonzeros(Aposzeros))
        map!(x -> x == -2 ? -0.0 : x, nonzeros(Anegzeros), nonzeros(Anegzeros))
        map!(x -> x == 2 ? 0.0 : x == -2 ? -0.0 : x, nonzeros(Abothsigns), nonzeros(Abothsigns))
        for Awithzeros in (Aposzeros, Anegzeros, Abothsigns)
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
    # test for issue #5169, modified for new behavior following #15242/#14798
    @test nnz(sparse([1, 1], [1, 2], [0.0, -0.0])) == 2
    @test nnz(dropzeros!(sparse([1, 1], [1, 2], [0.0, -0.0]))) == 0
    # test for issue #5437, modified for new behavior following #15242/#14798
    @test nnz(sparse([1, 2, 3], [1, 2, 3], [0.0, 1.0, 2.0])) == 3
    @test nnz(dropzeros!(sparse([1, 2, 3],[1, 2, 3],[0.0, 1.0, 2.0]))) == 2
end

@testset "similar should not alias the input sparse array" begin
    a = sparse(rand(3,3) .+ 0.1)
    b = similar(a, Float32, Int32)
    c = similar(b, Float32, Int32)
    SparseArrays.dropstored!(b, 1, 1)
    @test length(rowvals(c)) == 9
    @test length(nonzeros(c)) == 9
end

@testset "similar with type conversion" begin
    local A = sparse(1.0I, 5, 5)
    @test size(similar(A, ComplexF64, Int)) == (5, 5)
    @test typeof(similar(A, ComplexF64, Int)) == SparseMatrixCSC{ComplexF64, Int}
    @test size(similar(A, ComplexF64, Int8)) == (5, 5)
    @test typeof(similar(A, ComplexF64, Int8)) == SparseMatrixCSC{ComplexF64, Int8}
    @test similar(A, ComplexF64,(6, 6)) == spzeros(ComplexF64, 6, 6)
    @test convert(Matrix, A) == Array(A) # lolwut, are you lost, test?
end

@testset "similar for SparseMatrixCSC" begin
    local A = sparse(1.0I, 5, 5)
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
    simA = similar(A, Float32, Int8)
    @test typeof(simA) == SparseMatrixCSC{Float32,Int8}
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
    simA = similar(A, Float32, Int8, (6,6))
    @test typeof(simA) == SparseMatrixCSC{Float32, Int8}
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
    # test entry points to similar with entry type, index type, and non-Dims shape specification
    @test similar(A, Float32, Int8, 6, 6) == similar(A, Float32, Int8, (6, 6))
    @test similar(A, Float32, Int8, 6) == similar(A, Float32, Int8, (6,))
end

@testset "similar should preserve underlying storage type and uplo flag" begin
    m, n = 4, 3
    sparsemat = sprand(m, m, 0.5)
    for SymType in (Symmetric, Hermitian)
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
    sparsemat = sprand(m, m, 0.5)
    for TriType in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)
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
    d = view(b, :)

    @testset "nnz $n" for (n, M, nz) in (("A", A, 12), ("AU", AU, 11), ("AL", AL, 3),
                                         ("b", b, 4), ("c", c, 4), ("d", d, 4))
        @test nnz(M) == nz
        @test_throws BoundsError nzrange(M, 0)
        @test_throws BoundsError nzrange(M, size(M, 2) + 1)
    end
    @testset "nzrange(A, $i)" for (i, nzr) in ((1,1:0),(4,1:4),(5,5:8),(6,9:12),(9,13:12))
        @test nzrange(A, i) == nzr
    end
    @testset "nzrange(AU, $i)" for (i, nzr) in ((2,1:0),(4,1:3),(5,5:8),(6,9:12),(8,13:12))
        @test nzrange(AU, i) == nzr
    end
    @testset "nzrange(AL, $i)" for (i, nzr) in ((3,1:0),(4,3:4),(5,8:8),(6,13:12),(7,13:12))
        @test nzrange(AL, i) == nzr
    end
    @test nzrange(b, 1) == 1:4
    @test nzrange(c, 1) == 1:4
    @test nzrange(d, 1) == 1:4

    @test rowvals(A) == I
    @test rowvals(AL) == I
    @test rowvals(AL) == I
    @test rowvals(b) == I[1:4]
    @test rowvals(c) == I[5:8]
    @test rowvals(d) == I[1:4]

    @test nonzeros(A) == V
    @test nonzeros(AU) == V
    @test nonzeros(AL) == V
    @test nonzeros(b) == V[1:4]
    @test nonzeros(c) == V[5:8]
    @test nonzeros(d) == V[1:4]
end

@testset "copy a ReshapedArray of SparseMatrixCSC" begin
    A = sprand(20, 10, 0.2)
    rA = reshape(A, 10, 20)
    crA = copy(rA)
    @test reshape(crA, 20, 10) == A
end

@testset "SparseMatrixCSCView" begin
    A  = sprand(10, 10, 0.2)
    vA = view(A, :, 1:5) # a CSCView contains all rows and a UnitRange of the columns
    @test SparseArrays.getnzval(vA)  == SparseArrays.getnzval(A)
    @test SparseArrays.getrowval(vA) == SparseArrays.getrowval(A)
    @test SparseArrays.getcolptr(vA) == SparseArrays.getcolptr(A[:, 1:5])
end

@testset "fill! for SubArrays" begin
    a = sprand(10, 10, 0.2)
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
    struct Foo
       x::Int
    end
    Base.zero(::Type{Foo}) = Foo(0)
    Base.zero(::Foo) = zero(Foo)
    C = sparse([1], [1], [Foo(3)], 3, 3)
    sC = view(C, 1:1, 1:2)
    fill!(sC, zero(Foo))
    @test C[1:1, 1:2] == zeros(Foo, 1, 2)
end

using Base: swaprows!, swapcols!
@testset "swaprows!, swapcols!" begin
    S = sparse(
        [ 0   0  0  0  0   0
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
        @test Scopy == Sdense
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
        @test Scopy == Sdense
    end
end

@testset "count specializations" begin
    # count should throw for sparse arrays for which zero(eltype) does not exist
    @test_throws MethodError count(SparseMatrixCSC(2, 2, Int[1, 2, 3], Int[1, 2], Any[true, true]))
    @test_throws MethodError count(SparseVector(2, Int[1], Any[true]))
end

@testset "show" begin
    io = IOBuffer()

    A = spzeros(Float64, Int64, 0, 0)
    for (transform, showstring) in zip(
        (identity, adjoint, transpose), (
        "0×0 $SparseMatrixCSC{Float64, Int64} with 0 stored entries",
        "0×0 $Adjoint{Float64, $SparseMatrixCSC{Float64, Int64}} with 0 stored entries",
        "0×0 $Transpose{Float64, $SparseMatrixCSC{Float64, Int64}} with 0 stored entries"
        ))
        show(io, MIME"text/plain"(), transform(A))
        @test String(take!(io)) == showstring
    end

    A = sparse(Int64[1], Int64[1], [1.0])
    for (transform, showstring) in zip(
        (identity, adjoint, transpose), (
        "1×1 $SparseMatrixCSC{Float64, Int64} with 1 stored entry:\n 1.0",
        "1×1 $Adjoint{Float64, $SparseMatrixCSC{Float64, Int64}} with 1 stored entry:\n 1.0",
        "1×1 $Transpose{Float64, $SparseMatrixCSC{Float64, Int64}} with 1 stored entry:\n 1.0",
        ))
        show(io, MIME"text/plain"(), transform(A))
        @test String(take!(io)) == showstring
    end

    A = spzeros(Float32, Int64, 2, 2)
    for (transform, showstring) in zip(
        (identity, adjoint, transpose), (
        "2×2 $SparseMatrixCSC{Float32, Int64} with 0 stored entries:\n  ⋅    ⋅ \n  ⋅    ⋅ ",
        "2×2 $Adjoint{Float32, $SparseMatrixCSC{Float32, Int64}} with 0 stored entries:\n  ⋅    ⋅ \n  ⋅    ⋅ ",
        "2×2 $Transpose{Float32, $SparseMatrixCSC{Float32, Int64}} with 0 stored entries:\n  ⋅    ⋅ \n  ⋅    ⋅ ",
        ))
        show(io, MIME"text/plain"(), transform(A))
        @test String(take!(io)) == showstring
    end

    A = sparse(Int64[1, 1], Int64[1, 2], [1.0, 2.0])
    for (transform, showstring, braille) in zip(
        (identity, adjoint, transpose), (
        "1×2 $SparseMatrixCSC{Float64, Int64} with 2 stored entries:\n 1.0  2.0",
        "2×1 $Adjoint{Float64, $SparseMatrixCSC{Float64, Int64}} with 2 stored entries:\n 1.0\n 2.0",
        "2×1 $Transpose{Float64, $SparseMatrixCSC{Float64, Int64}} with 2 stored entries:\n 1.0\n 2.0",
        ),
        ("⎡⠁⠈⎤\n" *
         "⎣⠀⠀⎦",
         "⎡⠁⠀⎤\n" *
         "⎣⡀⠀⎦",
         "⎡⠁⠀⎤\n" *
         "⎣⡀⠀⎦"))
        show(io, MIME"text/plain"(), transform(A))
        @test String(take!(io)) == showstring
        _show_with_braille_patterns(convert(IOContext, io), transform(A))
        @test String(take!(io)) == braille
    end

    # every 1-dot braille pattern
    for (i, b) in enumerate(split("⠁⠂⠄⡀⠈⠐⠠⢀", ""))
        A = spzeros(Int64, Int64, 8, 4)
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
    A = spzeros(Int64, Int64, 4, 2)
    for transform in (identity, adjoint, transpose)
        expected = "⎡" * Char(10240)^2 * "⎤\n⎣" * Char(10240)^2 * "⎦"
        _show_with_braille_patterns(convert(IOContext, io), transform(A))
        @test String(take!(io)) == expected
    end

    A = sparse(Int64[1, 2, 4, 2, 3], Int64[1, 1, 1, 2, 2], Int64[1, 1, 1, 1, 1], 4, 2)
    for (transform, showstring, braille) in zip(
        (identity, adjoint, transpose), (
        "4×2 $SparseMatrixCSC{Int64, Int64} with 5 stored entries:\n 1  ⋅\n 1  1\n ⋅  1\n 1  ⋅",
        "2×4 $Adjoint{Int64, $SparseMatrixCSC{Int64, Int64}} with 5 stored entries:\n 1  1  ⋅  1\n ⋅  1  1  ⋅",
        "2×4 $Transpose{Int64, $SparseMatrixCSC{Int64, Int64}} with 5 stored entries:\n 1  1  ⋅  1\n ⋅  1  1  ⋅",
        ),
        ("⎡⠅⠠⎤\n" *
         "⎣⡀⠐⎦",
         "⎡⠉⠈⎤\n" *
         "⎣⢀⡀⎦",
         "⎡⠉⠈⎤\n" *
         "⎣⢀⡀⎦"))
        show(io, MIME"text/plain"(), transform(A))
        @test String(take!(io)) == showstring
        _show_with_braille_patterns(convert(IOContext, io), transform(A))
        @test String(take!(io)) == braille
    end

    A = sparse(Int64[1, 3, 2, 4], Int64[1, 1, 2, 2], Int64[1, 1, 1, 1], 7, 3)
    for (transform, showstring, braille) in zip(
        (identity, adjoint, transpose), (
        "7×3 $SparseMatrixCSC{Int64, Int64} with 4 stored entries:\n 1  ⋅  ⋅\n ⋅  1  ⋅\n 1  ⋅  ⋅\n ⋅  1  ⋅\n ⋅  ⋅  ⋅\n ⋅  ⋅  ⋅\n ⋅  ⋅  ⋅",
        "3×7 $Adjoint{Int64, $SparseMatrixCSC{Int64, Int64}} with 4 stored entries:\n 1  ⋅  1  ⋅  ⋅  ⋅  ⋅\n ⋅  1  ⋅  1  ⋅  ⋅  ⋅\n ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅",
        "3×7 $Transpose{Int64, $SparseMatrixCSC{Int64, Int64}} with 4 stored entries:\n 1  ⋅  1  ⋅  ⋅  ⋅  ⋅\n ⋅  1  ⋅  1  ⋅  ⋅  ⋅\n ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅",
        ),
        ("⎡⢕⠀⎤\n" *
         "⎣⠀⠀⎦",
         "⎡⢁⢁⠀⠀⎤\n" *
         "⎣⠀⠀⠀⠀⎦",
         "⎡⢁⢁⠀⠀⎤\n" *
         "⎣⠀⠀⠀⠀⎦"))
        show(io, MIME"text/plain"(), transform(A))
        @test String(take!(io)) == showstring
        _show_with_braille_patterns(convert(IOContext, io), transform(A))
        @test String(take!(io)) == braille
    end

    A = sparse(Int64[1:10;], Int64[1:10;], fill(Float64(1), 10))
    brailleString = "⎡⠑⢄⠀⠀⠀⎤\n" *
                    "⎢⠀⠀⠑⢄⠀⎥\n" *
                    "⎣⠀⠀⠀⠀⠑⎦"
    for transform in (identity, adjoint, transpose)
        _show_with_braille_patterns(convert(IOContext, io), transform(A))
        @test String(take!(io)) == brailleString
    end

    # Issue #30589
    @test repr("text/plain", sparse([true true])) == "1×2 $SparseMatrixCSC{Bool, $Int} with 2 stored entries:\n 1  1"

    function _filled_sparse(m::Integer, n::Integer)
        C = CartesianIndices((m, n))[:]
        Is = [Int64(x[1]) for x in C]
        Js = [Int64(x[2]) for x in C]
        return sparse(Is, Js, true, m, n)
    end

    # vertical scaling
    ioc = IOContext(io, :displaysize => (5, 80), :limit => true)
    _show_with_braille_patterns(ioc, _filled_sparse(10, 10))
    @test String(take!(io)) == "⎡⣿⣿⎤\n" *
                               "⎣⣿⣿⎦"

    _show_with_braille_patterns(ioc, _filled_sparse(20, 10))
    @test String(take!(io)) == "⎡⣿⣿⎤\n" *
                               "⎣⣿⣿⎦"

    # horizontal scaling
    ioc = IOContext(io, :displaysize => (80, 4), :limit => true)
    _show_with_braille_patterns(ioc, _filled_sparse(8, 8))
    @test String(take!(io)) == "⎡⣿⣿⎤\n" *
                               "⎣⣿⣿⎦"

    _show_with_braille_patterns(ioc, _filled_sparse(8, 16))
    @test String(take!(io)) == "⎡⣿⣿⎤\n" *
                               "⎣⣿⣿⎦"

    # respect IOContext while displaying J
    I, J, V = shuffle(1:50), shuffle(1:50), [1:50;]
    S = sparse(I, J, V)
    I, J, V = I[sortperm(J)], sort(J), V[sortperm(J)]
    @test repr(S) == "sparse($I, $J, $V, $(size(S,1)), $(size(S,2)))"
    limctxt(x) = repr(x, context=:limit=>true)
    expstr = "sparse($(limctxt(I)), $(limctxt(J)), $(limctxt(V)), $(size(S,1)), $(size(S,2)))"
    @test limctxt(S) == expstr
end

@testset "issparse for specialized matrix types" begin
    m = sprand(10, 10, 0.1)
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

@testset "expandptr" begin
    local A = sparse(1.0I, 5, 5)
    @test SparseArrays.expandptr(getcolptr(A)) == 1:5
    A[1,2] = 1
    @test SparseArrays.expandptr(getcolptr(A)) == [1; 2; 2; 3; 4; 5]
    @test_throws ArgumentError SparseArrays.expandptr([2; 3])
end

@testset "reverse" begin
    @testset "$name" for (name, S) in (("standard", sparse([2,2,4], [1,2,5], [-19, 73, -7])),
                            ("sprand", sprand(Float32, 15, 18, 0.2)),
                            ("zeros", spzeros(Int8, 20, 40)),
                            ("fixed", SparseArrays.fixed(sparse([2,2,4], [1,2,5], [-19, 73, -7]))))
        w = collect(S)
        revS = reverse(S)
        @test revS == reverse(w)
        @test nnz(revS) == nnz(S)
        if S isa SparseMatrixCSC
            S2 = copy(S)
            reverse!(S2)
            @test S2 == revS
            @test nnz(S2) == nnz(S)
        end
        for dims in 1:2
            revS = reverse(S; dims)
            @test revS == reverse(w; dims)
            @test nnz(revS) == nnz(S)
            if S isa SparseMatrixCSC
                S2 = copy(S)
                reverse!(S2; dims)
                @test S2 == revS
                @test nnz(S2) == nnz(S)
            end
        end
        revS = reverse(S, dims=(1,2))
        @test revS == reverse(w, dims=(1,2))
        @test nnz(revS) == nnz(S)
        if S isa SparseMatrixCSC
            S2 = copy(S)
            reverse!(S2, dims=(1,2))
            @test S2 == revS
            @test nnz(S2) == nnz(S)
        end
    end
end

end # module
