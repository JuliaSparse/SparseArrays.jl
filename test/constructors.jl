# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseConstructorTests

using Test
using SparseArrays
using SparseArrays: getcolptr, nonzeroinds, _show_with_braille_patterns
using LinearAlgebra
using Random
using Test: guardseed
include("forbidproperties.jl")

function same_structure(A, B)
    return all(getfield(A, f) == getfield(B, f) for f in (:m, :n, :colptr, :rowval))
end

@testset "uniform scaling should not change type #103" begin
    A = spzeros(Float32, Int8, 5, 5)
    B = I - A
    @test typeof(B) == typeof(A)
end

@testset "spzeros de-splatting" begin
    @test spzeros(Float64, Int64, (2, 2)) == spzeros(Float64, Int64, 2, 2)
    @test spzeros(Float64, Int32, (2, 2)) == spzeros(Float64, Int32, 2, 2)
    @test spzeros(Float32, (3, 2)) == spzeros(Float32, Int, 3, 2)
    @test spzeros((3, 2)) == spzeros((3, 2)...)
end

@testset "conversion to AbstractMatrix/SparseMatrix of same eltype" begin
    a = sprand(5, 5, 0.2)
    @test AbstractMatrix{eltype(a)}(a) == a
    @test SparseMatrixCSC{eltype(a)}(a) == a
    @test SparseMatrixCSC{eltype(a), Int}(a) == a
    @test SparseMatrixCSC{eltype(a)}(Array(a)) == a
    # a different eltype converts, as `SparseMatrixCSC{Tv,Ti}(::AbstractMatrix)` does
    @test SparseMatrixCSC{Float32}(Array(a))::SparseMatrixCSC{Float32,Int} == SparseMatrixCSC{Float32,Int}(Array(a))
    @test SparseMatrixCSC{ComplexF64}(Array(a)')::SparseMatrixCSC{ComplexF64,Int} == a'
    @test Array(SparseMatrixCSC{eltype(a), Int8}(a)) == Array(a)
    @test collect(a) == a
    # wrappers and views convert through the sparse kernels, not element by element
    c = sprand(ComplexF64, 5, 3, 0.4)
    for w in (a', transpose(c), view(c, :, 2:3), view(c, :, 1)', transpose(view(c, :, 1)))
        @test which(copyto!, Tuple{Matrix{eltype(w)}, typeof(w)}).module == SparseArrays
        @test Matrix(w)::Matrix{eltype(w)} == collect(w)
    end
    # issue #54
    b = SparseMatrixCSC{ComplexF64,Int32}(a)
    @test promote_type(typeof(a), typeof(b)) === SparseMatrixCSC{ComplexF64,Int}
    @test promote_type(typeof(a), Matrix{ComplexF64}) === Matrix{ComplexF64}
    @test promote_type(Matrix{Int}, typeof(a)) === Matrix{Float64}
    @test promote_type(SparseMatrixCSC{Int8,Int}, SparseMatrixCSC{Int16,Int}) === SparseMatrixCSC{Int16,Int}
    @test promote(a, b) == (a, b)
    @test eltype([a, b]) === SparseMatrixCSC{ComplexF64,Int}
end

@testset "sparse matrix construction" begin
    @test (A = fill(1.0+im,5,5); isequal(Array(sparse(A)), A))
    @test_throws ArgumentError sparse([1,2,3], [1,2], [1,2,3], 3, 3)
    @test_throws ArgumentError sparse([1,2,3], [1,2,3], [1,2], 3, 3)
    @test_throws ArgumentError sparse([1,2,3], [1,2,3], [1,2,3], 0, 1)
    @test_throws ArgumentError sparse([1,2,3], [1,2,3], [1,2,3], 1, 0)
    @test_throws ArgumentError sparse([1,2,4], [1,2,3], [1,2,3], 3, 3)
    @test_throws ArgumentError sparse([1,2,3], [1,2,4], [1,2,3], 3, 3)
    @test isequal(sparse(Int[], Int[], Int[], 0, 0), SparseMatrixCSC(0, 0, Int[1], Int[], Int[]))
    @test isequal(sparse(big.([1,1,1,2,2,3,4,5]),big.([1,2,3,2,3,3,4,5]),big.([1,2,4,3,5,6,7,8]), 6, 6),
        SparseMatrixCSC(6, 6, big.([1,2,4,7,8,9,9]), big.([1,1,2,1,2,3,4,5]), big.([1,2,3,4,5,6,7,8])))
    @test sparse(Any[1,2,3], Any[1,2,3], Any[1,1,1]) == sparse([1,2,3], [1,2,3], [1,1,1])
    @test sparse(Any[1,2,3], Any[1,2,3], Any[1,1,1], 5, 4) == sparse([1,2,3], [1,2,3], [1,1,1], 5, 4)
    # with combine
    @test sparse([1, 1, 2, 2, 2], [1, 2, 1, 2, 2], 1.0, 2, 2, +) == sparse([1, 1, 2, 2], [1, 2, 1, 2], [1.0, 1.0, 1.0, 2.0], 2, 2)
    @test sparse([1, 1, 2, 2, 2], [1, 2, 1, 2, 2], -1.0, 2, 2, *) == sparse([1, 1, 2, 2], [1, 2, 1, 2], [-1.0, -1.0, -1.0, 1.0], 2, 2)
    @test sparse(sparse(Int32.(1:5), Int32.(1:5), trues(5))') isa SparseMatrixCSC{Bool,Int32}
    # undef initializer
    sz = (3, 4)
    for m in (SparseMatrixCSC{Float32, Int16}(undef, sz...), SparseMatrixCSC{Float32, Int16}(undef, sz),
                 similar(SparseMatrixCSC{Float32, Int16}, sz))
        @test size(m) == sz
        @test eltype(m) === Float32
        @test m == spzeros(sz...)
    end
end

@testset "spzeros for pattern creation (structural zeros)" begin
    I = [1, 2, 3]
    J = [1, 3, 4]
    V = zeros(length(I))
    S = spzeros(I, J)
    S′ = sparse(I, J, V)
    @test S == S′
    @test same_structure(S, S′)
    @test eltype(S) == Float64
    S = spzeros(Float32, I, J)
    @test S == S′
    @test same_structure(S, S′)
    @test eltype(S) == Float32
    S = spzeros(I, J, 4, 5)
    S′ = sparse(I, J, V, 4, 5)
    @test S == S′
    @test same_structure(S, S′)
    @test eltype(S) == Float64
    S = spzeros(Float32, I, J, 4, 5)
    @test S == S′
    @test same_structure(S, S′)
    @test eltype(S) == Float32
end

@testset "sparsevec from matrices" begin
    X = Matrix(1.0I, 5, 5)
    M = rand(5,4)
    C = spzeros(3,3)
    SX = sparse(X); SM = sparse(M)
    VX = vec(X); VSX = vec(SX)
    VM = vec(M); VSM1 = vec(SM); VSM2 = sparsevec(M)
    VC = vec(C)
    @test VX == VSX
    @test VM == VSM1
    @test VM == VSM2
    @test size(VC) == (9,)
    @test nnz(VC) == 0
    @test nnz(VSX) == 5
end

@testset "test that sparse / sparsevec constructors work for AbstractMatrix subtypes" begin
    D = Diagonal(fill(1,10))
    sm = sparse(D)
    sv = sparsevec(D)

    @test count(!iszero, sm) == 10
    @test count(!iszero, sv) == 10

    @test count(!iszero, sparse(Diagonal(Int[]))) == 0
    @test count(!iszero, sparsevec(Diagonal(Int[]))) == 0
end

@testset "Sparse construction with empty/1x1 structured matrices" begin
    empty = spzeros(0, 0)

    @test sparse(Diagonal(zeros(0, 0))) == empty
    @test sparse(Bidiagonal(zeros(0, 0), :U)) == empty
    @test sparse(Bidiagonal(zeros(0, 0), :L)) == empty
    @test sparse(SymTridiagonal(zeros(0, 0))) == empty
    @test sparse(Tridiagonal(zeros(0, 0))) == empty

    one_by_one = rand(1,1)
    sp_one_by_one = sparse(one_by_one)

    @test sparse(Diagonal(one_by_one)) == sp_one_by_one
    @test sparse(Bidiagonal(one_by_one, :U)) == sp_one_by_one
    @test sparse(Bidiagonal(one_by_one, :L)) == sp_one_by_one
    @test sparse(Tridiagonal(one_by_one)) == sp_one_by_one

    s = SymTridiagonal(rand(1), rand(0))
    @test sparse(s) == s
end

@testset "avoid allocation for zeros in diagonal" begin
    x = [1, 0, 0, 5, 0]
    d = Diagonal(x)
    s = sparse(d)
    @test s == d
    @test nnz(s) == 2
end

@testset "float" begin
    local A
    A = sprand(Bool, 5, 5, 0.0)
    @test eltype(float(A)) == Float64  # issue #11658
    A = sprand(Bool, 5, 5, 0.2)
    @test float(A) == float(Array(A))
end

@testset "complex" begin
    A = sprand(Bool, 5, 5, 0.0)
    @test eltype(complex(A)) == Complex{Bool}
    A = sprand(Bool, 5, 5, 0.2)
    @test complex(A) == complex(Array(A))
end

@testset "one(A::SparseMatrixCSC)" begin
    @test_throws DimensionMismatch one(sparse([1 1 1; 1 1 1]))
    @test one(sparse([1 1; 1 1]))::SparseMatrixCSC == [1 0; 0 1]
end

struct MockTropical{T} <: Number begin
    n::T
    end
end
MockTropical{T}(x::MockTropical{T}) where {T} = x
Base.zero(::Type{MockTropical{T}}) where {T} = MockTropical{T}(typemin(T))
Base.zero(x::MockTropical{T}) where {T} = zero(MockTropical{T})
Base.one(::Type{MockTropical{T}}) where {T} = MockTropical{T}(zero(T))
Base.one(x::MockTropical{T}) where {T} = one(MockTropical{T})
Base.:*(a::MockTropical{T}, b::MockTropical{T}) where {T} = MockTropical{T}(a.n + b.n)
Base.:+(a::MockTropical{T}, b::MockTropical{T}) where {T} = MockTropical{T}(max(a.n, b.n))

@testset "issue #731" begin
    x = MockTropical{Float64}(1.0)
    B = sparse([1, 2], [1,2], [x, x])
    C = [one(x) zero(x); zero(x) one(x)]
    @test one(B) == C
    @test B^0 == C
end

@testset "sparsevec" begin
    local A = sparse(fill(1, 5, 5))
    @test sparsevec(A) == fill(1, 25)
    @test sparsevec([1:5;], 1) == fill(1, 5)
    @test_throws ArgumentError sparsevec([1:5;], [1:4;])
end

@testset "sparse" begin
    local A = sparse(fill(1, 5, 5))
    @test sparse(A) == A
    @test sparse([1:5;], [1:5;], 1) == sparse(1.0I, 5, 5)
end

@testset "test created type of sprand{T}(::Type{T}, m::Integer, n::Integer, density::AbstractFloat)" begin
    m = sprand(Float32, 10, 10, 0.1)
    @test eltype(m) == Float32
    m = sprand(Float64, 10, 10, 0.1)
    @test eltype(m) == Float64
    m = sprand(Int32, 10, 10, 0.1)
    @test eltype(m) == Int32
end

@testset "sprand" begin
    p=0.3; m=1000; n=2000;
    for s in 1:10
        # build a (dense) random matrix with randsubset + rand
        Random.seed!(s);
        v = randsubseq(1:m*n,p);
        x = zeros(m,n);
        x[v] .= rand(length(v));
        # redo the same with sprand
        Random.seed!(s);
        a = sprand(m,n,p);
        @test x == a
    end
end

@testset "sprandn with type $T" for T in (Float64, Float32, Float16, ComplexF64, ComplexF32, ComplexF16)
    @test sprandn(T, 5, 5, 0.5) isa AbstractSparseMatrix{T}
end

@testset "sprandn with invalid type $T" for T in (AbstractFloat, Complex)
    @test_throws MethodError sprandn(T, 5, 5, 0.5)
end

@testset "sparse! and spzeros!" begin
    using SparseArrays: sparse!, spzeros!, getcolptr, getrowval, nonzeros

    function allocate_arrays(m, n)
        N = round(Int, 0.5 * m * n)
        Tv, Ti = Float64, Int
        I = Ti[rand(1:m) for _ in 1:N]; I = Ti[I; I]
        J = Ti[rand(1:n) for _ in 1:N]; J = Ti[J; J]
        V = Tv.(I)
        csrrowptr = Vector{Ti}(undef, m + 1)
        csrcolval = Vector{Ti}(undef, length(I))
        csrnzval = Vector{Tv}(undef, length(I))
        klasttouch = Vector{Ti}(undef, n)
        csccolptr = Vector{Ti}(undef, n + 1)
        cscrowval = Vector{Ti}()
        cscnzval = Vector{Tv}()
        return I, J, V, klasttouch, csrrowptr, csrcolval, csrnzval, csccolptr, cscrowval, cscnzval
    end

    for (m, n) in ((10, 5), (5, 10), (10, 10))
        # Passing csr vectors
        I, J, V, klasttouch, csrrowptr, csrcolval, csrnzval = allocate_arrays(m, n)
        S  = sparse(I, J, V, m, n)
        S! = sparse!(I, J, V, m, n, +, klasttouch, csrrowptr, csrcolval, csrnzval)
        @test S == S!
        @test same_structure(S, S!)

        I, J, _, klasttouch, csrrowptr, csrcolval = allocate_arrays(m, n)
        S  = spzeros(I, J, m, n)
        S! = spzeros!(Float64, I, J, m, n, klasttouch, csrrowptr, csrcolval)
        @test S == S!
        @test iszero(S!)
        @test same_structure(S, S!)

        # Passing csr vectors + csccolptr
        I, J, V, klasttouch, csrrowptr, csrcolval, csrnzval, csccolptr = allocate_arrays(m, n)
        S  = sparse(I, J, V, m, n)
        S! = sparse!(I, J, V, m, n, +, klasttouch, csrrowptr, csrcolval, csrnzval, csccolptr)
        @test S == S!
        @test same_structure(S, S!)
        @test getcolptr(S!) === csccolptr

        I, J, _, klasttouch, csrrowptr, csrcolval, _, csccolptr = allocate_arrays(m, n)
        S  = spzeros(I, J, m, n)
        S! = spzeros!(Float64, I, J, m, n, klasttouch, csrrowptr, csrcolval, csccolptr)
        @test S == S!
        @test iszero(S!)
        @test same_structure(S, S!)
        @test getcolptr(S!) === csccolptr

        # Passing csr vectors, and csc vectors
        I, J, V, klasttouch, csrrowptr, csrcolval, csrnzval, csccolptr, cscrowval, cscnzval =
            allocate_arrays(m, n)
        S  = sparse(I, J, V, m, n)
        S! = sparse!(I, J, V, m, n, +, klasttouch, csrrowptr, csrcolval, csrnzval,
                     csccolptr, cscrowval, cscnzval)
        @test S == S!
        @test same_structure(S, S!)
        @test getcolptr(S!) === csccolptr
        @test getrowval(S!) === cscrowval
        @test nonzeros(S!) === cscnzval

        I, J, _, klasttouch, csrrowptr, csrcolval, _, csccolptr, cscrowval, cscnzval =
            allocate_arrays(m, n)
        S  = spzeros(I, J, m, n)
        S! = spzeros!(Float64, I, J, m, n, klasttouch, csrrowptr, csrcolval,
                      csccolptr, cscrowval, cscnzval)
        @test S == S!
        @test iszero(S!)
        @test same_structure(S, S!)
        @test getcolptr(S!) === csccolptr
        @test getrowval(S!) === cscrowval
        @test nonzeros(S!) === cscnzval

        # Passing csr vectors, and csc vectors of insufficient lengths
        I, J, V, klasttouch, csrrowptr, csrcolval, csrnzval, csccolptr, cscrowval, cscnzval =
            allocate_arrays(m, n)
        S  = sparse(I, J, V, m, n)
        S! = sparse!(I, J, V, m, n, +, klasttouch, csrrowptr, csrcolval, csrnzval,
                     resize!(csccolptr, 0), resize!(cscrowval, 0), resize!(cscnzval, 0))
        @test S == S!
        @test same_structure(S, S!)
        @test getcolptr(S!) === csccolptr
        @test getrowval(S!) === cscrowval
        @test nonzeros(S!) === cscnzval

        I, J, _, klasttouch, csrrowptr, csrcolval, _, csccolptr, cscrowval, cscnzval =
            allocate_arrays(m, n)
        S  = spzeros(I, J, m, n)
        S! = spzeros!(Float64, I, J, m, n, klasttouch, csrrowptr, csrcolval,
                      resize!(csccolptr, 0), resize!(cscrowval, 0), resize!(cscnzval, 0))
        @test S == S!
        @test iszero(S!)
        @test same_structure(S, S!)
        @test getcolptr(S!) === csccolptr
        @test getrowval(S!) === cscrowval
        @test nonzeros(S!) === cscnzval

        # Passing csr vectors, and csc vectors aliased with I, J, V
        I, J, V, klasttouch, csrrowptr, csrcolval, csrnzval = allocate_arrays(m, n)
        S  = sparse(I, J, V, m, n)
        S! = sparse!(I, J, V, m, n, +, klasttouch, csrrowptr, csrcolval, csrnzval, I, J, V)
        @test S == S!
        @test same_structure(S, S!)
        @test getcolptr(S!) === I
        @test getrowval(S!) === J
        @test nonzeros(S!) === V

        I, J, V, klasttouch, csrrowptr, csrcolval = allocate_arrays(m, n)
        S  = spzeros(I, J, m, n)
        S! = spzeros!(Float64, I, J, m, n, klasttouch, csrrowptr, csrcolval, I, J, V)
        @test S == S!
        @test iszero(S!)
        @test same_structure(S, S!)
        @test getcolptr(S!) === I
        @test getrowval(S!) === J
        @test nonzeros(S!) === V

        # Test reuse of I, J, V for the matrix buffers in
        # sparse!(I, J, V), sparse!(I, J, V, m, n), sparse!(I, J, V, m, n, combine),
        # spzeros!(T, I, J), and spzeros!(T, I, J, m, n).
        I, J, V = allocate_arrays(m, n)
        S = sparse(I, J, V)
        S! = sparse!(I, J, V)
        @test S == S!
        @test same_structure(S, S!)
        @test getcolptr(S!) === I
        @test getrowval(S!) === J
        @test nonzeros(S!) === V
        I, J, V = allocate_arrays(m, n)
        S = sparse(I, J, V, 2m, 2n)
        S! = sparse!(I, J, V, 2m, 2n)
        @test S == S!
        @test same_structure(S, S!)
        @test getcolptr(S!) === I
        @test getrowval(S!) === J
        @test nonzeros(S!) === V
        I, J, V = allocate_arrays(m, n)
        S = sparse(I, J, V, 2m, 2n, *)
        S! = sparse!(I, J, V, 2m, 2n, *)
        @test S == S!
        @test same_structure(S, S!)
        @test getcolptr(S!) === I
        @test getrowval(S!) === J
        @test nonzeros(S!) === V
        for T in (Float32, Float64)
            I, J, = allocate_arrays(m, n)
            S = spzeros(T, I, J)
            S! = spzeros!(T, I, J)
            @test S == S!
            @test same_structure(S, S!)
            @test eltype(S) == eltype(S!) == T
            @test getcolptr(S!) === I
            @test getrowval(S!) === J
            I, J, = allocate_arrays(m, n)
            S = spzeros(T, I, J, 2m, 2n)
            S! = spzeros!(T, I, J, 2m, 2n)
            @test S == S!
            @test same_structure(S, S!)
            @test eltype(S) == eltype(S!) == T
            @test getcolptr(S!) === I
            @test getrowval(S!) === J
        end
    end
end

end # module
