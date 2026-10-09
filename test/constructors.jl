# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseConstructorTests

using Test
using SparseArrays
using SparseArrays: getcolptr, nonzeroinds, _show_with_braille_patterns
using LinearAlgebra
using Random
using Test: guardseed
include("testhelpers.jl")

@static if COMPREHENSIVE
@testset "uniform scaling should not change type #103" begin
    A = spzeros(Float32, Int8, 5, 5)
    B = I - A
    @test typeof(B) == typeof(A)
end
end

@testset "spzeros de-splatting" begin
    @test spzeros(Float64, Int64, (2, 2)) == spzeros(Float64, Int64, 2, 2)
@static if COMPREHENSIVE
    @test spzeros(Float64, Int32, (2, 2)) == spzeros(Float64, Int32, 2, 2)
end
    @test spzeros(Float64, (3, 2)) == spzeros(Float64, Int, 3, 2)
    @test spzeros((3, 2)) == spzeros((3, 2)...)
end

@testset "conversion to AbstractMatrix/SparseMatrix of same eltype" begin
    a = fixture(Float64, 5, 3)
    @test mismatch(AbstractMatrix{eltype(a)}(a), Array(a); Ti=Int) === nothing
    @test mismatch(SparseMatrixCSC{eltype(a)}(a), Array(a); Ti=Int) === nothing
    @test mismatch(SparseMatrixCSC{eltype(a), Int}(a), Array(a); Ti=Int) === nothing
    @test mismatch(SparseMatrixCSC{eltype(a)}(Array(a)), Array(a); Ti=Int) === nothing
    # a different eltype converts, as `SparseMatrixCSC{Tv,Ti}(::AbstractMatrix)` does
    @test mismatch(SparseMatrixCSC{ComplexF64}(Array(a)), Array(a); Tv=ComplexF64, Ti=Int) === nothing
@static if COMPREHENSIVE
    @test SparseMatrixCSC{ComplexF64}(Array(a)')::SparseMatrixCSC{ComplexF64,Int} == a'
    @test Array(SparseMatrixCSC{eltype(a), Int8}(a)) == Array(a)
end
    @test collect(a) == a
    @test SparseMatrixCSC(a)::typeof(a) == a
    # an adjoint or transpose is materialized before the eltype conversion
    @test mismatch(SparseMatrixCSC{ComplexF64}(a'), Array(a'); Tv=ComplexF64, Ti=Int) === nothing
    @test mismatch(SparseMatrixCSC{ComplexF64}(transpose(a)), Array(a'); Tv=ComplexF64, Ti=Int) === nothing
    # wrappers and views convert through the sparse kernels, not element by element
    c = fixture(ComplexF64, 5, 3)
    for w in (transpose(c), (@static COMPREHENSIVE ? (view(c, :, 2:3), view(c, :, 1)') : ())...)
        @test which(copyto!, Tuple{Matrix{eltype(w)}, typeof(w)}).module == SparseArrays
        @test Matrix(w)::Matrix{eltype(w)} == collect(w)
    end
    # issue #54
    b = @static COMPREHENSIVE ? SparseMatrixCSC{ComplexF64,Int32}(a) : c
    @test promote_type(typeof(a), typeof(b)) === SparseMatrixCSC{ComplexF64,Int}
    @test promote_type(typeof(a), Matrix{ComplexF64}) === Matrix{ComplexF64}
    @test promote_type(Matrix{Int}, typeof(a)) === Matrix{Float64}
@static if COMPREHENSIVE
    @test promote_type(SparseMatrixCSC{Int8,Int}, SparseMatrixCSC{Int16,Int}) === SparseMatrixCSC{Int16,Int}
    @test promote(a, b) == (a, b)
    @test eltype([a, b]) === SparseMatrixCSC{ComplexF64,Int}
end
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
    # positions as Cartesian indices
    IJ = [CartesianIndex(3, 1), CartesianIndex(1, 2), CartesianIndex(3, 1)]
    @test mismatch(sparse(IJ, [1.0, 2.0, 4.0], 3, 4), [0 2.0 0 0; 0 0 0 0; 5.0 0 0 0]) === nothing
@static if COMPREHENSIVE
    @test mismatch(sparse(IJ, [1.0, 2.0, 4.0]), [0 2.0; 0 0; 5.0 0]) === nothing   # size from the indices
    @test mismatch(sparse(IJ, 1.5), [0 1.5; 0 0; 3.0 0]) === nothing && sparse(IJ, 1.5, 3, 4) == sparse([3, 1, 3], [1, 2, 1], 1.5, 3, 4)
    @test sparse(IJ, [1.0, 2.0, 4.0], 3, 4, max) == sparse([3, 1, 3], [1, 2, 1], [1.0, 2.0, 4.0], 3, 4, max)
    @test sparse(IJ, 2.0, 3, 4, *) == sparse([3, 1, 3], [1, 2, 1], 2.0, 3, 4, *)
    @test sparse(IJ, [true, false, true], 3, 4) == sparse([3, 1, 3], [1, 2, 1], [true, false, true], 3, 4)   # `|`, not `+`
    @test mismatch(sparse(StepRangeLen(CartesianIndex(1, 1), CartesianIndex(1, 1), 3), [1, 2, 3]), [1 0 0; 0 2 0; 0 0 3]) === nothing
    @test mismatch(sparse(CartesianIndex{2}[], Float64[], 2, 3), zeros(2, 3)) === nothing
    @test_throws ArgumentError sparse(IJ, [1.0, 2.0], 3, 4)
    @test_throws ArgumentError sparse(IJ, [1.0, 2.0, 4.0], 2, 4)
    @test_throws MethodError sparse([CartesianIndex(1, 1, 1)], [1.0], 3, 3)
    @test isequal(sparse(big.([1,1,1,2,2,3,4,5]),big.([1,2,3,2,3,3,4,5]),big.([1,2,4,3,5,6,7,8]), 6, 6),
        SparseMatrixCSC(6, 6, big.([1,2,4,7,8,9,9]), big.([1,1,2,1,2,3,4,5]), big.([1,2,3,4,5,6,7,8])))
    @test sparse(Any[1,2,3], Any[1,2,3], Any[1,1,1]) == sparse([1,2,3], [1,2,3], [1,1,1])
end
    # with combine
    @test sparse([1, 1, 2, 2, 2], [1, 2, 1, 2, 2], 1.0, 2, 2, +) == sparse([1, 1, 2, 2], [1, 2, 1, 2], [1.0, 1.0, 1.0, 2.0], 2, 2)
    # duplicates fold from the left in input order, which a commutative `combine` cannot show
    @test nonzeros(sparse([1, 2, 1, 1], [3, 1, 3, 3], [1.0, 5.0, 2.0, 3.0], 2, 3, -)) == [5.0, -4.0]
@static if COMPREHENSIVE
    # duplicates of `Bool` values combine with `|`
    @test mismatch(sparse([1, 1, 2], [1, 1, 3], [true, false, true], 2, 3), [true false false; false false true]; Ti=Int) === nothing
    @test sparse(sparse(Int32.(1:5), Int32.(1:5), trues(5))') isa SparseMatrixCSC{Bool,Int32}
end
    # row and column indices of different integer types are converted to `Int`
    for Ti in (Int16,)
        S = sparse(Ti[1,2,3], [1,2,3], [1.0, 2.0, 3.0], 3, 3)
        @test S::SparseMatrixCSC{Float64,Int} == sparse([1,2,3], [1,2,3], [1.0, 2.0, 3.0], 3, 3)
    end
    # undef initializer
    sz = (3, 4)
    Tv, Ti = Float64, Int
    for m in (SparseMatrixCSC{Tv, Ti}(undef, sz...), SparseMatrixCSC{Tv, Ti}(undef, sz),
                 similar(SparseMatrixCSC{Tv, Ti}, sz))
        @test size(m) == sz
        @test eltype(m) === Tv
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
    @test same_pattern(S, S′)
    @test eltype(S) == Float64
    S = spzeros(I, J, 4, 5)
    S′ = sparse(I, J, V, 4, 5)
    @test S == S′
    @test same_pattern(S, S′)
    @test eltype(S) == Float64
    S = spzeros(ComplexF64, I, J, 4, 5)
    @test S == S′
    @test same_pattern(S, S′)
    @test eltype(S) == ComplexF64
end

@testset "sparsevec from matrices" begin
    X = Matrix(1.0I, 5, 5)
    M = Matrix(fixture(Float64, 5, 4))
    C = spzeros(3,3)
    SX = sparse(X); SM = sparse(M)
    VX = vec(X); VSX = vec(SX)
    VM = vec(M); VSM1 = vec(SM); VSM2 = sparsevec(M)
    VC = vec(C)
    @test VX == VSX
    @test VM == VSM1
    @test mismatch(VSM2, VM; Ti=Int) === nothing
    @test size(VC) == (9,)
    @test nnz(VC) == 0
    @test nnz(VSX) == 5
end

@static if COMPREHENSIVE
@testset "test that sparse / sparsevec constructors work for AbstractMatrix subtypes" begin
    D = Diagonal(fill(1.0, 10))
    sm = sparse(D)
    sv = sparsevec(D)

    @test count(!iszero, sm) == 10
    @test count(!iszero, sv) == 10

    @test count(!iszero, sparse(Diagonal(eltype(D)[]))) == 0
    @test count(!iszero, sparsevec(Diagonal(eltype(D)[]))) == 0
end
end

@testset "Sparse construction with empty/1x1 structured matrices" begin
    empty = spzeros(0, 0)

    @test sparse(Diagonal(zeros(0, 0))) == empty
    @test sparse(Bidiagonal(zeros(0, 0), :U)) == empty
    @test sparse(Bidiagonal(zeros(0, 0), :L)) == empty
    @test sparse(SymTridiagonal(zeros(0, 0))) == empty
    @test sparse(Tridiagonal(zeros(0, 0))) == empty

    one_by_one = fill(2.5, 1, 1)
    sp_one_by_one = sparse(one_by_one)

    @test sparse(Diagonal(one_by_one)) == sp_one_by_one
    @test sparse(Bidiagonal(one_by_one, :U)) == sp_one_by_one
    @test sparse(Bidiagonal(one_by_one, :L)) == sp_one_by_one
    @test sparse(Tridiagonal(one_by_one)) == sp_one_by_one

    s = SymTridiagonal([2.5], Float64[])
    @test sparse(s) == s

    # with the diagonals all different, so that one taken for another changes the result
    M = reshape(Float64.(1:9), 3, 3)
    for T in (Bidiagonal(M, :U), Bidiagonal(M, :L), Tridiagonal(M))
        @test mismatch(sparse(T), Matrix(T); Ti=Int) === nothing
    end
end

@testset "avoid allocation for zeros in diagonal" begin
    x = [1.0, 0.0, 0.0, 5.0, 0.0]
    d = Diagonal(x)
    s = sparse(d)
    @test mismatch(s, Matrix(d); Ti=Int) === nothing
    @test nnz(s) == 2
end

@static if COMPREHENSIVE
@testset "float" begin
    local A
    A = spzeros(Bool, 5, 5)
    @test eltype(float(A)) == Float64  # issue #11658
    A = sparse([1, 3, 5, 2], [1, 2, 2, 5], [true, false, true, true], 5, 5)  # a stored `false`
    @test float(A) == float(Array(A))
end

@testset "complex" begin
    A = spzeros(Bool, 5, 5)
    @test eltype(complex(A)) == Complex{Bool}
    A = sparse([1, 3, 5, 2], [1, 2, 2, 5], [true, false, true, true], 5, 5)  # a stored `false`
    @test complex(A) == complex(Array(A))
end
end

@testset "one(A::SparseMatrixCSC)" begin
    @test_throws DimensionMismatch one(sparse(ones(2, 3)))
    @test one(sparse(ones(2, 2)))::SparseMatrixCSC == [1 0; 0 1]
end

@testset "SparseMatrixCSC construction from UniformScaling" begin
    # more columns than rows, so that the column pointers of the trailing empty columns are checked
    S = SparseMatrixCSC{Float64,Int}(2I, 3, 5)
    @test S::SparseMatrixCSC{Float64,Int} == sparse(2.0I, 3, 5)
    @test S == Matrix(2.0I, 3, 5)
    @test getcolptr(S) == [1, 2, 3, 4, 4, 4]
end

# linalg.jl compares the same call with `diagm` when COMPREHENSIVE
@static if !COMPREHENSIVE
@testset "spdiagm without diagonals" begin
    S = spdiagm(3, 4)
    @test S isa SparseMatrixCSC{Bool,Int} && size(S) == (3, 4) && nnz(S) == 0
end
end

@testset "conversion to special LinearAlgebra types" begin
    # a diagonal matrix is representable as each of the structured types
    S = sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0])
    @test convert(Diagonal, S)::Diagonal == S
    # `isa` only: `==` would compile once per structured type; linalg.jl compares the values
    for T in (SymTridiagonal, Tridiagonal, LowerTriangular, UpperTriangular)
        @test convert(T, S) isa T
    end
    @test_throws ArgumentError convert(Diagonal, sparse([1, 2, 1], [1, 2, 2], [1.0, 2.0, 3.0]))
end

@static if COMPREHENSIVE
@testset "issue #731" begin
    x = MockTropical{Float64}(1.0)
    B = sparse([1, 2], [1,2], [x, x])
    C = [one(x) zero(x); zero(x) one(x)]
    @test one(B) == C
    @test B^0 == C
end
end

@testset "sparsevec" begin
    x = 1.0
    local A = sparse(fill(x, 5, 5))
    @test mismatch(sparsevec(A), fill(x, 25); Ti=Int) === nothing
    @test sparsevec([1:5;], x) == fill(x, 5)
    @test_throws ArgumentError sparsevec([1:5;], [1:4;])
end

@testset "sparse" begin
    x = 1.0
    local A = sparse(fill(x, 5, 5))
    @test sparse(A) == A
    @test sparse([1:5;], [1:5;], x) == sparse(1.0I, 5, 5)
end

@testset "test created type of sprand{T}(::Type{T}, m::Integer, n::Integer, density::AbstractFloat)" begin
    Random.seed!(1)
@static if COMPREHENSIVE
    m = sprand(Float32, 10, 10, 0.1)
    @test eltype(m) == Float32
end
    m = sprand(Float64, 10, 10, 0.1)
    @test eltype(m) == Float64
    m = sprand(ComplexF64, 10, 10, 0.1)
    @test eltype(m) == ComplexF64
end

@testset "sprand" begin
    p=0.3; m=1000; n=2000;
    for s in 1:(@static COMPREHENSIVE ? 2 : 1)
        # build a (dense) random matrix with randsubset + rand
        Random.seed!(s);
        v = randsubseq(1:m*n,p);
        x = zeros(m,n);
        x[v] .= rand(length(v));
        # redo the same with sprand
        Random.seed!(s);
        a = sprand(m,n,p);
        @test mismatch(a, x; Ti=Int) === nothing
    end
end

@static if COMPREHENSIVE
@testset "sprandn with type $T" for T in (Float64, ComplexF32)
    Random.seed!(1)
    @test sprandn(T, 5, 5, 0.5) isa AbstractSparseMatrix{T}
end
end

@testset "sprandn with invalid type $T" for T in (AbstractFloat, Complex)
    @test_throws MethodError sprandn(T, 5, 5, 0.5)
end

@testset "sparse! and spzeros!" begin
    using SparseArrays: sparse!, spzeros!, getcolptr, getrowval, nonzeros

    function allocate_arrays(m, n)
        N = round(Int, 0.5 * m * n)
        Tv, Ti = Float64, Int
        # unsorted, with repeated entries, and with rows and columns left empty
        I = Ti[mod1(k * k, m) for k in 1:N]; I = Ti[I; I]
        J = Ti[mod1(k + k ÷ 4, n) for k in 1:N]; J = Ti[J; J]
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

    for (m, n) in ((10, 5), (@static COMPREHENSIVE ? ((5, 10), (10, 10)) : ())...)
        # Passing csr vectors
        I, J, V, klasttouch, csrrowptr, csrcolval, csrnzval = allocate_arrays(m, n)
        S  = sparse(I, J, V, m, n)
        S! = sparse!(I, J, V, m, n, +, klasttouch, csrrowptr, csrcolval, csrnzval)
        @test S == S!
        @test same_pattern(S, S!)

        I, J, _, klasttouch, csrrowptr, csrcolval = allocate_arrays(m, n)
        S  = spzeros(I, J, m, n)
        S! = spzeros!(Float64, I, J, m, n, klasttouch, csrrowptr, csrcolval)
        @test S == S!
        @test iszero(S!)
        @test same_pattern(S, S!)

        # Passing csr vectors + csccolptr
        I, J, V, klasttouch, csrrowptr, csrcolval, csrnzval, csccolptr = allocate_arrays(m, n)
        S  = sparse(I, J, V, m, n)
        S! = sparse!(I, J, V, m, n, +, klasttouch, csrrowptr, csrcolval, csrnzval, csccolptr)
        @test S == S!
        @test same_pattern(S, S!)
        @test getcolptr(S!) === csccolptr

@static if COMPREHENSIVE
        I, J, _, klasttouch, csrrowptr, csrcolval, _, csccolptr = allocate_arrays(m, n)
        S  = spzeros(I, J, m, n)
        S! = spzeros!(Float64, I, J, m, n, klasttouch, csrrowptr, csrcolval, csccolptr)
        @test S == S!
        @test iszero(S!)
        @test same_pattern(S, S!)
        @test getcolptr(S!) === csccolptr
end

        # Passing csr vectors, and csc vectors
        I, J, V, klasttouch, csrrowptr, csrcolval, csrnzval, csccolptr, cscrowval, cscnzval =
            allocate_arrays(m, n)
        S  = sparse(I, J, V, m, n)
        S! = sparse!(I, J, V, m, n, +, klasttouch, csrrowptr, csrcolval, csrnzval,
                     csccolptr, cscrowval, cscnzval)
        @test S == S!
        @test same_pattern(S, S!)
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
        @test same_pattern(S, S!)
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
        @test same_pattern(S, S!)
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
        @test same_pattern(S, S!)
        @test getcolptr(S!) === csccolptr
        @test getrowval(S!) === cscrowval
        @test nonzeros(S!) === cscnzval

        # Passing csr vectors, and csc vectors aliased with I, J, V
        I, J, V, klasttouch, csrrowptr, csrcolval, csrnzval = allocate_arrays(m, n)
        S  = sparse(I, J, V, m, n)
        S! = sparse!(I, J, V, m, n, +, klasttouch, csrrowptr, csrcolval, csrnzval, I, J, V)
        @test S == S!
        @test same_pattern(S, S!)
        @test getcolptr(S!) === I
        @test getrowval(S!) === J
        @test nonzeros(S!) === V

        I, J, V, klasttouch, csrrowptr, csrcolval = allocate_arrays(m, n)
        S  = spzeros(I, J, m, n)
        S! = spzeros!(Float64, I, J, m, n, klasttouch, csrrowptr, csrcolval, I, J, V)
        @test S == S!
        @test iszero(S!)
        @test same_pattern(S, S!)
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
        @test same_pattern(S, S!)
        @test getcolptr(S!) === I
        @test getrowval(S!) === J
        @test nonzeros(S!) === V
@static if COMPREHENSIVE
        I, J, V = allocate_arrays(m, n)
        S = sparse(I, J, V, 2m, 2n)
        S! = sparse!(I, J, V, 2m, 2n)
        @test S == S!
        @test same_pattern(S, S!)
        @test getcolptr(S!) === I
        @test getrowval(S!) === J
        @test nonzeros(S!) === V
        I, J, V = allocate_arrays(m, n)
        S = sparse(I, J, V, 2m, 2n, *)
        S! = sparse!(I, J, V, 2m, 2n, *)
        @test S == S!
        @test same_pattern(S, S!)
        @test getcolptr(S!) === I
        @test getrowval(S!) === J
        @test nonzeros(S!) === V
end
        for T in (@static COMPREHENSIVE ? (Float32, Float64) : (Float64,))
            I, J, = allocate_arrays(m, n)
            S = spzeros(T, I, J)
            S! = spzeros!(T, I, J)
            @test S == S!
            @test same_pattern(S, S!)
            @test eltype(S) == eltype(S!) == T
            @test getcolptr(S!) === I
            @test getrowval(S!) === J
@static if COMPREHENSIVE
            I, J, = allocate_arrays(m, n)
            S = spzeros(T, I, J, 2m, 2n)
            S! = spzeros!(T, I, J, 2m, 2n)
            @test S == S!
            @test same_pattern(S, S!)
            @test eltype(S) == eltype(S!) == T
            @test getcolptr(S!) === I
            @test getrowval(S!) === J
end
        end
    end
end

end # module
