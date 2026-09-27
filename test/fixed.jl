using Test, SparseArrays, LinearAlgebra
using SparseArrays: AbstractSparseVector, AbstractSparseMatrixCSC, FixedSparseCSC, FixedSparseVector, ReadOnly,
    getcolptr, rowvals, nonzeros, nonzeroinds, _is_fixed, fixed, move_fixed, fkeep!

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
    @test_throws ErrorException resize!(r, length(r) - 1)
    @test_throws ErrorException resize!(r, length(r) + 1)
    @test_throws ErrorException r[1] = r[1] + 1
    @test_throws ErrorException r[1] = r[1] - 1
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
end

struct_eq(A, B, C...) = struct_eq(A, B) && struct_eq(B, C...)
struct_eq(A::AbstractSparseMatrixCSC, B::AbstractSparseMatrixCSC) =
    getcolptr(A) == getcolptr(B) && rowvals(A) == rowvals(B)
struct_eq(A::AbstractSparseVector, B::AbstractSparseVector) =
    nonzeroinds(A) == nonzeroinds(B)

@testset "FixedSparseCSC" begin
    A = sprandn(10, 10, 0.3)

    F = FixedSparseCSC(copy(A))
    Ft = FixedSparseCSC{eltype(A),eltype(rowvals(A))}(A)
    @test typeof(Ft) == typeof(F)
    @test Ft == F

    @test struct_eq(F, A)
    nonzeros(F) .= 0
    @test struct_eq(F, A)
    dropzeros!(F)
    @test struct_eq(F, A)
    H = F ./ 1
    @test typeof(H) == typeof(F)
    @test struct_eq(F, H, A)
    H = map!(zero, copy(F), F)
    @test struct_eq(F, H, A)
    F .= false
    @test struct_eq(F, H, A)
    F .= A .+ A
    @test F == A .+ A
    @test struct_eq(F, H, A)
    F .= A .- A
    @test F == A .- A
    @test struct_eq(F, H, A)
    F .= H .* A
    @test F == H .* A
    @test struct_eq(F, H, A)

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
    @test struct_eq(B, F)
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
    y = sprandn(10, 0.3)
    x = FixedSparseVector(copy(y))
    @test struct_eq(x, y)
    nonzeros(x) .= 0
    @test struct_eq(x, y)
    dropzeros!(x)
    @test struct_eq(x, y)
    z = x ./ 2
    @test struct_eq(x, y, z)
    f(x, y, z) = @allocated(x .= y .+ y) +
        @allocated(x .= y .- y) +
        @allocated(x .= z .* y)
    f(x, y, z)
    f(x, y, z)
    @test f(x, y, z) == 0
    t = similar(x)
    @test typeof(t) == typeof(x)
    @test struct_eq(t, x)
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


@testset "Test factorization" begin
    b = sprandn(10, 10, 0.99) + I
    a = fixed(b)

    @test (lu(a) \ randn(10); true)
    @test b == a
    @test (qr(a + a') \ randn(10); true)
    @test b == a
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
