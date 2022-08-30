using Test, SparseArrays, LinearAlgebra
using SparseArrays: AbstractSparseVector, AbstractSparseMatrixCSC, FixedSparseCSC, FixedSparseVector, ReadOnly,
    getcolptr, rowvals, nonzeros, nonzeroinds, _is_fixed, fixed, move_fixed

@testset "ReadOnly" begin
    v = randn(100)
    r = ReadOnly(v)
    @test length(r) == length(v)
    @test (resize!(r, length(r)); true)
    @test_throws ErrorException resize!(r, length(r) - 1)
    @test_throws ErrorException resize!(r, length(r) + 1)
    @test_throws ErrorException r[1] = r[1] + 1
    @test_throws ErrorException r[1] = r[1] - 1
    @test (r[1] = r[1]; true)
end

struct_eq(A, B, C...) = struct_eq(A, B) && struct_eq(B, C...)
struct_eq(A::AbstractSparseMatrixCSC, B::AbstractSparseMatrixCSC) =
    getcolptr(A) == getcolptr(B) && rowvals(A) == rowvals(B)
struct_eq(A::AbstractSparseVector, B::AbstractSparseVector) =
    nonzeroinds(A) == nonzeroinds(B)
struct_eq(x, y, z...) = struct_eq(x, y) && (length(z) == 0 || struct_eq(y, z...))

@testset "FixedSparseCSC" begin
    A = sprandn(10, 10, 0.3)

    F = FixedSparseCSC(copy(A))
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

@testset "Test factorization" begin
    b = sprandn(10, 10, 0.99) + I
    a = fixed(b)

    @test (lu(a) \ randn(10); true)
    @test b == a
    @test (qr(a + a') \ randn(10); true)
    @test b == a
end

