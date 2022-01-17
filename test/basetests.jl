# As part of the migration of SparseArrays.jl into its own repo,
# these tests have been moved from other files in julia tests to
# the SparseArrays.jl repo

using Random, LinearAlgebra, SparseArrays

# From arrayops.jl

@testset "copy!" begin
    @testset "AbstractVector" begin
        s = Vector([1, 2])
        for a = ([1], UInt[1], [3, 4, 5], UInt[3, 4, 5])
            @test s === copy!(s, SparseVector(a)) == Vector(a)
        end
end

@testset "CartesianIndex" begin
   a = spzeros(2,3)
    @test CartesianIndices(size(a)) == eachindex(a)
    a[CartesianIndex{2}(2,3)] = 5
    @test a[2,3] == 5
    b = view(a, 1:2, 2:3)
    b[CartesianIndex{2}(1,1)] = 7
    @test a[1,2] == 7
end

@testset "Assignment of singleton array to sparse array (julia #43644)" begin
    K = spzeros(3,3)
    b = zeros(3,3)
    b[3,:] = [1,2,3]
    K[3,1:3] += [1.0 2.0 3.0]'
    @test K == b
    K[3:3,1:3] += zeros(1, 3)
    @test K == b
    K[3,1:3] += zeros(3)
    @test K == b
    K[3,:] += zeros(3,1)
    @test K == b
    @test_throws DimensionMismatch K[3,1:2] += [1.0 2.0 3.0]'
end

# From abstractarray.jl

@testset "julia #17088" begin
    n = 10
    M = rand(n, n)
    @testset "vector of vectors" begin
        v = [[M]; [M]] # using vcat
	@test size(v) == (2,)
        @test !issparse(v)
    end
    @testset "matrix of vectors" begin
        m1 = [[M] [M]] # using hcat
	m2 = [[M] [M];] # using hvcat
        @test m1 == m2
        @test size(m1) == (1,2)
        @test !issparse(m1)
        @test !issparse(m2)
    end
end

@testset "mapslices julia #21123" begin
    @test mapslices(nnz, sparse(1.0I, 3, 3), dims=1) == [1 1 1]
end

@testset "itr, iterate" begin
    r = sparse(2:3:8)
    itr = eachindex(r)
    y = iterate(itr)
    @test y !== nothing
    y = iterate(itr, y[2])
    y = iterate(itr, y[2])
    @test y !== nothing
    val, state = y
    @test r[val] == 8
    @test iterate(itr, state) == nothing
end

# From core.jl

# issue #12960
mutable struct T12960 end
import Base.zero
Base.zero(::Type{T12960}) = T12960()
Base.zero(x::T12960) = T12960()
let
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

# julia issue #12063
# NOTE: should have > MAX_TUPLETYPE_LEN arguments
f12063(tt, g, p, c, b, v, cu::T, d::AbstractArray{T, 2}, ve) where {T} = 1
f12063(args...) = 2
g12063() = f12063(0, 0, 0, 0, 0, 0, 0.0, spzeros(0,0), Int[])
@test g12063() == 1
