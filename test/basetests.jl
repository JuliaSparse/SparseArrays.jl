using Random, LinearAlgebra, SparseArrays

@testset "copy!" begin
    @testset "AbstractVector" begin
        s = Vector([1, 2])
        for a = ([1], UInt[1], [3, 4, 5], UInt[3, 4, 5])
            @test s === copy!(s, SparseVector(a)) == Vector(a)
        end
end

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
