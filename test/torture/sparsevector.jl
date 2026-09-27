# This file is a part of Julia. License is MIT: https://julialang.org/license

# Long-tail regression tests for sparse vectors. Each testset names the issue it guards;
# they run only when the `torture` suite is selected.

module TortureSparseVectorTests
using Test
using SparseArrays
using LinearAlgebra
include("../testhelpers.jl")
using SparseArrays: nonzeroinds
using Random

@testset "issue #7507" begin
    @test (i7507=sparsevec(Dict{Int64, Float64}(), 10))==spzeros(10)
end

@testset "issue #8363" begin
    @test_throws ArgumentError sparsevec(Dict(-1=>1,1=>2))
end

@testset "issparse for sparse vectors #34253" begin
    v = sprand(10, 0.5)
    @test issparse(v)
    @test issparse(v')
    @test issparse(transpose(v))
end

@testset "reinterpret (issue #289, pr #296)" begin
    s = spzeros(3)
    r = reinterpret(Int64, s)
    @test r == s

    r[1] = Int64(12)
    @test r[1] === Int64(12)
    @test s[1] === reinterpret(Float64, Int64(12))
    @test r != s

    r[2] = Int64(0)
    @test r[2] === Int64(0)
    @test s[2] === 0.0

    z = reinterpret(Int64, -0.0)
    r[3] = z
    @test r[3] === z
    @test s[3] === -0.0
end

# From Base's arrayops.jl
@testset "copy!" begin
    @testset "AbstractVector" begin
        s = Vector([1, 2])
        for a = ([1], UInt[1], [3, 4, 5], UInt[3, 4, 5])
            @test s === copy!(s, SparseVector(a)) == Vector(a)
        end
    end
end

@testset "Issue #334" begin
    x = sprand(10, .3);
    @test issorted(sort!(x; alg=Base.DEFAULT_STABLE));
    @test_throws MethodError sort!(x; banana=:blue); # From discussion at #335
end

# The long vectors of the hash comparison; the core suite runs lengths 10, 5 and 100.
@testset "hash matches dense, long vectors" begin
    n = 10^5
    v = spzeros(n); v[1] = 1
    w = copy(v); w[2] = 0.0   # explicitly stored zero must not change the hash
    @test hash(w) == hash(v) && isequal(w, v)
    for len in (40000,), x in (sprand(len, 0.1), sprandn(len, 0.3), spzeros(len))
        k = min(3, nnz(x)); nonzeros(x)[1:k] .= [NaN, -0.0, 0.0][1:k]
        @test hash(x) == hash(Vector(x))
        @test hash(x, UInt(7)) == hash(Vector(x), UInt(7))
    end
end

# The start/stop sweep of `reverse` over a random length-20 vector; the core suite
# keeps the fixed-pattern vectors.
@testset "reverse" begin
    @testset "$name" for (name, s) in (("random", sprand(Float32, 20, 0.4)),)
        w = collect(s)
        @testset for start in axes(s,1), stop in start:lastindex(s,1)
            srev = reverse(s, start, stop)
            @test nnz(srev) == nnz(s)
            @test srev == reverse(w, start, stop)
        end
    end
end

rnd_x0 = sprand(50, 0.6)
rnd_x0f = Array(rnd_x0)

rnd_x1 = sprand(50, 0.7) * 4.0
rnd_x1f = Array(rnd_x1)

# The math functions that the core testsets of the same names do not run.
@testset "Zero-preserving math functions: sparse -> sparse" begin
    x1operations = (ceil, trunc)
    x0operations = (expm1,  sinpi,
                    tan,    sind,   tand,
                    asin,   atan,   asind,  atand,
                    sinh,   tanh,   asinh)

    for (spvec, densevec, operations) in (
            (rnd_x0, rnd_x0f, x0operations),
            (rnd_x1, rnd_x1f, x1operations) )
        for op in operations
            spresvec = op.(spvec)
            @test spresvec == op.(densevec)
            @test all(!iszero, nonzeros(spresvec))
            resvaltype = typeof(op(zero(eltype(spvec))))
            resindtype = SparseArrays.indtype(spvec)
            @test isa(spresvec, SparseVector{resvaltype,resindtype})
        end
    end
end
@testset "Non-zero-preserving math functions: sparse -> dense" begin
    for op in (exp2, exp10, log2, log10,
            cosd, acos, cosh, cospi,
            csc, cscd, acot, csch, acsch,
            cot, cotd, acosd, coth,
            secd, acotd, sech, asech)
        spvec = rnd_x0
        densevec = rnd_x0f
        spresvec = op.(spvec)
        @test spresvec == op.(densevec)
        resvaltype = typeof(op(zero(eltype(spvec))))
        resindtype = SparseArrays.indtype(spvec)
        @test isa(spresvec, SparseVector{resvaltype,resindtype})
    end
end

# The full coefficient and wrapper grids of the BLAS Level-2 products; the core suite
# zips the coefficients and keeps a subset of the wrappers.
@testset "BLAS Level-2, full coefficient and wrapper grids" begin
    @testset "dense A * sparse x -> dense y" begin
        for TA in (Float64, ComplexF64), Tx in (Float64, ComplexF64)
            T = Base.promote_op(LinearAlgebra.matprod, TA, Tx)
            let A = randn(TA, 9, 16), x = sprand(Tx, 16, 0.7)
                xf = Array(x)
                for α in [0.0, 1.0, 2.0], β in [0.0, 0.5, 1.0]
                    y = rand(T, 9)
                    rr = α*A*xf + β*y
                    @test mul!(y, A, x, α, β) === y
                    @test y ≈ rr
                end
                y = A*x
                @test isa(y, Vector{T})
                @test A*x ≈ A*xf
            end

            let A = randn(TA, 16, 9), x = sprand(Tx, 16, 0.7)
                xf = Array(x)
                for α in [0.0, 1.0, 2.0], β in [0.0, 0.5, 1.0]
                    y = rand(T, 9)
                    rr = α*transpose(A)*xf + β*y
                    @test mul!(y, transpose(A), x, α, β) === y
                    @test y ≈ rr
                end
                y = *(transpose(A), x)
                @test isa(y, Vector{T})
                @test y ≈ *(transpose(A), xf)
            end

            let A = randn(TA, 16, 9), x = sprand(Tx, 16, 0.7)
                xf = Array(x)
                for α in [0.0, 1.0, 2.0], β in [0.0, 0.5, 1.0]
                    y = rand(T, 9)
                    rr = α*A'xf + β*y
                    @test mul!(y, adjoint(A), x, α, β) === y
                    @test y ≈ rr
                end
                y = *(adjoint(A), x)
                @test isa(y, Vector{T})
                @test y ≈ *(adjoint(A), xf)
            end

            let A = randn(TA, 16, 16), x = sprand(Tx, 16, 0.7)
                xf = Array(x)
                for wrap in (M -> Symmetric(M, :U), M -> Symmetric(M, :L),
                        M -> Hermitian(M, :U), M -> Hermitian(M, :L))
                    for α in (0.0, 1.0, 2.0), β in (0.0, 0.5, 1.0)
                        y = rand(T, 16)
                        rr = α*wrap(A)*xf + β*y
                        @test mul!(y, wrap(A), x, α, β) === y
                        @test y ≈ rr
                    end
                    y = *(wrap(A), x)
                    @test isa(y, Vector{T})
                    @test y ≈ *(wrap(A), xf)
                end
            end
        end
    end
    @testset "sparse A * sparse x -> dense y" begin
        let A = sprandn(9, 16, 0.5), x = sprand(16, 0.7)
            Af = Array(A)
            xf = Array(x)
            for α in [0.0, 1.0, 2.0], β in [0.0, 0.5, 1.0]
                y = rand(9)
                rr = α*Af*xf + β*y
                @test mul!(y, A, x, α, β) === y
                @test y ≈ rr
            end
            y = SparseArrays.densemv(A, x)
            @test isa(y, Vector{Float64})
            @test y ≈ Af*xf
        end

        let A = sprandn(16, 9, 0.5), x = sprand(16, 0.7)
            Af = Array(A)
            xf = Array(x)
            for α in [0.0, 1.0, 2.0], β in [0.0, 0.5, 1.0]
                y = rand(9)
                rr = α*Af'xf + β*y
                @test mul!(y, transpose(A), x, α, β) === y
                @test y ≈ rr
            end
            y = SparseArrays.densemv(A, x; trans='T')
            @test isa(y, Vector{Float64})
            @test y ≈ *(transpose(Af), xf)

            A32 = SparseMatrixCSC{Float64,Int32}(A)
            @test mul!(zeros(9), transpose(A32), x) ≈ transpose(Af) * xf
        end

        let A = sprandn(16, 16, 0.5), x = sprand(16, 0.7)
            Af = Array(A)
            xf = Array(x)
            for wrap in (M -> Symmetric(M, :U), M -> Symmetric(M, :L),
                M -> Hermitian(M, :U), M -> Hermitian(M, :L),
                M -> UpperTriangular(M), M -> UnitUpperTriangular(M),
                M -> LowerTriangular(M), M -> UnitLowerTriangular(M),
                M -> UpperTriangular(transpose(M)), M -> UnitUpperTriangular(transpose(M)),
                M -> LowerTriangular(transpose(M)), M -> UnitLowerTriangular(transpose(M)),
                M -> UpperTriangular(adjoint(M)), M -> UnitUpperTriangular(adjoint(M)),
                M -> LowerTriangular(adjoint(M)), M -> UnitLowerTriangular(adjoint(M)),
                M -> UpperTriangular(Symmetric(M)))
                for α in (0.0, 1.0, 2.0), β in (0.0, 0.5, 1.0)
                    y = rand(16)
                    rr = α*wrap(Af)*xf + β*y
                    @test mul!(y, wrap(A), x, α, β) === y
                    @test y ≈ rr
                end
                y = wrap(A) * x
                @test y ≈ *(wrap(Af), xf)
            end
        end

        let A = complex.(sprandn(7, 8, 0.5), sprandn(7, 8, 0.5)),
            x = complex.(sprandn(8, 0.6), sprandn(8, 0.6)),
            x2 = complex.(sprandn(7, 0.75), sprandn(7, 0.75))
            Af = Array(A)
            xf = Array(x)
            x2f = Array(x2)
            @test SparseArrays.densemv(A, x; trans='N') ≈ Af * xf
            @test SparseArrays.densemv(A, x2; trans='T') ≈ transpose(Af) * x2f
            @test SparseArrays.densemv(A, x2; trans='C') ≈ Af'x2f
            @test_throws ArgumentError SparseArrays.densemv(A, x; trans='D')
        end

        let A = sparse(bitrand(9, 16)), x = sparse(bitrand(16))
            Af = Array(A)
            xf = Array(x)
            y = SparseArrays.densemv(A, x)
            @test isa(y, Vector{Int})
            @test y == Af*xf
        end
    end
    @testset "sparse A * sparse x -> sparse y" begin
        let A = sprandn(9, 16, 0.5), x = sprand(16, 0.7), x2 = sprand(9, 0.7)
            Af = Array(A)
            xf = Array(x)
            x2f = Array(x2)

            y = A*x
            @test isa(y, SparseVector{Float64,Int})
            @test all(nonzeros(y) .!= 0.0)
            @test Array(y) ≈ Af * xf

            y = *(transpose(A), x2)
            @test isa(y, SparseVector{Float64,Int})
            @test all(nonzeros(y) .!= 0.0)
            @test Array(y) ≈ Af'x2f
        end

        let A = complex.(sprandn(7, 8, 0.5), sprandn(7, 8, 0.5)),
            x = complex.(sprandn(8, 0.6), sprandn(8, 0.6)),
            x2 = complex.(sprandn(7, 0.75), sprandn(7, 0.75))
            Af = Array(A)
            xf = Array(x)
            x2f = Array(x2)

            y = A*x
            @test isa(y, SparseVector{ComplexF64,Int})
            @test Array(y) ≈ Af * xf

            y = *(transpose(A), x2)
            @test isa(y, SparseVector{ComplexF64,Int})
            @test Array(y) ≈ transpose(Af) * x2f

            y = *(adjoint(A), x2)
            @test isa(y, SparseVector{ComplexF64,Int})
            @test Array(y) ≈ Af'x2f

            A32 = SparseMatrixCSC{ComplexF64,Int32}(A)
            for x32 in (x2, SparseVector{ComplexF64,Int32}(x2)), op in (transpose, adjoint)
                y = op(A32) * x32
                @test isa(y, SparseVector{ComplexF64,promote_type(Int32, eltype(nonzeroinds(x32)))})
                @test Array(y) ≈ op(Af) * x2f
            end
        end

        let A = sparse(bitrand(9, 16)), x = sparse(bitrand(16)), x2 = sparse(bitrand(9))
            Af = Array(A)
            xf = Array(x)
            x2f = Array(x2)

            y = A*x
            @test isa(y, SparseVector{Int, Int})
            @test Array(y) == Af*xf

            y = A'*x2
            @test isa(y, SparseVector{Int, Int})
            @test Array(y) == Af'x2f
        end
    end
    @testset "sparse A * dense x -> dense y" begin
        let A = sparse(bitrand(9, 16)), x = Vector(bitrand(16)), x2 = Vector(bitrand(9))
            Af = Array(A)
            xf = Array(x)
            x2f = Array(x2)

            y = A*x
            @test isa(y, Vector{Int})
            @test y == Af*xf

            y = A'*x2
            @test isa(y, Vector{Int})
            @test y == Af'x2f
        end
    end
end

# The longer vectors of the `dropzeros` check; the core suite runs length 10.
@testset "dropzeros[!] with length=$m" for m in (20, 30)
    Random.seed!(123)
    nzprob, targetnumposzeros, targetnumnegzeros = 0.4, 5, 5
    v = sprand(m, nzprob)
    struczerosv = findall(x -> x == 0, v)
    poszerosinds = unique(rand(struczerosv, targetnumposzeros))
    negzerosinds = unique(rand(struczerosv, targetnumnegzeros))
    vposzeros = copy(v)
    vposzeros[poszerosinds] .= 2
    vnegzeros = copy(v)
    vnegzeros[negzerosinds] .= -2
    vbothsigns = copy(vposzeros)
    vbothsigns[negzerosinds] .= -2
    map!(x -> x == 2 ? 0.0 : x, nonzeros(vposzeros), nonzeros(vposzeros))
    map!(x -> x == -2 ? -0.0 : x, nonzeros(vnegzeros), nonzeros(vnegzeros))
    map!(x -> x == 2 ? 0.0 : x == -2 ? -0.0 : x, nonzeros(vbothsigns), nonzeros(vbothsigns))
    for vwithzeros in (vposzeros, vnegzeros, vbothsigns)
        # Basic functionality / dropzeros!
        @test dropzeros!(copy(vwithzeros)) == v
        # Basic functionality / dropzeros
        @test dropzeros(vwithzeros) == v
        # Check trimming works as expected
        @test length(nonzeros(dropzeros!(copy(vwithzeros)))) == length(nonzeros(v))
        @test length(nonzeroinds(dropzeros!(copy(vwithzeros)))) == length(nonzeroinds(v))
    end
end

@testset "Issue 14013" begin
    s14013 = sparse([10.0 0.0 30.0; 0.0 1.0 0.0])
    a14013 = [10.0 0.0 30.0; 0.0 1.0 0.0]
    @test s14013 == a14013
    @test vec(s14013) == s14013[:] == a14013[:]
    @test Array(s14013)[1,:] == s14013[1,:] == a14013[1,:] == [10.0, 0.0, 30.0]
    @test Array(s14013)[2,:] == s14013[2,:] == a14013[2,:] == [0.0, 1.0, 0.0]
end

@testset "Issue 14046" begin
    s14046 = sprand(5, 1.0)
    @test spzeros(5) + s14046 == s14046
    @test 2*s14046 == s14046 + s14046
end

# The eltype and index-type pairs of `fill!` that the core suite does not run.
@testset "fill!" begin
    for Tv in [Float32, Float64, Int64, Int32, ComplexF64]
        for Ti in [Int16, Int32, Int64, BigInt]
            Tv in (Float32, ComplexF64) && Ti in (Int16, Int64) && continue
            sptypes = (SparseMatrixCSC{Tv, Ti}, SparseVector{Tv, Ti})
            sizes = [(3, 4), (3,)]
            for (siz, Sp) in zip(sizes, sptypes)
                arr = rand(Tv, siz...)
                sparr = Sp(arr)
                x = rand(Tv)
                @test fill!(sparr, x) == fill(x, siz)
                @test fill!(sparr, 0) == fill(0, siz)
            end
        end
    end
end

# Every fifth column of the column-view comparison; the core suite checks three.
@testset "Fast operations on full column views" begin
    n = 1000
    A = sprandn(n, n, 0.01)
    for j in 1:5:n
        Aj, Ajview = A[:, j], view(A, :, j)
        @test norm(Aj)          == norm(Ajview)
        @test dot(Aj, copy(Aj)) == dot(Ajview, Aj) # don't alias since it takes a different code path
        @test rmul!(Aj, 0.1)    == rmul!(Ajview, 0.1)
        @test Aj*0.1            == Ajview*0.1
        @test 0.1*Aj            == 0.1*Ajview
        @test Aj/0.1            == Ajview/0.1
        @test LinearAlgebra.axpy!(1.0, Aj,     sparse(fill(1., n))) ==
              LinearAlgebra.axpy!(1.0, Ajview, sparse(fill(1., n)))
        @test LinearAlgebra.lowrankupdate!(Matrix(1.0*I, n, n), fill(1.0, n), Aj) ==
              LinearAlgebra.lowrankupdate!(Matrix(1.0*I, n, n), fill(1.0, n), Ajview)
    end
end

end # module
