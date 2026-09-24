# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseLinalgSolversTests
using Test

using SparseArrays
using Random
using LinearAlgebra
using Serialization
using SparseArrays: FactorLock, @_readlock, @_writelock, _rdlock, _rdunlock,
    _wrlock, _wrunlock, _nreaders, _haswriter, _nwaiting

@testset "explicit zeros" begin
    a = SparseMatrixCSC(2, 2, [1, 3, 5], [1, 2, 1, 2], [1.0, 0.0, 0.0, 1.0])
    @test lu(a)\[2.0, 3.0] ≈ [2.0, 3.0]
    @test cholesky(a)\[2.0, 3.0] ≈ [2.0, 3.0]
end

@testset "complex left-division" begin
    for i = 1:5
        a = I + 0.1*sprandn(5, 5, 0.2)
        b = randn(5,3) + im*randn(5,3)
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())

        a = I + 0.1*sprandn(5, 5, 0.2) + 0.1*im*sprandn(5, 5, 0.2)
        b = randn(5,3)
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())
    end
end

@testset "sparse matrix cond" begin
    Random.seed!(1235)
    local A = sparse(reshape([1.0], 1, 1))
    @test cond(A, 1) == 1.0
    @test_throws ArgumentError cond(A,2)
    @test_throws ArgumentError cond(A,3)
    Arect = spzeros(10, 6)
    @test_throws DimensionMismatch cond(Arect, 1)
    @test_throws ArgumentError cond(Arect,2)
    @test_throws DimensionMismatch cond(Arect, Inf)
    Ac = sprandn(20, 20,.5) + im*sprandn(20, 20,.5)
    Ar = sprandn(20, 20,.5) + eps()*I
    # For a discussion of the tolerance, see #14778
    @test 0.99 <= cond(Ar, 1) \ opnorm(Ar, 1) * opnorm(inv(Array(Ar)), 1) < 3
    @test 0.99 <= cond(Ac, 1) \ opnorm(Ac, 1) * opnorm(inv(Array(Ac)), 1) < 3
    @test 0.99 <= cond(Ar, Inf) \ opnorm(Ar, Inf) * opnorm(inv(Array(Ar)), Inf) < 3
    @test 0.99 <= cond(Ac, Inf) \ opnorm(Ac, Inf) * opnorm(inv(Array(Ac)), Inf) < 3
    #issue 680
    A22 = sparse(randn(2,2))
    @test 0.99 ≤ cond(Array(A22), 1) / cond(A22, 1) < 3
    @test 0.99 ≤ cond(Array(A22), Inf) / cond(A22, Inf) < 3
end

@testset "sparse matrix opnormestinv" begin
    Random.seed!(1235)
    Ac = sprandn(20,20,.5) + im* sprandn(20,20,.5)
    Aci = ceil.(Int64, 100*sprand(20,20,.5)) + im*ceil.(Int64, sprand(20,20,.5))
    Ar = sprandn(20,20,.5)
    Ari = ceil.(Int64, 100*Ar)
    # NOTE: opnormestinv is probabilistic, so requires a fixed seed (set above in Random.seed!(1234))
    @test SparseArrays.opnormestinv(Ac,3) ≈ opnorm(inv(Array(Ac)),1) atol=1e-4
    @test SparseArrays.opnormestinv(Aci,3) ≈ opnorm(inv(Array(Aci)),1) atol=1e-4
    @test SparseArrays.opnormestinv(Ar) ≈ opnorm(inv(Array(Ar)),1) atol=1e-4
    @test_throws ArgumentError SparseArrays.opnormestinv(Ac,0)
    @test_throws ArgumentError SparseArrays.opnormestinv(Ac,21)
    @test_throws DimensionMismatch SparseArrays.opnormestinv(sprand(3,5,.9))
    #issue 680
    A33 = sparse(randn(3,3))
    @test SparseArrays.opnormestinv(A33,3) ≈ opnorm(inv(Array(A33)),1) atol=1e-4
end

@testset "factorization" begin
    Random.seed!(123)
    local A
    A = sparse(Diagonal(rand(5))) + sprandn(5, 5, 0.2) + im*sprandn(5, 5, 0.2)
    A = A + copy(A')
    @test abs(det(factorize(Hermitian(A)))) ≈ abs(det(factorize(Array(A))))
    A = sparse(Diagonal(rand(5))) + sprandn(5, 5, 0.2) + im*sprandn(5, 5, 0.2)
    A = A*A'
    @test abs(det(factorize(Hermitian(A)))) ≈ abs(det(factorize(Array(A))))
    A = sparse(Diagonal(rand(5))) + sprandn(5, 5, 0.2)
    A = A + copy(transpose(A))
    @test abs(det(factorize(Symmetric(A)))) ≈ abs(det(factorize(Array(A))))
    A = sparse(Diagonal(rand(5))) + sprandn(5, 5, 0.2)
    A = A*transpose(A)
    @test abs(det(factorize(Symmetric(A)))) ≈ abs(det(factorize(Array(A))))
    C, b = A[:, 1:4], fill(1., size(A, 1))
    @test factorize(C)\b ≈ Array(C)\b
end

@testset "type stability of linear solve" begin
    for relty in (Float16, Float32, Float64), elty in (relty, Complex{relty})
        A = sprand(elty, 2, 2, 1.0)
        B = randn(elty, 2, 2)
        b = randn(elty, 2)
        @inferred A \ b
        @inferred A \ B
    end
end

@testset "factorization of a fixed-pattern matrix" begin
    b = sprandn(10, 10, 0.99) + I
    a = SparseArrays.fixed(b)

    @test (lu(a) \ randn(10); true)
    @test b == a
    @test (qr(a + a') \ randn(10); true)
    @test b == a
end

@testset "FactorLock" begin
    # Cooperative tasks on this thread: a task that has started and is not done while this
    # task runs is blocked on the lock.
    # timedwait only guards against hangs
    poll(f) = timedwait(f, 60; pollint=0.001) === :ok
    blocked(t) = poll(() -> istaskstarted(t)) && !istaskdone(t)
    finishes(t) = poll(() -> istaskdone(t))

    @testset "reentrancy" begin
        l = FactorLock()
        r = @_writelock l begin
            @_writelock l begin
                @_readlock l begin
                    @test _haswriter(l) && _nreaders(l) == 1
                    @_readlock l 42
                end
            end
        end
        @test r == 42 && !_haswriter(l) && _nreaders(l) == 0
        @_readlock l @_readlock l @test _nreaders(l) == 2
        @test _nreaders(l) == 0
    end

    @testset "upgrade throws" begin
        l = FactorLock()
        @_readlock l begin
            @test_throws ConcurrencyViolationError @_writelock l nothing
            @test _nreaders(l) == 1 && !_haswriter(l) && _nwaiting(l) == 0
        end
        @test _nreaders(l) == 0
    end

    @testset "released on exceptions" begin
        l = FactorLock()
        @test_throws ErrorException @_writelock l error("boom")
        @test_throws ErrorException @_readlock l error("boom")
        @test !_haswriter(l) && _nreaders(l) == 0
        @test (@_writelock l 1) == 1
    end

    @testset "release by a task that does not hold the lock" begin
        l = FactorLock()
        @test_throws ConcurrencyViolationError _rdunlock(l)
        @test_throws ConcurrencyViolationError _wrunlock(l)
        _wrlock(l)
        @test fetch(@async try _wrunlock(l) catch e; e end) isa ConcurrencyViolationError
        @test _haswriter(l)
        _wrunlock(l)
        _rdlock(l)
        @test fetch(@async try _rdunlock(l) catch e; e end) isa ConcurrencyViolationError
        @test _nreaders(l) == 1
        _rdunlock(l)
    end

    @testset "shared reads, exclusive writes" begin
        l = FactorLock()
        _rdlock(l)
        @test fetch(@async @_readlock l _nreaders(l)) == 2
        w = @async @_writelock l _haswriter(l)
        @test poll(() -> _nwaiting(l) == 1)
        @test blocked(w)
        _rdunlock(l)
        @test finishes(w) && fetch(w)
        _wrlock(l)
        r = @async @_readlock l _nreaders(l)
        @test blocked(r)
        _wrunlock(l)
        @test finishes(r) && fetch(r) == 1
    end

    @testset "writer preference and ordering" begin
        l = FactorLock()
        order = Symbol[]
        _rdlock(l)
        w = @async @_writelock l push!(order, :w)
        @test poll(() -> _nwaiting(l) == 1)
        r = @async @_readlock l push!(order, :r)
        @test blocked(r)
        # a task already holding a read lock does not queue behind the writer
        @test (@_readlock l _nreaders(l)) == 2
        _rdunlock(l)
        @test finishes(w) && finishes(r)
        @test order == [:w, :r]
        @test _nreaders(l) == 0 && !_haswriter(l) && _nwaiting(l) == 0
    end

    @testset "interrupting a waiting writer releases the readers behind it" begin
        l = FactorLock()
        _rdlock(l)
        w = @async @_writelock l :w
        @test poll(() -> _nwaiting(l) == 1)
        r = @async @_readlock l :r
        @test blocked(r)
        schedule(w, InterruptException(); error=true)
        @test finishes(w) && istaskfailed(w)
        @test finishes(r) && fetch(r) === :r
        @test _nwaiting(l) == 0 && !_haswriter(l)
        _rdunlock(l)
        @test _nreaders(l) == 0
    end

    @testset "serialization gives a fresh lock" begin
        l = FactorLock()
        _wrlock(l)
        io = IOBuffer()
        serialize(io, (1, l))
        _, l2 = deserialize(seekstart(io))
        @test l2 isa FactorLock && l2 !== l
        @test !_haswriter(l2) && _nreaders(l2) == 0 && _nwaiting(l2) == 0
        @test _haswriter(l)
        _wrunlock(l)
    end

    @testset "ReentrantLock is exclusive in both modes" begin
        rl = ReentrantLock()
        @test (@_writelock rl @_readlock rl islocked(rl))
        @test !islocked(rl)
        @test_throws ErrorException @_readlock rl error("boom")
        @test !islocked(rl)
    end

    @testset "uncontended locking does not allocate" begin
        l = FactorLock()
        rd(l) = @_readlock l 1
        wr(l) = @_writelock l 1
        rd(l); wr(l)
        @test (@allocated rd(l)) == 0
        @test (@allocated wr(l)) == 0
    end
end

end # module
