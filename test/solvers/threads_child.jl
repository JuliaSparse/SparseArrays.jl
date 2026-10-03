# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseThreadsChildTests
# The threaded solver tests. Not a suite: `threads.jl` runs this file in a child process
# with a known thread count, and names the shared-factor cases in the environment. The
# child compiles everything it runs from scratch, so the file loads nothing it does not
# need, the shared test helpers included.
using Test, SparseArrays, LinearAlgebra, Random
Base.Experimental.@compiler_options optimize=0

Random.seed!(1234)

A = sprandn(120, 120, 0.2)
b = rand(120)

function test(n::Integer)
    _A = A[1:n, 1:n]
    _b = b[1:n]
    x = qr(_A) \ _b
    return norm(x)
end

res_threads = zeros(100)
Threads.@threads for i in eachindex(res_threads)
    res_threads[i] = test(i + 20)
end

@test res_threads ≈ [test(i + 20) for i in eachindex(res_threads)]

cases = map(split(get(ENV, "SPARSEARRAYS_TEST_THREADS_CASES", ""), ';'; keepempty=false)) do case
    factorize, T, Ti = split(case, ',')
    (getfield(LinearAlgebra, Symbol(factorize)), getfield(Base, Symbol(T)), getfield(Base, Symbol(Ti)))
end

@testset "shared $factorize factor, $T, $Ti" for (factorize, T, Ti) in cases
    n = 12
    offdiag = T <: Real ? one(T) : T(1 + im)
    S = SparseMatrixCSC{T,Ti}(spdiagm(-1 => fill(offdiag, n - 1),
        0 => fill(T(4), n), 1 => fill(conj(offdiag), n - 1)))
    F = factorize(S)
    b = T.(1:n) .* offdiag
    for rhs in (b, hcat(b, reverse(b)))
        inputs = [rhs .* i .+ (i % 3) for i in 1:30]
        expected = [F \ input for input in inputs]
        outputs = similar.(expected)
        Threads.@threads :static for i in eachindex(outputs)
            ldiv!(outputs[i], F, inputs[i])
        end
        @test all(i -> outputs[i] ≈ expected[i], eachindex(outputs))
    end
    if factorize === lu
        G = lu!(copy(F))
        @test G \ b ≈ Matrix(S) \ b
    end
end

end # module
