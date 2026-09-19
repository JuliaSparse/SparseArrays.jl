# This file is a part of Julia. License is MIT: https://julialang.org/license

using Test, LinearAlgebra, SparseArrays, Random

Random.seed!(1234)

A = sprandn(200, 200, 0.2)
b = rand(200)

function test(n::Integer)
    _A = A[1:n, 1:n]
    _b = b[1:n]
    x = qr(_A) \ _b
    return norm(x)
end

res_threads = zeros(100)
Threads.@threads for i in 1:100
    res_threads[i] = test(i + 100)
end

@test res_threads ≈ [test(i + 100) for i in 1:100]

@testset "shared $factorize factor, $T, $Ti" for factorize in (lu, cholesky, qr),
    T in (Float64, ComplexF64), Ti in (sizeof(Int) == 4 ? (Int32,) : (Int32, Int64))
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
