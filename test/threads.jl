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
    S = SparseMatrixCSC{T,Ti}(sparse(T[4 1 0; 1 4 1; 0 1 4]))
    F = factorize(S)
    for b in (T[1, 2, 3], T[1 2; 2 3; 3 4])
        expected = F \ b
        outputs = [similar(expected) for _ in 1:30]
        Threads.@threads for i in eachindex(outputs)
            ldiv!(outputs[i], F, b)
        end
        @test all(x -> x ≈ expected, outputs)
    end
    if factorize === lu
        G = lu!(copy(F))
        @test G \ ones(T, 3) ≈ Matrix(S) \ ones(T, 3)
    end
end
