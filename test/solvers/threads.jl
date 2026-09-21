# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseProcessTests
using Test, SparseArrays

# This file has two roles. Included by the test runner, it starts fresh Julia processes.
# The ones for the threaded tests include this file again with `CHILD_ENV` set, which
# selects the threaded tests themselves, so that they run with a known thread count.
const CHILD_ENV = "SPARSEARRAYS_TEST_THREADS_CHILD"

if haskey(ENV, CHILD_ENV)

using LinearAlgebra, Random

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

else

include("testprocess.jl")

@testset "threaded SuiteSparse tests" begin
    for nt in (1, 4)
        @testset "default threads = $nt" begin
            script = "include($(repr(@__FILE__)))"
            @test success(pipeline(testprocess(script; threads=nt, env=[CHILD_ENV => "1"]); stdout, stderr))
        end
    end
end
@testset "CHOLMOD ownership and lifetime" begin
    script = "include($(repr(joinpath(@__DIR__, "cholmod_lifetime.jl"))))"
    @test success(pipeline(testprocess(script); stdout, stderr))
end

@testset "SuiteSparse library directory override (#250)" begin
    L = SparseArrays.LibSuiteSparse
    original_dir = L.libdir()
    @test isdir(original_dir)
    @test_throws ArgumentError L.set_libdir!(joinpath(original_dir, "does-not-exist"))
    @test samefile(L.libdir(), original_dir)
    mktempdir() do dir
        for name in keys(L.SUITESPARSE_LIBRARIES)
            src = L._jll_path(name)
            cp(src, joinpath(dir, basename(src)); follow_symlinks=true)
        end
        check = """
            using LinearAlgebra, Libdl
            L = SparseArrays.LibSuiteSparse
            @test samefile(L.libdir(), $(repr(dir)))
            A = sparse([4.0 1; 1 3]); b = [1.0, 2.0]; x = Matrix(A) \\ b
            @test cholesky(A) \\ b ≈ x
            @test lu(A) \\ b ≈ x
            @test qr(A) \\ b ≈ x
            for lib in L.SUITESPARSE_LIBRARIES
                @test samefile(dirname(dlpath(lib)), $(repr(dir)))
            end
            @test_throws ArgumentError L.set_libdir!(nothing)
            @test samefile(L.libdir(), $(repr(dir)))
            """
        @test success(pipeline(testprocess(check; env=[L.LIBDIR_ENV => dir]); stdout, stderr))
        script = "SparseArrays.LibSuiteSparse.set_libdir!($(repr(dir)))\n" * check
        @test success(pipeline(testprocess(script); stdout, stderr))
    end
end

end # CHILD_ENV

end # module
