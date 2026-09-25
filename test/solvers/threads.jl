# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseProcessTests
using Test, SparseArrays

# Tests that need a fresh Julia process.

include("testprocess.jl")

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

end # module
