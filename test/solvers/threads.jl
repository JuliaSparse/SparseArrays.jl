# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseProcessTests
using Test, SparseArrays, LinearAlgebra
include("../testhelpers.jl")

include("testprocess.jl")

# A child process compiles everything it runs from scratch, so the standard suite starts
# one: `threads_child.jl` with four threads and one shared `lu` and `cholesky` factor.
@testset "threaded SuiteSparse tests" begin
    others = @static COMPREHENSIVE ? setdiff(pairwise((lu, cholesky, qr), (Float64, ComplexF64), itypes), threads_standard_cases()) : []
    for nt in (@static COMPREHENSIVE ? (1, 4) : (4,))
        @testset "default threads = $nt" begin
            cases = nt == 1 ? others[1:2:end] : [threads_standard_cases(); others[2:2:end]]
            @test success(pipeline(threadsprocess(cases; threads=nt); stdout, stderr))
        end
    end
end
@static if COMPREHENSIVE
@testset "CHOLMOD ownership and lifetime, $Ti, $Tv" for (Ti, Tv) in ((first(itypes), Float64), (last(itypes), Float32))
    @test success(pipeline(lifetimeprocess(Ti, Tv); stdout, stderr))
end
end

@testset "SuiteSparse library directory override (#250)" begin
    L = SparseArrays.LibSuiteSparse
    original_dir = L.libdir()
    @test isdir(original_dir)
    @test_throws ArgumentError L.set_libdir!(joinpath(original_dir, "does-not-exist"))
    @test samefile(L.libdir(), original_dir)
    @static if COMPREHENSIVE
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
end

end # module
