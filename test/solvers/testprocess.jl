# This file is a part of Julia. License is MIT: https://julialang.org/license

# Fresh Julia processes for the tests that need one: a known thread count, a process
# whose garbage collections the test controls, or a SuiteSparse library directory chosen
# before the libraries load. Included by `threads.jl`.

function testprocess(script; threads=1, env=Pair{String,String}[])
    project = Base.active_project()
    projectflag = isnothing(project) ? `` : `--project=$project`
    prelude = """
        using Test, SparseArrays
        @test samefile(pathof(SparseArrays), $(repr(pathof(SparseArrays))))
        @test Threads.nthreads(:default) == $threads
        """
    cmd = `$(Base.julia_cmd()) $projectflag --startup-file=no --depwarn=error --threads=$threads,0 -e $(prelude * script)`
    loadpath = join(Base.load_path(), Sys.iswindows() ? ";" : ":")
    return addenv(cmd, "JULIA_LOAD_PATH" => loadpath, env...)
end

# A child process costs its startup and the compilation of everything it runs, so the
# caller chooses the cases a child runs and passes them through the environment.

# `threads_child.jl` in a child: the threaded tests, with the shared-factor tests for the
# given `(factorization, eltype, index type)` cases.
const THREADS_CASES_ENV = "SPARSEARRAYS_TEST_THREADS_CASES"
threadsprocess(cases; threads) =
    testprocess("include($(repr(joinpath(@__DIR__, "threads_child.jl"))))"; threads,
                env=[THREADS_CASES_ENV => join((join(case, ',') for case in cases), ';')])
# The shared-factor cases of the standard suite: UMFPACK and CHOLMOD once each. A child
# compiles everything it runs from scratch, which is its whole cost, so each case is a
# second or two; the iteration counts are nearly free.
threads_standard_cases() = Any[(lu, Float64, Int), (cholesky, Float64, Int)]

# `cholmod_lifetime.jl` for one index type and one real type.
const LIFETIME_TYPES_ENV = "SPARSEARRAYS_TEST_LIFETIME_TYPES"
lifetimeprocess(Ti, Tv) =
    testprocess("include($(repr(joinpath(@__DIR__, "cholmod_lifetime.jl"))))";
                env=[LIFETIME_TYPES_ENV => "$Ti,$Tv"])
