# This file is a part of Julia. License is MIT: https://julialang.org/license
using Test, LinearAlgebra, SparseArrays

include("util/gha.jl")

if Base.get_bool_env("SPARSEARRAYS_AQUA_TEST", false)
    include("ambiguous.jl")
end

for file in readlines(joinpath(@__DIR__, "testgroups"))
    file == "" && continue # skip empty lines
    include(file * ".jl")
end

if Base.USE_GPL_LIBS

    nt = @static if isdefined(Threads, :maxthreadid)
        Threads.maxthreadid()
    else
        Threads.nthreads()
    end

    # 1. If the OS is Windows and we are in GitHub Actions CI, we do NOT run
    #    the `threads` tests.
    # 2. If the OS is Windows and we are NOT in GitHub Actions CI, we DO run
    #    the `threads` tests.
    #        - So, just as an example, if the OS is Windows and we are in
    #          Buildkite CI, we DO run the `threads` tests.
    # 3. If the OS is NOT Windows, we DO run the `threads` tests.
    if Sys.iswindows() && is_github_actions_ci()
        @warn "Skipping `threads` tests on Windows on GitHub Actions CI"
        @test_skip false
    else
        @debug "Beginning the `threads` tests..."

        # Test multithreaded execution
        @testset "threaded SuiteSparse tests" verbose = true begin
            @testset "threads = $nt" begin
                include("threads.jl")
            end
            # test both nthreads==1 and nthreads>1. spawn a process to test whichever
            # case we are not running currently.
            other_nthreads = nt == 1 ? 4 : 1
            @testset "threads = $other_nthreads" begin
                let p, cmd = `$(Base.julia_cmd()) --depwarn=error --startup-file=no threads.jl`
                    p = run(
                            pipeline(
                                setenv(
                                    cmd,
                                    "JULIA_NUM_THREADS" => other_nthreads,
                                    dir=@__DIR__()),
                                stdout = stdout,
                                stderr = stderr),
                            wait = false)
                    if !success(p)
                        error("SuiteSparse threads test failed with nthreads == $other_nthreads")
                    else
                        @test true # mimic the one @test in threads.jl
                    end
                end
            end
        end

        @debug "Finished the `threads` tests..."
    end
end
