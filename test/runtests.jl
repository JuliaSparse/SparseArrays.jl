# This file is a part of Julia. License is MIT: https://julialang.org/license

for file in readlines(joinpath(@__DIR__, "testgroups"))
    include(file * ".jl")
end
include("basetests.jl")
if Base.USE_GPL_LIBS
    include("solvers/ambiguous.jl")
    include("solvers/umfpack.jl")
    include("solvers/cholmod.jl")
    include("solvers/spqr.jl")
    include("solvers/linalg_tests.jl")

    # Test multithreaded execution
    @testset "threaded SuiteSparse tests" verbose = true begin
        @testset "threads = $(Threads.nthreads())" begin
            include("solvers/threads.jl")
        end
        # test both nthreads==1 and nthreads>1. spawn a process to test whichever
        # case we are not running currently.
        other_nthreads = Threads.nthreads() == 1 ? 4 : 1
        @testset "threads = $other_nthreads" begin
            let p, cmd = `$(Base.julia_cmd()) --depwarn=error --startup-file=no solvers/threads.jl`
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
end
