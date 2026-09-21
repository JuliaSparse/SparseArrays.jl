# This file is a part of Julia. License is MIT: https://julialang.org/license

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
