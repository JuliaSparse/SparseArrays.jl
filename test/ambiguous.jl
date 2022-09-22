# This file is a part of Julia. License is MIT: https://julialang.org/license

using Test, SparseArrays

if "exec" in ARGS
    let ambig = detect_ambiguities(SparseArrays; recursive=true)
        @test isempty(ambig)
        ambig = Set{Any}(((m1.sig, m2.sig) for (m1, m2) in ambig))
        expect = []
        good = true
        while !isempty(ambig)
            sigs = pop!(ambig)
            i = findfirst(==(sigs), expect)
            if i === nothing
                println(stderr, "push!(expect, (", sigs[1], ", ", sigs[2], "))")
                good = false
                continue
            end
            deleteat!(expect, i)
        end
        @test isempty(expect)
        @test good
    end
else
    @testset "method ambiguity" begin
        # Ambiguity test is run inside a clean process.
        # https://github.com/JuliaLang/julia/issues/28804
        script = @__FILE__
        cmd = `$(Base.julia_cmd()) --startup-file=no $script exec`
        @test success(pipeline(cmd; stdout=stdout, stderr=stderr))
    end
end
