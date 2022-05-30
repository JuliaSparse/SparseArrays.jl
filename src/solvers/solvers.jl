# This file is a part of Julia. License is MIT: https://julialang.org/license

module Solvers

include("LibSuiteSparse.jl")
using .LibSuiteSparse

if Base.USE_GPL_LIBS
    include("umfpack.jl")
    include("cholmod.jl")
    include("spqr.jl")
end

end # module Solvers
