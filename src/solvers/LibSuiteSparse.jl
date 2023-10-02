module LibSuiteSparse

using SuiteSparse_jll

const TRUE  = Int32(1)
const FALSE = Int32(0)

include("wrappers.jl")

# exports
const PREFIXES = ["cholmod_", "CHOLMOD_", "umfpack_"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

end # module
