# This file is a part of Julia. License is MIT: https://julialang.org/license

# Whether inference gives some local or SSA value of `f(::types...)` a `Union` that
# contains both `T1` and `T2`. A counter seeded from an `Int32` index array and then
# incremented with an `Int` literal is the usual way such a union appears; the public
# call still infers, so `@inferred` cannot see it.
function hasunionlocal(f, types, T1, T2)
    ci, _ = only(Base.code_typed(f, types; optimize=false))
    slots = ci.slottypes === nothing ? Any[] : ci.slottypes
    any(T -> T isa Union && T1 <: T && T2 <: T, Iterators.flatten((slots, ci.ssavaluetypes)))
end
