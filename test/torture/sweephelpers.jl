# This file is a part of Julia. License is MIT: https://julialang.org/license

# The seeded sparse-versus-dense sweep machinery for `sweeps.jl`. A plain definition
# file: `sweeps.jl` includes it after `../testhelpers.jl`.
#
# A sweep draws `SWEEP_CASES` random cases from `SWEEP_SEED`, and for each one builds a
# sparse operand together with its dense twin in the same form (plain, column view, fixed
# pattern, transpose or adjoint), applies one kernel to both with identical random draws,
# and compares the two outcomes. Every case runs in a testset whose name carries the seed,
# the case index and the case description, so a failure line is its own reproducer:
# `reproduce(name, seed, i)` rebuilds that case and returns both operands and results.

using Random: Xoshiro, randperm

const SWEEP_SEED = parse(Int, get(ENV, "SPARSEARRAYS_SWEEP_SEED", "1"))
const SWEEP_CASES = parse(Int, get(ENV, "SPARSEARRAYS_SWEEP_CASES", "100"))

caserng(seed, i) = Xoshiro(hash((seed, i)))

const SWEEP_SHAPES = (0, 1, 2, 3, 7, 16, 33)
const SWEEP_DENSITIES = (0.0, 0.05, 0.3, 1.0)
const SWEEP_ELTYPES = (Float64, ComplexF64, Int, Float32)
const SWEEP_FORMS = (:plain, :colview, :fixed, :transpose, :adjoint)
const SWEEP_INDEXTYPES = (Int, Int32)

struct SweepCase
    m::Int
    n::Int
    density::Float64
    T::DataType
    storedzeros::Int
    form::Symbol
    Ti::DataType
end

gencase(rng) = SweepCase(rand(rng, SWEEP_SHAPES), rand(rng, SWEEP_SHAPES),
                         rand(rng, SWEEP_DENSITIES), rand(rng, SWEEP_ELTYPES),
                         rand(rng, 0:3), rand(rng, SWEEP_FORMS), rand(rng, SWEEP_INDEXTYPES))

describe(c::SweepCase) =
    "$(c.m)x$(c.n) $(c.T) $(c.Ti) density=$(c.density) storedzeros=$(c.storedzeros) form=$(c.form)"

# `k` nonzero values of eltype `T`
sweepvalues(rng, ::Type{T}, k) where {T<:AbstractFloat} = T[x + copysign(T(0.5), x) for x in randn(rng, T, k)]
sweepvalues(rng, ::Type{Complex{T}}, k) where {T} = complex.(sweepvalues(rng, T, k), sweepvalues(rng, T, k))
sweepvalues(rng, ::Type{Int}, k) = rand(rng, (-9:-1) ∪ (1:9), k)

# A random `m`x`n` sparse matrix of the given density and eltype with `storedzeros`
# entries stored as explicit zeros. A stored zero may land on a stored nonzero.
function sweepmatrix(rng, ::Type{T}, ::Type{Ti}, m, n, density, storedzeros) where {T,Ti}
    S = SparseMatrixCSC{T,Ti}(sprand(rng, m, n, density, (r, k) -> sweepvalues(r, T, k), T))
    for _ in 1:min(storedzeros, m * n)
        i, j = rand(rng, 1:m), rand(rng, 1:n)
        S[i, j] = one(T)
        k = findfirst(==(i), view(rowvals(S), nzrange(S, j)))
        nonzeros(S)[first(nzrange(S, j)) + k - 1] = zero(T)
    end
    return S
end

# The vector analogue of `sweepmatrix`.
function sweepvector(rng, ::Type{T}, ::Type{Ti}, n, density, storedzeros) where {T,Ti}
    x = SparseVector{T,Ti}(sprand(rng, n, density, sweepvalues, T))
    for _ in 1:min(storedzeros, n)
        i = rand(rng, 1:n)
        x[i] = one(T)
        nonzeros(x)[findfirst(==(i), nonzeroinds(x))] = zero(T)
    end
    return x
end

# The sparse operand of a case and its dense twin in the same form. `rng` continues from
# `gencase`, so a case and its rng position determine the operand.
function materialize(c::SweepCase, rng)
    S = sweepmatrix(rng, c.T, c.Ti, c.m, c.n, c.density, c.storedzeros)
    M = Matrix(S)
    if c.form === :plain
        return S, M
    elseif c.form === :colview
        lo = rand(rng, 1:(c.n + 1))
        hi = rand(rng, (lo - 1):c.n)
        return view(S, :, lo:hi), view(M, :, lo:hi)
    elseif c.form === :fixed
        return FixedSparseCSC(S), M
    elseif c.form === :transpose
        return transpose(S), transpose(M)
    else
        return adjoint(S), adjoint(M)
    end
end

# A second operand for a kernel, sparse when `A` is sparse and dense otherwise, so that a
# kernel receives sparse inputs on the sparse side and dense inputs on the dense side.
like(A, S) = issparse(A) ? S : Array(S)
companion(rng, A, m, n; density=rand(rng, SWEEP_DENSITIES), storedzeros=rand(rng, 0:2)) =
    like(A, sweepmatrix(rng, eltype(A), Int, m, n, density, storedzeros))
companionvector(rng, A, n; density=rand(rng, SWEEP_DENSITIES), storedzeros=rand(rng, 0:2)) =
    like(A, sweepvector(rng, eltype(A), Int, n, density, storedzeros))
# a dense second operand on both sides
densecompanion(rng, A, m, n) = Array(sweepmatrix(rng, eltype(A), Int, m, n, rand(rng, SWEEP_DENSITIES), 0))
densevector(rng, A, n) = Array(sweepvector(rng, eltype(A), Int, n, rand(rng, SWEEP_DENSITIES), 0))

# A random unit range inside `1:n`, possibly empty, and a random index vector of length
# up to `n` with repeats.
randrange(rng, n) = (lo = rand(rng, 1:(n + 1)); lo:rand(rng, (lo - 1):n))
randindices(rng, n) = n == 0 ? Int[] : rand(rng, 1:n, rand(rng, 0:n))

# `isequal` up to the sign of zero, the comparator of the exact kernels: a sparse array
# does not store the sign of an unstored zero, while the dense twin of a complex adjoint
# carries `conj(0.0 + 0.0im) == 0.0 - 0.0im`, and a dense product of an unstored zero with
# a negative entry is `-0.0`.
exact(a, b) = isequal(a, b) || a == b

# The outcome of `f()`: `(:ok, value)` or `(:error, exception)`.
outcome(f) = try
    (:ok, f())
catch err
    (:error, err)
end

# The registry behind `reproduce`: the kernel of each sweep and whether it is in-place.
const SWEEPS = Dict{String,Tuple{Any,Bool}}()

# Whether the sparse and dense outcomes of a case agree: the same result under `cmp`, or
# the same exception type.
function agree(cmp, (tags, rs), (tagd, rd))
    tags === tagd || return false
    return tags === :ok ? cmp(rs, rd) : typeof(rs) === typeof(rd)
end

"""
    sweep(name, kernel; cases=SWEEP_CASES, seed=SWEEP_SEED, cmp=exact, check=Returns(true),
          broken=Returns(false), checkbroken=Returns(false))

Run `kernel(A, rng)` on the sparse operand and on its dense twin of each generated case,
with the same `rng` state for both, and test that the outcomes agree under `cmp` (or that
both throw the same exception type) and that `check(case, result)` holds for the sparse
result. A kernel may mutate its operand: every sweep builds the operands afresh. A case
for which `broken(A, rng)` holds, with the `rng` state the kernel sees, is a known bug
and is tested with `@test_broken`; one for which `checkbroken(case)` holds has a known
wrong result type, and only its `check` is `@test_broken`.
"""
function sweep(name, kernel; cases=SWEEP_CASES, seed=SWEEP_SEED, cmp=exact, check=Returns(true),
               broken=Returns(false), checkbroken=Returns(false))
    SWEEPS[name] = (kernel, false)
    @testset "$name" begin
        for i in 1:cases
            rng = caserng(seed, i)
            c = gencase(rng)
            @testset "$name seed=$seed case=$i $(describe(c))" begin
                A, D = materialize(c, rng)
                known = broken(A, copy(rng))
                rs = outcome(() -> kernel(A, copy(rng)))
                rd = outcome(() -> kernel(D, rng))
                if known
                    @test_broken agree(cmp, rs, rd)
                else
                    @test agree(cmp, rs, rd)
                end
                if rs[1] === :ok && !known
                    if checkbroken(c)
                        @test_broken check(c, rs[2])
                    else
                        @test check(c, rs[2])
                    end
                end
            end
        end
    end
end

"""
    sweep!(name, kernel!; cases=SWEEP_CASES, seed=SWEEP_SEED, cmp=exact, broken=Returns(false))

Run `kernel!(A, D, rng)` on the sparse operand and its dense twin of each generated case
and test that the two agree under `cmp` afterwards. The kernel sees both operands, so it
can choose in-pattern positions for a fixed-pattern operand and assert on its own. A case
for which `broken(A, rng)` holds, with the `rng` state the kernel sees, is a known bug
and is tested with `@test_broken`.
"""
function sweep!(name, kernel!; cases=SWEEP_CASES, seed=SWEEP_SEED, cmp=exact, broken=Returns(false))
    SWEEPS[name] = (kernel!, true)
    @testset "$name" begin
        for i in 1:cases
            rng = caserng(seed, i)
            c = gencase(rng)
            @testset "$name seed=$seed case=$i $(describe(c))" begin
                A, D = materialize(c, rng)
                if broken(A, copy(rng))
                    @test_broken (kernel!(A, D, rng); cmp(A, D))
                else
                    kernel!(A, D, rng)
                    @test cmp(A, D)
                end
            end
        end
    end
end

"""
    reproduce(name, seed, i)

Rebuild case `i` of the sweep `name` from `seed`, as printed in a failing testset's name,
and return the case, the sparse operand, its dense twin, and the sparse and dense outcomes
of the kernel. Include this file with `SPARSEARRAYS_SWEEP_CASES=0` to register the sweeps
without running them.
"""
function reproduce(name, seed, i)
    kernel, inplace = SWEEPS[name]
    rng = caserng(seed, i)
    c = gencase(rng)
    A, D = materialize(c, rng)
    if inplace
        rs = outcome(() -> kernel(A, D, rng))
        return (case=c, sparse=A, dense=D, outcome=rs)
    end
    rs = outcome(() -> kernel(A, copy(rng)))
    rd = outcome(() -> kernel(D, rng))
    return (case=c, sparse=A, dense=D, sparse_outcome=rs, dense_outcome=rd)
end
