# This file is a part of Julia. License is MIT: https://julialang.org/license

# Seeded sparse-versus-dense sweeps: every kernel below runs on random sparse operands and
# on their dense twins, and the two must agree. `sweephelpers.jl` has the generator and
# the `sweep`/`sweep!` drivers; `SPARSEARRAYS_SWEEP_SEED` and `SPARSEARRAYS_SWEEP_CASES`
# choose the seed and the case count. A failing testset's name is its reproducer: with
# `SPARSEARRAYS_SWEEP_CASES=0`, including this file registers the sweeps without running
# them, and `reproduce(name, seed, i)` rebuilds the case in the REPL.

module TortureSweepTests
using Test
using LinearAlgebra
using SparseArrays
include("../testhelpers.jl")
include("sweephelpers.jl")

approx(a, b) = isapprox(a, b)
issparse_result(c, r) = issparse(r)
isvector_result(c, r) = r isa AbstractSparseVector

nonempty(A) = !isempty(A)
randvalue(rng, ::Type{T}) where {T} = rand(rng, (zero(T), -zero(T), one(T), sweepvalues(rng, T, 1)[1]))
# whether writing `v` outside the pattern stores an entry: only an unsigned zero is dropped
inserts(v) = v !== zero(v)
inpattern(A::AbstractSparseMatrixCSC, i, j) = i in view(rowvals(A), nzrange(A, j))
wholepattern(A::AbstractSparseMatrixCSC, I, J) = all(inpattern(A, i, j) for j in J, i in I)
anystored(f, A::AbstractSparseMatrixCSC) =
    any(f(rowvals(A)[p], j) && !iszero(nonzeros(A)[p]) for j in axes(A, 2) for p in nzrange(A, j))
iswrapped(c::SweepCase) = c.form in (:colview, :transpose, :adjoint)

# a random mask and the values to write through it, zero where a fixed pattern cannot take them
function maskwrite(rng, A)
    mask = rand(rng, Bool, size(A))
    x = Array(sweepvector(rng, eltype(A), Int, count(mask), rand(rng, SWEEP_DENSITIES), 1))
    if A isa FixedSparseCSC
        any(k -> mask[k] && !inpattern(A, Tuple(CartesianIndices(A)[k])...), eachindex(A)) &&
            fill!(x, zero(eltype(A)))
    end
    return mask, x
end

# Known bug: the logical-mask `setindex!` kernel of `SparseMatrixCSC` reads each column
# after its first insertion through the already-shifted column pointer, so leading stored
# entries of later columns are skipped: the result is wrong or its buffers are corrupt.
# Which cases go wrong depends on the interleaving of mask, pattern and values, so the
# predicate replays the write on a copy.
function logicalsetindexbroken(A, mask, x)
    A isa SparseMatrixCSC || return false
    B, M = copy(A), Matrix(A)
    M[mask] = x
    return outcome(() -> B[mask] = x)[1] === :error || Matrix(B) != M
end
# Known bug: `reverse` of a fixed-pattern matrix with no rows throws a `BoundsError`.
fixedreversebroken(A) = A isa FixedSparseCSC && size(A, 1) == 0
# Known bug: `istriu(A, k)` scans the columns `1:min(n, m-1)` and `istril(A, k)` the
# columns `2:n`, the ranges for `k == 0`, so a nonzero outside the band in a column those
# ranges miss goes unnoticed. Transposition swaps the two tests and negates `k`.
function bandbroken(A, k)
    A isa Union{Adjoint,Transpose} && return bandbroken(parent(A), -k)
    A isa AbstractSparseMatrixCSC || return false
    m = size(A, 1)
    triu = k > 0 && !anystored((i, j) -> j < m && i > j - k, A) && anystored((i, j) -> j >= m && i > j - k, A)
    tril = k < 0 && !anystored((i, j) -> j > 1 && i < j - k, A) && anystored((i, j) -> j == 1 && i < j - k, A)
    return triu || tril
end

@testset "sweeps" begin

@testset "getindex" begin
    sweep("getindex scalar", (A, rng) -> begin
        m, n = size(A)
        [A[rand(rng, 1:m), rand(rng, 1:n)] for _ in 1:(nonempty(A) ? 5 : 0)]
    end)
    sweep("getindex linear", (A, rng) -> [A[k] for k in randindices(rng, length(A))])
    sweep("getindex ranges", (A, rng) -> A[randrange(rng, size(A, 1)), randrange(rng, size(A, 2))];
          check=issparse_result)
    sweep("getindex column", (A, rng) -> size(A, 2) == 0 ? A[:, 1:0] : A[:, rand(rng, 1:size(A, 2))];
          check=issparse_result)
    sweep("getindex row", (A, rng) -> size(A, 1) == 0 ? A[1:0, :] : A[rand(rng, 1:size(A, 1)), :];
          check=issparse_result)
    sweep("getindex index vectors", (A, rng) -> A[randindices(rng, size(A, 1)), randindices(rng, size(A, 2))];
          check=issparse_result)
    sweep("getindex colon and index vector", (A, rng) ->
          rand(rng, Bool) ? A[:, randindices(rng, size(A, 2))] : A[randindices(rng, size(A, 1)), :];
          check=issparse_result)
    sweep("getindex logical matrix", (A, rng) -> A[rand(rng, Bool, size(A))]; check=isvector_result)
    sweep("getindex sparse logical matrix", (A, rng) -> A[like(A, sparse(rand(rng, Bool, size(A))))];
          check=isvector_result)
    sweep("getindex logical vector", (A, rng) -> A[rand(rng, Bool, length(A))]; check=isvector_result)
end

@testset "setindex!" begin
    sweep!("setindex! scalar", (A, D, rng) -> begin
        m, n = size(A)
        nonempty(A) || return
        for _ in 1:6
            i, j = rand(rng, 1:m), rand(rng, 1:n)
            v = randvalue(rng, eltype(A))
            if A isa FixedSparseCSC && inserts(v) && !inpattern(A, i, j)
                @test_throws ArgumentError A[i, j] = v
                @test A[i, j] == D[i, j]
            else
                A[i, j] = v
                D[i, j] = v
            end
        end
    end)
    sweep!("setindex! linear", (A, D, rng) -> begin
        nonempty(A) || return
        for _ in 1:6
            k = rand(rng, 1:length(A))
            v = randvalue(rng, eltype(A))
            if A isa FixedSparseCSC && inserts(v) && !inpattern(A, Tuple(CartesianIndices(A)[k])...)
                @test_throws ArgumentError A[k] = v
                @test A[k] == D[k]
            else
                A[k] = v
                D[k] = v
            end
        end
    end)
    sweep!("setindex! scalar into ranges", (A, D, rng) -> begin
        m, n = size(A)
        for _ in 1:3
            I, J = randrange(rng, m), randrange(rng, n)
            v = randvalue(rng, eltype(A))
            # a fixed pattern takes nonzero writes only inside itself
            A isa FixedSparseCSC && inserts(v) && !wholepattern(A, I, J) && (v = zero(v))
            A[I, J] .= v
            D[I, J] .= v
        end
    end)
    sweep!("setindex! array into ranges", (A, D, rng) -> begin
        m, n = size(A)
        for _ in 1:3
            I, J = randrange(rng, m), randrange(rng, n)
            B = sweepmatrix(rng, eltype(A), Int, length(I), length(J), rand(rng, SWEEP_DENSITIES), 1)
            A isa FixedSparseCSC && !wholepattern(A, I, J) && (B = spzeros(eltype(A), size(B)...))
            if rand(rng, Bool)
                A[I, J] = B
                D[I, J] = B
            else
                A[I, J] = Array(B)
                D[I, J] = Array(B)
            end
        end
    end)
    sweep!("setindex! array into index vectors", (A, D, rng) -> begin
        m, n = size(A)
        I = randperm(rng, m)[1:rand(rng, 0:m)]
        J = randperm(rng, n)[1:rand(rng, 0:n)]
        B = Array(sweepmatrix(rng, eltype(A), Int, length(I), length(J), rand(rng, SWEEP_DENSITIES), 1))
        A isa FixedSparseCSC && !wholepattern(A, I, J) && fill!(B, zero(eltype(A)))
        A[I, J] = B
        D[I, J] = B
    end)
    sweep!("setindex! vector into column", (A, D, rng) -> begin
        m, n = size(A)
        n == 0 && return
        j = rand(rng, 1:n)
        x = sweepvector(rng, eltype(A), Int, m, rand(rng, SWEEP_DENSITIES), 1)
        A isa FixedSparseCSC && !wholepattern(A, 1:m, j) && (x = spzeros(eltype(A), m))
        if rand(rng, Bool)
            A[:, j] = x
            D[:, j] = x
        else
            A[:, j] = Array(x)
            D[:, j] = Array(x)
        end
    end)
    sweep!("setindex! into logical mask", (A, D, rng) -> begin
        mask, x = maskwrite(rng, A)
        A[mask] = x
        D[mask] = x
    end; broken=(A, rng) -> logicalsetindexbroken(A, maskwrite(rng, A)...))
end

@testset "map and broadcast" begin
    sweep("map zero-preserving", (A, rng) -> map(x -> 2x, A); check=issparse_result)
    sweep("map not zero-preserving", (A, rng) -> map(x -> x + 1, A))
    sweep("broadcast unary zero-preserving", (A, rng) -> abs.(A); check=issparse_result)
    sweep("broadcast unary not zero-preserving", (A, rng) -> (A .+ 1) .- 1)
    sweep("broadcast scalar product", (A, rng) -> A .* 3; check=issparse_result)
    sweep("broadcast two sparse", (A, rng) -> begin
        B = companion(rng, A, size(A)...)
        (A .+ B, A .* B, max.(A, B))
    end; cmp=(a, b) -> all(map(exact, a, b)), check=(c, r) -> all(issparse, r))
    sweep("broadcast dense vector along columns", (A, rng) -> A .* densevector(rng, A, size(A, 1)))
    sweep("broadcast dense row along rows", (A, rng) -> A .+ permutedims(densevector(rng, A, size(A, 2))))
    sweep("broadcast sparse and dense matrix", (A, rng) -> A .* densecompanion(rng, A, size(A)...))
    sweep("broadcast three arguments", (A, rng) -> begin
        B = companion(rng, A, size(A)...)
        v = densevector(rng, A, size(A, 1))
        A .* B .+ v
    end)
    sweep("broadcast!", (A, rng) -> begin
        B = companion(rng, A, size(A)...)
        C = similar(B)
        C .= A .* 2 .+ B
        C
    end)
    sweep("map!", (A, rng) -> begin
        B = companion(rng, A, size(A)...)
        map!(x -> 3x, B, A)
        B
    end; check=issparse_result)
end

@testset "arithmetic" begin
    sweep("sparse plus sparse", (A, rng) -> A + companion(rng, A, size(A)...); check=issparse_result)
    sweep("sparse minus sparse", (A, rng) -> A - companion(rng, A, size(A)...); check=issparse_result)
    sweep("sparse plus dense", (A, rng) -> A + densecompanion(rng, A, size(A)...))
    sweep("dense minus sparse", (A, rng) -> densecompanion(rng, A, size(A)...) - A)
    sweep("unary minus", (A, rng) -> -A; check=issparse_result)
    sweep("scalar times", (A, rng) -> (2 * A, A * 2, A / 2); cmp=(a, b) -> all(map(approx, a, b)),
          check=(c, r) -> all(issparse, r))
    sweep("sparse times sparse", (A, rng) -> A * companion(rng, A, size(A, 2), rand(rng, SWEEP_SHAPES));
          cmp=approx, check=issparse_result)
    sweep("sparse times dense matrix", (A, rng) -> A * densecompanion(rng, A, size(A, 2), rand(rng, SWEEP_SHAPES));
          cmp=approx)
    sweep("dense matrix times sparse", (A, rng) -> densecompanion(rng, A, rand(rng, SWEEP_SHAPES), size(A, 1)) * A;
          cmp=approx)
    sweep("sparse times dense vector", (A, rng) -> A * densevector(rng, A, size(A, 2)); cmp=approx)
    sweep("sparse times sparse vector", (A, rng) -> A * companionvector(rng, A, size(A, 2));
          cmp=approx, check=isvector_result)
    sweep("transposed vector times sparse", (A, rng) -> begin
        v = densevector(rng, A, size(A, 1))
        (transpose(v) * A, v' * A)
    end; cmp=(a, b) -> all(map(approx, a, b)))
    sweep("transformed products", (A, rng) -> begin
        B = companion(rng, A, size(A, 1), rand(rng, SWEEP_SHAPES))
        (transpose(A) * B, A' * B, transpose(B) * A, B' * A)
    end; cmp=(a, b) -> all(map(approx, a, b)), check=(c, r) -> all(issparse, r))
end

@testset "mul!" begin
    sweep!("mul! sparse-dense into dense", (A, D, rng) -> begin
        m, n = size(A)
        k = rand(rng, SWEEP_SHAPES)
        B = densecompanion(rng, A, n, k)
        C = densecompanion(rng, A, m, k)
        Cs, Cd = copy(C), copy(C)
        α, β = rand(rng, (0, 1, 2)), rand(rng, (0, 1, 2))
        mul!(Cs, A, B, α, β)
        mul!(Cd, D, B, α, β)
        @test Cs ≈ Cd
    end)
    sweep!("mul! dense-sparse into dense", (A, D, rng) -> begin
        m, n = size(A)
        k = rand(rng, SWEEP_SHAPES)
        B = densecompanion(rng, A, k, m)
        C = densecompanion(rng, A, k, n)
        Cs, Cd = copy(C), copy(C)
        α, β = rand(rng, (0, 1, 2)), rand(rng, (0, 1, 2))
        mul!(Cs, B, A, α, β)
        mul!(Cd, B, D, α, β)
        @test Cs ≈ Cd
    end)
    sweep!("mul! matrix-vector into dense", (A, D, rng) -> begin
        m, n = size(A)
        x = rand(rng, Bool) ? densevector(rng, A, n) : sweepvector(rng, eltype(A), Int, n, 0.3, 1)
        y = densevector(rng, A, m)
        ys, yd = copy(y), copy(y)
        α, β = rand(rng, (0, 1, 2)), rand(rng, (0, 1, 2))
        mul!(ys, A, x, α, β)
        mul!(yd, D, x, α, β)
        @test ys ≈ yd
    end)
    sweep!("mul! sparse-sparse into sparse", (A, D, rng) -> begin
        m, n = size(A)
        k = rand(rng, SWEEP_SHAPES)
        B = sweepmatrix(rng, eltype(A), Int, n, k, 0.3, 1)
        C = sweepmatrix(rng, eltype(A), Int, m, k, 0.3, 1)
        Cd = Matrix(C)
        α, β = rand(rng, (0, 1, 2)), rand(rng, (0, 1, 2))
        mul!(C, A, B, α, β)
        mul!(Cd, D, Matrix(B), α, β)
        @test C ≈ Cd
    end)
end

@testset "reductions" begin
    for (f, cmp) in ((sum, approx), (prod, approx), (maximum, exact), (minimum, exact), (extrema, exact))
        sweep("$f over dims", (A, rng) -> f(A, dims=rand(rng, (1, 2, (1, 2)))); cmp)
        sweep("$f over all", (A, rng) -> f(A); cmp)
    end
    sweep("sum of abs over dims", (A, rng) -> sum(abs, A, dims=rand(rng, (1, 2))); cmp=approx)
    sweep("mapreduce over dims", (A, rng) -> mapreduce(x -> x * x, +, A, dims=rand(rng, (1, 2))); cmp=approx)
    sweep("count and any and all", (A, rng) -> (count(!iszero, A), any(!iszero, A), all(iszero, A),
                                                count(!iszero, A, dims=1), count(!iszero, A, dims=2)))
    sweep("cumsum over dims", (A, rng) -> cumsum(A, dims=rand(rng, (1, 2))); cmp=approx)
end

@testset "reverse, circshift and rotations" begin
    sweep("reverse", (A, rng) -> reverse(A); check=issparse_result, broken=(A, rng) -> fixedreversebroken(A))
    sweep("reverse over dims", (A, rng) -> reverse(A, dims=rand(rng, (1, 2))); check=issparse_result,
          broken=(A, rng) -> fixedreversebroken(A))
    sweep("circshift", (A, rng) -> circshift(A, (rand(rng, -40:40), rand(rng, -40:40))); check=issparse_result)
    sweep("circshift one dimension", (A, rng) -> circshift(A, rand(rng, -40:40)); check=issparse_result)
    sweep("rot180", (A, rng) -> rot180(A); check=issparse_result)
    sweep("rotl90", (A, rng) -> rotl90(A); check=issparse_result)
    sweep("rotr90", (A, rng) -> rotr90(A); check=issparse_result)
end

@testset "concatenation" begin
    sweep("hcat sparse", (A, rng) -> hcat(A, companion(rng, A, size(A, 1), rand(rng, SWEEP_SHAPES)));
          check=issparse_result)
    sweep("vcat sparse", (A, rng) -> vcat(A, companion(rng, A, rand(rng, SWEEP_SHAPES), size(A, 2)));
          check=issparse_result)
    sweep("hcat dense and sparse", (A, rng) -> [densecompanion(rng, A, size(A, 1), rand(rng, SWEEP_SHAPES)) A];
          check=issparse_result)
    sweep("vcat sparse and dense", (A, rng) -> [A; densecompanion(rng, A, rand(rng, SWEEP_SHAPES), size(A, 2))];
          check=issparse_result)
    sweep("hcat sparse and vectors", (A, rng) -> begin
        m = size(A, 1)
        [A companionvector(rng, A, m) densevector(rng, A, m)]
    end; check=issparse_result)
    sweep("vcat sparse and transposed vector", (A, rng) -> [A; permutedims(densevector(rng, A, size(A, 2)))];
          check=issparse_result)
    sweep("hvcat sparse blocks", (A, rng) -> begin
        m, n = size(A)
        k, l = rand(rng, SWEEP_SHAPES), rand(rng, SWEEP_SHAPES)
        [A companion(rng, A, m, l); companion(rng, A, k, n) companion(rng, A, k, l)]
    end; check=issparse_result)
    sweep("hvcat mixed blocks", (A, rng) -> begin
        m, n = size(A)
        k, l = rand(rng, SWEEP_SHAPES), rand(rng, SWEEP_SHAPES)
        [A densecompanion(rng, A, m, l); densecompanion(rng, A, k, n) companion(rng, A, k, l)]
    end; check=issparse_result)
    sweep("hcat with a uniform scaling", (A, rng) -> [A 2I]; check=issparse_result)
end

@testset "transposition and permutation" begin
    sweep("copy of transpose", (A, rng) -> copy(transpose(A)); check=issparse_result)
    sweep("copy of adjoint", (A, rng) -> copy(adjoint(A)); check=issparse_result)
    sweep("permutedims", (A, rng) -> permutedims(A); check=issparse_result)
    sweep("permutedims with dims", (A, rng) -> permutedims(A, rand(rng, ((1, 2), (2, 1)))); check=issparse_result)
    sweep("permute rows and columns", (A, rng) -> begin
        p, q = randperm(rng, size(A, 1)), randperm(rng, size(A, 2))
        A isa AbstractSparseMatrixCSC ? permute(A, p, q) : A[p, q]
    end; check=issparse_result)
    sweep("transpose!", (A, rng) -> begin
        B = Array(A)
        C = similar(B, reverse(size(B)))
        transpose!(C, A)
        C
    end)
    sweep("conj and real and imag", (A, rng) -> (conj(A), real(A), imag(A));
          cmp=(a, b) -> all(map(exact, a, b)), check=(c, r) -> all(issparse, r))
end

@testset "triangular parts and differences" begin
    sweep("triu", (A, rng) -> triu(A, rand(rng, -size(A, 1):size(A, 2))); check=issparse_result)
    sweep("tril", (A, rng) -> tril(A, rand(rng, -size(A, 1):size(A, 2))); check=issparse_result)
    sweep("triu and tril in place", (A, rng) -> begin
        k = rand(rng, -size(A, 1):size(A, 2))
        B = A isa FixedSparseCSC ? SparseMatrixCSC(A) : copy(A)
        rand(rng, Bool) ? triu!(B, k) : tril!(B, k)
    end)
    # known bug: `diff` of a view, transpose or adjoint of a sparse matrix is dense
    sweep("diff over dims", (A, rng) -> diff(A, dims=rand(rng, (1, 2))); check=issparse_result,
          checkbroken=iswrapped)
    sweep("istriu and istril", (A, rng) -> (k = rand(rng, -3:3); (istriu(A, k), istril(A, k), isdiag(A)));
          broken=(A, rng) -> bandbroken(A, rand(rng, -3:3)))
end

@testset "norms and dot" begin
    sweep("norm", (A, rng) -> (norm(A), norm(A, 1), norm(A, Inf), norm(A, 3));
          cmp=(a, b) -> all(map(approx, a, b)))
    sweep("opnorm", (A, rng) -> (opnorm(A, 1), opnorm(A, Inf)); cmp=(a, b) -> all(map(approx, a, b)))
    sweep("dot of two sparse", (A, rng) -> dot(A, companion(rng, A, size(A)...)); cmp=approx)
    sweep("dot of sparse and dense", (A, rng) -> begin
        B = densecompanion(rng, A, size(A)...)
        (dot(A, B), dot(B, A))
    end; cmp=(a, b) -> all(map(approx, a, b)))
    sweep("three-argument dot", (A, rng) -> begin
        x = densevector(rng, A, size(A, 1))
        y = densevector(rng, A, size(A, 2))
        (dot(x, A, y), dot(companionvector(rng, A, size(A, 1)), A, companionvector(rng, A, size(A, 2))))
    end; cmp=(a, b) -> all(map(approx, a, b)))
    sweep("tr", (A, rng) -> tr(A); cmp=approx)
end

@testset "kron" begin
    # known bug: `kron` of a column view of a sparse matrix is dense
    sweep("kron of two sparse", (A, rng) -> kron(A, companion(rng, A, rand(rng, SWEEP_SHAPES), rand(rng, SWEEP_SHAPES)));
          check=issparse_result, checkbroken=c -> c.form === :colview)
    sweep("kron of sparse and dense", (A, rng) -> begin
        B = densecompanion(rng, A, rand(rng, SWEEP_SHAPES), rand(rng, SWEEP_SHAPES))
        (kron(A, B), kron(B, A))
    end; cmp=(a, b) -> all(map(exact, a, b)))
    sweep("kron of sparse and vectors", (A, rng) -> begin
        x = companionvector(rng, A, rand(rng, SWEEP_SHAPES))
        v = densevector(rng, A, rand(rng, SWEEP_SHAPES))
        (kron(A, x), kron(x, A), kron(A, v), kron(v, A))
    end; cmp=(a, b) -> all(map(exact, a, b)))
end

end # testset sweeps

end # module
