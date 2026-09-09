# This file is a part of Julia. License is MIT: https://julialang.org/license

module CHOLMODTests
using Test

@static if !Base.USE_GPL_LIBS
    @info "This Julia build excludes the use of SuiteSparse GPL libraries. Skipping CHOLMOD tests"
else

using SparseArrays.CHOLMOD
using SparseArrays.CHOLMOD: getcommon
using Random
using Serialization
using LinearAlgebra:
    I, cholesky, cholesky!, det, diag, eigmax, ishermitian, isposdef, issuccess,
    issymmetric, ldiv!, ldlt, ldlt!, logdet, norm, opnorm, Diagonal, Hermitian, Symmetric,
    PosDefException, ZeroPivotException, RowMaximum
using SparseArrays
using SparseArrays: getcolptr
using SparseArrays.LibSuiteSparse
using SparseArrays.LibSuiteSparse: cholmod_l_allocate_sparse, cholmod_allocate_sparse,
    cholmod_l_allocate_dense, cholmod_allocate_dense

# CHOLMOD tests
itypes = sizeof(Int) == 4 ? (Int32,) : (Int32, Int64)
for Ti ∈ itypes, Tv ∈ (Float32, Float64)
Random.seed!(123)

@testset "based on deps/SuiteSparse-4.0.2/CHOLMOD/Demo/ index type $Ti" begin

# chm_rdsp(joinpath(Sys.BINDIR, "../../deps/SuiteSparse-4.0.2/CHOLMOD/Demo/Matrix/bcsstk01.tri"))
# because the file may not exist in binary distributions and when a system suitesparse library
# is used

## Result from C program
## ---------------------------------- cholmod_demo:
## norm (A,inf) = 3.57095e+09
## norm (A,1)   = 3.57095e+09
## CHOLMOD sparse:  A:  48-by-48, nz 224, upper.  OK
## CHOLMOD dense:   B:  48-by-1,   OK
## bnorm 1.97917
## Analyze: flop 6009 lnz 489
## Factorizing A
## CHOLMOD factor:  L:  48-by-48  simplicial, LDL'. nzmax 489.  nz 489  OK
## Ordering: AMD     fl/lnz       12.3  lnz/anz        2.2
## ints in L: 782, doubles in L: 489
## factor flops 6009 nnz(L)             489 (w/no amalgamation)
## nnz(A*A'):             224
## flops / nnz(L):      12.3
## nnz(L) / nnz(A):      2.2
## analyze cputime:        0.0000
## factor  cputime:         0.0000 mflop:      0.0
## solve   cputime:         0.0000 mflop:      0.0
## overall cputime:         0.0000 mflop:      0.0
## peak memory usage:            0 (MB)
## residual  2.5e-19 (|Ax-b|/(|A||x|+|b|))
## residual  1.3e-19 (|Ax-b|/(|A||x|+|b|)) after iterative refinement
## rcond     9.5e-06

    n = 48
    A = CHOLMOD.Sparse(n, n,
        Ti[0,1,2,3,6,9,12,15,18,20,25,30,34,36,39,43,47,52,58,
        62,67,71,77,84,90,93,95,98,103,106,110,115,119,123,130,136,142,146,150,155,
        161,167,174,182,189,197,207,215,224], # zero-based column pointers
        Ti[0,1,2,1,2,3,0,2,4,0,1,5,0,4,6,1,3,7,2,8,1,3,7,8,9,
        0,4,6,8,10,5,6,7,11,6,12,7,11,13,8,10,13,14,9,13,14,15,8,10,12,14,16,7,11,
        12,13,16,17,0,12,16,18,1,5,13,15,19,2,4,14,20,3,13,15,19,20,21,2,4,12,16,18,
        20,22,1,5,17,18,19,23,0,5,24,1,25,2,3,26,2,3,25,26,27,4,24,28,0,5,24,29,6,
        11,24,28,30,7,25,27,31,8,9,26,32,8,9,25,27,31,32,33,10,24,28,30,32,34,6,11,
        29,30,31,35,12,17,30,36,13,31,35,37,14,15,32,34,38,14,15,33,37,38,39,16,32,
        34,36,38,40,12,17,31,35,36,37,41,12,16,17,18,23,36,40,42,13,14,15,19,37,39,
        43,13,14,15,20,21,38,43,44,13,14,15,20,21,37,39,43,44,45,12,16,17,22,36,40,
        42,46,12,16,17,18,23,41,42,46,47],
        Tv[2.83226851852e6,1.63544753086e6,1.72436728395e6,-2.0e6,-2.08333333333e6,
        1.00333333333e9,1.0e6,-2.77777777778e6,1.0675e9,2.08333333333e6,
        5.55555555555e6,1.53533333333e9,-3333.33333333,-1.0e6,2.83226851852e6,
        -6666.66666667,2.0e6,1.63544753086e6,-1.68e6,1.72436728395e6,-2.0e6,4.0e8,
        2.0e6,-2.08333333333e6,1.00333333333e9,1.0e6,2.0e8,-1.0e6,-2.77777777778e6,
        1.0675e9,-2.0e6,2.08333333333e6,5.55555555555e6,1.53533333333e9,-2.8e6,
        2.8360994695e6,-30864.1975309,-5.55555555555e6,1.76741074446e6,
        -15432.0987654,2.77777777778e6,517922.131816,3.89003806848e6,
        -3.33333333333e6,4.29857058902e6,-2.6349902747e6,1.97572063531e9,
        -2.77777777778e6,3.33333333333e8,-2.14928529451e6,2.77777777778e6,
        1.52734651547e9,5.55555555555e6,6.66666666667e8,2.35916180402e6,
        -5.55555555555e6,-1.09779731332e8,1.56411143711e9,-2.8e6,-3333.33333333,
        1.0e6,2.83226851852e6,-30864.1975309,-5.55555555555e6,-6666.66666667,
        -2.0e6,1.63544753086e6,-15432.0987654,2.77777777778e6,-1.68e6,
        1.72436728395e6,-3.33333333333e6,2.0e6,4.0e8,-2.0e6,-2.08333333333e6,
        1.00333333333e9,-2.77777777778e6,3.33333333333e8,-1.0e6,2.0e8,1.0e6,
        2.77777777778e6,1.0675e9,5.55555555555e6,6.66666666667e8,-2.0e6,
        2.08333333333e6,-5.55555555555e6,1.53533333333e9,-28935.1851852,
        -2.08333333333e6,60879.6296296,-1.59791666667e6,3.37291666667e6,
        -28935.1851852,2.08333333333e6,2.41171296296e6,-2.08333333333e6,
        1.0e8,-2.5e6,-416666.666667,1.5e9,-833333.333333,1.25e6,5.01833333333e8,
        2.08333333333e6,1.0e8,416666.666667,5.025e8,-28935.1851852,
        -2.08333333333e6,-4166.66666667,-1.25e6,3.98587962963e6,-1.59791666667e6,
        -8333.33333333,2.5e6,3.41149691358e6,-28935.1851852,2.08333333333e6,
        -2.355e6,2.43100308642e6,-2.08333333333e6,1.0e8,-2.5e6,5.0e8,2.5e6,
        -416666.666667,1.50416666667e9,-833333.333333,1.25e6,2.5e8,-1.25e6,
        -3.47222222222e6,1.33516666667e9,2.08333333333e6,1.0e8,-2.5e6,
        416666.666667,6.94444444444e6,2.16916666667e9,-28935.1851852,
        -2.08333333333e6,-3.925e6,3.98587962963e6,-1.59791666667e6,
        -38580.2469136,-6.94444444444e6,3.41149691358e6,-28935.1851852,
        2.08333333333e6,-19290.1234568,3.47222222222e6,2.43100308642e6,
        -2.08333333333e6,1.0e8,-4.16666666667e6,2.5e6,-416666.666667,
        1.50416666667e9,-833333.333333,-3.47222222222e6,4.16666666667e8,
        -1.25e6,3.47222222222e6,1.33516666667e9,2.08333333333e6,1.0e8,
        6.94444444445e6,8.33333333333e8,416666.666667,-6.94444444445e6,
        2.16916666667e9,-3830.95098171,1.14928529451e6,-275828.470683,
        -28935.1851852,-2.08333333333e6,-4166.66666667,1.25e6,64710.5806113,
        -131963.213599,-517922.131816,-2.29857058902e6,-1.59791666667e6,
        -8333.33333333,-2.5e6,3.50487988027e6,-517922.131816,-2.16567078453e6,
        551656.941366,-28935.1851852,2.08333333333e6,-2.355e6,517922.131816,
        4.57738374749e6,2.29857058902e6,-551656.941367,4.8619365099e8,
        -2.08333333333e6,1.0e8,2.5e6,5.0e8,-4.79857058902e6,134990.2747,
        2.47238730198e9,-1.14928529451e6,2.29724661236e8,-5.57173510779e7,
        -833333.333333,-1.25e6,2.5e8,2.39928529451e6,9.61679848804e8,275828.470683,
        -5.57173510779e7,1.09411960038e7,2.08333333333e6,1.0e8,-2.5e6,
        140838.195984,-1.09779731332e8,5.31278103775e8], 1)
    @test CHOLMOD.norm_sparse(A, 0) ≈ 3.570948074697437e9
    @test CHOLMOD.norm_sparse(A, 1) ≈ 3.570948074697437e9
    @test_throws ArgumentError CHOLMOD.norm_sparse(A, 2)
    @test CHOLMOD.isvalid(A)

    x = fill(Tv(1.), n)
    b = A*x

    chma = ldlt(A)                      # LDL' form
    @test CHOLMOD.isvalid(chma)
    @test unsafe_load(pointer(chma)).is_ll == 0    # check that it is in fact an LDLt
    @test chma\b ≈ x
    @test nnz(ldlt(A, perm=1:size(A,1))) > nnz(chma)
    @test size(chma) == size(A)
    chmal = CHOLMOD.FactorComponent(chma, :L)
    @test size(chmal) == size(A)
    @test size(chmal, 1) == size(A, 1)

    chma = cholesky(A)                      # LL' form
    @test CHOLMOD.isvalid(chma)
    @test unsafe_load(pointer(chma)).is_ll == 1    # check that it is in fact an LLt
    @test chma\b ≈ x
    x2 = zero(x)
    @inferred ldiv!(x2, chma, b)
    @test x2 ≈ x
    @test nnz(chma) == 489
    @test nnz(cholesky(A, perm=1:size(A,1))) > nnz(chma)
    @test size(chma) == size(A)
    chmal = CHOLMOD.FactorComponent(chma, :L)
    @test size(chmal) == size(A)
    @test size(chmal, 1) == size(A, 1)

    @testset "eltype" begin
        @test eltype(Dense(fill(Tv(1.), 3))) == Tv
        @test eltype(A) == Tv
        @test eltype(chma) == Tv
    end
end


for Tv2 ∈ (Float32, Float64)
@testset "lp_afiro example ($Tv, $Ti) \\ ($Tv2, $Ti)" begin
    afiro = CHOLMOD.Sparse(27, 51,
        Ti[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
        23,25,27,29,33,37,41,45,47,49,51,53,55,57,59,63,65,67,69,71,75,79,83,87,89,
        91,93,95,97,99,101,102],
        Ti[2,3,6,7,8,9,12,13,16,17,18,19,20,21,22,23,24,25,26,
        0,1,2,23,0,3,0,21,1,25,4,5,6,24,4,5,7,24,4,5,8,24,4,5,9,24,6,20,7,20,8,20,9,
        20,3,4,4,22,5,26,10,11,12,21,10,13,10,23,10,20,11,25,14,15,16,22,14,15,17,
        22,14,15,18,22,14,15,19,22,16,20,17,20,18,20,19,20,13,15,15,24,14,26,15],
        Tv[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
        1.0,-1.0,-1.06,1.0,0.301,1.0,-1.0,1.0,-1.0,1.0,1.0,-1.0,-1.06,1.0,0.301,
        -1.0,-1.06,1.0,0.313,-1.0,-0.96,1.0,0.313,-1.0,-0.86,1.0,0.326,-1.0,2.364,
        -1.0,2.386,-1.0,2.408,-1.0,2.429,1.4,1.0,1.0,-1.0,1.0,1.0,-1.0,-0.43,1.0,
        0.109,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,1.0,-0.43,1.0,1.0,0.109,-0.43,1.0,1.0,
        0.108,-0.39,1.0,1.0,0.108,-0.37,1.0,1.0,0.107,-1.0,2.191,-1.0,2.219,-1.0,
        2.249,-1.0,2.279,1.4,-1.0,1.0,-1.0,1.0,1.0,1.0], 0)
    afiro2 = CHOLMOD.aat(afiro, Ti[0:50;], Ti(1))
    CHOLMOD.change_stype!(afiro2, -1)
    chmaf = cholesky(afiro2)
    y = afiro'*fill(one(Tv), size(afiro,1))
    sol = @test_nowarn chmaf\convert(Dense{Tv2}, (afiro*y)) # least squares solution
    @test eltype(sol) == promote_type(Tv, Tv2)
    @test CHOLMOD.isvalid(sol)
    pred = afiro'*sol
    @test norm(afiro * (convert(Matrix, y) - convert(Matrix, pred))) <
        √(eps(Float32 <: Union{Tv, Tv2} ? Float32 : Float64)) # is this reasonable?
end
end

@testset "Issue 9160 $Ti" begin
    local A, B
    A = sprand(10, 10, 0.1)
    A = convert(SparseMatrixCSC{Tv,Ti}, A)
    cmA = CHOLMOD.Sparse(A)

    B = sprand(10, 10, 0.1)
    B = convert(SparseMatrixCSC{Tv,Ti}, B)
    cmB = CHOLMOD.Sparse(B)

    # Ac_mul_B
    @test sparse(cmA'*cmB) ≈ A'*B

    # A_mul_Bc
    @test sparse(cmA*cmB') ≈ A*B'

    # A_mul_Ac
    @test sparse(cmA*cmA') ≈ A*A'

    # Ac_mul_A
    @test sparse(cmA'*cmA) ≈ A'*A

    # A_mul_Ac for symmetric A
    A = 0.5*(A + copy(A'))
    cmA = CHOLMOD.Sparse(A)
    @test sparse(cmA*cmA') ≈ A*A'
end

@testset "Check inputs to Sparse. Related to #20024" for t_ in (
    (2, 2, [1, 2], Ti[], Tv[]),
    (2, 2, [1, 2, 3], Ti[1], Tv[]),
    (2, 2, [1, 2, 3], Ti[], Tv[1.0]),
    (2, 2, [1, 2, 3], Ti[1], Tv[1.0]))
    @test_throws ArgumentError SparseMatrixCSC(t_...)
    @test_throws ArgumentError CHOLMOD.Sparse(t_[1], t_[2], t_[3] .- 1, t_[4] .- 1, t_[5])
end

## The struct pointer must be constructed by the library constructor and then modified afterwards to checks that the method throws
# The constructors must free the pointer before throwing, so the Common's
# allocation count must be back to its previous value afterwards.
malloc_count(T) = getcommon(T)[].malloc_count
@testset "illegal dtype" begin
    nmalloc = malloc_count(Ti)
    p = Ti == Int64 ? cholmod_l_allocate_sparse(1, 1, 1, true, true, 0, CHOLMOD.xdtyp(Tv), getcommon(Ti)) :
        cholmod_allocate_sparse(1, 1, 1, true, true, 0, CHOLMOD.xdtyp(Tv), getcommon(Ti))
    @test malloc_count(Ti) > nmalloc
    puint = convert(Ptr{UInt32}, p)
    # The second argument 5 is the invalid `dtype`.
    # CHOLMOD_DOUBLE (0) and CHOLMOD_SINGLE (4) are both valid.
    unsafe_store!(puint, 5, 3*div(sizeof(Csize_t), 4) + 5*div(sizeof(Ptr{Cvoid}), 4) + 4)
    @test_throws CHOLMOD.CHOLMODException CHOLMOD.Sparse(p)
    @test malloc_count(Ti) == nmalloc
end

@testset "illegal xtype" begin
    nmalloc = malloc_count(Ti)
    p = Ti == Int64 ? cholmod_l_allocate_sparse(1, 1, 1, true, true, 0, CHOLMOD.xdtyp(Tv), getcommon(Ti)) :
        cholmod_allocate_sparse(1, 1, 1, true, true, 0, CHOLMOD.xdtyp(Tv), getcommon(Ti))
    @test malloc_count(Ti) > nmalloc
    puint = convert(Ptr{UInt32}, p)
    # The second argument 3 is the invalid `xtype`.
    # CHOLMOD_REAL (1), CHOLMOD_COMPLEX (2) are valid.
    unsafe_store!(puint, 3, 3*div(sizeof(Csize_t), 4) + 5*div(sizeof(Ptr{Cvoid}), 4) + 3)
    @test_throws CHOLMOD.CHOLMODException CHOLMOD.Sparse(p)
    @test malloc_count(Ti) == nmalloc
end

@testset "illegal dense xtype" begin
    # `free!(::Ptr{cholmod_dense})` always uses the native-Int Common
    nmalloc = malloc_count(Int)
    p = sizeof(Int) == 8 ? cholmod_l_allocate_dense(1, 1, 1, CHOLMOD.xdtyp(Tv), getcommon(Int)) :
        cholmod_allocate_dense(1, 1, 1, CHOLMOD.xdtyp(Tv), getcommon(Int))
    @test malloc_count(Int) > nmalloc
    xtype_offset = fieldoffset(LibSuiteSparse.cholmod_dense, findfirst(==(:xtype), fieldnames(LibSuiteSparse.cholmod_dense)))
    unsafe_store!(Ptr{Cint}(p + xtype_offset), 3) # CHOLMOD_ZOMPLEX is not supported
    @test_throws CHOLMOD.CHOLMODException CHOLMOD.Dense(p)
    @test malloc_count(Int) == nmalloc
end

# Test that a bogus `itype` raises the expected exception.
# With an invalid itype the constructor cannot know which Common allocated the
# pointer, so only the total count over all Commons is guaranteed to balance.
@testset "illegal itype I" begin
    nmalloc = sum(malloc_count, itypes)
    p = Ti == Int64 ? cholmod_l_allocate_sparse(1, 1, 1, true, true, 0, CHOLMOD.xdtyp(Tv), getcommon(Ti)) :
        cholmod_allocate_sparse(1, 1, 1, true, true, 0, CHOLMOD.xdtyp(Tv), getcommon(Ti))
    @test sum(malloc_count, itypes) > nmalloc
    puint = convert(Ptr{UInt32}, p)
    # The second argument to `unsafe_store!` is the illegal `itype`
    unsafe_store!(puint, 123, 3*div(sizeof(Csize_t), 4) + 5*div(sizeof(Ptr{Cvoid}), 4) + 2)
    @test_throws CHOLMOD.CHOLMODException CHOLMOD.Sparse(p)
    @test sum(malloc_count, itypes) == nmalloc
end

@testset "illegal itype II" begin
    nmalloc = sum(malloc_count, itypes)
    p = Ti == Int64 ? cholmod_l_allocate_sparse(1, 1, 1, true, true, 0, CHOLMOD.xdtyp(Tv), getcommon(Ti)) :
        cholmod_allocate_sparse(1, 1, 1, true, true, 0, CHOLMOD.xdtyp(Tv), getcommon(Ti))
    @test sum(malloc_count, itypes) > nmalloc
    puint = convert(Ptr{UInt32}, p)
    unsafe_store!(puint,  5, 3*div(sizeof(Csize_t), 4) + 5*div(sizeof(Ptr{Cvoid}), 4) + 2)
    @test_throws CHOLMOD.CHOLMODException CHOLMOD.Sparse(p)
    @test sum(malloc_count, itypes) == nmalloc
end
@testset "test free! $Ti" begin
    p = Ti == Int64 ? cholmod_l_allocate_sparse(1, 1, 1, true, true, 0, CHOLMOD.xdtyp(Tv), getcommon(Ti)) :
        cholmod_allocate_sparse(1, 1, 1, true, true, 0, CHOLMOD.xdtyp(Tv), getcommon(Ti))
    @test CHOLMOD.free!(p, Ti)
end

@testset "Check common is still in default state" begin
    # This test intentionally depends on all the above tests!
    current_common = CHOLMOD.getcommon(Ti)
    default_common = Ref(cholmod_common())
    result = Ti === Int64 ? cholmod_l_start(default_common) : cholmod_start(default_common)
    @test result == CHOLMOD.TRUE
    @test current_common[].print == 0
    for name in (
        :nmethods,
        :postorder,
        :final_ll,
        :supernodal,
    )
        @test getproperty(current_common[], name) == getproperty(default_common[], name)
    end
end

@testset "ldiv! $Tv $Ti" begin
    local A, x, x2, b, X, X2, B
    A = sprand(10, 10, 0.1)
    A = I + A * A'
    A = convert(SparseMatrixCSC{Tv,Ti}, A)
    factor = cholesky(A)

    x = fill(Tv(1), 10)
    b = A * x
    x2 = zero(x)
    @inferred ldiv!(x2, factor, b)
    @test x2 ≈ x

    X = fill(Tv(1), 10, 5)
    B = A * X
    X2 = zero(X)
    @inferred ldiv!(X2, factor, B)
    @test X2 ≈ X

    # reuse across multiple calls (Y/E buffers kept in workspace)
    fill!(x2, 0)
    ldiv!(x2, factor, b)
    @test x2 ≈ x

    # Y/E buffers are reused across calls, remaining 16 bytes come from CHOLMOD internals
    allocs = @allocated ldiv!(x2, factor, b)
    @test allocs <= 16

    c = fill(Tv(1), size(x, 1) + 1)
    C = fill(Tv(1), size(X, 1) + 1, size(X, 2))
    y = fill(Tv(1), size(x, 1) + 1)
    Y = fill(Tv(1), size(X, 1) + 1, size(X, 2))
    @test_throws DimensionMismatch ldiv!(y, factor, b)
    @test_throws DimensionMismatch ldiv!(Y, factor, B)
    @test_throws DimensionMismatch ldiv!(x2, factor, c)
    @test_throws DimensionMismatch ldiv!(X2, factor, C)
    @test_throws DimensionMismatch ldiv!(X2, factor, b)
    @test_throws DimensionMismatch ldiv!(x2, factor, B)
end

@testset "ldiv! no memory leak $Tv $Ti" begin
    local A, b, x, F
    A = sprand(10, 10, 0.1)
    A = I + A * A'
    A = convert(SparseMatrixCSC{Tv,Ti}, A)
    F = cholesky(A)
    b = A * fill(Tv(1), 10)
    x = zero(b)

    ldiv!(x, F, b) # allocate buffers
    GC.gc()
    before = getcommon(Ti)[].memory_inuse
    for _ in 1:1000
        ldiv!(x, F, b)
    end
    after = getcommon(Ti)[].memory_inuse
    @test before == after
end

@testset "free!(Factor) releases Y/E through the matching Common $Tv $Ti" begin
    local A, b, x, F
    A = sprand(10, 10, 0.1)
    A = I + A * A'
    A = convert(SparseMatrixCSC{Tv,Ti}, A)
    b = A * fill(Tv(1), 10)
    x = zero(b)
    # warm up so that the Common's persistent workspace is already allocated
    F = cholesky(A)
    ldiv!(x, F, b)
    finalize(F)
    GC.gc() # collect the temporary CHOLMOD objects created by `cholesky`
    nmalloc = Tuple(malloc_count(T) for T in itypes)
    F = cholesky(A)
    ldiv!(x, F, b) # allocates the Y/E buffers in getcommon(Ti)
    @test malloc_count(Ti) > nmalloc[findfirst(==(Ti), itypes)]
    finalize(F)    # must free Y/E with getcommon(Ti) as well
    GC.gc()
    @test Tuple(malloc_count(T) for T in itypes) == nmalloc
end

@testset "copy(Factor) buffer isolation $Tv $Ti" begin
    local A, x, b, x2, x3
    A = sprand(10, 10, 0.1)
    A = I + A * A'
    A = convert(SparseMatrixCSC{Tv,Ti}, A)
    factor = cholesky(A)
    factor2 = copy(factor)

    x = fill(Tv(1), 10)
    b = A * x
    x2 = zero(x)
    x3 = zero(x)

    ldiv!(x2, factor, b)
    ldiv!(x3, factor2, b)
    @test x2 ≈ x
    @test x3 ≈ x

    # Verify each copy has its own independent buffers
    @test getfield(factor, :Y) !== getfield(factor2, :Y)
end

end #end for Ti ∈ itypes

for Tv ∈ (Float32, Float64)
@testset "per-type buffers should be concretely typed" begin
    @test @inferred(SparseArrays.CHOLMOD.getcommon()) isa Base.RefValue
    F = cholesky(sparse(Tv[2 1; 1 2]))
    @test @inferred((F -> getfield(F, :Y)[])(F)) isa Ptr
end

@testset "Issue #9915" begin
    sparseI = sparse(Tv(1.0)I, 2, 2)
    @test sparseI \ sparseI == sparseI
end

@testset "test Sparse constructor Symmetric and Hermitian input (and issymmetric and ishermitian)" begin
    ACSC = sprandn(Tv, 10, 10, 0.3) + I
    @test issymmetric(Sparse(Symmetric(ACSC, :L)))
    @test issymmetric(Sparse(Symmetric(ACSC, :U)))
    @test ishermitian(Sparse(Hermitian(complex(ACSC), :L)))
    @test ishermitian(Sparse(Hermitian(complex(ACSC), :U)))
end

@testset "test Sparse constructor and read_sparse" begin
    # avoid dependenting on delimited files
    function writedlm(fn, title="", xs...)
        open(fn, "w") do file
            println(file, title)
            for i in xs
                println(file, i)
            end
        end
    end
    mktempdir() do temp_dir
        testfile = joinpath(temp_dir, "tmp.mtx")

        writedlm(testfile, "%%MatrixMarket matrix coordinate real symmetric","3 3 4","1 1 1","2 2 1","3 2 0.5","3 3 1")
        @test sparse(CHOLMOD.Sparse(testfile)) == [1 0 0;0 1 0.5;0 0.5 1]
        rm(testfile)

        writedlm(testfile, "%%MatrixMarket matrix coordinate complex Hermitian",
                        "3 3 4","1 1 1.0 0.0","2 2 1.0 0.0","3 2 0.5 0.5","3 3 1.0 0.0")
        @test sparse(CHOLMOD.Sparse(testfile)) == [1 0 0;0 1 0.5-0.5im;0 0.5+0.5im 1]
        rm(testfile)

        # this also tests that the error message is correctly retrieved from the library
        writedlm(testfile, "%%MatrixMarket matrix coordinate real symmetric","%3 3 4","1 1 1","2 2 1","3 2 0.5","3 3 1")
        @test_throws CHOLMOD.CHOLMODException("indices out of range") sparse(CHOLMOD.Sparse(testfile))
        rm(testfile)
    end
end

@testset "High level interface" for elty in (Tv, Complex{Tv})
    local A, b
    if elty <: Real
        A = randn(Tv, 5, 5)
        b = randn(Tv, 5)
    else
        A = complex.(randn(Tv, 5, 5), randn(Tv, 5, 5))
        b = complex.(randn(Tv, 5), randn(Tv, 5))
    end
    ADense = CHOLMOD.Dense(A)
    bDense = CHOLMOD.Dense(b)

    @test_throws BoundsError ADense[6, 1]
    @test_throws BoundsError ADense[1, 6]
    @test copy(ADense) == ADense
    @test CHOLMOD.norm_dense(ADense, 1) ≈ opnorm(A, 1)
    @test CHOLMOD.norm_dense(ADense, 0) ≈ opnorm(A, Inf)
    @test_throws ArgumentError CHOLMOD.norm_dense(ADense, 2)
    @test_throws ArgumentError CHOLMOD.norm_dense(ADense, 3)

    @test CHOLMOD.norm_dense(bDense, 2) ≈ norm(b)
    @test CHOLMOD.check_dense(bDense)

    AA = CHOLMOD.eye(3, Tv)
    unsafe_store!(convert(Ptr{Csize_t}, pointer(AA)), 2, 1) # change size, but not stride, of Dense
    @test convert(Matrix, AA) == Matrix(I, 2, 3)
end

@testset "Low level interface" begin
    @test isa(CHOLMOD.zeros(3, 3, Tv), CHOLMOD.Dense{Tv})
    @test isa(CHOLMOD.zeros(3, 3), CHOLMOD.Dense{Float64})
    @test isa(CHOLMOD.ones(3, 3, Tv), CHOLMOD.Dense{Tv})
    @test isa(CHOLMOD.ones(3, 3), CHOLMOD.Dense{Float64})
    @test isa(CHOLMOD.eye(3, 4, Tv), CHOLMOD.Dense{Tv})
    @test isa(CHOLMOD.eye(3, 4), CHOLMOD.Dense{Float64})
    @test isa(CHOLMOD.eye(3, Tv), CHOLMOD.Dense{Tv})
    @test isa(CHOLMOD.eye(3), CHOLMOD.Dense{Float64})
end

end # for Tv ∈ (Float32, Float64)

end # Base.USE_GPL_LIBS

end # module
