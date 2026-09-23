# This file is a part of Julia. License is MIT: https://julialang.org/license

module CHOLMODLifetimeTests
using Test

using SparseArrays
using SparseArrays.CHOLMOD
using SparseArrays.CHOLMOD: getcommon
using SparseArrays.LibSuiteSparse
using SparseArrays.LibSuiteSparse: cholmod_l_allocate_sparse, cholmod_allocate_sparse,
    cholmod_l_allocate_dense, cholmod_allocate_dense
using LinearAlgebra: I, cholesky, diag, ldiv!, ldlt, qr, Symmetric
using Random

# Run in a fresh process: intentional collections exercise finalization and rooting.
# The constructors must free the pointer before throwing, so the Common's
# allocation count must be back to its previous value afterwards. The GC is
# disabled around each measurement so that finalizers of unrelated CHOLMOD
# objects cannot change the count between the baseline read and the check.
malloc_count(T) = getcommon(T)[].malloc_count
function with_gc_disabled(f)
    GC.gc()
    enabled = GC.enable(false)
    try
        f()
    finally
        GC.enable(enabled)
    end
end
itypes = sizeof(Int) == 4 ? (Int32,) : (Int32, Int64)
for Ti ∈ itypes, Tv ∈ (Float32, Float64)
Random.seed!(123)

@testset "illegal dtype" begin
    p = Ti == Int64 ? cholmod_l_allocate_sparse(1, 1, 1, true, true, 0, CHOLMOD.xdtyp(Tv), getcommon(Ti)) :
        cholmod_allocate_sparse(1, 1, 1, true, true, 0, CHOLMOD.xdtyp(Tv), getcommon(Ti))
    puint = convert(Ptr{UInt32}, p)
    # The second argument 5 is the invalid `dtype`.
    # CHOLMOD_DOUBLE (0) and CHOLMOD_SINGLE (4) are both valid.
    unsafe_store!(puint, 5, 3*div(sizeof(Csize_t), 4) + 5*div(sizeof(Ptr{Cvoid}), 4) + 4)
    @test_throws CHOLMOD.CHOLMODException CHOLMOD.Sparse(p)
end

@testset "illegal xtype" begin
    with_gc_disabled() do
        nmalloc = malloc_count(Ti)
        p = Ti == Int64 ? cholmod_l_allocate_sparse(1, 1, 1, true, true, 0, CHOLMOD.xdtyp(Tv), getcommon(Ti)) :
            cholmod_allocate_sparse(1, 1, 1, true, true, 0, CHOLMOD.xdtyp(Tv), getcommon(Ti))
        puint = convert(Ptr{UInt32}, p)
        # The second argument 3 is the invalid `xtype`.
        # CHOLMOD_REAL (1), CHOLMOD_COMPLEX (2) are valid.
        unsafe_store!(puint, 3, 3*div(sizeof(Csize_t), 4) + 5*div(sizeof(Ptr{Cvoid}), 4) + 3)
        @test_throws CHOLMOD.CHOLMODException CHOLMOD.Sparse(p)
        @test malloc_count(Ti) == nmalloc
    end
end

@testset "illegal dense xtype" begin
    with_gc_disabled() do
        # `free!(::Ptr{cholmod_dense})` always uses the native-Int Common
        nmalloc = malloc_count(Int)
        p = sizeof(Int) == 8 ? cholmod_l_allocate_dense(1, 1, 1, CHOLMOD.xdtyp(Tv), getcommon(Int)) :
            cholmod_allocate_dense(1, 1, 1, CHOLMOD.xdtyp(Tv), getcommon(Int))
        xtype_offset = fieldoffset(LibSuiteSparse.cholmod_dense, findfirst(==(:xtype), fieldnames(LibSuiteSparse.cholmod_dense)))
        unsafe_store!(Ptr{Cint}(p + xtype_offset), 3) # CHOLMOD_ZOMPLEX is not supported
        @test_throws CHOLMOD.CHOLMODException CHOLMOD.Dense(p)
        @test malloc_count(Int) == nmalloc
    end
end

# Test that a bogus `itype` raises the expected exception
@testset "illegal itype I" begin
    p = Ti == Int64 ? cholmod_l_allocate_sparse(1, 1, 1, true, true, 0, CHOLMOD.xdtyp(Tv), getcommon(Ti)) :
        cholmod_allocate_sparse(1, 1, 1, true, true, 0, CHOLMOD.xdtyp(Tv), getcommon(Ti))
    puint = convert(Ptr{UInt32}, p)
    # The second argument to `unsafe_store!` is the illegal `itype`
    unsafe_store!(puint, 123, 3*div(sizeof(Csize_t), 4) + 5*div(sizeof(Ptr{Cvoid}), 4) + 2)
    @test_throws CHOLMOD.CHOLMODException CHOLMOD.Sparse(p)
end

@testset "illegal itype II" begin
    p = Ti == Int64 ? cholmod_l_allocate_sparse(1, 1, 1, true, true, 0, CHOLMOD.xdtyp(Tv), getcommon(Ti)) :
        cholmod_allocate_sparse(1, 1, 1, true, true, 0, CHOLMOD.xdtyp(Tv), getcommon(Ti))
    puint = convert(Ptr{UInt32}, p)
    unsafe_store!(puint,  5, 3*div(sizeof(Csize_t), 4) + 5*div(sizeof(Ptr{Cvoid}), 4) + 2)
    @test_throws CHOLMOD.CHOLMODException CHOLMOD.Sparse(p)
end

@testset "test free! $Ti" begin
    p = Ti == Int64 ? cholmod_l_allocate_sparse(1, 1, 1, true, true, 0, CHOLMOD.xdtyp(Tv), getcommon(Ti)) :
        cholmod_allocate_sparse(1, 1, 1, true, true, 0, CHOLMOD.xdtyp(Tv), getcommon(Ti))
    @test CHOLMOD.free!(p, Ti)

    # Object-level free! must null the wrapper's pointer so that a second
    # free! (and the finalizer) is a no-op rather than a double free.
    D = CHOLMOD.Dense(rand(Tv, 3))
    @test CHOLMOD.free!(D)
    @test getfield(D, :ptr) == C_NULL
    @test_throws ArgumentError pointer(D)
    @test !CHOLMOD.free!(D)

    S = CHOLMOD.Sparse(convert(SparseMatrixCSC{Tv,Ti}, sparse(I, 3, 3)))
    @test CHOLMOD.free!(S)
    @test getfield(S, :ptr) == C_NULL
    @test_throws ArgumentError pointer(S)
    @test !CHOLMOD.free!(S)

    # A Factor that has been used in ldiv! owns Y/E scratch buffers; free! must
    # release them and null the handles as well as the factor pointer.
    A = convert(SparseMatrixCSC{Tv,Ti}, sparse(Tv[4 1 0; 1 4 1; 0 1 4]))
    F = cholesky(A)
    b = fill(Tv(1), 3)
    ldiv!(similar(b), F, b)
    # cholmod_solve2 always allocates Y; E is only allocated when needed.
    @test getfield(F, :Y)[] != C_NULL
    @test CHOLMOD.free!(F)
    @test getfield(F, :ptr) == C_NULL
    @test getfield(F, :Y)[] == C_NULL
    @test getfield(F, :E)[] == C_NULL
    @test_throws ArgumentError pointer(F)
    @test !CHOLMOD.free!(F)

    D = S = F = nothing
    GC.gc()
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

# For an Int64 factor both Commons coincide, so the check is only meaningful
# for Ti == Int32.
if Ti == Int32 && Int64 in itypes
@testset "free!(Factor) releases Y/E through the matching Common $Tv $Ti" begin
    local A, b, x, F
    A = sprand(10, 10, 0.1)
    A = I + A * A'
    A = convert(SparseMatrixCSC{Tv,Ti}, A)
    b = A * fill(Tv(1), 10)
    x = zero(b)
    with_gc_disabled() do
        n64 = malloc_count(Int64)
        F = cholesky(A)
        ldiv!(x, F, b) # allocates the Y/E buffers in the Int32 Common
        CHOLMOD.free!(F)
        # Y/E must be released through the Int32 Common as well; freeing them
        # through the Int64 Common would decrement its count by two.
        @test malloc_count(Int64) == n64
    end
end
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

@testset "temporaries stay rooted while reading raw pointers $Tv $Ti" begin
    # The conversions below read through the raw CHOLMOD buffers of a wrapper
    # that is otherwise dead after `unsafe_load(pointer(A))`. If the wrapper is
    # not kept rooted, a GC triggered by an allocation during the copy can run
    # its finalizer and free the buffers mid-read. Not a deterministic
    # reproducer, but exercises the preserved paths under GC pressure.
    local S, SPD, Fref
    S = convert(SparseMatrixCSC{Tv,Ti}, sprand(400, 300, 0.05))
    SPD = convert(SparseMatrixCSC{Tv,Ti}, S[1:300, :] * S[1:300, :]' + 300I)
    Fref = cholesky(SPD)
    for _ in 1:20
        @test SparseMatrixCSC(CHOLMOD.Sparse(S)) == S
        GC.gc(false)
        @test sparse(CHOLMOD.Sparse(S)) == S
        GC.gc(false)
        @test sparsevec(CHOLMOD.Sparse(S[:, 1])) == S[:, 1]
        GC.gc(false)
        @test sparse(CHOLMOD.Sparse(Symmetric(SPD))) == Symmetric(SPD)
        GC.gc(false)
        @test diag(cholesky(SPD)) ≈ diag(Fref)
        GC.gc(false)
        @test cholesky(SPD).p == Fref.p
        GC.gc(false)
        @test CHOLMOD.get_perm(ldlt(SPD)) == ldlt(SPD).p
        GC.gc(false)
        @test CHOLMOD.Sparse(S)[7, 3] == S[7, 3]
        @test CHOLMOD.Dense(Vector(S[:, 2]))[5] == S[5, 2]
        GC.gc(false)
        @test Matrix(CHOLMOD.Dense(Matrix(S[1:20, 1:20]))) == Matrix(S[1:20, 1:20])
        GC.gc(false)
    end
end

# For Int64 both Commons coincide, so the check is only meaningful for Ti == Int32.
if Ti == Int32 && Int64 in itypes && Tv == Float64
@testset "qr releases its outputs through the matching Common $Ti" begin
    A = SparseMatrixCSC{Tv,Ti}(sprand(20, 10, 0.3) + sparse(1:10, 1:10, 1.0, 20, 10))
    qr(A)
    GC.gc()
    n32, n64 = getcommon(Int32)[].memory_inuse, getcommon(Int64)[].memory_inuse
    for _ in 1:10
        qr(A)
    end
    GC.gc()
    @test getcommon(Int32)[].memory_inuse == n32
    @test getcommon(Int64)[].memory_inuse == n64
end
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

end # Ti, Tv

end # module
