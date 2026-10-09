# This file is a part of Julia. License is MIT: https://julialang.org/license

module UMFPACKTests
using Test

using SparseArrays
using Serialization
using LinearAlgebra:
    LinearAlgebra, I, det, diag, issuccess, ldiv!, lu, lu!, Transpose, SingularException, Diagonal, logabsdet, Symmetric, Hermitian
using SparseArrays: nnz, sparse, SparseMatrixCSC, UMFPACK, increment!
include("../testhelpers.jl")

function umfpack_report(l::UMFPACK.UmfpackLU)
    UMFPACK.umfpack_report_numeric(l, 0)
    UMFPACK.umfpack_report_symbolic(l, 0)
    return
end

_isnull_numeric(F::UMFPACK.UmfpackLU) = F.numeric.p == C_NULL

const TransposeFact = isdefined(LinearAlgebra, :TransposeFactorization) ?
    LinearAlgebra.TransposeFactorization :
    Transpose

# A standard run factorizes with the build's `Int` indices, and `Float64` elements where
# the element type is not the point of the test.
const ITYPES = core_itypes
# The C entry points are generated per index type. The testsets that between them call
# each one (solve and report, the factors and the determinant, a column ordering) take
# both index types in a comprehensive run; the others would repeat them.
const KERNEL_ITYPES = @static COMPREHENSIVE ? itypes : core_itypes
const ELTYPES = @static COMPREHENSIVE ? STD_ELTYPES : (Float64,)

for itype in UMFPACK.UmfpackIndexTypes
    sol_r = Symbol(UMFPACK.umf_nm("solve", :Float64, itype))
    sol_c = Symbol(UMFPACK.umf_nm("solve", :ComplexF64, itype))
    @eval begin
        function alloc_solve!(x::StridedVector{Float64}, lu::UMFPACK.UmfpackLU{Float64,$itype}, b::StridedVector{Float64}, typ::Integer)
            if x === b
                throw(ArgumentError("output array must not be aliased with input array"))
            end
            if stride(x, 1) != 1 || stride(b, 1) != 1
                throw(ArgumentError("in and output vectors must have unit strides"))
            end
            UMFPACK.umfpack_numeric!(lu)
            (size(b,1) == lu.m) && (size(b) == size(x)) || throw(DimensionMismatch())
            UMFPACK.@isok UMFPACK.$sol_r(typ, lu.colptr, lu.rowval, lu.nzval,
                        x, b, lu.numeric, lu.control,
                        lu.info)
            return x
        end
        function alloc_solve!(x::StridedVector{ComplexF64}, lu::UMFPACK.UmfpackLU{ComplexF64,$itype}, b::StridedVector{ComplexF64}, typ::Integer)
            if x === b
                throw(ArgumentError("output array must not be aliased with input array"))
            end
            if stride(x, 1) != 1 || stride(b, 1) != 1
                throw(ArgumentError("in and output vectors must have unit strides"))
            end
            UMFPACK.umfpack_numeric!(lu)
            (size(b, 1) == lu.m) && (size(b) == size(x)) || throw(DimensionMismatch())
            UMFPACK.@isok UMFPACK.$sol_c(typ, lu.colptr, lu.rowval, lu.nzval, C_NULL, x, C_NULL, b,
                        C_NULL, lu.numeric, lu.control, lu.info)
            return x
        end
    end
end

@testset "Workspace management" begin
    A0 = fixture(Float64, 100, 100) + 100I
    b0 = Float64.(1:100)
    bn0 = reshape(Float64.(1:2000), 100, 20)
    @testset "Core functionality for $Tv elements" for Tv in ELTYPES
        for Ti in KERNEL_ITYPES
            A = convert(SparseMatrixCSC{Tv,Ti}, A0)
            Af = lu(A)
            umfpack_report(Af)
            b = convert(Vector{Tv}, b0)
            x = alloc_solve!(
                similar(b),
                Af, b,
                UMFPACK.UMFPACK_A)
            @test (@static COMPREHENSIVE ? A : Af) \ b == x
            bn = convert(Matrix{Tv}, bn0)
            xn = similar(bn)
            for i in 1:20
                xn[:, i] .= alloc_solve!(
                    similar(bn[:, i]),
                    Af, bn[:, i],
                    UMFPACK.UMFPACK_A)
            end
            @test (@static COMPREHENSIVE ? A : Af) \ bn == xn
            umfpack_report(Af)
        end
    end
    function f(Tv, Ti)
        A = convert(SparseMatrixCSC{Tv,Ti}, A0)
        Af = lu(A)
        umfpack_report(Af)
        b = convert(Vector{Tv}, b0)
        x = similar(b)
        ws = UMFPACK.UmfpackWS(Af)
        ldiv!(x, Af, b; workspace = ws)
        aloc1 = @allocated ldiv!(x, Af, b; workspace = ws)
        bn = convert(Matrix{Tv}, bn0)
        xn = similar(bn)
        ldiv!(xn, Af, bn; workspace = ws)
        aloc2 = @allocated ldiv!(xn, Af, bn; workspace = ws)
        umfpack_report(Af)
        return aloc1 + aloc2
    end
    @testset "Allocations" begin
        for Tv in Base.uniontypes(UMFPACK.UMFVTypes),
            Ti in ITYPES
            f(Tv, Ti)
            f(Tv, Ti)
            @test f(Tv, Ti) == 0
        end
    end
    @testset "test similar" begin
        Af = lu(A0)
        umfpack_report(Af)
        ws = UMFPACK.UmfpackWS(Af)
        sim = similar(ws)
        for f in [typeof, length],
            p in [:Wi, :W]
            @test f(getproperty(sim, p)) == f(getproperty(ws, p))
            @test getproperty(sim, p) !== getproperty(ws, p)
        end
        umfpack_report(Af)
    end
    function test_ws_dup(Af, Af1)
        for i in [:colptr, :rowval, :nzval, :control, :info]
            @test getproperty(Af, i) == getproperty(Af1, i)
            @test getproperty(Af, i) !== getproperty(Af1, i)
        end
        for i in [:n, :m, :status]
            @test getproperty(Af, i) == getproperty(Af1, i)
        end
        @test Af1.lock !== Af.lock
        @test Af1.symbolic.p != Af.symbolic.p && Af1.numeric.p != Af.numeric.p
    end
    @testset "test copy(UmfpackLU)" begin
        Af = lu(A0)
        umfpack_report(Af)
        test_ws_dup(Af, copy(Af))
        # a copied wrapper has a factorization of its own, as a copied factorization has
        test_ws_dup(Af, parent(copy(transpose(Af))))
        test_ws_dup(Af, parent(copy(adjoint(Af))))
        @test copy(transpose(Af)) isa typeof(transpose(Af))
        # the workspace argument is accepted for compatibility
        test_ws_dup(Af, copy(Af, UMFPACK.UmfpackWS(Af)))
        umfpack_report(Af)
    end
end

@testset "UMFPACK wrappers" begin
    @static if COMPREHENSIVE
    se33 = sparse(1.0I, 3, 3)
    do33 = fill(1., 3)
    @test isequal(se33 \ do33, do33)
    end

    # based on deps/Suitesparse-4.0.2/UMFPACK/Demo/umfpack_di_demo.c

    A0 = sparse(increment!([0,4,1,1,2,2,0,1,2,3,4,4]),
                increment!([0,4,0,2,1,2,1,4,3,2,1,2]),
                [2.,1.,3.,4.,-1.,-3.,3.,6.,2.,1.,4.,2.], 5, 5)

    @testset "Core functionality for $Tv elements" for Tv in (Float64, ComplexF64)
        # We might be able to support two index sizes one day
        for Ti in ITYPES
            A = convert(SparseMatrixCSC{Tv,Ti}, A0)
            lua = lu(A)
            umfpack_report(lua)
            @test nnz(lua) == 18
            @test_throws isdefined(Base, :FieldError) ? FieldError : ErrorException lua.Z
            L,U,p,q,Rs = lua.:(:)
            @test L == lua.L
            @test U == lua.U
            @test p == lua.p
            @test q == lua.q
            @test Rs == lua.Rs
            @test (Diagonal(Rs) * A)[p,q] ≈ L * U

            @test det(lua) ≈ det(Array(A))
            logdet_lua, sign_lua = logabsdet(lua)
            logdet_A, sign_A = logabsdet(Array(A))
            @test logdet_lua ≈ logdet_A
            @test sign_lua ≈ sign_A

            b = [8., 45., -3., 3., 19.]
            x = lua\b
            @test x ≈ float([1:5;])

            @test A*x ≈ b
            z = complex.(b)
            x = ldiv!(lua, z)
            @test x ≈ float([1:5;])
            @test z === x
            y = similar(z)
            ldiv!(y, lua, complex.(b))
            @test y ≈ x

            @test A*x ≈ b

            b = [8., 20., 13., 6., 17.]
            x = lua'\b
            @test x ≈ float([1:5;])

            @test A'*x ≈ b
            z = complex.(b)
            x = ldiv!(adjoint(lua), z)
            @test x ≈ float([1:5;])
            @test x === z
            y = similar(x)
            ldiv!(y, adjoint(lua), complex.(b))
            @test y ≈ x

            @test A'*x ≈ b
            @test transpose(lua) isa TransposeFact
            x = transpose(lua) \ b
            @test x ≈ float([1:5;])

            @test transpose(A) * x ≈ b
            x = ldiv!(transpose(lua), complex.(b))
            @test x ≈ float([1:5;])
            y = similar(x)
            ldiv!(y, transpose(lua), complex.(b))
            @test y ≈ x

            @test transpose(A) * x ≈ b

            lua = lu(A')
            x = lua \ b
            @test A'*x ≈ b

            lua = lu(transpose(A))
            x = lua \ b
            @test transpose(A)*x ≈ b

            for W in (@static COMPREHENSIVE ? (Symmetric(A), Hermitian(view(A, 1:3, 1:3))) : (Tv <: Real ? Symmetric(A) : Hermitian(A),))
                F = lu(W)
                @test F isa UMFPACK.UmfpackLU
                @test Matrix(W) * (F \ b[1:size(W, 1)]) ≈ b[1:size(W, 1)]
            end

            # Element promotion and type inference
            @inferred lua\fill(1, size(A, 2))
            umfpack_report(lua)
        end
    end

    @static if COMPREHENSIVE
    @testset "More tests for complex cases" begin
        Ac0 = complex.(A0,A0)
        for Ti in Base.uniontypes(UMFPACK.UMFITypes)
            Ac = convert(SparseMatrixCSC{ComplexF64,Ti}, Ac0)
            x  = fill(1.0 + im, size(Ac,1))
            lua = lu(Ac)
            umfpack_report(lua)
            L,U,p,q,Rs = lua.:(:)
            @test (Diagonal(Rs) * Ac)[p,q] ≈ L * U
            b  = Ac*x
            @test Ac\b ≈ x
            b  = Ac'*x
            @test Ac'\b ≈ x
            b  = transpose(Ac)*x
            @test transpose(Ac)\b ≈ x
            umfpack_report(lua)
        end
    end
    end

    @static if COMPREHENSIVE
    @testset "Rectangular cases. elty=$elty, m=$m, n=$n" for
        elty in (Float64, ComplexF64),
            (m, n) in ((10,5), (5, 10))

        # UMFPACK takes the pivots from the first min(m, n) columns of its ordering and
        # reports a zero pivot when those are dependent, although the matrix has full rank.
        # The fixture's columns 1, 4, 7 and 10 span two rows, so the entries that give it
        # full rank go in the last columns, where the ordering finds nonzero pivots.
        k = min(m, n)
        A = fixture(elty, m, n) + sparse(1:k, n-k+1:n, elty == Float64 ? Float64.(1:k) : complex.(1.0:k, -1.0), m, n)
        F = lu(A)
        umfpack_report(F)
        L, U, p, q, Rs = F.:(:)
        @test (Diagonal(Rs) * A)[p,q] ≈ L * U
        @test (L, U, p, q, Rs) == (F.L, F.U, F.p, F.q, F.Rs)
        umfpack_report(F)
    end
    end

    @static if COMPREHENSIVE
    @testset "Issue #4523 - complex sparse \\" begin
        A, b = sparse((1.0 + im)I, 2, 2), fill(1., 2)
        @test A * (lu(A)\b) ≈ b

        @test det(sparse([1,3,3,1], [1,1,3,3], [1,1,1,1])) == 0
    end
    end

    @testset "UMFPACK_ERROR_n_nonpositive" begin
        @test_throws ArgumentError lu(sparse(Int[], Int[], Float64[], 5, 0))
    end

    @testset "Issue #15099" begin
        testtypes = [
            (ComplexF32, ComplexF64),
            (Float32, Float64),
            (Int, Float64),
            (Float16, Float64),
        ]
        testtypes = @static COMPREHENSIVE ? testtypes : [(ComplexF32, ComplexF64), (Int, Float64)]

        for (Tin, Tout) in testtypes
            F = lu(sparse(fill(Tin(1), 1, 1)))
            umfpack_report(F)
            L = sparse(fill(Tout(1), 1, 1))
            @test F.p == F.q == [1]
            @test F.Rs == [1.0]
            @test mismatch(F.L, Matrix(L)) === nothing
            @test mismatch(F.U, Matrix(L)) === nothing
            @test F.:(:) == (L, L, [1], [1], [1.0])
            umfpack_report(F)
        end
    end

    @testset "BigFloat not supported" for T in (BigFloat, (@static COMPREHENSIVE ? (Complex{BigFloat},) : ())...)
        @test_throws ArgumentError lu(sparse(fill(T(1), 1, 1)))
    end

    @testset "size(::UmfpackLU)" begin
        m = n = 1
        F = lu(sparse(fill(1., m, n)))
        umfpack_report(F)
        @test size(F) == (m, n)
        @test size(F, 1) == m
        @test size(F, 2) == n
        @test size(F, 3) == 1
        @test_throws ArgumentError size(F,-1)
        umfpack_report(F)
    end

    @testset "aliased solution and right-hand side" begin
        A = sparse([2.0 1 0; 1 3 1; 0 1 4])
        F = lu(A)
        # iterative refinement reads the right-hand side again after writing the solution
        F.control[UMFPACK.JL_UMFPACK_IRSTEP] = 2
        B = A * [1.0 2; 3 4; 5 6]
        @test ldiv!(view(B, :, 1), F, view(vec(B), 1:3)) ≈ [1, 3, 5]
        B = A * [1.0 2 3; 4 5 6; 7 8 9]
        @test ldiv!(view(B, :, 2:3), F, view(B, :, 1:2)) ≈ [1.0 2; 4 5; 7 8]
    end

    @testset "propertynames(::UmfpackLU)" begin
        F = lu(sparse([4.0 1 0; 1 4 1; 0 1 4]))
        @test propertynames(F) == (:L, :U, :p, :q, :Rs, :(:))
        @test hasproperty(F, :(:))
        @test :numeric ∉ propertynames(F)
        @test :numeric ∈ propertynames(F, true)
    end

    @static if COMPREHENSIVE
    @testset "Issues #18246,18244 - lu sparse pivot" begin
        A = sparse(1.0I, 4, 4)
        A[1:2,1:2] = [-.01 -200; 200 .001]
        F = lu(A)
        umfpack_report(F)
        @test F.p == [3 ; 4 ; 2 ; 1]
    end
    end

    @testset "Test that A[c|t]_ldiv_B!{T<:Complex}(X::StridedMatrix{T}, lu::UmfpackLU{Float64}, B::StridedMatrix{T}) works as expected." begin
        N = 10
        A = N*I + fixture(Float64, N, N)
        X = zeros(ComplexF64, N, N)
        B = Matrix(fixture(ComplexF64, N, N))
        luA, lufA = lu(A), lu(Array(A))
        umfpack_report(luA)
        @test ldiv!(copy(X), luA, B) ≈ ldiv!(copy(X), lufA, B)
        # a vector right-hand side has a kernel of its own
        @test ldiv!(X[:, 1], luA, B[:, 1]) ≈ ldiv!(copy(X), lufA, B)[:, 1]
        @static if COMPREHENSIVE
        @test ldiv!(copy(X), adjoint(luA), B) ≈ ldiv!(copy(X), adjoint(lufA), B)
        @test ldiv!(copy(X), transpose(luA), B) ≈ ldiv!(copy(X), transpose(lufA), B)
        end
        umfpack_report(luA)
    end

    @testset "singular matrix" begin
        for A in sparse.((Float64[1 2; 0 0], (@static COMPREHENSIVE ? (ComplexF64[1 2; 0 0],) : ())...))
            @test_throws SingularException lu(A)
            @test !issuccess(lu(A; check = false))
        end
    end

    @testset "rcond (#118) for $Tv, $Ti" for Tv in ELTYPES, Ti in ITYPES
        # the number is min/max of |diag(U)| of the row-scaled matrix UMFPACK factorized
        F = lu(SparseMatrixCSC{Tv,Ti}(sparse(Tv[1 3; 0 1])))
        @test UMFPACK.rcond(F) === 0.25
        @test UMFPACK.rcond(F) === minimum(abs, diag(F.U)) / maximum(abs, diag(F.U))
        # row scaling is on by default, so a diagonal matrix is perfectly conditioned
        @test UMFPACK.rcond(lu(SparseMatrixCSC{Tv,Ti}(sparse(Diagonal(Tv[1, 2, 4]))))) === 1.0
        # 1-by-1 and singular special cases
        @test UMFPACK.rcond(lu(SparseMatrixCSC{Tv,Ti}(sparse(Diagonal(Tv[3]))))) === 1.0
        @test UMFPACK.rcond(lu(SparseMatrixCSC{Tv,Ti}(sparse(Tv[1 2; 0 0])); check=false)) === 0.0
        # a factor without a numeric decomposition gets one on demand
        G = UMFPACK.UmfpackLU(SparseMatrixCSC{Tv,Ti}(sparse(Tv[1 3; 0 1])))
        @test UMFPACK.rcond(G) === 0.25
        # lu! refreshes the estimate
        lu!(F, SparseMatrixCSC{Tv,Ti}(sparse(Tv[1 1; 0 1])))
        @test UMFPACK.rcond(F) === 0.5
    end

    @static if COMPREHENSIVE
    @testset "deserialization" begin
        A  = 10*I + fixture(Float64, 10, 10)
        F1 = lu(A)

        umfpack_report(F1)
        b  = IOBuffer()
        serialize(b, F1)
        seekstart(b)
        F2 = deserialize(b)
        for nm in (:colptr, :m, :n, :nzval, :rowval, :status)
            @test getfield(F1, nm) == getfield(F2, nm)
        end
        b1 = IOBuffer()
        serialize(b1, (a=F1, b=F2))
        seekstart(b1)
        x = deserialize(b1)
        lu!(x.a)
        lu!(x.b)
        for nm in (:colptr, :m, :n, :nzval, :rowval, :status)
            @test getfield(F1, nm) == getfield(x.a, nm) == getfield(x.b, nm)
        end

        umfpack_report(F1)
        umfpack_report(F2)
        umfpack_report(x.a)
        umfpack_report(x.b)
    end
    end

    @testset "Do/do not reuse symbolic LU factorization" for reuse ∈ (true, false)
        A1 = sparse(increment!([0,4,1,1,2,2,0,1,2,3,4,4]),
                    increment!([0,4,0,2,1,2,1,4,3,2,1,2]),
                    [2.,1.,3.,4.,-1.,-3.,3.,9.,2.,1.,4.,2.], 5, 5)
        testtypes = [ComplexF64, Float64]
        cases = @static COMPREHENSIVE ? eachvalue((true, false), testtypes, itypes) : [(reuse, Float64, Int)]
        for (r, Tv, Ti) in cases
            # (Float64, Int) runs once under each `reuse`, whether or not the grid pairs them
            if r == reuse || (Tv, Ti) == (Float64, Int) && (reuse, Tv, Ti) ∉ cases
                A = convert(SparseMatrixCSC{Tv,Ti}, A0)
                B = convert(SparseMatrixCSC{Tv,Ti}, A1)
                b = Tv[8., 45., -3., 3., 19.]
                F = lu(A)
                umfpack_report(F)
                lu!(F, B; reuse_symbolic=reuse)
                umfpack_report(F)
                @test F\b ≈ Matrix(B)\b
                @static COMPREHENSIVE && @test B\b ≈ Matrix(B)\b

                # singular matrix
                C = copy(B)
                C[4, 3] = Tv(0)
                F = lu(A)
                umfpack_report(F)
                @test_throws SingularException lu!(F, C; reuse_symbolic=reuse)
                # change of nonzero pattern
                D = copy(B)
                D[5, 1] = Tv(1.0)
                F = lu(A)
                umfpack_report(F)
                if reuse
                    # rejected before F is written to, so F still factorizes A
                    @test_throws ArgumentError lu!(F, D; reuse_symbolic=reuse)
                    umfpack_report(F)
                    @test F\b ≈ Matrix(A)\b
                else
                    lu!(F, D; reuse_symbolic=reuse)
                    umfpack_report(F)
                    @test F\b ≈ Matrix(D)\b
                    @static COMPREHENSIVE && @test D\b ≈ Matrix(D)\b
                end
            end
        end
    end

    @testset "F.Rs and logabsdet when UMFPACK divides by the scale factors, $Tv, $Ti" for
            Tv in ELTYPES, Ti in KERNEL_ITYPES
        # UMFPACK stores reciprocal scale factors for badly scaled rows
        A = SparseMatrixCSC{Tv,Ti}(sparse(Tv[1e-20 2e-20 0; 0 1 3; 1 0 1]))
        F = lu(A)
        @test F.L * F.U ≈ (F.Rs .* A)[F.p, F.q]
        L, U, p, q, Rs = F.:(:)
        @test Rs == F.Rs
        @test all(logabsdet(F) .≈ logabsdet(Matrix(A)))
        @test det(F) ≈ det(Matrix(A))
        @static if COMPREHENSIVE
        B = SparseMatrixCSC{Tv,Ti}(1e-15 * (fixture(Tv, 50, 50) + 50I))
        @test all(logabsdet(lu(B)) .≈ logabsdet(Matrix(B)))
        end
    end

    @testset "factors are rebuilt on demand, $Tv, $Ti" for
            Tv in ELTYPES, Ti in ITYPES
        A = SparseMatrixCSC{Tv,Ti}(sparse(Tv[4 1; 1 3]))
        for G in (UMFPACK.UmfpackLU(A), deserialize(seekstart(let io = IOBuffer(); serialize(io, lu(A)); io; end)))
            @test det(G) ≈ det(Matrix(A))
            @test nnz(G) == nnz(lu(A))
        end
        # a failed factorization stays failed across serialization
        S = SparseMatrixCSC{Tv,Ti}(sparse(Tv[1 2; 2 4]))
        io = IOBuffer(); serialize(io, lu(S; check=false)); seekstart(io)
        G = deserialize(io)
        @test !issuccess(G)
        @test occursin("Failed factorization", sprint(show, MIME"text/plain"(), G))
    end

    @testset "failed lu! drops the old numeric factorization, $Tv, $Ti" for
            Tv in ELTYPES, Ti in ITYPES
        A = SparseMatrixCSC{Tv,Ti}(sparse(Tv[4 1 0; 1 4 1; 0 1 4]))
        B = SparseMatrixCSC{Tv,Ti}(sparse(Tv[5 1 0; 1 5 1; 0 1 5]))
        F = lu(A)
        @test_throws ArgumentError lu!(F, B; reuse_symbolic=false, q=[1, 1, 2])
        @test !issuccess(F)
        @test _isnull_numeric(F)
        # anything needing the factors refactorizes the matrix now held, B
        @test all(logabsdet(F) .≈ logabsdet(Matrix(B)))
        @test F \ ones(3) ≈ Matrix(B) \ ones(3)
    end

    @testset "lu!(F, S) validates S before mutating F, $Ti" for Ti in ITYPES
        A = SparseMatrixCSC{Float64,Ti}(sparse([4.0 1; 1 3]))
        F = lu(A)
        L, U = F.L, F.U
        @test_throws ArgumentError lu!(F, sparse(ComplexF64[4 1 0; 1 3 0; 0 0 1im]))
        @test size(F) == (2, 2)
        @test F.L == L && F.U == U
        @test F \ [1.0, 2.0] ≈ Matrix(A) \ [1.0, 2.0]
        # integer and differently indexed inputs convert, and the workspace follows the new size
        C = sparse([4 1 0; 1 3 0; 0 0 1])
        lu!(F, C)
        @test size(F) == (3, 3)
        @test F \ [1.0, 2.0, 3.0] ≈ Matrix(C) \ [1.0, 2.0, 3.0]
        lu!(F, sparse([4 1; 1 3]))
        @test F \ [1.0, 2.0] ≈ Matrix(A) \ [1.0, 2.0]
        Fc = lu(SparseMatrixCSC{ComplexF64,Ti}(A))
        lu!(Fc, sparse([4 1 0; 1 3 0; 0 0 1]))
        @test Fc \ ComplexF64[1, 2, 3] ≈ Matrix(C) \ ComplexF64[1, 2, 3]
    end

    @static if COMPREHENSIVE
    @testset "lu!(F, S) does not reuse the symbolic factors of another size, $Ti" for Ti in ITYPES
        F = lu(SparseMatrixCSC{Float64,Ti}(sparse([4.0 1; 1 3])))
        C = sparse([4.0 1 0; 1 3 0; 0 0 7])
        lu!(F, C)
        @test size(F.L) == (3, 3)
        @test F \ [1.0, 2.0, 3.0] ≈ Matrix(C) \ [1.0, 2.0, 3.0]
        @test det(F) ≈ det(Matrix(C))
    end

    @testset "factors are read under the lock, $Ti" for Ti in ITYPES
        A = SparseMatrixCSC{Float64,Ti}(sparse([4.0 1; 1 3]))
        F = lu(A)
        for f in (F -> F.L, F -> F.:(:), logabsdet, F -> lu!(F, A))
            lock(F.lock)
            t = @async f(F)
            yield()
            @test !istaskdone(t)
            unlock(F.lock)
            @test timedwait(() -> istaskdone(t), 60; pollint=0.001) === :ok
        end
        @test F \ [1.0, 2.0] ≈ Matrix(A) \ [1.0, 2.0]
    end
    end

    @testset "keywords reach converted eltypes and any q vector, $Ti" for Ti in KERNEL_ITYPES
        A = sparse([4.0 1 0; 1 4 1; 0 1 4])
        b = [1.0, 2.0, 3.0]
        x = Matrix(A) \ b
        # the element type is converted before the keywords reach anything index-specific
        for S in (SparseMatrixCSC{Float32,Ti}(A), (@static COMPREHENSIVE ? (Ti == Int ? (SparseMatrixCSC{ComplexF32,Ti}(A),
                  SparseMatrixCSC{Int,Ti}(A)) : ()) : ())..., SparseMatrixCSC{Float64,Ti}(A))
            @test lu(S; q=[3, 2, 1]) \ b ≈ x
            @static if COMPREHENSIVE
            @test lu(S; q=Int32[2, 1, 0]) \ b ≈ x
            end
            @test lu(S; q=3:-1:1) \ b ≈ x
            # an ordering UMFPACK does not choose by itself, and an odd permutation
            F = lu(S; q=[1, 3, 2])
            @test F.q == [1, 3, 2] != lu(S).q
            @test all(logabsdet(F) .≈ logabsdet(Matrix(A)))
            @test lu(S; control=UMFPACK.get_umfpack_control(Float64, Ti)) \ b ≈ x
        end
        @test lu(SparseMatrixCSC{ComplexF64,Ti}(A); q=[3, 2, 1]) \ complex(b) ≈ x
        @test_throws DimensionMismatch lu(SparseMatrixCSC{Float64,Ti}(A); q=[1, 2])
    end

    @testset "non-square det and \\ throw DimensionMismatch, $Tv, $m×$n" for
            Tv in (Float64, ComplexF64), (m, n) in FIXTURE_SHAPES[1:2]
        A = fixture(Tv, m, n)
        # the fixture is rank deficient, which UMFPACK reports and still factorizes
        F = lu(A; check=false)
        @test_throws DimensionMismatch det(F)
        @test_throws DimensionMismatch F \ ones(m)
        # each factor read by itself has the shape and the values it has in the tuple
        L, U, p, q, Rs = F.:(:)
        @test size(L) == (m, min(m, n)) && size(U) == (min(m, n), n)
        @test mismatch(F.L, Matrix(L)) === nothing
        @test mismatch(F.U, Matrix(U)) === nothing
        @test (F.p, F.q, F.Rs) == (p, q, Rs)
        @test L * U ≈ (Diagonal(Rs) * A)[p, q]
    end

    @testset "ldiv! DimensionMismatch names the sizes and leaves the output unchanged" begin
        F = lu(sparse([4.0 1 0; 1 4 1; 0 1 4]))
        X = fill(7.0, 3)
        @test_throws DimensionMismatch ldiv!(X, F, [1.0, 2])
        @test_throws r"3×3.*2 rows" ldiv!(X, F, [1.0, 2])
        @test_throws r"\(3,\).*\(3, 1\)" ldiv!(X, F, reshape([1.0, 2, 3], 3, 1))
        @test X == fill(7.0, 3)
    end

    @testset "ldiv! with strided and adjoint/transpose right-hand sides, $Tv, $Ti" for
            Tv in ELTYPES, Ti in ITYPES
        A = SparseMatrixCSC{Tv,Ti}(sparse(Tv[4 1 0 0; 1 4 1 0; 0 1 4 1; 0 0 1 4.5]))
        F = lu(A)
        Ad = Matrix(A)
        w = Tv.(collect(1.0:8.0))
        v = view(w, 1:2:8)
        @test ldiv!(F, v) ≈ Ad \ Tv.(1:2:8)
        @test w[2:2:8] == 2:2:8
        @test ldiv!(zeros(Tv, 4), F, view(Tv.(collect(1.0:8.0)), 1:2:8)) ≈ Ad \ Tv.(1:2:8)
        # the ComplexF64 solve! has a strided branch of its own, which a standard run reaches only here
        @static COMPREHENSIVE || @test ldiv!(zeros(ComplexF64, 4), lu(SparseMatrixCSC{ComplexF64,Ti}(A)), view(complex(collect(1.0:8.0)), 1:2:8)) ≈ Ad \ Tv.(1:2:8)
        # a matrix and a right-hand side that differ from their conjugates tell the transpose
        # from the adjoint solve, and check the imaginary part of the determinant
        Ac = SparseMatrixCSC{ComplexF64,Ti}(fixture(ComplexF64, 4, 4) + 5I)
        Fc = lu(Ac)
        bc = complex.(1.0:4.0, 4.0:-1.0:1.0)
        @test transpose(Ac) * ldiv!(zeros(ComplexF64, 4), transpose(Fc), bc) ≈ bc
        @test Ac' * ldiv!(zeros(ComplexF64, 4), Fc', bc) ≈ bc
        @test det(Fc) ≈ det(Matrix(Ac))
        @test all(logabsdet(Fc) .≈ logabsdet(Matrix(Ac)))
        M = Tv.(reshape(1.0:24.0, 8, 3))
        Y = zeros(Tv, 8, 3)
        ldiv!(view(Y, 1:2:8, :), transpose(F), view(M, 2:2:8, :))
        @test Y[1:2:8, :] ≈ transpose(Ad) \ M[2:2:8, :]
        @test iszero(Y[2:2:8, :])
        for (op, G) in ((op, wrap(F)) for (tv, ti, op, wrap) in unique(((Float64, Int, adjoint, identity), (Float64, Int, transpose, adjoint),
                (@static COMPREHENSIVE ? ((Float64, Int, adjoint, transpose), (ComplexF64, Int, transpose, identity),
                    (ComplexF64, Int, adjoint, adjoint), (ComplexF64, Int, transpose, transpose)) : ())...)) if (tv, ti) == (Tv, Ti))
            B = Tv.(reshape(1.0:12.0, 3, 4))
            Bw = op(copy(B))
            @test ldiv!(G, Bw) === Bw
            @test Bw ≈ (G === F ? Ad : G isa TransposeFact ? transpose(Ad) : Ad') \ op(B)
        end
    end
end

@testset "REPL printing of UmfpackLU" begin
    # regular matrix
    A = sparse([1, 2], [1, 2], Float64[1.0, 1.0])
    F = lu(A)
    facstring = sprint((t, s) -> show(t, "text/plain", s), F)
    lstring = sprint((t, s) -> show(t, "text/plain", s), F.L)
    ustring = sprint((t, s) -> show(t, "text/plain", s), F.U)
    @test facstring == "$(summary(F))\nL factor:\n$lstring\nU factor:\n$ustring"

    # singular matrix
    B = sparse(zeros(Float64, 2, 2))
    F = lu(B; check=false)
    umfpack_report(F)
    facstring = sprint((t, s) -> show(t, "text/plain", s), F)
    @test facstring == "Failed factorization of type $(summary(F))"
    umfpack_report(F)

    # factors not computed yet
    F = UMFPACK.UmfpackLU(A)
    facstring = sprint((t, s) -> show(t, "text/plain", s), F)
    @test startswith(facstring, summary(F))
    @test occursin("not computed", facstring)
end


@static if COMPREHENSIVE
@testset "UMFPACK's lu with custom permutation" begin
    A = sparse([1.0 0.0 0.9778920565882165 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 1.0 0.0 0.0 0.0 1.847311282254734 0.0 0.0 0.0 0.0;
    0.0 0.0 1.0 0.0 0.0 0.04863647201402087 0.0 0.0 0.0 -1.1593207405039443;
    0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.5145863988424498;
    0.0421803353935357 0.0 -1.2818900361848549 0.0 1.0 0.0 0.1116124255865398 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.5457237331767308;
    -0.4983003278517826 -0.9974658316950679 1.0734689365455168 -1.0511956770913033 0.0 -0.37409855916460416 1.999357231970987 0.0 0.0 -0.9620788056415616;
    -1.5784683379261246 0.0 0.0 0.0 -0.4147349268116999 0.0 0.8539293641597945 1.0 0.0 0.0;
    0.0 0.0 0.0 0.0 -0.039051958043171624 0.0 0.0 -0.3814599389272203 1.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0])
    q1 = [9, 8, 5, 1, 7, 2, 3, 4, 6, 10]
    q0 = q1 .- 1
    for i in 1:10
        b = Float64.((1:10) .== i)
        x = lu(A) \ b
        x0 = lu(A; q=q0) \ b
        x1 = lu(A; q=q1) \ b
        @test x ≈ x0
        @test x ≈ x1
    end
end
end

@testset "a workspace grows when refinement is turned on" begin
    A = lu(fixture(Float64, 100, 100) + 100I)
    umfpack_report(A)
    b = Float64.(1:100)
    ws = UMFPACK.UmfpackWS(A)
    @test length(ws.Wi) == 100
    @test length(ws.W) == 100
    x = ldiv!(similar(b), A, b; workspace = ws)
    A.control[UMFPACK.JL_UMFPACK_IRSTEP] = 2
    y = ldiv!(similar(b), A, b; workspace = ws)
    @test x ≈ y
    @test length(ws.Wi) == 100
    @test length(ws.W) == 500
    # a smaller factorization does not shrink it
    @test ldiv!(zeros(2), lu(sparse([4.0 1; 1 3])), [1.0, 2.0]; workspace = ws) ≈ [4.0 1; 1 3] \ [1.0, 2.0]
    @test length(ws.Wi) == 100
    umfpack_report(A)
end


@testset "copy is independent of the original" begin
    A = sparse([4.0 1 0; 1 4 1; 0 1 4])
    b = [1.0, 2, 3]
    F = lu(A)
    G = copy(F)
    lu!(F, 2A)
    @test G \ b ≈ Matrix(A) \ b
    # a factorization without factors, or a failed one, copies as one
    @test _isnull_numeric(copy(UMFPACK.UmfpackLU(A)))
    @test !issuccess(copy(lu(sparse([1.0 2; 2 4]); check=false)))
end

@testset "deepcopy does not duplicate the C pointers" begin
    S = sparse([4.0 1 0; 1 3 1; 0 1 2])
    b = [1.0, 2, 3]
    F = lu(S)
    G = deepcopy(F)
    @test G.numeric.p != F.numeric.p
    @test deepcopy(F') \ b ≈ Matrix(S)' \ b
end


@testset "reports at print level 0 do not throw" begin
    A = fixture(Float64, 100, 100) + 100I
    Af = lu(A)
    UMFPACK.umfpack_report_numeric(Af, 0)
    UMFPACK.umfpack_report_symbolic(Af, 0)
end

end # module
