using LinearAlgebra, SparseArrays, Test

# Test the CSR interface with a transposed SparseMatrixCSC
struct TransposeSparseMatrixCSC{Tv, Ti} <: SparseArrays.AbstractSparseMatrixCSR{Tv, Ti}
    A::Transpose{Tv, SparseMatrixCSC{Tv, Ti}}
end

function TransposeSparseMatrixCSC(A::SparseMatrixCSC)
    return TransposeSparseMatrixCSC(Transpose(A))
end

function TransposeSparseMatrixCSC{Tv,Ti}(m::Integer, n::Integer, rowptr::Vector{Ti}, colval::Vector{Ti}, nzval::Vector{Tv}) where {Ti,Tv}
    A = SparseMatrixCSC(n, m, rowptr, colval, nzval)
    return TransposeSparseMatrixCSC(Transpose(A))
end

# Dispatches
SparseArrays.getrowptr(A::TransposeSparseMatrixCSC) = SparseArrays.getcolptr(A.A.parent)
SparseArrays.colvals(A::TransposeSparseMatrixCSC)   = SparseArrays.rowvals(A.A.parent)
SparseArrays.size(A::TransposeSparseMatrixCSC)      = size(A.A)
SparseArrays.nonzeros(A::TransposeSparseMatrixCSC)  = nonzeros(A.A.parent)

@testset "AbstractSparseMatrixCSC interface" begin
    A  = sprandn(4, 4, 0.5)
    AT = TransposeSparseMatrixCSC(A)
    # index
    @test A[1,1:end] == AT[1:end,1]
    @test A[1,1:end] == AT[1:end,1]
    @test A[1,1:end] == AT[1:end,1]
    @test A[2,2:3]  == AT[2:3,2]
    @test A[3,4]        == AT[4,3]
    @test any(A .== AT[1:end,1:end])
    @test A[1:end,1:end] == transpose(AT[1:end,1:end])

    @test issparse(AT)
    @test nnz(A) == nnz(AT)

    ATATT = AT * AT'
    @test typeof(ATATT) <: SparseArrays.AbstractSparseMatrixCSR
    @test all(ATATT .== A'A)

    ATTAT = AT' * AT
    @test typeof(ATTAT) <: SparseArrays.AbstractSparseMatrixCSR
    @test all(ATTAT .== A*A')

    A  = TransposeSparseMatrixCSC(sprandn(3,2,0.5))
    B1 = TransposeSparseMatrixCSC(sprandn(3,4,0.5))
    B2 = TransposeSparseMatrixCSC(sprandn(3,3,0.5))
    B3 = TransposeSparseMatrixCSC(sprandn(4,3,0.5))
    @test A*B1' == Matrix(A)*Matrix(B1')
    @test A*B2  == Matrix(A)*Matrix(B2)
    @test A*B3  == Matrix(A)*Matrix(B3)
end
