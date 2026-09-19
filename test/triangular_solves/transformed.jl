# This file is a part of Julia. License is MIT: https://julialang.org/license

# PR 28242
@testset "forward and backward solving of transpose/adjoint triangular matrices" begin
    rng = MersenneTwister(20180730)
    n = 10
    A = sprandn(rng, n, n, 0.8); A += Diagonal((1:n) - diag(A))
    B = ones(n, 2)
    for (Ttri, triul ) in ((UpperTriangular, triu), (LowerTriangular, tril))
        for trop in (adjoint, transpose)
            AT = Ttri(A)           # ...Triangular wrapped
            AC = triul(A)          # copied part of A
            ATa = trop(AT)         # wrapped Adjoint
            ACa = sparse(trop(AC)) # copied and adjoint
            @test AT \ B ≈ AC \ B
            @test ATa \ B ≈ ACa \ B
            @test ATa \ sparse(B) ≈ ATa \ B
            @test Matrix(ATa) \ B ≈ ATa \ B
            @test ATa * ( ATa \ B ) ≈ B
        end
    end
end
