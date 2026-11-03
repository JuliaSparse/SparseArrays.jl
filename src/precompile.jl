# This file is a part of Julia. License is MIT: https://julialang.org/license

# Runs the common operations on Float64 arrays while the package image is built, so that their
# native code is cached; `show` and the `\` polyalgorithm otherwise take seconds on first call.
if Base.generating_output()
    let
        A = sparse([1, 2, 3, 1, 3], [1, 2, 3, 3, 1], [4.0, 5.0, 6.0, 1.0, 1.0])
        x = [1.0, 2.0, 3.0]
        y = similar(x)
        X = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        v = sparsevec([1, 3], [1.0, 2.0], 3)
        spzeros(3, 3); sparse(1.0I, 3, 3); sparse(X); sparse(x)
        spdiagm(-1 => [1.0, 1.0], 0 => x, 1 => [1.0, 1.0])
        A[1, 2]; A[1:2, 2:3]; A[:, 2]; A[2, :]; A[[3, 1], [1, 3]]; A[A .> 2.5]; A[2, 2] = 5.0
        view(A, :, 2:3) * [1.0, 1.0]
        A * x; A * X; A * A; A' * x; transpose(A) * x; A * v; A + A; A - A; 2 * A
        mul!(y, A, x); mul!(y, A, x, 2.0, 1.0)
        A .* 2; A .+ A; A .* A; A .+ x; abs.(A)
        copy(A'); sparse(A'); sum(A); sum(A; dims = 1); sum(A; dims = 2); maximum(A)
        norm(A); opnorm(A, 1); tr(A); findnz(A); findall(!iszero, A); findmax(A)
        dropzeros(A); droptol!(copy(A), 1.5); diag(A); triu(A); tril(A); issymmetric(A)
        kron(A, A); blockdiag(A, A); permute(A, [3, 2, 1], [1, 2, 3])
        hcat(A, A); vcat(A, A); [A A; A A]; hcat(A, X)
        Symmetric(A) * x; Diagonal(x) * A; A * Diagonal(x)
        LowerTriangular(A) \ x; UpperTriangular(A) \ x
        v + v; v .* v; dot(v, v); norm(v); v[2:3]; Vector(v)
        sprand(3, 3, 0.5)
        sprint(show, MIME"text/plain"(), A); sprint(show, A)
        sprint(show, MIME"text/plain"(), v); sprint(show, v)
        @static if Base.USE_GPL_LIBS
            A \ x; qr(A) \ x
            F = lu(A); F \ x; lu!(F, A); det(F)
            H = Hermitian(A' * A + 10 * LinearAlgebra.I)
            C = cholesky(H); C \ x; logdet(C); ldlt(H) \ x
        end
    end
end
