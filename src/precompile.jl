# This file is a part of Julia. License is MIT: https://julialang.org/license

# Runs the common operations on Float64 arrays while the package image is built, so that their
# native code is cached; `show` and the `\` polyalgorithm otherwise take seconds on first call.
if Base.generating_output()
    let
        A = sparse([1, 2, 3, 1, 3], [1, 2, 3, 3, 1], [4.0, 5.0, 6.0, 1.0, 1.0])
        x = [1.0, 2.0, 3.0]
        v = sparsevec([1, 3], [1.0, 2.0], 3)
        A[1, 2]; A[1:2, 2:3]; A[:, 2]; A[2, 2] = 5.0
        A * x; A * A; A' * x; A * v; A + A; A - A; 2 * A
        A .* 2; A .+ A; A .* A
        copy(A'); sum(A); findnz(A); dropzeros(A); diag(A); kron(A, A)
        hcat(A, A); vcat(A, A); [A A; A A]
        v + v; dot(v, v)
        sprand(3, 3, 0.5)
        sprint(show, MIME"text/plain"(), A); sprint(show, A)
        sprint(show, MIME"text/plain"(), v); sprint(show, v)
        @static if Base.USE_GPL_LIBS
            A \ x; lu(A) \ x; qr(A) \ x
            H = Hermitian(A' * A + 10 * LinearAlgebra.I)
            cholesky(H) \ x; ldlt(H) \ x
        end
    end
end
