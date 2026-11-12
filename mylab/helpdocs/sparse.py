"""Help text for sparse matrices and the iterative solvers."""

_KRYLOV_ARGS = ("A is a square matrix (sparse or full) or a function handle that returns A*x. tol is the "
                "relative residual to reach (default 1e-6) and maxit the iteration limit. M1 and M2 are "
                "optional preconditioner factors (the method works with inv(M1*M2)*A); either may be a "
                "function handle that returns M\\x. x0 is the starting guess. flag is 0 when the tolerance "
                "was met and 1 when the iterations ran out, relres is norm(b-A*x)/norm(b), iter the iteration "
                "count and resvec the residual history. With a single output a one-line convergence report "
                "is printed.")

HELP = {
    "sparse": (["S = sparse(A)", "S = sparse(m, n)", "S = sparse(i, j, v)", "S = sparse(i, j, v, m, n)"],
               "Creates a sparse matrix, which stores only its nonzero elements. sparse(A) converts a full "
               "matrix, sparse(m, n) is an all-zero m-by-n matrix, and sparse(i, j, v) puts v(k) at row i(k), "
               "column j(k); values that land on the same position are added, which is the usual way to "
               "assemble finite-element matrices. m and n fix the size (otherwise it is max(i) by max(j)). "
               "A sparse matrix has class double (or logical); use issparse to tell it from a full one. "
               "Operators, indexing, \\ and most matrix functions work on it directly; functions without a "
               "sparse version receive the full matrix."),
    "full": ("A = full(S)", "Converts a sparse matrix to an ordinary full matrix. A full input is returned "
             "unchanged."),
    "issparse": ("tf = issparse(A)", "True when A is stored as a sparse matrix."),
    "speye": (["S = speye(n)", "S = speye(m, n)", "S = speye([m n])"],
              "Sparse identity matrix: ones on the main diagonal and nothing stored elsewhere."),
    "spones": ("R = spones(S)", "A sparse matrix with the same nonzero pattern as S and every nonzero replaced "
               "by 1."),
    "spalloc": ("S = spalloc(m, n, nz)", "An all-zero m-by-n sparse matrix. The third input, the expected number "
                "of nonzeros, is accepted and ignored: storage grows as needed."),
    "spfun": ("R = spfun(f, S)", "Applies the function f to the nonzero elements of S only, keeping the zeros. "
              "For example spfun(@exp, S) leaves the zeros alone where exp(S) would turn them into ones."),
    "nzmax": ("n = nzmax(S)", "Amount of storage allocated for nonzeros: nnz(S) (at least 1) for a sparse "
              "matrix and numel(A) for a full one."),
    "nonzeros": ("v = nonzeros(A)", "Column vector of the nonzero elements of A, in column-major order."),
    "spconvert": ("S = spconvert(D)", "Builds a sparse matrix from a table of triplets: each row of D is "
                  "[row column value], or [row column real imag] for complex values."),
    "sprand": (["R = sprand(m, n, density)", "R = sprand(S)"],
               "Random sparse matrix with uniformly distributed nonzeros in (0, 1). About density*m*n elements "
               "are nonzero; sprand(S) keeps the pattern of S."),
    "sprandn": (["R = sprandn(m, n, density)", "R = sprandn(S)"],
                "Random sparse matrix with normally distributed nonzeros. About density*m*n elements are "
                "nonzero; sprandn(S) keeps the pattern of S."),
    "sprandsym": (["R = sprandsym(n, density)", "R = sprandsym(S)"],
                  "Random symmetric sparse matrix with normally distributed nonzeros, either n-by-n with "
                  "roughly the given density or with the pattern of the lower triangle of S."),
    "spdiags": (["A = spdiags(B, d, m, n)", "A = spdiags(B, d, A)", "B = spdiags(A, d)", "[B, d] = spdiags(A)"],
                "Builds a sparse matrix from diagonals or extracts diagonals. spdiags(B, d, m, n) puts the "
                "columns of B on the diagonals numbered d (0 is the main diagonal, positive above, negative "
                "below) of an m-by-n matrix; with m >= n the element in column j of the matrix comes from "
                "row j of B. spdiags(B, d, A) replaces those diagonals of A. spdiags(A, d) returns the "
                "requested diagonals as columns, and spdiags(A) returns all nonzero diagonals and their "
                "numbers."),
    "blkdiag": ("B = blkdiag(A1, A2, ...)", "Block-diagonal matrix with A1, A2, ... along the diagonal and "
                "zeros elsewhere. The result is sparse when any input is sparse."),
    "bandwidth": (["[lower, upper] = bandwidth(A)", "b = bandwidth(A, 'lower')", "b = bandwidth(A, 'upper')"],
                  "Number of nonzero diagonals below (lower) and above (upper) the main diagonal."),
    "isbanded": ("tf = isbanded(A, lower, upper)", "True when A has no nonzeros beyond the given lower and "
                 "upper bandwidths."),
    "normest": ("n = normest(S)", "Estimate of the 2-norm (largest singular value) of a matrix without "
                "converting it to full form."),
    "condest": ("c = condest(A)", "Estimate of the 1-norm condition number of a square matrix, computed from "
                "a sparse LU factorization. A singular matrix gives Inf."),
    "sprank": ("r = sprank(A)", "Structural rank: the largest rank any matrix with the nonzero pattern of A "
               "could have."),
    "symrcm": ("p = symrcm(A)", "Reverse Cuthill-McKee ordering: a permutation p such that A(p, p) has its "
               "nonzeros close to the diagonal."),
    "amd": ("p = amd(A)", "Minimum-degree style ordering of a symmetric pattern: factorizing A(p, p) usually "
            "creates fewer new nonzeros than factorizing A."),
    "symamd": ("p = symamd(A)", "Minimum-degree style ordering for a symmetric matrix (same as amd here)."),
    "colamd": ("p = colamd(A)", "Column ordering that reduces fill-in when A(:, p) is factorized."),
    "ilu": (["[L, U] = ilu(A)", "[L, U] = ilu(A, setup)", "[L, U, P] = ilu(A, setup)"],
            "Incomplete LU factorization of a sparse square matrix, for use as a preconditioner: L*U "
            "approximates A while staying sparse. setup is a struct; setup.type is 'nofill' (default) or "
            "'ilutp', and setup.droptol sets the drop tolerance for 'ilutp'."),
    "ichol": (["L = ichol(A)", "L = ichol(A, opts)"],
              "Incomplete Cholesky factor of a sparse symmetric positive definite matrix: a lower-triangular "
              "L with L*L' close to A, used to precondition pcg as pcg(A, b, tol, maxit, L, L'). By default "
              "L keeps the nonzero pattern of the lower triangle of A. opts.type = 'ict' with opts.droptol "
              "allows fill-in; opts.diagcomp adds a diagonal shift. An error is raised if a pivot is not "
              "positive."),
    "eigs": (["d = eigs(A)", "d = eigs(A, k)", "d = eigs(A, k, sigma)", "[V, D] = eigs(...)",
              "d = eigs(A, B, k, sigma)"],
             "A few eigenvalues of a large (usually sparse) square matrix: the k largest in magnitude by "
             "default (k = 6). sigma chooses which ones: 'largestabs' ('lm'), 'smallestabs' ('sm'), "
             "'largestreal', 'smallestreal', 'bothendsreal', or a number to get the eigenvalues closest to "
             "it. With two outputs the columns of V are eigenvectors and D is a diagonal matrix. A second "
             "matrix B solves the generalized problem A*v = lambda*B*v. Function-handle operators are not "
             "supported."),
    "svds": (["s = svds(A)", "s = svds(A, k)", "s = svds(A, k, 'smallest')", "[U, S, V] = svds(...)"],
             "A few singular values of a large matrix: the k largest (default 6) in decreasing order, or the "
             "smallest ones with 'smallest'. With three outputs A is approximated by U*S*V'."),
    "pcg": (["x = pcg(A, b)", "x = pcg(A, b, tol, maxit)", "x = pcg(A, b, tol, maxit, M1, M2, x0)",
             "[x, flag, relres, iter, resvec] = pcg(...)"],
            "Solves A*x = b for a symmetric positive definite A with the preconditioned conjugate gradient "
            "method. " + _KRYLOV_ARGS + " maxit defaults to min(n, 20)."),
    "gmres": (["x = gmres(A, b)", "x = gmres(A, b, restart, tol, maxit, M1, M2, x0)",
               "[x, flag, relres, iter, resvec] = gmres(...)"],
              "Solves A*x = b for a general square A with the generalized minimum residual method, restarted "
              "every `restart` iterations ([] for no restart). " + _KRYLOV_ARGS + " iter is [outer inner]."),
    "bicgstab": (["x = bicgstab(A, b)", "x = bicgstab(A, b, tol, maxit, M1, M2, x0)",
                  "[x, flag, relres, iter, resvec] = bicgstab(...)"],
                 "Solves A*x = b for a general square A with the stabilized biconjugate gradient method. "
                 + _KRYLOV_ARGS),
    "bicg": (["x = bicg(A, b)", "x = bicg(A, b, tol, maxit)", "[x, flag, relres, iter] = bicg(...)"],
             "Solves A*x = b with the biconjugate gradient method. A must be a matrix (the method also needs "
             "A'), and preconditioners are not used. tol defaults to 1e-6 and maxit to min(n, 20)."),
    "cgs": (["x = cgs(A, b)", "x = cgs(A, b, tol, maxit, M1, M2, x0)", "[x, flag, relres, iter] = cgs(...)"],
            "Solves A*x = b for a general square A with the conjugate gradient squared method. " + _KRYLOV_ARGS),
    "minres": (["x = minres(A, b)", "x = minres(A, b, tol, maxit, M1, M2, x0)",
                "[x, flag, relres, iter] = minres(...)"],
               "Solves A*x = b for a symmetric (possibly indefinite) A with the minimum residual method. "
               + _KRYLOV_ARGS),
    "lsqr": (["x = lsqr(A, b)", "x = lsqr(A, b, tol, maxit)", "[x, flag, relres, iter] = lsqr(...)"],
             "Least-squares solution of A*x = b for a rectangular or square matrix, by an iterative method "
             "that only needs products with A and A'. tol defaults to 1e-6."),
    "spy": (["spy(S)", "spy(S, marker)", "spy(S, marker, markersize)"],
            "Plots the nonzero pattern of a matrix: one marker per nonzero, with row 1 at the top, and the "
            "number of nonzeros under the axes."),
}
