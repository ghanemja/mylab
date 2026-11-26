"""Help text for the functions in mylab.builtins.linalg_more."""

HELP = {
    "schur": (["T = schur(A)", "[U, T] = schur(A)", "[U, T] = schur(A, 'real')", "[U, T] = schur(A, 'complex')"],
              "Schur form of a square matrix: A = U*T*U' with U unitary. For a real A the default is the "
              "real form, where T is quasi-triangular (2-by-2 blocks for complex eigenvalue pairs); "
              "'complex' gives a triangular T with the eigenvalues on its diagonal."),
    "ordeig": (["e = ordeig(T)", "e = ordeig(AA, BB)"],
               "Eigenvalues of a Schur form T, or of a QZ pair (AA, BB), in the order in which they "
               "appear on the diagonal. Use it to build a selection for ordschur and ordqz."),
    "ordschur": (["[US, TS] = ordschur(U, T, select)", "[US, TS] = ordschur(U, T, keyword)",
                  "[US, TS] = ordschur(U, T, clusters)"],
                 "Reorders a Schur form so that the chosen eigenvalues come first, with US*TS*US' still "
                 "equal to U*T*U'. select is a logical vector with one entry per eigenvalue; keyword is "
                 "'lhp', 'rhp', 'udi' or 'udo' (left/right half plane, inside/outside the unit disk); "
                 "clusters is a vector of cluster numbers, and the largest number ends up first."),
    "qz": (["[AA, BB, Q, Z] = qz(A, B)", "[AA, BB, Q, Z] = qz(A, B, 'real')",
            "[AA, BB, Q, Z, V, W] = qz(A, B)"],
           "Generalized Schur (QZ) factorization of two square matrices: Q*A*Z = AA and Q*B*Z = BB, "
           "with Q and Z unitary. By default the result is complex with triangular AA and BB; 'real' "
           "keeps real matrices real, with AA quasi-triangular. V and W hold the right and left "
           "generalized eigenvectors."),
    "ordqz": (["[AAS, BBS, QS, ZS] = ordqz(AA, BB, Q, Z, select)", "... = ordqz(AA, BB, Q, Z, keyword)",
               "... = ordqz(AA, BB, Q, Z, clusters)"],
              "Reorders a QZ factorization so that the chosen generalized eigenvalues come first. The "
              "selection works as in ordschur: a logical vector, one of 'lhp', 'rhp', 'udi', 'udo', "
              "or a vector of cluster numbers."),
    "hess": (["H = hess(A)", "[P, H] = hess(A)", "[AA, BB, Q, Z] = hess(A, B)"],
             "Hessenberg form: A = P*H*P' with P unitary and H zero below the first subdiagonal. With "
             "two matrices it returns Q*A*Z = AA (upper Hessenberg) and Q*B*Z = BB (upper triangular)."),
    "balance": (["B = balance(A)", "[T, B] = balance(A)", "[S, P, B] = balance(A)", "... = balance(A, 'noperm')"],
                "Scales (and permutes) a square matrix so that its rows and columns have similar norms, "
                "which makes eigenvalue computations more accurate: B = T\\A*T. With three outputs the "
                "scaling vector S and the permutation P are returned separately (T(:,P) = diag(S)). "
                "'noperm' scales without permuting."),
    "cholupdate": (["R1 = cholupdate(R, x)", "R1 = cholupdate(R, x, '+')", "R1 = cholupdate(R, x, '-')",
                    "[R1, p] = cholupdate(R, x, '-')"],
                   "Rank-one update of a Cholesky factor. If R = chol(A), R1 is the upper triangular "
                   "factor of A + x*x' ('+', the default) or of A - x*x' ('-'). With two outputs a "
                   "failed downdate returns p = 1 instead of an error (p = 2: R is not a valid factor)."),
    "qrupdate": ("[Q1, R1] = qrupdate(Q, R, u, v)",
                 "QR factorization of A + u*v', computed from the factorization A = Q*R without starting "
                 "again. u and v are column vectors."),
    "qrinsert": (["[Q1, R1] = qrinsert(Q, R, j, x)", "[Q1, R1] = qrinsert(Q, R, j, x, 'row')"],
                 "QR factorization of the matrix obtained from A = Q*R by inserting the column x before "
                 "column j, or ('row') the row x before row j."),
    "qrdelete": (["[Q1, R1] = qrdelete(Q, R, j)", "[Q1, R1] = qrdelete(Q, R, j, 'row')"],
                 "QR factorization of the matrix obtained from A = Q*R by removing column j, or ('row') "
                 "row j."),
    "planerot": ("[G, y] = planerot(x)",
                 "Plane (Givens) rotation for a two-element column vector x: G is orthogonal, y = G*x and "
                 "y(2) is zero."),
    "givens": (["G = givens(x, y)", "[c, s] = givens(x, y)"],
               "2-by-2 rotation G = [c s; -s' c] that maps [x; y] to [r; 0]. With two outputs it returns "
               "the cosine and sine instead of the matrix."),
    "subspace": ("theta = subspace(A, B)",
                 "Angle in radians between the column spaces of A and B: the largest principal angle. "
                 "It is 0 when one space contains the other and pi/2 when they are orthogonal."),
    "lscov": (["x = lscov(A, b)", "x = lscov(A, b, w)", "x = lscov(A, b, V)", "x = lscov(A, b, V, alg)",
               "[x, stdx, mse, S] = lscov(...)"],
              "Least squares with known weights or covariance. w is a vector of weights, one per row; "
              "V is the covariance matrix of the errors in b. Also returns the standard errors of x, "
              "the mean squared error and the covariance matrix of x. alg is 'chol' (default) or "
              "'orth' (for a semidefinite V)."),
    "lsqminnorm": (["x = lsqminnorm(A, b)", "x = lsqminnorm(A, b, tol)", "x = lsqminnorm(..., 'warn')"],
                   "Least-squares solution of A*x = b that has the smallest norm when the solution is not "
                   "unique (rank-deficient or underdetermined A). tol is the rank tolerance; 'warn' "
                   "prints a warning when A is rank deficient."),
    "decomposition": (["dA = decomposition(A)", "dA = decomposition(A, type)",
                       "dA = decomposition(A, type, 'upper' | 'lower')",
                       "dA = decomposition(..., 'CheckCondition', tf, 'RankTolerance', tol)"],
                      "Factorizes A once and stores the result, so that many systems can be solved "
                      "quickly with dA\\b and b/dA (also dA'\\b). type is 'auto', 'qr', 'cod', 'lu', "
                      "'ldl', 'chol', 'triangular', 'permutedTriangular', 'banded', 'hessenberg' or "
                      "'diagonal'. Properties: Type, MatrixSize, IsReal, IsSparse, "
                      "IsConjugateTransposed, ScaleFactor, CheckCondition, Datatype. Functions: "
                      "isIllConditioned(dA), rcond(dA), rank(dA) for 'qr' and 'cod'."),
    "funm": (["F = funm(A, fun)", "F = funm(A, fun, options)", "[F, exitflag, output] = funm(...)"],
             "Evaluates a function of a square matrix by the Schur-Parlett method. fun is @exp, @log, "
             "@sin, @cos, @sinh, @cosh, or a handle fun(x, k) that returns the k-th derivative of the "
             "function at x. options is a struct with TolBlk, TolTay, MaxTerms and Display. exitflag "
             "is 0 on success and 1 when a Taylor series did not converge."),
    "tensorprod": (["C = tensorprod(A, B)", "C = tensorprod(A, B, dim)", "C = tensorprod(A, B, dimA, dimB)",
                    "C = tensorprod(A, B, 'all')", "C = tensorprod(..., 'NumDimensionsA', n)"],
                   "Tensor product of two arrays. With no dimensions it is the outer product, of size "
                   "[size(A) size(B)]. Listed dimensions of A and B are multiplied and summed over "
                   "(contracted); 'all' contracts every dimension and returns a scalar."),
    "pagetranspose": ("Y = pagetranspose(X)",
                      "Transposes every page (2-D slice along the first two dimensions) of an N-D array."),
    "pagectranspose": ("Y = pagectranspose(X)",
                       "Complex conjugate transpose of every page of an N-D array."),
    "pageinv": ("Y = pageinv(X)", "Inverse of every page of an N-D array; each page must be square."),
    "pagemldivide": ("X = pagemldivide(A, B)",
                     "Solves A(:,:,k) \\ B(:,:,k) for every page. Page dimensions of size 1 expand."),
    "pagemrdivide": ("X = pagemrdivide(B, A)",
                     "Computes B(:,:,k) / A(:,:,k) for every page. Page dimensions of size 1 expand."),
    "pagenorm": (["n = pagenorm(X)", "n = pagenorm(X, p)"],
                 "Matrix norm of every page, as a 1-by-1-by-pages array. p is 1, 2 (default), Inf or 'fro'."),
    "pagesvd": (["S = pagesvd(X)", "[U, S, V] = pagesvd(X)", "... = pagesvd(X, 'econ')",
                 "... = pagesvd(..., 'vector' | 'matrix')"],
                "Singular value decomposition of every page of an N-D array."),
    "pageeig": (["e = pageeig(X)", "[V, D] = pageeig(X)", "[V, D, W] = pageeig(X)"],
                "Eigenvalues and eigenvectors of every page of an N-D array; each page must be square. "
                "The options of eig ('vector', 'matrix', 'balance', 'nobalance') are accepted."),
    "pagepinv": (["Y = pagepinv(X)", "Y = pagepinv(X, tol)"],
                 "Moore-Penrose pseudoinverse of every page of an N-D array."),
    "pagelsqminnorm": (["X = pagelsqminnorm(A, B)", "X = pagelsqminnorm(A, B, tol)"],
                       "Minimum-norm least-squares solution of A(:,:,k)*X = B(:,:,k) for every page."),
    "gallery": (["A = gallery(name, n, ...)", "A = gallery(3)", "A = gallery(5)"],
                "Test matrices. Families: binomial, cauchy, chebspec, chebvand, chow, circul, clement, "
                "condex, cycol, dorr, dramadah, fiedler, forsythe, frank, gcdmat, gearmat, grcar, hanowa, "
                "house, integerdata, invhess, invol, ipjfact, jordbloc, kahan, kms, krylov, lauchli, "
                "lehmer, lesp, lotkin, minij, moler, neumann, normaldata, orthog, parter, pei, poisson, "
                "prolate, qmult, randcolu, randcorr, randhess, randjorth, rando, randsvd, redheff, "
                "riemann, ris, smoke, toeppd, toeppen, tridiag, triw, uniformdata, wathen, wilk. "
                "'poisson', 'tridiag', 'toeppen', 'neumann', 'dorr' and 'wathen' return sparse "
                "matrices. The random families use the current random stream; integerdata, normaldata "
                "and uniformdata are repeatable for a given seed, but their values differ from other "
                "systems."),
    "hankel": (["H = hankel(c)", "H = hankel(c, r)"],
               "Hankel matrix: constant along the anti-diagonals. hankel(c) has first column c and zeros "
               "below the main anti-diagonal; hankel(c, r) has first column c and last row r."),
    "rosser": ("A = rosser",
               "The 8-by-8 Rosser matrix, a classic symmetric test matrix for eigenvalue algorithms "
               "(a double eigenvalue, three nearly equal ones and a zero eigenvalue)."),
    "wilkinson": ("W = wilkinson(n)",
                  "Wilkinson's eigenvalue test matrix: symmetric and tridiagonal, with ones next to the "
                  "diagonal and abs(-(n-1)/2 : (n-1)/2) on it. Its largest eigenvalues come in close pairs."),
    "hadamard": ("H = hadamard(n)",
                 "Hadamard matrix of order n: entries +1 and -1 with H'*H = n*eye(n). n must be a power "
                 "of 2, or 12 or 20 times a power of 2."),
    "condeig": (["c = condeig(A)", "[V, D, s] = condeig(A)"],
                "Condition numbers of the eigenvalues of A: the reciprocals of the cosines of the angles "
                "between left and right eigenvectors. Large values mean sensitive eigenvalues."),
    "normest1": (["c = normest1(A)", "c = normest1(A, t)", "[c, v, w, it] = normest1(A, t, X0)",
                  "c = normest1(afun, t, X0, p1, ...)"],
                 "Estimates the 1-norm of a square matrix from a few products with A and A'. t is the "
                 "number of columns used per step (default 2). v and w satisfy w = A*v and "
                 "norm(w,1) = c*norm(v,1); it = [iterations, products]. A function handle is called as "
                 "afun('dim'), afun('real'), afun('notransp', X) and afun('transp', X)."),
    "residue": (["[r, p, k] = residue(b, a)", "[b, a] = residue(r, p, k)"],
                "Partial fraction expansion of b(s)/a(s): residues r, poles p and direct (polynomial) "
                "term k. For a pole repeated m times the residues are those of 1/(s-p), 1/(s-p)^2, ... "
                "in that order. With three inputs it converts back to the two polynomials."),
    "residuez": (["[r, p, k] = residuez(b, a)", "[b, a] = residuez(r, p, k)"],
                 "Partial fraction expansion of a ratio of polynomials in z^-1 (a discrete-time transfer "
                 "function): b(z)/a(z) = sum of r(i)/(1 - p(i)*z^-1) plus k(1) + k(2)*z^-1 + ... "
                 "With three inputs it converts back."),
    "polyvalm": ("Y = polyvalm(p, X)",
                 "Evaluates a polynomial at a square matrix, with matrix powers: "
                 "p(1)*X^n + ... + p(n)*X + p(n+1)*eye(size(X))."),
    "polyeig": (["e = polyeig(A0, A1, ..., Ap)", "[X, e] = polyeig(...)", "[X, e, s] = polyeig(...)"],
                "Polynomial eigenvalue problem: finds lambda and x with "
                "(A0 + lambda*A1 + ... + lambda^p*Ap)*x = 0. X has one eigenvector per column and s "
                "holds condition numbers of the eigenvalues."),
    "mpoles": (["[mults, idx] = mpoles(p)", "[mults, idx] = mpoles(p, tol)", "... = mpoles(p, tol, reorder)"],
               "Finds repeated poles. p(idx) lists the poles with nearly equal ones (within tol, default "
               "0.001) next to each other, largest magnitude first; mults counts 1, 2, ... within each "
               "group. reorder = 0 keeps the original order of the groups."),
    "filtic": (["z = filtic(b, a, y)", "z = filtic(b, a, y, x)"],
               "Initial conditions for filter(b, a, ..., z) that continue a signal whose past outputs "
               "were y = [y(-1), y(-2), ...] and past inputs x = [x(-1), x(-2), ...]."),
    "bicgstabl": (["x = bicgstabl(A, b)", "x = bicgstabl(A, b, tol, maxit, M1, M2, x0)",
                   "[x, flag, relres, iter, resvec] = bicgstabl(...)"],
                  "Solves A*x = b with the stabilized biconjugate gradient method BiCGstab(2). A can be "
                  "a matrix or a function handle that returns A*x. flag is 0 when the relative residual "
                  "tol (default 1e-6) was reached within maxit iterations."),
    "qmr": (["x = qmr(A, b)", "x = qmr(A, b, tol, maxit, M1, M2, x0)", "[x, flag, relres, iter, resvec] = qmr(...)"],
            "Solves A*x = b with the quasi-minimal residual method. A function handle is called as "
            "afun(x, 'notransp') for A*x and afun(x, 'transp') for A'*x."),
    "tfqmr": (["x = tfqmr(A, b)", "x = tfqmr(A, b, tol, maxit, M1, M2, x0)",
               "[x, flag, relres, iter, resvec] = tfqmr(...)"],
              "Solves A*x = b with the transpose-free quasi-minimal residual method."),
    "symmlq": (["x = symmlq(A, b)", "x = symmlq(A, b, tol, maxit, M1, M2, x0)",
                "[x, flag, relres, iter, resvec] = symmlq(...)"],
               "Solves A*x = b for a symmetric (possibly indefinite) A with the symmetric LQ method. "
               "A preconditioner must be symmetric positive definite."),
}
