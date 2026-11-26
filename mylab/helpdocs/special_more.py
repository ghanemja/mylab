"""Help text for the functions in mylab.builtins.special_more."""

HELP = {
    "erfcx": ("y = erfcx(x)",
              "Scaled complementary error function exp(x.^2) .* erfc(x), computed without overflow or "
              "loss of accuracy for large x."),
    "betaln": ("y = betaln(z, w)", "Logarithm of the beta function, log(beta(z, w)), without overflow."),
    "betaincinv": (["x = betaincinv(y, z, w)", "x = betaincinv(y, z, w, 'upper')"],
                   "Inverse of the incomplete beta function: the x with betainc(x, z, w) = y. 'upper' "
                   "inverts the upper tail."),
    "gammaincinv": (["x = gammaincinv(y, a)", "x = gammaincinv(y, a, 'upper')"],
                    "Inverse of the incomplete gamma function: the x with gammainc(x, a) = y. 'upper' "
                    "inverts the upper tail."),
    "besselh": (["H = besselh(nu, Z)", "H = besselh(nu, K, Z)", "H = besselh(nu, K, Z, scale)"],
                "Hankel function (Bessel function of the third kind) of order nu. K is 1 (default) for "
                "J + i*Y or 2 for J - i*Y. scale = 1 multiplies by exp(-i*Z) for K = 1 and exp(i*Z) for K = 2."),
    "ellipj": (["[sn, cn, dn] = ellipj(u, m)", "[sn, cn, dn] = ellipj(u, m, tol)"],
               "Jacobi elliptic functions with argument u and parameter m (0 <= m <= 1). They satisfy "
               "sn.^2 + cn.^2 = 1 and dn.^2 + m*sn.^2 = 1."),
    "legendre": (["P = legendre(n, X)", "P = legendre(n, X, 'unnorm' | 'sch' | 'norm')"],
                 "Associated Legendre functions of degree n and orders 0 to n, one order per row, at the "
                 "points X in [-1, 1]. 'sch' gives the Schmidt semi-normalized functions and 'norm' the "
                 "fully normalized ones."),
    "secd": ("y = secd(x)", "Secant of an angle in degrees."),
    "cscd": ("y = cscd(x)", "Cosecant of an angle in degrees."),
    "cotd": ("y = cotd(x)", "Cotangent of an angle in degrees."),
    "asecd": ("y = asecd(x)", "Inverse secant, in degrees."),
    "acscd": ("y = acscd(x)", "Inverse cosecant, in degrees."),
    "acotd": ("y = acotd(x)", "Inverse cotangent, in degrees."),
    "sinpi": ("y = sinpi(x)", "sin(pi*x), exact for integers and half-integers (sinpi(1) is 0, not 1.2e-16)."),
    "cospi": ("y = cospi(x)", "cos(pi*x), exact for integers and half-integers (cospi(0.5) is 0)."),
    "reallog": ("y = reallog(x)", "Natural logarithm that raises an error for negative input instead of "
                                  "returning a complex number."),
    "realpow": ("z = realpow(x, y)", "Element-wise power x.^y that raises an error when the result would be complex."),
    "rms": (["y = rms(x)", "y = rms(x, dim)", "y = rms(x, 'all')", "y = rms(x, vecdim)", "y = rms(..., 'omitnan')"],
            "Root mean square: sqrt(mean(abs(x).^2)) along the first non-singleton dimension, a chosen "
            "dimension, several dimensions or all elements."),
    "topkrows": (["B = topkrows(X, k)", "B = topkrows(X, k, col)", "B = topkrows(X, k, col, direction)",
                  "[B, I] = topkrows(...)", "B = topkrows(T, k, vars, direction)"],
                 "The first k rows of X after sorting its rows in descending order (by all columns, or "
                 "by the listed ones; a negative column number sorts that column ascending). direction "
                 "is 'descend', 'ascend' or one per column. I holds the row numbers. Also works on tables."),
    "issortedrows": (["tf = issortedrows(A)", "tf = issortedrows(A, col)", "tf = issortedrows(A, col, direction)"],
                     "True if the rows of A are in sorted order, as sortrows would leave them. direction "
                     "may also be 'monotonic', 'strictascend', 'strictdescend' or 'strictmonotonic'."),
    "histcounts2": (["N = histcounts2(X, Y)", "N = histcounts2(X, Y, nbins)", "N = histcounts2(X, Y, Xedges, Yedges)",
                     "[N, Xedges, Yedges, binX, binY] = histcounts2(...)",
                     "N = histcounts2(..., 'Normalization', type)"],
                    "Two-dimensional histogram counts: N(i, j) is the number of points with X in bin i "
                    "and Y in bin j. Options: XBinEdges, YBinEdges, NumBins, BinWidth, XBinLimits, "
                    "YBinLimits, BinMethod, Normalization."),
    "movprod": (["M = movprod(A, k)", "M = movprod(A, [kb kf])", "M = movprod(..., dim)",
                 "M = movprod(..., 'omitnan')"],
                "Moving product over a sliding window of k elements (or kb before and kf after)."),
    "movmad": (["M = movmad(A, k)", "M = movmad(A, [kb kf])", "M = movmad(..., dim)", "M = movmad(..., 'omitnan')"],
               "Moving median absolute deviation over a sliding window."),
    "movfun": (["y = movfun(fun, A, k)", "y = movfun(fun, A, [kb kf])", "y = movfun(..., 'Endpoints', e, 'dim', d)"],
               "Applies a function to each sliding window of A; fun takes a column vector and returns a "
               "scalar. Endpoints is 'shrink' (default), 'fill', 'periodic', 'same' or a fill value."),
    "rmoutliers": (["B = rmoutliers(A)", "B = rmoutliers(A, method)", "[B, TF, L, U, C] = rmoutliers(...)",
                    "B = rmoutliers(..., 'MinNumOutliers', n)"],
                   "Removes outliers from a vector, or the rows (dim 1) or columns (dim 2) of a matrix "
                   "that contain outliers. The detection options are those of isoutlier."),
    "filloutliers": (["B = filloutliers(A, fillmethod)", "B = filloutliers(A, fillmethod, findmethod)",
                      "[B, TF, L, U, C] = filloutliers(...)"],
                     "Replaces outliers. fillmethod is a number, 'center', 'clip', 'previous', 'next', "
                     "'nearest', 'linear', 'spline', 'pchip' or 'makima'. The detection options are "
                     "those of isoutlier."),
    "ismembertol": (["LIA = ismembertol(A, B)", "LIA = ismembertol(A, B, tol)", "[LIA, LocB] = ismembertol(...)",
                     "... = ismembertol(..., 'ByRows', tf, 'OutputAllIndices', tf, 'DataScale', ds)"],
                    "Set membership within a tolerance: true where an element of A is within "
                    "tol*max(abs([A(:); B(:)])) of an element of B (default tol 1e-12, 1e-6 for single)."),
}
