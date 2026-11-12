"""Help text for the smaller numeric functions."""

HELP = {
    "realsqrt": ("y = realsqrt(x)",
                 "Square root of each element. Unlike sqrt, it raises an error when an element is "
                 "negative instead of returning a complex number."),
    "pow2": (["y = pow2(x)", "y = pow2(f, e)"],
             "pow2(x) is 2.^x. pow2(f, e) is f .* 2.^e, the inverse of [f, e] = log2(x)."),
    "invhilb": ("A = invhilb(n)",
                "The exact inverse of the n-by-n Hilbert matrix hilb(n). All its entries are integers."),
    "compan": ("A = compan(p)",
               "Companion matrix of the polynomial with coefficients p (highest power first). "
               "Its eigenvalues are the roots of the polynomial."),
    "del2": (["L = del2(U)", "L = del2(U, h)", "L = del2(U, hx, hy)"],
             "Discrete Laplacian. For a vector it is one quarter of the second difference; for a "
             "matrix it is (d2U/dx2 + d2U/dy2)/4. Values on the edges are extrapolated from the "
             "interior. h, hx and hy are scalar spacings between points (default 1)."),
    "rat": (["[n, d] = rat(x)", "[n, d] = rat(x, tol)", "s = rat(x)"],
            "Rational approximation by continued fractions: n./d is close to x, within tol "
            "(default 1e-6*norm(x(:), 1)). With one output it returns the continued fraction as "
            "text, for example rat(pi) is '3 + 1/(7 + 1/16)'."),
    "anynan": ("tf = anynan(A)", "True if at least one element of A is NaN."),
    "allfinite": ("tf = allfinite(A)", "True if no element of A is NaN or Inf."),
    "standardizeMissing": ("B = standardizeMissing(A, indicator)",
                           "Replaces the values listed in indicator with the standard missing value: "
                           "NaN in a numeric array, '' in a cell array of text. For example "
                           "standardizeMissing([1 -99 3], -99) is [1 NaN 3]."),
    "isoutlier": (["tf = isoutlier(A)", "tf = isoutlier(A, method)", "tf = isoutlier(A, method, dim)",
                   "tf = isoutlier(..., 'ThresholdFactor', f)"],
                  "Marks the outliers in A with true. The default method 'median' flags values more "
                  "than 3 scaled median absolute deviations from the median. 'mean' flags values more "
                  "than 3 standard deviations from the mean, and 'quartiles' flags values more than "
                  "1.5 interquartile ranges outside the quartiles. A matrix is checked column by column."),
    "uniquetol": (["C = uniquetol(A)", "C = uniquetol(A, tol)", "[C, ia, ic] = uniquetol(...)"],
                  "Unique values of A, where values closer than tol*max(abs(A(:))) count as equal. "
                  "The default tol is 1e-12 (1e-6 for single). The result is sorted; ia and ic are "
                  "index vectors with C = A(ia) and A approximately equal to C(ic)."),
}
