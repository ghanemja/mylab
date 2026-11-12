"""Help text for geometry, scattered interpolation, N-D transforms and extra statistics."""

_RND = ("Random numbers from the {dist} distribution. Parameters may be scalars or arrays of the result's "
        "size; trailing arguments give the size, e.g. {name}({args}, 3, 2) or {name}({args}, [3 2]). "
        "Invalid parameters give NaN.")

HELP = {
    # -- distances ------------------------------------------------------------------------------------
    "pdist2": (["D = pdist2(X, Y)", "D = pdist2(X, Y, distance)", "[D, I] = pdist2(X, Y, distance, 'Smallest', k)"],
               "Distances between each row of X and each row of Y: D(i, j) is the distance from X(i, :) "
               "to Y(j, :). distance is 'euclidean' (default), 'squaredeuclidean', 'cityblock', "
               "'chebychev', 'minkowski' (with the exponent as next argument), 'cosine', 'correlation', "
               "'hamming', 'jaccard', 'seuclidean', 'spearman' or 'mahalanobis'. With 'Smallest' or "
               "'Largest', k, each column holds the k nearest (or farthest) distances for one row of Y, "
               "and I the row numbers in X."),
    "knnsearch": (["idx = knnsearch(X, Y)", "[idx, D] = knnsearch(X, Y, 'K', k, 'Distance', distance)"],
                  "For each row of Y, finds the nearest rows of X. idx(i, :) are the row numbers in X of "
                  "the k nearest points (default k = 1), closest first, and D the distances. Distance "
                  "names are the same as for pdist2."),
    "dsearchn": (["k = dsearchn(X, Q)", "[k, d] = dsearchn(X, Q)"],
                 "For each row of Q, the row number of the nearest point in X and the distance to it."),

    # -- polygons -------------------------------------------------------------------------------------
    "inpolygon": (["in = inpolygon(xq, yq, xv, yv)", "[in, on] = inpolygon(xq, yq, xv, yv)"],
                  "True for the query points (xq, yq) that lie inside or on the edge of the polygon with "
                  "corners (xv, yv). on is true for points exactly on the edge. Separate several loops "
                  "with NaN; a loop inside another makes a hole."),
    "polyarea": (["A = polyarea(x, y)", "A = polyarea(x, y, dim)"],
                 "Area of the polygon with corners (x, y). For matrices, each column is one polygon."),
    "rectint": ("A = rectint(R1, R2)",
                "Overlap area of rectangles given as rows [x y width height]. A(i, j) is the area shared "
                "by R1(i, :) and R2(j, :)."),
    "convhull": (["k = convhull(x, y)", "k = convhull(P)", "[k, v] = convhull(x, y, z)"],
                 "Convex hull. In 2-D, k lists the corner points counterclockwise, starting from the "
                 "leftmost point and ending where it starts, so plot(x(k), y(k)) draws the hull; v is the "
                 "area. In 3-D, each row of k is a triangle of the hull surface and v the volume. Points "
                 "that are all on one line (or plane) raise an error."),
    "convhulln": ("[K, v] = convhulln(P)",
                  "Convex hull of the points in the rows of P in any dimension: each row of K is a facet, "
                  "v the enclosed volume."),
    "delaunay": (["T = delaunay(x, y)", "T = delaunay(x, y, z)", "T = delaunay(P)"],
                 "Delaunay triangulation. Each row of T holds the point numbers of one triangle "
                 "(counterclockwise), or of one tetrahedron in 3-D. Draw it with triplot(T, x, y) or "
                 "trisurf(T, x, y, z)."),
    "delaunayn": ("T = delaunayn(P)", "Delaunay triangulation of the points in the rows of P, in any dimension."),

    # -- scattered data -------------------------------------------------------------------------------
    "griddata": (["vq = griddata(x, y, v, xq, yq)", "vq = griddata(x, y, v, xq, yq, method)",
                  "vq = griddata(x, y, z, v, xq, yq, zq)"],
                 "Interpolates scattered samples v at (x, y) onto the query points, usually a grid from "
                 "meshgrid. Methods: 'linear' (default), 'nearest', 'natural', 'cubic' and 'v4'. Query "
                 "points outside the convex hull of the samples give NaN (except 'nearest' and 'v4'). "
                 "'natural' uses linear interpolation in mylab. Repeated sample points are averaged, with "
                 "a warning."),
    "scatteredInterpolant": (["F = scatteredInterpolant(x, y, v)", "F = scatteredInterpolant(x, y, z, v)",
                              "F = scatteredInterpolant(P, v, method, extrapolation)", "vq = F(xq, yq)"],
                             "Builds an interpolating function from scattered samples; call it like a "
                             "function: vq = F(xq, yq) or F(Pq). method is 'linear' (default), 'nearest' "
                             "or 'natural' (treated as linear). extrapolation is 'linear' (default), "
                             "'nearest' or 'none' (NaN outside the samples). Linear extrapolation extends "
                             "the plane of the nearest triangle, which approximates MATLAB's result. "
                             "Properties Points, Values, Method and ExtrapolationMethod can be read and "
                             "changed."),
    "TriScatteredInterp": ("F = TriScatteredInterp(x, y, v)",
                           "Older form of scatteredInterpolant. vq = F(xq, yq) interpolates linearly and "
                           "returns NaN outside the convex hull of the samples."),

    # -- shapes ---------------------------------------------------------------------------------------
    "sphere": (["[X, Y, Z] = sphere", "[X, Y, Z] = sphere(n)", "sphere(n)"],
               "Coordinates of a unit sphere as three (n+1)-by-(n+1) matrices (n = 20 by default), for "
               "surf(X, Y, Z). Without outputs it draws the sphere."),
    "cylinder": (["[X, Y, Z] = cylinder", "[X, Y, Z] = cylinder(r)", "[X, Y, Z] = cylinder(r, n)"],
                 "Coordinates of a cylinder of height 1 with n points around (default 20). r is the "
                 "radius, or a vector of radii along the height for a surface of revolution. Without "
                 "outputs it draws the surface."),
    "ellipsoid": (["[X, Y, Z] = ellipsoid(xc, yc, zc, xr, yr, zr)", "[X, Y, Z] = ellipsoid(..., n)"],
                  "Coordinates of an ellipsoid with center (xc, yc, zc) and semi-axes xr, yr, zr, as "
                  "(n+1)-by-(n+1) matrices (n = 20 by default). Without outputs it draws the surface."),

    # -- N-D transforms -------------------------------------------------------------------------------
    "fftn": (["Y = fftn(X)", "Y = fftn(X, siz)"],
             "N-dimensional discrete Fourier transform: the FFT along every dimension of X. siz pads or "
             "cuts X to that size first."),
    "ifftn": (["X = ifftn(Y)", "X = ifftn(Y, siz)", "X = ifftn(..., 'symmetric')"],
              "Inverse N-dimensional discrete Fourier transform. The result is real when its imaginary "
              "part is only round-off; 'symmetric' always returns the real part."),
    "convn": (["C = convn(A, B)", "C = convn(A, B, shape)"],
              "N-dimensional convolution. shape is 'full' (default), 'same' (the central part, the size "
              "of A) or 'valid' (only where B fits completely inside A)."),
    "rcond": ("r = rcond(A)",
              "Estimate of the reciprocal condition number of the square matrix A in the 1-norm. Near 1 "
              "means well conditioned; near 0 (about eps or less) means A is singular to working "
              "precision."),
    "interpn": (["Vq = interpn(X1, X2, ..., V, Xq1, Xq2, ...)", "Vq = interpn(V, Xq1, Xq2, ...)",
                 "Vq = interpn(..., method, extrapval)"],
                "Interpolates values on an N-dimensional grid made with ndgrid. Methods: 'linear' "
                "(default), 'nearest', 'cubic', 'spline'. Points outside the grid give NaN, or extrapval."),
    "interp3": (["Vq = interp3(X, Y, Z, V, Xq, Yq, Zq)", "Vq = interp3(V, Xq, Yq, Zq)",
                 "Vq = interp3(..., method, extrapval)"],
                "Interpolates values on a 3-D grid made with meshgrid (X varies along the columns). "
                "Methods: 'linear' (default), 'nearest', 'cubic', 'spline'. Points outside the grid give "
                "NaN, or extrapval."),

    # -- sampling and distributions -----------------------------------------------------------------------
    "datasample": (["y = datasample(data, k)", "[y, idx] = datasample(data, k, dim)",
                    "y = datasample(..., 'Replace', false)", "y = datasample(..., 'Weights', w)"],
                   "Picks k random elements of a vector, or k random rows of a matrix or table (columns "
                   "with dim = 2). Sampling is with replacement unless 'Replace', false. idx are the "
                   "positions that were picked."),
    "mvnrnd": (["R = mvnrnd(mu, Sigma)", "R = mvnrnd(mu, Sigma, n)"],
               "Random vectors from the multivariate normal distribution with mean mu (a row) and "
               "covariance matrix Sigma; one sample per row of R."),
    "mvnpdf": (["p = mvnpdf(X)", "p = mvnpdf(X, mu)", "p = mvnpdf(X, mu, Sigma)"],
               "Density of the multivariate normal distribution at each row of X. mu defaults to zero and "
               "Sigma to the identity matrix."),
    "gamrnd": ("r = gamrnd(a, b, ...)", _RND.format(dist="gamma (shape a, scale b)", name="gamrnd", args="a, b")),
    "betarnd": ("r = betarnd(a, b, ...)", _RND.format(dist="beta", name="betarnd", args="a, b")),
    "chi2rnd": ("r = chi2rnd(v, ...)", _RND.format(dist="chi-square (v degrees of freedom)", name="chi2rnd",
                                                  args="v")),
    "trnd": ("r = trnd(v, ...)", _RND.format(dist="Student t (v degrees of freedom)", name="trnd", args="v")),
    "frnd": ("r = frnd(v1, v2, ...)", _RND.format(dist="F", name="frnd", args="v1, v2")),
    "lognrnd": ("r = lognrnd(mu, sigma, ...)", _RND.format(dist="lognormal", name="lognrnd", args="mu, sigma")),
    "wblrnd": ("r = wblrnd(a, b, ...)", _RND.format(dist="Weibull (scale a, shape b)", name="wblrnd",
                                                   args="a, b")),
    "raylrnd": ("r = raylrnd(b, ...)", _RND.format(dist="Rayleigh", name="raylrnd", args="b")),
    "geornd": ("r = geornd(p, ...)", _RND.format(dist="geometric (failures before the first success)",
                                                name="geornd", args="p")),
    "nbinrnd": ("r = nbinrnd(r, p, ...)", _RND.format(dist="negative binomial", name="nbinrnd", args="r, p")),
    "unidrnd": ("r = unidrnd(n, ...)", _RND.format(dist="discrete uniform (integers 1 to n)", name="unidrnd",
                                                  args="n")),
    "nanmedian": (["m = nanmedian(x)", "m = nanmedian(x, dim)"], "Median that ignores NaN values."),
    "nanvar": (["v = nanvar(x)", "v = nanvar(x, dim)"], "Variance that ignores NaN values."),
    "geomean": (["m = geomean(x)", "m = geomean(x, dim)"],
                "Geometric mean: the n-th root of the product of n values. Negative values give NaN."),
    "harmmean": (["m = harmmean(x)", "m = harmmean(x, dim)"],
                 "Harmonic mean: n divided by the sum of the reciprocals."),
    "pca": (["coeff = pca(X)", "[coeff, score, latent, tsquared, explained, mu] = pca(X)",
             "coeff = pca(X, 'NumComponents', k)"],
            "Principal component analysis of the data in X (one observation per row, one variable per "
            "column). The columns of coeff are the principal directions, score the data in those "
            "coordinates, latent the variance along each, explained the percentage of the total variance "
            "and mu the column means. Rows with NaN are skipped. The sign of each direction is chosen so "
            "that its largest element is positive. 'Centered', false skips subtracting the means."),
    "grp2idx": ("[idx, names, levels] = grp2idx(g)",
                "Turns a grouping variable into group numbers 1, 2, .... Numbers are grouped in sorted "
                "order, text in order of first appearance and categorical arrays in category order. "
                "names is a cell array of the group names. Missing values get NaN."),
    "ksdensity": (["[f, xi] = ksdensity(x)", "f = ksdensity(x, pts)", "[f, xi, bw] = ksdensity(x, ..., "
                   "'Bandwidth', bw)"],
                  "Kernel density estimate of the sample x: f is the estimated density at the points xi "
                  "(100 points covering the data by default). Options: 'Bandwidth', 'Kernel' ('normal', "
                  "'box', 'triangle', 'epanechnikov'), 'Function' ('pdf', 'cdf', 'survivor'), "
                  "'NumPoints', 'Support' ('positive'), 'Weights'."),

    # -- functions from text -------------------------------------------------------------------------
    "inline": (["f = inline(expr)", "f = inline(expr, arg1, arg2, ...)"],
               "Creates a function from text, e.g. f = inline('x.^2 + 1'); f(3). Without argument names "
               "the variables in the expression are used in alphabetical order. The result is an "
               "anonymous function; new code should write @(x) x.^2 + 1."),
    "builtin": ("[out1, ...] = builtin(name, arg1, ...)",
                "Calls the built-in function with that name, even when a file or class method with the "
                "same name exists."),
    "random": (["r = random(name, a)", "r = random(name, a, b, m, n)"],
               "Random numbers from the distribution with that name, e.g. random('Normal', 0, 1, 3, 2) or "
               "random('Poisson', 4). Names: Normal, Uniform, Exponential, Poisson, Binomial, Gamma, Beta, "
               "Chisquare, T, F, Lognormal, Weibull, Rayleigh, Geometric, Negative Binomial, Discrete "
               "Uniform. The parameters and size arguments are those of the matching function (normrnd, "
               "unifrnd, ...)."),
    "spectrogram": (["spectrogram(x)", "spectrogram(x, window, noverlap, nfft, fs)",
                     "[S, F, T, P] = spectrogram(x, window, noverlap, nfft, fs)", "spectrogram(..., 'yaxis')"],
                    "Short-time Fourier transform of the signal x. window is a window vector or a length "
                    "(a Hamming window; default: 8 segments), noverlap the number of shared samples "
                    "(default half the window), nfft the FFT length and fs the sample rate in Hz. S holds "
                    "one spectrum per column, F the frequencies, T the centre time of each segment and P "
                    "the power spectral density. Without outputs the result is drawn in decibels; 'yaxis' "
                    "puts frequency on the vertical axis."),
    "ldl": (["[L, D] = ldl(A)", "[L, D, P] = ldl(A)"],
            "Block LDL factorization of a symmetric matrix: A = L*D*L' with D block diagonal (1-by-1 and "
            "2-by-2 blocks). With three outputs L is lower triangular and P'*A*P = L*D*L'."),
    "upsample": (["y = upsample(x, n)", "y = upsample(x, n, phase)"],
                 "Inserts n-1 zeros after each sample of x (each row of a matrix)."),
    "downsample": (["y = downsample(x, n)", "y = downsample(x, n, phase)"],
                   "Keeps every n-th sample of x, starting with the first (or after skipping phase samples)."),
    "native2unicode": (["s = native2unicode(bytes)", "s = native2unicode(bytes, encoding)"],
                       "Converts bytes in the given character encoding (default UTF-8) to text."),
    "unicode2native": (["bytes = unicode2native(s)", "bytes = unicode2native(s, encoding)"],
                       "Converts text to bytes (uint8) in the given character encoding (default UTF-8)."),
    "findprop": ("p = findprop(obj, name)",
                 "Describes a property of an object of a classdef class: a struct with Name, GetAccess, "
                 "SetAccess, Dependent, Constant and more. Returns [] when there is no such property."),
}
