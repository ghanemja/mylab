"""Help text for the curve fitting functions (fit, fittype, fitoptions, cfit, sfit) and griddedInterpolant."""

_F = "f is a cfit or sfit object made by fit. Also callable as f.{name}(...)."
_T = "Works on fittype, cfit and sfit objects."

HELP = {
    "fit": (["fitobject = fit(x, y, fitType)", "fitobject = fit([x, y], z, fitType)",
             "fitobject = fit(x, y, fitType, fitOptions)", "fitobject = fit(x, y, fitType, Name, Value)",
             "[fitobject, gof, output] = fit(...)"],
            "Fit a curve (x a column vector) or a surface (x a two-column matrix) to the column vector y. fitType is "
            "a library model name ('poly1'-'poly9', 'exp1', 'exp2', 'power1', 'power2', 'gauss1'-'gauss8', "
            "'sin1'-'sin8', 'fourier1'-'fourier8', 'rat01'-'rat55', 'weibull', 'linearinterp', 'nearestinterp', "
            "'cubicinterp', 'pchipinterp', 'smoothingspline'; for surfaces 'poly11'-'poly55', 'lowess', "
            "'loess', 'linearinterp', 'nearestinterp', 'cubicinterp', 'biharmonicinterp' and "
            "'thinplateinterp'), an equation such as "
            "'a*exp(-b*x)+c', a cell array of linear terms or a fittype. Options (as a fitoptions object or "
            "Name,Value pairs): Normalize, Exclude, Weights, StartPoint, Lower, Upper, Robust ('off', 'LAR', "
            "'Bisquare'), Algorithm ('Trust-Region', 'Levenberg-Marquardt'), MaxIter, MaxFunEvals, TolFun, TolX, "
            "Display, SmoothingParam, Span (lowess and loess: the fraction of the points in each local fit, "
            "default 0.25) and 'problem' for the values of problem parameters. lowess and loess fit, around "
            "each point, a weighted linear or quadratic surface to the nearest Span share of the data with "
            "tricube distance weights (Robust 'Bisquare' or 'LAR' refits with weights that play down outliers). "
            "Surface interpolants pass through the data: linear, nearest and cubic over a Delaunay triangulation "
            "(NaN outside the data's convex hull), thin-plate and biharmonic splines everywhere; repeated (x, y) "
            "points are averaged. Linear models are solved "
            "by least squares; nonlinear ones start from data-driven starting points (a random one, with a "
            "warning, for custom equations without StartPoint). gof has sse, rsquare, dfe, adjrsquare and rmse; "
            "output has numobs, numparam, residuals, Jacobian, exitflag, algorithm and, for nonlinear fits, "
            "iterations, funcCount and firstorderopt. Data with NaN or Inf is an error: remove it with "
            "prepareCurveData or exclude it."),
    "fittype": (["ft = fittype(libraryModelName)", "ft = fittype(expression)", "ft = fittype(expression, Name, Value)",
                 "ft = fittype(anonymousFunction)", "ft = fittype(linearModelTerms)"],
                "Model for fit. A string that is not a library model is a custom equation; its coefficients are "
                "the variables other than the independent variable (alphabetically), and the operators are made "
                "element-wise. Options: 'independent' (default 'x'; {'x','y'} for surfaces), 'dependent', "
                "'coefficients', 'problem' (fixed parameters given to fit with 'problem') and 'options'. "
                "For an anonymous function the last input is the independent variable. A cell array such as "
                "{'x','sin(x)','1'} is a linear model with coefficients a, b, c. ft(a, b, x) and feval(ft, a, b, x) "
                "evaluate it; formula, coeffnames, argnames, numcoeffs, type, category and islinear describe it."),
    "fitoptions": (["opts = fitoptions(libraryModelName)", "opts = fitoptions(fitType)",
                    "opts = fitoptions('Method', method, Name, Value)", "opts = fitoptions(opts, Name, Value)"],
                   "Fit options object for fit. Methods: 'NonlinearLeastSquares' (StartPoint, Lower, Upper, "
                   "Algorithm, DiffMinChange, DiffMaxChange, Display, MaxFunEvals, MaxIter, TolFun, TolX, Robust), "
                   "'LinearLeastSquares' (Robust, Lower, Upper), 'SmoothingSpline' (SmoothingParam), the "
                   "interpolant methods and 'None'; all have Normalize, Exclude, Weights and Method. Options of a "
                   "library model include its default bounds (for example c >= 0 for Gaussian widths). Properties "
                   "can also be set with opts.Name = value."),
    "cfit": (["f = cfit(fittype, coef1, coef2, ...)"],
             "Curve fit object with the given coefficient values (fit returns cfit objects too). f(x) and "
             "feval(f, x) evaluate it. Methods: coeffvalues, coeffnames, confint, predint, differentiate, "
             "integrate, formula, numcoeffs, argnames, dependnames, indepnames, probnames, probvalues, type, "
             "category, islinear and plot. Coefficients are read as f.a, f.p1 and so on."),
    "sfit": (["f = sfit(fittype, coef1, coef2, ...)"],
             "Surface fit object (fit with [x, y] data returns sfit objects). f(x, y) or f([x, y]) evaluates it; "
             "it has the cfit methods except integrate, and differentiate returns [fx, fy, fxx, fxy, fyy]."),
    "coeffvalues": ("values = coeffvalues(f)",
                    "Row vector of the coefficient values of a fit (a cell holding the piecewise-polynomial "
                    "structure for interpolants and smoothing splines). " + _F.format(name="coeffvalues")),
    "coeffnames": ("names = coeffnames(f)", "Column cell array of the coefficient names. " + _T),
    "probnames": ("names = probnames(f)", "Column cell array of the problem parameter names. " + _T),
    "probvalues": ("values = probvalues(f)", "Row vector of the problem parameter values of a fit. "
                   + _F.format(name="probvalues")),
    "indepnames": ("names = indepnames(f)", "Column cell array of the independent variable names. " + _T),
    "dependnames": ("names = dependnames(f)", "Cell array holding the dependent variable name. " + _T),
    "numcoeffs": ("n = numcoeffs(f)", "Number of coefficients of the model. " + _T),
    "numargs": ("n = numargs(f)", "Number of inputs of the model: coefficients, problem parameters and "
                "independent variables. " + _T),
    "category": ("c = category(f)", "Model category: 'library', 'custom', 'interpolant' or 'spline'. " + _T),
    "islinear": ("tf = islinear(f)", "True for models that are linear in their coefficients (polynomials and "
                 "linear-term models). " + _T),
    "setoptions": ("ft = setoptions(ft, opts)", "Attach a fitoptions object to a fittype; fit then starts from "
                   "these options."),
    "confint": (["ci = confint(f)", "ci = confint(f, level)"],
                "Confidence bounds of the coefficients (2-by-p, lower bounds in the first row), from the "
                "t distribution with the fit's degrees of freedom; level defaults to 0.95. "
                + _F.format(name="confint")),
    "predint": (["ci = predint(f, x)", "ci = predint(f, x, level)", "ci = predint(f, x, level, intopt, simopt)",
                 "[ci, y] = predint(...)"],
                "Prediction bounds at x (n-by-2): intopt 'observation' (default, for a new observation) or "
                "'functional' (for the fitted curve); simopt 'off' (default, nonsimultaneous) or 'on' "
                "(simultaneous). y is the fit at x. " + _F.format(name="predint")),
    "differentiate": (["fx = differentiate(f, x)", "[fx, fxx] = differentiate(f, x)",
                       "[fx, fy, fxx, fxy, fyy] = differentiate(sfit, x, y)"],
                      "Derivatives of a fit at the points x (analytic for polynomials and piecewise polynomials, "
                      "numerical otherwise). " + _F.format(name="differentiate")),
    "integrate": ("int = integrate(f, x, x0)",
                  "Integral of a curve fit from x0 to each element of x. " + _F.format(name="integrate")),
    "prepareCurveData": (["[x, y] = prepareCurveData(x, y)", "[x, y, w] = prepareCurveData(x, y, w)"],
                         "Data as column vectors for fit, with the points that have NaN or Inf removed (with a "
                         "warning)."),
    "prepareSurfaceData": (["[x, y, z] = prepareSurfaceData(x, y, z)", "[x, y, z, w] = prepareSurfaceData(x, y, z, w)"],
                           "Surface data as column vectors for fit, without NaN or Inf points. A row x, a column "
                           "y and a matrix z with size [numel(y) numel(x)] are expanded to a grid first."),
    "griddedInterpolant": (["F = griddedInterpolant(v)", "F = griddedInterpolant(x, v)",
                            "F = griddedInterpolant(X1, X2, ..., Xn, V)", "F = griddedInterpolant({x1, ..., xn}, V)",
                            "F = griddedInterpolant(..., method)", "F = griddedInterpolant(..., method, extrapMethod)",
                            "Vq = F(Xq)", "Vq = F(Xq1, ..., Xqn)", "Vq = F({xq1, ..., xqn})"],
                           "Interpolant on a full grid (ndgrid format, strictly increasing grid vectors). Methods: "
                           "'linear' (default), 'nearest', 'next', 'previous', 'pchip', 'cubic' (cubic convolution "
                           "on a uniform grid, otherwise a spline), 'spline' and 'makima', applied dimension by "
                           "dimension. Extrapolation uses the interpolation method unless extrapMethod is given; "
                           "'none' gives NaN outside the grid. F(Xq) takes one query point per row (or any array "
                           "for 1-D data); F({xq1, ...}) evaluates on the grid of query vectors. V can have one "
                           "extra trailing dimension for several values per grid point. The properties "
                           "GridVectors, Values, Method and ExtrapolationMethod can be read and assigned."),
}
