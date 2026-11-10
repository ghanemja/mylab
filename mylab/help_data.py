"""Help entries: HELP[name] = (usage or [usages], description).

Written from scratch for mylab; do not paste text from other products' documentation.
"""

_SIZE_FORMS = ("n gives an n-by-n result; (m, n) or [m n] gives m-by-n; more sizes such as (m, n, p) "
               "give an N-D array.")

HELP: dict = {
    # -- array construction ---------------------------------------------------------------
    "zeros": (["A = zeros(n)", "A = zeros(m, n)", "A = zeros(m, n, p, ...)", "A = zeros([m n ...])",
               "A = zeros(..., classname)",
               "A = zeros(..., 'like', p)"],
              f"Array filled with 0. With no size it returns the scalar 0; {_SIZE_FORMS} "
              "A class name such as 'int32' or 'single' sets the type. "
              "Example: zeros(2, 3) is a 2-by-3 matrix of zeros."),
    "ones": (["A = ones(n)", "A = ones(m, n)", "A = ones(m, n, p, ...)", "A = ones([m n ...])",
              "A = ones(..., classname)",
              "A = ones(..., 'like', p)"],
             f"Array filled with 1. With no size it returns the scalar 1; {_SIZE_FORMS} "
             "A class name sets the type."),
    "NaN": (["x = NaN", "A = NaN(n)", "A = NaN(m, n)", "A = NaN([m n])", "A = NaN(..., classname)"],
            f"Not-a-Number, the result of undefined operations such as 0/0. {_SIZE_FORMS} "
            "NaN is unequal to everything, itself included; test for it with isnan."),
    "Inf": (["x = Inf", "A = Inf(n)", "A = Inf(m, n)", "A = Inf([m n])", "A = Inf(..., classname)"],
            f"Positive infinity; -Inf is negative infinity. {_SIZE_FORMS} "
            "Example: 1/0 gives Inf."),
    "true": (["t = true", "T = true(n)", "T = true(m, n)", "T = true([m n])"],
             f"Logical 1, or a logical array of ones. {_SIZE_FORMS}"),
    "false": (["f = false", "F = false(n)", "F = false(m, n)", "F = false([m n])"],
              f"Logical 0, or a logical array of zeros. {_SIZE_FORMS}"),
    "eye": (["I = eye(n)", "I = eye(m, n)", "I = eye([m n])", "I = eye(..., classname)"],
            "Identity matrix: ones on the main diagonal, zeros elsewhere. A non-square size puts ones "
            "on the diagonal starting at the top-left. N-D sizes are not supported. "
            "Example: eye(2) is [1 0; 0 1]."),
    "rand": (["r = rand", "R = rand(n)", "R = rand(m, n)", "R = rand([m n])"],
             f"Uniform random numbers from 0 up to (not including) 1. {_SIZE_FORMS} "
             "Values come from the session's generator, which rng can seed; the stream differs from "
             "other implementations even with the same seed."),
    "randn": (["r = randn", "R = randn(n)", "R = randn(m, n)", "R = randn([m n])"],
              f"Normally distributed random numbers with mean 0 and standard deviation 1. {_SIZE_FORMS} "
              "Uses the generator controlled by rng."),
    "randi": (["r = randi(imax)", "R = randi(imax, n)", "R = randi(imax, m, n)", "R = randi([imin imax], ...)"],
              "Random integers drawn uniformly from 1..imax, or from imin..imax when a two-element range "
              "is given. The result is of class double. Example: randi(6, 1, 3) simulates three dice."),
    "randperm": (["p = randperm(n)", "p = randperm(n, k)"],
                 "Row vector holding a random ordering of the integers 1..n. With k, only the first k "
                 "entries of such an ordering are returned, so they are distinct."),
    "rng": (["rng(seed)", "rng('default')", "rng('shuffle')"],
            "Resets the random number generator used by rand, randn, randi and randperm. A numeric seed "
            "gives a repeatable stream, 'default' is the same as seed 0, and 'shuffle' reseeds from the "
            "operating system. Returns nothing; saving and restoring a generator state is not supported."),
    "linspace": (["y = linspace(a, b)", "y = linspace(a, b, n)"],
                 "Row vector of n points evenly spaced from a to b, both ends included. n defaults to 100. "
                 "Example: linspace(0, 1, 5) is [0 0.25 0.5 0.75 1]."),
    "logspace": (["y = logspace(a, b)", "y = logspace(a, b, n)"],
                 "Row vector of n points spaced evenly on a log scale from 10^a to 10^b. n defaults to 50. "
                 "Example: logspace(0, 2, 3) is [1 10 100]."),
    "colon": (["v = colon(a, b)", "v = colon(a, step, b)"],
              "Function form of the a:b and a:step:b range operators. Returns a row vector, empty when "
              "the range runs the wrong way or the step is 0."),
    "magic": ("M = magic(n)",
              "n-by-n magic square: every row, column and both main diagonals add up to the same value, "
              "n*(n^2+1)/2. Each integer 1..n^2 appears once. magic(2) is returned as [4 3; 1 2] even "
              "though no true 2-by-2 magic square exists."),
    "repmat": (["B = repmat(A, n)", "B = repmat(A, m, n)", "B = repmat(A, [m n])", "B = repmat(A, [m n p ...])"],
               "Tiles A m times vertically and n times horizontally (n times both ways for a single n); "
               "further counts repeat it along higher dimensions. Works with numeric, logical, char, cell "
               "and struct arrays. "
               "Example: repmat([1 2], 2, 2) is [1 2 1 2; 1 2 1 2]."),
    "reshape": (["B = reshape(A, m, n)", "B = reshape(A, m, n, p, ...)", "B = reshape(A, [m n ...])",
                 "B = reshape(A, ..., [], ...)"],
                "Rearranges the elements of A into an array of the given size, reading and writing in "
                "column-major order. One size may be given as [] and is computed from the element count, "
                "which must not change. Works on cells and structs too. "
                "Example: reshape(1:6, 2, 3) is [1 3 5; 2 4 6]."),
    "cat": ("C = cat(dim, A, B, ...)",
            "Joins arrays along dimension dim: 1 stacks them vertically, 2 side by side, 3 stacks them "
            "as pages of an N-D array, and so on. Sizes must agree in every other dimension."),
    "horzcat": ("C = horzcat(A, B, ...)",
                "Joins arrays side by side, the same as [A, B, ...]. They must have the same number of rows."),
    "vertcat": ("C = vertcat(A, B, ...)",
                "Stacks arrays on top of each other, the same as [A; B; ...]. They must have the same "
                "number of columns."),
    "meshgrid": (["[X, Y] = meshgrid(x)", "[X, Y] = meshgrid(x, y)", "[X, Y, Z] = meshgrid(x, y, z)"],
                 "Builds grid coordinates from vectors: each row of X is a copy of x and each column of Y "
                 "is a copy of y, so X and Y are numel(y)-by-numel(x). Handy for evaluating a function of "
                 "two variables over a grid. With three vectors it builds 3-D grids."),
    "ndgrid": (["[X, Y] = ndgrid(x)", "[X, Y] = ndgrid(x, y)", "[X, Y, Z] = ndgrid(x, y, z)"],
               "Like meshgrid but the first vector runs along dimension 1 (down the rows), the second "
               "along dimension 2 and so on: X and Y are numel(x)-by-numel(y)."),
    "diag": (["D = diag(v)", "D = diag(v, k)", "d = diag(A)", "d = diag(A, k)"],
             "For a vector v, builds a square matrix with v on the k-th diagonal (k = 0 main, k > 0 above, "
             "k < 0 below). For a matrix A, extracts the k-th diagonal as a column vector. "
             "Example: diag([1 2]) is [1 0; 0 2]."),
    # -- constants --------------------------------------------------------------------------
    "pi": ("p = pi", "The ratio of a circle's circumference to its diameter, 3.14159..., in double precision."),
    "e": ("x = e", "Euler's number 2.71828..., the base of the natural logarithm. Same as exp(1)."),
    "i": ("z = i", "The imaginary unit, sqrt(-1). Can be overwritten by a variable named i; writing 1i is safer."),
    "realmax": ("x = realmax", "Largest finite double-precision number, about 1.7977e308."),
    "realmin": ("x = realmin", "Smallest positive normalized double-precision number, about 2.2251e-308."),
    "flintmax": ("x = flintmax",
                 "2^53, the largest value up to which every integer is exactly representable as a double. "
                 "A class argument is not supported."),
    "eps": (["d = eps", "d = eps(x)", "d = eps('double')", "d = eps('single')"],
            "With no input, the gap between 1 and the next larger double, about 2.2204e-16. eps(x) gives "
            "that gap at the magnitude of each element of x. eps('single') gives the single-precision gap."),
    "intmax": (["v = intmax", "v = intmax(classname)"],
               "Largest value an integer class can hold, returned in that class. Defaults to 'int32'. "
               "Example: intmax('uint8') is 255."),
    "intmin": (["v = intmin", "v = intmin(classname)"],
               "Smallest value an integer class can hold, returned in that class. Defaults to 'int32'. "
               "Example: intmin('int8') is -128."),
    # -- size and shape queries -----------------------------------------------------------
    "size": (["sz = size(A)", "n = size(A, dim)", "v = size(A, [d1 d2 ...])", "[r, c] = size(A)",
              "[d1, d2, ..., dn] = size(A)"],
             "Dimensions of A. With one output, a row vector holding the length of every dimension; "
             "size(A, dim) gives one length (1 for a dimension beyond ndims(A)) and a vector of dims gives "
             "several. With several outputs, the last output absorbs all remaining dimensions (their "
             "product), so [r, c] = size(A) puts the product of dimensions 2 and up in c. "
             "Example: size(ones(2, 5, 3)) is [2 5 3]."),
    "numel": ("n = numel(A)", "Total number of elements in A, the product of all its dimensions."),
    "length": ("n = length(A)",
               "Size of the longest dimension of A, or 0 if A is empty. Example: length(ones(3, 7)) is 7."),
    "ndims": ("n = ndims(A)", "Number of dimensions of A, never less than 2. Example: ndims(zeros(2, 3, 4)) is 3."),
    "isempty": ("tf = isempty(A)", "True if A has no elements, i.e. at least one dimension is 0."),
    "isscalar": ("tf = isscalar(A)", "True if A is 1-by-1."),
    "isvector": ("tf = isvector(A)", "True if A is 1-by-n or n-by-1 with at least one element."),
    "isrow": ("tf = isrow(A)", "True if A has exactly one row (a 1-by-0 array counts)."),
    "iscolumn": ("tf = iscolumn(A)", "True if A has exactly one column (a 0-by-1 array counts)."),
    "ismatrix": ("tf = ismatrix(A)", "True if A has exactly two dimensions (any size, including empty)."),
    "issquare": ("tf = issquare(A)", "True if A has the same number of rows and columns."),
    # -- type queries -----------------------------------------------------------------------
    "class": ("c = class(x)",
              "Name of the type of x as a char vector, for example 'double', 'char', 'logical', 'int8', "
              "'cell', 'struct', 'string' or 'function_handle'. Objects report their class name, such as "
              "'table', 'containers.Map' or the name of a classdef class."),
    "isnumeric": ("tf = isnumeric(x)",
                  "True for double, single and integer arrays (real or complex). False for logical, char, "
                  "cell, struct and string values."),
    "isfloat": ("tf = isfloat(x)", "True if x is a double or single array."),
    "isinteger": ("tf = isinteger(x)",
                  "True if x has an integer class such as int8 or uint32. A double holding a whole number "
                  "is not an integer type."),
    "islogical": ("tf = islogical(x)", "True if x is a logical (boolean) array."),
    "ischar": ("tf = ischar(x)", "True if x is a char array, e.g. text in single quotes."),
    "isstring": ("tf = isstring(x)", "True if x is a string scalar, e.g. text in double quotes."),
    "istext": ("tf = istext(x)", "True if x is either a char array or a string."),
    "iscell": ("tf = iscell(x)", "True if x is a cell array."),
    "isstruct": ("tf = isstruct(x)", "True if x is a struct or struct array."),
    "isreal": ("tf = isreal(x)",
               "False only when x is stored as a complex array; a complex array whose imaginary parts are "
               "all zero still counts as not real. True for non-numeric values."),
    "is_function_handle": ("tf = is_function_handle(x)", "True if x is a function handle such as @sin or @(t) t.^2."),
    "isa": ("tf = isa(x, classname)",
            "True if class(x) equals classname. Also accepts the group names 'numeric' (double, single or "
            "integer), 'float' (double or single) and 'integer' (any integer class). For objects of a "
            "classdef class it is also true for any superclass, e.g. isa(dog, 'Animal')."),
    "isequal": ("tf = isequal(A, B, ...)",
                "True if all inputs have the same size and contents. Numeric type is ignored, so "
                "isequal(1, int8(1)) is true. NaN is never equal to NaN; see isequaln."),
    "isequaln": ("tf = isequaln(A, B, ...)", "Like isequal, but NaN values in matching positions count as equal."),
    "squeeze": ("B = squeeze(A)",
                "Removes dimensions of length 1 from A. The result always keeps at least two dimensions, "
                "so 2-D inputs, including row vectors, are returned unchanged. Example: squeeze(ones(1, 1, 3)) "
                "is a 3-by-1 column."),
    "permute": ("B = permute(A, order)",
                "Rearranges the dimensions of A in the given order, a permutation of 1..n. permute(A, [2 1]) "
                "is the same as transposing a matrix without conjugation."),
    "ipermute": ("A = ipermute(B, order)",
                 "Undoes permute: ipermute(permute(A, order), order) gives back A."),
    "pagemtimes": ("Z = pagemtimes(X, Y)",
                   "Matrix multiplication applied page by page: each 2-D slice X(:,:,k) is multiplied by the "
                   "matching slice Y(:,:,k). The inner dimensions of each pair of pages must agree."),
    "issparse": ("tf = issparse(A)", "Always false: mylab stores every matrix densely."),
    # -- conversions ------------------------------------------------------------------------
    "double": ("y = double(x)",
               "Converts to double precision. Char input gives character codes; a string or string array is "
               "parsed as numbers element by element (NaN where the text is not a number or is <missing>). "
               "Cells and structs cannot be converted. Example: double('A') is 65; double(\"2.5\") is 2.5."),
    "single": ("y = single(x)", "Converts to single precision (32-bit floating point)."),
    "logical": ("tf = logical(x)",
                "Converts to logical: nonzero becomes true, zero becomes false. NaN input is an error."),
    "char": (["c = char(codes)", "c = char(str)", "c = char(C)", "c = char(a, b, ...)"],
             "Makes a char array. Numbers are taken as character codes. A cell array, a string array or "
             "several text inputs become rows of a char matrix, padded with spaces to equal length. "
             "Example: char(72, 105) is a two-row char array; char([72 105]) is 'Hi'."),
    "cast": ("y = cast(x, classname)",
             "Converts x to the named class: 'double', 'single', 'logical' or an integer class."),
    "typecast": ("y = typecast(x, classname)",
                 "Reinterprets the bytes of a real numeric scalar or vector as another numeric type ('double', "
                 "'single' or an integer class) without changing the bits, using little-endian byte order. "
                 "The total byte count must divide evenly into the new type. The result is a row, or a "
                 "column for column input. Example: typecast(single(1), 'uint32') is 1065353216."),
    "full": ("A = full(S)", "Returns its input unchanged, since mylab has no separate sparse storage."),
    "sparse": (["S = sparse(A)", "S = sparse(m, n)", "S = sparse(i, j, v)", "S = sparse(i, j, v, m, n)"],
               "Accepted for compatibility but always produces an ordinary dense matrix. sparse(m, n) is an "
               "m-by-n zero matrix; sparse(i, j, v, ...) places v(k) at row i(k), column j(k), adding up "
               "values that land on the same position."),

    # -- elementwise math -------------------------------------------------------------------
    "sqrt": ("y = sqrt(x)",
             "Square root of each element. Negative inputs give complex results, e.g. sqrt(-4) is 0 + 2i. "
             "Integer inputs are converted to double first."),
    "exp": ("y = exp(x)", "e raised to the power of each element. Example: exp(0) is 1."),
    "log": ("y = log(x)",
            "Natural logarithm of each element. log(0) is -Inf and negative inputs give complex results."),
    "log2": ("y = log2(x)", "Base-2 logarithm of each element; negative inputs give complex results. "
             "The two-output form [f, e] = log2(x) is not supported."),
    "log10": ("y = log10(x)", "Base-10 logarithm of each element; negative inputs give complex results."),
    "log1p": ("y = log1p(x)", "log(1 + x) computed accurately when x is tiny. Inputs below -1 give NaN."),
    "expm1": ("y = expm1(x)", "exp(x) - 1 computed accurately when x is tiny."),
    "sin": ("y = sin(x)", "Sine of each element, with x in radians."),
    "cos": ("y = cos(x)", "Cosine of each element, with x in radians."),
    "tan": ("y = tan(x)", "Tangent of each element, with x in radians."),
    "sec": ("y = sec(x)", "Secant, 1./cos(x), with x in radians."),
    "csc": ("y = csc(x)", "Cosecant, 1./sin(x), with x in radians."),
    "cot": ("y = cot(x)", "Cotangent, 1./tan(x), with x in radians."),
    "asin": ("y = asin(x)", "Inverse sine in radians. Inputs outside [-1, 1] give complex results."),
    "acos": ("y = acos(x)", "Inverse cosine in radians. Inputs outside [-1, 1] give complex results."),
    "atan": ("y = atan(x)", "Inverse tangent in radians, between -pi/2 and pi/2. See atan2 for the full circle."),
    "atan2": ("theta = atan2(y, x)",
              "Four-quadrant angle in radians, between -pi and pi, of the point (x, y). Note that y comes "
              "first. Inputs expand against each other. Example: atan2(1, -1) is 3*pi/4."),
    "sinh": ("y = sinh(x)", "Hyperbolic sine of each element."),
    "cosh": ("y = cosh(x)", "Hyperbolic cosine of each element."),
    "tanh": ("y = tanh(x)", "Hyperbolic tangent of each element."),
    "asinh": ("y = asinh(x)", "Inverse hyperbolic sine of each element."),
    "acosh": ("y = acosh(x)", "Inverse hyperbolic cosine. Inputs below 1 give complex results."),
    "atanh": ("y = atanh(x)", "Inverse hyperbolic tangent. Inputs outside (-1, 1) give complex or infinite results."),
    "sind": ("y = sind(x)", "Sine of angles given in degrees. Multiples of 180 give exactly 0."),
    "cosd": ("y = cosd(x)", "Cosine of angles given in degrees. Odd multiples of 90 give exactly 0."),
    "tand": ("y = tand(x)", "Tangent of angles given in degrees; 0 and multiples of 180 give exactly 0."),
    "asind": ("y = asind(x)", "Inverse sine, result in degrees."),
    "acosd": ("y = acosd(x)", "Inverse cosine, result in degrees."),
    "atand": ("y = atand(x)", "Inverse tangent, result in degrees."),
    "atan2d": ("theta = atan2d(y, x)", "Four-quadrant inverse tangent like atan2, but the result is in degrees."),
    "deg2rad": ("r = deg2rad(d)", "Converts angles from degrees to radians (multiplies by pi/180)."),
    "rad2deg": ("d = rad2deg(r)", "Converts angles from radians to degrees (multiplies by 180/pi)."),
    "hypot": ("c = hypot(a, b)", "sqrt(a.^2 + b.^2) element by element, computed without overflow for large values."),
    "abs": ("y = abs(x)", "Absolute value of each element; for complex x, the magnitude. Example: abs(3 - 4i) is 5."),
    "sign": ("s = sign(x)",
             "-1, 0 or 1 according to the sign of each element. For complex x, returns x./abs(x) "
             "(0 where x is 0)."),
    "floor": ("y = floor(x)", "Rounds each element down, toward -Inf. Example: floor(-2.5) is -3."),
    "ceil": ("y = ceil(x)", "Rounds each element up, toward +Inf. Example: ceil(2.1) is 3."),
    "fix": ("y = fix(x)", "Rounds each element toward zero, dropping the fractional part. Example: fix(-2.7) is -2."),
    "round": (["y = round(x)", "y = round(x, n)"],
              "Rounds to the nearest integer, with halves going away from zero (round(2.5) is 3, "
              "round(-2.5) is -3). With n, rounds to n digits after the decimal point; negative n rounds "
              "to tens, hundreds and so on. Integer-class input is returned unchanged."),
    "mod": ("r = mod(a, b)",
            "Remainder after division, defined as a - floor(a./b).*b, so a nonzero result has the sign of "
            "b. mod(a, 0) is a. Example: mod(-1, 3) is 2. See rem for the truncating version."),
    "rem": ("r = rem(a, b)",
            "Remainder after division, defined as a - fix(a./b).*b, so a nonzero result has the sign of a. "
            "rem(a, 0) is NaN. Example: rem(-1, 3) is -1."),
    "nthroot": ("y = nthroot(x, n)",
                "Real n-th root of each element; for negative x with odd n the real negative root is "
                "returned, e.g. nthroot(-8, 3) is -2. Negative x with an even or non-integer n is an error."),
    "real": ("r = real(z)", "Real part of each element."),
    "imag": ("y = imag(z)", "Imaginary part of each element (0 for real input)."),
    "conj": ("w = conj(z)", "Complex conjugate of each element: the sign of the imaginary part is flipped."),
    "angle": ("theta = angle(z)", "Phase angle of each complex element in radians, between -pi and pi."),
    "complex": (["z = complex(a)", "z = complex(a, b)"],
                "Builds a complex array a + b*i. With one input the imaginary part is zero but the result "
                "is still stored as complex, so isreal reports false."),
    "isnan": ("tf = isnan(x)", "Logical array that is true where x is NaN."),
    "isinf": ("tf = isinf(x)", "Logical array that is true where x is Inf or -Inf."),
    "isfinite": ("tf = isfinite(x)", "Logical array that is true where x is neither NaN nor infinite."),
    "gamma": ("y = gamma(x)", "Gamma function of each element; gamma(n) equals factorial(n-1) for positive integers."),
    "gammaln": ("y = gammaln(x)", "Natural log of the gamma function, useful when gamma itself would overflow."),
    "beta": ("y = beta(a, b)", "Beta function, gamma(a).*gamma(b)./gamma(a+b), element by element."),
    "erf": ("y = erf(x)", "Error function of each element."),
    "erfc": ("y = erfc(x)", "Complementary error function, 1 - erf(x), accurate for large x."),
    "erfinv": ("x = erfinv(y)", "Inverse error function; defined for y in [-1, 1]."),
    "erfcinv": ("x = erfcinv(y)", "Inverse complementary error function; defined for y in [0, 2]."),
    # -- operators in function form ---------------------------------------------------------
    "plus": ("C = plus(A, B)", "Same as A + B."),
    "minus": ("C = minus(A, B)", "Same as A - B."),
    "uminus": ("B = uminus(A)", "Same as -A."),
    "uplus": ("B = uplus(A)", "Same as +A; returns A unchanged."),
    "times": ("C = times(A, B)", "Same as A .* B (element-wise product)."),
    "rdivide": ("C = rdivide(A, B)", "Same as A ./ B (element-wise right division)."),
    "ldivide": ("C = ldivide(A, B)", "Same as A .\\ B, i.e. B ./ A."),
    "power": ("C = power(A, B)", "Same as A .^ B (element-wise power)."),
    "mtimes": ("C = mtimes(A, B)", "Same as A * B (matrix product)."),
    "mrdivide": ("X = mrdivide(B, A)", "Same as B / A, which solves X*A = B."),
    "mldivide": ("X = mldivide(A, B)", "Same as A \\ B, which solves A*X = B (least squares if A is not square)."),
    "mpower": ("C = mpower(A, B)", "Same as A ^ B (matrix power)."),
    "eq": ("tf = eq(A, B)", "Same as A == B."),
    "ne": ("tf = ne(A, B)", "Same as A ~= B."),
    "lt": ("tf = lt(A, B)", "Same as A < B."),
    "le": ("tf = le(A, B)", "Same as A <= B."),
    "gt": ("tf = gt(A, B)", "Same as A > B."),
    "ge": ("tf = ge(A, B)", "Same as A >= B."),
    "and": ("tf = and(A, B)", "Same as A & B (element-wise logical AND)."),
    "or": ("tf = or(A, B)", "Same as A | B (element-wise logical OR)."),
    "not": ("tf = not(A)", "Same as ~A (element-wise logical NOT)."),
    "xor": ("tf = xor(A, B)", "Element-wise exclusive OR: true where exactly one of A and B is nonzero."),
    "transpose": ("B = transpose(A)", "Same as A.' : swaps rows and columns without conjugating."),
    "ctranspose": ("B = ctranspose(A)", "Same as A' : swaps rows and columns and conjugates complex values."),
    # -- number theory and integers ---------------------------------------------------------
    "factorial": ("f = factorial(n)", "n! for each element of n, e.g. factorial(5) is 120. Returned as double."),
    "nchoosek": (["c = nchoosek(n, k)", "C = nchoosek(v, k)"],
                 "With scalar n, the number of ways to pick k items from n (the binomial coefficient). "
                 "With a vector v, every k-element combination of its entries, one per row. "
                 "Example: nchoosek(5, 2) is 10."),
    "primes": ("p = primes(n)",
               "Row vector of all prime numbers less than or equal to n. Example: primes(10) is [2 3 5 7]."),
    "isprime": ("tf = isprime(x)", "Logical array that is true where x is a prime number."),
    "factor": ("f = factor(n)", "Row vector of the prime factors of the scalar n in ascending order, with "
               "repeats. Example: factor(12) is [2 2 3]."),
    "gcd": ("g = gcd(a, b)", "Greatest common divisor of corresponding integer elements of a and b."),
    "lcm": ("m = lcm(a, b)", "Least common multiple of corresponding integer elements of a and b."),
    "dec2bin": (["s = dec2bin(d)", "s = dec2bin(d, n)"],
                "Binary representation of the non-negative integer d as a char vector, zero-padded to at "
                "least n digits. Only scalar input is supported. Example: dec2bin(5) is '101'."),
    "bin2dec": ("d = bin2dec(s)", "Value of a binary digit string; spaces are ignored. Example: bin2dec('101') is 5."),
    "dec2hex": (["s = dec2hex(d)", "s = dec2hex(d, n)"],
                "Hexadecimal representation (upper-case digits) of the non-negative integer d, zero-padded "
                "to at least n digits. Only scalar input is supported. Example: dec2hex(255) is 'FF'."),
    "hex2dec": ("d = hex2dec(s)", "Value of a hexadecimal digit string. Example: hex2dec('ff') is 255."),
    "bitand": ("c = bitand(a, b)", "Bitwise AND of non-negative integer values; the result is double."),
    "bitor": ("c = bitor(a, b)", "Bitwise OR of non-negative integer values; the result is double."),
    "bitxor": ("c = bitxor(a, b)", "Bitwise exclusive OR of non-negative integer values; the result is double."),
    "bitshift": ("c = bitshift(a, n)",
                 "Shifts the bits of integer values left by n places (right when n is negative). n must be "
                 "a scalar. Example: bitshift(1, 3) is 8."),
    # -- reductions and statistics ----------------------------------------------------------
    "sum": (["s = sum(x)", "s = sum(x, dim)", "s = sum(x, 'all')", "s = sum(..., 'omitnan')"],
            "Sum of elements. For a vector, the total; for a matrix, a row of column sums. dim = 1 sums "
            "down columns, dim = 2 across rows, and 'all' sums everything. 'omitnan' skips NaN values. "
            "sum([]) is 0. Integer inputs keep their class and saturate. Example: sum([1 2; 3 4]) is [4 6]."),
    "prod": (["p = prod(x)", "p = prod(x, dim)", "p = prod(x, 'all')", "p = prod(..., 'omitnan')"],
             "Product of elements, working along the first dimension whose size is not 1 unless dim or "
             "'all' is given. 'omitnan' skips NaN values. prod([]) is 1."),
    "mean": (["m = mean(x)", "m = mean(x, dim)", "m = mean(x, 'all')", "m = mean(..., 'omitnan')"],
             "Arithmetic average, along the first dimension whose size is not 1 unless dim or 'all' is "
             "given. 'omitnan' ignores NaN values. The result is double, also for integer input; mean([]) "
             "is NaN. Example: mean([1 2; 3 4]) is [2 3]."),
    "median": (["m = median(x)", "m = median(x, dim)", "m = median(x, 'all')", "m = median(..., 'omitnan')"],
               "Middle value after sorting (the average of the two middle values for an even count), along "
               "the first dimension whose size is not 1 unless dim or 'all' is given. Any NaN makes the "
               "result NaN unless 'omitnan' is given."),
    "mode": (["m = mode(x)", "m = mode(x, dim)"],
             "Most frequent value, along the first dimension whose size is not 1 unless dim is given. Ties "
             "go to the smallest value and NaN is ignored. Only the first output is supported."),
    "std": (["s = std(x)", "s = std(x, w)", "s = std(x, w, dim)", "s = std(..., 'omitnan')"],
            "Standard deviation along the first dimension whose size is not 1, or along dim. w = 0 (the "
            "default, also used when w is []) divides by N-1; w = 1 divides by N. A single value gives 0. "
            "'omitnan' ignores NaN. Example: std([2 4 4 4 5 5 7 9], 1) is 2."),
    "var": (["v = var(x)", "v = var(x, w)", "v = var(x, w, dim)", "v = var(..., 'omitnan')"],
            "Variance, the square of std. w = 0 (default) divides by N-1 and w = 1 divides by N. Works "
            "along the first dimension whose size is not 1 unless dim is given."),
    "max": (["m = max(A)", "[m, idx] = max(A)", "m = max(A, [], dim)", "m = max(A, [], 'all')", "C = max(A, B)"],
            "Largest element along the first dimension whose size is not 1 (or along dim, or over 'all'). "
            "idx gives the position of the first maximum. NaN values are skipped unless every value is "
            "NaN; complex values are compared by magnitude. max(A, B) takes the larger of A and B element "
            "by element. Example: [m, i] = max([3 9 2]) gives m = 9, i = 2."),
    "min": (["m = min(A)", "[m, idx] = min(A)", "m = min(A, [], dim)", "m = min(A, [], 'all')", "C = min(A, B)"],
            "Smallest element along the first dimension whose size is not 1 (or along dim, or over 'all'). "
            "idx gives the position of the first minimum. NaN values are skipped unless every value is "
            "NaN. min(A, B) takes the smaller of A and B element by element."),
    "bounds": ("[lo, hi] = bounds(x)",
               "Smallest and largest elements of x in one call, ignoring NaN. mylab always looks at all "
               "elements; a dimension argument is not supported."),
    "range": (["r = range(x)", "r = range(x, dim)"],
              "max minus min along the first dimension whose size is not 1, or along dim."),
    "any": (["tf = any(x)", "tf = any(x, dim)", "tf = any(x, 'all')"],
            "True if at least one element is nonzero, tested down columns for a matrix (a row of results), "
            "along dim, or over 'all' elements. any([]) is false."),
    "all": (["tf = all(x)", "tf = all(x, dim)", "tf = all(x, 'all')"],
            "True if every element is nonzero, tested down columns for a matrix, along dim, or over 'all' "
            "elements. all([]) is true."),
    "cumsum": (["c = cumsum(x)", "c = cumsum(x, dim)", "c = cumsum(..., 'reverse')"],
               "Running total along the first dimension whose size is not 1, or along dim. 'reverse' "
               "accumulates from the end. Integer inputs keep their class and saturate. NaN flags are "
               "accepted but have no effect. Example: cumsum([1 2 3]) is [1 3 6]."),
    "cumprod": (["c = cumprod(x)", "c = cumprod(x, dim)", "c = cumprod(..., 'reverse')"],
                "Running product along the first dimension whose size is not 1, or along dim."),
    "cummax": (["c = cummax(x)", "c = cummax(x, dim)", "c = cummax(..., 'reverse')"],
               "Running maximum along the first dimension whose size is not 1, or along dim. "
               "Example: cummax([1 3 2 5]) is [1 3 3 5]."),
    "cummin": (["c = cummin(x)", "c = cummin(x, dim)", "c = cummin(..., 'reverse')"],
               "Running minimum along the first dimension whose size is not 1, or along dim."),
    "diff": (["d = diff(x)", "d = diff(x, n)", "d = diff(x, n, dim)"],
             "Differences between neighbouring elements, x(2)-x(1), x(3)-x(2), ..., along the first "
             "dimension whose size is not 1 unless dim is given. n repeats the operation n times, so the "
             "result is n elements shorter. Example: diff([1 4 9]) is [3 5]."),
    "trapz": (["q = trapz(y)", "q = trapz(x, y)"],
              "Approximate integral of sampled data by the trapezoid rule, with unit spacing or the points "
              "in x. Works along the first dimension whose size is not 1; a dim argument is not supported. "
              "Example: trapz([1 2 3]) is 4."),
    "cumtrapz": (["q = cumtrapz(y)", "q = cumtrapz(x, y)"],
                 "Running trapezoid-rule integral, same size as y and starting at 0. Uses unit spacing or "
                 "the points in x, along the first dimension whose size is not 1."),
    "dot": ("d = dot(a, b)",
            "Dot product. For two vectors with the same number of elements, sum(conj(a).*b); for matrices, "
            "the dot products of matching columns as a row vector. A dim argument is not supported."),
    "cross": ("c = cross(a, b)",
              "Cross product of two 3-element vectors, shaped like a. Matrix inputs are not supported. "
              "Example: cross([1 0 0], [0 1 0]) is [0 0 1]."),
    "kron": ("K = kron(A, B)", "Kronecker product: every element A(i,j) is replaced by the block A(i,j)*B."),
    # -- find, sort and sets ----------------------------------------------------------------
    "find": (["k = find(x)", "k = find(x, n)", "k = find(x, n, 'last')", "[r, c] = find(x)", "[r, c, v] = find(x)"],
             "Linear indices of the nonzero elements of x, in column-major order. n limits the result to "
             "the first (or 'last') n hits. With two or three outputs, returns row and column subscripts "
             "and the values. The result is a row for row-vector input, otherwise a column. "
             "Example: find([0 5 0 7]) is [2 4]."),
    "nnz": ("n = nnz(x)", "Number of nonzero elements in x."),
    "sort": (["B = sort(A)", "B = sort(A, dim)", "B = sort(..., 'descend')", "[B, idx] = sort(...)"],
             "Sorts in ascending order (or 'descend') along the first dimension whose size is not 1, or "
             "along dim. idx holds the original positions. Equal values keep their order; NaN goes last "
             "when ascending and first when descending; complex values sort by magnitude. Cell arrays of "
             "char vectors and string arrays are sorted alphabetically (by character code). "
             "Example: sort([3 1 2]) is [1 2 3]."),
    "unique": (["u = unique(x)", "u = unique(x, 'stable')", "u = unique(A, 'rows')", "[u, ia, ic] = unique(...)"],
               "The distinct values of x, sorted unless 'stable' is given (first-appearance order). The "
               "result is a row if x is a row, otherwise a column. ia and ic are index columns with "
               "u = x(ia) and x = u(ic); 'last' makes ia point to last occurrences. 'rows' finds distinct "
               "rows. Also works on cell arrays of char vectors and on string arrays. "
               "Example: unique([3 1 3 2]) is [1 2 3]."),
    "union": (["c = union(a, b)", "c = union(..., 'stable')", "[c, ia, ib] = union(...)"],
              "Values found in a or b, without repeats, sorted unless 'stable' is given. The result is a "
              "row unless a is a column. Also accepts cell arrays of text. In mylab ia and ib list the "
              "position in a (and in b) of each result value found there. The 'rows' option is ignored."),
    "intersect": (["c = intersect(a, b)", "c = intersect(..., 'stable')", "[c, ia, ib] = intersect(...)"],
                  "Values found in both a and b, without repeats, sorted unless 'stable' is given; "
                  "c = a(ia) = b(ib). The result is a row unless a is a column. Accepts cell arrays of "
                  "text. The 'rows' option is ignored. Example: intersect([3 1 2], [2 3 4]) is [2 3]."),
    "setdiff": (["c = setdiff(a, b)", "c = setdiff(..., 'stable')", "[c, ia] = setdiff(...)"],
                "Values in a that are not in b, without repeats, sorted unless 'stable' is given; "
                "c = a(ia). Accepts cell arrays of text. The 'rows' option is ignored."),
    "setxor": (["c = setxor(a, b)", "c = setxor(..., 'stable')", "[c, ia, ib] = setxor(...)"],
               "Values found in exactly one of a and b, without repeats, sorted unless 'stable' is given. "
               "Accepts cell arrays of text. The 'rows' option is ignored."),
    # -- rearranging --------------------------------------------------------------------------
    "fliplr": ("B = fliplr(A)", "Reverses the order of the columns (left-right mirror). Works on cell arrays too."),
    "flipud": ("B = flipud(A)", "Reverses the order of the rows (up-down mirror). Works on cell arrays too."),
    "flip": (["B = flip(A)", "B = flip(A, dim)"],
             "Reverses the order of elements along the first dimension whose size is not 1, or along dim. "
             "Example: flip([1 2 3]) is [3 2 1]."),
    "rot90": (["B = rot90(A)", "B = rot90(A, k)"],
              "Rotates a matrix by 90 degrees counter-clockwise, k times (negative k rotates clockwise). "
              "Example: rot90([1 2; 3 4]) is [2 4; 1 3]."),
    "circshift": (["B = circshift(A, k)", "B = circshift(A, k, dim)", "B = circshift(A, [r c])"],
                  "Shifts elements circularly by k places, wrapping the ones that fall off the end back to "
                  "the start. Without dim the shift follows the first dimension whose size is not 1; a "
                  "two-element shift moves rows and columns. Example: circshift([1 2 3], 1) is [3 1 2]."),
    "triu": (["U = triu(A)", "U = triu(A, k)"],
             "Upper triangular part of A: entries below the k-th diagonal are set to zero (k = 0 is the "
             "main diagonal)."),
    "tril": (["L = tril(A)", "L = tril(A, k)"],
             "Lower triangular part of A: entries above the k-th diagonal are set to zero."),

    # -- linear algebra -----------------------------------------------------------------------
    "inv": ("Y = inv(A)",
            "Inverse of a square matrix, so that A*Y is the identity. A warning is printed when A is "
            "singular or nearly so, and the result is then unreliable (possibly Inf). For solving A*x = b, "
            "A\\b is faster and more accurate."),
    "pinv": ("Y = pinv(A)",
             "Moore-Penrose pseudoinverse, computed from the singular value decomposition. Works for "
             "rectangular and singular matrices. A tolerance argument is not supported."),
    "det": ("d = det(A)", "Determinant of a square matrix. Example: det([1 2; 3 4]) is -2."),
    "rank": ("r = rank(A)",
             "Number of linearly independent rows or columns, estimated from the singular values. "
             "A tolerance argument is not supported."),
    "trace": ("t = trace(A)", "Sum of the main diagonal of a square matrix."),
    "norm": (["n = norm(v)", "n = norm(v, p)", "n = norm(A)", "n = norm(A, p)"],
             "For a vector, the p-norm (sum(abs(v).^p))^(1/p); p defaults to 2, the Euclidean length, and "
             "Inf or -Inf give the largest or smallest magnitude. For a matrix, p may be 1, 2 (default, "
             "the largest singular value), Inf or 'fro' (Frobenius). Example: norm([3 4]) is 5."),
    "eig": (["e = eig(A)", "[V, D] = eig(A)", "e = eig(A, B)", "[V, D] = eig(A, B)"],
            "Eigenvalues of a square matrix as a column vector, or eigenvector matrix V and diagonal "
            "eigenvalue matrix D with A*V = V*D. With B, solves the generalized problem A*V = B*V*D. "
            "Symmetric (Hermitian) input gets real eigenvalues in ascending order; for other matrices the "
            "order and eigenvector scaling may differ from other implementations."),
    "svd": (["s = svd(A)", "[U, S, V] = svd(A)", "[U, S, V] = svd(A, 'econ')"],
            "Singular value decomposition A = U*S*V'. With one output, the singular values as a column in "
            "decreasing order. Passing any second argument (such as 'econ' or 0) gives the economy-size "
            "factors."),
    "qr": (["R = qr(A)", "[Q, R] = qr(A)", "[Q, R] = qr(A, 0)"],
           "QR factorization A = Q*R with Q orthogonal and R upper triangular. Any second argument "
           "(such as 0 or 'econ') gives the economy-size version. Column pivoting is not supported."),
    "lu": (["[L, U] = lu(A)", "[L, U, P] = lu(A)"],
           "LU factorization with partial pivoting. With three outputs, P*A = L*U where L is unit lower "
           "triangular. With two outputs, L already includes the row permutation, so A = L*U."),
    "chol": (["R = chol(A)", "L = chol(A, 'lower')", "[R, p] = chol(A)"],
             "Cholesky factor of a symmetric positive definite matrix: upper triangular R with R'*R = A, or "
             "lower triangular L with L*L' = A. If A is not positive definite it is an error, unless p is "
             "requested: then p is nonzero and R is empty (mylab does not return a partial factor)."),
    "expm": ("Y = expm(A)", "Matrix exponential of a square matrix (not the element-wise exp)."),
    "logm": ("Y = logm(A)", "Matrix logarithm, the inverse of expm. May be complex."),
    "sqrtm": ("Y = sqrtm(A)", "Matrix square root: Y*Y equals A. Not the same as the element-wise sqrt."),
    "null": ("Z = null(A)", "Orthonormal basis for the null space of A: the columns of Z satisfy A*Z = 0."),
    "orth": ("Q = orth(A)", "Orthonormal basis for the column space (range) of A."),
    "linsolve": ("x = linsolve(A, b)",
                 "Solves the linear system A*x = b, the same as A\\b. An options struct is not supported."),
    "cond": ("c = cond(A)",
             "2-norm condition number, the ratio of the largest to smallest singular value. Large values "
             "mean that solving with A loses accuracy. Other norms are not supported."),
    "vander": ("V = vander(v)",
               "Vandermonde matrix: column j holds v raised to the power n-j, so the last column is all ones."),
    "hilb": ("H = hilb(n)", "n-by-n Hilbert matrix with entries 1/(i+j-1), a classic badly conditioned example."),
    "toeplitz": (["T = toeplitz(c)", "T = toeplitz(c, r)"],
                 "Toeplitz matrix, constant along each diagonal, with first column c and first row r. With "
                 "one input the matrix is symmetric. Example: toeplitz([1 2 3]) is [1 2 3; 2 1 2; 3 2 1]."),
    # -- text: formatting and conversion -----------------------------------------------------
    "sprintf": ("s = sprintf(fmt, a1, a2, ...)",
                "Formats data into text using % conversions (%d, %i, %u, %f, %e, %g, %x, %o, %c, %s, %%) "
                "with optional flags, width and precision (a * takes them from the data). Escapes such as "
                "\\n and \\t are expanded. Array arguments are taken element by element and the format is "
                "reused until all data is consumed. A non-integer given to %d is shown in %e style; an integer "
                "given to %s is taken as a character code, so sprintf('%s', 65) is 'A'. Returns "
                "a char vector, or a string if fmt is a string. Example: sprintf('%d-%d', 1, 2) is '1-2'."),
    "fprintf": (["fprintf(fmt, a1, a2, ...)", "fprintf(fid, fmt, ...)", "n = fprintf(...)"],
                "Formats data like sprintf and writes it out. Without a file id, or with fid 1 or 2, the "
                "text goes to the command window; otherwise to a file opened with fopen. No newline is "
                "added automatically. The optional output is the number of bytes written."),
    "num2str": (["s = num2str(x)", "s = num2str(x, precision)", "s = num2str(x, format)"],
                "Converts numbers to text. Integers print in full; other values keep four digits after "
                "the decimal point (at least five significant digits for small values), with trailing "
                "zeros dropped. precision sets the number of significant digits and format is a sprintf "
                "format. Matrices give a char matrix with one row per row. Text input is returned as is. "
                "Example: num2str(pi) is '3.1416'."),
    "int2str": ("s = int2str(x)", "Rounds x to integers and converts the result to text like num2str."),
    "mat2str": (["s = mat2str(A)", "s = mat2str(A, n)"],
                "Text that recreates A when evaluated, such as '[1 2;3 4]'. Uses up to n significant digits "
                "(15 by default). Logical values print as true/false and char input is returned in quotes."),
    "str2num": (["x = str2num(s)", "[x, ok] = str2num(s)"],
                "Evaluates text as a bracketed expression, so '1 2 3' gives [1 2 3] and '[1;2]' a column. "
                "On failure returns [] and ok is false. Because the text is executed, prefer str2double "
                "for untrusted input."),
    "string": (["str = string(x)", "str = string()"],
               "Converts to string. A char row becomes a string scalar; a char matrix gives one string per "
               "row; a cell array of text gives a string array of the same size; numbers are converted "
               "element by element to their short text form, with NaN becoming <missing>. string() is the "
               "empty string \"\". Example: string([1 2]) is [\"1\" \"2\"]."),
    "strlength": ("n = strlength(s)",
                  "Number of characters in a char vector or string scalar. For a string array or cell array "
                  "of text, an array of lengths of the same size (NaN for <missing>)."),
    "newline": ("c = newline", "A char vector holding the line-feed character (code 10)."),
    "blanks": ("s = blanks(n)", "Char vector of n spaces."),
    # -- text: comparison and search ---------------------------------------------------------
    "strcmp": ("tf = strcmp(a, b)",
               "True if two texts are identical (case-sensitive). Either input may be a cell array or string "
               "array, giving an element-wise logical array. Non-text inputs give false rather than an error."),
    "strcmpi": ("tf = strcmpi(a, b)", "Like strcmp but ignores upper/lower case."),
    "strncmp": ("tf = strncmp(a, b, n)",
                "True if the first n characters of two texts match (case-sensitive). Texts shorter than n "
                "never match. Cell arrays compare element by element."),
    "strncmpi": ("tf = strncmpi(a, b, n)", "Like strncmp but ignores upper/lower case."),
    "strtrim": ("t = strtrim(s)",
                "Removes leading and trailing whitespace (and null characters). Works on char arrays, "
                "strings, string arrays and cell arrays of text."),
    "deblank": ("t = deblank(s)", "Removes trailing whitespace and null characters only."),
    "reverse": ("t = reverse(s)", "Reverses the order of the characters. Example: reverse('abc') is 'cba'."),
    "strrep": ("t = strrep(s, old, new)",
               "Replaces every occurrence of old in s with new. s may be a cell array or string array. "
               "Example: strrep('a-b-c', '-', '+') is 'a+b+c'."),
    "strfind": ("k = strfind(s, pattern)",
                "Starting indices of every occurrence of pattern in s, overlapping ones included, as a row "
                "vector ([] if none). For a cell array, returns a cell of index vectors. "
                "Example: strfind('abcabc', 'bc') is [2 5]."),
    "contains": (["tf = contains(s, pattern)", "tf = contains(s, pattern, 'IgnoreCase', true)"],
                 "True if s contains pattern anywhere. pattern may be a cell or string array, meaning any of "
                 "them. s may be a cell or string array, giving a logical array."),
    "startsWith": (["tf = startsWith(s, pattern)", "tf = startsWith(s, pattern, 'IgnoreCase', true)"],
                   "True if s begins with pattern (or any pattern in a cell array). Works element-wise on a "
                   "cell array s."),
    "endsWith": (["tf = endsWith(s, pattern)", "tf = endsWith(s, pattern, 'IgnoreCase', true)"],
                 "True if s ends with pattern (or any pattern in a cell array). Works element-wise on a "
                 "cell array s."),
    "strjoin": (["s = strjoin(C)", "s = strjoin(C, delimiter)"],
                "Joins the texts in a cell array or string array into one piece of text, separated by "
                "delimiter (a space by default). Escapes such as '\\n' in the delimiter are expanded. Returns "
                "a char vector for cell input and a string for string input."),
    "strcat": ("s = strcat(s1, s2, ...)",
               "Concatenates texts horizontally. Trailing whitespace of char inputs is removed first; "
               "strings keep theirs. If any input is a cell array the result is a cell array, joined "
               "element by element. Example: strcat('a ', 'b') is 'ab'."),
    "strvcat": ("S = strvcat(s1, s2, ...)",
                "Stacks texts as rows of a char matrix padded with spaces. Empty inputs are skipped."),
    "isspace": ("tf = isspace(s)", "True for each whitespace character in a char array; false for non-char input."),
    "isletter": ("tf = isletter(s)", "True for each alphabetic character in a char array; false for non-char input."),
    "isstrprop": ("tf = isstrprop(s, category)",
                  "Tests each character of s for a category: 'alpha', 'digit', 'alphanum', 'wspace', "
                  "'upper', 'lower', 'punct' or 'cntrl'."),
    # -- regular expressions -----------------------------------------------------------------
    "regexp": (["[tok, mat] = regexp(s, expr, 'tokens', 'match')", "out = regexp(s, expr, keyword, ...)",
                "[st, en, te, mat, tok, nm, sp] = regexp(s, expr)", "out = regexp(..., 'once')",
                "out = regexp(..., 'ignorecase')"],
               "Regular-expression search. Keywords 'match', 'tokens', 'names', 'start', 'end' and 'split' "
               "choose which results are returned, in the order given; without them the outputs are start "
               "indices, end indices, token extents (always empty in mylab), matches, tokens, named tokens "
               "and split pieces. 'once' returns only the first match, not wrapped in a cell. Named groups "
               "are written (?<name>...); 'names' gives a 1-by-N struct array with one element per match (a "
               "scalar struct with 'once'). Matching uses Python's regex engine, so rare constructs may "
               "differ. A cell array s gives a cell of results."),
    "regexpi": (["out = regexpi(s, expr, keyword, ...)"],
                "Case-insensitive form of regexp: the same outputs and options, but letters match "
                "regardless of case."),
    "regexprep": (["t = regexprep(s, expr, replacement)", "t = regexprep(..., 'ignorecase')",
                   "t = regexprep(..., 'once')"],
                  "Replaces every match of a regular expression. In the replacement, $1, $2, ... insert "
                  "captured groups and $0 the whole match. 'once' replaces only the first match. s may be a "
                  "cell array or string array. Example: regexprep('abc', '(b)', '[$1]') is 'a[b]c'."),
    # -- cell arrays --------------------------------------------------------------------------
    "cell": (["C = cell(n)", "C = cell(m, n)", "C = cell([m n])"],
             "Cell array whose cells all hold the empty matrix []. cell(n) is n-by-n."),
    "iscellstr": ("tf = iscellstr(C)",
                  "True if C is a cell array in which every cell holds a char vector (or empty char)."),
    "cellstr": ("C = cellstr(s)",
                "Converts text to a cell array of char vectors: each row of a char matrix becomes one cell "
                "(trailing blanks removed), giving a column; a string array gives a cell array of the same "
                "size (<missing> becomes ''); a cell array is returned as is."),
    "cell2mat": ("A = cell2mat(C)",
                 "Concatenates the contents of a cell array into one array, joining each row of cells side "
                 "by side and then stacking the rows. The pieces must fit together. "
                 "Example: cell2mat({1 2; 3 4}) is [1 2; 3 4]."),
    "num2cell": ("C = num2cell(A)", "Cell array the same size as A, with each element of A in its own cell."),
    "deal": (["[a, b, ...] = deal(x)", "[a, b, ...] = deal(x1, x2, ...)"],
             "Copies inputs to outputs: one input is copied to every output; otherwise there must be as "
             "many inputs as outputs and each goes to its own output."),
    "cellfun": (["A = cellfun(f, C)", "A = cellfun(f, C1, C2, ...)", "[A, B, ...] = cellfun(...)",
                 "R = cellfun(..., 'UniformOutput', false)", "A = cellfun(..., 'ErrorHandler', h)"],
                "Calls f on the contents of each cell (on matching cells of several same-size inputs). By "
                "default every call must return a scalar and the results form an array the size of C; with "
                "'UniformOutput' false they are collected in a cell array. f may be a handle or a function "
                "name; the names 'isempty', 'length', 'numel', 'isreal', 'islogical' and 'ndims' are handled "
                "directly. An error handler h is called with an error struct and the inputs. "
                "Example: cellfun(@numel, {'ab', 'c'}) is [2 1]."),
    "arrayfun": (["A = arrayfun(f, X)", "A = arrayfun(f, X1, X2, ...)", "[A, B, ...] = arrayfun(...)",
                  "R = arrayfun(..., 'UniformOutput', false)", "A = arrayfun(..., 'ErrorHandler', h)"],
                 "Calls f on each element of one or more same-size arrays, the way cellfun does for cells. "
                 "Elements of a struct array are passed as 1-by-1 structs and elements of a cell array as "
                 "1-by-1 cells. Example: arrayfun(@(k) k^2, 1:3) is [1 4 9]."),
    # -- structs ----------------------------------------------------------------------------------
    "struct": (["s = struct()", "s = struct(name1, value1, name2, value2, ...)"],
               "Creates a struct from field name/value pairs. A value that is a cell array creates a struct "
               "array the size of that cell, one element per cell; wrap a cell in another cell, e.g. "
               "{{1, 2}}, to store it as a single field value. Example: s = struct('x', 1, 'y', 'hi')."),
    "fieldnames": ("names = fieldnames(s)",
                   "Field names of a struct as a column cell array of char vectors, in creation order."),
    "isfield": (["tf = isfield(s, name)", "tf = isfield(s, {name1, name2, ...})"],
                "True if s is a struct with a field of that name; a cell array of names gives a logical "
                "array. Non-struct input gives false."),
    "rmfield": (["s = rmfield(s, name)", "s = rmfield(s, {name1, name2, ...})"],
                "Returns a copy of the struct without the named field(s). Naming a missing field is an error."),
    "setfield": ("s = setfield(s, name, value)",
                 "Returns a copy of the struct with the field set to value, the same as s.(name) = value. "
                 "Only a single field level is supported."),
    "getfield": ("v = getfield(s, name)",
                 "Value of the named field, the same as s.(name). Only a single field level is supported."),
    "struct2cell": ("C = struct2cell(s)",
                    "Column cell array holding the field values of a scalar struct, in field order."),
    "cell2struct": ("s = cell2struct(C, fields)",
                    "Builds a scalar struct whose fields, named by a cell array (or a single name), take "
                    "their values from the cells of C in order. A dimension argument and struct-array "
                    "results are not supported."),
    "orderfields": ("s = orderfields(s)",
                    "Returns the struct with its fields sorted alphabetically (by character code)."),
    "structfun": (["A = structfun(f, s)", "t = structfun(f, s, 'UniformOutput', false)"],
                  "Calls f on the value of each field of a scalar struct. By default the scalar results "
                  "form a column vector; with 'UniformOutput' false the results are returned as a struct "
                  "with the same field names."),

    # -- output and errors ------------------------------------------------------------------------
    "disp": ("disp(x)",
             "Print the value of x without showing a variable name. An empty char prints a blank line; "
             "other empty values print nothing. Example: disp('hello') prints hello."),
    "display": ("display(x)", "Prints x the way an unterminated statement would, with an 'ans =' header."),
    "error": (["error(msg)", "error(fmt, a1, ...)", "error(id, fmt, a1, ...)", "error(errstruct)"],
              "Throws an error that stops the program unless caught by try/catch. The message is formatted "
              "like sprintf when extra arguments or escapes are present. A first argument such as "
              "'pkg:name' (with a colon and no spaces) followed by more text is taken as the identifier. "
              "A struct with message and identifier fields can be passed instead. error([]) does nothing."),
    "rethrow": ("rethrow(err)",
                "Throws again an error caught with try/catch; err is a struct with message and identifier "
                "fields, such as the variable from catch err."),
    "throw": ("throw(ME)",
              "Throws the error described by ME, a struct with message and identifier fields such as one "
              "made by MException."),
    "MException": ("ME = MException(id, fmt, a1, ...)",
                   "Creates an error value with identifier, formatted message and an empty stack field, "
                   "ready to pass to throw. In mylab it is a plain struct, not an object."),
    "warning": (["warning(msg)", "warning(fmt, a1, ...)", "warning(id, fmt, a1, ...)", "warning('off')",
                 "warning('on')", "warning('off', 'all')"],
                "Prints a warning message and carries on. warning('off') or warning('off', 'all') silences "
                "all warnings and warning('on') restores them; turning off a single identifier is accepted "
                "but has no effect."),
    "lastwarn": ("msg = lastwarn", "Text of the most recent warning, or an empty char if there was none."),
    "assert": (["assert(cond)", "assert(cond, msg, a1, ...)", "assert(cond, id, msg, a1, ...)"],
               "Throws an error if cond is false (or not all true). The default message is 'Assertion "
               "failed.'; a custom message and identifier follow the same rules as error."),
    # -- timing and dates ------------------------------------------------------------------------
    "tic": (["tic", "t = tic"],
            "Starts a stopwatch. With an output, also returns a start-time value that can be passed to toc."),
    "toc": (["toc", "elapsed = toc", "elapsed = toc(t)"],
            "Seconds since the last tic (or since tic returned t). Without an output it prints "
            "'Elapsed time is ... seconds.'"),
    "pause": ("pause(n)",
              "Waits n seconds (at most an hour). pause with no argument, or with text such as 'on', returns "
              "immediately instead of waiting for a key press."),
    "clock": ("c = clock", "Current local date and time as [year month day hour minute seconds]."),
    "cputime": ("t = cputime",
                "Processor time used by mylab so far, in seconds. Differences between two calls time code."),
    "now": ("t = now",
            "Current date and time as a serial day number (days counted from year 0), with the time of day as a "
            "fraction."),
    "datestr": (["s = datestr", "s = datestr(t)"],
                "Formats a serial day number (now if omitted) as text like '05-Mar-2024 14:07:30' in local "
                "time. A format argument is not supported."),
    # -- workspace ---------------------------------------------------------------------------------
    "clc": ("clc", "Clears the command window when the host program supports it; otherwise does nothing."),
    "clear": (["clear", "clear name1 name2 ...", "clear pattern*", "clear all", "clear global", "clear global name"],
              "Removes variables from the current workspace: all of them, the named ones, or those matching "
              "a wildcard pattern. 'clear all' also clears globals and closes figures; 'clear global' "
              "removes global variables. 'clear functions' is accepted and does nothing."),
    "who": (["who", "names = who"],
            "Lists the names of the variables in the current workspace, or returns them as a column cell array."),
    "whos": ("whos",
             "Prints a table of the current variables with their sizes, byte counts and classes. Byte counts "
             "are reported for numeric, logical and char arrays only."),
    "exist": (["code = exist(name)", "code = exist(name, kind)"],
              "Tells what a name refers to: 1 for a variable, 2 for a file or a function defined in a .m "
              "file or in the session, 5 for a built-in function, 7 for a folder, 0 if not found. kind "
              "('var', 'file', 'dir' or 'builtin') limits the search."),
    "isvarname": ("tf = isvarname(s)",
                  "True if s is a valid variable name: a letter followed by letters, digits or underscores, "
                  "at most 63 characters, and not a keyword."),
    "iskeyword": (["tf = iskeyword(s)", "list = iskeyword"],
                  "True if s is a reserved word of the language such as 'for' or 'end'. With no input, "
                  "returns all keywords as a column cell array."),
    "nargin": ("n = nargin",
               "Inside a function, the number of inputs the caller passed. nargin(fname) is not supported."),
    "nargout": ("n = nargout", "Inside a function, the number of outputs the caller asked for."),
    "narginchk": ("narginchk(minargs, maxargs)",
                  "Inside a function, throws an error if the number of inputs passed is below minargs or "
                  "above maxargs."),
    "nargoutchk": ("nargoutchk(minargs, maxargs)", "Accepted for compatibility; mylab does not check anything."),
    "exit": (["exit", "exit(code)"],
             "Ends the mylab session, optionally with a numeric exit status for the operating system."),
    # -- evaluation ------------------------------------------------------------------------------
    "feval": ("[y1, ...] = feval(f, x1, ...)",
              "Calls the function f, given as a handle or a name, with the remaining arguments. "
              "Example: feval('max', [3 8 1]) is 8."),
    "func2str": ("s = func2str(f)",
                 "Text form of a function handle: the function name, or the full expression starting with @ "
                 "for an anonymous function."),
    "str2func": ("f = str2func(s)",
                 "Makes a function handle from a name such as 'sin' or from anonymous-function text such as "
                 "'@(x) x + 1'."),
    "eval": (["eval(code)", "eval(code, catchcode)", "[y1, ...] = eval(expr)"],
             "Runs text as code in the current workspace. If it fails and catchcode is given, that is run "
             "instead. With outputs, code must be a single expression whose values are returned. Building "
             "variable names with eval is error-prone; prefer arrays, cells or structs."),
    "evalc": ("s = evalc(code)", "Runs text as code and returns everything it would have printed, as a char vector."),
    "input": (["x = input(prompt)", "s = input(prompt, 's')"],
              "Shows prompt and reads a line from the user. The reply is evaluated as an expression, so "
              "typing 2*3 gives 6; an empty reply gives []. With 's' the raw text is returned instead."),
    "keyboard": ("keyboard", "Accepted for compatibility; mylab has no interactive debugger, so it does nothing."),
    "format": (["format", "format long", "format short", "format longG", "format shortG"],
               "Chooses how many digits numbers display with: long shows about 15 significant digits, short "
               "(the default) about 5. 'compact' and 'loose' are accepted but change nothing."),
    "more": ("more on|off", "Accepted for compatibility; output is never paged, so it does nothing."),
    "beep": ("beep", "Accepted for compatibility; makes no sound."),
    "commandwindow": ("commandwindow", "Accepted for compatibility; does nothing."),
    "pack": ("pack", "Accepted for compatibility; does nothing."),
    "dbstop": ("dbstop ...", "Accepted for compatibility; mylab has no debugger, so no breakpoint is set."),
    "profile": ("profile on|off|viewer", "Accepted for compatibility; no profiling is done."),
    # -- environment -----------------------------------------------------------------------------
    "version": ("v = version", "mylab's version as text, e.g. 'mylab 0.1.0'."),
    "ver": ("ver", "Prints the versions of mylab, Python and NumPy."),
    "isunix": ("tf = isunix", "True when running on a POSIX system (Linux or macOS)."),
    "ispc": ("tf = ispc", "True when running on Windows."),
    "ismac": ("tf = ismac", "True when running on macOS."),
    "isdeployed": ("tf = isdeployed", "Always false: mylab code never runs as a compiled standalone application."),
    "computer": ("s = computer",
                 "Name of the platform as reported by Python, e.g. 'linux', 'darwin' or 'win32'. These codes "
                 "differ from the architecture strings other implementations use."),
    "getenv": ("v = getenv(name)", "Value of an operating-system environment variable, or '' if it is not set."),
    "help": (["help name", "s = help(name)"],
             "Shows the calling forms and a short description of a function. For functions and classdef "
             "classes in your own .m files it shows the comment block at the top of the file. With an "
             "output the text is returned instead of printed."),
    "doc": ("doc name", "Same as help: prints the help text in the command window (no browser is opened)."),
    # -- files and folders -----------------------------------------------------------------------
    "pwd": ("d = pwd", "Current working folder as a char vector."),
    "cd": (["cd", "cd(folder)", "d = cd"],
           "Changes the current working folder (a leading ~ means your home folder). With no input, prints "
           "or returns the current folder."),
    "addpath": ("addpath(folder1, folder2, ...)",
                "Adds folders to the front of the search path used to find .m files. A single argument may "
                "list several folders separated by the system path separator. Flags such as '-end' are "
                "ignored, so folders are always added at the front."),
    "rmpath": ("rmpath(folder1, ...)", "Removes folders from the search path."),
    "path": ("p = path", "Current search path as one char vector with folders separated by the system path separator."),
    "fullfile": ("f = fullfile(part1, part2, ...)",
                 "Joins folder and file names with the operating system's separator. "
                 "Example: fullfile('data', 'run1.csv') is 'data/run1.csv' on Linux."),
    "fileparts": ("[folder, name, ext] = fileparts(f)",
                  "Splits a file path into its folder, base name and extension (with the dot). "
                  "Example: fileparts('a/b.txt') gives 'a', 'b' and '.txt'."),
    "isfile": ("tf = isfile(f)", "True if f names an existing file (not a folder)."),
    "isfolder": ("tf = isfolder(f)", "True if f names an existing folder."),
    "mkdir": (["mkdir(folder)", "mkdir(parent, folder)", "ok = mkdir(...)"],
              "Creates a folder, including any missing parent folders. It is not an error if the folder "
              "already exists. The optional output is true."),
    "delete": ("delete(file1, file2, ...)",
               "Deletes the named files. Names that do not exist are skipped silently; wildcards are not supported."),
    "fileread": ("s = fileread(filename)", "Entire contents of a text file (read as UTF-8) as one char vector."),
    "fopen": (["fid = fopen(filename)", "fid = fopen(filename, permission)", "[fid, msg] = fopen(...)"],
              "Opens a text file and returns a file id (3 or higher) for fprintf, fgetl, fgets, feof and "
              "fclose, or -1 on failure with the reason in msg. permission is 'r' (default), 'w', 'a', "
              "'r+', 'w+' or 'a+'; a 't' or 'b' suffix is accepted. Files are read and written as UTF-8 text."),
    "fclose": (["fclose(fid)", "fclose('all')", "status = fclose(...)"],
               "Closes a file opened with fopen, or all of them. An unknown fid is an error."),
    "fgetl": ("line = fgetl(fid)",
              "Reads the next line from a file without its line ending. Returns -1 at end of file."),
    "fgets": ("line = fgets(fid)", "Reads the next line from a file, keeping its newline. Returns -1 at end of file."),
    "feof": ("tf = feof(fid)", "True if there is nothing more to read from the file."),
    "save": (["save", "save(filename)", "save(filename, 'var1', 'var2', ...)", "save filename var1 var2"],
             "Writes variables from the workspace to a .mat file (the extension is added if missing; the "
             "default name is matlab.mat). Only numeric, logical, char and string variables are saved; "
             "cells, structs and handles are skipped. Option flags such as -ascii are ignored."),
    "load": (["load(filename)", "S = load(filename)", "load filename"],
             "Reads variables from a .mat file into the workspace, or into the fields of S when an output is "
             "given. Numeric data is loaded as double. A .csv, .txt or .dat file is read as a numeric matrix "
             "and stored in a variable named after the file (or returned)."),
    # -- polynomials ---------------------------------------------------------------------------
    "roots": ("r = roots(p)", "Roots of the polynomial with coefficients p, as a column vector (possibly complex)."),
    "poly": (["p = poly(r)", "p = poly(A)"],
             "Polynomial coefficients from a vector of roots r, or the characteristic polynomial of a "
             "square matrix A. Example: poly([1 2]) is [1 -3 2]."),
    "polyder": ("dp = polyder(p)",
                "Coefficients of the derivative of a polynomial. The product and quotient forms with two "
                "inputs are not supported."),
    "polyint": (["ip = polyint(p)", "ip = polyint(p, k)"],
                "Coefficients of the antiderivative of a polynomial, with constant of integration k (default 0)."),
    "conv": (["w = conv(u, v)", "w = conv(u, v, shape)"],
             "Convolution of two vectors, which is also polynomial multiplication. shape is 'full' (default, "
             "length numel(u)+numel(v)-1), 'same' (central part the size of u) or 'valid'. The result is a "
             "column if u is a column. Example: conv([1 1], [1 -1]) is [1 0 -1]."),
    "deconv": ("[q, r] = deconv(b, a)",
               "Polynomial division: b = conv(a, q) + r. r has the same length as b, padded with leading zeros."),
    "filter": ("y = filter(b, a, x)",
               "Applies the rational digital filter with numerator b and denominator a to the data x, along "
               "a row vector or down the columns of a matrix. Initial conditions are not supported. "
               "Example: filter(1, [1 -0.5], [1 1 1]) is [1 1.5 1.75]."),
    # -- interpolation ---------------------------------------------------------------------------
    # -- solving and optimization ----------------------------------------------------------------
    "optimset": (["opts = optimset(name, value, ...)"],
                 "Builds an options struct from name/value pairs for fzero, fminsearch and similar solvers. "
                 "mylab reads TolX, TolFun, MaxIter and MaxFunEvals where they apply and ignores other names."),
    "optimoptions": ("opts = optimoptions(solvername, name, value, ...)",
                     "Same as optimset; the solver name is accepted and ignored. Returns a plain struct."),
    "fsolve": (["x = fsolve(f, x0)", "[x, fval, exitflag] = fsolve(...)"],
               "Solves the system of equations f(x) = 0, starting from x0. f returns a vector of residuals. "
               "Options are ignored."),
    # -- integration -----------------------------------------------------------------------------
    "integral": (["q = integral(f, a, b)", "q = integral(f, a, b, 'AbsTol', at, 'RelTol', rt)"],
                 "Numerical integral of f from a to b by adaptive quadrature; a and b may be Inf or -Inf. "
                 "mylab calls f with one scalar at a time, so f need not be vectorized. Other options such "
                 "as 'ArrayValued' or 'Waypoints' are not supported. Example: integral(@(x) x.^2, 0, 3) is 9."),
    "quad": (["q = quad(f, a, b)", "q = quadgk(f, a, b)", "q = quadl(f, a, b)"],
             "Older names for numerical integration; in mylab quad, quadgk and quadl all behave exactly like "
             "integral, including the 'AbsTol' and 'RelTol' options. A positional tolerance argument is "
             "ignored."),
    "integral2": ("q = integral2(f, xmin, xmax, ymin, ymax)",
                  "Numerical double integral of f(x, y) over xmin <= x <= xmax and ymin <= y <= ymax. ymin "
                  "and ymax may be numbers or functions of x. f is called with scalars. Options are not "
                  "supported."),
    # -- ODE solvers -----------------------------------------------------------------------------
    # -- plotting: lines and charts --------------------------------------------------------------
    "plot": (["plot(y)", "plot(x, y)", "plot(x, y, linespec)", "plot(x1, y1, s1, x2, y2, s2, ...)",
              "plot(..., Name, Value)"],
             "Draws 2-D lines. With one input, y is plotted against its index (complex y plots real against "
             "imaginary part). A matrix y draws one line per column. linespec combines a colour "
             "(r g b c m y k w), a marker (o + * . x s d ^ v < > p h) and a line style (- -- : -.), e.g. "
             "'r--o'. Name/Value options: LineWidth, Color, MarkerSize, Marker, LineStyle, DisplayName, "
             "MarkerFaceColor, MarkerEdgeColor. Replaces the current plot unless hold is on; no handle is "
             "returned. Example: plot(0:0.1:1, (0:0.1:1).^2, 'b-')."),
    "semilogx": (["semilogx(x, y)", "semilogx(x, y, linespec)", "semilogx(..., Name, Value)"],
                 "Like plot, but with a logarithmic x axis."),
    "semilogy": (["semilogy(x, y)", "semilogy(x, y, linespec)", "semilogy(..., Name, Value)"],
                 "Like plot, but with a logarithmic y axis."),
    "loglog": (["loglog(x, y)", "loglog(x, y, linespec)", "loglog(..., Name, Value)"],
               "Like plot, but with logarithmic x and y axes."),
    "plot3": (["plot3(x, y, z)", "plot3(x, y, z, linespec)", "plot3(..., Name, Value)"],
              "Draws a line through 3-D points on 3-D axes. Accepts the same line options as plot."),
    "scatter": (["scatter(x, y)", "scatter(x, y, sz)", "scatter(x, y, sz, c)", "scatter(..., 'filled')",
                 "scatter(x, y, sz, c, marker)"],
                "Draws a marker at each point (x, y). sz is the marker area (default 36), scalar or per "
                "point; c is a colour name, an RGB triplet or one value per point mapped through the "
                "colormap. Markers are hollow unless 'filled' is given. DisplayName sets the legend label."),
    "bar": (["bar(y)", "bar(x, y)"],
            "Vertical bar chart of y, at positions x or 1..n. A matrix y draws groups of bars, one group per "
            "row and one colour per column. Width and style arguments are ignored."),
    "barh": (["barh(y)", "barh(x, y)"], "Horizontal bar chart of the vector y, at positions x or 1..n."),
    "histogram": (["histogram(x)", "histogram(x, nbins)", "histogram(x, edges)"],
                  "Plots a histogram of the finite values in x, with automatic bins, a bin count or bin "
                  "edges. No histogram object is returned."),
    "hist": (["hist(x)", "hist(x, nbins)", "counts = hist(x, nbins)", "[counts, centers] = hist(...)"],
             "Without outputs, plots a histogram like histogram. With outputs, returns the bin counts (and "
             "bin centres) without plotting, using 10 equal-width bins by default."),
    "stem": (["stem(y)", "stem(x, y)"],
             "Plots each value as a vertical stem from zero topped with a marker, as for discrete signals."),
    "stairs": (["stairs(y)", "stairs(x, y)"], "Draws y as a staircase that holds each value until the next x."),
    "area": (["area(y)", "area(x, y)"],
             "Plots the curve y and shades the region between it and zero. Only a single vector is supported."),
    "fill": (["fill(x, y, c)", "fill(x, y)"],
             "Draws a filled polygon with corners (x(k), y(k)); c is a colour name or RGB triplet."),
    "errorbar": (["errorbar(y, err)", "errorbar(x, y, err)", "errorbar(..., linespec)"],
                 "Plots y with vertical error bars of half-length err at each point. Only symmetric error "
                 "bars are supported."),
    "pie": (["pie(x)", "pie(x, labels)"],
            "Pie chart of the values in x, each slice proportional to its share of the total. Without "
            "labels each slice shows its percentage; labels is a cell array of text."),
    # -- plotting: surfaces and images -------------------------------------------------------------
    "surf": (["surf(Z)", "surf(X, Y, Z)"],
             "Draws a coloured 3-D surface with heights Z over the grid X, Y (vectors are expanded to a "
             "grid; without them the indices are used). It always replaces the figure's contents. surfc "
             "currently draws the same plot without the contour lines underneath."),
    "mesh": (["mesh(Z)", "mesh(X, Y, Z)"],
             "Draws a 3-D wireframe surface. It replaces the figure's contents. meshc currently draws the "
             "same plot without contour lines."),
    "contour": (["contour(Z)", "contour(X, Y, Z)", "contour(X, Y, Z, n)"],
                "Draws contour lines (lines of equal height) of Z, optionally over the grid X, Y and with n "
                "levels. It replaces the figure's contents."),
    "contourf": (["contourf(Z)", "contourf(X, Y, Z)", "contourf(X, Y, Z, n)"],
                 "Like contour, but fills the regions between the contour lines with colour."),
    "imagesc": (["imagesc(C)", "imagesc(x, y, C)"],
                "Shows the matrix C as an image, mapping its value range onto the current colormap; row 1 "
                "is at the top. The x and y arguments are accepted but the axes always show indices. image "
                "and imshow behave the same way in mylab."),
    "colorbar": ("colorbar",
                 "Adds a colour scale beside the current axes for the most recent image, surface or contour plot."),
    "colormap": ("colormap(name)",
                 "Chooses the colour map by name for later surfaces and images, e.g. 'jet', 'hot', 'gray' or "
                 "'viridis' (the default); any matplotlib colormap name works. Numeric colormap matrices are "
                 "not supported."),
    "view": (["view(az, el)", "view(2)"],
             "Sets the viewing angle of the current 3-D axes: azimuth az and elevation el in degrees. view(2) "
             "looks straight down. Has no effect on 2-D axes."),
    # -- plotting: figures and axes ----------------------------------------------------------------
    "figure": (["figure", "figure(n)", "n = figure(...)", "figure(..., 'Name', title)"],
               "Opens a new figure window, or makes figure n current (creating it if needed). Later plots go "
               "to the current figure. The output is the figure number, not a graphics object."),
    "gcf": ("n = gcf", "Number of the current figure, creating one if none exists."),
    "gca": ("n = gca",
            "Makes sure the current figure has axes and returns the figure number. mylab has no graphics "
            "objects, so the result cannot be used with set or get."),
    "clf": ("clf", "Clears everything from the current figure."),
    "close": (["close", "close(n)", "close all"], "Closes the current figure, figure n, or every figure."),
    "hold": (["hold on", "hold off", "hold"],
             "With hold on, new plots are added to the current axes instead of replacing them; hold off "
             "returns to replacing. hold alone toggles."),
    "ishold": ("tf = ishold", "True if hold is on for the current axes."),
    "subplot": (["subplot(m, n, p)", "subplot(mnp)", "subplot(m, n, [p1 p2 ...])"],
                "Splits the figure into an m-by-n grid of axes and makes axes p current, counting along "
                "rows from the top left. A vector p makes one axes spanning those cells. subplot(221) is "
                "shorthand for subplot(2, 2, 1)."),
    "title": (["title(txt)", "title(txt, 'FontSize', n)"],
              "Sets the title of the current axes. A cell array or multi-row char array gives several lines."),
    "xlabel": (["xlabel(txt)", "xlabel(txt, 'FontSize', n)"], "Labels the x axis of the current axes."),
    "ylabel": (["ylabel(txt)", "ylabel(txt, 'FontSize', n)"], "Labels the y axis of the current axes."),
    "zlabel": (["zlabel(txt)", "zlabel(txt, 'FontSize', n)"],
               "Labels the z axis of the current 3-D axes; ignored on 2-D axes."),
    "sgtitle": (["sgtitle(txt)", "sgtitle(txt, 'FontSize', n)"],
                "Sets one title for the whole figure, above all subplots. suptitle does the same."),
    "legend": (["legend(label1, label2, ...)", "legend({label1, label2, ...})", "legend(..., 'Location', loc)",
                "legend", "legend off"],
               "Adds a legend naming the plotted items in drawing order. With no labels it uses DisplayName "
               "values. loc is a compass name such as 'northeast' (top right), 'southwest' or 'best'. "
               "legend off removes it."),
    "grid": (["grid on", "grid off", "grid minor", "grid"],
             "Shows or hides grid lines on the current axes; grid alone toggles."),
    "box": ("box on|off", "Accepted for compatibility; the axes box is not changed."),
    "axis": (["axis([xmin xmax ymin ymax])", "axis([xmin xmax ymin ymax zmin zmax])", "axis equal", "axis square",
              "axis tight", "axis off", "axis on", "axis auto", "axis ij", "lims = axis"],
             "Sets axis limits or appearance: 'equal' makes units the same length on both axes, 'square' "
             "makes the plot box square, 'tight' fits the limits to the data, 'off' hides the axes, 'auto' "
             "restores automatic limits and 'ij' puts the y origin at the top. With an output and no input, "
             "returns [xmin xmax ymin ymax]."),
    "xlim": (["xlim([xmin xmax])", "lims = xlim", "xlim auto"],
             "Sets or returns the x-axis limits of the current axes."),
    "ylim": (["ylim([ymin ymax])", "lims = ylim", "ylim auto"],
             "Sets or returns the y-axis limits of the current axes."),
    "zlim": (["zlim([zmin zmax])", "lims = zlim", "zlim auto"],
             "Sets or returns the z-axis limits of the current 3-D axes."),
    "xticks": ("xticks(v)", "Puts the x-axis tick marks at the values in v. Querying the ticks is not supported."),
    "yticks": ("yticks(v)", "Puts the y-axis tick marks at the values in v. Querying the ticks is not supported."),
    "xticklabels": ("xticklabels(labels)",
                    "Accepted for compatibility but currently ignored; tick labels are not changed."),
    "text": ("text(x, y, txt)", "Writes text on the current axes with its left end at data point (x, y)."),
    "drawnow": ("drawnow",
                "Updates figure windows so that pending drawing appears, e.g. inside an animation loop. "
                "shg and refresh do the same."),
    # -- plotting: saving -------------------------------------------------------------------------
    "saveas": (["saveas(fig, filename)", "saveas(fig, filename, format)"],
               "Saves figure number fig (e.g. from gcf) as an image file. The type comes from the file "
               "extension (png, pdf, svg, jpg, ...) or from format when the name has no extension."),
    "print": (["print(filename)", "print('-dpng', filename)", "print(fig, '-dpdf', filename)"],
              "Saves the current figure (or figure fig) to a file. A -d option such as -dpng, -djpeg or "
              "-dpdf picks the format when the file name has no extension; PNG is the default. Resolution "
              "options are ignored."),
    "exportgraphics": ("exportgraphics(fig, filename)",
                       "Saves figure fig (gcf or gca both work) to an image file, with the format taken from "
                       "the extension. Same as saveas."),
}

# -- aliases that share an entry ---------------------------------------------------------------
_ALIASES = {
    "nan": "NaN", "inf": "Inf", "j": "i",
    "home": "clc", "clearvars": "clear", "quit": "exit", "isdir": "isfolder",
    "patch": "fill", "image": "imagesc", "imshow": "imagesc",
    "surfc": "surf", "meshc": "mesh", "suptitle": "sgtitle",
    "yticklabels": "xticklabels", "shg": "drawnow", "refresh": "drawnow",
}
for _alias, _target in _ALIASES.items():
    HELP[_alias] = HELP[_target]

# -- integer conversions -------------------------------------------------------------------------
_INT_RANGES = {
    "int8": "-128..127", "int16": "-32768..32767", "int32": "-2147483648..2147483647",
    "int64": "about -9.2e18..9.2e18", "uint8": "0..255", "uint16": "0..65535",
    "uint32": "0..4294967295", "uint64": "0..about 1.8e19",
}
for _cls, _range in _INT_RANGES.items():
    HELP[_cls] = (f"y = {_cls}(x)",
                  f"Converts to the {_cls} integer class: values are rounded to the nearest integer (halves "
                  f"away from zero), clipped to the range {_range}, and NaN becomes 0. "
                  f"Example: {_cls}(2.5) is 3.")
HELP["quadgk"] = HELP["quadl"] = HELP["quad"]

# -- string arrays, objects and containers ------------------------------------------------------
HELP.update({
    "strings": (["s = strings(n)", "s = strings(m, n, ...)"],
                "Creates a string array of the given size filled with empty strings \"\". strings(1, 1) is the "
                "empty string scalar."),
    "ismissing": ("tf = ismissing(x)",
                  "True where an element is missing: <missing> in a string array, NaN in a numeric array, or "
                  "an empty char vector in a cell array. Returns a logical array the size of x."),
    "isobject": ("tf = isobject(x)",
                 "True if x is an object: a containers.Map, a table, or an instance of a classdef class."),
    "containers.Map": (["m = containers.Map()", "m = containers.Map(keys, values)",
                        "m = containers.Map('KeyType', kt, 'ValueType', vt)"],
                       "Creates a map from keys (char vectors or numbers) to values. Read and write entries "
                       "with m(key) and m(key) = value; use keys(m), values(m), isKey(m, k), remove(m, k) and "
                       "m.Count. A map is a handle: copying the variable shares the same map."),
})

# -- tables -------------------------------------------------------------------------------------
HELP.update({
    "cell2table": (["T = cell2table(C)", "T = cell2table(C, 'VariableNames', names)"],
                   "Makes a table with one variable per column of the cell array C. Columns holding only "
                   "numeric scalars become numeric variables; others stay cell columns."),
    "struct2table": ("T = struct2table(S)",
                     "Converts a struct array (one row per element) or a scalar struct of column vectors into a "
                     "table whose variables are the struct fields."),
    "table2struct": ("S = table2struct(T)", "Converts a table into an n-by-1 struct array, one element per row."),
    "table2array": ("A = table2array(T)",
                    "Concatenates all table variables side by side into one array. The variables must have "
                    "compatible types."),
    "height": ("n = height(T)", "Number of rows of a table (or of an array)."),
    "width": ("n = width(T)", "Number of variables (columns) of a table."),
    "head": (["T2 = head(T)", "T2 = head(T, k)"], "First k rows of a table or array (default 8)."),
    "tail": (["T2 = tail(T)", "T2 = tail(T, k)"], "Last k rows of a table or array (default 8)."),
    "removevars": ("T2 = removevars(T, vars)", "Removes the named (or numbered) variables from a table."),
    "addvars": (["T2 = addvars(T, var1, ...)", "T2 = addvars(..., 'NewVariableNames', names)"],
                "Appends variables (columns) to the right of a table."),
    "renamevars": ("T2 = renamevars(T, oldNames, newNames)", "Renames table variables."),
    "istable": ("tf = istable(x)", "True if x is a table."),
    "rows2vars": ("T2 = rows2vars(T)",
                  "Transposes a numeric table: each original row becomes a variable, and a first variable "
                  "OriginalVariableNames lists the old column names."),
    "readtable": (["T = readtable(file)", "T = readtable(file, 'Delimiter', d)",
                   "T = readtable(..., 'ReadVariableNames', false)", "T = readtable(..., 'TextType', 'string')"],
                  "Reads a delimited text file (CSV, tab, semicolon...) into a table. The first row gives the "
                  "variable names; all-numeric columns become numbers (blank cells become NaN), other columns "
                  "become cell arrays of text (or strings with 'TextType', 'string')."),
    "writetable": (["writetable(T, file)", "writetable(T, file, 'Delimiter', d)",
                    "writetable(..., 'WriteVariableNames', false)"],
                   "Writes a table to a delimited text file with a header row of variable names."),
})
HELP["summary"] = ("summary(T)",
                   "Prints a short description of each table variable: its size and type, plus the minimum, "
                   "median and maximum of numeric variables.")
HELP["run"] = ("run(file)",
               "Runs a script file (with or without the .m extension) in the current workspace, so its variables "
               "stay available afterwards. The script's folder is added to the search path.")

# -- text toolbox functions (char, string, string array or cell array input) ----------------------
_TEXT_IN = ("s may be a char vector, a string, a string array or a cell array of text; string input gives "
            "string output.")
HELP.update({
    "split": (["parts = split(s)", "parts = split(s, delimiter)", "parts = split(..., 'CollapseDelimiters', tf)"],
              "Splits text at whitespace, or at a delimiter (a cell or string array lists several). Runs of "
              "delimiters count as one break unless 'CollapseDelimiters' is false. One input text gives a "
              "column of pieces (a cell of char vectors, or a string array for string input); a cell array "
              "whose elements all split into the same number of pieces gives a matrix. "
              "Example: split(\"a,b\", \",\") is [\"a\"; \"b\"]."),
    "compose": (["str = compose(fmt)", "str = compose(fmt, A)", "str = compose(fmt, A1, A2, ...)"],
                "Formats data into text with sprintf-style conversions, giving one piece of text per row "
                "rather than one long result. With a single array, each row is fed to the format, using as "
                "many columns per piece as the format has conversions; with several arrays, matching "
                "elements are formatted together. Returns a cell array of char vectors, or a string array "
                "when fmt is a string. Example: compose('%d-%d', [1 2; 3 4]) is {'1-2'; '3-4'}."),
    "replace": ("t = replace(s, old, new)",
                "Replaces every occurrence of the literal text old with new (no regular expressions). old may "
                "list several patterns, with new either one replacement for all or one per pattern; longer "
                f"patterns win where they overlap. {_TEXT_IN}"),
    "erase": ("t = erase(s, pattern)",
              f"Deletes every occurrence of pattern (or of each pattern in a list) from the text. {_TEXT_IN} "
              "Example: erase('a-b-c', '-') is 'abc'."),
    "count": (["n = count(s, pattern)", "n = count(s, pattern, 'IgnoreCase', true)"],
              "Number of non-overlapping occurrences of pattern (or of any pattern in a list) in s. A cell "
              "array of text gives an array of counts. Example: count('banana', 'an') is 2."),
    "extractBefore": (["t = extractBefore(s, pattern)", "t = extractBefore(s, pos)"],
                      "Text before the first occurrence of pattern, or before character position pos. If the "
                      f"pattern does not occur the result is empty text. {_TEXT_IN}"),
    "extractAfter": (["t = extractAfter(s, pattern)", "t = extractAfter(s, pos)"],
                     "Text after the first occurrence of pattern, or after character position pos. If the "
                     f"pattern does not occur the result is empty text. {_TEXT_IN}"),
    "extractBetween": (["t = extractBetween(s, startPat, endPat)", "t = extractBetween(s, startPos, endPos)",
                        "t = extractBetween(..., 'Boundaries', 'inclusive')"],
                       "Text between start and end markers (excluded unless 'Boundaries' is 'inclusive'), or "
                       "between two character positions (both included). For a char vector every match is "
                       "returned as a column cell array; for a cell array each element must contain exactly "
                       "one match. Example: extractBetween('a[b]c[d]', '[', ']') is {'b'; 'd'}."),
    "insertAfter": (["t = insertAfter(s, pattern, new)", "t = insertAfter(s, pos, new)"],
                    "Inserts new after every occurrence of pattern, or after character position pos. "
                    f"{_TEXT_IN}"),
    "insertBefore": (["t = insertBefore(s, pattern, new)", "t = insertBefore(s, pos, new)"],
                     "Inserts new before every occurrence of pattern, or before character position pos. "
                     f"{_TEXT_IN}"),
    "pad": (["t = pad(s)", "t = pad(s, n)", "t = pad(s, side)", "t = pad(s, n, side, ch)"],
            "Pads text with spaces (or the single character ch) to length n. side is 'right' (default), "
            "'left' or 'both'. Without n, the elements of a cell or string array are padded to the length "
            "of the longest one. Text already longer than n is left alone. Example: pad('ab', 5, 'left', '*') "
            "is '***ab'."),
    "strip": (["t = strip(s)", "t = strip(s, side)", "t = strip(s, side, ch)", "t = strip(s, ch)"],
              "Removes leading and trailing whitespace, or the single character ch. side is 'both' "
              f"(default), 'left' or 'right'. {_TEXT_IN}"),
    "strjust": (["T = strjust(C)", "T = strjust(C, mode)"],
                "Moves the text in each row of a char matrix to the 'right' (default), 'left' or 'center', "
                "keeping the width. Only char arrays are accepted; strings and cell arrays are an error."),
    "sscanf": (["A = sscanf(s, fmt)", "A = sscanf(s, fmt, sz)", "[A, count, errmsg, next] = sscanf(...)"],
               "Reads values from text according to a format, the reverse of sprintf. Supports %d, %i, %u, "
               "%o, %x, %f, %e, %g, %s, %c and %[...] with optional field width and * to skip a field; the "
               "format repeats until the text is used up or does not match. Numbers come back as a column; "
               "if everything read is text the result is a char row. sz (n, Inf or [m n]) shapes the output, "
               "filling columns first. count is the number of fields read; next is always one past the end. "
               "Example: sscanf('3 4 5', '%d') is [3; 4; 5]."),
})

# -- indexing, array and cell helpers -----------------------------------------------------------
HELP.update({
    "sub2ind": ("ind = sub2ind(sz, r, c, ...)",
                "Converts subscripts into linear indices for an array of size sz (column-major numbering). "
                "Subscript arrays must have matching sizes; scalars are expanded. With fewer subscripts than "
                "dimensions the last one spans the remaining dimensions. Out-of-range subscripts are an "
                "error. Example: sub2ind([3 4], 2, 3) is 8."),
    "ind2sub": (["[r, c] = ind2sub(sz, ind)", "[d1, d2, ..., dn] = ind2sub(sz, ind)"],
                "Converts linear indices into subscripts for an array of size sz. With fewer outputs than "
                "dimensions, the last output counts through the remaining dimensions. "
                "Example: [r, c] = ind2sub([3 4], 8) gives r = 2, c = 3."),
    "perms": ("P = perms(v)",
              "Every ordering of the elements of v, one per row, n! rows in total, starting with the reverse "
              "of v. Also works on cell arrays. Limited to 11 elements. Example: perms([1 2]) is [2 1; 1 2]."),
    "bsxfun": ("C = bsxfun(f, A, B)",
               "Expands singleton dimensions of A and B so their sizes match, then calls the binary function "
               "f once with the expanded arrays. Plain operators already expand like this, e.g. A - mean(A)."),
    "idivide": (["C = idivide(a, b)", "C = idivide(a, b, opt)"],
                "Integer division with a chosen rounding: 'fix' (toward zero, the default), 'floor', 'ceil' or "
                "'round'. At least one input must be an integer type (two integer inputs must share a class) "
                "and the result has that class. Example: idivide(int32(7), int32(2)) is 3."),
    "vecnorm": (["n = vecnorm(A)", "n = vecnorm(A, p)", "n = vecnorm(A, p, dim)"],
                "p-norm (default 2, the Euclidean length) of each column of A, or of each vector along dim; "
                "a vector input gives a single norm. p = Inf gives the largest magnitude. "
                "Example: vecnorm([3 0; 4 1]) is [5 1]."),
    "nthargout": (["v = nthargout(n, f, x1, ...)", "v = nthargout(n, nout, f, x1, ...)"],
                  "Calls f with the given inputs and returns only its n-th output. A vector n returns several "
                  "outputs as a cell array; nout sets how many outputs f is asked for. "
                  "Example: nthargout(2, @max, [3 9 2]) is 2."),
    "mat2cell": (["C = mat2cell(A, r)", "C = mat2cell(A, r, c)"],
                 "Cuts a 2-D array into blocks and returns them in a cell array. r lists the block heights and "
                 "c the block widths; they must add up to the size of A (c defaults to all columns). "
                 "Example: mat2cell(magic(4), [2 2], [3 1]) gives a 2-by-2 cell."),
    "celldisp": (["celldisp(C)", "celldisp(C, name)"],
                 "Prints the contents of every cell, labelled like C{1} or C{2,3}, descending into nested "
                 "cell arrays. name replaces the label 'C'."),
    "isdiag": ("tf = isdiag(A)", "True if every entry of the numeric matrix A off the main diagonal is zero."),
    "istriu": ("tf = istriu(A)", "True if every entry below the main diagonal of the numeric matrix A is zero."),
    "istril": ("tf = istril(A)", "True if every entry above the main diagonal of the numeric matrix A is zero."),
    "issymmetric": (["tf = issymmetric(A)", "tf = issymmetric(A, 'skew')", "tf = issymmetric(A, tol)"],
                    "True if the square matrix A equals its transpose ('skew': equals minus its transpose). "
                    "With tol, small differences relative to the size of A are allowed."),
    "ishermitian": (["tf = ishermitian(A)", "tf = ishermitian(A, 'skew')", "tf = ishermitian(A, tol)"],
                    "True if the square matrix A equals its conjugate transpose A' ('skew': equals -A'). With "
                    "tol, small differences relative to the size of A are allowed."),
    "validateattributes": ("validateattributes(A, classes, attributes, funcname, varname, argindex)",
                           "Checks an input and throws a descriptive error if it fails. classes is a cell "
                           "array of allowed class names (or {} for any; 'numeric', 'float' and 'integer' "
                           "work as groups). attributes is a cell array of tests such as 'scalar', 'vector', "
                           "'row', 'column', '2d', 'square', 'nonempty', 'real', 'finite', 'nonnan', "
                           "'integer', 'positive', 'nonnegative', 'nonzero', 'binary', 'even', 'odd', "
                           "'increasing', 'nondecreasing', or pairs like '>', 0 / 'numel', 3 / "
                           "'size', [NaN 2] (NaN = any). funcname, varname and argindex are optional and "
                           "only shape the message."),
    "validatestring": (["s = validatestring(str, options)", "s = validatestring(str, options, funcname, ...)"],
                       "Matches str against a list of allowed texts, ignoring case and accepting an "
                       "unambiguous leading part, and returns the full matching option. No match or an "
                       "ambiguous match is an error. Example: validatestring('lin', {'linear', 'cubic'}) "
                       "is 'linear'."),
})

# -- angles, coordinates and special functions ----------------------------------------------------
HELP.update({
    "wrapTo360": ("a = wrapTo360(a)", "Wraps angles in degrees into [0, 360]; positive multiples of 360 map to 360."),
    "wrapTo180": ("a = wrapTo180(a)", "Wraps angles in degrees into [-180, 180]; values already in range are kept."),
    "wrapTo2Pi": ("a = wrapTo2Pi(a)",
                  "Wraps angles in radians into [0, 2*pi]; positive multiples of 2*pi map to 2*pi."),
    "wrapToPi": ("a = wrapToPi(a)", "Wraps angles in radians into [-pi, pi]; values already in range are kept."),
    "cart2pol": (["[theta, rho] = cart2pol(x, y)", "[theta, rho, z] = cart2pol(x, y, z)", "P = cart2pol(C)"],
                 "Converts Cartesian coordinates to polar (or cylindrical): theta is the angle in radians and "
                 "rho the distance from the origin. A single matrix with 2 or 3 columns may be given and, with "
                 "one output, a matrix of the same layout is returned."),
    "pol2cart": (["[x, y] = pol2cart(theta, rho)", "[x, y, z] = pol2cart(theta, rho, z)", "C = pol2cart(P)"],
                 "Converts polar (or cylindrical) coordinates, angle in radians, to Cartesian."),
    "cart2sph": (["[az, el, r] = cart2sph(x, y, z)", "S = cart2sph(C)"],
                 "Converts Cartesian coordinates to spherical: azimuth in the x-y plane, elevation from that "
                 "plane (both in radians) and radius."),
    "sph2cart": (["[x, y, z] = sph2cart(az, el, r)", "C = sph2cart(S)"],
                 "Converts spherical coordinates (azimuth, elevation in radians, radius) to Cartesian."),
    "besselj": (["J = besselj(nu, z)", "J = besselj(nu, z, 1)"],
                "Bessel function of the first kind J_nu(z), element by element. The third argument 1 returns "
                "the exponentially scaled version. Results may be complex for negative z."),
    "bessely": (["Y = bessely(nu, z)", "Y = bessely(nu, z, 1)"],
                "Bessel function of the second kind Y_nu(z). Negative z gives complex results; 1 scales."),
    "besseli": (["I = besseli(nu, z)", "I = besseli(nu, z, 1)"],
                "Modified Bessel function of the first kind I_nu(z); 1 returns the exponentially scaled form."),
    "besselk": (["K = besselk(nu, z)", "K = besselk(nu, z, 1)"],
                "Modified Bessel function of the second kind K_nu(z); 1 returns the exponentially scaled form."),
    "airy": (["W = airy(z)", "W = airy(k, z)", "W = airy(k, z, 1)"],
             "Airy functions: k = 0 gives Ai (default), 1 its derivative, 2 Bi, 3 the derivative of Bi. The "
             "third argument 1 returns the exponentially scaled version."),
    "ellipke": (["K = ellipke(m)", "[K, E] = ellipke(m)"],
                "Complete elliptic integrals of the first (K) and second (E) kind for parameter m in [0, 1]. "
                "A tolerance argument is not supported."),
    "expint": ("y = expint(x)",
               "Exponential integral E1(x), the integral of exp(-t)/t from x to Inf. Negative real x gives "
               "complex results."),
    "gammainc": (["y = gammainc(x, a)", "y = gammainc(x, a, 'upper')"],
                 "Regularized incomplete gamma function: the lower tail P(a, x) by default, or the upper tail "
                 "1 - P with 'upper'. Note that x comes first. a must be non-negative."),
    "betainc": (["y = betainc(x, a, b)", "y = betainc(x, a, b, 'upper')"],
                "Regularized incomplete beta function for x in [0, 1], lower tail by default or upper tail "
                "with 'upper'."),
    "psi": (["y = psi(x)", "y = psi(k, x)"],
            "Digamma function, the derivative of gammaln. psi(k, x) gives the k-th derivative of digamma "
            "(polygamma), for non-negative integer k."),
})

# -- signal processing ----------------------------------------------------------------------------
HELP.update({
    "fft": (["Y = fft(x)", "Y = fft(x, n)", "Y = fft(x, n, dim)"],
            "Fast Fourier transform (discrete Fourier transform) along the first dimension whose size is not "
            "1, or along dim; each column of a matrix is transformed. n zero-pads or truncates to n points. "
            "The result is always complex; abs(Y) is the amplitude spectrum. "
            "Example: fft([1 0 0 0]) is [1 1 1 1]."),
    "ifft": (["x = ifft(Y)", "x = ifft(Y, n)", "x = ifft(Y, n, dim)", "x = ifft(..., 'symmetric')"],
             "Inverse fast Fourier transform. The result is returned as real when its imaginary parts are "
             "negligible; 'symmetric' always returns the real part."),
    "fft2": (["Y = fft2(X)", "Y = fft2(X, m, n)", "Y = fft2(X, [m n])"],
             "Two-dimensional fast Fourier transform of a matrix, optionally padded or cropped to m-by-n."),
    "ifft2": (["X = ifft2(Y)", "X = ifft2(Y, m, n)", "X = ifft2(..., 'symmetric')"],
              "Two-dimensional inverse FFT; real output when the imaginary parts are negligible or with "
              "'symmetric'."),
    "fftshift": (["Y = fftshift(X)", "Y = fftshift(X, dim)"],
                 "Swaps the halves of a spectrum so the zero-frequency term sits in the middle, which suits "
                 "plotting fft output against centred frequencies."),
    "ifftshift": (["X = ifftshift(Y)", "X = ifftshift(Y, dim)"], "Undoes fftshift (they differ for odd lengths)."),
    "nextpow2": ("p = nextpow2(n)",
                 "Smallest integer p with 2^p >= abs(n), element by element; handy for choosing an FFT length. "
                 "Example: nextpow2(100) is 7."),
    "filtfilt": ("y = filtfilt(b, a, x)",
                 "Zero-phase digital filtering: runs the filter forward and then backward, so the output has "
                 "no phase delay and the filter's effect is squared. Works along a row vector or down columns. "
                 "The data must be longer than three times the filter order."),
    "butter": (["[b, a] = butter(n, Wn)", "[b, a] = butter(n, Wn, ftype)", "[z, p, k] = butter(...)",
                "[b, a] = butter(n, Wn, 's')"],
               "Butterworth filter design (maximally flat IIR filter) of order n. Wn is the cutoff as a "
               "fraction of the Nyquist frequency (between 0 and 1); a two-element Wn designs a band-pass "
               "filter. ftype is 'low', 'high', 'bandpass' or 'stop'. 's' designs an analog filter with Wn in "
               "rad/s. Example: [b, a] = butter(4, 0.2) is a 4th-order low-pass filter for use with filter."),
    "cheby1": (["[b, a] = cheby1(n, Rp, Wn)", "[b, a] = cheby1(n, Rp, Wn, ftype)", "[z, p, k] = cheby1(...)"],
               "Chebyshev Type I filter design: order n, Rp decibels of ripple allowed in the passband, and "
               "a sharper roll-off than Butterworth. Wn and ftype work as in butter."),
    "freqz": (["h = freqz(b, a)", "[h, w] = freqz(b, a, n)", "[h, f] = freqz(b, a, n, fs)", "h = freqz(b, a, w)",
               "[h, w] = freqz(..., 'whole')"],
              "Frequency response of a digital filter at n points (default 512) from 0 up to the Nyquist "
              "frequency ('whole' goes around the full circle). w is in radians per sample, or f in Hz when "
              "the sample rate fs is given. A vector of frequencies may be given instead of n. No plot is "
              "drawn; plot abs(h) yourself."),
    "hamming": (["w = hamming(n)", "w = hamming(n, 'periodic')"], "Hamming window of length n as a column vector."),
    "hann": (["w = hann(n)", "w = hann(n, 'periodic')"],
             "Hann (raised-cosine) window of length n as a column vector; the end points are zero."),
    "hanning": (["w = hanning(n)", "w = hanning(n, 'periodic')"],
                "Hann window variant without the zero end points, as a column vector."),
    "blackman": (["w = blackman(n)", "w = blackman(n, 'periodic')"], "Blackman window of length n as a column vector."),
    "rectwin": ("w = rectwin(n)", "Rectangular window: a column of n ones."),
    "kaiser": (["w = kaiser(n)", "w = kaiser(n, beta)"],
               "Kaiser window of length n as a column; larger beta (default 0.5) gives lower side lobes."),
    "xcorr": (["r = xcorr(x)", "r = xcorr(x, y)", "r = xcorr(..., maxlag)", "r = xcorr(..., scaleopt)",
               "[r, lags] = xcorr(...)"],
              "Cross-correlation of two vectors (autocorrelation with one input) at every lag from -(N-1) to "
              "N-1, or only up to maxlag. scaleopt is 'none' (default), 'biased', 'unbiased' or 'coeff' "
              "(peak 1 at lag 0 for autocorrelation). A shorter vector is zero-padded. Only vectors are "
              "supported."),
    "periodogram": (["pxx = periodogram(x)", "pxx = periodogram(x, window)", "pxx = periodogram(x, window, nfft)",
                     "[pxx, f] = periodogram(x, window, nfft, fs)", "[pxx, f] = periodogram(..., range)"],
                    "Power spectral density estimate of a signal vector from its squared FFT magnitude. The "
                    "window defaults to rectangular, nfft to max(256, next power of 2). For real signals the "
                    "result is one-sided unless range is 'twosided' or 'centered'. f is in rad/sample, or Hz "
                    "when fs is given. No plot is drawn."),
    "pwelch": (["pxx = pwelch(x)", "pxx = pwelch(x, window)", "pxx = pwelch(x, window, noverlap)",
                "pxx = pwelch(x, window, noverlap, nfft)", "[pxx, f] = pwelch(x, window, noverlap, nfft, fs)"],
               "Welch power spectral density estimate: averages windowed periodograms of overlapping "
               "segments, giving a smoother spectrum than periodogram. Defaults: Hamming window of about "
               "length/4.5 samples, 50% overlap, nfft max(256, next power of 2). window may be a length or a "
               "vector. One-sided for real signals; no plot is drawn."),
    "findpeaks": (["pks = findpeaks(y)", "[pks, locs] = findpeaks(y)", "[pks, locs] = findpeaks(y, x)",
                   "[...] = findpeaks(..., Name, Value)"],
                  "Finds local maxima (peaks) of a vector: points higher than both neighbours, with a flat "
                  "top reported at its first sample; the end points are never peaks. locs are indices, or "
                  "positions from x (a scalar x is a sample rate). Options: MinPeakHeight, MinPeakProminence, "
                  "MinPeakDistance, Threshold, SortStr ('ascend'/'descend') and NPeaks. Widths, prominences "
                  "and plotting are not supported. Example: findpeaks([1 3 2 5 1]) is [3 5]."),
    "resample": (["y = resample(x, p, q)", "y = resample(x, p, q, n)", "y = resample(x, p, q, n, beta)"],
                 "Changes the sample rate by the factor p/q, with an anti-aliasing low-pass FIR filter "
                 "(Kaiser window; n sets its length, beta its shape). Works along a row vector or down columns."),
    "decimate": (["y = decimate(x, r)", "y = decimate(x, r, n)", "y = decimate(x, r, 'fir')"],
                 "Reduces the sample rate by the integer factor r after low-pass filtering: by default an "
                 "8th-order Chebyshev Type I filter applied forward and backward (zero phase), or a 30-tap "
                 "FIR filter with 'fir'. n changes the filter order."),
    "interpft": (["y = interpft(x, n)", "y = interpft(x, n, dim)"],
                 "Resamples periodic data to n points by zero-padding (or trimming) its Fourier transform."),
    "sinc": ("y = sinc(x)", "Normalized sinc function sin(pi*x)./(pi*x), with sinc(0) = 1."),
    "sawtooth": (["y = sawtooth(t)", "y = sawtooth(t, width)"],
                 "Sawtooth wave with period 2*pi running between -1 and 1. width (0..1, default 1) sets where "
                 "in the period the peak is; 0.5 gives a triangle wave."),
    "square": (["y = square(t)", "y = square(t, duty)"],
               "Square wave with period 2*pi taking the values +1 and -1; duty is the percentage of each "
               "period spent at +1 (default 50)."),
    "chirp": (["y = chirp(t, f0, t1, f1)", "y = chirp(t, f0, t1, f1, method)", "y = chirp(t, f0, t1, f1, method, phi)"],
              "Swept-frequency cosine whose frequency moves from f0 at time 0 to f1 at time t1 (defaults 0, "
              "1 and 100 Hz). method is 'linear' (default), 'quadratic' or 'logarithmic'; phi is the initial "
              "phase in degrees."),
    "medfilt1": (["y = medfilt1(x)", "y = medfilt1(x, n)", "y = medfilt1(x, n, [], dim)", "y = medfilt1(..., padding)",
                  "y = medfilt1(..., nanflag)"],
                 "One-dimensional median filter of order n (default 3): each output is the median of a window "
                 "around the sample, which removes spikes while keeping edges. padding is 'zeropad' (default) "
                 "or 'truncate'; 'omitnan' ignores NaN."),
    "sgolayfilt": (["y = sgolayfilt(x, order, framelen)", "y = sgolayfilt(x, order, framelen, [], dim)"],
                   "Savitzky-Golay smoothing filter: fits a polynomial of the given order to each window of "
                   "framelen samples (odd, larger than order). Weights are not supported."),
    "detrend": (["y = detrend(x)", "y = detrend(x, n)", "y = detrend(x, 'constant')", "y = detrend(x, 'linear')"],
                "Removes the best-fit straight line (default), mean ('constant') or polynomial of degree n from "
                "a vector or from each column of a matrix. Breakpoints are not supported."),
    "unwrap": (["q = unwrap(p)", "q = unwrap(p, tol)", "q = unwrap(p, tol, dim)"],
               "Removes jumps larger than tol (default pi) from phase angles in radians by adding multiples of "
               "2*pi, giving a continuous phase curve."),
    "db2mag": ("m = db2mag(d)", "Converts decibels to amplitude ratio: 10.^(d/20)."),
    "mag2db": ("d = mag2db(m)", "Converts amplitude ratio to decibels: 20*log10(m). Negative values give NaN."),
    "db2pow": ("p = db2pow(d)", "Converts decibels to power ratio: 10.^(d/10)."),
    "pow2db": ("d = pow2db(p)", "Converts power ratio to decibels: 10*log10(p). Negative values are an error."),
    "hilbert": (["z = hilbert(x)", "z = hilbert(x, n)"],
                "Analytic signal of real data: the real part is x and the imaginary part its Hilbert "
                "transform, so abs(z) is the instantaneous amplitude. Works along a row vector or down "
                "columns; any imaginary part of x is ignored."),
    "envelope": (["up = envelope(x)", "[up, lo] = envelope(x)"],
                 "Upper and lower envelopes of a signal from the magnitude of its analytic signal, centred on "
                 "the mean. Only the 'analytic' method is supported; a window length argument is ignored."),
})

# -- probability distributions --------------------------------------------------------------------
_UPPER = "Add 'upper' as the last argument to get the upper tail 1 - P."
HELP.update({
    "normpdf": (["y = normpdf(x)", "y = normpdf(x, mu, sigma)"],
                "Normal (Gaussian) probability density at x, with mean mu (default 0) and standard deviation "
                "sigma (default 1). Inputs expand against scalars. sigma <= 0 gives NaN."),
    "normcdf": (["p = normcdf(x)", "p = normcdf(x, mu, sigma)", "p = normcdf(..., 'upper')"],
                "Normal cumulative distribution function: the probability that a value is at most x. Defaults "
                "mu = 0, sigma = 1. 'upper' gives the accurate upper tail. Example: normcdf(0) is 0.5."),
    "norminv": (["x = norminv(p)", "x = norminv(p, mu, sigma)"],
                "Inverse of normcdf (normal quantile). Example: norminv(0.975) is about 1.96."),
    "normrnd": (["r = normrnd(mu, sigma)", "r = normrnd(mu, sigma, m, n, ...)", "r = normrnd(mu, sigma, [m n])"],
                "Normally distributed random numbers with the given mean and standard deviation, shaped like "
                "the parameters or by the size arguments. Uses the generator controlled by rng."),
    "unifpdf": (["y = unifpdf(x)", "y = unifpdf(x, a, b)"],
                "Density of the continuous uniform distribution on [a, b] (default [0, 1])."),
    "unifcdf": (["p = unifcdf(x)", "p = unifcdf(x, a, b)"],
                f"Cumulative probability of the uniform distribution on [a, b] (default [0, 1]). {_UPPER}"),
    "unifrnd": (["r = unifrnd(a, b)", "r = unifrnd(a, b, m, n, ...)"], "Uniform random numbers on [a, b]."),
    "exppdf": (["y = exppdf(x)", "y = exppdf(x, mu)"],
               "Exponential density with mean mu (default 1); note mu is the mean, not the rate."),
    "expcdf": (["p = expcdf(x)", "p = expcdf(x, mu)"],
               f"Exponential cumulative probability with mean mu (default 1). {_UPPER}"),
    "exprnd": (["r = exprnd(mu)", "r = exprnd(mu, m, n, ...)"], "Exponential random numbers with mean mu."),
    "binopdf": ("y = binopdf(x, n, p)",
                "Binomial probability of exactly x successes in n trials with success probability p. "
                "Example: binopdf(1, 2, 0.5) is 0.5."),
    "binocdf": ("y = binocdf(x, n, p)", f"Binomial probability of at most x successes in n trials. {_UPPER}"),
    "binornd": (["r = binornd(n, p)", "r = binornd(n, p, m, k, ...)"],
                "Binomial random counts for n trials with probability p."),
    "poisspdf": ("y = poisspdf(x, lambda)", "Poisson probability of exactly x events when the mean count is lambda."),
    "poisscdf": ("y = poisscdf(x, lambda)", f"Poisson probability of at most x events with mean lambda. {_UPPER}"),
    "poissrnd": (["r = poissrnd(lambda)", "r = poissrnd(lambda, m, n, ...)"],
                 "Poisson random counts with mean lambda."),
    "tpdf": ("y = tpdf(x, nu)", "Student's t density with nu degrees of freedom."),
    "tcdf": ("p = tcdf(x, nu)", f"Student's t cumulative probability with nu degrees of freedom. {_UPPER}"),
    "tinv": ("x = tinv(p, nu)",
             "Inverse of tcdf (t quantile), used for confidence intervals. Example: tinv(0.975, 10) is about 2.228."),
    "chi2pdf": ("y = chi2pdf(x, nu)", "Chi-square density with nu degrees of freedom."),
    "chi2cdf": ("p = chi2cdf(x, nu)", f"Chi-square cumulative probability with nu degrees of freedom. {_UPPER}"),
    "chi2inv": ("x = chi2inv(p, nu)", "Inverse of chi2cdf (chi-square quantile)."),
    "fpdf": ("y = fpdf(x, nu1, nu2)", "F distribution density with nu1 and nu2 degrees of freedom."),
    "fcdf": ("p = fcdf(x, nu1, nu2)", f"F distribution cumulative probability. {_UPPER}"),
    "finv": ("x = finv(p, nu1, nu2)", "Inverse of fcdf (F quantile)."),
    "gampdf": (["y = gampdf(x, a)", "y = gampdf(x, a, b)"], "Gamma density with shape a and scale b (default 1)."),
    "gamcdf": (["p = gamcdf(x, a)", "p = gamcdf(x, a, b)"],
               f"Gamma cumulative probability with shape a and scale b (default 1). {_UPPER}"),
    "betapdf": ("y = betapdf(x, a, b)", "Beta distribution density on [0, 1] with shape parameters a and b."),
    "betacdf": ("p = betacdf(x, a, b)", f"Beta distribution cumulative probability. {_UPPER}"),
    "randsample": (["y = randsample(n, k)", "y = randsample(population, k)", "y = randsample(..., replace)",
                    "y = randsample(n, k, true, w)"],
                   "Random sample of k values from 1..n or from the elements of a population vector, without "
                   "replacement unless replace is true. Weights w make the draw weighted (and with replacement "
                   "unless replace is given as false). Uses the generator controlled by rng."),
})

# -- descriptive statistics -----------------------------------------------------------------------
HELP.update({
    "prctile": (["q = prctile(x, p)", "q = prctile(x, p, dim)"],
                "Percentiles of the data for percentages p (0..100), along the first dimension whose size is "
                "not 1 or along dim; NaN values are ignored. Sorted values are treated as sitting at "
                "percentages 100*(i-0.5)/n with linear interpolation between them. "
                "Example: prctile([1 2 3 4], 50) is 2.5."),
    "quantile": (["q = quantile(x, p)", "q = quantile(x, p, dim)"],
                 "Quantiles for probabilities p (0..1), computed the same way as prctile."),
    "iqr": (["r = iqr(x)", "r = iqr(x, dim)"],
            "Interquartile range: the 75th minus the 25th percentile, a spread measure robust to outliers."),
    "mad": (["y = mad(x)", "y = mad(x, flag)", "y = mad(x, flag, dim)"],
            "Mean absolute deviation from the mean (flag 0, default) or median absolute deviation from the "
            "median (flag 1). NaN values are ignored."),
    "zscore": (["z = zscore(x)", "z = zscore(x, flag)", "z = zscore(x, flag, dim)", "[z, mu, sigma] = zscore(...)"],
               "Standardizes data: subtracts the mean and divides by the standard deviation (N-1 normalization, "
               "or N with flag 1) along the first dimension whose size is not 1. Constant data gives zeros."),
    "skewness": (["y = skewness(x)", "y = skewness(x, flag)", "y = skewness(x, flag, dim)"],
                 "Sample skewness, a measure of asymmetry (0 for symmetric data). flag 1 (default) gives the "
                 "plain moment estimate, flag 0 the bias-corrected one. NaN values are ignored."),
    "kurtosis": (["k = kurtosis(x)", "k = kurtosis(x, flag)", "k = kurtosis(x, flag, dim)"],
                 "Sample kurtosis, a measure of how heavy the tails are; a normal distribution gives 3 (this is "
                 "not the excess kurtosis). flag 0 applies the bias correction. NaN values are ignored."),
    "cov": (["C = cov(x)", "C = cov(X)", "C = cov(x, y)", "C = cov(..., w)", "C = cov(..., 'omitrows')"],
            "Covariance. For a vector, the variance; for a matrix whose columns are variables, the covariance "
            "matrix; for two vectors, their 2-by-2 covariance matrix. w = 1 normalizes by N instead of N-1. "
            "'omitrows' drops rows containing NaN."),
    "corrcoef": (["R = corrcoef(X)", "R = corrcoef(x, y)", "[R, P] = corrcoef(...)"],
                 "Pearson correlation coefficients between the columns of X (or between x and y), with ones on "
                 "the diagonal. P holds p-values for testing zero correlation. Name-value options are ignored."),
    "corr": (["rho = corr(X)", "rho = corr(X, Y)", "rho = corr(..., 'Type', 'Spearman')", "[rho, pval] = corr(...)"],
             "Linear (Pearson, default) or rank (Spearman) correlation between the columns of X, or between "
             "each column of X and each column of Y. Kendall's tau is not supported."),
    "nanmean": (["m = nanmean(x)", "m = nanmean(x, dim)"],
                "Mean that ignores NaN values; same as mean(..., 'omitnan')."),
    "nansum": (["s = nansum(x)", "s = nansum(x, dim)"], "Sum that ignores NaN values; same as sum(..., 'omitnan')."),
    "nanstd": (["s = nanstd(x)", "s = nanstd(x, flag)", "s = nanstd(x, flag, dim)"],
               "Standard deviation that ignores NaN values; flag 1 normalizes by N instead of N-1."),
    "nanmax": (["m = nanmax(x)", "[m, i] = nanmax(x)", "m = nanmax(x, [], dim)", "m = nanmax(a, b)"],
               "Maximum that ignores NaN values, like max."),
    "nanmin": (["m = nanmin(x)", "[m, i] = nanmin(x)", "m = nanmin(x, [], dim)", "m = nanmin(a, b)"],
               "Minimum that ignores NaN values, like min."),
    "histc": (["n = histc(x, edges)", "n = histc(x, edges, dim)", "[n, bin] = histc(...)"],
              "Counts values falling in each interval edges(k) <= x < edges(k+1); the last count is for values "
              "equal to the last edge. bin gives each value's interval (0 if outside). Older function; "
              "histcounts is usually more convenient."),
    "tabulate": (["tabulate(x)", "T = tabulate(x)"],
                 "Frequency table of the values in x with columns Value, Count and Percent. For positive "
                 "integers every value from 1 to max(x) is listed; text in a cell array gives a cell result. "
                 "Without an output the table is printed."),
    "movmean": (["M = movmean(x, k)", "M = movmean(x, [kb kf])", "M = movmean(..., dim)", "M = movmean(..., 'omitnan')",
                 "M = movmean(..., 'Endpoints', e)"],
                "Moving (rolling, sliding-window) average over windows of k samples centred on each point "
                "(for even k, one more sample before than after), or kb before and kf after. Near the ends "
                "the window shrinks by default; 'Endpoints' can be 'discard', 'fill' (NaN) or a number. "
                "SamplePoints are not supported. Example: movmean([1 2 3 4 5], 3) is [1.5 2 3 4 4.5]."),
    "movsum": (["M = movsum(x, k)", "M = movsum(x, [kb kf])", "M = movsum(..., dim)", "M = movsum(..., 'omitnan')"],
               "Moving sum over a sliding window; windows and options work as in movmean."),
    "movmedian": (["M = movmedian(x, k)", "M = movmedian(x, [kb kf])", "M = movmedian(..., dim)"],
                  "Moving median over a sliding window, useful for removing spikes; options as in movmean."),
    "movmax": (["M = movmax(x, k)", "M = movmax(x, [kb kf])", "M = movmax(..., dim)"],
               "Moving maximum over a sliding window; options as in movmean."),
    "movmin": (["M = movmin(x, k)", "M = movmin(x, [kb kf])", "M = movmin(..., dim)"],
               "Moving minimum over a sliding window; options as in movmean."),
    "movstd": (["M = movstd(x, k)", "M = movstd(x, k, w)", "M = movstd(x, k, w, dim)"],
               "Moving standard deviation over a sliding window; w = 1 normalizes by the window size instead "
               "of size-1. Other options as in movmean."),
    "movvar": (["M = movvar(x, k)", "M = movvar(x, k, w)", "M = movvar(x, k, w, dim)"],
               "Moving variance over a sliding window; options as in movstd."),
    "smoothdata": (["y = smoothdata(x)", "y = smoothdata(x, method)", "y = smoothdata(x, method, window)",
                    "y = smoothdata(x, dim, ...)", "y = smoothdata(..., 'Degree', d)"],
                   "Smooths noisy data along the first dimension whose size is not 1. method is 'movmean' "
                   "(default), 'movmedian', 'gaussian' or 'sgolay' (Savitzky-Golay with polynomial degree d, "
                   "default 2); the lowess/loess methods are not supported. Without a window, about a tenth "
                   "of the data length is used, which may differ from other implementations. NaN values are "
                   "ignored by default."),
    "normalize": (["y = normalize(x)", "y = normalize(x, dim)", "y = normalize(..., method)",
                   "y = normalize(..., method, option)"],
                  "Rescales data along the first dimension whose size is not 1. method is 'zscore' (default; "
                  "option 'std' or 'robust'), 'range' (to [0 1] or a given [a b]), 'norm' (divide by the "
                  "p-norm, default 2), 'center' (subtract the mean, the median or a value) or 'scale' (divide "
                  "by the std, 'mad', 'first', 'iqr' or a value)."),
    "rescale": (["y = rescale(x)", "y = rescale(x, l, u)", "y = rescale(..., 'InputMin', a, 'InputMax', b)"],
                "Linearly maps the smallest element of x to l (default 0) and the largest to u (default 1), "
                "using all elements. InputMin and InputMax set the source range, clipping values outside it. "
                "Constant data maps to l."),
    "pdist": (["d = pdist(X)", "d = pdist(X, metric)", "d = pdist(X, 'minkowski', p)"],
              "Pairwise distances between the rows of X, returned as a row vector (use squareform to get a "
              "matrix). metric: 'euclidean' (default), 'squaredeuclidean', 'seuclidean', 'cityblock', "
              "'chebychev', 'minkowski', 'cosine', 'correlation', 'spearman', 'hamming' or 'jaccard'."),
    "squareform": (["D = squareform(d)", "d = squareform(D)", "Y = squareform(X, 'tovector')",
                    "Y = squareform(X, 'tomatrix')"],
                   "Converts between the vector of pairwise distances returned by pdist and the full symmetric "
                   "distance matrix with zeros on the diagonal."),
    "kmeans": (["idx = kmeans(X, k)", "[idx, C, sumd, D] = kmeans(X, k)", "[...] = kmeans(..., Name, Value)"],
               "k-means clustering of the rows of X into k groups. idx gives each row's cluster number, C "
               "the cluster centres, sumd the within-cluster sums of distances and D the distance from every "
               "point to every centre. Options: 'Distance' ('sqeuclidean' default or 'cityblock'), "
               "'Replicates' (keep the best of several runs), 'MaxIter' (default 100) and 'Start' ('plus' "
               "k-means++ default, 'sample', 'uniform' or a k-by-p matrix). Rows with NaN get idx NaN. "
               "Results are random; set rng for repeatable runs."),
})

# -- hypothesis tests and regression ----------------------------------------------------------------
HELP.update({
    "ttest": (["h = ttest(x)", "h = ttest(x, m)", "h = ttest(x, y)", "[h, p, ci, stats] = ttest(..., Name, Value)"],
              "One-sample t-test that the data have mean m (default 0), or paired t-test on x - y when y has "
              "the same size. h = 1 rejects the hypothesis at the significance level 'Alpha' (default 0.05). "
              "'Tail' is 'both' (default), 'right' or 'left'; 'Dim' picks the dimension. ci is the confidence "
              "interval for the mean and stats holds tstat, df and sd. NaN values are ignored."),
    "ttest2": (["h = ttest2(x, y)", "[h, p, ci, stats] = ttest2(x, y, Name, Value)"],
               "Two-sample t-test that x and y come from populations with equal means. Assumes equal variances "
               "unless 'Vartype' is 'unequal' (Welch's test). 'Alpha', 'Tail' and 'Dim' work as in ttest; ci "
               "is the interval for the difference of means."),
    "ranksum": (["p = ranksum(x, y)", "[p, h, stats] = ranksum(x, y, Name, Value)"],
                "Wilcoxon rank sum test (equivalent to the Mann-Whitney U test), a non-parametric test that "
                "two samples have equal medians. Two-sided only. Uses the exact distribution for small samples "
                "without ties and a normal approximation otherwise ('method' 'exact' or 'approximate' "
                "overrides). 'alpha' sets the level for h."),
    "anova1": (["p = anova1(y)", "p = anova1(y, group)", "[p, tbl, stats] = anova1(...)"],
               "One-way analysis of variance: tests whether several groups share the same mean. Groups are the "
               "columns of y, or are given by labels in group (numbers, a char matrix or a cell array). tbl is "
               "the ANOVA table as a cell array; stats holds the group names, sizes and means. No figure is "
               "shown."),
    "regress": (["b = regress(y, X)", "[b, bint, r, rint, stats] = regress(y, X)", "[...] = regress(y, X, alpha)"],
                "Multiple linear regression by least squares: finds b so that X*b best fits y. Include a "
                "column of ones in X to fit an intercept. bint gives 100*(1-alpha)% confidence intervals for b, "
                "r the residuals, rint residual intervals and stats = [R^2, F, p-value, error variance]. Rows "
                "containing NaN are ignored. Example: b = regress(y, [ones(size(x)) x]) fits a line."),
})

# -- optimization (linear, quadratic, nonlinear programming, least squares) --------------------
_LP_ARGS = "Pass [] for constraints that are not needed."
HELP.update({
    "linprog": (["x = linprog(f, A, b)", "x = linprog(f, A, b, Aeq, beq)", "x = linprog(f, A, b, Aeq, beq, lb, ub)",
                 "[x, fval, exitflag, output] = linprog(...)"],
                "Linear programming: minimizes f'*x subject to A*x <= b, Aeq*x = beq and lb <= x <= ub, using "
                f"the HiGHS solver. {_LP_ARGS} exitflag is 1 when solved, 0 at the iteration limit, -2 when "
                "infeasible and -3 when unbounded (then x is []). An options argument is ignored."),
    "intlinprog": (["x = intlinprog(f, intcon, A, b)", "x = intlinprog(f, intcon, A, b, Aeq, beq, lb, ub)",
                    "[x, fval, exitflag] = intlinprog(...)"],
                   "Mixed-integer linear programming: like linprog, but the variables listed in intcon must "
                   f"take integer values. {_LP_ARGS} Options are ignored."),
    "quadprog": (["x = quadprog(H, f)", "x = quadprog(H, f, A, b)", "x = quadprog(H, f, A, b, Aeq, beq, lb, ub)",
                  "x = quadprog(H, f, A, b, Aeq, beq, lb, ub, x0)", "[x, fval, exitflag] = quadprog(...)"],
                 "Quadratic programming: minimizes 0.5*x'*H*x + f'*x subject to linear constraints and bounds. "
                 "Without constraints and with H positive definite the minimum is solved directly; otherwise "
                 "a sequential quadratic programming method is used, which finds a local minimum. "
                 "Options are ignored."),
    "fmincon": (["x = fmincon(fun, x0, A, b)", "x = fmincon(fun, x0, A, b, Aeq, beq, lb, ub)",
                 "x = fmincon(fun, x0, A, b, Aeq, beq, lb, ub, nonlcon)", "x = fmincon(..., nonlcon, options)",
                 "[x, fval, exitflag] = fmincon(...)"],
                "Constrained nonlinear optimization: minimizes fun(x) from the start x0 subject to linear "
                "constraints, bounds and nonlinear constraints, where nonlcon returns [c, ceq] with c <= 0 and "
                "ceq = 0 required. Uses sequential quadratic programming with numerical gradients and finds a "
                "local minimum. Only the MaxIterations (or MaxIter) option is read. exitflag is 1 on success, "
                "-2 if the result is infeasible."),
    "lsqnonneg": (["x = lsqnonneg(C, d)", "[x, resnorm, residual, exitflag] = lsqnonneg(C, d)"],
                  "Non-negative least squares: minimizes norm(C*x - d) subject to x >= 0. resnorm is the "
                  "squared residual norm."),
    "lsqcurvefit": (["x = lsqcurvefit(fun, x0, xdata, ydata)", "x = lsqcurvefit(fun, x0, xdata, ydata, lb, ub)",
                     "x = lsqcurvefit(..., lb, ub, options)", "[x, resnorm, residual, exitflag] = lsqcurvefit(...)"],
                    "Nonlinear curve fitting by least squares: finds parameters x so that fun(x, xdata) matches "
                    "ydata as closely as possible, starting from x0. Uses Levenberg-Marquardt, or a "
                    "trust-region method when bounds are given. Options StepTolerance (TolX) and "
                    "FunctionTolerance (TolFun) are read. "
                    "Example: lsqcurvefit(@(p, t) p(1)*exp(p(2)*t), [1 -1], t, y)."),
    "lsqnonlin": (["x = lsqnonlin(fun, x0)", "x = lsqnonlin(fun, x0, lb, ub)",
                   "x = lsqnonlin(fun, x0, lb, ub, options)",
                   "[x, resnorm, residual, exitflag] = lsqnonlin(...)"],
                  "Nonlinear least squares: minimizes the sum of squares of the residual vector returned by "
                  "fun(x), starting from x0, optionally within bounds. Methods and options as in lsqcurvefit."),
})

# -- validators and object utilities ------------------------------------------------------------
_VALIDATORS = {
    "mustBePositive": "every element is greater than zero",
    "mustBeNonnegative": "every element is zero or greater",
    "mustBeNegative": "every element is less than zero",
    "mustBeNonpositive": "every element is zero or less",
    "mustBeNonzero": "no element is zero",
    "mustBeInteger": "every element is a finite whole number",
    "mustBeFinite": "no element is Inf or NaN",
    "mustBeNonNan": "no element is NaN",
    "mustBeReal": "the value has no imaginary part",
    "mustBeNumeric": "the value is a numeric array",
    "mustBeNumericOrLogical": "the value is numeric or logical",
    "mustBeNonempty": "the value is not empty",
    "mustBeVector": "the value is a row or column vector",
    "mustBeText": "the value is a char vector, string, or cell array of char vectors",
    "mustBeTextScalar": "the value is one piece of text (char row vector or string scalar)",
    "mustBeNonzeroLengthText": "the value is text with at least one character",
    "mustBeFile": "the value names an existing file",
    "mustBeFolder": "the value names an existing folder",
}
for _name, _rule in _VALIDATORS.items():
    HELP[_name] = (f"{_name}(value)",
                   f"Validation function: does nothing if {_rule}, and raises an error otherwise. "
                   "Useful in classdef property declarations, e.g. r (1,1) double {" + _name + "}.")
for _name, _rule in {"mustBeGreaterThan": "greater than c", "mustBeGreaterThanOrEqual": "at least c",
                     "mustBeLessThan": "less than c", "mustBeLessThanOrEqual": "at most c"}.items():
    HELP[_name] = (f"{_name}(value, c)",
                   f"Validation function: raises an error unless every element of value is {_rule}.")
HELP.update({
    "mustBeInRange": ("mustBeInRange(value, lower, upper)",
                      "Validation function: raises an error unless every element lies between lower and upper "
                      "(inclusive)."),
    "mustBeMember": ("mustBeMember(value, allowed)",
                     "Validation function: raises an error unless value (number, text or cell of text) is one of "
                     "the allowed values."),
    "mustBeA": ("mustBeA(value, classNames)",
                "Validation function: raises an error unless value is of one of the named classes (a class "
                "name, or a cell/string array of names; subclasses count)."),
    "isvalid": ("tf = isvalid(h)",
                "True for a handle object (classdef handle class, containers.Map, ...) that has not been "
                "deleted with delete(h)."),
    "properties": (["properties(obj)", "names = properties(obj)", "properties('ClassName')"],
                   "Lists the public properties of an object or class; with an output, returns them as a cell "
                   "column."),
    "methods": (["methods(obj)", "names = methods(obj)", "methods('ClassName')"],
                "Lists the public methods of an object or class; with an output, returns them as a cell column."),
    "isprop": ("tf = isprop(obj, name)", "True if obj has a public property called name."),
    "ismethod": ("tf = ismethod(obj, name)", "True if obj has a public method called name."),
})
HELP["delete"] = (["delete(filename)", "delete(h)"],
                  "Deletes the named file(s), or, for a handle object, runs the class's delete method (if any) "
                  "and marks the object invalid so later use raises an error.")
# -- graphics handles, layouts and extra plot types ---------------------------------------------
HELP.update({
    "get": (["v = get(h, name)", "c = get(h, {name1, name2})", "s = get(h)"],
            "Reads a property of a graphics handle h (from gcf, gca, plot, title, ...). Property names "
            "ignore case. When h holds several handles (e.g. all lines returned by one plot call) the "
            "result is a cell array with one row per handle. get(h) returns a struct of the common "
            "properties. A figure number or 0 (the graphics root) also works as h. Dot syntax, e.g. "
            "h.LineWidth, reads the same properties."),
    "set": (["set(h, name, value, ...)", "set(h, s)"],
            "Changes properties of graphics handles, e.g. set(h, 'LineWidth', 2, 'Color', 'r'). If h holds "
            "several handles each one is changed. A struct s of property values is also accepted. set(0, "
            "'DefaultLineLineWidth', w) and a few other root defaults change later plots. Dot assignment, "
            "e.g. h.LineWidth = 2, does the same for one property; on an array of handles it changes all "
            "of them, and reading a property from an array gives a cell array."),
    "groot": ("r = groot", "Handle to the graphics root, the parent of all figures (the same as handle 0). "
                           "Its properties include CurrentFigure and ScreenSize."),
    "isgraphics": (["tf = isgraphics(h)", "tf = isgraphics(h, type)"],
                   "True for each element of h that is a valid (not deleted or closed) graphics handle, "
                   "optionally of the given type such as 'line' or 'figure'. Figure numbers count as handles "
                   "of open figures. ishandle and ishghandle are the same."),
    "ishandle": ("tf = ishandle(h)", "Same as isgraphics: true for valid graphics handles."),
    "ishghandle": ("tf = ishghandle(h)", "Same as isgraphics: true for valid graphics handles."),
    "line": (["line(x, y)", "line(x, y, z)", "line('XData', x, 'YData', y, ...)", "h = line(...)"],
             "Draws a line in the current axes without clearing it, whatever the hold state. A matrix "
             "draws one line per column. Accepts the same Name/Value options as plot."),
    "fplot": (["fplot(f)", "fplot(f, [a b])", "fplot(f, [a b], linespec)", "fplot(fx, fy, [a b])",
               "h = fplot(...)"],
              "Plots the function handle f over the interval [a b] (default [-5 5]) by sampling it at many "
              "points; f is called with a vector and, if that fails, one point at a time. With two handles "
              "draws the parametric curve (fx(t), fy(t))."),
    "polarplot": (["polarplot(theta, rho)", "polarplot(theta, rho, linespec)", "polarplot(rho)",
                   "h = polarplot(...)"],
                  "Plots points given by angle theta (radians) and radius rho on polar axes."),
    "quiver": (["quiver(U, V)", "quiver(X, Y, U, V)", "quiver(..., scale)", "quiver(..., linespec)",
                "h = quiver(...)"],
               "Draws arrows with components (U, V) starting at the points (X, Y), or on a grid of indices. "
               "Arrows are scaled automatically to fit between grid points; scale multiplies that factor and "
               "0 turns scaling off ('AutoScale', 'off' does the same)."),
    "scatter3": (["scatter3(x, y, z)", "scatter3(x, y, z, sz, c)", "scatter3(..., 'filled')"],
                 "Draws a marker at each 3-D point (x, y, z) on 3-D axes."),
    "tiledlayout": (["tiledlayout(m, n)", "tiledlayout('flow')",
                     "t = tiledlayout(..., 'TileSpacing', s, 'Padding', p)"],
                    "Starts a grid of axes in the current figure (clearing it). An m-by-n layout has fixed "
                    "tiles; 'flow' grows and rearranges the grid as tiles are added ('vertical' and "
                    "'horizontal' stack them). Use nexttile to create each axes. title(t, txt), xlabel(t, "
                    "txt) and ylabel(t, txt) label the whole layout. TileSpacing and Padding accept 'loose', "
                    "'compact', 'tight' or 'none'."),
    "nexttile": (["nexttile", "nexttile(k)", "nexttile([r c])", "nexttile(k, [r c])", "ax = nexttile(...)"],
                 "Creates the next axes of the current tiled layout and makes it current; nexttile(k) "
                 "goes to tile k (counting along rows) and [r c] makes an axes spanning r rows and c "
                 "columns. Without a layout a 'flow' layout is started."),
    "shading": (["shading flat", "shading interp", "shading faceted"],
                "Sets how surfaces are drawn: 'faceted' (the default) outlines each patch, 'flat' and "
                "'interp' hide the grid lines. Applies to the surfaces in the current axes and to later "
                "surf calls. mylab draws 'interp' like 'flat' with smoothing."),
    "peaks": (["Z = peaks", "Z = peaks(n)", "Z = peaks(v)", "Z = peaks(X, Y)", "[X, Y, Z] = peaks(...)",
               "peaks"],
              "A smooth test surface with several hills and dips, given by a sum of Gaussian bumps. The "
              "default grid is 49-by-49 points on [-3, 3]; peaks(n) uses n points, a vector v builds the "
              "grid from v, and X, Y give the points directly. Without outputs it draws the surface."),
    "parula": (["c = parula", "c = parula(n)"],
               "n-by-3 matrix of RGB colours (default 256) running from dark blue through green to yellow. "
               "mylab uses matplotlib's perceptually uniform 'viridis' map for parula, also the default "
               "colormap. Use with colormap, e.g. colormap(parula(16))."),
    "jet": (["c = jet", "c = jet(n)"], "n-by-3 rainbow colormap (default 256 rows) from dark blue to dark red."),
    "gray": (["c = gray", "c = gray(n)"], "n-by-3 colormap of greys from black to white (default 256 rows)."),
    "hot": (["c = hot", "c = hot(n)"], "n-by-3 colormap from black through red and yellow to white."),
    "cool": (["c = cool", "c = cool(n)"], "n-by-3 colormap from cyan to magenta."),
    "hsv": (["c = hsv", "c = hsv(n)"], "n-by-3 colormap cycling through the hues of the colour wheel."),
    "bone": (["c = bone", "c = bone(n)"], "n-by-3 greyscale colormap with a slight blue tint."),
    "copper": (["c = copper", "c = copper(n)"], "n-by-3 colormap from black to a light copper colour."),
    "spring": (["c = spring", "c = spring(n)"], "n-by-3 colormap from magenta to yellow."),
    "summer": (["c = summer", "c = summer(n)"], "n-by-3 colormap from green to yellow."),
    "autumn": (["c = autumn", "c = autumn(n)"], "n-by-3 colormap from red through orange to yellow."),
    "winter": (["c = winter", "c = winter(n)"], "n-by-3 colormap from blue to green."),
    "turbo": (["c = turbo", "c = turbo(n)"], "n-by-3 rainbow colormap with smoother brightness than jet."),
    "pink": (["c = pink", "c = pink(n)"], "n-by-3 colormap of pastel pinks, from dark to white."),
    "lines": (["c = lines", "c = lines(n)"],
              "n-by-3 matrix repeating the default line colour order (default 7 rows)."),
    "yyaxis": (["yyaxis left", "yyaxis right", "yyaxis(ax, side)"],
               "Gives the current axes a second y axis on the right. Later plots, labels and limits apply "
               "to the chosen side; each side has its own colour."),
    "cla": (["cla", "cla(ax)", "cla reset"],
            "Removes everything drawn in the current axes, keeping the figure and any other axes."),
    "subtitle": (["subtitle(txt)", "subtitle(ax, txt)", "t = subtitle(...)"],
                 "Adds a smaller second line of text under the title of the current axes. "
                 "title(txt, subtxt) sets both at once."),
    "xtickangle": (["xtickangle(angle)", "angle = xtickangle"],
                   "Rotates the x-axis tick labels by angle degrees, or returns the current rotation."),
    "ytickangle": (["ytickangle(angle)", "angle = ytickangle"],
                   "Rotates the y-axis tick labels by angle degrees, or returns the current rotation."),
    "zticks": (["zticks(v)", "v = zticks"], "Sets or returns the z-axis tick positions of 3-D axes."),
    "annotation": (["annotation('line', x, y)", "annotation('arrow', x, y)", "annotation('doublearrow', x, y)",
                    "annotation('textarrow', x, y, 'String', s)", "annotation('textbox', pos, 'String', s)",
                    "annotation('rectangle', pos)", "annotation('ellipse', pos)"],
                   "Draws a shape on the figure (not tied to axes) in normalized figure units from 0 to 1. "
                   "x = [x1 x2] and y = [y1 y2] give the start and end points; pos = [left bottom width "
                   "height]."),
})
# -- data analysis, interpolation, solvers and text-file input/output (library fixes) ------------
HELP.update({
    "accumarray": (["A = accumarray(subs, vals)", "A = accumarray(subs, vals, sz)",
                    "A = accumarray(subs, vals, sz, fun)", "A = accumarray(subs, vals, sz, fun, fillval)"],
                   "Groups the values vals by the subscripts in subs and combines each group. Each row of "
                   "subs is one subscript (a column of indices for a vector result, two columns for a "
                   "matrix, or a cell of index vectors). sz fixes the output size ([] lets the largest "
                   "subscript decide). fun is called with each group's values as a column and must return "
                   "a scalar (default: sum); if it returns a cell such as @(x) {x}, the result is a cell "
                   "array. Positions that receive no values hold fillval (default 0, or [] for cells). "
                   "Example: accumarray([1; 2; 1], [10; 20; 30], [], @max) is [30; 20]."),
    "histcounts": (["n = histcounts(x)", "n = histcounts(x, nbins)", "n = histcounts(x, edges)",
                    "n = histcounts(..., 'BinWidth', w)", "n = histcounts(..., 'NumBins', k)",
                    "n = histcounts(..., 'BinEdges', e)", "n = histcounts(..., 'BinLimits', [lo hi])",
                    "n = histcounts(..., 'Normalization', type)", "[n, edges, bin] = histcounts(...)"],
                   "Counts how many values of x fall in each bin; every bin includes its left edge and "
                   "the last bin also its right edge. Automatic bins use a round width (1, 2, 2.5 or 5 "
                   "times a power of ten) based on the spread of the data, with edges on multiples of it; "
                   "integer data over a small range gets unit-width bins centred on the integers. "
                   "Normalization may be 'count', 'probability', 'pdf', 'cdf', 'countdensity' or "
                   "'cumcount'. bin gives the bin number of each element (0 when outside). "
                   "Example: histcounts(1:5) is [1 1 1 1 1] with edges 0.5:1:5.5."),
    "sortrows": (["B = sortrows(A)", "B = sortrows(A, cols)", "B = sortrows(A, 'descend')",
                  "[B, idx] = sortrows(...)", "T2 = sortrows(T, vars)"],
                 "Sorts the rows of a matrix, char array or cell array. Rows are compared by the listed "
                 "columns in turn (by default all columns, left to right); a negative column number sorts "
                 "that column in descending order. Cell array columns must hold text or numeric scalars. "
                 "Ties keep their original order. idx is the row permutation. "
                 "Example: sortrows([3 1; 1 2; 3 0], [1 -2]) is [1 2; 3 1; 3 0]."),
    "ismember": (["tf = ismember(a, s)", "[tf, loc] = ismember(a, s)", "[tf, loc] = ismember(A, B, 'rows')"],
                 "True where an element of a is found in s; loc gives the index of the first match in s "
                 "(0 when absent). Works for numbers, text and cell arrays of text. With 'rows', each row "
                 "of A is looked up among the rows of B and column vectors are returned. "
                 "Example: ismember([1 2; 5 6], [5 6; 0 0], 'rows') is [false; true]."),
    "strsplit": (["C = strsplit(s)", "C = strsplit(s, delimiter)", "C = strsplit(..., 'CollapseDelimiters', tf)",
                  "C = strsplit(..., 'DelimiterType', 'RegularExpression')", "[C, matches] = strsplit(...)"],
                 "Splits text at a delimiter (default: whitespace) or at any of a cell array of "
                 "delimiters, which may contain escapes such as '\\t'. Runs of consecutive delimiters "
                 "count as one unless 'CollapseDelimiters' is false. A string input gives a string row; "
                 "a char input gives a cell row. matches holds the delimiter text that was found. "
                 "Example: strsplit('a,,b', ',') is {'a', 'b'}; with 'CollapseDelimiters', false it is "
                 "{'a', '', 'b'}."),
    "strtok": (["tok = strtok(s)", "tok = strtok(s, delimiters)", "[tok, rest] = strtok(...)"],
               "First token of the text: leading delimiter characters (default: whitespace) are skipped "
               "and the token runs up to the next delimiter. rest is the remainder, starting at that "
               "delimiter. Works elementwise on cell and string arrays. "
               "Example: [t, r] = strtok('  red green') gives t = 'red' and r = ' green'."),
    "splitlines": ("lines = splitlines(s)",
                   "Splits text at line breaks (\\n, \\r\\n or \\r). A string gives a string column; a char "
                   "vector gives a cell column of char vectors."),
    "missing": ("m = missing",
                "A missing value. Here it is a string scalar holding <missing>; ismissing(m) is true."),
    "str2double": ("x = str2double(s)",
                   "Converts text to a double. Returns NaN when the text is not a number. Cell arrays and "
                   "string arrays are converted element by element (a <missing> string gives NaN); a "
                   "numeric input gives NaN. Example: str2double([\"1.5\" \"x\"]) is [1.5 NaN]."),
    "upper": ("t = upper(s)", "Converts text to upper case, character by character, so the length never "
                              "changes (a letter whose upper-case form is longer, such as the German sharp s, "
                              "is kept). Works on char arrays, strings, string arrays and cells."),
    "lower": ("t = lower(s)", "Converts text to lower case, character by character, so the length never "
                              "changes. Works on char arrays, strings, string arrays and cells."),
    "repelem": (["B = repelem(v, n)", "B = repelem(A, r, c)"],
                "Repeats elements. For a vector v, each element is repeated n times (n may be a vector "
                "with one count per element). For a matrix, each row is repeated r times and each column "
                "c times. Example: repelem([1 2], [2 3]) is [1 1 2 2 2]."),
    "issorted": (["tf = issorted(A)", "tf = issorted(A, 'descend')", "tf = issorted(A, 'monotonic')",
                  "tf = issorted(A, 'rows')"],
                 "True if the elements (or rows, with 'rows') are already in order. Other modes: "
                 "'strictascend', 'strictdescend' and 'strictmonotonic'. Works on cell arrays of text too."),
    "maxk": (["B = maxk(A, k)", "B = maxk(A, k, dim)", "[B, I] = maxk(...)"],
             "The k largest elements along the first non-singleton dimension (or dim), largest first. "
             "NaN values come last. I gives their indices. Example: maxk([3 9 4 7], 2) is [9 7]."),
    "mink": (["B = mink(A, k)", "B = mink(A, k, dim)", "[B, I] = mink(...)"],
             "The k smallest elements along the first non-singleton dimension (or dim), smallest first. "
             "NaN values come last. Example: mink([3 9 4 7], 2) is [3 4]."),
    "pascal": (["P = pascal(n)", "L = pascal(n, 1)"],
               "Pascal matrix: P(i, j) is the binomial coefficient (i+j-2 choose j-1), a symmetric "
               "positive definite matrix of integers. pascal(n, 1) is a lower-triangular factor with "
               "alternating column signs that is its own inverse."),
    "gradient": (["g = gradient(f)", "g = gradient(f, h)", "[gx, gy] = gradient(F)",
                  "[gx, gy] = gradient(F, h)", "[gx, gy] = gradient(F, hx, hy)"],
                 "Numerical derivative: central differences in the interior and one-sided differences at "
                 "the ends. Spacing may be a scalar step or a vector of coordinates. For a matrix, gx "
                 "differentiates along each row (across columns) and gy along each column. "
                 "Example: gradient([1 4 9 16]) is [3 4 6 7]."),
    "conv2": (["C = conv2(A, B)", "C = conv2(A, B, shape)", "C = conv2(u, v, A)", "C = conv2(u, v, A, shape)"],
              "Two-dimensional convolution. shape 'full' (default) returns the whole result, 'same' the "
              "central part with the size of A, and 'valid' only the part computed without zero padding. "
              "With two vectors u and v, A is convolved with column u and then with row v."),
    "interp1": (["yq = interp1(x, y, xq)", "yq = interp1(y, xq)", "yq = interp1(..., method)",
                 "yq = interp1(..., method, 'extrap')", "yq = interp1(..., method, extrapval)"],
                "One-dimensional interpolation of the samples (x, y) at the points xq. Methods: "
                "'linear' (default), 'nearest', 'previous', 'next', 'spline' (not-a-knot cubic spline), "
                "'pchip' and 'cubic' (shape-preserving piecewise cubic) and 'makima' (modified Akima). "
                "Outside the range of x the result is NaN for the first four methods and extrapolated "
                "for the cubic ones, unless 'extrap' or a fill value is given. If y is a matrix, each "
                "column is interpolated. Example: interp1([1 2 3], [10 20 30], 2.5) is 25."),
    "spline": (["pp = spline(x, y)", "yq = spline(x, y, xq)"],
               "Cubic spline through the points (x, y) with not-a-knot end conditions; if y has two more "
               "values than x, the first and last give the end slopes. With xq, returns the values there; "
               "otherwise returns a piecewise polynomial for ppval. If y is a matrix, each row is a "
               "component of a vector-valued spline."),
    "pchip": (["pp = pchip(x, y)", "yq = pchip(x, y, xq)"],
              "Shape-preserving piecewise cubic Hermite interpolation: no overshoot between data points, "
              "and monotone data stay monotone. Returns values at xq or a piecewise polynomial for ppval."),
    "makima": (["pp = makima(x, y)", "yq = makima(x, y, xq)"],
               "Modified Akima piecewise cubic interpolation, which avoids most overshoot without "
               "flattening the curve as much as pchip. Returns values at xq or a piecewise polynomial."),
    "ppval": ("v = ppval(pp, xq)",
              "Evaluates a piecewise polynomial (from spline, pchip, makima or mkpp) at xq. Points "
              "outside the breaks use the first or last piece."),
    "mkpp": (["pp = mkpp(breaks, coefs)", "pp = mkpp(breaks, coefs, d)"],
             "Builds a piecewise polynomial. Row k of coefs holds the coefficients of piece k (highest "
             "power first) in the local variable x - breaks(k); d is the number of components for a "
             "vector-valued polynomial. Example: ppval(mkpp([0 1 2], [1 0; -1 1]), 1.5) is 0.5."),
    "unmkpp": ("[breaks, coefs, l, k, d] = unmkpp(pp)",
               "Takes a piecewise polynomial apart: its breaks, coefficient matrix, number of pieces l, "
               "order k and number of components d."),
    "interp2": (["Zq = interp2(X, Y, Z, Xq, Yq)", "Zq = interp2(Z, Xq, Yq)", "Zq = interp2(..., method)",
                 "Zq = interp2(..., method, extrapval)", "Zq = interp2(Z, k)", "Zq = interp2(Z)"],
                "Interpolation on a 2-D grid. X and Y are grid vectors or meshgrid matrices (default "
                "1:n and 1:m). A row vector Xq with a column vector Yq is expanded to a grid. Methods: "
                "'linear' (default), 'nearest', 'cubic', 'spline' and 'makima'. Points outside the grid "
                "give NaN or extrapval. interp2(Z, k) refines the grid by halving the spacing k times "
                "(once if k is omitted)."),
    "polyfit": (["p = polyfit(x, y, n)", "[p, S] = polyfit(x, y, n)", "[p, S, mu] = polyfit(x, y, n)"],
                "Least-squares polynomial of degree n through the points (x, y), highest power first. S "
                "holds the triangular factor R, the degrees of freedom df, the residual norm normr and "
                "rsquared, for error estimates with polyval. With mu, x is first centred and scaled: "
                "the fit uses (x - mu(1)) / mu(2), where mu = [mean(x); std(x)]; this helps with large "
                "or badly scaled x."),
    "polyval": (["y = polyval(p, x)", "[y, delta] = polyval(p, x, S)", "y = polyval(p, x, [], mu)",
                 "[y, delta] = polyval(p, x, S, mu)"],
                "Evaluates the polynomial p (highest power first) at each element of x. With mu from "
                "polyfit, x is centred and scaled first. With S from polyfit, delta is an error estimate "
                "such that y +/- delta covers new observations about 68% of the time for normal errors."),
    "fzero": (["x = fzero(f, x0)", "x = fzero(f, [a b])", "x = fzero(f, x0, options)",
               "[x, fval, exitflag, output] = fzero(...)"],
              "Finds a zero of a function of one variable, starting from x0 (a sign change is searched "
              "for around it) or inside the interval [a b] whose ends have opposite signs. output holds "
              "iterations, funcCount, algorithm and message."),
    "fminsearch": (["x = fminsearch(f, x0)", "x = fminsearch(f, x0, options)",
                    "[x, fval, exitflag, output] = fminsearch(...)"],
                   "Minimizes a function of several variables without derivatives (Nelder-Mead simplex), "
                   "starting at x0. Options TolX, TolFun, MaxIter and MaxFunEvals are read. output holds "
                   "iterations, funcCount, algorithm and message."),
    "fminbnd": (["x = fminbnd(f, a, b)", "x = fminbnd(f, a, b, options)", "[x, fval, exitflag, output] = fminbnd(...)"],
                "Minimizes a function of one variable on the interval [a, b] (golden section search with "
                "parabolic steps). output holds iterations, funcCount, algorithm and message."),
    "fminunc": (["x = fminunc(f, x0)", "[x, fval, exitflag, output, grad, hessian] = fminunc(...)"],
                "Minimizes a smooth function of several variables without constraints (quasi-Newton BFGS). "
                "grad is the gradient and hessian an approximation of the Hessian at the solution."),
    "ode45": (["[t, y] = ode45(f, tspan, y0)", "[t, y] = ode45(f, tspan, y0, options)",
               "[t, y, te, ye, ie] = ode45(f, tspan, y0, options)", "sol = ode45(...)"],
              "Solves the ODE y' = f(t, y) from y(tspan(1)) = y0 with an explicit Runge-Kutta (4,5) pair. "
              "If tspan has two elements, the output holds the solver steps, refined with four points per "
              "step taken from the solver's own interpolant; with more elements, the solution at exactly "
              "those times. Options from odeset: RelTol (default 1e-3), AbsTol (1e-6), MaxStep, "
              "InitialStep, Refine and Events. An event function [value, isterminal, direction] = "
              "g(t, y) reports zero crossings of value (direction 1: rising only, -1: falling only, 0: "
              "both); te, ye and ie give their times, states and indices, and a terminal event stops the "
              "integration. sol is a structure with x, y (and xe, ye, ie) for use with deval."),
    "ode23": (["[t, y] = ode23(f, tspan, y0)", "[t, y] = ode23(f, tspan, y0, options)",
               "[t, y, te, ye, ie] = ode23(...)", "sol = ode23(...)"],
              "Like ode45 but with a lower-order (2,3) Runge-Kutta pair, which can be cheaper at loose "
              "tolerances. Supports the same options and outputs."),
    "ode113": (["[t, y] = ode113(f, tspan, y0)", "[t, y] = ode113(f, tspan, y0, options)",
                "[t, y, te, ye, ie] = ode113(...)", "sol = ode113(...)"],
               "Solver for non-stiff problems at tight tolerances. mylab uses a high-order (8th order) "
               "Runge-Kutta method instead of a multistep method; options and outputs are as for ode45."),
    "ode78": (["[t, y] = ode78(f, tspan, y0)", "sol = ode78(...)"],
              "High-order Runge-Kutta solver for smooth non-stiff problems at tight tolerances. Options "
              "and outputs are as for ode45."),
    "ode89": (["[t, y] = ode89(f, tspan, y0)", "sol = ode89(...)"],
              "High-order Runge-Kutta solver (the same method as ode78 here). Options and outputs are as "
              "for ode45."),
    "ode15s": (["[t, y] = ode15s(f, tspan, y0)", "[t, y] = ode15s(f, tspan, y0, options)",
                "[t, y, te, ye, ie] = ode15s(...)", "sol = ode15s(...)"],
               "Solver for stiff problems (implicit backward differentiation formulas). ode23t and ode23tb "
               "use the same method and ode23s an implicit Runge-Kutta method. Options and outputs are "
               "as for ode45; mass matrices are not supported."),
    "odeset": (["opts = odeset(name, value, ...)", "opts = odeset(oldopts, name, value, ...)"],
               "Creates or updates an options structure for the ODE solvers, e.g. odeset('RelTol', 1e-6, "
               "'Events', @myevents)."),
    "odeget": (["v = odeget(opts, name)", "v = odeget(opts, name, default)"],
               "Reads one option from an odeset structure (the name is not case sensitive), or returns "
               "default (or []) when it is not set."),
    "deval": (["y = deval(sol, t)", "y = deval(sol, t, idx)", "[y, yp] = deval(...)", "y = deval(t, sol)"],
              "Evaluates a solution structure from an ODE solver at the times t, using the solver's own "
              "continuous interpolant, so the values are as accurate as the solution itself. Each column "
              "of y is the state at one time; idx selects components and yp holds the derivatives. "
              "Times must lie within the solved interval."),
    "etime": ("s = etime(t1, t0)",
              "Seconds elapsed from date vector t0 to date vector t1 ([year month day hour minute "
              "seconds], as returned by clock)."),
    "timeit": (["t = timeit(f)", "t = timeit(f, numOutputs)"],
               "Measures how long the function handle f (taking no inputs) takes to run: calls it several "
               "times and returns the median time in seconds. numOutputs is the number of outputs to "
               "request on each call (default 0)."),
    # -- tables ----------------------------------------------------------------------------------
    "table": (["T = table(var1, var2, ...)", "T = table(..., 'VariableNames', names)",
               "T = table(..., 'RowNames', rownames)",
               "T = table('Size', [m n], 'VariableTypes', types, 'VariableNames', names)"],
              "Creates a table: columns (variables) of equal height that may hold numbers, logicals, "
              "text in cells or strings. Variables are named after the input variables, or Var1, Var2, "
              "... Rows can be named with 'RowNames' (or later through T.Properties.RowNames) and then "
              "indexed by name, as in T('row2', :). Access a variable with T.name, a sub-table with "
              "T(rows, vars) and the contents with T{rows, vars}, which can also be assigned."),
    "array2table": (["T = array2table(A)", "T = array2table(A, 'VariableNames', names)",
                     "T = array2table(A, 'RowNames', rownames)"],
                    "Turns each column of the matrix A into a table variable."),
    "table2cell": ("C = table2cell(T)",
                   "Cell array with one cell per table entry: numbers as numeric values, text as char "
                   "vectors or strings."),
    "varfun": (["T2 = varfun(f, T)", "T2 = varfun(f, T, 'InputVariables', vars)",
                "T2 = varfun(f, T, 'GroupingVariables', groupvars)", "A = varfun(f, T, 'OutputFormat', 'uniform')"],
               "Applies the function f to each table variable. Result variables are named like "
               "mean_Height. With 'GroupingVariables', f is applied to each group of rows separately and "
               "the result has one row per group, with the group values and a GroupCount column. "
               "'OutputFormat' 'uniform' returns a numeric row and 'cell' a cell row instead of a table."),
    "findgroups": (["G = findgroups(A)", "[G, ID] = findgroups(A)", "[G, ID1, ID2, ...] = findgroups(A1, A2, ...)",
                    "[G, TID] = findgroups(T)"],
                   "Numbers the groups formed by the unique values (or unique combinations of values) of "
                   "the grouping vectors, in sorted order of those values. G gives the group number of "
                   "each element (NaN when a value is missing); ID holds each group's value. Works for "
                   "numeric, logical, cell arrays of text and string vectors, and for all variables of a "
                   "table. Example: findgroups([5 3 5]) is [2 1 2]."),
    "splitapply": (["Y = splitapply(f, X, G)", "Y = splitapply(f, X1, X2, ..., G)", "[Y1, Y2] = splitapply(...)"],
                   "Splits the data into groups given by the group numbers G (from findgroups), calls f "
                   "on each group and stacks the results, one row per group. Matrices are split by rows. "
                   "f must return the same size for each group; return a cell such as @(x) {x} for "
                   "anything else. Example: splitapply(@sum, [1 2 3], [1 2 1]) is [4 2]."),
    "groupsummary": (["S = groupsummary(T, groupvars)", "S = groupsummary(T, groupvars, method)",
                      "S = groupsummary(T, groupvars, method, datavars)", "B = groupsummary(X, G, method)"],
                     "Summary statistics for each group of table rows. The result has one row per group "
                     "with the grouping values, a GroupCount column and one column per method and data "
                     "variable, named like mean_Height. Methods: 'mean', 'sum', 'min', 'max', 'median', "
                     "'mode', 'std', 'var', 'range', 'numel', 'nnz', 'nummissing', 'all', or a function "
                     "handle; NaN values are left out of the statistics. datavars defaults to all other "
                     "numeric variables."),
    "innerjoin": (["T = innerjoin(Tleft, Tright)", "T = innerjoin(Tleft, Tright, 'Keys', keys)",
                   "T = innerjoin(..., 'LeftKeys', lk, 'RightKeys', rk)", "[T, ileft, iright] = innerjoin(...)"],
                  "Combines rows of two tables that have equal key values (by default all variables with "
                  "the same name). Every matching pair of rows gives one output row; the result is sorted "
                  "by key. Non-key variables of the right table are appended; names used in both tables "
                  "get _left and _right suffixes. 'LeftVariables' and 'RightVariables' choose the "
                  "variables to keep."),
    "outerjoin": (["T = outerjoin(Tleft, Tright)", "T = outerjoin(..., 'Keys', keys)",
                   "T = outerjoin(..., 'MergeKeys', true)", "T = outerjoin(..., 'Type', 'left')",
                   "[T, ileft, iright] = outerjoin(...)"],
                  "Like innerjoin but also keeps rows without a match ('Type' 'full', 'left' or 'right'), "
                  "filling the missing side with NaN, empty text or <missing>. Without 'MergeKeys' the key "
                  "variables of both tables are kept (with _left and _right suffixes); with it they are "
                  "merged into one."),
    "join": (["s = join(C)", "s = join(C, delimiter)", "s = join(C, delimiter, dim)", "T = join(Tleft, Tright)",
              "T = join(Tleft, Tright, 'Keys', keys)"],
             "Text: joins the strings in C with a delimiter (default a space) along the last dimension, "
             "or along dim. Tables: adds to each row of Tleft the variables of the row of Tright with the "
             "same key values (by default the variables with the same name); every left row must match "
             "exactly one right row, and the left row order is kept."),
    "movevars": (["T2 = movevars(T, vars)", "T2 = movevars(T, vars, 'Before', location)",
                  "T2 = movevars(T, vars, 'After', location)"],
                 "Moves table variables to a new position, before or after another variable (given by "
                 "name or number), or to the end."),
    "rmmissing": (["B = rmmissing(A)", "B = rmmissing(A, dim)", "[B, TF] = rmmissing(...)",
                   "B = rmmissing(..., 'MinNumMissing', n)"],
                  "Removes missing entries: NaN in numeric data, empty char vectors in cell arrays, "
                  "<missing> strings. A vector loses the missing elements, a matrix or table the rows "
                  "(or columns, with dim 2) that contain at least n missing values (default 1). TF marks "
                  "what was removed."),
    "fillmissing": (["F = fillmissing(A, 'constant', v)", "F = fillmissing(A, method)",
                     "F = fillmissing(A, method, dim)", "F = fillmissing(..., 'EndValues', e)",
                     "[F, TF] = fillmissing(...)"],
                    "Replaces missing entries. Methods: 'constant' (the value v), 'previous', 'next', "
                    "'nearest' and, for numeric data, 'linear', 'spline', 'pchip' and 'makima'. Matrices "
                    "are filled column by column (or along dim); tables variable by variable. Leading or "
                    "trailing gaps are extrapolated by the interpolating methods unless 'EndValues' is "
                    "'none', 'nearest' or a constant. TF marks the entries that were filled. "
                    "Example: fillmissing([1 NaN 3], 'linear') is [1 2 3]."),
    # -- text-file input and output ------------------------------------------------------------
    "readmatrix": (["A = readmatrix(filename)", "A = readmatrix(filename, 'NumHeaderLines', n)",
                    "A = readmatrix(filename, 'Delimiter', d)"],
                   "Reads delimited numeric text into a matrix. The delimiter (comma, tab, semicolon, bar "
                   "or whitespace) is detected when not given. Leading lines with only text, such as a "
                   "header row, are skipped automatically unless 'NumHeaderLines' says how many to skip. "
                   "Empty or non-numeric fields become NaN, and short rows are padded with NaN."),
    "csvread": (["A = csvread(filename)", "A = csvread(filename, r, c)"],
                "Reads a comma-separated numeric file; empty fields become 0. r and c are zero-based row "
                "and column offsets of the first value to read."),
    "dlmread": (["A = dlmread(filename)", "A = dlmread(filename, delimiter)", "A = dlmread(filename, delimiter, r, c)"],
                "Reads a delimited numeric file; empty fields become 0. r and c are zero-based offsets."),
    "writematrix": (["writematrix(A, filename)", "writematrix(A, filename, 'Delimiter', d)",
                     "writematrix(A, filename, 'WriteMode', 'append')"],
                    "Writes a matrix to a delimited text file with full precision, one row per line. The "
                    "delimiter may be 'comma' (default; tab for .tsv files), 'tab', 'space', 'semi', 'bar' "
                    "or any single character. Cell arrays, strings and char arrays are written as text, "
                    "quoted when they contain the delimiter."),
    "csvwrite": (["csvwrite(filename, A)", "csvwrite(filename, A, r, c)"],
                 "Writes a numeric matrix as comma-separated values with five significant digits; r and c "
                 "shift the data down and right, padding with zeros."),
    "dlmwrite": (["dlmwrite(filename, A)", "dlmwrite(filename, A, delimiter)", "dlmwrite(filename, A, delimiter, r, c)",
                  "dlmwrite(filename, A, 'delimiter', d, 'precision', p)", "dlmwrite(filename, A, '-append', ...)"],
                 "Writes a numeric matrix to a delimited text file (comma by default; '\\t' or 'tab' for "
                 "tabs). precision is a number of significant digits or a format such as '%.3f' "
                 "(default five significant digits)."),
    "writecell": (["writecell(C, filename)", "writecell(C, filename, 'Delimiter', d)"],
                  "Writes a cell array of numbers and text to a delimited text file, one row of the cell "
                  "array per line."),
    "readcell": (["C = readcell(filename)", "C = readcell(filename, 'Delimiter', d)",
                  "C = readcell(filename, 'NumHeaderLines', n)"],
                 "Reads a delimited text file into a cell array: numeric fields become numbers, other "
                 "fields char vectors and empty fields <missing>."),
    "readlines": (["lines = readlines(filename)", "lines = readlines(filename, 'EmptyLineRule', 'skip')",
                   "lines = readlines(filename, 'WhitespaceRule', 'trim')"],
                  "Reads a text file into a string column, one element per line (a final line break does "
                  "not add an empty line)."),
    "writelines": (["writelines(lines, filename)", "writelines(lines, filename, 'WriteMode', 'append')"],
                   "Writes text (a string array, cell array or char vector) to a file, one element per "
                   "line, each followed by a line break."),
    "type": ("type filename",
             "Prints the contents of a text file. If the file is not found as given, name.m and the "
             "search path are tried."),
    "importdata": (["A = importdata(filename)", "A = importdata(filename, delimiter)",
                    "A = importdata(filename, delimiter, nheaderlines)",
                    "[A, delimiter, nheaderlines] = importdata(...)"],
                   "Loads numeric text data. Without header lines the result is a matrix. With header "
                   "lines it is a structure with data (the matrix), textdata (the header text) and, when "
                   "the last header line names the columns, colheaders. The delimiter and number of "
                   "header lines are detected when not given. .mat files are loaded as with load."),
    "fscanf": (["A = fscanf(fid, format)", "A = fscanf(fid, format, size)", "[A, count] = fscanf(...)"],
               "Reads formatted data from the rest of an open file, like sscanf: conversions such as %d, "
               "%f, %g and %s are applied repeatedly. size may be n, Inf or [m n] (with n possibly Inf) "
               "and fills the result column by column."),
    "textscan": (["C = textscan(fid, format)", "C = textscan(str, format)", "C = textscan(..., N)",
                  "C = textscan(..., 'Delimiter', d, 'HeaderLines', n)"],
                 "Reads columns of formatted text into a 1-by-n cell array, one cell per conversion: %f "
                 "and %n give double columns, %d and %u integer columns (int32/uint32, or with a size "
                 "suffix such as %d8), %s and %q cell columns of text (%q removes double quotes) and %c "
                 "characters; %*s skips a field. Fields are separated by whitespace or by 'Delimiter'; "
                 "'HeaderLines' skips lines first and N limits the number of records. Other options: "
                 "'MultipleDelimsAsOne', 'CollectOutput', 'CommentStyle', 'EmptyValue' and 'TextType'."),
    "dir": (["dir", "dir(name)", "listing = dir(...)"],
            "Lists a folder, the files matching a wildcard pattern such as '*.csv', or a single file. "
            "With an output, returns a column struct array with fields name, folder, date, bytes, isdir "
            "and datenum. A folder listing includes '.' and '..'."),
    "rmdir": (["rmdir(folder)", "rmdir(folder, 's')", "[status, msg] = rmdir(...)"],
              "Removes an empty folder, or with 's' the folder and everything in it. With outputs, "
              "failure is reported through status and msg instead of an error."),
    "tempdir": ("d = tempdir", "The system folder for temporary files, ending with a file separator."),
    "tempname": (["f = tempname", "f = tempname(folder)"],
                 "A unique name for a temporary file or folder (nothing is created)."),
})
HELP["ode23s"] = HELP["ode23t"] = HELP["ode23tb"] = HELP["ode15s"]
