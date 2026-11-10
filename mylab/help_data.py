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
            "on the diagonal starting at the top-left. Example: eye(2) is [1 0; 0 1]."),
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
    "size": (["sz = size(A)", "n = size(A, dim)", "[r, c] = size(A)", "[d1, d2, ..., dn] = size(A)"],
             "Dimensions of A. With one output, a row vector holding the length of every dimension; "
             "size(A, dim) gives one length, and 1 for a dimension beyond ndims(A). With several outputs, "
             "the last output absorbs all remaining dimensions (their product), so [r, c] = size(A) puts "
             "the product of dimensions 2 and up in c. Example: size(ones(2, 5)) is [2 5]."),
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
              "'cell', 'struct', 'string' or 'function_handle'."),
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
            "integer), 'float' (double or single) and 'integer' (any integer class)."),
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
               "Converts to double precision. Char input gives character codes; a string scalar is parsed "
               "as a number (NaN if it is not one). Cells and structs cannot be converted. "
               "Example: double('A') is 65."),
    "single": ("y = single(x)", "Converts to single precision (32-bit floating point)."),
    "logical": ("tf = logical(x)",
                "Converts to logical: nonzero becomes true, zero becomes false. NaN input is an error."),
    "char": (["c = char(codes)", "c = char(str)", "c = char(C)", "c = char(a, b, ...)"],
             "Makes a char array. Numbers are taken as character codes. A cell array or several text inputs "
             "become rows of a char matrix, padded with spaces to equal length. Example: char(72, 105) is "
             "a two-row char array; char([72 105]) is 'Hi'."),
    "cast": ("y = cast(x, classname)",
             "Converts x to the named class: 'double', 'single', 'logical' or an integer class."),
    "typecast": ("y = typecast(x, classname)",
                 "Reinterpreting the bytes of an array as another type is not supported yet; calling it "
                 "raises an error."),
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
               "accumulates from the end. NaN flags are accepted but have no effect. "
               "Example: cumsum([1 2 3]) is [1 3 6]."),
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
    "histcounts": (["n = histcounts(x)", "n = histcounts(x, nbins)", "n = histcounts(x, edges)",
                    "[n, edges] = histcounts(...)"],
                   "Counts how many values of x fall in each bin. Bins are chosen automatically, set by a "
                   "bin count, or given as an edge vector; the last bin includes its right edge. The "
                   "automatic bin choice may differ from other implementations."),
    "accumarray": ("A = accumarray(subs, vals)",
                   "Adds each vals(k) into position subs(k) of the output. A column of indices gives a "
                   "column vector; a two-column index matrix gives a matrix. vals may be a scalar. Only "
                   "summing is supported (no size or function arguments). "
                   "Example: accumarray([1; 2; 1], [10; 20; 30]) is [40; 20]."),
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
             "when ascending and first when descending; complex values sort by magnitude. A cell array of "
             "char vectors is sorted alphabetically. Example: sort([3 1 2]) is [1 2 3]."),
    "sortrows": (["B = sortrows(A)", "B = sortrows(A, cols)", "B = sortrows(A, 'descend')", "[B, idx] = sortrows(...)"],
                 "Sorts the rows of a matrix, comparing the first column, then the second for ties, and so "
                 "on. cols picks the columns to compare; a negative column number sorts that column in "
                 "descending order. idx gives the original row numbers. Cell arrays are not supported."),
    "unique": (["u = unique(x)", "u = unique(x, 'stable')", "u = unique(A, 'rows')", "[u, ia, ic] = unique(...)"],
               "The distinct values of x, sorted unless 'stable' is given (first-appearance order). The "
               "result is a row if x is a row, otherwise a column. ia and ic are index columns with "
               "u = x(ia) and x = u(ic); 'last' makes ia point to last occurrences. 'rows' finds distinct "
               "rows. Also works on cell arrays of char vectors. Example: unique([3 1 3 2]) is [1 2 3]."),
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
    "ismember": (["tf = ismember(a, s)", "[tf, loc] = ismember(a, s)"],
                 "True for each element of a that also occurs in s; loc gives the index of its first match "
                 "in s, or 0. Works for numbers, and for text against a cell array of text. "
                 "Example: ismember([1 5], [5 6]) is [false true]."),
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
                "reused until all data is consumed. A non-integer given to %d is shown in %e style. Returns "
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
    "str2double": ("x = str2double(s)",
                   "Parses text as one number. Accepts decimal, exponent and complex forms, Inf and NaN, and "
                   "ignores thousands commas. Gives NaN if the text is not a valid number. A cell array "
                   "gives an array of results. Example: str2double('2.5e3') is 2500."),
    "str2num": (["x = str2num(s)", "[x, ok] = str2num(s)"],
                "Evaluates text as a bracketed expression, so '1 2 3' gives [1 2 3] and '[1;2]' a column. "
                "On failure returns [] and ok is false. Because the text is executed, prefer str2double "
                "for untrusted input."),
    "string": ("str = string(x)",
               "Converts a char vector or a numeric scalar into a string scalar. Only scalars are "
               "supported because mylab has no multi-element string arrays yet."),
    "strlength": ("n = strlength(s)", "Number of characters in a char vector or string scalar."),
    "newline": ("c = newline", "A char vector holding the line-feed character (code 10)."),
    "blanks": ("s = blanks(n)", "Char vector of n spaces."),
    # -- text: comparison and search ---------------------------------------------------------
    "strcmp": ("tf = strcmp(a, b)",
               "True if two texts are identical (case-sensitive). Either input may be a cell array, giving "
               "an element-wise logical array. Non-text inputs give false rather than an error."),
    "strcmpi": ("tf = strcmpi(a, b)", "Like strcmp but ignores upper/lower case."),
    "strncmp": ("tf = strncmp(a, b, n)",
                "True if the first n characters of two texts match (case-sensitive). Texts shorter than n "
                "never match. Cell arrays compare element by element."),
    "strncmpi": ("tf = strncmpi(a, b, n)", "Like strncmp but ignores upper/lower case."),
    "upper": ("t = upper(s)", "Converts text to upper case. Works on char arrays, strings and cell arrays of text."),
    "lower": ("t = lower(s)", "Converts text to lower case. Works on char arrays, strings and cell arrays of text."),
    "strtrim": ("t = strtrim(s)",
                "Removes leading and trailing whitespace (and null characters). Works on char, string and "
                "cell arrays of text."),
    "deblank": ("t = deblank(s)", "Removes trailing whitespace and null characters only."),
    "reverse": ("t = reverse(s)", "Reverses the order of the characters. Example: reverse('abc') is 'cba'."),
    "strrep": ("t = strrep(s, old, new)",
               "Replaces every occurrence of old in s with new. s may be a cell array of text. "
               "Example: strrep('a-b-c', '-', '+') is 'a+b+c'."),
    "strfind": ("k = strfind(s, pattern)",
                "Starting indices of every occurrence of pattern in s, overlapping ones included, as a row "
                "vector ([] if none). For a cell array, returns a cell of index vectors. "
                "Example: strfind('abcabc', 'bc') is [2 5]."),
    "contains": (["tf = contains(s, pattern)", "tf = contains(s, pattern, 'IgnoreCase', true)"],
                 "True if s contains pattern anywhere. pattern may be a cell array, meaning any of them. "
                 "s may be a cell array, giving a logical array."),
    "startsWith": (["tf = startsWith(s, pattern)", "tf = startsWith(s, pattern, 'IgnoreCase', true)"],
                   "True if s begins with pattern (or any pattern in a cell array). Works element-wise on a "
                   "cell array s."),
    "endsWith": (["tf = endsWith(s, pattern)", "tf = endsWith(s, pattern, 'IgnoreCase', true)"],
                 "True if s ends with pattern (or any pattern in a cell array). Works element-wise on a "
                 "cell array s."),
    "strsplit": (["C = strsplit(s)", "C = strsplit(s, delimiter)"],
                 "Splits text into a 1-by-N cell array. By default it splits on whitespace, treating a run "
                 "of whitespace as one break. A delimiter (or a cell array of delimiters, with escapes such "
                 "as '\\t' allowed) splits at each occurrence, so adjacent delimiters give empty pieces. "
                 "Example: strsplit('a,b', ',') is {'a', 'b'}."),
    "strjoin": (["s = strjoin(C)", "s = strjoin(C, delimiter)"],
                "Joins the texts in a cell array into one char vector, separated by delimiter (a space by "
                "default). Escapes such as '\\n' in the delimiter are expanded."),
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
               "are written (?<name>...). Matching uses Python's regex engine, so rare constructs may "
               "differ. A cell array s gives a cell of results."),
    "regexpi": (["out = regexpi(s, expr, keyword, ...)"],
                "Case-insensitive form of regexp: the same outputs and options, but letters match "
                "regardless of case."),
    "regexprep": (["t = regexprep(s, expr, replacement)", "t = regexprep(..., 'ignorecase')",
                   "t = regexprep(..., 'once')"],
                  "Replaces every match of a regular expression. In the replacement, $1, $2, ... insert "
                  "captured groups and $0 the whole match. 'once' replaces only the first match. s may be a "
                  "cell array. Example: regexprep('abc', '(b)', '[$1]') is 'a[b]c'."),
    # -- cell arrays --------------------------------------------------------------------------
    "cell": (["C = cell(n)", "C = cell(m, n)", "C = cell([m n])"],
             "Cell array whose cells all hold the empty matrix []. cell(n) is n-by-n."),
    "iscellstr": ("tf = iscellstr(C)",
                  "True if C is a cell array in which every cell holds a char vector (or empty char)."),
    "cellstr": ("C = cellstr(s)",
                "Converts text to a cell array: each row of a char matrix becomes one cell (trailing blanks "
                "removed), giving a column. A string becomes a 1-by-1 cell; a cell array is returned as is."),
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
             "Shows the calling forms and a short description of a function. For functions in your own .m "
             "files it shows the comment block at the top of the file. With an output the text is returned "
             "instead of printed."),
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
    "writematrix": ("writematrix(A, filename)",
                    "Writes a numeric matrix to a text file as comma-separated values with up to 15 "
                    "significant digits. Other delimiters and options are not supported."),
    "csvwrite": ("csvwrite(filename, A)", "Writes a numeric matrix as comma-separated values, like writematrix. "
                 "Row and column offsets are not supported."),
    "dlmwrite": ("dlmwrite(filename, A)",
                 "Writes a numeric matrix to a text file. mylab always uses commas; a delimiter argument is "
                 "not supported."),
    "readmatrix": ("A = readmatrix(filename)",
                   "Reads a comma-separated text file into a numeric matrix. Fields that are not numbers become "
                   "NaN. Options are not supported."),
    "csvread": ("A = csvread(filename)", "Reads a comma-separated file into a numeric matrix, like readmatrix. "
                "Row and column offsets are not supported."),
    "dlmread": ("A = dlmread(filename)",
                "Reads a comma-separated file into a numeric matrix. mylab always splits on commas; other "
                "delimiters and ranges are not supported."),
    "save": (["save", "save(filename)", "save(filename, 'var1', 'var2', ...)", "save filename var1 var2"],
             "Writes variables from the workspace to a .mat file (the extension is added if missing; the "
             "default name is matlab.mat). Only numeric, logical, char and string variables are saved; "
             "cells, structs and handles are skipped. Option flags such as -ascii are ignored."),
    "load": (["load(filename)", "S = load(filename)", "load filename"],
             "Reads variables from a .mat file into the workspace, or into the fields of S when an output is "
             "given. Numeric data is loaded as double. A .csv, .txt or .dat file is read as a numeric matrix "
             "and stored in a variable named after the file (or returned)."),
    # -- polynomials ---------------------------------------------------------------------------
    "polyfit": (["p = polyfit(x, y, n)", "[p, S] = polyfit(x, y, n)"],
                "Least-squares polynomial of degree n through the points (x, y). p is a row vector of "
                "coefficients, highest power first. S has fields normr (norm of the residuals) and df "
                "(degrees of freedom). Centering and scaling (a third output) are not supported. "
                "Example: polyfit([0 1 2], [1 3 5], 1) is about [2 1]."),
    "polyval": ("y = polyval(p, x)",
                "Evaluates the polynomial with coefficients p (highest power first) at every element of x. "
                "Example: polyval([1 0 -1], 3) is 8."),
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
    "interp1": (["yq = interp1(x, y, xq)", "yq = interp1(y, xq)", "yq = interp1(..., method)",
                 "yq = interp1(..., method, 'extrap')", "yq = interp1(..., method, fillvalue)"],
                "1-D interpolation of the data (x, y) at the query points xq. When x is omitted it is "
                "1:n. method is 'linear' (default), 'nearest', 'previous', 'next', 'pchip' or 'spline'; "
                "'cubic' and 'makima' are treated as 'spline' in mylab. Points outside the data give NaN, "
                "or fillvalue, or are extrapolated with 'extrap' (the default for 'spline'). If y is a matrix "
                "each column is interpolated. 'spline' needs at least four data points."),
    "spline": ("yq = spline(x, y, xq)",
               "Cubic spline through the points (x, y) with not-a-knot end conditions, evaluated at xq. "
               "The form without query points (returning a piecewise polynomial) is not supported."),
    "interp2": (["Zq = interp2(X, Y, Z, Xq, Yq)", "Zq = interp2(X, Y, Z, Xq, Yq, method)"],
                "2-D interpolation on a grid such as one made by meshgrid. method is 'linear' (default) or "
                "'nearest' (any other name is treated as 'nearest'). Points outside the grid give NaN. "
                "The grid coordinates must be given."),
    # -- solving and optimization ----------------------------------------------------------------
    "optimset": (["opts = optimset(name, value, ...)"],
                 "Builds an options struct from name/value pairs for fzero, fminsearch and similar solvers. "
                 "mylab reads TolX, TolFun, MaxIter and MaxFunEvals where they apply and ignores other names."),
    "optimoptions": ("opts = optimoptions(solvername, name, value, ...)",
                     "Same as optimset; the solver name is accepted and ignored. Returns a plain struct."),
    "fzero": (["x = fzero(f, x0)", "x = fzero(f, [a b])", "x = fzero(f, x0, options)",
               "[x, fval, exitflag] = fzero(...)"],
              "Finds a point where the scalar function f crosses zero. With an interval [a b], f(a) and "
              "f(b) must have opposite signs; with a single start x0 it first searches outward for such an "
              "interval and fails if none is found. Example: fzero(@cos, [1 2]) is pi/2."),
    "fminsearch": (["x = fminsearch(f, x0)", "x = fminsearch(f, x0, options)", "[x, fval, exitflag] = fminsearch(...)"],
                   "Minimizes a function of one or more variables, starting at x0, with the derivative-free "
                   "Nelder-Mead simplex method. Options TolX, TolFun, MaxIter and MaxFunEvals are honoured. "
                   "Finds a local minimum only."),
    "fminbnd": (["x = fminbnd(f, a, b)", "[x, fval, exitflag] = fminbnd(...)"],
                "Minimizes a scalar function of one variable on the interval [a, b]. An options argument is "
                "not supported."),
    "fminunc": (["x = fminunc(f, x0)", "[x, fval, exitflag] = fminunc(...)"],
                "Minimizes a smooth function of several variables from x0 with a quasi-Newton (BFGS) method "
                "using numerical gradients. Options are ignored."),
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
    "ode45": (["[t, y] = ode45(f, tspan, y0)", "[t, y] = ode45(f, tspan, y0, options)", "sol = ode45(...)"],
              "Solves the ODE system dy/dt = f(t, y) from y0, using an adaptive explicit Runge-Kutta (4,5) "
              "method. f receives t and a column y and returns the derivatives. With tspan = [t0 tf] the "
              "solver's own steps are returned (each step refined into four points); with more entries the "
              "solution is reported at those times. Each row of y is the state at t. With one output, sol "
              "is a struct with fields x, y and solver for use with deval. RelTol and AbsTol options apply."),
    "ode23": (["[t, y] = ode23(f, tspan, y0)", "[t, y] = ode23(f, tspan, y0, options)", "sol = ode23(...)"],
              "Like ode45 but with a lower-order Runge-Kutta (2,3) pair; can be cheaper at loose tolerances."),
    "ode113": (["[t, y] = ode113(f, tspan, y0)", "[t, y] = ode113(f, tspan, y0, options)", "sol = ode113(...)"],
               "Same calling forms as ode45. mylab solves it with a high-order (8th) explicit Runge-Kutta "
               "method rather than a multistep method, suited to tight tolerances."),
    "ode15s": (["[t, y] = ode15s(f, tspan, y0)", "[t, y] = ode15s(f, tspan, y0, options)", "sol = ode15s(...)"],
               "Solver for stiff problems, with the same calling forms as ode45. mylab uses an implicit "
               "backward differentiation (BDF) method; ode23s, ode23t and ode23tb use the same method."),
    "odeset": ("opts = odeset(name, value, ...)",
               "Builds an options struct for the ODE solvers from name/value pairs. The solvers use RelTol "
               "(default 1e-3) and AbsTol (default 1e-6); other names are stored but ignored."),
    "deval": ("y = deval(sol, t)",
              "Evaluates a solution struct returned by an ODE solver at the times t; each row of the result "
              "is one state component. mylab interpolates linearly between the stored solution points."),
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
    "ode23s": "ode15s", "ode23t": "ode15s", "ode23tb": "ode15s",
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
    "table": (["T = table(var1, var2, ...)", "T = table(..., 'VariableNames', names)",
               "T = table('Size', [h w], 'VariableTypes', types)"],
              "Creates a table whose columns are the given variables (each must have the same number of rows; "
              "row vectors are turned into columns). Column names come from the argument variable names, or "
              "Var1, Var2, ... Access a column with T.Name, a subset with T(rows, vars) and the raw data with "
              "T{rows, vars}."),
    "array2table": (["T = array2table(A)", "T = array2table(A, 'VariableNames', names)"],
                    "Makes a table with one variable per column of the matrix A."),
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
    "varfun": ("T2 = varfun(f, T)",
               "Applies the function f to each table variable and returns a table of the results, with "
               "variables named f_Name."),
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
