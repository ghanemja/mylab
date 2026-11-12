"""Help text for the newer text, struct and data-cleaning functions."""

HELP = {
    "dec2base": (["s = dec2base(d, base)", "s = dec2base(d, base, n)", "s = dec2base(d, symbols)"],
                 "Text for the non-negative integer d in the given base (2 to 36), with at least n digits. "
                 "An array gives a char matrix with one row per number. A char vector of symbols can be given "
                 "instead of the base."),
    "base2dec": (["d = base2dec(s, base)", "d = base2dec(s, symbols)"],
                 "Number written by the text s in the given base (2 to 36). A char matrix or cell array gives a "
                 "column of numbers; text that is not a valid number gives NaN."),
    "dec2bin": (["s = dec2bin(d)", "s = dec2bin(d, n)"],
                "Binary digits of the non-negative integer d as text, with at least n digits. An array gives a "
                "char matrix with one row per number. Example: dec2bin(5) is '101'."),
    "dec2hex": (["s = dec2hex(d)", "s = dec2hex(d, n)"],
                "Hexadecimal digits (upper case) of the non-negative integer d as text, with at least n digits. "
                "An array gives a char matrix with one row per number. Example: dec2hex(255) is 'FF'."),
    "bin2dec": ("d = bin2dec(s)",
                "Value of a binary digit string; spaces are ignored. A char matrix or cell array gives a column "
                "of numbers. Example: bin2dec('101') is 5."),
    "hex2dec": ("d = hex2dec(s)",
                "Value of a hexadecimal digit string. A char matrix or cell array gives a column of numbers. "
                "Example: hex2dec('ff') is 255."),
    "orderfields": (["s = orderfields(s)", "s = orderfields(s, names)", "s = orderfields(s, perm)",
                     "s = orderfields(s, s2)", "[s, p] = orderfields(...)"],
                    "Reorders the fields of a struct: alphabetically, in the order of a cell array of names, by a "
                    "permutation vector, or like the fields of another struct. p is the permutation that was "
                    "applied."),
    "mat2str": (["s = mat2str(A)", "s = mat2str(A, n)", "s = mat2str(A, 'class')"],
                "Text that rebuilds the matrix A when evaluated, with n significant digits (15 by default). "
                "With 'class' the text is wrapped in the class name, e.g. int8([1 2 3]). An empty array is "
                "written as zeros(m,n)."),
    "num2cell": (["C = num2cell(A)", "C = num2cell(A, dims)"],
                 "Cell array with each element of A in its own cell. With dims, the listed dimensions are kept "
                 "together: num2cell(A, 1) puts each column of A in one cell."),
    "numfields": ("n = numfields(s)", "Number of fields of the struct s (0 if s is not a struct)."),
    "toupper": ("t = toupper(s)", "Same as upper: converts text to upper case."),
    "tolower": ("t = tolower(s)", "Same as lower: converts text to lower case."),
    "isStringScalar": ("tf = isStringScalar(x)", "True if x is a string array with exactly one element."),
    "convertStringsToChars": (["a = convertStringsToChars(a)", "[a, b, ...] = convertStringsToChars(a, b, ...)"],
                              "Converts string scalars to char vectors and string arrays to cell arrays of char "
                              "vectors. Other values are returned unchanged."),
    "convertCharsToStrings": (["a = convertCharsToStrings(a)", "[a, b, ...] = convertCharsToStrings(a, b, ...)"],
                              "Converts char arrays to string scalars and cell arrays of char vectors to string "
                              "arrays. Other values are returned unchanged."),
    "lasterr": (["msg = lasterr", "[msg, id] = lasterr", "lasterr(msg, id)"],
                "Message (and identifier) of the most recent error. It is reliable inside a catch block; "
                "prefer 'catch err' and err.message in new code."),
    "lasterror": (["s = lasterror", "lasterror('reset')"],
                  "Struct with the message, identifier and stack of the most recent error. It is reliable inside "
                  "a catch block; prefer 'catch err' in new code."),
    "anynan": ("tf = anynan(A)", "True if any element of A is NaN."),
    "allfinite": ("tf = allfinite(A)", "True if no element of A is NaN or infinite."),
    "isoutlier": (["tf = isoutlier(A)", "tf = isoutlier(A, method)", "tf = isoutlier(..., dim)",
                   "tf = isoutlier(..., 'ThresholdFactor', f)"],
                  "Marks the outliers of A along its first non-singleton dimension. The default method 'median' "
                  "flags values more than 3 scaled median absolute deviations from the median; 'mean' uses 3 "
                  "standard deviations from the mean and 'quartiles' 1.5 interquartile ranges outside the "
                  "quartiles."),
    "standardizeMissing": ("B = standardizeMissing(A, indicators)",
                           "Replaces the values of A that equal one of the indicators with the standard missing "
                           "value: NaN for numbers, '' in a cell array of char vectors, <missing> for strings."),
    "uniquetol": (["C = uniquetol(A)", "C = uniquetol(A, tol)", "[C, ia, ic] = uniquetol(...)"],
                  "Unique values of A where values closer than tol*max(abs(A(:))) count as equal (tol is 1e-12 "
                  "for double and 1e-6 for single by default)."),
}
