"""Help text for the functions in builtins/text_more.py."""

_OCT = " This is an Octave function that mylab accepts so that Octave scripts run; MATLAB does not have it."
_VALID = (" Use it in an arguments block or a property declaration, e.g. x {NAME}, or call it directly; "
          "it returns nothing and raises an error with identifier MATLAB:validators:NAME when the check fails.")


def _v(name: str, usage: str, what: str):
    return (usage, what + _VALID.replace("NAME", name))


def _is(name: str, what: str):
    return (f"tf = {name}(s)", f"Logical array of the size of the char array s that is true where a character is "
            f"{what}. A cell array gives a cell array of results; a value that is not text gives false." + _OCT)


HELP = {
    # -- Octave output --------------------------------------------------------------------------------
    "printf": ("printf(format, a, b, ...)",
               "Writes formatted text to the command window, like fprintf(format, ...)." + _OCT),
    "puts": ("puts(text)", "Writes text to the command window exactly as given: no format codes are "
             "interpreted and no newline is added." + _OCT),
    "fputs": ("fputs(fid, text)", "Writes text, exactly as given, to the file with identifier fid "
              "(1 and 2 mean the command window)." + _OCT),
    "fdisp": ("fdisp(fid, x)", "Writes what disp(x) shows to the file with identifier fid "
              "(1 and 2 mean the command window)." + _OCT),
    "stdout": ("fid = stdout", "The file identifier of the command window output, 1." + _OCT),
    "stderr": ("fid = stderr", "The file identifier of the error output, 2." + _OCT),
    "merge": ("r = merge(mask, tval, fval)",
              "Chooses between two values. With a logical scalar mask the result is tval when it is true and "
              "fval otherwise (the values can be of any type). With a logical array mask the result takes "
              "elements of tval where mask is true and of fval elsewhere; each value is a scalar or has the "
              "size of mask. ifelse is another name for the same function." + _OCT),
    "ifelse": ("r = ifelse(mask, tval, fval)", "Another name for merge(mask, tval, fval)." + _OCT),
    "rows": ("n = rows(a)", "Number of rows of a, the same as size(a, 1)." + _OCT),
    "columns": ("n = columns(a)", "Number of columns of a, the same as size(a, 2)." + _OCT),
    "isargout": ("tf = isargout(k)", "True when output number k of the running function was requested by "
                 "the caller. Outputs skipped with ~ are not tracked, so they count as requested." + _OCT),
    "print_usage": ("print_usage()", "Stops with an error that says the running function was called in "
                    "a wrong way." + _OCT),
    "usage": ("usage(format, ...)", "Stops with an error whose message starts with 'usage: '." + _OCT),
    "NA": (["x = NA", "x = NA(m, n)"], "Octave's marker for a missing value. In mylab it is NaN." + _OCT),
    "e": (["x = e", "x = e(m, n)"], "The base of the natural logarithm, 2.71828..., the same as exp(1). A "
          "variable called e takes precedence, as for every function." + _OCT),
    # -- character tests ------------------------------------------------------------------------------
    "isalpha": _is("isalpha", "a letter (the same as isletter)"),
    "isdigit": _is("isdigit", "one of the decimal digits 0 to 9"),
    "isupper": _is("isupper", "an upper-case letter"),
    "islower": _is("islower", "a lower-case letter"),
    "ispunct": _is("ispunct", "a punctuation character (printable, and neither a letter, a digit nor a space)"),
    "isalnum": _is("isalnum", "a letter or a digit"),
    "isxdigit": _is("isxdigit", "a hexadecimal digit (0-9, a-f, A-F)"),
    "iscntrl": _is("iscntrl", "a control character (code below 32, or 127)"),
    "isgraph": _is("isgraph", "a printable character other than a space"),
    "isprint": _is("isprint", "a printable character, including the space"),
    "isascii": _is("isascii", "an ASCII character (code below 128)"),
    # -- small text functions --------------------------------------------------------------------------
    "do_string_escapes": ("t = do_string_escapes(s)", "Replaces escape sequences such as \\n and \\t in s "
                          "by the characters they stand for." + _OCT),
    "undo_string_escapes": ("t = undo_string_escapes(s)", "Replaces special characters in s (newline, tab, "
                            "backslash, double quote ...) by their escape sequences." + _OCT),
    "postpad": (["y = postpad(x, n)", "y = postpad(x, n, c)", "y = postpad(x, n, c, dim)"],
                "Makes x have length n along its first non-singleton dimension (or dim) by adding the value "
                "c (default 0) at the end, or by removing elements from the end when x is longer." + _OCT),
    "prepad": (["y = prepad(x, n)", "y = prepad(x, n, c)", "y = prepad(x, n, c, dim)"],
               "Makes x have length n along its first non-singleton dimension (or dim) by adding the value "
               "c (default 0) at the start, or by removing elements from the start when x is longer." + _OCT),
    "sumsq": (["s = sumsq(x)", "s = sumsq(x, dim)"], "Sum of the squared magnitudes of the elements, "
              "sum(abs(x).^2, dim)." + _OCT),
    "vec": ("v = vec(x)", "All elements of x as one column, the same as x(:)." + _OCT),
    "index": (["k = index(s, t)", "k = index(s, t, 'last')"], "Position of the first (or, with 'last', the "
              "last) occurrence of the text t in s, or 0 when t does not occur." + _OCT),
    "rindex": ("k = rindex(s, t)", "Position of the last occurrence of the text t in s, or 0." + _OCT),
    "substr": (["t = substr(s, offset)", "t = substr(s, offset, len)"], "The part of s that starts at "
               "position offset and is len characters long (to the end of s by default)." + _OCT),
    "ostrsplit": (["c = ostrsplit(s, seps)", "c = ostrsplit(s, seps, true)"],
                  "Splits s at every character that occurs in seps and returns the pieces in a cell array. "
                  "With true as third argument empty pieces are left out." + _OCT),
    "terminal_size": ("rc = terminal_size()", "The [rows columns] of the command window that the display "
                      "of wide matrices is laid out for." + _OCT),
    "isglobal": ("tf = isglobal(name)", "True when the variable called name (a char vector or string) is "
                 "declared global in the current workspace."),
    "isindex": (["tf = isindex(v)", "tf = isindex(v, n)"], "True when v can be used as an index: a logical "
                "array, or positive whole numbers (not larger than n when it is given)." + _OCT),
    # -- errors -----------------------------------------------------------------------------------------
    "MException.last": (["ME = MException.last", "MException.last('reset')"],
                        "The MException of the error that was caught most recently by try/catch in this "
                        "session (an MException with empty identifier and message if there was none). "
                        "'reset' forgets it. In MATLAB the function reports the last uncaught error at the "
                        "prompt; mylab also counts errors that were caught."),
    # -- validators ----------------------------------------------------------------------------------------
    "mustBeFloat": _v("mustBeFloat", "mustBeFloat(x)", "Checks that x is a double or single array."),
    "mustBeUnderlyingType": _v("mustBeUnderlyingType", "mustBeUnderlyingType(x, types)",
                               "Checks that the class of x is one of the names in types (text or a list)."),
    "mustBeScalarOrEmpty": _v("mustBeScalarOrEmpty", "mustBeScalarOrEmpty(x)",
                              "Checks that x has at most one element."),
    "mustBeRow": _v("mustBeRow", "mustBeRow(x)", "Checks that x is 1-by-N."),
    "mustBeColumn": _v("mustBeColumn", "mustBeColumn(x)", "Checks that x is N-by-1."),
    "mustBeMatrix": _v("mustBeMatrix", "mustBeMatrix(x)", "Checks that x has two dimensions."),
    "mustBeValidVariableName": _v("mustBeValidVariableName", "mustBeValidVariableName(name)",
                                  "Checks that name is text that can be used as a variable name."),
    "mustBeNonmissing": _v("mustBeNonmissing", "mustBeNonmissing(x)",
                           "Checks that x has no missing element (NaN, NaT, <missing>, <undefined>)."),
    "mustBeNonsparse": _v("mustBeNonsparse", "mustBeNonsparse(x)", "Checks that x is not a sparse matrix."),
    "mustBeSparse": _v("mustBeSparse", "mustBeSparse(x)", "Checks that x is a sparse matrix."),
    "mustBeBetween": _v("mustBeBetween", ["mustBeBetween(x, lo, hi)", "mustBeBetween(x, lo, hi, intervalType)"],
                        "Checks that every element of x lies between lo and hi. intervalType is 'closed' "
                        "(default), 'open', 'openleft', 'openright', 'closedleft' or 'closedright'."),
}
