"""Help text for pattern objects and the text functions that search with them."""

_USE = (" Patterns are combined with + (one after the other) and | (either one) and can be given to contains, "
        "matches, startsWith, endsWith, count, extract, extractBefore, extractAfter, extractBetween, replace, "
        "erase, insertBefore, insertAfter, replaceBetween, eraseBetween, split, strip and strfind.")
_COUNT = (" With no argument it matches one or more of them, with N exactly N, and with (min, max) between min "
          "and max (max may be Inf).")
_KIND = " The type is 'start', 'end' or 'either' (the default)."
_NOT = " Put ~ in front of the pattern to match the places where it does not hold."


def _boundary(name: str, what: str):
    return ([f"pat = {name}", f"pat = {name}(type)"],
            f"A pattern that matches a position, not characters: {what}." + _KIND + _NOT + _USE)


HELP = {
    "pattern": (["pat = pattern(text)", "pat = pattern"],
                "A pattern that matches the given text literally. With a string array or a cell array of "
                "character vectors it matches any one of the texts. Other patterns are built with functions "
                "such as digitsPattern, lettersPattern and wildcardPattern." + _USE),
    "ispattern": ("tf = ispattern(x)", "True when x is a pattern object."),
    "digitsPattern": (["pat = digitsPattern", "pat = digitsPattern(N)", "pat = digitsPattern(min, max)"],
                      "A pattern that matches digit characters 0-9." + _COUNT + _USE
                      + ' Example: extract("abc123def45", digitsPattern) gives ["123"; "45"].'),
    "lettersPattern": (["pat = lettersPattern", "pat = lettersPattern(N)", "pat = lettersPattern(min, max)"],
                       "A pattern that matches letters (of any alphabet)." + _COUNT + _USE),
    "alphanumericsPattern": (["pat = alphanumericsPattern", "pat = alphanumericsPattern(N)",
                              "pat = alphanumericsPattern(min, max)"],
                             "A pattern that matches letters and digits." + _COUNT + _USE),
    "whitespacePattern": (["pat = whitespacePattern", "pat = whitespacePattern(N)",
                           "pat = whitespacePattern(min, max)"],
                          "A pattern that matches whitespace characters (space, tab, line breaks)." + _COUNT + _USE),
    "wildcardPattern": (["pat = wildcardPattern", "pat = wildcardPattern(N)", "pat = wildcardPattern(min, max)",
                         "pat = wildcardPattern(..., 'Except', pat2)"],
                        "A pattern that matches any characters, as few as possible: zero or more by default, "
                        "exactly N, or between min and max. With 'Except' the matched characters must not "
                        "start a match of pat2." + _USE + ' Example: "<" + wildcardPattern + ">" matches one tag.'),
    "characterListPattern": (["pat = characterListPattern(chars)", "pat = characterListPattern(first, last)"],
                             "A pattern that matches one character out of the given ones, or one character in "
                             "the range from first to last." + _USE),
    "optionalPattern": ("pat = optionalPattern(p)", "A pattern that matches p if it is there, and nothing "
                        "otherwise." + _USE),
    "asManyOfPattern": (["pat = asManyOfPattern(p)", "pat = asManyOfPattern(p, min)",
                         "pat = asManyOfPattern(p, min, max)"],
                        "A pattern that matches p repeated as many times as possible, at least min times "
                        "(default 1) and at most max times." + _USE),
    "asFewOfPattern": (["pat = asFewOfPattern(p)", "pat = asFewOfPattern(p, min)", "pat = asFewOfPattern(p, min, max)"],
                       "A pattern that matches p repeated as few times as possible, at least min times "
                       "(default 0) and at most max times." + _USE),
    "caseInsensitivePattern": ("pat = caseInsensitivePattern(p)", "A pattern that matches p without regard to "
                               "upper and lower case." + _USE),
    "caseSensitivePattern": ("pat = caseSensitivePattern(p)", "A pattern that matches p with the exact case, even "
                             "in a function that is called with 'IgnoreCase', true." + _USE),
    "regexpPattern": (["pat = regexpPattern(expr)", "pat = regexpPattern(expr, Name, Value)"],
                      "A pattern that matches the regular expression expr. Options: 'IgnoreCase', "
                      "'DotExceptNewline', 'MultilineAnchors' and 'FreeSpacing' (true or false)." + _USE),
    "namedPattern": (["pat = namedPattern(p, name)", "pat = namedPattern(p, name, description)"],
                     "Gives the pattern p a name, which is shown when the pattern is displayed. It matches "
                     "what p matches." + _USE),
    "textBoundary": _boundary("textBoundary", "the start or the end of the text"),
    "lineBoundary": _boundary("lineBoundary", "the start or the end of a line"),
    "letterBoundary": _boundary("letterBoundary", "a place where a letter meets a character that is not a letter"),
    "digitBoundary": _boundary("digitBoundary", "a place where a digit meets a character that is not a digit"),
    "alphanumericBoundary": _boundary("alphanumericBoundary", "a place where a letter or digit meets another "
                                      "kind of character"),
    "whitespaceBoundary": _boundary("whitespaceBoundary", "a place where whitespace meets another character"),
    "lookAheadBoundary": ("pat = lookAheadBoundary(p)", "A pattern that matches the position in front of a match "
                          "of p without taking the characters of p." + _NOT + _USE),
    "lookBehindBoundary": ("pat = lookBehindBoundary(p)", "A pattern that matches the position right after a "
                           "match of p without taking its characters. p may have a variable length within "
                           "limits (a repetition counts up to 16 extra characters)." + _NOT + _USE),
    "matches": (["tf = matches(str, pat)", "tf = matches(str, pat, 'IgnoreCase', true)"],
                "True for each element of str (a string array, char vector or cell array of char vectors) "
                "that matches pat as a whole. pat is text, a list of texts (any one may match) or a pattern."),
    "extract": ("parts = extract(str, pat)",
                "All the pieces of str that match the pattern (or text) pat. For one text the result is an "
                "N-by-1 string array, or cell array of char vectors when str is a char vector. For an array "
                "of texts every element must have the same number of matches."),
    "replaceBetween": (["new = replaceBetween(str, startPat, endPat, newText)",
                        "new = replaceBetween(str, startPos, endPos, newText)",
                        "new = replaceBetween(..., 'Boundaries', bounds)"],
                       "Replaces the text between a start and an end marker by newText. Markers are text or "
                       "patterns (every occurrence is replaced, the markers stay) or character positions (the "
                       "characters at the positions are replaced too). 'Boundaries' is 'inclusive' or "
                       "'exclusive' and overrides that default."),
    "eraseBetween": (["new = eraseBetween(str, startPat, endPat)", "new = eraseBetween(str, startPos, endPos)",
                      "new = eraseBetween(..., 'Boundaries', bounds)"],
                     "Removes the text between a start and an end marker. Markers are text or patterns (the "
                     "markers stay) or character positions (the characters at the positions are removed too). "
                     "'Boundaries' is 'inclusive' or 'exclusive' and overrides that default."),
    "append": ("s = append(s1, s2, ..., sN)",
               "Joins text element by element. Unlike strcat it keeps trailing blanks of char vectors. Arrays "
               "must have the same size; single texts are repeated. The result is a string array if any "
               "input is a string, else a cell array if any input is a cell array, else a char vector."),
    "regexptranslate": (["s = regexptranslate('escape', text)", "s = regexptranslate('wildcard', text)",
                         "s = regexptranslate('flexible', text, expr)"],
                        "Rewrites text for use in a regular expression. 'escape' puts a backslash before "
                        "every special character so that the text is matched literally. 'wildcard' turns "
                        "the wildcards * and ? into .* and . and escapes dots. 'flexible' replaces every "
                        "character that is not a letter, digit or underscore by expr."),
}
