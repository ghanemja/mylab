Write a MATLAB function

    lines = wrap_text(s, width)

that wraps the text in the char vector `s` into lines of at most `width`
characters using the greedy method: words (maximal runs of non-whitespace
characters) are placed on the current line, separated by single spaces, as
long as the line stays within `width`; otherwise a new line starts. A word
longer than `width` is not split but placed on a line of its own.

Return an `m-by-1` **cell array** of char vectors (no leading or trailing
spaces in any line). Any whitespace (spaces, tabs, newlines) in `s` only
separates words. If `s` has no words, return a `0-by-1` cell.

Example:

    wrap_text('small tasks make the benchmark easy to grow over time', 12)
    % returns {'small tasks'; 'make the'; 'benchmark'; 'easy to grow'; 'over time'}
