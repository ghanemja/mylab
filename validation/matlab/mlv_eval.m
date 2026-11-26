function s = mlv_eval(expr__)
%MLV_EVAL  Evaluate one expression and return its one-line text form (see mlv_repr).
%   The same file runs in MATLAB and in mylab, so the two results can be compared.
try
    v__ = eval(expr__);
    s = mlv_repr(v__);
catch err__
    s = ['ERROR ' strrep(err__.message, newline, ' ')];
end
end
