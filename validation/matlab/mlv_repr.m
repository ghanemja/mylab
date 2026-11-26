function s = mlv_repr(v)
%MLV_REPR  One-line text form of a value, identical in MATLAB and in mylab.
%   Used to compare results between engines: class, size and contents with
%   17 significant digits. Cells are listed element by element in column order.
if ischar(v)
    t = v';
    t = strrep(strrep(t(:)', sprintf('\n'), '\n'), sprintf('\t'), '\t');
    s = sprintf('char[%dx%d] "%s"', size(v, 1), size(v, 2), t);
elseif iscell(v)
    parts = cell(1, numel(v));
    for k = 1:numel(v)
        parts{k} = mlv_repr(v{k});
    end
    s = sprintf('cell[%s] {%s}', strtrim(sprintf('%d ', size(v))), strjoin(parts, ' ; '));
elseif isnumeric(v) || islogical(v)
    d = double(v(:))';
    if isreal(v)
        s = sprintf('%s[%s] %s', class(v), strtrim(sprintf('%d ', size(v))), strtrim(sprintf('%.17g ', d)));
    else
        s = sprintf('%s[%s] complex %s | %s', class(v), strtrim(sprintf('%d ', size(v))), ...
            strtrim(sprintf('%.17g ', real(d))), strtrim(sprintf('%.17g ', imag(d))));
    end
elseif isstring(v)
    parts = cell(1, numel(v));
    for k = 1:numel(v)
        if ismissing(v(k))
            parts{k} = '<missing>';
        else
            parts{k} = ['"' char(v(k)) '"'];
        end
    end
    s = sprintf('string[%s] %s', strtrim(sprintf('%d ', size(v))), strjoin(parts, ' '));
else
    s = sprintf('<%s>[%s]', class(v), strtrim(sprintf('%d ', size(v))));
end
end
