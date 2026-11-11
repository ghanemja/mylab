function out = names_last_first(names)
%NAMES_LAST_FIRST Rewrite "Given Middle Family" as "Family, Given M.".
out = strings(size(names));
for k = 1:numel(names)
    t = strtrim(char(names(k)));
    if isempty(t)
        out(k) = "";
        continue
    end
    parts = strsplit(t, ' ');
    if numel(parts) == 1
        out(k) = string(parts{1});
        continue
    end
    s = [parts{end} ', ' parts{1}];
    for j = 2:numel(parts) - 1
        s = [s ' ' parts{j}(1) '.']; %#ok<AGROW>
    end
    out(k) = string(s);
end
end
