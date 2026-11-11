function lines = wrap_text(s, width)
%WRAP_TEXT Greedy word wrap into a column cell of lines.
words = regexp(s, '\S+', 'match');
lines = cell(0, 1);
cur = '';
for k = 1:numel(words)
    w = words{k};
    if isempty(cur)
        cur = w;
    elseif numel(cur) + 1 + numel(w) <= width
        cur = [cur ' ' w]; %#ok<AGROW>
    else
        lines{end + 1, 1} = cur; %#ok<AGROW>
        cur = w;
    end
end
if ~isempty(cur)
    lines{end + 1, 1} = cur;
end
end
