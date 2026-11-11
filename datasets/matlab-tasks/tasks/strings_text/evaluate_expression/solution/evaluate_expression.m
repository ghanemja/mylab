function v = evaluate_expression(s)
%EVALUATE_EXPRESSION Evaluate + - * / ( ) expressions by recursive descent.
toks = tokenize(s);
[v, p] = parse_sum(toks, 1);
if p <= numel(toks)
    syntax_error();
end
end

function toks = tokenize(s)
toks = {};
i = 1;
n = numel(s);
while i <= n
    c = s(i);
    if isspace(c)
        i = i + 1;
    elseif any(c == '+-*/()')
        toks{end + 1} = c; %#ok<AGROW>
        i = i + 1;
    elseif (c >= '0' && c <= '9') || c == '.'
        j = i;
        while j <= n && ((s(j) >= '0' && s(j) <= '9') || s(j) == '.')
            j = j + 1;
        end
        word = s(i:j-1);
        if sum(word == '.') > 1 || all(word == '.')
            syntax_error();
        end
        toks{end + 1} = str2double(word); %#ok<AGROW>
        i = j;
    else
        syntax_error();
    end
end
end

function [v, p] = parse_sum(t, p)
[v, p] = parse_product(t, p);
while p <= numel(t) && ischar(t{p}) && any(t{p} == '+-')
    op = t{p};
    [r, p] = parse_product(t, p + 1);
    if op == '+'
        v = v + r;
    else
        v = v - r;
    end
end
end

function [v, p] = parse_product(t, p)
[v, p] = parse_unary(t, p);
while p <= numel(t) && ischar(t{p}) && any(t{p} == '*/')
    op = t{p};
    [r, p] = parse_unary(t, p + 1);
    if op == '*'
        v = v * r;
    else
        v = v / r;
    end
end
end

function [v, p] = parse_unary(t, p)
if p > numel(t)
    syntax_error();
end
tok = t{p};
if isnumeric(tok)
    v = tok;
    p = p + 1;
elseif tok == '-'
    [v, p] = parse_unary(t, p + 1);
    v = -v;
elseif tok == '('
    [v, p] = parse_sum(t, p + 1);
    if p > numel(t) || ~isequal(t{p}, ')')
        syntax_error();
    end
    p = p + 1;
else
    syntax_error();
end
end

function syntax_error()
error('evaluate_expression:syntax', 'Malformed expression.');
end
