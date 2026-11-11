function v = get_by_path(data, path)
%GET_BY_PATH Follow a path such as 'a.b(2).c{3}' into nested data.
steps = parse_path(path);
v = data;
for k = 1:size(steps, 1)
    kind = steps{k, 1};
    arg = steps{k, 2};
    switch kind
        case '.'
            if ~(isstruct(v) && isscalar(v) && isfield(v, arg))
                not_found();
            end
            v = v.(arg);
        case '('
            if arg > numel(v)
                not_found();
            end
            v = v(arg);
        case '{'
            if ~iscell(v) || arg > numel(v)
                not_found();
            end
            v = v{arg};
    end
end
end

function steps = parse_path(path)
steps = cell(0, 2);
pos = 1;
n = numel(path);
while pos <= n
    c = path(pos);
    if c == '.' || (pos == 1 && is_letter(c))
        if c == '.'
            if pos == 1
                bad_path();
            end
            pos = pos + 1;
        end
        name = regexp(path(pos:end), '^[A-Za-z]\w*', 'match', 'once');
        if isempty(name)
            bad_path();
        end
        steps(end + 1, :) = {'.', name}; %#ok<AGROW>
        pos = pos + numel(name);
    elseif c == '(' || c == '{'
        if c == '('
            closer = ')';
        else
            closer = '}';
        end
        digits = regexp(path(pos+1:end), '^\d+', 'match', 'once');
        after = pos + 1 + numel(digits);
        if isempty(digits) || after > n || path(after) ~= closer
            bad_path();
        end
        k = str2double(digits);
        if k < 1
            bad_path();
        end
        steps(end + 1, :) = {c, k}; %#ok<AGROW>
        pos = after + 1;
    else
        bad_path();
    end
end
end

function tf = is_letter(c)
tf = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
end

function bad_path()
error('get_by_path:badPath', 'Malformed path.');
end

function not_found()
error('get_by_path:notFound', 'Path does not exist in the data.');
end
