function s = cf_dump_value(v)
% Serialize any value to a JSON object: class, size and contents.
% Numbers are written as text (17 significant digits) inside a JSON string
% so NaN, Inf, -Inf and -0 survive. Characters are written as code points.
c = class(v);
head = ['{"class":"' c '","size":[' strrep(cf_ints(size(v)), ' ', ',') ']'];
if isnumeric(v)
    if isreal(v)
        s = [head ',"re":"' cf_nums(v) '"}'];
    else
        s = [head ',"re":"' cf_nums(real(v)) '","im":"' cf_nums(imag(v)) '"}'];
    end
elseif islogical(v)
    s = [head ',"re":"' cf_ints(double(v(:))) '"}'];
elseif ischar(v)
    s = [head ',"re":"' cf_ints(double(v(:))) '"}'];
elseif iscell(v)
    s = [head ',"items":['];
    for k = 1:numel(v)
        if k > 1
            s = [s ','];
        end
        s = [s cf_dump_value(v{k})];
    end
    s = [s ']}'];
elseif isstruct(v)
    f = fieldnames(v);
    s = [head ',"fields":['];
    for j = 1:numel(f)
        if j > 1
            s = [s ','];
        end
        s = [s '"' f{j} '"'];
    end
    s = [s '],"items":['];
    for k = 1:numel(v)
        if k > 1
            s = [s ','];
        end
        s = [s '['];
        for j = 1:numel(f)
            if j > 1
                s = [s ','];
            end
            s = [s cf_dump_value(v(k).(f{j}))];
        end
        s = [s ']'];
    end
    s = [s ']}'];
elseif isa(v, 'function_handle')
    t = func2str(v);
    s = [head ',"text":"' cf_ints(double(t(:))) '"}'];
elseif isa(v, 'containers.Map')
    s = [head ',"count":' sprintf('%d', v.Count) ',"keys":' cf_dump_value(keys(v)) ...
        ',"values":' cf_dump_value(values(v)) '}'];
elseif isa(v, 'string')
    s = [head ',"items":['];
    for k = 1:numel(v)
        if k > 1
            s = [s ','];
        end
        if ismissing(v(k))
            s = [s '{"class":"missing","size":[1,1]}'];
        else
            s = [s cf_dump_value(char(v(k)))];
        end
    end
    s = [s ']}'];
else
    s = [head ',"opaque":1}'];
end
end

function t = cf_ints(x)
t = sprintf('%d ', x);
t = t(1:end-1);
end

function t = cf_nums(x)
x = x(:);
if isinteger(x)
    t = sprintf('%d ', x);
else
    t = sprintf('%.17g ', double(x));
end
t = t(1:end-1);
end
