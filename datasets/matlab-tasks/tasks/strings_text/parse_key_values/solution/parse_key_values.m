function cfg = parse_key_values(s)
%PARSE_KEY_VALUES Parse 'key=value; key=value' into a struct.
cfg = struct();
entries = strsplit(s, ';');
for k = 1:numel(entries)
    e = strtrim(entries{k});
    if isempty(e)
        continue
    end
    eq = find(e == '=', 1);
    if isempty(eq)
        error('parse_key_values:syntax', 'Entry "%s" has no "=".', e);
    end
    key = strtrim(e(1:eq-1));
    val = strtrim(e(eq+1:end));
    if ~isvarname(key)
        error('parse_key_values:badKey', 'Invalid key "%s".', key);
    end
    num = str2double(val);
    if strcmpi(val, 'true')
        v = true;
    elseif strcmpi(val, 'false')
        v = false;
    elseif ~isnan(num)
        v = num;
    else
        v = val;
    end
    cfg.(key) = v;
end
end
