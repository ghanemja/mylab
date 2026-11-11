function s = read_settings(filename)
%READ_SETTINGS Parse key = value lines with optional [section] headers.
s = struct();
fid = fopen(filename, 'r');
if fid < 0
    error('read_settings:open', 'Cannot open %s', filename);
end
section = '';
line = fgetl(fid);
while ischar(line)
    line = strtrim(line);
    if ~isempty(line) && line(1) ~= '#' && line(1) ~= ';'
        if line(1) == '[' && line(end) == ']'
            section = strtrim(line(2:end-1));
        else
            eq = find(line == '=', 1);
            if ~isempty(eq)
                key = strtrim(line(1:eq-1));
                val = strtrim(line(eq+1:end));
                if strcmpi(val, 'true')
                    v = true;
                elseif strcmpi(val, 'false')
                    v = false;
                else
                    v = str2double(val);
                    if isnan(v)
                        v = val;
                    end
                end
                if isempty(section)
                    s.(key) = v;
                else
                    s.([section '_' key]) = v;
                end
            end
        end
    end
    line = fgetl(fid);
end
fclose(fid);
end
