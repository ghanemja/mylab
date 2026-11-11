function C = parse_quoted_csv(filename)
%PARSE_QUOTED_CSV Read a CSV file with quoted fields into a cell array.
rows = {};
fid = fopen(filename, 'r');
if fid < 0
    error('parse_quoted_csv:open', 'Cannot open %s', filename);
end
line = fgetl(fid);
while ischar(line)
    if ~isempty(line) && line(end) == char(13)
        line = line(1:end-1);
    end
    if ~isempty(line)
        rows{end + 1} = split_record(line); %#ok<AGROW>
    end
    line = fgetl(fid);
end
fclose(fid);
if isempty(rows)
    C = cell(0, 0);
    return
end
C = cell(numel(rows), numel(rows{1}));
for r = 1:numel(rows)
    C(r, :) = rows{r};
end
end

function fields = split_record(line)
fields = {};
n = numel(line);
i = 1;
while true
    if i <= n && line(i) == '"'
        buf = '';
        i = i + 1;
        while i <= n
            if line(i) == '"'
                if i < n && line(i + 1) == '"'
                    buf(end + 1) = '"'; %#ok<AGROW>
                    i = i + 2;
                else
                    i = i + 1;   % closing quote
                    break
                end
            else
                buf(end + 1) = line(i); %#ok<AGROW>
                i = i + 1;
            end
        end
    else
        j = i;
        while j <= n && line(j) ~= ','
            j = j + 1;
        end
        buf = line(i:j - 1);
        i = j;
    end
    if isempty(buf)
        buf = '';
    end
    fields{end + 1} = buf; %#ok<AGROW>
    if i > n
        break
    end
    i = i + 1;   % skip the comma
    if i > n
        fields{end + 1} = ''; %#ok<AGROW>
        break
    end
end
end
