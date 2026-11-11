function [counts, firstError] = scan_log(filename)
%SCAN_LOG Count ERROR/WARN/INFO lines and return the first error message.
counts = zeros(1, 3);
firstError = '';
levels = {'ERROR', 'WARN', 'INFO'};
pat = '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[(ERROR|WARN|INFO|DEBUG)\](?: (.*))?$';
fid = fopen(filename, 'r');
if fid < 0
    error('scan_log:open', 'Cannot open %s', filename);
end
line = fgetl(fid);
while ischar(line)
    if ~isempty(line) && line(end) == char(13)
        line = line(1:end-1);
    end
    tok = regexp(line, pat, 'tokens', 'once');
    if ~isempty(tok)
        idx = find(strcmp(tok{1}, levels));
        if ~isempty(idx)
            counts(idx) = counts(idx) + 1;
            if idx == 1 && counts(1) == 1
                if numel(tok) >= 2
                    firstError = strtrim(tok{2});
                end
            end
        end
    end
    line = fgetl(fid);
end
fclose(fid);
end
