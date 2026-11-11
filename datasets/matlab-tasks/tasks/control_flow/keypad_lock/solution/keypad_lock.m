function [state, wrong] = keypad_lock(code, keys)
%KEYPAD_LOCK Simulate a keypad lock with three allowed wrong attempts.
if ~all(ismember(keys, '0123456789*#'))
    error('keypad_lock:badKey', 'Invalid key pressed.');
end
state = 'locked';
wrong = 0;
buffer = '';
for key = keys
    if ~strcmp(state, 'locked')
        break
    end
    switch key
        case '*'
            buffer = '';
        case '#'
            if isempty(buffer)
                continue
            end
            if strcmp(buffer, code)
                state = 'open';
            else
                wrong = wrong + 1;
                if wrong >= 3
                    state = 'blocked';
                end
            end
            buffer = '';
        otherwise
            buffer = [buffer key]; %#ok<AGROW>
    end
end
end
