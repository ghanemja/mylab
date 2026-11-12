function cfh_err(kind)
% Raises errors of several kinds.
switch kind
    case 'id'
        error('cfh:tooBig', 'Value %d is too big for %s', 5, 'x');
    case 'plain'
        error('plain message');
    case 'format'
        error('count=%d ratio=%.2f', 3, 0.5);
    case 'struct'
        s.message = 'from struct';
        s.identifier = 'cfh:fromStruct';
        error(s);
    case 'nested'
        cfh_err('id');
    case 'idonly'
        error('cfh:only', 'only');
    case 'percent'
        error('cfh:pct', '100%% done');
    case 'newline'
        error('cfh:nl', 'line1\nline2');
end
end
