function r = cfh_cleanup(v)
% try/catch inside a function, rethrowing with a new identifier.
try
    if v < 0
        error('cfh:neg', 'negative: %d', v);
    end
    r = sqrt(v);
catch err
    error('cfh:wrapped', 'wrapped <%s>', err.message);
end
end
