function varargout = cfh_varout(n)
% Returns n outputs: 10, 20, 30, ...
for k = 1:n
    varargout{k} = 10 * k;
end
end
