% MATLAB-style identifiers of common built-in errors; expectations are in overrides.json.
try, undefined_thing_cfh(1); catch e, id1 = e.identifier; end
try, x = [1 2 3]; x(7); catch e, id2 = e.identifier; end
try, [1 2 3] * [4 5 6]; catch e, id3 = e.identifier; end
try, s.a = 1; s.b; catch e, id4 = e.identifier; end
try, x = [1 2 3]; x(0); catch e, id5 = e.identifier; end
try, x = [1 2 3]; x(1.5); catch e, id8 = e.identifier; end
try, c = {1, 2}; c{3}; catch e, id9 = e.identifier; end
try, error('abc'); catch e, id11 = e.identifier; end
try, ones(2, 3) * ones(2, 3); catch e, id12 = e.identifier; end
try, cell2mat(5); catch e, id13 = e.identifier; end
try, q = zeros(2); q(:, 3) = [1 2 3]; catch e, id14 = e.identifier; end
try, str2func({1}); catch e, id15 = ischar(e.identifier); end
