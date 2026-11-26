"""Tables and timetables: indexing, properties, display, reshaping, grouping, sets, missing data, retime.

Each line is MATLAB code that must run without an error; the expected values in the
assertions were worked out by hand from the documented behaviour of each function.
"""

import pytest

TABLES = [
    ("T = table([1; 2; 3], {'a'; 'b'; 'c'}, [true; false; true], 'VariableNames', {'N', 'S', "
     "'L'}); assert(height(T) == 3 && width(T) == 3); assert(isequal(T.N, [1; 2; 3])); assert(T{2,"
     " 'N'} == 2); assert(isequal(size(T), [3 3]));"),
    ("T = table([1; 2; 3], {'a'; 'b'; 'c'}, 'VariableNames', {'N', 'S'}); U = T(T.N > 1, :); "
     "assert(height(U) == 2); T.D = T.N * 2; assert(isequal(T.Properties.VariableNames, {'N', 'S',"
     " 'D'})); assert(isequal(T{:, {'N', 'D'}}, [1 2; 2 4; 3 6]));"),
    ("T = table([1; 2; 3], {'a'; 'b'; 'c'}, 'VariableNames', {'N', 'S'}, 'RowNames', {'r1', 'r2', "
     "'r3'}); assert(T{'r3', 'N'} == 3); assert(isequal(T({'r1', 'r3'}, :).N, [1; 3])); "
     "assert(isequal(T.Row, {'r1'; 'r2'; 'r3'})); assert(isequal(T.Properties.RowNames, {'r1'; "
     "'r2'; 'r3'}));"),
    ("T = table([1; 2; 3], [4; 5; 6], 'VariableNames', {'a', 'b'}); assert(isequal(T.Variables, [1"
     ' 4; 2 5; 3 6])); assert(isequal(T{:, :}, [1 4; 2 5; 3 6])); assert(isequal(T{end, end}, 6));'
     " assert(isequal(T(end, :).a, 3)); assert(isequal(T{[true false true], 'b'}, [4; 6]));"),
    ("T = table([1; 2; 3], [4; 5; 6], 'VariableNames', {'a', 'b'}); T.a(2) = 20; assert(T.a(2) == "
     "20); T{3, 'b'} = 60; assert(T.b(3) == 60); T(1, :) = {100, 400}; assert(T.a(1) == 100 && "
     'T.b(1) == 400); T.c = T.a + T.b; assert(width(T) == 3);'),
    ("T = table([1; 2; 3], [4; 5; 6], 'VariableNames', {'a', 'b'}); T(2, :) = []; assert(height(T)"
     " == 2); T(:, 'b') = []; assert(width(T) == 1); T.a = []; assert(width(T) == 0);"),
    ("T = table([1; 2], [3; 4], 'VariableNames', {'a', 'b'}); T2 = [T; T]; assert(height(T2) == "
     "4); T3 = [T table([5; 6], 'VariableNames', {'c'})]; assert(width(T3) == 3); threw = false; "
     'try, [T T]; catch, threw = true; end; assert(threw);'),
    ("T = table([1; 2], [3; 4], 'VariableNames', {'a', 'b'}); T4 = [T; table([9; 8], [7; 6], "
     "'VariableNames', {'b', 'a'})]; assert(isequal(T4.a, [1; 2; 7; 6])); threw = false; try, [T; "
     "table(1, 'VariableNames', {'z'})]; catch, threw = true; end; assert(threw);"),
    ("a = [1; 2]; b = {'x'; 'y'}; T = table(a, b); assert(isequal(T.Properties.VariableNames, "
     "{'a', 'b'})); T = table(a, a * 2); assert(isequal(T.Properties.VariableNames, {'a', "
     "'Var2'})); E = table; assert(isequal(size(E), [0 0]) && istable(E));"),
    ("T = table('Size', [2 3], 'VariableTypes', {'double', 'string', 'logical'}, 'VariableNames', "
     "{'x', 's', 'l'}); assert(isequal(size(T), [2 3]) && isstring(T.s) && islogical(T.l) && "
     'all(T.x == 0));'),
    ('T = table([1 2; 3 4], ["a"; "b"], \'VariableNames\', {\'M\', \'S\'}); assert(isequal(size(T.M), [2'
     " 2])); assert(isequal(T.M(2, :), [3 4])); assert(isequal(T{1, 'M'}, [1 2])); assert(T.S(2) "
     '== "b"); assert(isequal(size(T), [2 2]));'),
    ("T = table([1; 2], table([3; 4], [5; 6], 'VariableNames', {'p', 'q'}), 'VariableNames', {'a',"
     " 'nested'}); assert(istable(T.nested) && isequal(T.nested.q, [5; 6])); assert(T.nested.p(2) "
     '== 4); U = T(2, :); assert(U.nested.q == 6);'),
    ("T = table([3; 1; 2], {'c'; 'a'; 'b'}, 'VariableNames', {'n', 's'}); assert(isequal(T(:, "
     "1).Properties.VariableNames, {'n'})); assert(isequal(T(:, [true false]).n, [3; 1; 2])); "
     "assert(isequal(T(:, vartype('numeric')).Properties.VariableNames, {'n'}));"),
    ('T = table([3; 1; 2], {\'c\'; \'a\'; \'b\'}, ["x"; "y"; "z"], \'VariableNames\', {\'n\', \'s\', \'t\'}); '
     "assert(isequal(T(:, vartype('cellstr')).Properties.VariableNames, {'s'})); assert(width(T(:,"
     " vartype('string'))) == 1 || width(T(:, vartype('char'))) == 1);"),
    ("T = table([1; 2], [3; 4], 'VariableNames', {'a', 'b'}); T.Properties.VariableUnits = {'m', "
     "'s'}; assert(isequal(T.Properties.VariableUnits, {'m', 's'})); "
     "T.Properties.VariableDescriptions = {'first', 'second'}; "
     "assert(strcmp(T.Properties.VariableDescriptions{2}, 'second'));"),
    ("T = table([1; 2], [3; 4], 'VariableNames', {'a', 'b'}); T.Properties.Description = 'my "
     "table'; assert(strcmp(T.Properties.Description, 'my table')); T.Properties.UserData = "
     "struct('k', 1); assert(T.Properties.UserData.k == 1);"),
    ("T = table([1; 2], [3; 4], 'VariableNames', {'a', 'b'}); T.Properties.DimensionNames = "
     "{'Obs', 'Vars'}; assert(isequal(T.Properties.DimensionNames, {'Obs', 'Vars'})); "
     "assert(isequal(T.Vars, [1 3; 2 4])); T.Properties.RowNames = {'x', 'y'}; "
     "assert(isequal(T.Obs, {'x'; 'y'}));"),
    ("T = table([1; 2], [3; 4], 'VariableNames', {'a', 'b'}); T.Properties.VariableNames{2} = 'z';"
     " assert(isequal(T.Properties.VariableNames, {'a', 'z'})); T.Properties.VariableNames = {'p',"
     " 'q'}; assert(T.q(1) == 3); T.Properties.VariableUnits = {'m', 's'}; U = T(:, 'q'); "
     "assert(isequal(U.Properties.VariableUnits, {'s'}));"),
    ("T = table([1; 2], 'VariableNames', {'a'}); T = addprop(T, {'Source', 'Precision'}, {'table',"
     " 'variable'}); T.Properties.CustomProperties.Source = 'lab'; "
     "assert(strcmp(T.Properties.CustomProperties.Source, 'lab')); T = rmprop(T, 'Source'); "
     "assert(~isfield(T.Properties.CustomProperties, 'Source'));"),
    ("T = table([1; 2], [3; 4], 'VariableNames', {'a', 'b'}); T.Properties.VariableUnits = {'m', "
     "'s'}; T2 = renamevars(T, 'a', 'aa'); assert(isequal(T2.Properties.VariableNames, {'aa', "
     "'b'})); T3 = sortrows(T, 'a', 'descend'); assert(isequal(T3.Properties.VariableUnits, {'m', "
     "'s'}));"),
    ("T = table([1; 2; 3], {'a'; 'bcd'; 'c'}, [true; false; true], 'VariableNames', {'N', 'S', "
     "'L'}); s = evalc('T'); assert(strcmp(s, sprintf('\\nT =\\n\\n  3×3 table\\n\\n    N       S"
     "         L  \\n    _    _______    _____\\n\\n    1    {''a''  }    true \\n    2    {''bcd''}"
     "    false\\n    3    {''c''  }    true \\n\\n')), s);"),
    ("T = table([38; 43], [71; 69], [176; 163], 'VariableNames', {'Age', 'Height', 'Weight'}, "
     "'RowNames', {'Sanchez', 'Li'}); s = evalc('disp(T)'); assert(strcmp(s, sprintf('"
     '               Age    Height    Weight\\n               ___    ______    ______\\n\\n    '
     "Sanchez    38       71       176  \\n    Li         43       69       163  \\n\\n')), s);"),
    ('T = table([0.5; 1000.123; NaN], ["x"; "yy"; missing], \'VariableNames\', {\'X\', \'Str\'}); s = '
     "evalc('disp(T)'); assert(strcmp(s, sprintf('      X          Str   \\n    ______    "
     '_________\\n\\n       0.5    "x"      \\n    1000.1    "yy"     \\n       NaN    '
     "<missing>\\n\\n')), s);"),
    ("T = table([124 93; 109 77], 'VariableNames', {'BloodPressure'}); s = evalc('disp(T)'); "
     "assert(strcmp(s, sprintf('    BloodPressure\\n    _____________\\n\\n     124     93  \\n     "
     "109     77  \\n\\n')), s);"),
    ("s = evalc('E = table'); assert(strcmp(s, sprintf('\\nE =\\n\\n  0×0 empty table\\n\\n')), s); T ="
     " table(datetime(2024, 1, 1:2)', categorical({'red'; 'blue'}), 'VariableNames', {'when', "
     "'color'}); s = evalc('disp(T)'); assert(contains(s, '01-Jan-2024    red ') && contains(s, "
     "'02-Jan-2024    blue'));"),
    ("T = table([1; 2], [10; 20], [30; 40], 'VariableNames', {'id', 'a', 'b'}); S = stack(T, {'a',"
     " 'b'}); assert(height(S) == 4); assert(isequal(S.Properties.VariableNames, {'id', "
     "'a_b_Indicator', 'a_b'})); assert(isequal(S.a_b, [10; 30; 20; 40])); assert(isequal(S.id, "
     '[1; 1; 2; 2]));'),
    ("T = table([1; 2], [10; 20], [30; 40], 'VariableNames', {'id', 'a', 'b'}); S = stack(T, {'a',"
     " 'b'}); assert(iscategorical(S.a_b_Indicator) && isequal(cellstr(S.a_b_Indicator), {'a'; "
     "'b'; 'a'; 'b'})); U = unstack(S, 'a_b', 'a_b_Indicator'); assert(height(U) == 2 && "
     'isequal(U.a, [10; 20]) && isequal(U.b, [30; 40]));'),
    ("T = table([1; 2], [10; 20], [30; 40], 'VariableNames', {'id', 'a', 'b'}); S = stack(T, 2:3, "
     "'NewDataVariableName', 'val', 'IndexVariableName', 'which'); "
     "assert(isequal(S.Properties.VariableNames, {'id', 'which', 'val'})); [S2, iu] = stack(T, "
     "{'a', 'b'}); assert(isequal(iu, [1; 1; 2; 2]));"),
    ("S = table({'x'; 'x'; 'y'}, {'p'; 'q'; 'p'}, [1; 2; 3], 'VariableNames', {'g', 'k', 'v'}); U "
     "= unstack(S, 'v', 'k'); assert(isequal(U.Properties.VariableNames, {'g', 'p', 'q'})); "
     'assert(isequaln(U.p, [1; 3]) && isequaln(U.q, [2; NaN]));'),
    ("S = table([1; 1; 1; 2], {'p'; 'p'; 'q'; 'p'}, [1; 2; 3; 4], 'VariableNames', {'g', 'k', "
     "'v'}); U = unstack(S, 'v', 'k'); assert(isequaln(U.p, [3; 4])); U = unstack(S, 'v', 'k', "
     "'AggregationFunction', @mean); assert(isequaln(U.p, [1.5; 4]));"),
    ("T = table([1; 2], [10; 20], [30; 40], 'VariableNames', {'id', 'a', 'b'}); R = rowfun(@(a, b)"
     " a + b, T, 'InputVariables', {'a', 'b'}, 'OutputVariableNames', 'sum'); "
     'assert(isequal(R.sum, [40; 60]));'),
    ("T = table([1; 2], [10; 20], 'VariableNames', {'a', 'b'}); R = rowfun(@(a, b) a * b, T); "
     "assert(isequal(R.Var1, [10; 40])); r = rowfun(@(a, b) a * b, T, 'OutputFormat', 'uniform'); "
     "assert(isequal(r, [10; 40])); c = rowfun(@(a, b) a * b, T, 'OutputFormat', 'cell'); "
     'assert(iscell(c) && c{2} == 40);'),
    ("T = table([1; 2], [10; 20], 'VariableNames', {'a', 'b'}); R = rowfun(@(a, b) deal(a + b, a -"
     " b), T, 'NumOutputs', 2, 'OutputVariableNames', {'s', 'd'}); assert(isequal(R.d, [-9; "
     "-18])); R = rowfun(@(x) sum(x), T, 'SeparateInputs', false); assert(isequal(R.Var1, [11; "
     '22]));'),
    ("T = table({'a'; 'b'; 'a'}, [1; 2; 3], 'VariableNames', {'g', 'x'}); R = rowfun(@sum, T, "
     "'GroupingVariables', 'g', 'InputVariables', 'x', 'OutputVariableNames', 'total'); "
     'assert(isequal(R.total, [4; 2]) && isequal(R.GroupCount, [2; 1]));'),
    ("T = table([1; 2; 3], [4; 5; 6], 'VariableNames', {'x', 'y'}); V = varfun(@mean, T); "
     "assert(isequal(V.Properties.VariableNames, {'mean_x', 'mean_y'}) && V.mean_y == 5); V = "
     'varfun(@(v) v * 2, T); assert(isequal(V.Fun_x, [2; 4; 6]));'),
    ("T = table([1; 2; 3], [4; 5; 6], 'VariableNames', {'x', 'y'}); v = varfun(@mean, T, "
     "'OutputFormat', 'uniform'); assert(isequal(v, [2 5])); c = varfun(@(v) v', T, "
     "'OutputFormat', 'cell'); assert(iscell(c) && isequal(c{2}, [4 5 6])); V = varfun(@max, T, "
     "'InputVariables', 'y'); assert(V.max_y == 6);"),
    ("T = table({'a'; 'b'; 'a'; 'b'}, [1; 2; 3; 4], 'VariableNames', {'g', 'x'}); V = "
     "varfun(@mean, T, 'InputVariables', 'x', 'GroupingVariables', 'g'); assert(isequal(V.mean_x, "
     "[2; 3]) && isequal(V.GroupCount, [2; 2])); V = varfun(@mean, T, 'InputVariables', "
     '@isnumeric); assert(V.mean_x == 2.5);'),
    ("g = {'a'; 'b'; 'a'; 'b'}; x = [1; 2; 3; 4]; T = table(g, x); S = groupsummary(T, 'g', "
     "'mean', 'x'); assert(isequal(S.mean_x, [2; 3])); assert(isequal(S.GroupCount, [2; 2])); C = "
     "groupcounts(T, 'g'); assert(isequal(C.GroupCount, [2; 2]) && isequal(C.Percent, [50; 50]));"),
    ("g = {'a'; 'b'; 'a'; 'b'}; x = [1; 2; 3; 4]; T = table(g, x); S = groupsummary(T, 'g', "
     "{'sum', 'max'}, 'x'); assert(isequal(S.Properties.VariableNames, {'g', 'GroupCount', "
     "'sum_x', 'max_x'}) && isequal(S.max_x, [3; 4])); S = groupsummary(T, 'g'); assert(width(S) "
     '== 2);'),
    ("g = {'a'; 'b'; 'a'; 'b'}; x = [1; 2; 3; 4]; T = table(g, x); S = groupsummary(T, 'g', @(v) "
     "max(v) - min(v), 'x'); assert(isequal(S.fun1_x, [2; 2])); S = groupsummary(T, 'g', 'all', "
     "'x'); assert(any(strcmp(S.Properties.VariableNames, 'median_x')) && "
     "any(strcmp(S.Properties.VariableNames, 'std_x')));"),
    ("T = table([1; 5; 7; 12], [10; 20; 30; 40], 'VariableNames', {'x', 'y'}); S = groupsummary(T,"
     " 'x', [0 6 12], 'sum', 'y'); assert(isequal(S.sum_y, [30; 70])); "
     "assert(isequal(S.Properties.VariableNames, {'disc_x', 'GroupCount', 'sum_y'})); "
     "assert(isequal(cellstr(S.disc_x), {'[0, 6)'; '[6, 12]'}));"),
    ("T = table(datetime(2024, [1; 1; 2], [5; 20; 3]), [1; 2; 3], 'VariableNames', {'d', 'v'}); S "
     "= groupsummary(T, 'd', 'month', 'sum', 'v'); assert(isequal(S.sum_v, [3; 3])); "
     "assert(isequal(S.Properties.VariableNames{1}, 'month_d'));"),
    ("T = table({'a'; 'b'; 'a'}, [1; 1; 2], [10; 20; 30], 'VariableNames', {'g', 'h', 'v'}); S = "
     "groupsummary(T, {'g', 'h'}, 'sum', 'v'); assert(height(S) == 3 && isequal(S.sum_v, [10; 30; "
     "20])); [B, BG] = groupsummary([10; 20; 30], [1; 2; 1], 'sum'); assert(isequal(B, [40; 20]) "
     '&& isequal(BG, [1; 2]));'),
    ("T = table({'a'; 'b'; 'a'; 'c'}, [1; 2; 3; 4], 'VariableNames', {'g', 'x'}); F = "
     "groupfilter(T, 'g', @(v) numel(v) > 1); assert(height(F) == 2 && isequal(F.x, [1; 3])); F = "
     "groupfilter(T, 'g', @(v) v > 1, 'x'); assert(isequal(F.x, [2; 3; 4]));"),
    ("T = table({'a'; 'b'; 'a'; 'b'}, [1; 2; 3; 6], 'VariableNames', {'g', 'x'}); G = "
     "grouptransform(T, 'g', 'meancenter'); assert(isequal(G.x, [-1; -2; 1; 2])); G = "
     "grouptransform(T, 'g', @(v) v / sum(v), 'x'); assert(isequal(G.x, [0.25; 0.25; 0.75; "
     '0.75]));'),
    ("T = table({'a'; 'b'; 'a'; 'b'}, [1; 2; 3; 6], 'VariableNames', {'g', 'x'}); G = "
     "grouptransform(T, 'g', 'rescale', 'x', 'ReplaceValues', false); assert(isequal(G.rescale_x, "
     "[0; 0; 1; 1]) && isequal(G.x, T.x)); G = grouptransform(T, 'g', 'zscore'); assert(abs(G.x(1)"
     ' + 0.7071) < 1e-3);'),
    ('[c, bg] = groupcounts([1; 2; 1; 3; 1]); assert(isequal(c, [3; 1; 1]) && isequal(bg, [1; 2; '
     "3])); c = groupcounts({'a'; 'b'; 'a'}); assert(isequal(c, [2; 1])); [G, names] = "
     "findgroups({'a'; 'b'; 'a'}); assert(isequal(G, [1; 2; 1])); assert(isequal(splitapply(@sum, "
     '[1; 2; 3], G), [4; 2]));'),
    ("T = table({'a'; 'b'; 'a'; 'b'}, {'x'; 'x'; 'y'; 'y'}, [1; 2; 3; 4], 'VariableNames', {'r', "
     "'c', 'v'}); P = pivot(T, Rows='r', Columns='c'); assert(isequal(P.Properties.VariableNames, "
     "{'r', 'x', 'y'}) && isequal(P.x, [1; 1])); P = pivot(T, Rows='r', Columns='c', "
     "DataVariable='v'); assert(isequal(P.y, [3; 4]));"),
    ("T = table({'a'; 'b'; 'a'; 'b'}, {'x'; 'x'; 'y'; 'y'}, [1; 2; 3; 4], 'VariableNames', {'r', "
     "'c', 'v'}); P = pivot(T, Rows='r', DataVariable='v', Method='mean'); assert(isequal(P.mean, "
     "[2; 3])); P = pivot(T, Columns='c', DataVariable='v', Method='max'); assert(P.x == 2 && P.y "
     '== 4);'),
    ("T = table([1; 2], [3; 4], [5; 6], 'VariableNames', {'a', 'b', 'c'}); M = mergevars(T, {'a', "
     "'b'}); assert(isequal(M.Properties.VariableNames, {'Var1', 'c'}) && isequal(M.Var1, [1 3; 2 "
     "4])); M = mergevars(T, {'b', 'c'}, 'NewVariableName', 'bc'); assert(isequal(M.bc, [3 5; 4 "
     '6]));'),
    ("T = table([1; 2], [3; 4], [5; 6], 'VariableNames', {'a', 'b', 'c'}); M = mergevars(T, {'a', "
     "'b'}, 'MergeAsTable', true, 'NewVariableName', 'ab'); assert(istable(M.ab) && "
     "isequal(M.ab.b, [3; 4])); S = splitvars(M); assert(isequal(S.Properties.VariableNames, {'a',"
     " 'b', 'c'}));"),
    ("T = table([1 2; 3 4], [5; 6], 'VariableNames', {'m', 'c'}); S = splitvars(T); "
     "assert(isequal(S.Properties.VariableNames, {'m_1', 'm_2', 'c'}) && isequal(S.m_2, [2; 4])); "
     "S = splitvars(T, 'm', 'NewVariableNames', {'p', 'q'}); assert(isequal(S.q, [2; 4]));"),
    ("T = table([1; 2], [3; 4], [5; 6], 'VariableNames', {'a', 'b', 'c'}); M = movevars(T, 'c', "
     "'Before', 'a'); assert(isequal(M.Properties.VariableNames, {'c', 'a', 'b'})); M = "
     "movevars(T, 'a', 'After', 'c'); assert(isequal(M.Properties.VariableNames, {'b', 'c', "
     "'a'})); M = movevars(T, {'a', 'b'}, 'After', 'c'); "
     "assert(strcmp(M.Properties.VariableNames{1}, 'c'));"),
    ("T = table([1; 2], {'3'; '4'}, 'VariableNames', {'a', 'b'}); C = convertvars(T, 'b', "
     "'string'); assert(isstring(C.b)); C = convertvars(T, 'a', @(x) x * 10); assert(isequal(C.a, "
     "[10; 20])); C = convertvars(T, {'a'}, 'single'); assert(isa(C.a, 'single')); C = "
     "convertvars(T, @iscellstr, 'categorical'); assert(iscategorical(C.b));"),
    ("T = table([1; 2], [3; 4], 'VariableNames', {'a', 'b'}); A = addvars(T, [7; 8], "
     "'NewVariableNames', 'y'); assert(isequal(A.Properties.VariableNames, {'a', 'b', 'y'})); A = "
     "addvars(T, [7; 8], 'Before', 'a', 'NewVariableNames', 'y'); "
     "assert(strcmp(A.Properties.VariableNames{1}, 'y'));"),
    ("T = table([1; 2], [3; 4], 'VariableNames', {'a', 'b'}); A = addvars(T, [7; 8], [9; 10], "
     "'After', 'a', 'NewVariableNames', {'y', 'z'}); assert(isequal(A.Properties.VariableNames, "
     "{'a', 'y', 'z', 'b'})); w = [5; 6]; A = addvars(T, w); "
     "assert(isequal(A.Properties.VariableNames, {'a', 'b', 'w'}));"),
    ("T = table([1; 2], [3; 4], 'VariableNames', {'a', 'b'}); R = removevars(T, 'a'); "
     "assert(isequal(R.Properties.VariableNames, {'b'})); R = removevars(T, {'a', 'b'}); "
     "assert(width(R) == 0); R = renamevars(T, {'a', 'b'}, {'x', 'y'}); "
     "assert(isequal(R.Properties.VariableNames, {'x', 'y'}));"),
    ("T = table([1; 2], [3; 4], 'VariableNames', {'a', 'b'}, 'RowNames', {'r1', 'r2'}); V = "
     "rows2vars(T); assert(isequal(V.Properties.VariableNames, {'OriginalVariableNames', 'r1', "
     "'r2'}) && isequal(V.r2, [2; 4]) && isequal(V.OriginalVariableNames, {'a'; 'b'}));"),
    ("T = table({'k1'; 'k2'}, [3; 4], 'VariableNames', {'key', 'b'}); V = rows2vars(T, "
     "'VariableNamesSource', 'key'); assert(isequal(V.Properties.VariableNames, "
     "{'OriginalVariableNames', 'k1', 'k2'}) && V.k2 == 4); T = table([1; 2], [3; 4]); V = "
     'rows2vars(T); assert(isequal(V.Var1, [1; 3]));'),
    ("T = table(table([1; 2], [3; 4], 'VariableNames', {'x', 'y'}), table([5; 6], [7; 8], "
     "'VariableNames', {'x', 'y'}), 'VariableNames', {'A', 'B'}); I = inner2outer(T); "
     "assert(isequal(I.Properties.VariableNames, {'x', 'y'}) && isequal(I.x.B, [5; 6]));"),
    ("T = table([2; 1; 2; 1], [4; 3; 1; 2], {'a'; 'b'; 'c'; 'd'}, 'VariableNames', {'p', 'q', "
     "'s'}); S = sortrows(T, {'p', 'q'}); assert(isequal(S.q, [2; 3; 1; 4])); S = sortrows(T, "
     "{'p', 'q'}, {'ascend', 'descend'}); assert(isequal(S.q, [3; 2; 4; 1])); [S, i] = sortrows(T,"
     " 'q'); assert(isequal(i, [3; 4; 2; 1]));"),
    ("T = table([2; 1; 2; 1], [4; 3; 1; 2], {'a'; 'b'; 'c'; 'd'}, 'VariableNames', {'p', 'q', "
     "'s'}); S = sortrows(T, [1 -2]); assert(isequal(S.q, [3; 2; 4; 1])); S = sortrows(T, 's', "
     "'descend'); assert(isequal(S.s, {'d'; 'c'; 'b'; 'a'})); S = sortrows(T); assert(isequal(S.q,"
     ' [2; 3; 1; 4]));'),
    ("T = table([2; NaN; 1], 'VariableNames', {'x'}); S = sortrows(T, 'x'); assert(isequaln(S.x, "
     "[1; 2; NaN])); S = sortrows(T, 'x', 'descend'); assert(isequaln(S.x, [NaN; 2; 1])); S = "
     "sortrows(T, 'x', 'MissingPlacement', 'first'); assert(isequaln(S.x, [NaN; 1; 2]));"),
    ("T = table([3; 1; 2], 'VariableNames', {'x'}, 'RowNames', {'c', 'a', 'b'}); S = sortrows(T, "
     "'RowNames'); assert(isequal(S.x, [1; 2; 3])); K = topkrows(T, 2); assert(isequal(K.x, [3; "
     "2])); K = topkrows(T, 2, 'x', 'ascend'); assert(isequal(K.x, [1; 2])); "
     "assert(issortedrows(sortrows(T, 'x'), 'x'));"),
    ("T = table([1; 2; 1], {'a'; 'b'; 'a'}, 'VariableNames', {'n', 's'}); [U, ia, ic] = unique(T);"
     ' assert(height(U) == 2 && isequal(ia, [1; 2]) && isequal(ic, [1; 2; 1])); U = unique(T(:, '
     "'s')); assert(isequal(U.s, {'a'; 'b'})); U = unique(T, 'stable'); assert(isequal(U.n, [1; "
     '2]));'),
    ("A = table([1; 2; 3], {'a'; 'b'; 'c'}, 'VariableNames', {'n', 's'}); B = table([3; 4], {'c'; "
     "'d'}, 'VariableNames', {'n', 's'}); assert(isequal(ismember(A, B), [false; false; true])); I"
     ' = intersect(A, B); assert(height(I) == 1 && I.n == 3); U = union(A, B); assert(height(U) =='
     ' 4);'),
    ("A = table([1; 2; 3], {'a'; 'b'; 'c'}, 'VariableNames', {'n', 's'}); B = table([3; 4], {'c'; "
     "'d'}, 'VariableNames', {'n', 's'}); D = setdiff(A, B); assert(isequal(D.n, [1; 2])); X = "
     'setxor(A, B); assert(isequal(X.n, [1; 2; 4])); [tf, loc] = ismember(A, B); '
     'assert(isequal(loc, [0; 0; 1]));'),
    ('T = table([1; NaN; 3], {\'a\'; \'\'; \'c\'}, ["x"; "y"; missing], \'VariableNames\', {\'n\', \'s\', '
     "'t'}); M = ismissing(T); assert(isequal(M, [false false false; true true false; false false "
     'true])); R = rmmissing(T); assert(height(R) == 1);'),
    ("T = table([1; NaN; 3], {'a'; ''; 'c'}, 'VariableNames', {'n', 's'}); F = fillmissing(T, "
     "'constant', 0, 'DataVariables', 'n'); assert(isequal(F.n, [1; 0; 3])); F = fillmissing(T, "
     "'previous'); assert(F.n(2) == 1); S = standardizeMissing(T, 3); assert(isnan(S.n(3)));"),
    ("T = table([1; -99; 3], {'a'; 'NA'; 'c'}, 'VariableNames', {'n', 's'}); S = "
     "standardizeMissing(T, {-99, 'NA'}); assert(isnan(S.n(2)) && isempty(S.s{2})); "
     "assert(isequal(ismissing(T, {-99, 'NA'}), [false false; true true; false false]));"),
    ("T = table([1; 2; 3; 4; 5; 6; 7; 8; 9; 10], 'VariableNames', {'x'}); H = head(T); "
     'assert(height(H) == 8); H = head(T, 3); assert(isequal(H.x, [1; 2; 3])); L = tail(T, 2); '
     'assert(isequal(L.x, [9; 10])); L = tail(T); assert(height(L) == 8);'),
    ("T = table([1; 2; 3], {'a'; 'b'; 'c'}, [true; false; true], 'VariableNames', {'N', 'S', "
     "'L'}); s = evalc('summary(T)'); assert(contains(s, 'Variables:') && contains(s, 'N: 3×1 "
     "double') && contains(s, 'Median') && contains(s, 'cell array of character vectors') && "
     "contains(s, 'True'));"),
    ("T = table([1; 2; 3], 'VariableNames', {'N'}); T.Properties.VariableUnits = {'kg'}; s = "
     "evalc('summary(T)'); assert(contains(s, 'Units') && contains(s, 'kg')); st = summary(T); "
     "assert(st.N.Min == 1 && st.N.Max == 3 && st.N.Median == 2 && strcmp(st.N.Type, 'double'));"),
    ("A = array2table([1 2; 3 4]); assert(isequal(A.Properties.VariableNames, {'Var1', 'Var2'})); "
     "A = array2table([1 2; 3 4], 'VariableNames', {'p', 'q'}, 'RowNames', {'r1', 'r2'}); "
     "assert(A{'r2', 'q'} == 4); m = [1 2; 3 4]; A = array2table(m); "
     "assert(isequal(A.Properties.VariableNames, {'m1', 'm2'}));"),
    ("K = cell2table({1, 'a'; 2, 'b'}, 'VariableNames', {'n', 'c'}); assert(isequal(K.n, [1; 2]) "
     '&& isequal(K.c, {\'a\'; \'b\'})); K = cell2table({1, "x"; 2, "y"}); assert(isstring(K.Var2)); c '
     "= {1, 2; 3, 4}; K = cell2table(c); assert(isequal(K.Properties.VariableNames, {'c1', "
     "'c2'}));"),
    ("S = struct('a', {1, 2, 3}, 'b', {'x', 'y', 'z'}); T = struct2table(S); assert(height(T) == 3"
     " && isequal(T.a, [1; 2; 3]) && isequal(T.b, {'x'; 'y'; 'z'})); S = struct('a', [1; 2], 'b', "
     "{{'x'; 'y'}}); T = struct2table(S); assert(height(T) == 2 && isequal(T.b, {'x'; 'y'}));"),
    ("S = struct('a', [1 2 3], 'b', 'text'); T = struct2table(S, 'AsArray', true); "
     "assert(height(T) == 1 && isequal(T.a, [1 2 3])); T = table([1; 2], {'x'; 'y'}, "
     "'VariableNames', {'a', 'b'}); S = table2struct(T); assert(numel(S) == 2 && S(2).a == 2 && "
     "strcmp(S(2).b, 'y'));"),
    ("T = table([1; 2], {'x'; 'y'}, 'VariableNames', {'a', 'b'}); S = table2struct(T, 'ToScalar', "
     'true); assert(isequal(S.a, [1; 2]) && iscell(S.b)); C = table2cell(T); assert(isequal(C, {1,'
     " 'x'; 2, 'y'})); assert(isequal(table2array(T(:, 'a')), [1; 2]));"),
    ("A = table([1; 2; 3], {'a'; 'b'; 'c'}, 'VariableNames', {'k', 'va'}); B = table([2; 3; 4], "
     "[20; 30; 40], 'VariableNames', {'k', 'vb'}); J = innerjoin(A, B); assert(isequal(J.k, [2; "
     "3]) && isequal(J.vb, [20; 30])); O = outerjoin(A, B, 'MergeKeys', true); assert(height(O) =="
     ' 4); L = join(A(2:3, :), B); assert(isequal(L.vb, [20; 30]));'),
]


TIMETABLES = [
    ("t = datetime(2024, 1, 1) + hours(0:3)'; TT = timetable(t, [1; 2; 3; 4], 'VariableNames', "
     "{'v'}); assert(height(TT) == 4); R = retime(TT, 'daily', 'sum'); assert(R.v(1) == 10); "
     'assert(istimetable(TT) && ~istable(TT) || istimetable(TT));'),
    ("t = datetime(2024, 1, 1) + hours(0:3)'; TT = timetable(t, [1; 2; 3; 4], {'a'; 'b'; 'c'; "
     "'d'}, 'VariableNames', {'v', 's'}); assert(isequal(size(TT), [4 2])); assert(isequal(TT.v, "
     '[1; 2; 3; 4])); assert(isequal(TT.Time, t)); assert(isequal(TT.Properties.RowTimes, t));'),
    ("TT = timetable([1; 2; 3], 'RowTimes', datetime(2024, 1, 1:3)'); "
     "assert(isequal(TT.Properties.VariableNames, {'Var1'})); "
     "assert(isequal(TT.Properties.DimensionNames, {'Time', 'Variables'})); assert(TT.Time(2) == "
     'datetime(2024, 1, 2));'),
    ("TT = timetable([1; 2; 3], 'SampleRate', 100); assert(isduration(TT.Time)); "
     'assert(isequal(seconds(TT.Time), [0; 0.01; 0.02])); assert(TT.Properties.SampleRate == 100);'
     ' assert(seconds(TT.Properties.TimeStep) == 0.01);'),
    ("TT = timetable([1; 2; 3], 'TimeStep', minutes(5)); assert(isequal(minutes(TT.Time), [0; 5; "
     "10])); TT = timetable([1; 2; 3], 'TimeStep', hours(1), 'StartTime', datetime(2024, 1, 1)); "
     'assert(TT.Time(3) == datetime(2024, 1, 1, 2, 0, 0));'),
    ("TT = timetable([1; 2; 3], 'TimeStep', calmonths(1), 'StartTime', datetime(2024, 1, 31)); "
     'assert(TT.Time(2) == datetime(2024, 2, 29)); assert(TT.Properties.StartTime == '
     'datetime(2024, 1, 31));'),
    ('TT = timetable(seconds([0; 1; 2]), [5; 6; 7]); assert(isequal(TT.Var1, [5; 6; 7])); '
     "assert(isduration(TT.Properties.RowTimes)); x = [7; 8; 9]; tm = datetime(2024, 1, 1:3)'; TT "
     "= timetable(tm, x); assert(isequal(TT.Properties.VariableNames, {'x'}) && isequal(TT.tm, "
     'tm));'),
    ("TT = timetable('Size', [2 2], 'VariableTypes', {'double', 'string'}, 'RowTimes', "
     "datetime(2024, 1, 1:2)', 'VariableNames', {'a', 'b'}); assert(isequal(size(TT), [2 2]) && "
     'isstring(TT.b));'),
    ("T = table(datetime(2024, 1, 1:3)', [1; 2; 3], 'VariableNames', {'when', 'v'}); TT = "
     'table2timetable(T); assert(istimetable(TT) && width(TT) == 1 && '
     "isequal(TT.Properties.DimensionNames{1}, 'when')); T2 = timetable2table(TT); "
     "assert(istable(T2) && isequal(T2.Properties.VariableNames, {'when', 'v'}));"),
    ("T = table([1; 2; 3], 'VariableNames', {'v'}); TT = table2timetable(T, 'RowTimes', "
     "datetime(2024, 1, 1:3)'); assert(TT.Time(1) == datetime(2024, 1, 1)); TT = "
     "table2timetable(T, 'SampleRate', 1); assert(isequal(seconds(TT.Time), [0; 1; 2])); T3 = "
     "timetable2table(TT, 'ConvertRowTimes', false); assert(width(T3) == 1);"),
    ("TT = array2timetable([1 2; 3 4], 'RowTimes', datetime(2024, 1, 1:2)'); "
     "assert(isequal(TT.Properties.VariableNames, {'Var1', 'Var2'})); TT = array2timetable([1 2; 3"
     " 4], 'TimeStep', hours(1), 'VariableNames', {'a', 'b'}); assert(isequal(TT.b, [2; 4]));"),
    ("t = datetime(2024, 1, 1) + days(0:4)'; TT = timetable(t, (1:5)', 'VariableNames', {'v'}); S "
     '= TT(timerange(datetime(2024, 1, 2), datetime(2024, 1, 4)), :); assert(isequal(S.v, [2; '
     "3])); S = TT(timerange(datetime(2024, 1, 2), datetime(2024, 1, 4), 'closed'), :); "
     'assert(isequal(S.v, [2; 3; 4]));'),
    ("t = datetime(2024, 1, 1) + days(0:4)'; TT = timetable(t, (1:5)', 'VariableNames', {'v'}); S "
     "= TT(timerange('2024-01-03', '2024-01-05'), :); assert(isequal(S.v, [3; 4])); S = "
     'TT(timerange(datetime(2024, 1, 4), Inf), :); assert(isequal(S.v, [4; 5])); S = '
     'TT(timerange(-Inf, datetime(2024, 1, 2)), :); assert(isequal(S.v, 1));'),
    ("t = datetime(2024, 1, 1) + days(0:4)'; TT = timetable(t, (1:5)', 'VariableNames', {'v'}); S "
     "= TT(datetime(2024, 1, 3), :); assert(S.v == 3); S = TT('2024-01-02', :); assert(S.v == 2); "
     'S = TT([1 5], :); assert(isequal(S.v, [1; 5])); S = TT(TT.v > 3, :); assert(height(S) == 2);'),
    ("t = datetime(2024, 1, 1) + days(0:4)'; TT = timetable(t, (1:5)', 'VariableNames', {'v'}); S "
     '= TT(withtol(datetime(2024, 1, 3, 0, 0, 5), seconds(10)), :); assert(S.v == 3); assert(TT{2,'
     " 'v'} == 2); assert(isequal(TT{:, :}, (1:5)')); assert(isequal(TT.Variables, (1:5)'));"),
    ("t = datetime(2024, 1, 1) + days(0:4)'; TT = timetable(t, (1:5)', 'VariableNames', {'v'}); "
     'TT.w = TT.v * 2; assert(width(TT) == 2); TT(2, :) = []; assert(height(TT) == 4 && TT.Time(2)'
     " == datetime(2024, 1, 3)); U = TT(:, 'w'); assert(istimetable(U) && isequal(U.w, [2; 6; 8; "
     '10]));'),
    ("t = datetime(2024, 1, 1) + days(0:2)'; A = timetable(t, [1; 2; 3], 'VariableNames', {'a'}); "
     "B = timetable(t, [4; 5; 6], 'VariableNames', {'b'}); C = [A B]; assert(isequal(C.b, [4; 5; "
     '6])); D = [A; A]; assert(height(D) == 6); S = sortrows(D); assert(isequal(S.a, [1; 1; 2; 2; '
     '3; 3]));'),
    ("t = datetime(2024, 1, [3 1 2])'; TT = timetable(t, [3; 1; 2], 'VariableNames', {'v'}); S = "
     "sortrows(TT); assert(isequal(S.v, [1; 2; 3])); S = sortrows(TT, 'Time', 'descend'); "
     'assert(isequal(S.v, [3; 2; 1])); assert(~issorted(TT) && issorted(sortrows(TT))); S = '
     "sortrows(TT, 'v'); assert(S.Time(1) == datetime(2024, 1, 1));"),
    ("t = datetime(2024, 1, 1) + days(0:4)'; TT = timetable(t, (1:5)', 'VariableNames', {'v'}); H "
     '= head(TT, 2); assert(istimetable(H) && height(H) == 2); TT.Time = TT.Time + days(1); '
     "assert(TT.Time(1) == datetime(2024, 1, 2)); TT.Properties.DimensionNames{1} = 'Date'; "
     'assert(TT.Date(1) == datetime(2024, 1, 2));'),
    ("t = datetime(2024, 1, 1) + hours([0 6 12 30])'; TT = timetable(t, [1; 2; 3; 4], "
     "'VariableNames', {'v'}); R = retime(TT, 'daily', 'sum'); assert(isequal(R.v, [6; 4])); "
     "assert(isequal(R.Time, datetime(2024, 1, 1:2)')); R = retime(TT, 'daily', 'mean'); "
     'assert(isequal(R.v, [2; 4]));'),
    ("t = datetime(2024, 1, 1) + hours([0 6 12 30])'; TT = timetable(t, [1; 2; 3; 4], "
     "'VariableNames', {'v'}); R = retime(TT, 'daily', 'count'); assert(isequal(R.v, [3; 1])); R ="
     " retime(TT, 'daily', 'max'); assert(isequal(R.v, [3; 4])); R = retime(TT, 'daily', "
     "'firstvalue'); assert(isequal(R.v, [1; 4])); R = retime(TT, 'daily', 'lastvalue'); "
     'assert(isequal(R.v, [3; 4]));'),
    ("t = datetime(2024, 1, 1) + hours([0 6 12 30])'; TT = timetable(t, [1; 2; 3; 4], "
     "'VariableNames', {'v'}); R = retime(TT, 'daily', @(x) max(x) - min(x)); assert(isequal(R.v, "
     "[2; 0])); R = retime(TT, 'daily', 'min'); assert(isequal(R.v, [1; 4])); R = retime(TT, "
     "'daily', 'prod'); assert(isequal(R.v, [6; 4]));"),
    ("t = datetime(2024, 1, 1) + hours([0 2])'; TT = timetable(t, [10; 30], 'VariableNames', "
     "{'v'}); R = retime(TT, 'hourly', 'linear'); assert(isequal(R.v, [10; 20; 30])); R = "
     "retime(TT, 'hourly', 'previous'); assert(isequal(R.v, [10; 10; 30])); R = retime(TT, "
     "'hourly', 'next'); assert(isequal(R.v, [10; 30; 30]));"),
    ("t = datetime(2024, 1, 1) + hours([0 2])'; TT = timetable(t, [10; 30], 'VariableNames', "
     "{'v'}); R = retime(TT, 'hourly'); assert(isequaln(R.v, [10; NaN; 30])); R = retime(TT, "
     "'hourly', 'fillwithconstant', 'Constant', 0); assert(isequal(R.v, [10; 0; 30])); R = "
     "retime(TT, 'hourly', 'nearest'); assert(R.v(1) == 10 && R.v(3) == 30);"),
    ("t = datetime(2024, 1, 1) + hours([0 2])'; TT = timetable(t, [10; 30], 'VariableNames', "
     "{'v'}); R = retime(TT, 'regular', 'linear', 'TimeStep', minutes(30)); assert(isequal(R.v, "
     "[10; 15; 20; 25; 30])); nt = datetime(2024, 1, 1) + hours([1 3])'; R = retime(TT, nt, "
     "'linear'); assert(isequal(R.v, [20; 40]));"),
    ("t = datetime(2024, 1, 1) + hours([0 2])'; TT = timetable(t, [10; 30], {'a'; 'b'}, "
     "'VariableNames', {'v', 's'}); R = retime(TT, 'hourly', 'previous'); assert(isequal(R.s, "
     "{'a'; 'a'; 'b'})); R = retime(TT, 'hourly'); assert(isequal(R.s, {'a'; ''; 'b'}) && "
     'isnan(R.v(2)));'),
    ("t = datetime(2024, 1, [1 15 40])'; TT = timetable(t, [1; 2; 3], 'VariableNames', {'v'}); R ="
     " retime(TT, 'monthly', 'sum'); assert(isequal(R.v, [3; 3]) && R.Time(2) == datetime(2024, 2,"
     " 1)); R = retime(TT, 'yearly', 'sum'); assert(R.v == 6); R = retime(TT, 'weekly', 'count'); "
     'assert(sum(R.v) == 3);'),
    ("TT = timetable(seconds([0; 1; 2; 3]), [1; 2; 3; 4], 'VariableNames', {'v'}); R = retime(TT, "
     "seconds([0; 2]), 'sum'); assert(R.v(1) == 3); R = retime(TT, 'regular', 'mean', 'TimeStep', "
     "seconds(2)); assert(isequal(R.v, [1.5; 3.5])); R = retime(TT, 'regular', 'linear', "
     "'SampleRate', 2); assert(numel(R.v) == 7 && R.v(2) == 1.5);"),
    ("t = datetime(2024, 1, 1) + hours([0 1 2 3])'; TT = timetable(t, [1; 4; 9; 16], "
     "'VariableNames', {'v'}); R = retime(TT, t(1) + minutes([30 90])', 'spline'); "
     "assert(all(abs(R.v - [2.25; 6.25]) < 1e-9)); R = retime(TT, t(1) + minutes([30 90])', "
     "'pchip'); assert(R.v(1) > 1 && R.v(1) < 4);"),
    ("t1 = datetime(2024, 1, 1:3)'; t2 = datetime(2024, 1, 2:4)'; A = timetable(t1, [1; 2; 3], "
     "'VariableNames', {'a'}); B = timetable(t2, [10; 20; 30], 'VariableNames', {'b'}); S = "
     'synchronize(A, B); assert(height(S) == 4 && isequaln(S.a, [1; 2; 3; NaN]) && isequaln(S.b, '
     '[NaN; 10; 20; 30]));'),
    ("t1 = datetime(2024, 1, 1:3)'; t2 = datetime(2024, 1, 2:4)'; A = timetable(t1, [1; 2; 3], "
     "'VariableNames', {'a'}); B = timetable(t2, [10; 20; 30], 'VariableNames', {'b'}); S = "
     "synchronize(A, B, 'intersection'); assert(isequal(S.a, [2; 3]) && isequal(S.b, [10; 20])); S"
     " = synchronize(A, B, 'first'); assert(height(S) == 3 && isnan(S.b(1)));"),
    ("t1 = datetime(2024, 1, 1:3)'; t2 = datetime(2024, 1, 2:4)'; A = timetable(t1, [1; 2; 3], "
     "'VariableNames', {'a'}); B = timetable(t2, [10; 20; 30], 'VariableNames', {'b'}); S = "
     "synchronize(A, B, 'union', 'linear'); assert(isequal(S.a, [1; 2; 3; 4]) && isequal(S.b, [0; "
     "10; 20; 30])); S = synchronize(A, B, 'commonrange'); assert(height(S) == 2);"),
    ("t1 = datetime(2024, 1, 1:3)'; A = timetable(t1, [1; 2; 3], 'VariableNames', {'v'}); B = "
     "timetable(t1, [4; 5; 6], 'VariableNames', {'v'}); S = synchronize(A, B); "
     "assert(isequal(S.Properties.VariableNames, {'v_A', 'v_B'})); S = synchronize(A, B, 'daily', "
     "'sum'); assert(isequal(S.v_B, [4; 5; 6]));"),
    ("t1 = datetime(2024, 1, 1) + hours(0:2)'; A = timetable(t1, [1; 2; 3], 'VariableNames', "
     "{'a'}); B = timetable(t1 + minutes(30), [10; 20; 30], 'VariableNames', {'b'}); S = "
     "synchronize(A, B, 'hourly', 'mean'); assert(isequal(S.b, [10; 20; 30]) && isequal(S.a, [1; "
     "2; 3])); S = synchronize(A, B, t1, 'previous'); assert(isequaln(S.b, [NaN; 10; 20]));"),
    ("t = datetime(2024, 1, 1:4)'; TT = timetable(t, [1; 2; 3; 4], 'VariableNames', {'v'}); L = "
     'lag(TT); assert(isequaln(L.v, [NaN; 1; 2; 3]) && isequal(L.Time, t)); L = lag(TT, 2); '
     'assert(isequaln(L.v, [NaN; NaN; 1; 2])); L = lag(TT, -1); assert(isequaln(L.v, [2; 3; 4; '
     'NaN])); L = lag(TT, days(1)); assert(isnan(L.v(1)));'),
    ("t = datetime(2024, 1, 1:4)'; TT = timetable(t, [1; 2; 3; 4], 'VariableNames', {'v'}); "
     'assert(isregular(TT)); [tf, dt] = isregular(TT); assert(tf && dt == days(1)); TT2 = TT([1 2 '
     "4], :); assert(~isregular(TT2)); assert(isregular(TT, 'days')); M = timetable(datetime(2024,"
     " 1:3, 1)', [1; 2; 3]); assert(~isregular(M) && isregular(M, 'months'));"),
    ("t = datetime(2024, 1, 1) + hours(0:1)'; TT = timetable(t, [1; 2], 'VariableNames', {'v'}); s"
     " = evalc('TT'); assert(strcmp(s, sprintf('\\nTT =\\n\\n  2×1 timetable\\n\\n             t"
     '              v\\n    ____________________    _\\n\\n    01-Jan-2024 00:00:00    1\\n    '
     "01-Jan-2024 01:00:00    2\\n\\n')), s);"),
    ("TT = timetable([1; 2], 'RowTimes', seconds([0; 1])); s = evalc('disp(TT)'); "
     "assert(contains(s, 'Time') && contains(s, '0 sec') && contains(s, 'Var1')); "
     "assert(strcmp(class(TT), 'timetable')); assert(isa(TT, 'timetable')); "
     'assert(isequal(size(TT), [2 1]));'),
    ("t = datetime(2024, 1, 1:3)'; TT = timetable(t, [1; NaN; 3], 'VariableNames', {'v'}); R = "
     "rmmissing(TT); assert(height(R) == 2 && istimetable(R)); F = fillmissing(TT, 'linear'); "
     'assert(F.v(2) == 2); V = varfun(@(x) x * 2, TT); assert(isequaln(V.Fun_v, [2; NaN; 6]));'),
    ("t = datetime(2024, 1, 1:3)'; TT = timetable(t, [1; 2; 3], [4; 5; 6], 'VariableNames', {'a', "
     "'b'}); T = removevars(TT, 'a'); assert(istimetable(T) && width(T) == 1); T = renamevars(TT, "
     "'a', 'z'); assert(isequal(T.Properties.VariableNames, {'z', 'b'})); T = addvars(TT, [7; 8; "
     "9], 'NewVariableNames', 'c'); assert(width(T) == 3 && istimetable(T));"),
    ("t = datetime(2024, 1, 1:3)'; TT = timetable(t, [1; 2; 3], 'VariableNames', {'a'}); s = "
     "evalc('summary(TT)'); assert(contains(s, 'RowTimes') && contains(s, 'Variables')); "
     "TT.Properties.VariableUnits = {'m'}; assert(isequal(TT.Properties.VariableUnits, {'m'})); G "
     "= groupsummary(TT, 'a', 'sum') ;"),
]


@pytest.mark.parametrize("code", TABLES)
def test_tables(m, code):
    m.run(code)


@pytest.mark.parametrize("code", TIMETABLES)
def test_timetables(m, code):
    m.run(code)


