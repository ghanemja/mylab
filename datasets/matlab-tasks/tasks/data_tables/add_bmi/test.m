T = table({'A'; 'B'}, [50; 90], [1.80; 1.75], 'VariableNames', {'Name', 'MassKg', 'HeightM'});
R = add_bmi(T);
assert(isequal(R.Properties.VariableNames, {'Name', 'MassKg', 'HeightM', 'BMI', 'Band'}), 'variable order');
assert(all(abs(R.BMI - [15.4; 29.4]) < 1e-9), 'example BMI');
assert(isequal(R.Band, {'low'; 'high'}), 'example bands');
assert(isequal(R.Name, T.Name) && isequal(R.MassKg, T.MassKg), 'original data kept');
% boundaries: 18.5 and 25 exactly (after rounding)
U = table([74; 100; 59.94; 64.96], [2; 2; 1.8; 1.8], 'VariableNames', {'MassKg', 'HeightM'});
R = add_bmi(U);
assert(all(abs(R.BMI - [18.5; 25; 18.5; 20]) < 1e-9), 'rounded values');
assert(isequal(R.Band, {'normal'; 'high'; 'normal'; 'normal'}), 'band boundaries use rounded BMI');
V = add_bmi(table(45, 1.7, 'VariableNames', {'MassKg', 'HeightM'}));
assert(height(V) == 1 && iscell(V.Band) && strcmp(V.Band{1}, 'low'), 'single row');
