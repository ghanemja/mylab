function T = make_roster(names, ages)
%MAKE_ROSTER Table of names, ages and an adult flag.
names = names(:);
ages = double(ages(:));
T = table(names, ages, ages >= 18, 'VariableNames', {'Name', 'Age', 'Adult'});
end
