function R = top_students(filename, k)
%TOP_STUDENTS Highest mean scores from a CSV file.
T = readtable(filename);
names = cellstr(T.Student);
scores = [T.Math, T.Physics, T.Chemistry];
m = mean(scores, 2, 'omitnan');
[names, i1] = sort(names);
m = m(i1);
[m, i2] = sort(m, 'descend');
names = names(i2);
k = min(k, numel(m));
R = table(names(1:k), m(1:k), 'VariableNames', {'Student', 'Mean'});
end
