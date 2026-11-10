% tables_and_strings.m - tables, string arrays and maps

% A table built from column vectors
Name = ["Ana"; "Ben"; "Chi"; "Dee"];
Age = [28; 35; 41; 23];
Score = [91.5; 78; 88; 95];
T = table(Name, Age, Score);
disp(T)

top = T(T.Score > 85, :);
fprintf('%d people scored above 85\n', height(top));
T = sortrows(T, 'Score', 'descend');
fprintf('Best: %s (%g)\n', T.Name(1), T.Score(1));
T.Passed = T.Score >= 80;

% String arrays
words = split("the quick brown fox jumps over the lazy dog", " ");
fprintf('%d words, %d unique\n', numel(words), numel(unique(words)));
longest = words(strlength(words) == max(strlength(words)));
disp(upper(longest(1)))

% Counting with a containers.Map
counts = containers.Map();
for w = words'
    key = char(w);
    if isKey(counts, key)
        counts(key) = counts(key) + 1;
    else
        counts(key) = 1;
    end
end
fprintf('"the" appears %d times\n', counts('the'));
