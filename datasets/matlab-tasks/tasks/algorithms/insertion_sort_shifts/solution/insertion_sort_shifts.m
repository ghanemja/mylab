function [s, shifts] = insertion_sort_shifts(v)
%INSERTION_SORT_SHIFTS Insertion sort that also counts element shifts.
s = v;
shifts = 0;
for i = 2:numel(s)
    key = s(i);
    j = i - 1;
    while j >= 1 && s(j) > key
        s(j + 1) = s(j);
        j = j - 1;
        shifts = shifts + 1;
    end
    s(j + 1) = key;
end
end
