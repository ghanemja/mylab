Write a MATLAB function `R = top_students(filename, k)`.

`filename` names a CSV file with a header line
`Student,Math,Physics,Chemistry` followed by one line per student: a name and
three numeric scores. A score may be missing (an empty field); missing scores
are ignored when averaging. Every student has at least one score.

Read the file with `readtable`, compute each student's mean score, and return
a table `R` with exactly the variables `Student` (cell array of character
vectors) and `Mean` (double), containing the `k` students with the highest
mean, best first. Students with equal means are ordered by name in ascending
order. If `k` exceeds the number of students, return all of them.

Example: for a file

```
Student,Math,Physics,Chemistry
Lena,80,90,70
Omar,95,,85
Kim,60,70,80
```

`top_students(f, 2)` returns students `{'Omar'; 'Lena'}` with means
`[90; 80]`.
