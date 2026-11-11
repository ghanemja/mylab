Write a MATLAB function `T = add_bmi(T)`.

`T` is a table with (at least) the double variables `MassKg` and `HeightM`.
Return the same table with two new variables appended at the end, in this
order:

- `BMI` — body mass index `MassKg ./ HeightM.^2`, rounded to one decimal
  place (use `round(x, 1)`).
- `Band` — a cell array of character vectors classifying the **rounded** BMI:
  `'low'` if BMI < 18.5, `'normal'` if 18.5 <= BMI < 25, and `'high'` if
  BMI >= 25.

All existing variables and rows stay unchanged.

Example:

```matlab
T = table({'A'; 'B'}, [50; 90], [1.80; 1.75], 'VariableNames', {'Name', 'MassKg', 'HeightM'});
T = add_bmi(T);
% T.BMI = [15.4; 29.4], T.Band = {'low'; 'high'}
```
