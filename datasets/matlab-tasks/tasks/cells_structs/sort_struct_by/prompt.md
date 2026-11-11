Write a MATLAB function

    s = sort_struct_by(s, field, direction)

that sorts the struct array `s` (a row or column vector, possibly empty) by
the values of the field named `field`. In all elements that field holds
either a real numeric scalar, or a char row vector (compare by character
codes, so `'B' < 'a'`). `direction` is `'ascend'` (the default when omitted)
or `'descend'`.

The sort must be **stable in both directions**: elements with equal values
keep their original relative order, also when sorting in descending order.
The result keeps the orientation of `s`. If `s` has no field with that name,
throw an error with identifier `sort_struct_by:noField`.

Example:

    s = struct('name', {'Ola', 'Bea', 'Kim', 'Ari'}, 'age', {31, 25, 31, 40});
    r = sort_struct_by(s, 'age', 'descend');
    {r.name}   % {'Ari', 'Ola', 'Kim', 'Bea'}
