function c = in_exactly_one(a, b)
%IN_EXACTLY_ONE Sorted symmetric difference as a row vector.
c = setxor(a(:), b(:));
c = reshape(c, 1, []);
end
