% MATLAB wraps integer and single values in the class name only with 'class'.
k1 = mat2str(int8([1 2 3]));
try, k2 = mat2str(int8([1 2 3]), 'class'); catch, end
try, k6 = mat2str(uint16(7), 'class'); catch, end
k9 = mat2str(int8(5));
