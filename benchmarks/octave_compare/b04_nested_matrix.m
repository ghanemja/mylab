function r = b04_nested_matrix()
A = zeros(400);
for i = 1:400
    for j = 1:400
        A(i, j) = i + j;
    end
end
r = A(end);
end
