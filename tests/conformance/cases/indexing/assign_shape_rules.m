A = zeros(2, 3);
A(1, :) = [1 2 3];
A(2, :) = [4; 5; 6];
B = zeros(3, 2);
B(:, 1) = [1 2 3];
C = zeros(2, 2);
C(:) = [1 2 3 4];
D = zeros(2, 2);
D(:) = [1 2; 3 4];
E = zeros(1, 4);
E(1:4) = [1; 2; 3; 4];
F = zeros(2, 3);
F(:, [1 2]) = [1 2; 3 4];
t1 = 0;
try
    A(1, :) = [1 2];
catch
    t1 = 1;
end
t2 = 0;
try
    C(:) = [1 2 3];
catch
    t2 = 1;
end
t3 = 0;
try
    tmpF = F; tmpF(1:2, 1:2) = [1 2 3 4];
catch
    t3 = 1;
end
