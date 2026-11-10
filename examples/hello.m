% hello.m - a first tour of the language
disp('Hello from mylab!')

% Vectors and matrices
x = 1:5
A = magic(3)

% Elementwise vs matrix operations
squares = x .^ 2
product = A * A

% Solve a linear system A*y = b
b = [1; 2; 3];
y = A \ b

% Formatted output
fprintf('The sum of 1..100 is %d\n', sum(1:100));
fprintf('pi to 6 places: %.6f\n', pi);
