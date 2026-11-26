function r = b21_matmul_1500()
A = ones(1500) + eye(1500); B = A';
C = A * B;
r = C(1);
end
