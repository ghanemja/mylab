R = axis_angle_rotation([0 0 1], pi/2);
assert(isequal(size(R), [3 3]), 'size');
assert(all(abs(R*[1; 0; 0] - [0; 1; 0]) < 1e-15), 'quarter turn about z');
R1 = axis_angle_rotation([1 2 2], 0.7);
assert(max(max(abs(R1 - [0.7909708331417675  -0.3772211664439025  0.48173574987301876; 0.48173574987301876  0.8693567707136046  -0.11022464565011408; -0.3772211664439025  0.31925381250834656  0.8693567707136046]))) < 1e-14, 'general axis');
assert(max(max(abs(R1'*R1 - eye(3)))) < 1e-14 && abs(det(R1) - 1) < 1e-14, 'proper orthogonal');
assert(all(abs(R1*[1; 2; 2] - [1; 2; 2]) < 1e-14), 'axis is unchanged');
R2 = axis_angle_rotation([0; -3; 4], -2.1);
assert(max(max(abs(R2 - [-0.5048461045998577  0.6905674933190991  0.5179256199893243; -0.6905674933190991  0.03689849305609108  -0.7223261302079317; -0.5179256199893243  -0.7223261302079317  0.4582554023440514]))) < 1e-14, 'unnormalized column axis and negative angle');
assert(max(max(abs(axis_angle_rotation([5 0 0], 0) - eye(3)))) < 1e-15, 'zero angle');
Rx = axis_angle_rotation([2 0 0], pi);
assert(max(max(abs(Rx - diag([1 -1 -1])))) < 1e-15, 'half turn about x');
threw = false;
try
    axis_angle_rotation([0 0 0], 1);
catch
    threw = true;
end
assert(threw, 'zero axis must throw');
