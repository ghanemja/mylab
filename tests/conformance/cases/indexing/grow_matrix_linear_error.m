A = [1 2; 3 4];
threw1 = 0;
try
    A(7) = 1;
catch
    threw1 = 1;
end
v = 1:3;
threw2 = 0;
try
    tmp = v(4);
catch
    threw2 = 1;
end
threw3 = 0;
try
    tmp = v(0);
catch
    threw3 = 1;
end
threw4 = 0;
try
    tmp = v(1.5);
catch
    threw4 = 1;
end
threw5 = 0;
try
    tmp = A(3, 1);
catch
    threw5 = 1;
end
threw6 = 0;
try
    tmp = v(-1);
catch
    threw6 = 1;
end
