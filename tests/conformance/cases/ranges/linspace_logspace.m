l1 = linspace(0, 1, 5);
l2 = linspace(1, 0, 5);
l3 = linspace(0, 1, 1);
l4 = size(linspace(0, 1, 0));
l5 = numel(linspace(0, 1));
l6 = linspace(5, 5, 3);
l7 = linspace(0, 1, 2);
l8 = logspace(0, 2, 3);
l9 = numel(logspace(0, 1));
l10 = linspace(-1, 1, 4);
l11 = linspace(0, 10, 11) == 0:10;
l12 = linspace(1, 2, 3)';
try, l13 = linspace(0, 1, 2.5); catch, end
l14 = linspace(1 + 1i, 2 + 2i, 3);
l15 = class(linspace(single(0), 1, 3));
l16 = logspace(1, 3, 2);
