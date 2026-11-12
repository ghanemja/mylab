% With several matches MATLAB returns a 1xN struct array, one element per match.
n = regexp('a=1,b=2', '(?<k>\w)=(?<v>\d)', 'names');
sn = size(n);
k1 = n(1).k;
k2 = n(end).k;
v2 = n(end).v;
allk = [n.k];
