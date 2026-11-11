function Z = column_zscores(X)
%COLUMN_ZSCORES Standardize columns (or a vector) to zero mean, unit std.
if isempty(X)
    Z = X;
    return
end
isVec = isvector(X);
if isVec
    sz = size(X);
    X = X(:);
end
mu = mean(X, 1);
sd = std(X, 0, 1);
Z = (X - mu) ./ sd;
Z(:, sd == 0) = 0;
if isVec
    Z = reshape(Z, sz);
end
end
