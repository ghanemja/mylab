function c = deep_merge(a, b)
%DEEP_MERGE Recursively merge scalar struct B into scalar struct A.
c = a;
names = fieldnames(b);
for k = 1:numel(names)
    f = names{k};
    vb = b.(f);
    if isfield(c, f) && is_scalar_struct(c.(f)) && is_scalar_struct(vb)
        c.(f) = deep_merge(c.(f), vb);
    else
        c.(f) = vb;
    end
end
end

function tf = is_scalar_struct(x)
tf = isstruct(x) && isscalar(x);
end
