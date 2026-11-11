function [p, r] = project_onto_span(V, x)
%PROJECT_ONTO_SPAN Orthogonal projection of x onto range(V).
x = x(:);
if size(V, 1) ~= numel(x)
    error('project_onto_span:size', 'V must have numel(x) rows.');
end
if isempty(V) || all(V(:) == 0)
    p = zeros(size(x));
else
    Q = orth(V);
    p = Q * (Q' * x);
end
r = x - p;
end
