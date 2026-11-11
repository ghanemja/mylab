function out = all_dependencies(deps, name)
%ALL_DEPENDENCIES Sorted transitive dependencies of a package.
seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');
stack = {name};
while ~isempty(stack)
    cur = stack{end};
    stack(end) = [];
    if isKey(deps, cur)
        d = deps(cur);
        for k = 1:numel(d)
            if ~isKey(seen, d{k})
                seen(d{k}) = true;
                stack{end + 1} = d{k}; %#ok<AGROW>
            end
        end
    end
end
if isKey(seen, name)
    remove(seen, name);
end
out = sort(keys(seen));
out = reshape(out, 1, []);
end
