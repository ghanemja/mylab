function s = name_initials(name)
%NAME_INITIALS Uppercase first letters of the space/hyphen separated parts.
parts = regexp(name, '[^\s\-]+', 'match');
s = '';
for k = 1:numel(parts)
    s(end + 1) = upper(parts{k}(1)); %#ok<AGROW>
end
end
