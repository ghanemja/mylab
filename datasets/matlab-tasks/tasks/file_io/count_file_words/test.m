assert(count_file_words('fox.txt') == 8, 'example');
assert(count_file_words('blank.txt') == 0, 'whitespace only');
assert(count_file_words('empty.txt') == 0, 'empty file');
assert(count_file_words('crlf.txt') == 5, 'windows line endings and no final newline');
assert(count_file_words('single.txt') == 1, 'single word');
n = count_file_words('fox.txt');
assert(isa(n, 'double') && isscalar(n), 'double scalar');
