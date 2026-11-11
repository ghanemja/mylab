assert(bounce_count(10, 0.5, 1) == 3, 'example');
assert(bounce_count(10, 0.5, 1.25) == 3, 'equal height counts');
assert(bounce_count(10, 0.5, 1.26) == 2, 'just above');
assert(bounce_count(1, 0.9, 2) == 0, 'dropped from below hmin');
assert(bounce_count(8, 0, 0.1) == 0, 'no bounce at all');
assert(bounce_count(100, 0.8, 1) == 20, 'many bounces');
assert(bounce_count(2, 0.5, 1) == 1, 'single bounce exactly at hmin');
