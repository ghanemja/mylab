assert(max_disjoint_bookings([1 3; 2 4; 3 5]) == 2, 'example 1');
assert(max_disjoint_bookings([1 10; 2 3; 4 5; 6 7]) == 3, 'example 2');
assert(max_disjoint_bookings(zeros(0, 2)) == 0, 'empty');
assert(max_disjoint_bookings([5 6]) == 1, 'single');
assert(max_disjoint_bookings([1 2; 1 2; 1 2]) == 1, 'identical requests');
assert(max_disjoint_bookings([0 5; 5 10; 10 15]) == 3, 'touching allowed');
assert(max_disjoint_bookings([8 9; 0 6; 1 2; 5 7; 3 4; 6 8]) == 4, 'unsorted input');
assert(max_disjoint_bookings([-3 -1; -2 0; -1 1; 0.5 0.75]) == 2, 'negative and fractional times');
