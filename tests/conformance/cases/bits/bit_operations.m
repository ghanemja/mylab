b1 = bitand(12, 10);
b2 = bitor(12, 10);
b3 = bitxor(12, 10);
b4 = bitshift(1, 3);
b5 = bitshift(16, -2);
b6 = bitshift(1, 10);
b7 = bitand(uint8(255), uint8(15));
c7 = class(b7);
b8 = bitor(uint8(1), 2);
c8 = class(b8);
b9 = bitxor(uint16(65535), uint16(255));
b10 = bitshift(uint8(255), 1);
b11 = bitshift(uint8(255), -1);
b12 = bitshift(uint8(1), 8);
b13 = bitand([1 2 3 4], 1);
b14 = bitor([1 2; 4 8], 1);
b15 = bitget(5, 1:4);
b16 = bitset(0, 3);
b17 = bitset(7, 1, 0);
b18 = bitcmp(uint8(5));
b19 = bitand(5, 3) + bitor(5, 3);
b20 = bitshift(5, 2);
b21 = bitshift([1 2 4], [1 2 3]);
b22 = bitget(uint8(255), 8);
b23 = bitset(uint8(0), 8);
b24 = bitand(int8(7), int8(5));
b26 = bitshift(1, 52);
b27 = bitand(2^52 + 1, 1);
b28 = bitcmp(uint16(0));
b29 = bitshift(uint8(16), -5);
b30 = bitor(0, 0);
t1 = 0; try, tmp = bitand(1.5, 1); catch, t1 = 1; end
t2 = 0; try, tmp = bitand(-1, 1); catch, t2 = 1; end
t3 = 0; try, tmp = bitand(uint8(1), uint16(1)); catch, t3 = 1; end
try, b31 = swapbytes(uint16(1)); catch, end
b32 = bitshift(int8(-8), -1);
b33 = bitshift(int8(64), 1);
