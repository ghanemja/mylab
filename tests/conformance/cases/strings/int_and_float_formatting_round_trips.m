vals = [0 1 -1 0.5 1e-3 123456789 1e10 2^31 -2^31 0.1 1/3 1e-10 pi * 1e5];
ok = true(size(vals));
for k = 1:numel(vals)
    ok(k) = str2double(sprintf('%.17g', vals(k))) == vals(k);
end
ok2 = true(size(vals));
for k = 1:numel(vals)
    ok2(k) = str2double(num2str(vals(k), 17)) == vals(k);
end
ok3 = isequal(str2num(mat2str(vals, 17)), vals);
r1 = sprintf('%d', int32(12345));
r2 = str2double(r1) == 12345;
r3 = num2str(str2double('0.1'));
r4 = sprintf('%.3f', str2double('2.0005'));
r5 = sprintf('%g', str2double('1e5'));
r6 = sprintf('%g', str2double('100000'));
r7 = sprintf('%g', str2double('1e-5'));
r8 = sprintf('%g', 123456789);
r9 = sprintf('%g', 0.1 + 0.2);
r10 = sprintf('%.15g', 0.1 + 0.2);
r11 = sprintf('%.16g', 0.1 + 0.2);
r12 = sprintf('%.17g', 0.1 + 0.2);
r13 = mat2str(0.1 + 0.2);
r14 = mat2str(0.1 + 0.2, 17);
r15 = num2str(0.1 + 0.2);
r16 = num2str(1 - 1e-10);
r17 = num2str(123456.7891);
r18 = sprintf('%10.4e', 12345.6789);
r19 = sprintf('%x', 4095);
r20 = sprintf('%o', 64);
r21 = sprintf('%c', 72, 105);
r22 = sprintf('%5.1f|%-8s|%3d', 3.14159, 'ab', 7);
r23 = str2double(sprintf('%e', 0.000123));
r24 = sprintf('%d%%', 50);
r25 = int2str(str2double('41.5'));
r26 = dec2bin(bin2dec('110101'));
r27 = hex2dec(dec2hex(48879));
r28 = num2str(str2num('3*4'));
