% Hexadecimal and binary literals give the smallest unsigned integer type.
try, h1 = 0x1F; catch, end
try, h2 = 0xFF + 0; catch, end
try, h3 = 0x100; catch, end
try, h4 = 0b101; catch, end
try, h5 = class(0x1F); catch, end
try, h6 = class(0x1FF); catch, end
try, h7 = 0x7Fs8; catch, end
try, h8 = class(0xFFu16); catch, end
