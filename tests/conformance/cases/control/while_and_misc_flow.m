n = 27;
steps = 0;
while n ~= 1
    if mod(n, 2) == 0
        n = n / 2;
    else
        n = 3 * n + 1;
    end
    steps = steps + 1;
end
x = 1;
its = 0;
while abs(x^2 - 2) > 1e-12
    x = x - (x^2 - 2) / (2 * x);
    its = its + 1;
end
total = 0;
for i = 1:3
    for j = i:3
        total = total + i * j;
    end
end
primes_found = [];
for cand = 2:30
    isp = true;
    for d = 2:floor(sqrt(cand))
        if mod(cand, d) == 0
            isp = false;
            break;
        end
    end
    if isp
        primes_found(end+1) = cand;
    end
end
fizz = cell(1, 15);
for k = 1:15
    if mod(k, 15) == 0
        fizz{k} = 'FizzBuzz';
    elseif mod(k, 3) == 0
        fizz{k} = 'Fizz';
    elseif mod(k, 5) == 0
        fizz{k} = 'Buzz';
    else
        fizz{k} = num2str(k);
    end
end
