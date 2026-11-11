function stack = stack_machine(program)
%STACK_MACHINE Interpret a small stack-machine program.
stack = zeros(1, 0);
pc = 1;
steps = 0;
n = numel(program);
while pc <= n
    steps = steps + 1;
    if steps > 10000
        error('stack_machine:tooManySteps', 'Step limit exceeded.');
    end
    parts = strsplit(strtrim(program{pc}));
    op = parts{1};
    nargs = numel(parts) - 1;
    pc = pc + 1;
    switch op
        case {'push', 'jz', 'jmp'}
            if nargs ~= 1
                error('stack_machine:badInstruction', 'Bad argument count.');
            end
            arg = str2double(parts{2});
            if isnan(arg) || arg ~= round(arg)
                error('stack_machine:badInstruction', 'Bad argument.');
            end
        case {'pop', 'dup', 'over', 'swap', 'rot', 'add', 'sub', 'mul', 'halt'}
            if nargs ~= 0
                error('stack_machine:badInstruction', 'Unexpected argument.');
            end
        otherwise
            error('stack_machine:badInstruction', 'Unknown instruction "%s".', op);
    end
    switch op
        case 'push'
            stack(end + 1) = arg; %#ok<AGROW>
        case 'pop'
            need(stack, 1);
            stack(end) = [];
        case 'dup'
            need(stack, 1);
            stack(end + 1) = stack(end); %#ok<AGROW>
        case 'over'
            need(stack, 2);
            stack(end + 1) = stack(end - 1); %#ok<AGROW>
        case 'swap'
            need(stack, 2);
            stack([end - 1, end]) = stack([end, end - 1]);
        case 'rot'
            need(stack, 3);
            stack(end - 2:end) = stack([end - 1, end, end - 2]);
        case {'add', 'sub', 'mul'}
            need(stack, 2);
            a = stack(end);
            b = stack(end - 1);
            stack(end - 1:end) = [];
            switch op
                case 'add'
                    r = b + a;
                case 'sub'
                    r = b - a;
                otherwise
                    r = b * a;
            end
            stack(end + 1) = r; %#ok<AGROW>
        case {'jz', 'jmp'}
            if arg < 1 || arg > n
                error('stack_machine:badInstruction', 'Jump target out of range.');
            end
            if strcmp(op, 'jmp')
                pc = arg;
            else
                need(stack, 1);
                a = stack(end);
                stack(end) = [];
                if a == 0
                    pc = arg;
                end
            end
        case 'halt'
            return
    end
end
end

function need(stack, k)
if numel(stack) < k
    error('stack_machine:underflow', 'Stack underflow.');
end
end
