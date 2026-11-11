assert(evaluate_expression('1 + 2 * 3') == 7, 'precedence');
assert(evaluate_expression('(1 + 2) * 3') == 9, 'parentheses');
assert(evaluate_expression('-(2.5 * 4) + -3') == -13, 'unary minus');
assert(evaluate_expression('64 / 4 / 2') == 8 && evaluate_expression('8 - 3 - 2') == 3, 'left associativity');
assert(evaluate_expression('--3*-2') == -6, 'repeated unary minus');
assert(evaluate_expression(' .5+4.  ') == 4.5, 'number forms and whitespace');
assert(evaluate_expression('3*(2+(4-1)*2)/ 4') == 6, 'nested');
assert(evaluate_expression('10/4-3') == -0.5 && isinf(evaluate_expression('1/0')), 'division');
bad = {'', '2 +', '(1 + 2', '1 + 2)', '3 $ 4', '1 2', '1.2.3', '*4', '()'};
ok = true;
for k = 1:numel(bad)
    try
        evaluate_expression(bad{k});
        ok = false;
    catch err
        ok = ok && strcmp(err.identifier, 'evaluate_expression:syntax');
    end
end
assert(ok, 'malformed expressions');
