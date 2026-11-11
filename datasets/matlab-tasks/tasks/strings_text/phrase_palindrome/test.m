assert(phrase_palindrome('No lemon, no melon') == true, 'classic phrase');
assert(phrase_palindrome('ab-c') == false, 'not a palindrome');
assert(phrase_palindrome('') == true, 'empty');
assert(phrase_palindrome('?!') == true, 'only punctuation');
assert(phrase_palindrome('Racecar') == true, 'case-insensitive');
assert(phrase_palindrome('12/21') == true, 'digits');
assert(phrase_palindrome('12/31') == false, 'digits differ');
assert(phrase_palindrome('ab Ba x') == false, 'nearly');
assert(islogical(phrase_palindrome('aa')), 'logical output');
