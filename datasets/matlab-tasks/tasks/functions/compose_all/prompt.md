Write a MATLAB function

    h = compose_all(varargin)

that takes any number of function handles `f1, f2, ..., fn` (each of one
input and one output) and returns a single handle `h` with

    h(x) = f1(f2(...fn(x)...))

so the **last** function is applied first. With no inputs, `h` is the
identity function. If any input is not a function handle, throw an error with
identifier `compose_all:notHandle`.

Examples:

    h = compose_all(@(x) x + 1, @(x) x * 2);
    h(3)      % returns 7

    h = compose_all(@upper, @fliplr);
    h('abc')  % returns 'CBA'
