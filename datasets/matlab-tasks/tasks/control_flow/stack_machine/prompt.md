Write a MATLAB function

    stack = stack_machine(program)

that runs a program for a small stack machine and returns the final stack as a
**row vector** (bottom of the stack first, `1-by-0` when empty).

`program` is a cell array of char vectors, one instruction per cell.
Instructions are numbered from 1. An instruction is a lowercase name,
optionally followed by whitespace and one integer argument (leading and
trailing whitespace is allowed). Below, `a` is the top of the stack and `b`
the element under it.

| instruction | effect |
|-------------|--------|
| `push N`    | push the integer `N` (may be negative) |
| `pop`       | remove `a` |
| `dup`       | push a copy of `a` |
| `over`      | push a copy of `b` |
| `swap`      | exchange `a` and `b` |
| `rot`       | move the third element from the top to the top (`... c b a` becomes `... b a c`) |
| `add`, `sub`, `mul` | pop `a` and `b`, push `b+a`, `b-a`, `b*a` |
| `jz L`      | pop `a`; if it is zero continue at instruction `L` |
| `jmp L`     | continue at instruction `L` |
| `halt`      | stop |

The program also stops after its last instruction. Errors (use exactly these
identifiers):

- `stack_machine:underflow` when an instruction needs more elements than the
  stack holds;
- `stack_machine:badInstruction` for an unknown name, a missing or extra
  argument, or a jump target outside `1..numel(program)`;
- `stack_machine:tooManySteps` when more than 10000 instructions have been
  executed.

Example (computes 3 * (4 + 5)):

    stack_machine({'push 3', 'push 4', 'push 5', 'add', 'mul'})   % returns 27
