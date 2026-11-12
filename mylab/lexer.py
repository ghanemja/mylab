"""Tokenizer for the MATLAB-compatible language.

The tricky parts of MATLAB lexing are all context dependent:

* ``'`` is a transpose operator right after a value, and a string start otherwise.
* Inside ``[...]`` and ``{...}`` whitespace separates elements (``[1 -2]`` is two
  elements, ``[1 - 2]`` is one) and newlines separate rows.
* ``%`` starts a comment, ``%{ ... %}`` is a block comment and ``...`` continues
  a line.
* "Command syntax": ``hold on`` means ``hold('on')``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import MatlabError

KEYWORDS = {
    "if", "elseif", "else", "end", "for", "while", "switch", "case", "otherwise",
    "try", "catch", "break", "continue", "return", "function", "global",
    "persistent", "parfor", "classdef",
}

# Longest first so that greedy matching works.
OPERATORS = [
    "...",
    ".^", ".*", "./", ".\\", ".'",
    "==", "~=", "!=", "<=", ">=", "&&", "||",
    "+", "-", "*", "/", "\\", "^", "<", ">", "&", "|", "~", "!",
    "=", "(", ")", "[", "]", "{", "}", ",", ";", ":", ".", "@",
]

# Octave block terminators, accepted as 'end' so Octave-flavoured files also run.
OCTAVE_ENDS = {"endfunction", "endif", "endfor", "endwhile", "endswitch", "end_try_catch", "endparfor",
               "endclassdef", "endmethods", "endproperties", "endevents", "endenumeration"}

# Commands that are commonly typed with command syntax (``hold on``).
COMMAND_WORDS = {
    "hold", "format", "close", "clear", "clc", "grid", "axis", "figure",
    "disp", "help", "more", "warning", "clf", "pkg", "load", "save", "cd",
    "type", "which", "what", "doc", "echo", "drawnow", "tic", "toc",
    "box", "rng", "lookfor", "edit", "mkdir", "addpath", "rmpath", "ls", "dir", "import",
}


@dataclass
class Token:
    kind: str  # NUM, STR, DSTR, ID, KW, OP, NL, EOF
    value: object
    line: int
    col: int
    space_before: bool = False
    auto: bool = False  # separator inserted by the lexer from whitespace

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Token({self.kind}, {self.value!r}, {self.line}:{self.col})"


def _ends_value(tok: Token | None) -> bool:
    """True if ``tok`` can be the last token of an operand."""
    if tok is None:
        return False
    if tok.kind in ("NUM", "STR", "DSTR", "ID"):
        return True
    if tok.kind == "KW" and tok.value == "end":
        return True
    if tok.kind == "OP" and tok.value in (")", "]", "}", "'", ".'"):
        return not getattr(tok, "lambda_params_end", False)
    return False


class Lexer:
    def __init__(self, text: str):
        self.text = text.replace("\r\n", "\n").replace("\r", "\n")
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: list[Token] = []
        # Stack of open brackets: '(', '[', '{'
        self.brackets: list[str] = []
        import re
        self.in_classdef = bool(re.search(r"^\s*classdef\b", self.text, re.MULTILINE))
        self.in_arguments = False

    # -- helpers ---------------------------------------------------------------
    def peek(self, offset: int = 0) -> str:
        p = self.pos + offset
        return self.text[p] if p < len(self.text) else "\0"

    def advance(self, n: int = 1) -> str:
        s = self.text[self.pos:self.pos + n]
        for ch in s:
            if ch == "\n":
                self.line += 1
                self.col = 1
            else:
                self.col += 1
        self.pos += n
        return s

    def error(self, msg: str) -> MatlabError:
        err = MatlabError(f"Parse error at line {self.line}, column {self.col}: {msg}")
        err.line = self.line  # type: ignore[attr-defined]
        return err

    @property
    def last(self) -> Token | None:
        return self.tokens[-1] if self.tokens else None

    def in_matrix(self) -> bool:
        return bool(self.brackets) and self.brackets[-1] in "[{"

    def add(self, kind: str, value, line: int, col: int, space: bool) -> None:
        self.tokens.append(Token(kind, value, line, col, space))

    # -- main loop ---------------------------------------------------------------
    def tokenize(self) -> list[Token]:
        space = False
        while self.pos < len(self.text):
            ch = self.peek()

            # Block comments: '%{' alone on a line.
            if ch == "%" and self.peek(1) == "{" and self._line_is_only("%{"):
                self._skip_block_comment()
                continue

            if ch in "%" or (ch == "#" and not self.in_matrix()):
                while self.pos < len(self.text) and self.peek() != "\n":
                    self.advance()
                continue

            if ch in " \t":
                self.advance()
                space = True
                continue

            # Line continuation.
            if self.text.startswith("...", self.pos):
                while self.pos < len(self.text) and self.peek() != "\n":
                    self.advance()
                if self.pos < len(self.text):
                    self.advance()
                space = True
                continue

            line, col = self.line, self.col

            if ch == "\n":
                self.advance()
                if self.in_matrix():
                    self._matrix_separator(";", line, col)
                elif self.brackets:
                    pass  # newline inside parentheses is ignored
                else:
                    self.add("NL", "\n", line, col, space)
                space = False
                continue

            # Whitespace inside a matrix can act as an element separator.
            if space and self.in_matrix() and _ends_value(self.last):
                if self._starts_new_element():
                    self.add("OP", ",", line, col, True)
                    self.last.auto = True

            if ch.isdigit() or (ch == "." and self.peek(1).isdigit()):
                self._number(line, col, space)
            elif ch.isalpha() or ch == "_":
                self._identifier(line, col, space)
            elif ch == "'":
                if _ends_value(self.last) and not space:
                    self.advance()
                    self.add("OP", "'", line, col, space)
                elif _ends_value(self.last) and space and not self.in_matrix():
                    # ``a '`` outside a matrix: still a transpose.
                    self.advance()
                    self.add("OP", "'", line, col, space)
                else:
                    self._string("'", "STR", line, col, space)
            elif ch == '"':
                self._string('"', "DSTR", line, col, space)
            else:
                self._operator(line, col, space)
            space = False

        if self.brackets:
            raise self.error(f"unbalanced '{self.brackets[-1]}'")
        self.add("NL", "\n", self.line, self.col, False)
        self.add("EOF", None, self.line, self.col, False)
        return self.tokens

    # -- pieces ---------------------------------------------------------------
    def _line_is_only(self, marker: str) -> bool:
        start = self.text.rfind("\n", 0, self.pos) + 1
        end = self.text.find("\n", self.pos)
        if end == -1:
            end = len(self.text)
        return self.text[start:end].strip() == marker

    def _skip_block_comment(self) -> None:
        depth = 0
        while self.pos < len(self.text):
            start = self.pos
            end = self.text.find("\n", start)
            if end == -1:
                end = len(self.text)
            content = self.text[start:end].strip()
            if content == "%{":
                depth += 1
            elif content == "%}":
                depth -= 1
            self.advance(end - start)
            if depth == 0:
                return
            if self.pos < len(self.text):
                self.advance()  # newline
        raise self.error("unterminated block comment")

    def _starts_new_element(self) -> bool:
        ch = self.peek()
        nxt = self.peek(1)
        if ch in "+-":
            # [1 -2] -> two elements; [1 - 2] -> one; [a -= ...] n/a
            return nxt not in " \t\n" and nxt != "="
        if ch in "~!":
            return nxt != "="
        if ch == ".":
            return nxt.isdigit()
        if ch == "'" or ch == '"':
            return True
        return ch.isalnum() or ch in "_([{@"

    def _matrix_separator(self, sep: str, line: int, col: int) -> None:
        last = self.last
        if last is not None and last.kind == "OP" and last.value in ("[", "{", ";", ","):
            if last.value == "," and sep == ";":
                last.value = ";"
            return
        self.add("OP", sep, line, col, False)

    def _radix_literal(self, line: int, col: int, space: bool) -> bool:
        """0x1F / 0b101 with an optional s8..s64 / u8..u64 suffix; False if this is not one."""
        digits = "01" if self.peek(1) in "bB" else "0123456789abcdefABCDEF"
        k = 2
        while self.peek(k) and self.peek(k) in digits:
            k += 1
        if k == 2:
            return False
        value = int(self.text[self.pos + 2:self.pos + k], 2 if digits == "01" else 16)
        cls = None
        if self.peek(k) and self.peek(k) in "suSU":
            for width in ("8", "16", "32", "64"):
                if self.text[self.pos + k + 1:self.pos + k + 1 + len(width)] == width:
                    cls = ("int" if self.peek(k) in "sS" else "uint") + width
                    k += 1 + len(width)
                    break
        if self.peek(k) and (self.peek(k).isalnum() or self.peek(k) == "_"):
            return False
        if cls is None:  # the smallest unsigned type that holds the value
            cls = next((f"uint{b}" for b in (8, 16, 32, 64) if value < 2 ** b), None)
            if cls is None:
                raise MatlabError(f"The literal at line {line} is too large for a 64-bit integer.")
        else:
            bits = int(cls.lstrip("uint"))
            if value >= 2 ** bits:
                raise MatlabError(f"The literal at line {line} does not fit in {cls}.")
            if cls[0] == "i" and value >= 2 ** (bits - 1):
                value -= 2 ** bits  # signed literals give the two's complement bit pattern
        text = self.text[self.pos:self.pos + k]
        for _ in range(k):
            self.advance()
        self.add("NUM", (value, cls, text), line, col, space)
        return True

    def _number(self, line: int, col: int, space: bool) -> None:
        if self.peek() == "0" and self.peek(1) and self.peek(1) in "xXbB" \
                and self._radix_literal(line, col, space):
            return
        start = self.pos
        while self.peek().isdigit():
            self.advance()
        if self.peek() == "." and self.peek(1) not in ("*", "/", "\\", "^", "'"):
            self.advance()
            while self.peek().isdigit():
                self.advance()
        if self.peek() in "eEdD" and (
            self.peek(1).isdigit() or (self.peek(1) in "+-" and self.peek(2).isdigit())
        ):
            self.advance()
            if self.peek() in "+-":
                self.advance()
            while self.peek().isdigit():
                self.advance()
        text = self.text[start:self.pos].replace("d", "e").replace("D", "e")
        value: complex | float = float(text)
        if value >= 9007199254740992.0 and text.isdigit():
            from .ast import ExactInt
            value = ExactInt(value)
            value.exact = int(text)
        if self.peek() in "ij" and not (self.peek(1).isalnum() or self.peek(1) == "_"):
            self.advance()
            value = complex(0, value)
        self.add("NUM", value, line, col, space)

    def _identifier(self, line: int, col: int, space: bool) -> None:
        start = self.pos
        while self.peek().isalnum() or self.peek() == "_":
            self.advance()
        word = self.text[start:self.pos]
        after_dot = self.last is not None and self.last.kind == "OP" and self.last.value == "."
        at_start = len(self.tokens) == 0 or self.tokens[-1].kind == "NL" or (
            self.tokens[-1].kind == "OP" and self.tokens[-1].value in (";", ","))
        if word == "arguments" and at_start and not self.brackets:
            rest = self.text[self.pos:self.text.find("\n", self.pos) if "\n" in self.text[self.pos:] else None]
            if not rest.split("%")[0].strip() or rest.lstrip().startswith("("):
                self.in_arguments = True
        elif word in ("end", *OCTAVE_ENDS) and at_start and self.in_arguments and not after_dot:
            self.in_arguments = False
        if word in OCTAVE_ENDS and not after_dot:
            word = "end"  # Octave's endfunction / endif / ... close blocks like 'end'
        if word in KEYWORDS and not after_dot:
            self.add("KW", word, line, col, space)
            if word == "end" and self.brackets:
                self.last.in_index = True  # type: ignore[attr-defined]
            return
        self.add("ID", word, line, col, space)
        if not self.brackets and self._is_command_syntax(word):
            self._command_args()

    def _is_command_syntax(self, word: str) -> bool:
        """Detect ``hold on`` style calls at the start of a statement."""
        if len(self.tokens) >= 2:
            prev = self.tokens[-2]
            if not (prev.kind == "NL" or (prev.kind == "OP" and prev.value in (";", ","))):
                return False
        ch = self.peek()
        if ch not in " \t":
            return False
        rest_end = self.text.find("\n", self.pos)
        if rest_end == -1:
            rest_end = len(self.text)
        rest = self.text[self.pos:rest_end].strip()
        if not rest or rest[0] in "%,;":
            return False
        first = rest[0]
        # ``disp (x)``, ``hold = 3`` or ``tic + 1`` are expressions, not commands.
        if first in "=(" or (first in "+-*/^<>&|" and len(rest) > 1 and rest[1] in " \t"):
            return False
        if word in COMMAND_WORDS and (first.isdigit() or first == "_"):
            return True  # e.g. figure 2, pause 1
        if (self.in_classdef or self.in_arguments) and word not in COMMAND_WORDS:
            return False  # 'name string' in properties/arguments blocks is a declaration
        # Any other name: ``shading interp``, ``colormap jet``, ``clearvars -except y``,
        # ``addpath ../lib``, ``web http://example.org``, ``title 'Results'``.
        # The first argument must look like a word, a path or quoted text, so expressions
        # such as ``a -1``, ``x - y``, ``x ./ y`` or ``a '...`` are left alone.
        if word in KEYWORDS:
            return False
        nxt = rest[1] if len(rest) > 1 else ""
        nxt2 = rest[2] if len(rest) > 2 else ""
        if first.isalpha():
            return not (nxt in " \t" and len(rest) > 2 and rest[2] in "=")  # 'a b = 1' is not a command
        if first == "-":
            return nxt.isalpha()
        if first in "/\\~":
            return nxt not in " \t=" and nxt != ""
        if first == ".":
            return (nxt == "/" and nxt2 not in " \t") or (nxt == "." and nxt2 in "/\\;, \t%")
        if first == "'":
            return "'" in rest[1:]
        return False

    def _command_args(self) -> None:
        """Turn ``word arg1 arg2`` into ``word('arg1', 'arg2')``."""
        line, col = self.line, self.col
        self.add("OP", "(", line, col, False)
        args: list[str] = []
        while self.peek() in " \t":
            self.advance()
        while self.pos < len(self.text) and self.peek() not in "\n,;%":
            if self.peek() == "'":
                self.advance()
                buf = []
                while self.pos < len(self.text) and self.peek() not in "\n":
                    if self.peek() == "'":
                        if self.peek(1) == "'":
                            buf.append("'")
                            self.advance(2)
                            continue
                        self.advance()
                        break
                    buf.append(self.advance())
                args.append("".join(buf))
            else:
                start = self.pos
                while self.pos < len(self.text) and self.peek() not in " \t\n,;%":
                    self.advance()
                args.append(self.text[start:self.pos])
            while self.peek() in " \t":
                self.advance()
        for i, a in enumerate(args):
            if i:
                self.add("OP", ",", line, col, False)
            self.add("STR", a, line, col, False)
        self.add("OP", ")", line, col, False)

    def _string(self, quote: str, kind: str, line: int, col: int, space: bool) -> None:
        self.advance()
        buf = []
        while True:
            ch = self.peek()
            if ch == "\0" or ch == "\n":
                raise self.error("unterminated string")
            if ch == quote:
                if self.peek(1) == quote:
                    buf.append(quote)
                    self.advance(2)
                    continue
                self.advance()
                break
            buf.append(self.advance())
        self.add(kind, "".join(buf), line, col, space)

    def _operator(self, line: int, col: int, space: bool) -> None:
        for op in OPERATORS:
            if self.text.startswith(op, self.pos):
                self.advance(len(op))
                closes_lambda = False
                if op in ("(", "[", "{"):
                    after_at = op == "(" and self.last is not None and self.last.kind == "OP" \
                        and self.last.value == "@"
                    self.brackets.append("@(" if after_at else op)
                elif op in (")", "]", "}"):
                    expected = {")": "(", "]": "[", "}": "{"}[op]
                    top = self.brackets[-1] if self.brackets else None
                    if top == "@(" and op == ")":
                        closes_lambda = True
                    elif top != expected:
                        raise self.error(f"unbalanced '{op}'")
                    self.brackets.pop()
                    # Drop a trailing separator before a closing bracket.
                    last = self.last
                    if op in "]}" and last and last.kind == "OP" and last.value in (",", ";"):
                        self.tokens.pop()
                if op in (";", ",") and self.in_matrix():
                    self._matrix_separator(op, line, col)
                    return
                if op == "!=":
                    op = "~="
                elif op == "!":
                    op = "~"
                self.add("OP", op, line, col, space)
                if closes_lambda:
                    self.last.lambda_params_end = True  # type: ignore[attr-defined]
                return
        raise self.error(f"unexpected character {self.peek()!r}")


def tokenize(text: str) -> list[Token]:
    return Lexer(text).tokenize()
