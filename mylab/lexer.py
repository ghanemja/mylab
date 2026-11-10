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

# Commands that are commonly typed with command syntax (``hold on``).
COMMAND_WORDS = {
    "hold", "format", "close", "clear", "clc", "grid", "axis", "figure",
    "disp", "help", "more", "warning", "clf", "pkg", "load", "save", "cd",
    "type", "which", "what", "doc", "echo", "drawnow", "tic", "toc",
    "box", "rng", "lookfor", "edit", "mkdir", "addpath", "rmpath", "ls", "dir",
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

    def _number(self, line: int, col: int, space: bool) -> None:
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
        if self.peek() in "ij" and not (self.peek(1).isalnum() or self.peek(1) == "_"):
            self.advance()
            value = complex(0, value)
        self.add("NUM", value, line, col, space)

    def _identifier(self, line: int, col: int, space: bool) -> None:
        start = self.pos
        while self.peek().isalnum() or self.peek() == "_":
            self.advance()
        word = self.text[start:self.pos]
        if word in KEYWORDS and not (self.last and self.last.kind == "OP" and self.last.value == "."):
            self.add("KW", word, line, col, space)
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
        if word in COMMAND_WORDS:
            return first.isalnum() or first in "_-'"
        # Any other name: ``shading interp``, ``colormap jet``, ``clearvars -except y``.
        # Only plain words count, so expressions such as ``a -1`` or ``x '...`` are left alone.
        body = rest.split("%", 1)[0].rstrip().rstrip(";,").rstrip()
        if not body or word in KEYWORDS:
            return False
        import re
        return bool(re.fullmatch(r"-?[A-Za-z][\w.]*(\s+-?[A-Za-z_][\w.]*)*", body))

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
