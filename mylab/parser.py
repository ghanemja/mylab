"""Recursive-descent parser producing the tree defined in :mod:`mylab.ast`.

Operator precedence, lowest to highest::

    ||
    &&
    |
    &
    < <= > >= == ~=
    :            (ranges)
    + -          (binary)
    * / \\ .* ./ .\\
    + - ~        (unary)
    ^ .^         (left associative; -2^2 == -4)
    ' .' () {} . (postfix)
"""

from __future__ import annotations

from . import ast as A
from .errors import MatlabError
from .lexer import Token, tokenize

COMPARISON = ("<", "<=", ">", ">=", "==", "~=")
ADDITIVE = ("+", "-")
MULTIPLICATIVE = ("*", "/", "\\", ".*", "./", ".\\")
POWER = ("^", ".^")
BLOCK_END = ("end", "else", "elseif", "case", "otherwise", "catch", "function")


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0
        self.index_depth = 0

    # -- token helpers ----------------------------------------------------------
    @property
    def tok(self) -> Token:
        return self.tokens[self.pos]

    def peek(self, offset: int = 1) -> Token:
        p = min(self.pos + offset, len(self.tokens) - 1)
        return self.tokens[p]

    def next(self) -> Token:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def is_op(self, *values: str) -> bool:
        return self.tok.kind == "OP" and self.tok.value in values

    def is_kw(self, *values: str) -> bool:
        return self.tok.kind == "KW" and self.tok.value in values

    def error(self, msg: str, tok: Token | None = None) -> MatlabError:
        t = tok or self.tok
        return MatlabError(f"Parse error at line {t.line}, column {t.col}: {msg}")

    def expect_op(self, value: str) -> Token:
        if not self.is_op(value):
            raise self.error(f"expected '{value}' but found {self._describe(self.tok)}")
        return self.next()

    def expect_kw(self, value: str) -> Token:
        if not self.is_kw(value):
            raise self.error(f"expected '{value}' but found {self._describe(self.tok)}")
        return self.next()

    @staticmethod
    def _describe(t: Token) -> str:
        if t.kind == "EOF":
            return "end of input"
        if t.kind == "NL":
            return "end of line"
        return f"'{t.value}'"

    def skip_newlines(self) -> None:
        while self.tok.kind == "NL" or self.is_op(";", ","):
            self.next()

    # -- program -----------------------------------------------------------------
    def parse_program(self) -> A.Program:
        body: list[A.Node] = []
        functions: dict[str, A.Function] = {}
        self.skip_newlines()
        while self.tok.kind != "EOF":
            if self.is_kw("function"):
                fn = self.parse_function()
                functions[fn.name] = fn
            else:
                body.append(self.parse_statement())
            self.skip_newlines()
        return A.Program(body, functions)

    def parse_block(self, *terminators: str) -> list[A.Node]:
        stmts: list[A.Node] = []
        self.skip_newlines()
        while not (self.tok.kind == "KW" and self.tok.value in terminators):
            if self.tok.kind == "EOF":
                raise self.error(f"missing '{terminators[0]}'")
            if self.is_kw("function"):
                raise self.error("function definitions cannot be nested inside blocks")
            stmts.append(self.parse_statement())
            self.skip_newlines()
        return stmts

    def parse_function(self) -> A.Function:
        start = self.expect_kw("function")
        outputs: list[str] = []
        # function [a, b] = name(...)  |  function a = name(...)  |  function name(...)
        if self.is_op("["):
            self.next()
            while not self.is_op("]"):
                if self.is_op(","):
                    self.next()
                    continue
                outputs.append(self._ident_name())
            self.next()
            self.expect_op("=")
            name = self._ident_name()
        else:
            first = self._ident_name()
            if self.is_op("="):
                self.next()
                outputs = [first]
                name = self._ident_name()
            else:
                name = first
        # Allow dotted names such as set.Prop (ignored beyond the first part).
        while self.is_op("."):
            self.next()
            name += "." + self._ident_name()
        params: list[str] = []
        if self.is_op("("):
            self.next()
            while not self.is_op(")"):
                if self.is_op(","):
                    self.next()
                    continue
                if self.is_op("~"):
                    self.next()
                    params.append("~")
                    continue
                params.append(self._ident_name())
            self.next()
        stmts: list[A.Node] = []
        self.skip_newlines()
        while not (self.is_kw("end", "function") or self.tok.kind == "EOF"):
            stmts.append(self.parse_statement())
            self.skip_newlines()
        if self.is_kw("end"):
            self.next()
        fn = A.Function(name, params, outputs, stmts)
        fn.line = start.line
        return fn

    def _ident_name(self) -> str:
        t = self.tok
        if t.kind != "ID":
            raise self.error(f"expected a name but found {self._describe(t)}")
        self.next()
        return t.value

    # -- statements -------------------------------------------------------------
    def end_statement(self) -> bool:
        """Consume a statement terminator; return True if output should show."""
        if self.is_op(";"):
            self.next()
            if self.is_op(","):
                self.next()
            return False
        if self.is_op(","):
            self.next()
            return True
        if self.tok.kind == "NL":
            self.next()
            return True
        if self.tok.kind == "EOF":
            return True
        if self.tok.kind == "KW" and self.tok.value in BLOCK_END:
            return True
        raise self.error(f"unexpected {self._describe(self.tok)}")

    def parse_statement(self) -> A.Node:
        t = self.tok
        line = t.line
        node: A.Node
        if t.kind == "KW":
            kw = t.value
            if kw == "if":
                node = self.parse_if()
            elif kw in ("for", "parfor"):
                node = self.parse_for()
            elif kw == "while":
                node = self.parse_while()
            elif kw == "switch":
                node = self.parse_switch()
            elif kw == "try":
                node = self.parse_try()
            elif kw == "break":
                self.next()
                node = A.Break()
                self.end_statement()
            elif kw == "continue":
                self.next()
                node = A.Continue()
                self.end_statement()
            elif kw == "return":
                self.next()
                node = A.Return()
                self.end_statement()
            elif kw in ("global", "persistent"):
                self.next()
                names = []
                while self.tok.kind == "ID":
                    names.append(self.next().value)
                node = A.Global(names)
                self.end_statement()
            else:
                raise self.error(f"unexpected keyword '{kw}'")
            node.line = line
            return node

        if self.is_op("[") and self._is_multi_assign():
            node = self.parse_multi_assign()
            node.line = line
            return node

        expr = self.parse_expr()
        if self.is_op("="):
            self.next()
            if not self._is_lvalue(expr):
                raise self.error("invalid assignment target", t)
            value = self.parse_expr()
            show = self.end_statement()
            node = A.Assign([expr], value, show)
        else:
            show = self.end_statement()
            node = A.ExprStmt(expr, show)
        node.line = line
        return node

    def _is_lvalue(self, node: A.Node) -> bool:
        if isinstance(node, A.Ident):
            return True
        if isinstance(node, (A.Index, A.CellIndex, A.Field, A.DynField)):
            return self._is_lvalue(node.base)
        return False

    def _is_multi_assign(self) -> bool:
        depth = 0
        i = self.pos
        while i < len(self.tokens):
            t = self.tokens[i]
            if t.kind in ("NL", "EOF"):
                return False
            if t.kind == "OP" and t.value in ("[", "(", "{"):
                depth += 1
            elif t.kind == "OP" and t.value in ("]", ")", "}"):
                depth -= 1
                if depth == 0:
                    nxt = self.tokens[i + 1]
                    return nxt.kind == "OP" and nxt.value == "="
            i += 1
        return False

    def parse_multi_assign(self) -> A.Assign:
        self.expect_op("[")
        targets: list[A.Node] = []
        while not self.is_op("]"):
            if self.is_op(","):
                self.next()
                continue
            if self.is_op("~"):
                self.next()
                targets.append(A.Ident("~"))
                continue
            target = self.parse_postfix()
            if not self._is_lvalue(target):
                raise self.error("invalid assignment target")
            targets.append(target)
        self.next()
        self.expect_op("=")
        value = self.parse_expr()
        show = self.end_statement()
        return A.Assign(targets, value, show)

    def parse_if(self) -> A.If:
        self.expect_kw("if")
        clauses = []
        cond = self.parse_expr()
        body = self.parse_block("elseif", "else", "end")
        clauses.append((cond, body))
        orelse = None
        while True:
            if self.is_kw("elseif"):
                self.next()
                cond = self.parse_expr()
                body = self.parse_block("elseif", "else", "end")
                clauses.append((cond, body))
            elif self.is_kw("else"):
                self.next()
                orelse = self.parse_block("end")
            else:
                break
        self.expect_kw("end")
        self.end_statement()
        return A.If(clauses, orelse)

    def parse_for(self) -> A.For:
        self.next()
        paren = False
        if self.is_op("("):
            # for (i = 1:n) ... end
            paren = True
            self.next()
        var = A.Ident(self._ident_name())
        self.expect_op("=")
        iterable = self.parse_expr()
        if paren:
            self.expect_op(")")
        body = self.parse_block("end")
        self.expect_kw("end")
        self.end_statement()
        return A.For(var, iterable, body)

    def parse_while(self) -> A.While:
        self.expect_kw("while")
        cond = self.parse_expr()
        body = self.parse_block("end")
        self.expect_kw("end")
        self.end_statement()
        return A.While(cond, body)

    def parse_switch(self) -> A.Switch:
        self.expect_kw("switch")
        subject = self.parse_expr()
        self.skip_newlines()
        cases = []
        otherwise = None
        while True:
            if self.is_kw("case"):
                self.next()
                value = self.parse_expr()
                body = self.parse_block("case", "otherwise", "end")
                cases.append((value, body))
            elif self.is_kw("otherwise"):
                self.next()
                otherwise = self.parse_block("case", "otherwise", "end")
            else:
                break
        self.expect_kw("end")
        self.end_statement()
        return A.Switch(subject, cases, otherwise)

    def parse_try(self) -> A.Try:
        self.expect_kw("try")
        body = self.parse_block("catch", "end")
        ident = None
        handler: list[A.Node] = []
        if self.is_kw("catch"):
            catch_tok = self.next()
            if self.tok.kind == "ID" and self.tok.line == catch_tok.line:
                nxt = self.peek()
                if nxt.kind == "NL" or (nxt.kind == "OP" and nxt.value in (";", ",")):
                    ident = self.next().value
            handler = self.parse_block("end")
        self.expect_kw("end")
        self.end_statement()
        return A.Try(body, ident, handler)

    # -- expressions ------------------------------------------------------------
    def parse_expr(self) -> A.Node:
        return self.parse_oror()

    def _binary(self, sub, ops) -> A.Node:
        left = sub()
        while self.tok.kind == "OP" and self.tok.value in ops:
            op = self.next().value
            right = sub()
            left = A.BinOp(op, left, right)
        return left

    def parse_oror(self) -> A.Node:
        return self._binary(self.parse_andand, ("||",))

    def parse_andand(self) -> A.Node:
        return self._binary(self.parse_or, ("&&",))

    def parse_or(self) -> A.Node:
        return self._binary(self.parse_and, ("|",))

    def parse_and(self) -> A.Node:
        return self._binary(self.parse_cmp, ("&",))

    def parse_cmp(self) -> A.Node:
        return self._binary(self.parse_range, COMPARISON)

    def parse_range(self) -> A.Node:
        start = self.parse_add()
        if self.is_op(":") and not self._colon_is_bare():
            self.next()
            second = self.parse_add()
            if self.is_op(":"):
                self.next()
                third = self.parse_add()
                return A.Range(start, second, third)
            return A.Range(start, None, second)
        return start

    def _colon_is_bare(self) -> bool:
        nxt = self.peek()
        return nxt.kind == "OP" and nxt.value in (")", ",", "}")

    def parse_add(self) -> A.Node:
        return self._binary(self.parse_mul, ADDITIVE)

    def parse_mul(self) -> A.Node:
        return self._binary(self.parse_unary, MULTIPLICATIVE)

    def parse_unary(self) -> A.Node:
        if self.is_op("-", "+", "~"):
            op = self.next().value
            operand = self.parse_unary()
            return A.UnaryOp(op, operand)
        return self.parse_power()

    def parse_power(self) -> A.Node:
        base = self.parse_postfix()
        while self.is_op(*POWER):
            op = self.next().value
            # The exponent may carry its own unary sign: 2^-1
            if self.is_op("-", "+", "~"):
                uop = self.next().value
                exponent: A.Node = A.UnaryOp(uop, self.parse_postfix())
            else:
                exponent = self.parse_postfix()
            base = A.BinOp(op, base, exponent)
        return base

    def parse_postfix(self) -> A.Node:
        node = self.parse_primary()
        while True:
            t = self.tok
            if t.kind != "OP":
                break
            if t.value == "(":
                self.next()
                node = A.Index(node, self.parse_args(")"))
            elif t.value == "{":
                self.next()
                node = A.CellIndex(node, self.parse_args("}"))
            elif t.value == "." and self.peek().kind in ("ID", "KW"):
                self.next()
                node = A.Field(node, self.next().value)
            elif t.value == "." and self.peek().kind == "OP" and self.peek().value == "(":
                self.next()
                self.next()
                expr = self.parse_expr()
                self.expect_op(")")
                node = A.DynField(node, expr)
            elif t.value in ("'", ".'"):
                self.next()
                node = A.Postfix(t.value, node)
            else:
                break
        return node

    def parse_args(self, closer: str) -> list[A.Node]:
        args: list[A.Node] = []
        self.index_depth += 1
        try:
            while not self.is_op(closer):
                if self.is_op(":") and self._colon_is_bare():
                    self.next()
                    args.append(A.Colon())
                else:
                    args.append(self.parse_expr())
                if self.is_op(","):
                    self.next()
                elif not self.is_op(closer):
                    raise self.error(f"expected ',' or '{closer}' but found {self._describe(self.tok)}")
            self.next()
        finally:
            self.index_depth -= 1
        return args

    def parse_primary(self) -> A.Node:
        t = self.tok
        if t.kind == "NUM":
            self.next()
            return A.Number(t.value)
        if t.kind == "STR":
            self.next()
            return A.String(t.value, False)
        if t.kind == "DSTR":
            self.next()
            return A.String(t.value, True)
        if t.kind == "ID":
            self.next()
            node = A.Ident(t.value)
            node.line = t.line
            return node
        if t.kind == "KW" and t.value == "end" and self.index_depth > 0:
            self.next()
            return A.End()
        if t.kind == "OP":
            if t.value == "(":
                self.next()
                expr = self.parse_expr()  # 'end' stays valid inside nested parens
                self.expect_op(")")
                return expr
            if t.value == "[":
                self.next()
                return A.Matrix(self.parse_rows("]"))
            if t.value == "{":
                self.next()
                return A.CellArray(self.parse_rows("}"))
            if t.value == "@":
                return self.parse_handle()
            if t.value == ":":
                self.next()
                return A.Colon()
        raise self.error(f"unexpected {self._describe(t)}")

    def parse_rows(self, closer: str) -> list[list[A.Node]]:
        rows: list[list[A.Node]] = []
        row: list[A.Node] = []
        while not self.is_op(closer):
            if self.is_op(";"):
                self.next()
                rows.append(row)
                row = []
                continue
            if self.is_op(","):
                self.next()
                continue
            row.append(self.parse_expr())
        self.next()
        if row or not rows:
            rows.append(row)
        return [r for r in rows if r] if any(rows) else []

    def parse_handle(self) -> A.Node:
        self.expect_op("@")
        if self.is_op("("):
            self.next()
            params = []
            while not self.is_op(")"):
                if self.is_op(","):
                    self.next()
                    continue
                if self.is_op("~"):
                    self.next()
                    params.append("~")
                    continue
                params.append(self._ident_name())
            self.next()
            # Inside [...] or {...} the lexer may have split "@(x) x" on the space.
            if self.is_op(",") and self.tok.auto:
                self.next()
            saved = self.index_depth
            self.index_depth = 0
            try:
                body = self.parse_expr()
            finally:
                self.index_depth = saved
            return A.Lambda(params, body)
        name = self._ident_name()
        while self.is_op(".") and self.peek().kind == "ID":
            self.next()
            name += "." + self.next().value
        return A.FuncHandle(name)


def parse(text: str) -> A.Program:
    return Parser(tokenize(text)).parse_program()
