"""Tree-walking interpreter."""

from __future__ import annotations

import os
import sys
from typing import Any, Callable

import numpy as np

from . import ast as A
from . import indexing as ix
from . import ops
from .display import format_display, function_handle_text, unparse
from .errors import BreakSignal, ContinueSignal, MatlabError, ReturnSignal
from .parser import parse
from .values import (
    Cell, FunctionHandle, MString, Struct, char, class_name, codes_to_char,
    empty_double, is_char, is_true, to_matlab, to_numeric, to_str, values_equal,
)

_CURRENT: list["Interpreter"] = []


def current_interpreter() -> "Interpreter | None":
    return _CURRENT[-1] if _CURRENT else None


class Frame:
    """A workspace: the base workspace or one function call's local variables."""

    def __init__(self, interp: "Interpreter", functions: dict[str, A.Function] | None = None,
                 name: str = "base", nargin: int = 0, nargout: int = 0, file: str | None = None):
        self.interp = interp
        self.vars: dict[str, Any] = {}
        self.functions = functions if functions is not None else {}
        self.globals: set[str] = set()
        self.name = name
        self.nargin = nargin
        self.nargout = nargout
        self.file = file

    def has(self, name: str) -> bool:
        if name in self.globals:
            return name in self.interp.global_vars
        return name in self.vars

    def get(self, name: str) -> Any:
        if name in self.globals:
            return self.interp.global_vars[name]
        return self.vars[name]

    def lookup(self, name: str) -> Any | None:
        if name in self.globals:
            return self.interp.global_vars.get(name)
        return self.vars.get(name)

    def set(self, name: str, value: Any) -> None:
        if name in self.globals:
            self.interp.global_vars[name] = value
        else:
            self.vars[name] = value


class Interpreter:
    """Runs MATLAB-language code.

    Typical use from Python::

        interp = Interpreter()
        interp.execute("x = linspace(0, 1, 5);")
        interp.get_var("x")     # -> numpy array
    """

    def __init__(self, out: Callable[[str], None] | None = None, path: list[str] | None = None):
        from .builtins import BUILTINS
        self.builtins = BUILTINS
        self.out = out or (lambda s: (sys.stdout.write(s), sys.stdout.flush()))
        self.global_vars: dict[str, Any] = {}
        self.base = Frame(self)
        self.frames: list[Frame] = [self.base]
        self.path: list[str] = path if path is not None else []
        self.end_stack: list[tuple[tuple[int, int], int, int]] = []
        self.warnings_enabled = True
        self.last_warning = ""
        self.depth = 0
        self.max_depth = 256
        self._file_cache: dict[str, tuple[float, A.Program]] = {}
        self.state: dict[str, Any] = {}  # scratch space for builtins (tic, rng, figures...)
        self.input_func: Callable[[str], str] = input

    # -- public API -------------------------------------------------------------------
    @property
    def frame(self) -> Frame:
        return self.frames[-1]

    def execute(self, code: str, file: str | None = None) -> None:
        """Run code in the base workspace (like typing it at the prompt)."""
        program = parse(code)
        self.run_program(program, self.base, file)

    def run_file(self, path: str) -> None:
        path = os.path.abspath(path)
        directory = os.path.dirname(path)
        if directory not in self.path:
            self.path.insert(0, directory)
        with open(path, encoding="utf-8") as fh:
            code = fh.read()
        program = parse(code)
        if not program.body and program.functions:
            # A function file: run its main function with no arguments.
            main = next(iter(program.functions.values()))
            self.call_user_function(main, [], 0, program.functions, file=path)
            return
        self.run_program(program, self.base, path)

    def run_program(self, program: A.Program, frame: Frame, file: str | None = None) -> None:
        _CURRENT.append(self)
        old_functions, old_file = frame.functions, frame.file
        try:
            if program.functions:
                frame.functions = {**frame.functions, **program.functions}
            if file:
                frame.file = file
            self.exec_block(program.body, frame)
        except ReturnSignal:
            pass
        finally:
            _CURRENT.pop()
            frame.functions, frame.file = old_functions, old_file
            if program.functions and frame is self.base and file is None:
                # Functions typed at the prompt stay defined for later commands.
                frame.functions = {**old_functions, **program.functions}

    def eval_expression(self, code: str) -> Any:
        """Evaluate a single expression and return its MATLAB value."""
        program = parse(code)
        if len(program.body) != 1 or not isinstance(program.body[0], A.ExprStmt):
            raise MatlabError("Expected a single expression.")
        _CURRENT.append(self)
        try:
            return self.eval(program.body[0].expr, self.base)
        finally:
            _CURRENT.pop()

    def get_var(self, name: str) -> Any:
        if not self.base.has(name):
            raise MatlabError(f"Unrecognized variable '{name}'.")
        return self.base.get(name)

    def set_var(self, name: str, value: Any) -> None:
        self.base.set(name, to_matlab(value))

    def warn(self, msg: str) -> None:
        self.last_warning = msg
        if self.warnings_enabled:
            self.out(f"Warning: {msg}\n")

    # -- statements -------------------------------------------------------------------
    def exec_block(self, stmts: list[A.Node], frame: Frame) -> None:
        for stmt in stmts:
            self.exec_stmt(stmt, frame)

    def exec_stmt(self, stmt: A.Node, frame: Frame) -> None:
        try:
            self._exec(stmt, frame)
        except MatlabError as e:
            if not getattr(e, "line", None):
                e.line = stmt.line  # type: ignore[attr-defined]
                e.file = frame.file  # type: ignore[attr-defined]
                e.func = frame.name  # type: ignore[attr-defined]
            raise
        except (BreakSignal, ContinueSignal, ReturnSignal):
            raise
        except RecursionError:
            raise MatlabError("Maximum recursion limit reached.")
        except KeyboardInterrupt:
            raise
        except Exception as e:  # internal failure: surface as a MATLAB error
            err = MatlabError(f"Internal error: {type(e).__name__}: {e}")
            err.line = stmt.line  # type: ignore[attr-defined]
            err.file = frame.file  # type: ignore[attr-defined]
            raise err from e

    def _exec(self, stmt: A.Node, frame: Frame) -> None:
        t = type(stmt)
        if t is A.ExprStmt:
            self.exec_expr_stmt(stmt, frame)
        elif t is A.Assign:
            self.exec_assign(stmt, frame)
        elif t is A.If:
            for cond, body in stmt.clauses:
                if is_true(self.eval(cond, frame)):
                    self.exec_block(body, frame)
                    return
            if stmt.orelse is not None:
                self.exec_block(stmt.orelse, frame)
        elif t is A.For:
            self.exec_for(stmt, frame)
        elif t is A.While:
            while is_true(self.eval(stmt.cond, frame)):
                try:
                    self.exec_block(stmt.body, frame)
                except BreakSignal:
                    break
                except ContinueSignal:
                    continue
        elif t is A.Switch:
            self.exec_switch(stmt, frame)
        elif t is A.Try:
            try:
                self.exec_block(stmt.body, frame)
            except MatlabError as e:
                if stmt.ident:
                    frame.set(stmt.ident, make_exception(e))
                self.exec_block(stmt.handler, frame)
        elif t is A.Break:
            raise BreakSignal()
        elif t is A.Continue:
            raise ContinueSignal()
        elif t is A.Return:
            raise ReturnSignal()
        elif t is A.Global:
            for name in stmt.names:
                frame.globals.add(name)
                if name not in self.global_vars:
                    self.global_vars[name] = frame.vars.pop(name, empty_double())
        else:  # pragma: no cover
            raise MatlabError(f"Unsupported statement {t.__name__}")

    def exec_expr_stmt(self, stmt: A.ExprStmt, frame: Frame) -> None:
        expr = stmt.expr
        if isinstance(expr, A.Ident) and frame.has(expr.name):
            if stmt.show:
                self.out(format_display(expr.name, frame.get(expr.name)))
            return
        values = self.eval_multi(expr, frame, 0)
        for v in values:
            frame.set("ans", v)
            if stmt.show:
                self.out(format_display("ans", v))

    def exec_assign(self, stmt: A.Assign, frame: Frame) -> None:
        targets = stmt.targets
        if len(targets) == 1:
            self.assign(targets[0], self.eval(stmt.value, frame), frame)
        else:
            values = self.eval_multi(stmt.value, frame, len(targets))
            if len(values) < len(targets):
                raise MatlabError("Too many output arguments.")
            for target, value in zip(targets, values):
                if isinstance(target, A.Ident) and target.name == "~":
                    continue
                self.assign(target, value, frame)
        if stmt.show:
            shown: list[str] = []
            for target in targets:
                name = root_name(target)
                if name and name != "~" and name not in shown and frame.has(name):
                    shown.append(name)
                    self.out(format_display(name, frame.get(name)))

    def exec_for(self, stmt: A.For, frame: Frame) -> None:
        iterable = self.eval(stmt.iterable, frame)
        name = stmt.var.name
        if isinstance(iterable, Cell):
            data = iterable.data
            columns = [Cell(data[:, j:j + 1].copy()) for j in range(data.shape[1])] if data.size else []
        elif isinstance(iterable, Struct):
            data = iterable.data
            columns = [Struct(data[:, j:j + 1].copy(), iterable.fields) for j in range(data.shape[1])]
        elif isinstance(iterable, (MString, FunctionHandle)):
            columns = [iterable]
        else:
            arr = iterable
            if arr.ndim > 2:  # iterate over columns of reshape(A, size(A, 1), [])
                arr = arr.reshape((arr.shape[0], -1), order="F")
            if arr.size == 0:
                columns = []
            elif arr.shape[0] == 1:
                columns = [arr[:, j:j + 1] for j in range(arr.shape[1])]
            else:
                columns = [arr[:, j:j + 1].copy() for j in range(arr.shape[1])]
        for col in columns:
            frame.set(name, col)
            try:
                self.exec_block(stmt.body, frame)
            except BreakSignal:
                break
            except ContinueSignal:
                continue

    def exec_switch(self, stmt: A.Switch, frame: Frame) -> None:
        subject = self.eval(stmt.subject, frame)
        for case_expr, body in stmt.cases:
            case_val = self.eval(case_expr, frame)
            options = case_val.values_f() if isinstance(case_val, Cell) else [case_val]
            if any(switch_matches(subject, opt) for opt in options):
                self.exec_block(body, frame)
                return
        if stmt.otherwise is not None:
            self.exec_block(stmt.otherwise, frame)

    # -- assignment ---------------------------------------------------------------------
    def assign(self, target: A.Node, value: Any, frame: Frame) -> None:
        if isinstance(target, A.Ident):
            frame.set(target.name, value)  # shadowing built-ins (e.g. i, pi) is allowed
            return
        current = self.lvalue_current(target, frame)
        new = self.assign_into(target, current, value, frame)
        self.assign(target.base, new, frame)

    def lvalue_current(self, target: A.Node, frame: Frame) -> Any:
        """Current value of ``target.base`` (None if it does not exist yet)."""
        return self.lookup_existing(target.base, frame)

    def lookup_existing(self, node: A.Node, frame: Frame) -> Any:
        if isinstance(node, A.Ident):
            return frame.lookup(node.name)
        base = self.lookup_existing(node.base, frame)
        if base is None:
            return None
        try:
            if isinstance(node, A.Index):
                if isinstance(base, FunctionHandle):
                    return None
                args = self.eval_index_args(node.args, base, frame)
                return index_value(base, args)
            if isinstance(node, A.CellIndex):
                if not isinstance(base, Cell):
                    return None
                args = self.eval_index_args(node.args, base, frame)
                vals = Cell(ix.get(base.data, args)).values_f()
                return vals[0] if len(vals) == 1 else None
            if isinstance(node, (A.Field, A.DynField)):
                if not isinstance(base, Struct) or base.numel() != 1:
                    return None
                name = self.field_name(node, frame)
                return base.data.flat[0].get(name)
        except MatlabError:
            return None
        return None

    def field_name(self, node: A.Node, frame: Frame) -> str:
        if isinstance(node, A.Field):
            return node.name
        return to_str(self.eval(node.expr, frame), "dynamic field name")

    def assign_into(self, target: A.Node, current: Any, value: Any, frame: Frame) -> Any:
        """Return the new value of ``target.base`` after assigning ``value``."""
        if isinstance(target, (A.Field, A.DynField)):
            name = self.field_name(target, frame)
            if current is None or (isinstance(current, np.ndarray) and current.size == 0):
                s = Struct()
            elif isinstance(current, Struct):
                if current.numel() != 1:
                    raise MatlabError("Scalar structure required for this assignment; index into "
                                      "the struct array first, e.g. s(2).field = value.")
                s = current.copy()
            else:
                raise MatlabError(f"Unable to perform assignment because dot indexing is not "
                                  f"supported for variables of type '{class_name(current)}'.")
            s.set_field(0, name, value)
            return s

        if isinstance(target, A.CellIndex):
            if current is None or (isinstance(current, np.ndarray) and current.size == 0):
                current = Cell(np.empty((0, 0), dtype=object))
            if not isinstance(current, Cell):
                raise MatlabError("Brace indexing is not supported for variables of this type; "
                                  f"'{root_name(target)}' is a {class_name(current)}.")
            args = self.eval_index_args(target.args, current, frame)
            wrapped = np.empty((1, 1), dtype=object)
            wrapped[0, 0] = value
            return Cell(ix.set_(current.data, args, wrapped, empty_double))

        if isinstance(target, A.Index):
            if isinstance(current, FunctionHandle):
                raise MatlabError("Cannot index into a function handle.")
            args = self.eval_index_args(target.args, current if current is not None else empty_double(), frame)
            is_delete = isinstance(value, np.ndarray) and value.shape == (0, 0) and value.dtype == float
            if is_delete and current is not None:
                if isinstance(current, Cell):
                    return Cell(ix.delete(current.data, args))
                if isinstance(current, Struct):
                    return Struct(ix.delete(current.data, args), current.fields)
                if isinstance(current, np.ndarray):
                    return ix.delete(current, args)
            if current is None or (isinstance(current, np.ndarray) and current.size == 0
                                   and isinstance(value, (Cell, Struct))):
                if isinstance(value, Cell):
                    current = Cell(np.empty((0, 0), dtype=object))
                elif isinstance(value, Struct):
                    current = Struct(np.empty((0, 0), dtype=object), value.fields)
                elif isinstance(value, MString):
                    raise MatlabError("String arrays are not supported yet; use a cell array {...}.")
                elif isinstance(value, np.ndarray):
                    current = np.zeros((0, 0), dtype=value.dtype)
                else:
                    raise MatlabError(f"Cannot index-assign a {class_name(value)}.")
            if isinstance(current, Cell):
                if not isinstance(value, Cell):
                    raise MatlabError("Conversion to cell from " + class_name(value) +
                                      " is not possible. Use c{i} = value to set cell contents.")
                return Cell(ix.set_(current.data, args, value.data, empty_double))
            if isinstance(current, Struct):
                if not isinstance(value, Struct):
                    raise MatlabError("Conversion to struct from " + class_name(value) + " is not possible.")
                fields = list(current.fields)
                for f in value.fields:
                    if f not in fields:
                        fields.append(f)
                data = current.copy().data
                for d in data.flat:
                    for f in fields:
                        d.setdefault(f, empty_double())
                vdata = value.copy().data
                for d in vdata.flat:
                    for f in fields:
                        d.setdefault(f, empty_double())
                fill = lambda: {f: empty_double() for f in fields}  # noqa: E731
                return Struct(ix.set_(data, args, vdata, fill), fields)
            if isinstance(current, np.ndarray):
                if isinstance(value, (Cell, Struct, FunctionHandle, MString)):
                    raise MatlabError(f"Unable to perform assignment: cannot put a {class_name(value)} "
                                      f"into a {class_name(current)} array.")
                cur, val = coerce_for_assignment(current, value)
                if cur.dtype.kind == "U":
                    fill = lambda: "\0"  # noqa: E731
                elif cur.dtype == bool:
                    fill = lambda: False  # noqa: E731
                else:
                    fill = lambda: 0  # noqa: E731
                return ix.set_(cur, args, val, fill)
            raise MatlabError(f"Cannot index into a value of type {class_name(current)}.")
        raise MatlabError("Invalid assignment target.")

    # -- expressions --------------------------------------------------------------------
    def eval(self, node: A.Node, frame: Frame) -> Any:
        t = type(node)
        if t is A.Number:
            v = node.value
            return np.array([[v]], dtype=complex if isinstance(v, complex) else float)
        if t is A.String:
            return MString(node.value) if node.double else char(node.value)
        if t is A.Ident:
            if frame.has(node.name):
                return frame.get(node.name)
            vals = self.call_named(node.name, [], 1, frame, node)
            if not vals:
                raise MatlabError(f"Function '{node.name}' does not return a value.")
            return vals[0]
        if t is A.BinOp:
            op = node.op
            if op == "&&" or op == "||":
                left = self.eval(node.left, frame)
                lv = self._short_circuit_operand(left, op)
                if op == "&&" and not lv:
                    return np.array([[False]])
                if op == "||" and lv:
                    return np.array([[True]])
                return np.array([[self._short_circuit_operand(self.eval(node.right, frame), op)]])
            return ops.binop(op, self.eval(node.left, frame), self.eval(node.right, frame))
        if t is A.UnaryOp:
            return ops.unary(node.op, self.eval(node.operand, frame))
        if t is A.Postfix:
            return ops.transpose(self.eval(node.operand, frame), node.op == "'")
        if t is A.Range:
            return self.eval_range(node, frame)
        if t is A.Matrix:
            rows = [self.eval_elements(row, frame) for row in node.rows]
            return ops.build_matrix(rows)
        if t is A.CellArray:
            return self.eval_cell_literal(node, frame)
        if t is A.End:
            if not self.end_stack:
                raise MatlabError("'end' can only be used inside an index expression.")
            shape, pos, count = self.end_stack[-1]
            return np.array([[float(ix.end_value(shape, pos, count))]])
        if t is A.Colon:
            return char(":")
        if t is A.FuncHandle:
            return FunctionHandle(name=node.name, closure={"__functions__": frame.functions})
        if t is A.Lambda:
            return self.make_lambda(node, frame)
        vals = self.eval_multi(node, frame, 1)
        if not vals:
            if isinstance(node, A.CellIndex):
                raise MatlabError("Brace indexing produced no results; the cell index is empty.")
            raise MatlabError("Expression does not produce a value (the function returns nothing).")
        return vals[0]

    def _short_circuit_operand(self, v: Any, op: str) -> bool:
        a = to_numeric(v)
        if a.size != 1:
            raise MatlabError(f"Operands to the {op} operator must be convertible to logical scalar values.")
        return bool(np.real(a.flat[0]) != 0)

    def make_lambda(self, node: A.Lambda, frame: Frame) -> FunctionHandle:
        captured: dict[str, Any] = {}
        for name in free_identifiers(node.body):
            if name not in node.params and frame.has(name):
                captured[name] = frame.get(name)
        captured["__functions__"] = frame.functions
        return FunctionHandle(params=node.params, body=node.body, closure=captured,
                              text=f"@({','.join(node.params)}){unparse(node.body)}")

    def eval_range(self, node: A.Range, frame: Frame) -> np.ndarray:
        start_v = self.eval(node.start, frame)
        stop_v = self.eval(node.stop, frame)
        step_v = self.eval(node.step, frame) if node.step is not None else None
        return colon_range(start_v, step_v, stop_v)

    def eval_cell_literal(self, node: A.CellArray, frame: Frame) -> Cell:
        row_cells = []
        for row in node.rows:
            items = []
            for elem in row:
                for v in self.eval_expanding(elem, frame):
                    items.append(v if isinstance(v, Cell) else _wrap_cell(v))
            row_cells.append(ops.concat(items, 1) if items else Cell(np.empty((0, 0), dtype=object)))
        if not row_cells:
            return Cell(np.empty((0, 0), dtype=object))
        return ops.concat(row_cells, 0)

    def eval_elements(self, nodes: list[A.Node], frame: Frame) -> list[Any]:
        out: list[Any] = []
        for n in nodes:
            out.extend(self.eval_expanding(n, frame))
        return out

    def eval_expanding(self, node: A.Node, frame: Frame) -> list[Any]:
        """Evaluate, expanding comma-separated lists such as ``c{:}`` and ``s.name``."""
        if isinstance(node, (A.CellIndex, A.Field, A.DynField)):
            return self.eval_multi(node, frame, -1)
        return [self.eval(node, frame)]

    def eval_index_args(self, args: list[A.Node], base: Any, frame: Frame) -> list[Any]:
        shape = base.shape if isinstance(base, (np.ndarray, Cell, Struct)) else (1, 1)
        count = len(args)
        out: list[Any] = []
        for i, arg in enumerate(args):
            if isinstance(arg, A.Colon):
                out.append(ix.COLON)
                continue
            self.end_stack.append((shape, i, count))
            try:
                vals = self.eval_expanding(arg, frame)
            finally:
                self.end_stack.pop()
            for v in vals:
                if is_char(v) and v.size == 1 and v.flat[0] == ":":
                    out.append(ix.COLON)
                else:
                    out.append(v)
        return out

    def eval_multi(self, node: A.Node, frame: Frame, nargout: int) -> list[Any]:
        """Evaluate ``node`` possibly producing several values.

        ``nargout`` is the number of requested outputs (0 for a bare statement,
        -1 meaning "all", for comma-separated list expansion).
        """
        t = type(node)
        if t is A.Ident:
            if frame.has(node.name):
                return [frame.get(node.name)]
            return self.call_named(node.name, [], nargout, frame, node)
        if t is A.Index:
            base = node.base
            if isinstance(base, A.Ident) and not frame.has(base.name):
                args = self.eval_elements(node.args, frame)
                args = [ix_colon_to_char(a) for a in args]
                return self.call_named(base.name, args, nargout, frame, node)
            base_val = self.eval(base, frame)
            if isinstance(base_val, FunctionHandle):
                args = self.eval_elements(node.args, frame)
                return self.call_handle(base_val, args, nargout)
            args = self.eval_index_args(node.args, base_val, frame)
            return [index_value(base_val, args)]
        if t is A.CellIndex:
            base_val = self.eval(node.base, frame)
            if not isinstance(base_val, Cell):
                raise MatlabError(f"Brace indexing is not supported for variables of type "
                                  f"'{class_name(base_val)}'.")
            args = self.eval_index_args(node.args, base_val, frame)
            return Cell(ix.get(base_val.data, args)).values_f()
        if t is A.Field or t is A.DynField:
            base_val = self.eval(node.base, frame)
            name = self.field_name(node, frame)
            if isinstance(base_val, Struct):
                if name not in base_val.fields:
                    raise MatlabError(f"Unrecognized field name \"{name}\".")
                return [d[name] for d in base_val.data.flatten(order="F")]
            raise MatlabError(f"Dot indexing is not supported for variables of type '{class_name(base_val)}'.")
        return [self.eval(node, frame)]

    # -- calls ----------------------------------------------------------------------------
    def call_named(self, name: str, args: list[Any], nargout: int, frame: Frame,
                   node: A.Node | None = None) -> list[Any]:
        fn = frame.functions.get(name)
        if fn is not None:
            return self.call_user_function(fn, args, nargout, frame.functions, frame.file)
        found = self.find_file_function(name)
        if found is not None:
            program, path = found
            if program.functions and not program.body:
                main = next(iter(program.functions.values()))
                return self.call_user_function(main, args, nargout, program.functions, path)
            if args:
                raise MatlabError(f"Attempt to execute SCRIPT {name} as a function.")
            self.run_program(program, frame, path)
            return []
        builtin = self.builtins.get(name)
        if builtin is not None:
            return self.call_builtin(name, builtin, args, nargout)
        raise MatlabError(f"Unrecognized function or variable '{name}'.", "MATLAB:UndefinedFunction")

    def function_exists(self, name: str, frame: Frame | None = None) -> bool:
        frame = frame or self.frame
        return name in frame.functions or name in self.builtins or self.find_file_function(name) is not None

    def call_builtin(self, name: str, fn: Callable, args: list[Any], nargout: int) -> list[Any]:
        try:
            result = fn(self, args, nargout)
        except MatlabError:
            raise
        except (BreakSignal, ContinueSignal, ReturnSignal, KeyboardInterrupt):
            raise
        except RecursionError:
            raise MatlabError("Maximum recursion limit reached.")
        except (ValueError, TypeError, IndexError, np.linalg.LinAlgError, ZeroDivisionError,
                OverflowError, KeyError) as e:
            raise MatlabError(f"Error using {name}: {e}") from e
        if result is None:
            return []
        if isinstance(result, tuple):
            return [to_matlab(r) for r in result]
        return [to_matlab(result)]

    def call_handle(self, handle: FunctionHandle, args: list[Any], nargout: int) -> list[Any]:
        functions = handle.closure.get("__functions__", {})
        if not handle.is_anonymous:
            fn = functions.get(handle.name)
            if fn is not None:
                return self.call_user_function(fn, args, nargout, functions, None)
            tmp = Frame(self, functions)
            return self.call_named(handle.name, args, nargout, tmp)
        params = handle.params
        if len(args) > len(params) and not (params and params[-1] == "varargin"):
            raise MatlabError("Too many input arguments.")
        frame = Frame(self, functions, name="anonymous function", nargin=len(args), nargout=nargout)
        for k, v in handle.closure.items():
            if k != "__functions__":
                frame.vars[k] = v
        for i, p in enumerate(params):
            if p == "varargin" and i == len(params) - 1:
                frame.vars[p] = Cell.from_list(args[i:])
                break
            if i < len(args) and p != "~":
                frame.vars[p] = args[i]
        self._enter()
        self.frames.append(frame)
        try:
            return self.eval_multi(handle.body, frame, max(nargout, 1) if nargout != 0 else 0)
        finally:
            self.frames.pop()
            self._leave()

    def call_function_value(self, f: Any, args: list[Any], nargout: int = 1) -> list[Any]:
        """Call a function handle or a function given by name (used by builtins)."""
        if isinstance(f, FunctionHandle):
            return self.call_handle(f, args, nargout)
        name = to_str(f, "function name")
        return self.call_named(name, args, nargout, self.frame)

    def _enter(self) -> None:
        self.depth += 1
        if self.depth > self.max_depth:
            self.depth = 0
            raise MatlabError(f"Maximum recursion limit of {self.max_depth} reached.")

    def _leave(self) -> None:
        self.depth = max(0, self.depth - 1)

    def call_user_function(self, fn: A.Function, args: list[Any], nargout: int,
                           functions: dict[str, A.Function], file: str | None) -> list[Any]:
        params = fn.params
        has_varargin = bool(params) and params[-1] == "varargin"
        fixed = params[:-1] if has_varargin else params
        if len(args) > len(fixed) and not has_varargin:
            raise MatlabError(f"Too many input arguments for function '{fn.name}'.")
        outputs = fn.outputs
        has_varargout = bool(outputs) and outputs[-1] == "varargout"
        fixed_out = outputs[:-1] if has_varargout else outputs
        if nargout > len(fixed_out) and not has_varargout:
            raise MatlabError(f"Too many output arguments for function '{fn.name}'.")
        frame = Frame(self, functions, name=fn.name, nargin=len(args), nargout=nargout, file=file)
        for i, p in enumerate(fixed):
            if i < len(args) and p != "~":
                frame.vars[p] = args[i]
        if has_varargin:
            frame.vars["varargin"] = Cell.from_list(args[len(fixed):])
        self._enter()
        self.frames.append(frame)
        _CURRENT.append(self)
        try:
            self.exec_block(fn.body, frame)
        except ReturnSignal:
            pass
        finally:
            _CURRENT.pop()
            self.frames.pop()
            self._leave()
        results: list[Any] = []
        wanted = max(nargout, 1)
        for i, name in enumerate(fixed_out):
            if i >= wanted:
                break
            if not frame.has(name):
                if i == 0 and nargout == 0:
                    return []
                raise MatlabError(f"Output argument \"{name}\" (and possibly others) not assigned a "
                                  f"value in the execution with \"{fn.name}\" function.")
            results.append(frame.get(name))
        if has_varargout and len(results) < wanted:
            extra = frame.lookup("varargout")
            if isinstance(extra, Cell):
                vals = extra.values_f()
                need = wanted - len(results)
                if len(vals) < need and nargout > 0:
                    raise MatlabError("Not enough values in varargout.")
                results.extend(vals[:need])
        return results

    # -- .m files on the path -------------------------------------------------------------
    def find_file_function(self, name: str) -> tuple[A.Program, str] | None:
        dirs = list(self.path)
        cwd = os.getcwd()
        if cwd not in dirs:
            dirs.append(cwd)
        for d in dirs:
            path = os.path.join(d, name + ".m")
            if os.path.isfile(path):
                mtime = os.path.getmtime(path)
                cached = self._file_cache.get(path)
                if cached and cached[0] == mtime:
                    return cached[1], path
                with open(path, encoding="utf-8") as fh:
                    program = parse(fh.read())
                self._file_cache[path] = (mtime, program)
                return program, path
        return None


# -- helpers --------------------------------------------------------------------------

def _wrap_cell(v: Any) -> Cell:
    data = np.empty((1, 1), dtype=object)
    data[0, 0] = v
    return Cell(data)


def ix_colon_to_char(v: Any) -> Any:
    return char(":") if v is ix.COLON else v


def root_name(node: A.Node) -> str | None:
    while not isinstance(node, A.Ident):
        node = getattr(node, "base", None)
        if node is None:
            return None
    return node.name


def index_value(base: Any, args: list[Any]) -> Any:
    if isinstance(base, np.ndarray):
        return ix.get(base, args)
    if isinstance(base, Cell):
        return Cell(ix.get(base.data, args))
    if isinstance(base, Struct):
        return Struct(ix.get(base.data, args), base.fields)
    if isinstance(base, MString):
        arr = np.empty((1, 1), dtype=object)
        arr[0, 0] = base
        res = ix.get(arr, args)
        if res.size != 1:
            raise MatlabError("String arrays are not supported yet.")
        return res.flat[0]
    raise MatlabError(f"Indexing is not supported for values of type '{class_name(base)}'.")


def coerce_for_assignment(current: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ck, vk = current.dtype.kind, value.dtype.kind
    if ck == "U":
        if vk == "U":
            return current, value
        return current, codes_to_char(value)
    if vk == "U":
        value = to_numeric(value)
        vk = value.dtype.kind
    if ck in "iu":
        from .values import saturate_int
        return current, saturate_int(value, current.dtype)
    if ck == "b":
        if vk == "b":
            return current, value
        current = current.astype(value.dtype if vk in "fciu" else float)
        return current, value
    if vk == "c" and ck != "c":
        return current.astype(complex), value
    if vk == "b":
        return current, value.astype(current.dtype)
    if vk in "iu" and current.size == 0:
        return current.astype(value.dtype), value
    return current, value


def colon_range(start_v: Any, step_v: Any, stop_v: Any) -> np.ndarray:
    def first(v: Any, what: str) -> float:
        a = to_numeric(v)
        if a.size == 0:
            return float("nan")
        return float(np.real(a.flat[0]))

    is_chr = is_char(start_v) and is_char(stop_v)
    start = first(start_v, "start")
    stop = first(stop_v, "stop")
    step = 1.0 if step_v is None else first(step_v, "step")
    if any(np.isnan(x) for x in (start, stop, step)) or step == 0 or \
            (step > 0 and start > stop) or (step < 0 and start < stop):
        out = np.zeros((1, 0))
    else:
        n = int(np.floor((stop - start) / step + 1e-10 * max(1.0, abs((stop - start) / step)))) + 1
        out = (start + step * np.arange(n, dtype=float)).reshape(1, -1)
    if is_chr:
        return codes_to_char(out)
    for v in (start_v, stop_v):
        if isinstance(v, np.ndarray) and v.dtype.kind in "iu":
            from .values import saturate_int
            return saturate_int(out, v.dtype)
    return out


def switch_matches(subject: Any, option: Any) -> bool:
    if isinstance(subject, MString) or is_char(subject) or isinstance(option, MString) or is_char(option):
        try:
            return to_str(subject) == to_str(option)
        except MatlabError:
            return False
    if isinstance(subject, np.ndarray) and isinstance(option, np.ndarray):
        if subject.size == 0 or option.size == 0:
            return False
        if option.size == 1 and subject.size == 1:
            return bool(to_numeric(subject).flat[0] == to_numeric(option).flat[0])
        return values_equal(subject, option)
    return False


def make_exception(e: MatlabError) -> Struct:
    stack = Struct.scalar({"file": char(getattr(e, "file", None) or ""),
                           "name": char(getattr(e, "func", "") or ""),
                           "line": np.array([[float(getattr(e, "line", 0) or 0)]])})
    return Struct.scalar({
        "message": char(e.message),
        "identifier": char(e.identifier or ""),
        "stack": stack,
    })


def free_identifiers(node: A.Node) -> set[str]:
    names: set[str] = set()

    def visit(n: Any) -> None:
        if isinstance(n, A.Ident):
            names.add(n.name)
        elif isinstance(n, A.Lambda):
            inner = free_identifiers(n.body) - set(n.params)
            names.update(inner)
        elif isinstance(n, A.Node):
            for value in vars(n).values():
                visit(value)
        elif isinstance(n, (list, tuple)):
            for item in n:
                visit(item)

    visit(node)
    return names


__all__ = ["Interpreter", "Frame", "current_interpreter", "function_handle_text"]
