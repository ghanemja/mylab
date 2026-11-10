"""Syntax tree node definitions."""

from __future__ import annotations

from dataclasses import dataclass, field


class Node:
    line: int = 0


# -- expressions ----------------------------------------------------------------

@dataclass
class Number(Node):
    value: float | complex


@dataclass
class String(Node):
    value: str
    double: bool = False  # "..." string vs '...' char array


@dataclass
class Ident(Node):
    name: str


@dataclass
class Colon(Node):
    """A bare ``:`` used as an index (select everything)."""


@dataclass
class End(Node):
    """``end`` used inside an index expression."""


@dataclass
class Range(Node):
    start: Node
    step: Node | None
    stop: Node


@dataclass
class BinOp(Node):
    op: str
    left: Node
    right: Node


@dataclass
class UnaryOp(Node):
    op: str
    operand: Node


@dataclass
class Postfix(Node):
    op: str  # "'" or ".'"
    operand: Node


@dataclass
class Index(Node):
    """``base(args)`` - either a function call or array indexing."""
    base: Node
    args: list[Node]


@dataclass
class CellIndex(Node):
    base: Node
    args: list[Node]


@dataclass
class Field(Node):
    base: Node
    name: str


@dataclass
class DynField(Node):
    base: Node
    expr: Node


@dataclass
class Matrix(Node):
    rows: list[list[Node]]


@dataclass
class CellArray(Node):
    rows: list[list[Node]]


@dataclass
class FuncHandle(Node):
    name: str


@dataclass
class Lambda(Node):
    params: list[str]
    body: Node


# -- statements -----------------------------------------------------------------

@dataclass
class ExprStmt(Node):
    expr: Node
    show: bool


@dataclass
class Assign(Node):
    targets: list[Node]  # one or more lvalues; Ident('~') ignores an output
    value: Node
    show: bool


@dataclass
class If(Node):
    clauses: list[tuple[Node, list[Node]]]
    orelse: list[Node] | None


@dataclass
class For(Node):
    var: Node
    iterable: Node
    body: list[Node]


@dataclass
class While(Node):
    cond: Node
    body: list[Node]


@dataclass
class Switch(Node):
    subject: Node
    cases: list[tuple[Node, list[Node]]]
    otherwise: list[Node] | None


@dataclass
class Try(Node):
    body: list[Node]
    ident: str | None
    handler: list[Node]


@dataclass
class Break(Node):
    pass


@dataclass
class Continue(Node):
    pass


@dataclass
class Return(Node):
    pass


@dataclass
class Global(Node):
    names: list[str]


@dataclass
class Function(Node):
    name: str
    params: list[str]
    outputs: list[str]
    body: list[Node]
    subfunctions: dict[str, "Function"] = field(default_factory=dict)


@dataclass
class PropertyDef(Node):
    name: str
    default: Node | None
    attrs: dict


@dataclass
class ClassDef(Node):
    name: str
    superclasses: list[str]
    attrs: dict
    properties: list[PropertyDef]
    methods: dict[str, Function]
    static_methods: set[str] = field(default_factory=set)
    method_attrs: dict[str, dict] = field(default_factory=dict)
    abstract_methods: set[str] = field(default_factory=set)


@dataclass
class SuperCall(Node):
    """``name@Parent(args)``: superclass constructor or method call."""
    name: str
    parent: str
    args: list[Node]


@dataclass
class Program(Node):
    body: list[Node]
    functions: dict[str, Function]
    classdefs: dict[str, ClassDef] = field(default_factory=dict)
