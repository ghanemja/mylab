"""annotation(container, type, ...): the annotation goes on the given figure (WEC-Sim's formatPlot.m)."""

import pytest

from mylab import Interpreter
from mylab.errors import MatlabError


def run(code: str) -> Interpreter:
    it = Interpreter()
    it.execute("f1 = figure('Visible','off'); f2 = figure('Visible','off');\n" + code)
    return it


@pytest.mark.parametrize("kind, where", [
    ("textbox", "[0.1 0.1 0.3 0.1], 'String', 'on f1'"),
    ("arrow", "[0.1 0.5], [0.1 0.5]"),
    ("rectangle", "[0.2 0.2 0.3 0.3]"),
])
def test_annotation_goes_on_the_given_figure(kind, where):
    it = run(f"h = annotation(f1, '{kind}', {where}); ok = isequal(ancestor(h, 'figure'), f1); cur = gcf == f2;")
    assert bool(it.get_var("ok")) and bool(it.get_var("cur"))


def test_annotation_without_container_uses_current_figure():
    it = run("h = annotation('line', [0 1], [0 1]); ok = isequal(ancestor(h, 'figure'), f2);")
    assert bool(it.get_var("ok"))


def test_annotation_refuses_an_axes_as_container():
    with pytest.raises(MatlabError, match="figure, panel or tab"):
        run("ax = axes(f1); annotation(ax, 'line', [0 1], [0 1]);")
