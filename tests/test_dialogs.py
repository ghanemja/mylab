"""Dialogs: text prompts through interp.input_func, non-interactive defaults and front-end handlers."""

import os

import numpy as np
import pytest

from mylab.builtins import dialogs
from mylab.errors import MatlabError
from mylab.values import Cell, to_str


def answers(m, *replies):
    """Make the session interactive: each prompt gets the next reply. Returns the list of prompts."""
    it = iter(replies)
    prompts = []

    def fake(prompt):
        prompts.append(prompt)
        return next(it)

    m.interp.input_func = fake
    return prompts


def no_input(m):
    def fail(prompt):
        raise EOFError("no stdin")
    m.interp.input_func = fail


def cellstr(v):
    assert isinstance(v, Cell)
    return [to_str(x) for x in v.values_f()]


# -- message boxes ---------------------------------------------------------------------------------------

def test_message_boxes_print_and_return_handles(m):
    out = m.run("""
    h = msgbox('All done', 'Result');
    e = errordlg({'Cannot open file', 'data.csv'}, 'File problem');
    w = warndlg('Low battery');
    p = helpdlg('Press any key');
    q = msgbox('Oops', 'T', 'error');
    msgbox(sprintf('Value %d', 4));
    a = isvalid(h); b = ishandle(h); close(h); c = isvalid(h); delete(e); d = ishandle(e);
    uiwait(w); uiwait; uiresume; uiresume(w); waitfor(w); waitfor(w, 'Visible'); k = waitforbuttonpress;
    """)
    assert out.splitlines() == ["Message (Result): All done", "Error (File problem): Cannot open file",
                                "  data.csv", "Warning: Low battery", "Help: Press any key", "Error (T): Oops",
                                "Message: Value 4"]
    assert m.num("a") and m.num("b") and not m.num("c") and not m.num("d") and m.num("k") == 0


# -- waitbar ---------------------------------------------------------------------------------------------

def test_waitbar_is_quiet_when_output_is_captured(m):
    out = m.run("""
    h = waitbar(0, 'Starting');
    for k = 1:10
        waitbar(k / 10, h, sprintf('Step %d', k));
    end
    waitbar(0.5, h); waitbar(0.25);
    v = isvalid(h); close(h); v2 = isvalid(h);
    g = waitbar(0.3, 'Again', 'Name', 'Progress'); n = g.Name; delete(g); v3 = ishandle(g);
    waitbar(1.5); close all
    """)
    assert out == ""
    assert m.num("v") and not m.num("v2") and not m.num("v3") and m.str("n") == "Progress"
    with pytest.raises(MatlabError):
        m.run("waitbar('text')")


def test_waitbar_draws_one_updating_line_in_a_terminal(m, monkeypatch):
    monkeypatch.setattr(dialogs, "_is_terminal", lambda interp: True)
    out = m.run("h = waitbar(0, 'Load'); waitbar(0.5, h); waitbar(0.5, h); waitbar(1, h, 'Done'); close(h);")
    parts = out.split("\r")
    assert parts[0] == "" and len(parts) == 4  # three different states, each redrawn in place
    assert parts[1].startswith("[" + "." * 30 + "]   0% Load")
    assert parts[2].startswith("[" + "#" * 15 + "." * 15 + "]  50% Load")
    assert parts[3].startswith("[" + "#" * 30 + "] 100% Done") and out.endswith("\n")
    assert out.count("\n") == 1


def test_waitbar_handler(m):
    seen = []

    def handler(kind, payload):
        seen.append((kind, payload["state"], payload["fraction"], payload["message"]))
        return True

    m.interp.state["dialog_handler"] = handler
    m.run("h = waitbar(0, 'Go'); waitbar(0.5, h, 'Half'); close(h); close(h);")
    assert [s[:2] for s in seen] == [("waitbar", "open"), ("waitbar", "update"), ("waitbar", "close")]
    assert seen[1][2:] == (0.5, "Half")


# -- questdlg ----------------------------------------------------------------------------------------------

def test_questdlg_defaults_when_nobody_can_answer(m):
    out = m.run("""
    a = questdlg('Continue?');
    b = questdlg('Save changes?', 'Editor', 'No');
    c = questdlg('Colour?', 'Pick', 'Red', 'Green', 'Green');
    d = questdlg('Size?', 'Pick', 'S', 'M', 'L', 'M');
    """)
    assert [m.str(v) for v in "abcd"] == ["Yes", "No", "Green", "M"]
    assert "[questdlg] Continue? -> Yes (default answer)" in out
    no_input(m)  # an input function that fails behaves the same way
    m.run("e = questdlg('Again?', 'T', 'One', 'Two', 'Two');")
    assert m.str("e") == "Two"
    with pytest.raises(MatlabError):
        m.run("questdlg()")


def test_questdlg_typed_answers(m):
    prompts = answers(m, "2", "", "gre", "nonsense", "3")
    out = m.run("""
    a = questdlg('Continue?', 'Check');
    b = questdlg('Continue?', 'Check', 'Cancel');
    c = questdlg('Colour?', 'Pick', 'Red', 'Green', 'Red');
    d = questdlg('Really?');
    """)
    assert [m.str(v) for v in "abcd"] == ["No", "Cancel", "Green", "Cancel"]
    assert prompts[0] == "Choose [Yes]: " and prompts[1] == "Choose [Cancel]: " and len(prompts) == 5
    assert "Check: Continue?" in out and "[1] Yes" in out and "[2] Green" in out
    assert "Please type" in out


# -- inputdlg -----------------------------------------------------------------------------------------------

def test_inputdlg_defaults(m):
    out = m.run("""
    a = inputdlg({'Name:', 'Age:'}, 'Person', 1, {'Ada', '36'});
    b = inputdlg('Value');
    c = inputdlg({'x', 'y'}, 'Two', [1 20], {'1'});
    age = str2double(a{2});
    """)
    assert cellstr(m.val("a")) == ["Ada", "36"] and m.val("a").data.shape == (2, 1)
    assert cellstr(m.val("b")) == [""] and cellstr(m.val("c")) == ["1", ""]
    assert m.num("age") == 36 and "[inputdlg] Person -> 'Ada', '36' (default answers)" in out


def test_inputdlg_typed_answers(m):
    prompts = answers(m, "Grace", "", "3.5")
    m.run("a = inputdlg({'Name:', 'Age:'}, 'Person', 1, {'Ada', '36'}); b = inputdlg('Length (m):');")
    assert cellstr(m.val("a")) == ["Grace", "36"] and cellstr(m.val("b")) == ["3.5"]
    assert prompts == ["Name [Ada]: ", "Age [36]: ", "Length (m): "]


def test_inputdlg_handler_and_cancel(m):
    m.interp.state["dialog_handler"] = lambda kind, p: ["7"] if p["title"] == "ok" else False
    m.run("a = inputdlg({'x', 'y'}, 'ok', 1, {'1', '2'}); b = inputdlg('x', 'cancel'); e = isempty(b); c = class(b);")
    assert cellstr(m.val("a")) == ["7", "2"] and m.num("e") and m.str("c") == "cell"


# -- listdlg and menu -----------------------------------------------------------------------------------

def test_listdlg(m):
    m.run("""
    [i, ok] = listdlg('ListString', {'a', 'b', 'c'});
    [j, ok2] = listdlg('ListString', {'a', 'b', 'c'}, 'InitialValue', [2 3], 'PromptString', 'Pick');
    k = listdlg('ListString', {'a', 'b', 'c'}, 'SelectionMode', 'single', 'InitialValue', [2 3]);
    """)
    assert m.num("i") == 1 and m.num("ok") == 1 and m.num("j").tolist() == [[2, 3]] and m.num("k") == 2
    prompts = answers(m, "3, 1", "9", "2", "c", "1 2")
    out = m.run("""
    [i, ok] = listdlg('ListString', {'a', 'b', 'c'}, 'Name', 'Letters');
    [j, ok2] = listdlg('ListString', {'a', 'b', 'c'});
    [k, ok3] = listdlg('ListString', {'a', 'b', 'c'});
    [s, ok4] = listdlg('ListString', {'a', 'b', 'c'}, 'SelectionMode', 'single');
    """)
    assert m.num("i").tolist() == [[1, 3]] and m.num("j") == 2
    assert m.num("k").size == 0 and m.num("ok3") == 0 and m.num("s") == 1 and m.num("ok4") == 1
    assert "Letters" in out and "[3] c" in out and "Please type numbers from 1 to 3." in out
    assert len(prompts) == 5
    with pytest.raises(MatlabError):
        m.run("listdlg('PromptString', 'x')")


def test_menu(m):
    out = m.run("k = menu('Choose a colour', 'Red', 'Green', 'Blue'); j = menu('T', {'a', 'b'});")
    assert m.num("k") == 1 and m.num("j") == 1 and "[menu] Choose a colour -> Red (first choice)" in out
    prompts = answers(m, "3", "gr", "x", "y", "z")
    out = m.run("""
    a = menu('Choose a colour', 'Red', 'Green', 'Blue');
    b = menu('Choose a colour', {'Red', 'Green', 'Blue'});
    c = menu('Choose a colour', 'Red', 'Green');
    """)
    assert m.num("a") == 3 and m.num("b") == 2 and m.num("c") == 0
    assert "[2] Green" in out and prompts[0] == "Choose a number: "
    with pytest.raises(MatlabError):
        m.run("menu('only a title')")


def test_endless_automatic_answers_are_stopped(m):
    with pytest.raises(MatlabError, match="automatically"):
        m.run("while true, k = menu('Again?', 'Yes', 'No'); if k == 2, break; end; end")


# -- files and folders ----------------------------------------------------------------------------------

def test_uigetfile_noninteractive(m, tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("1,2\n")
    out = m.run(f"""
    [f, p] = uigetfile('*.csv');
    cancelled = isequal(f, 0) && isequal(p, 0);
    [g, q, idx] = uigetfile('*.csv', 'Pick data', '{data}');
    full = fullfile(q, g);
    [h, r] = uigetfile('{data}');
    [x, y] = uigetfile('*.csv', 'Pick', 'missing_file_zz.csv'); c2 = isequal(x, 0);
    one = uigetfile;
    """)
    assert m.num("cancelled") and m.num("c2") and m.num("one") == 0
    assert m.str("g") == "data.csv" and m.str("q") == str(tmp_path) + os.sep and m.num("idx") == 1
    assert m.str("full") == str(data) and m.str("h") == "data.csv"
    assert "cancelled" in out and "(default file)" in out


def test_uigetfile_typed_path(m, tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("a")
    b.write_text("b")
    prompts = answers(m, str(tmp_path / "nope.txt"), str(a), "", f"{a};{b}")
    out = m.run("""
    [f, p] = uigetfile({'*.txt', 'Text files'; '*.*', 'All files'}, 'Open text');
    [g, q] = uigetfile('*.txt');
    [h, r] = uigetfile('*.txt', 'Several', 'MultiSelect', 'on');
    """)
    assert m.str("f") == "a.txt" and m.str("p") == str(tmp_path) + os.sep
    assert m.num("g") == 0 and cellstr(m.val("h")) == ["a.txt", "b.txt"]
    assert "not found" in out and prompts[0] == "Open text (*.txt; *.*) (empty = cancel): "


def test_uiputfile_and_uigetdir(m, tmp_path):
    out = m.run(f"""
    [f, p] = uiputfile('*.mat', 'Save results', 'results.mat');
    [g, q] = uiputfile('*.mat');
    [h, r] = uiputfile('report.txt');
    d = uigetdir; d2 = uigetdir('{tmp_path}', 'Data folder'); d3 = uigetdir('no_such_dir_zz');
    here = pwd;
    """)
    assert m.str("f") == "results.mat" and m.str("p") == os.getcwd() + os.sep
    assert m.num("g") == 0 and m.num("q") == 0 and m.str("h") == "report.txt"
    assert m.str("d") == m.str("here") and m.str("d2") == str(tmp_path) and m.str("d3") == m.str("here")
    assert "[uigetdir] Data folder" in out
    prompts = answers(m, str(tmp_path / "out.mat"), "", "c", "nowhere_zz", str(tmp_path))
    out = m.run("""
    [f, p] = uiputfile('*.mat', 'Save', 'x.mat');
    [g, q] = uiputfile('*.mat', 'Save', 'x.mat');
    d = uigetdir; e = uigetdir;
    """)
    assert m.str("f") == "out.mat" and m.str("p") == str(tmp_path) + os.sep and m.str("g") == "x.mat"
    assert m.num("d") == 0 and m.str("e") == str(tmp_path) and "not found" in out
    assert prompts[0] == "Save (*.mat) [x.mat]: "


# -- handler --------------------------------------------------------------------------------------------

def test_dialog_handler_payloads(m, tmp_path):
    seen = {}
    replies = {"msgbox": True, "questdlg": "Stop", "listdlg": [3, 1], "menu": 2,
               "uigetfile": [str(tmp_path / "a.csv"), str(tmp_path / "b.csv")],
               "uiputfile": str(tmp_path / "save.mat"), "uigetdir": False, "ginput": [[1, 2], [3, 4, 3]]}

    def handler(kind, payload):
        seen[kind] = payload
        return replies.get(kind)

    m.interp.state["dialog_handler"] = handler
    out = m.run("""
    errordlg('Bad input', 'Oops');
    a = questdlg('Go on?', 'Run', 'Go', 'Stop', 'Go');
    [i, ok] = listdlg('ListString', {'a', 'b', 'c'}, 'PromptString', 'Pick letters', 'Name', 'L');
    k = menu('Mode', 'fast', 'slow');
    [f, p] = uigetfile('*.csv', 'Open', 'MultiSelect', 'on');
    [g, q] = uiputfile({'*.mat', 'MAT-files'}, 'Save as', 'out.mat');
    d = uigetdir('.', 'Folder');
    plot(1:3); [x, y, b] = ginput(2); xy = ginput(2);
    c = inputdlg({'p1', 'p2'}, 'T', 1, {'d1', 'd2'});
    """)
    assert out.count("\n") == 1 and "[inputdlg]" in out  # only the unhandled dialog printed something
    assert seen["msgbox"] == {"message": "Bad input", "title": "Oops", "icon": "error", "function": "errordlg"}
    assert seen["questdlg"] == {"question": "Go on?", "title": "Run", "buttons": ["Go", "Stop"], "default": "Go"}
    assert seen["listdlg"] == {"items": ["a", "b", "c"], "prompt": "Pick letters", "title": "L",
                               "multiple": True, "initial": [1], "ok": "OK", "cancel": "Cancel"}
    assert seen["menu"] == {"title": "Mode", "items": ["fast", "slow"]}
    assert seen["uigetfile"] == {"title": "Open", "filters": [["*.csv", "*.csv"]], "default": "",
                                 "multiselect": True}
    assert seen["uiputfile"] == {"title": "Save as", "filters": [["*.mat", "MAT-files"]], "default": "out.mat"}
    assert seen["uigetdir"] == {"title": "Folder", "start": os.getcwd()}
    assert seen["ginput"] == {"n": 2, "figure": 1}
    assert seen["inputdlg"] == {"prompts": ["p1", "p2"], "title": "T", "defaults": ["d1", "d2"]}
    assert m.str("a") == "Stop" and m.num("i").tolist() == [[3, 1]] and m.num("k") == 2
    assert cellstr(m.val("f")) == ["a.csv", "b.csv"] and m.str("g") == "save.mat" and m.num("d") == 0
    assert m.num("x").tolist() == [[1], [3]] and m.num("y").tolist() == [[2], [4]]
    assert m.num("b").tolist() == [[1], [3]] and m.num("xy").tolist() == [[1, 2], [3, 4]]
    assert cellstr(m.val("c")) == ["d1", "d2"]


def test_failing_handler_falls_back(m):
    def handler(kind, payload):
        raise RuntimeError("front end crashed")

    m.interp.state["dialog_handler"] = handler
    m.run("a = questdlg('Continue?');")
    assert m.str("a") == "Yes"


# -- ginput and GUI functions -----------------------------------------------------------------------------

def test_ginput_needs_a_window(m):
    with pytest.raises(MatlabError, match="mouse clicks"):
        m.run("plot(1:3); [x, y] = ginput(1);")
    with pytest.raises(MatlabError):
        m.run("ginput(0)")


@pytest.mark.parametrize("call", ["uicontrol('Style', 'pushbutton')", "f = uifigure;", "uipanel", "uimenu(gcf)",
                                  "guide", "uitable('Data', magic(3))"])
def test_gui_functions_raise_one_clear_error(m, call):
    with pytest.raises(MatlabError) as e:
        m.run(call)
    assert e.value.message == "Graphical user interfaces are not supported in mylab"


def test_existing_console_functions_still_work(m):
    answers(m, "42", "hello")
    m.run("beep; keyboard; x = input('Number: '); s = input('Word: ', 's');")
    assert m.num("x") == 42 and m.str("s") == "hello"
    assert np.asarray(m.num("x")).size == 1
