"""App windows: components, properties, layout and the JSON description (no browser needed)."""

import json

import numpy as np
import pytest

from mylab import ui
from mylab.api import Session


def run(code: str) -> Session:
    s = Session()
    r = s.run(code)
    assert r.error is None, r.error
    s.last = r
    return s


def out(code: str) -> str:
    s = Session()
    r = s.run(code)
    assert r.error is None, r.error
    return r.output


def err(code: str) -> str:
    r = Session().run(code)
    assert r.error is not None
    return r.error


def node(tree: dict, type_: str, nth: int = 0) -> dict:
    found = []

    def walk(n):
        if n["type"] == type_:
            found.append(n)
        for c in n["children"]:
            walk(c)

    for f in tree["figures"]:
        walk(f)
    return found[nth]


def test_no_ui_no_tree():
    r = Session().run("x = 1;")
    assert r.ui is None
    assert "ui" not in r.to_dict()


def test_figure_and_button_description():
    s = run("fig = uifigure('Name', 'Hello', 'Position', [10 20 300 200]);"
            "b = uibutton(fig, 'Text', 'Go', 'Position', [20 30 80 22], 'Tag', 'go');")
    tree = s.last.ui
    json.dumps(tree)  # plain JSON data
    fig = tree["figures"][0]
    assert fig["type"] == "figure"
    assert fig["props"]["name"] == "Hello"
    assert (fig["props"]["width"], fig["props"]["height"]) == (300, 200)
    b = fig["children"][0]
    assert b["type"] == "button"
    assert b["props"]["text"] == "Go"
    assert b["props"]["pos"] == [19, 29, 80, 22]  # 0-based from the bottom-left corner
    assert b["props"]["tag"] == "go"


def test_classes_and_types():
    text = out("""
        f = uifigure;
        disp(class(f)); disp(class(uibutton(f))); disp(class(uibutton(f, 'state')));
        disp(class(uieditfield(f))); disp(class(uieditfield(f, 'numeric')));
        disp(class(uigridlayout(f))); disp(class(uiaxes(f))); disp(class(uitable(f)));
        disp(class(uiknob(f, 'discrete'))); disp(f.Type);
        b = uibutton(f); disp(isa(b, 'matlab.ui.control.Button')); disp(isa(b, 'handle'));
    """)
    assert text.split() == [
        "matlab.ui.Figure", "matlab.ui.control.Button", "matlab.ui.control.StateButton",
        "matlab.ui.control.EditField", "matlab.ui.control.NumericEditField", "matlab.ui.container.GridLayout",
        "matlab.ui.control.UIAxes", "matlab.ui.control.Table", "matlab.ui.control.DiscreteKnob", "figure",
        "1", "1"]


def test_get_set_and_dot_syntax_are_case_insensitive():
    text = out("""
        f = uifigure; b = uibutton(f);
        set(b, 'text', 'One', 'ENABLE', 'off');
        disp(get(b, 'Text')); disp(b.Enable);
        b.Text = "Two"; disp(b.Text); disp(class(b.Text));
        b.Enable = true; disp(b.Enable);
        v = get(b, {'Text', 'Visible'}); disp(v);
        s = get(b); disp(isfield(s, 'ButtonPushedFcn'));
    """)
    assert "One" in text and "off" in text and "Two" in text and "char" in text
    assert "'on'" in text


def test_unknown_property_is_an_error():
    assert "Unrecognized property 'Txt'" in err("b = uibutton(uifigure); b.Txt = 'x';")
    assert "Unrecognized property" in err("b = uibutton(uifigure, 'Nope', 1);")
    # a component that failed to build is not left behind
    s = Session()
    s.run("f = uifigure; try, uibutton(f, 'Nope', 1); catch, end")
    assert s.ui_snapshot()["figures"][0]["children"] == []


def test_defaults_do_not_send_colors_so_themes_work():
    s = run("f = uifigure; uilabel(f); uibutton(f); uieditfield(f);")
    for child in s.last.ui["figures"][0]["children"]:
        assert "style" not in child["props"]


def test_style_properties():
    s = run("f = uifigure; l = uilabel(f, 'Text', 'Hi', 'FontSize', 20, 'FontWeight', 'bold', "
            "'FontColor', 'red', 'BackgroundColor', [0 0 1], 'HorizontalAlignment', 'center', "
            "'Tooltip', 'tip', 'Visible', 'off', 'Enable', 'off');")
    p = node(s.last.ui, "label")["props"]
    assert p["style"] == {"color": "#ff0000", "background": "#0000ff", "fontSize": 20.0, "bold": True,
                          "align": "center"}
    assert p["tooltip"] == "tip" and p["visible"] is False and p["enabled"] is False


def test_colors():
    text = out("l = uilabel(uifigure); l.FontColor = '#ff8000'; disp(l.FontColor); "
               "l.FontColor = 'g'; disp(l.FontColor);")
    assert "0.5020" in text and "0     1     0" in text
    assert "not a color" in err("l = uilabel(uifigure); l.FontColor = 'pinkish';")


def test_parent_children_delete_isvalid():
    text = out("""
        f = uifigure; p = uipanel(f, 'Title', 'Box');
        a = uibutton(p); b = uilabel(p);
        disp(a.Parent == p); disp(numel(p.Children)); disp(class(p.Children));
        delete(a); disp(isvalid(a)); disp(isvalid(b)); disp(numel(p.Children));
        delete(p); disp(isvalid(b)); disp(isempty(f.Children));
        c = uibutton(f); c.Parent = uipanel(f); disp(class(c.Parent));
    """)
    assert text.split() == ["1", "2", "matlab.graphics.Graphics", "0", "1", "1", "0", "1",
                            "matlab.ui.container.Panel"]
    assert "deleted" in err("b = uibutton(uifigure); delete(b); b.Text = 'x';")


def test_children_array_indexing_and_loop():
    text = out("""
        f = uifigure;
        uibutton(f, 'Text', 'a'); uibutton(f, 'Text', 'b'); uibutton(f, 'Text', 'c');
        kids = f.Children; disp(numel(kids)); disp(kids(1).Text);
        for k = 1:numel(kids), fprintf('%s ', kids(k).Text); end
        set(kids, 'Enable', 'off'); disp(kids(3).Enable);
        both = [kids(1) kids(2)]; disp(size(both));
    """)
    assert "3\nc\nc b a" in text and "off" in text and "1     2" in text


def test_findobj_by_tag_and_type():
    text = out("""
        f = uifigure; g = uigridlayout(f);
        uibutton(g, 'Tag', 'ok', 'Text', 'OK'); uilabel(g, 'Tag', 'msg');
        h = findobj(f, 'Tag', 'ok'); disp(h.Text);
        disp(class(findobj(f, 'Tag', 'msg')));
        disp(isempty(findobj(f, 'Tag', 'nothing')));
        h2 = findall(f, 'Type', 'uibutton'); disp(h2.Text);
        disp(class(findobj('Tag', 'msg')));
    """)
    assert text.split() == ["OK", "matlab.ui.control.Label", "1", "OK", "matlab.ui.control.Label"]


def test_grid_layout_auto_placement_and_explicit_cells():
    s = run("""
        f = uifigure; g = uigridlayout(f, [2 3]);
        g.RowHeight = {'fit', '1x'}; g.ColumnWidth = {100, '2x', 'fit'};
        g.Padding = [5 5 5 5]; g.RowSpacing = 4; g.ColumnSpacing = 6;
        a = uibutton(g); b = uibutton(g); c = uibutton(g); d = uibutton(g);
        e = uilabel(g); e.Layout.Row = 3; e.Layout.Column = [1 3];
        disp(d.Layout.Row); disp(d.Layout.Column); disp(numel(g.RowHeight));
    """)
    assert s.last.output.split() == ["2", "1", "3"]
    g = node(s.last.ui, "gridlayout")
    assert g["props"]["rows"] == ["fit", "1x", "1x"]
    assert g["props"]["cols"] == [100.0, "2x", "fit"]
    assert g["props"]["padding"] == [5, 5, 5, 5]
    assert (g["props"]["rowGap"], g["props"]["colGap"]) == (4, 6)
    cells = [c["props"]["grid"] for c in g["children"]]
    assert cells == [[1, 1, 1, 1], [1, 1, 2, 2], [1, 1, 3, 3], [2, 2, 1, 1], [3, 3, 1, 3]]
    assert "pos" not in g["children"][0]["props"]


def test_grid_sizes_are_checked():
    assert "'1x'" in err("g = uigridlayout(uifigure); g.RowHeight = {'wide'};")


def test_value_components():
    s = run("""
        f = uifigure;
        e = uieditfield(f, 'Value', 'abc');
        n = uieditfield(f, 'numeric', 'Value', 2.5, 'Limits', [0 10]);
        t = uitextarea(f, 'Value', {'one'; 'two'});
        c = uicheckbox(f, 'Text', 'Yes', 'Value', true);
        sl = uislider(f, 'Limits', [0 5], 'Value', 9);
        sp = uispinner(f, 'Value', 3, 'Step', 0.5, 'Limits', [0 10]);
        sw = uiswitch(f, 'Value', 'On');
        kn = uiknob(f, 'Value', 30);
        ga = uigauge(f, 'Value', 40, 'Limits', [0 50]);
        la = uilamp(f, 'Color', 'red');
        disp(sl.Value); disp(class(c.Value)); disp(class(t.Value)); disp(sw.Value);
    """)
    assert s.last.output.split() == ["5", "logical", "cell", "On"]
    tree = s.last.ui
    assert node(tree, "editfield")["props"]["value"] == "abc"
    num = node(tree, "editfield", 1)["props"]
    assert num["numeric"] is True and num["value"] == "2.5" and (num["min"], num["max"]) == (0, 10)
    assert node(tree, "textarea")["props"]["value"] == "one\ntwo"
    assert node(tree, "checkbox")["props"] == {"pos": [99, 99, 84, 22], "text": "Yes", "value": True}
    assert node(tree, "slider")["props"]["value"] == 5
    assert node(tree, "spinner")["props"]["step"] == 0.5
    assert node(tree, "switch")["props"]["selected"] == 1
    assert node(tree, "knob")["props"]["value"] == 30
    assert node(tree, "gauge")["props"]["max"] == 50
    assert node(tree, "lamp")["props"]["color"] == "#ff0000"


def test_numeric_field_limits_are_enforced():
    assert "between 0 and 10" in err("n = uieditfield(uifigure, 'numeric', 'Limits', [0 10]); n.Value = 11;")
    assert "must be text" in err("e = uieditfield(uifigure); e.Value = 5;")


def test_dropdown_items_and_itemsdata():
    s = run("""
        f = uifigure;
        d = uidropdown(f, 'Items', {'Red', 'Green', 'Blue'});
        disp(d.Value); d.Value = 'Blue'; disp(d.Value);
        d.Items = {'A', 'B'}; disp(d.Value);
        d.ItemsData = [10 20]; disp(d.Value); d.Value = 20; disp(d.Value); disp(d.ValueIndex);
        lb = uilistbox(f, 'Items', ["x" "y" "z"], 'Multiselect', 'on');
        lb.Value = {'x', 'z'}; disp(numel(lb.Value));
    """)
    assert s.last.output.split() == ["Red", "Blue", "A", "10", "20", "2", "2"]
    assert node(s.last.ui, "dropdown")["props"]["selected"] == 1
    lb = node(s.last.ui, "listbox")["props"]
    assert lb["items"] == ["x", "y", "z"] and lb["selected"] == [0, 2] and lb["multi"] is True
    assert "one of the Items" in err("d = uidropdown(uifigure, 'Items', {'a'}); d.Value = 'zz';")


def test_tabs_and_button_group():
    s = run("""
        f = uifigure; tg = uitabgroup(f);
        t1 = uitab(tg, 'Title', 'One'); t2 = uitab(tg, 'Title', 'Two');
        disp(tg.SelectedTab.Title); tg.SelectedTab = t2; disp(tg.SelectedTab.Title);
        bg = uibuttongroup(t1, 'Title', 'Pick');
        r1 = uiradiobutton(bg, 'Text', 'a'); r2 = uiradiobutton(bg, 'Text', 'b');
        disp(bg.SelectedObject.Text); r2.Value = true; disp(bg.SelectedObject.Text); disp(r1.Value);
        disp(numel(bg.Buttons));
    """)
    assert s.last.output.split() == ["One", "Two", "a", "b", "0", "2"]
    tg = node(s.last.ui, "tabgroup")
    assert tg["props"]["selected"] == 1
    assert [t["props"]["title"] for t in tg["children"]] == ["One", "Two"]
    assert "tab group" in err("uitab(uifigure, 'Title', 'x');")


def test_table_description():
    s = run("""
        f = uifigure;
        t = uitable(f, 'Data', [1 2; 3 4.5], 'ColumnName', {'A', 'B'}, 'ColumnEditable', [true false]);
        c = uitable(f, 'Data', {'Milk', 2, true; 'Eggs', 12, false});
        T = table([1; 2], {'x'; 'y'}, 'VariableNames', {'N', 'S'});
        u = uitable(f, 'Data', T);
    """)
    t = node(s.last.ui, "table")["props"]
    assert [c["name"] for c in t["columns"]] == ["A", "B"]
    assert [c["editable"] for c in t["columns"]] == [True, False]
    assert t["rows"] == [["1", "2"], ["3", "4.5"]]
    c = node(s.last.ui, "table", 1)["props"]
    assert [x["type"] for x in c["columns"]] == ["text", "num", "bool"]
    assert c["rows"] == [["Milk", "2", True], ["Eggs", "12", False]]
    u = node(s.last.ui, "table", 2)["props"]
    assert [x["name"] for x in u["columns"]] == ["N", "S"]
    assert u["rows"] == [["1", "x"], ["2", "y"]]


def test_menu_image_and_hyperlink(tmp_path):
    png = tmp_path / "dot.png"
    from PIL import Image
    Image.new("RGB", (2, 2), "red").save(png)
    s = run(f"""
        f = uifigure;
        m = uimenu(f, 'Text', 'File'); uimenu(m, 'Text', 'Open', 'Separator', 'on');
        im = uiimage(f, 'ImageSource', '{png}');
        im2 = uiimage(f, 'ImageSource', zeros(4, 4, 3, 'uint8'));
        uihyperlink(f, 'Text', 'Site', 'URL', 'https://example.org');
    """)
    tree = s.last.ui
    assert node(tree, "menu")["props"]["text"] == "File"
    assert node(tree, "menu", 1)["props"]["separator"] is True
    assert node(tree, "image")["props"]["src"].startswith("data:image/png;base64,")
    assert node(tree, "image", 1)["props"]["src"].startswith("data:image/png;base64,")
    assert node(tree, "hyperlink")["props"]["url"] == "https://example.org"


def test_components_without_parent_get_a_window():
    s = run("b = uibutton('Text', 'Alone');")
    assert node(s.last.ui, "button")["props"]["text"] == "Alone"


def test_parent_must_be_a_container():
    assert "cannot hold other components" in err("f = uifigure; b = uibutton(f); uilabel(b);")


def test_display_and_handles_in_workspace():
    text = out("f = uifigure; b = uibutton(f, 'Text', 'Go')")
    assert "Button with properties:" in text and "Text: 'Go'" in text
    s = run("f = uifigure; b = uibutton(f);")
    assert {v.name: v.cls for v in s.variables()}["b"] == "matlab.ui.control.Button"


def test_update_reports_only_changes():
    s = run("f = uifigure; b = uibutton(f, 'Text', 'a'); l = uilabel(f);")
    s.interp.execute("b.Text = 'b';")
    up = ui.update(s.interp)
    b = ui.find(s.interp, type="button")
    assert up == {"full": False, "patch": {str(b.uid): {"text": "b"}}, "dialogs": []}
    assert ui.update(s.interp)["patch"] == {}
    s.run("uilabel(f);")
    assert ui.update(s.interp)["patch"] == {}  # run() already sent the new tree
    s.interp.execute("delete(l);")
    assert ui.update(s.interp)["full"] is True


def test_reset_and_close():
    s = run("f = uifigure; uibutton(f);")
    s.ui_close()
    assert s.ui_snapshot() is None
    s = run("f = uifigure; uibutton(f); close(f); disp(isvalid(f))")
    assert s.last.ui is None and s.last.output.strip() == "0"


def test_units_and_normalized_positions():
    s = run("""
        f = uifigure('Position', [0 0 400 200]);
        p = uipanel(f, 'Units', 'normalized', 'Position', [0.5 0 0.5 1]);
        b = uibutton(p, 'Position', [11 11 50 20]);
        b.Units = 'normalized'; disp(b.Position);
    """)
    assert node(s.last.ui, "panel")["props"]["npos"] == [0.5, 0.0, 0.5, 1.0]
    assert np.allclose([float(x) for x in s.last.output.split()], [0.05, 0.05, 0.25, 0.1])


def test_dialogs_alert_and_progress():
    s = run("""
        f = uifigure;
        uialert(f, 'Bad input', 'Oops');
        d = uiprogressdlg(f, 'Title', 'Working', 'Message', 'Step 1', 'Cancelable', 'on');
        d.Value = 0.5; disp(d.CancelRequested);
    """)
    kinds = [(d["kind"], d.get("title")) for d in s.last.ui["dialogs"]]
    assert kinds == [("alert", "Oops"), ("progress", "Working")]
    prog = s.last.ui["dialogs"][1]
    assert prog["value"] == 0.5 and prog["cancelable"] is True
    r = s.ui_event({"id": prog["id"], "event": "dialog", "value": "Cancel"})
    assert r["error"] is None
    s.run("disp(d.CancelRequested); close(d);")
    alert = s.last.ui if hasattr(s, "last") else None
    assert alert is not None
    tree = s.ui_snapshot()
    assert [d["kind"] for d in tree["dialogs"]] == ["alert"]
    s.ui_event({"id": tree["dialogs"][0]["id"], "event": "dialog", "value": "OK"})
    assert s.ui_snapshot()["dialogs"] == []


def test_confirm_uses_default_or_handler():
    s = Session()
    r = s.run("f = uifigure; a = uiconfirm(f, 'Delete?', 'Confirm', 'Options', {'Yes', 'No'}, "
              "'DefaultOption', 2); disp(a)")
    assert r.output.strip().endswith("No")
    s.interp.state["dialog_handler"] = lambda kind, payload: "Yes" if kind == "questdlg" else None
    r = s.run("a = uiconfirm(f, 'Delete?', 'Confirm', 'Options', {'Yes', 'No'}); disp(a)")
    assert r.output.strip() == "Yes"


def test_unsupported_tools_say_so():
    assert ".mlapp" in err("appdesigner")
    assert "not available" in err("guide")
    assert "not supported" in err("uitree(uifigure)")


@pytest.mark.parametrize("name", ["uifigure", "uibutton", "uicontrol", "uigridlayout", "uitable", "guidata"])
def test_help_exists(name):
    text = Session().help(name)
    assert "not found" not in text
    assert "not supported" not in text
