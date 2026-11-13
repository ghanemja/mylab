"""App windows: events from a front end run callbacks, and the answer lists what changed."""

import textwrap

from mylab import ui
from mylab.api import Session


def start(code: str, tmp_path=None) -> Session:
    s = Session(cwd=str(tmp_path)) if tmp_path is not None else Session()
    r = s.run(textwrap.dedent(code))
    assert r.error is None, r.error
    s.first = r
    return s


def send(s: Session, tag: str, event: str = "push", **more) -> dict:
    n = ui.find(s.interp, tag=tag)
    assert n is not None, f"no component with Tag '{tag}'"
    r = s.ui_event({"id": n.uid, "event": event, **more})
    assert r["error"] is None, r["error"]
    return r


def prop(s: Session, tag: str, name: str):
    n = ui.find(s.interp, tag=tag)
    return n.serialize()["props"].get(name)


COUNTER = """
    fig = uifigure('Name', 'Counter');
    g = uigridlayout(fig, [1 2]);
    lbl = uilabel(g, 'Text', '0', 'Tag', 'count');
    btn = uibutton(g, 'Text', 'Add', 'Tag', 'add');
    fig.UserData = 0;
    btn.ButtonPushedFcn = @(src, event) add(fig, lbl, event);
    function add(fig, lbl, event)
        fig.UserData = fig.UserData + 1;
        lbl.Text = num2str(fig.UserData);
        fprintf('%s from %s\\n', event.EventName, class(event.Source));
    end
"""


def test_button_callback_keeps_state_between_events():
    s = start(COUNTER)
    r = send(s, "add")
    lbl = ui.find(s.interp, tag="count")
    assert r["update"] == {"full": False, "patch": {str(lbl.uid): {"text": "1"}}, "dialogs": []}
    assert r["output"] == "ButtonPushed from matlab.ui.control.Button\n"
    send(s, "add")
    assert send(s, "add")["update"]["patch"][str(lbl.uid)] == {"text": "3"}


def test_callback_error_is_reported_and_app_keeps_running():
    s = start("""
        f = uifigure; b = uibutton(f, 'Tag', 'b', 'ButtonPushedFcn', @(s, e) error('boom %d', 3));
        l = uilabel(f, 'Tag', 'l');
    """)
    r = s.ui_event({"id": ui.find(s.interp, tag="b").uid, "event": "push"})
    assert r["error"].startswith("boom 3")
    assert s.ui_snapshot() is not None


def test_value_changed_event_fields():
    s = start("""
        f = uifigure;
        e = uieditfield(f, 'Tag', 'name', 'Value', 'old');
        e.ValueChangedFcn = @(src, ev) fprintf('%s -> %s (%s)\\n', ev.PreviousValue, ev.Value, src.Value);
        n = uieditfield(f, 'numeric', 'Tag', 'num', 'Limits', [0 10], 'Value', 1);
        n.ValueChangedFcn = @(src, ev) fprintf('%g -> %g\\n', ev.PreviousValue, ev.Value);
    """)
    assert send(s, "name", "value", value="new")["output"] == "old -> new (new)\n"
    assert send(s, "name", "value", value="new")["output"] == ""  # unchanged: no callback
    assert send(s, "num", "value", value="7.5")["output"] == "1 -> 7.5\n"
    assert send(s, "num", "value", value="99")["output"] == ""  # outside the limits: rejected
    assert send(s, "num", "value", value="abc")["output"] == ""
    assert prop(s, "num", "value") == "7.5"


def test_slider_changing_and_changed():
    s = start("""
        f = uifigure; l = uilabel(f, 'Tag', 'l');
        sl = uislider(f, 'Tag', 's', 'Limits', [0 10]);
        sl.ValueChangingFcn = @(src, ev) set(l, 'Text', sprintf('moving %g (%g)', ev.Value, src.Value));
        sl.ValueChangedFcn = @(src, ev) set(l, 'Text', sprintf('at %g', src.Value));
    """)
    send(s, "s", "value", value=4, changing=True)
    assert prop(s, "l", "text") == "moving 4 (0)"
    send(s, "s", "value", value=6)
    assert prop(s, "l", "text") == "at 6"
    send(s, "s", "value", value=60)
    assert prop(s, "s", "value") == 10


def test_checkbox_dropdown_listbox_switch_spinner():
    s = start("""
        f = uifigure;
        c = uicheckbox(f, 'Tag', 'c', 'ValueChangedFcn', @(s, e) fprintf('check %d\\n', e.Value));
        d = uidropdown(f, 'Tag', 'd', 'Items', {'a', 'b', 'c'}, 'ItemsData', [10 20 30], ...
            'ValueChangedFcn', @(s, e) fprintf('drop %d (was %d)\\n', e.Value, e.PreviousValue));
        lb = uilistbox(f, 'Tag', 'lb', 'Items', {'x', 'y', 'z'}, 'Multiselect', 'on', ...
            'ValueChangedFcn', @(s, e) fprintf('list %s\\n', strjoin(e.Value, '+')));
        sw = uiswitch(f, 'Tag', 'sw', 'ValueChangedFcn', @(s, e) fprintf('switch %s\\n', e.Value));
        sp = uispinner(f, 'Tag', 'sp', 'ValueChangedFcn', @(s, e) fprintf('spin %g\\n', e.Value));
        sb = uibutton(f, 'state', 'Tag', 'sb', 'ValueChangedFcn', @(s, e) fprintf('state %d\\n', s.Value));
    """)
    assert send(s, "c", "value", value=True)["output"] == "check 1\n"
    assert send(s, "d", "value", value=2)["output"] == "drop 30 (was 10)\n"
    assert send(s, "lb", "value", value=[0, 2])["output"] == "list x+z\n"
    assert send(s, "sw", "value", value=1)["output"] == "switch On\n"
    assert send(s, "sp", "value", value=4)["output"] == "spin 4\n"
    assert send(s, "sb", "value", value=True)["output"] == "state 1\n"


def test_radio_buttons_and_tabs_selection_changed():
    s = start("""
        f = uifigure; tg = uitabgroup(f, 'Tag', 'tg');
        t1 = uitab(tg, 'Title', 'One'); t2 = uitab(tg, 'Title', 'Two');
        tg.SelectionChangedFcn = @(src, ev) fprintf('%s -> %s\\n', ev.OldValue.Title, ev.NewValue.Title);
        bg = uibuttongroup(t1, 'SelectionChangedFcn', @(src, ev) fprintf('%s -> %s\\n', ...
            ev.OldValue.Text, ev.NewValue.Text));
        uiradiobutton(bg, 'Text', 'a', 'Tag', 'ra'); uiradiobutton(bg, 'Text', 'b', 'Tag', 'rb');
    """)
    assert send(s, "tg", "select", value=1)["output"] == "One -> Two\n"
    assert prop(s, "tg", "selected") == 1
    assert send(s, "rb", "value", value=True)["output"] == "a -> b\n"
    assert prop(s, "ra", "value") is False and prop(s, "rb", "value") is True
    assert send(s, "rb", "value", value=True)["output"] == ""


def test_disabled_components_ignore_events():
    s = start("f = uifigure; b = uibutton(f, 'Tag', 'b', 'Enable', 'off', "
              "'ButtonPushedFcn', @(s, e) disp('pushed'));")
    assert send(s, "b")["output"] == ""


def test_callback_forms_cell_array_and_text():
    s = start("""
        f = uifigure;
        uibutton(f, 'Tag', 'a', 'ButtonPushedFcn', {@greet, 'cell', 2});
        uibutton(f, 'Tag', 'b', 'ButtonPushedFcn', 'disp(''text form'')');
        uibutton(f, 'Tag', 'c', 'ButtonPushedFcn', @() disp('no inputs'));
        uibutton(f, 'Tag', 'd', 'ButtonPushedFcn', @nested);
        function greet(src, event, word, n)
            fprintf('%s %d %s\\n', word, n, src.Tag);
        end
        function nested(src, ~)
            disp(src.Tag);
        end
    """)
    assert send(s, "a")["output"] == "cell 2 a\n"
    assert send(s, "b")["output"] == "text form\n"
    assert send(s, "c")["output"] == "no inputs\n"
    assert send(s, "d")["output"] == "d\n"


def test_nested_function_callbacks_share_variables(tmp_path):
    (tmp_path / "counterapp.m").write_text(textwrap.dedent("""
        function counterapp
            n = 0;
            fig = uifigure;
            lbl = uilabel(fig, 'Tag', 'lbl', 'Text', '0');
            uibutton(fig, 'Tag', 'up', 'ButtonPushedFcn', @up);
            function up(~, ~)
                n = n + 1;
                lbl.Text = sprintf('%d', n);
            end
        end
    """))
    s = start("counterapp", tmp_path)
    send(s, "up")
    send(s, "up")
    assert prop(s, "lbl", "text") == "2"


def test_table_edit_and_selection():
    s = start("""
        f = uifigure;
        t = uitable(f, 'Tag', 't', 'Data', {'Milk', 2, false; 'Eggs', 12, true}, ...
            'ColumnName', {'Item', 'Qty', 'Done'}, 'ColumnEditable', [false true true]);
        t.CellEditCallback = @(src, ev) fprintf('(%d,%d) %g -> %g "%s" total %g\\n', ev.Indices(1), ...
            ev.Indices(2), ev.PreviousData, ev.NewData, ev.EditData, sum(cell2mat(src.Data(:, 2))));
        t.CellSelectionCallback = @(src, ev) fprintf('selected %d %d\\n', ev.Indices);
    """)
    r = send(s, "t", "cell", row=0, col=1, value="5")
    assert r["output"] == '(1,2) 2 -> 5 "5" total 17\n'
    assert prop(s, "t", "rows") == [["Milk", "5", False], ["Eggs", "12", True]]
    assert send(s, "t", "cell", row=0, col=0, value="Tea")["output"] == ""  # column is not editable
    r = send(s, "t", "cell", row=0, col=2, value=True)
    assert r["output"].startswith("(1,3) 0 -> 1")
    assert send(s, "t", "select", row=1, col=0)["output"] == "selected 2 1\n"
    assert s.run("disp(class(t.Data{1, 3})); disp(t.Data{1, 2})").output.split() == ["logical", "5"]


def test_table_edit_numeric_matrix_and_table_data():
    s = start("""
        f = uifigure;
        a = uitable(f, 'Tag', 'a', 'Data', magic(3), 'ColumnEditable', true);
        T = table([1; 2], ["x"; "y"], 'VariableNames', {'N', 'S'});
        b = uitable(f, 'Tag', 'b', 'Data', T, 'ColumnEditable', true);
    """)
    send(s, "a", "cell", row=1, col=2, value="2.5")
    send(s, "b", "cell", row=0, col=0, value="9")
    send(s, "b", "cell", row=1, col=1, value="zed")
    r = s.run("disp(a.Data(2, 3)); disp(b.Data.N(1)); disp(b.Data.S(2))")
    assert r.output.split() == ["2.5000", "9", "zed"]


def test_uiaxes_plotting_updates_image():
    s = start("""
        f = uifigure; g = uigridlayout(f, [2 1]);
        ax = uiaxes(g, 'Tag', 'ax');
        b = uibutton(g, 'Tag', 'b', 'ButtonPushedFcn', @(s, e) draw(ax));
        h = plot(ax, 1:3, [1 4 9]); title(ax, 'Squares'); xlabel(ax, 'n'); hold(ax, 'on');
        ax.XLim = [0 4]; grid(ax, 'on');
        f.UserData = h;
        function draw(ax)
            h = ax.Parent.Parent.UserData;
            h.YData = [9 4 1];
            plot(ax, 1:3, [2 2 2], 'r--');
            title(ax, 'Changed');
        end
    """)
    assert s.first.figures == []  # an axes in an app is not an ordinary figure
    before = prop(s, "ax", "image")
    assert before.startswith("data:image/png;base64,")
    assert prop(s, "ax", "alt").startswith("Squares")
    r = send(s, "b")
    ax = ui.find(s.interp, tag="ax")
    patch = r["update"]["patch"][str(ax.uid)]
    assert patch["image"] != before and patch["alt"].startswith("Changed (2")
    assert s.run("disp(class(ax)); disp(ax.XLim); disp(numel(ax.Children)); disp(ax.Layout.Row)").output.split() == [
        "matlab.ui.control.UIAxes", "0", "4", "2", "1"]


def test_axes_size_event_changes_the_rendered_size():
    s = start("f = uifigure; g = uigridlayout(f, [1 1]); ax = uiaxes(g, 'Tag', 'ax'); plot(ax, 1:3);")
    ax = ui.find(s.interp, tag="ax")
    r = s.ui_event({"id": ax.uid, "event": "size", "width": 300, "height": 200})
    assert r["update"]["patch"][str(ax.uid)]["width"] == 300


def test_ordinary_figures_drawn_by_callbacks_are_returned():
    s = start("f = uifigure; uibutton(f, 'Tag', 'b', 'ButtonPushedFcn', @(s, e) plot(1:3));")
    assert len(send(s, "b")["figures"]) == 1


def test_close_request_and_delete():
    s = start("""
        f = uifigure('Tag', 'f', 'CloseRequestFcn', @(src, ev) disp('asked'));
        uibutton(f, 'Tag', 'quit', 'ButtonPushedFcn', @(src, ev) delete(f));
    """)
    r = send(s, "f", "close")
    assert r["output"] == "asked\n" and s.ui_snapshot() is not None
    r = send(s, "quit")
    assert r["update"] == {"full": True, "figures": [], "dialogs": []}
    assert s.ui_snapshot() is None
    s = start("f = uifigure('Tag', 'f'); uilabel(f);")
    send(s, "f", "close")
    assert s.ui_snapshot() is None


def test_size_changed():
    s = start("f = uifigure('Tag', 'f', 'SizeChangedFcn', @(src, ev) fprintf('%d x %d\\n', src.Position(3:4)));")
    assert send(s, "f", "resize", width=500, height=300)["output"] == "500 x 300\n"
    assert send(s, "f", "resize", width=500, height=300)["output"] == ""


def test_confirm_with_close_function_is_answered_later():
    s = start("""
        f = uifigure; l = uilabel(f, 'Tag', 'l');
        uiconfirm(f, 'Save?', 'Quit', 'Options', {'Save', 'Discard'}, ...
            'CloseFcn', @(src, ev) set(l, 'Text', [ev.SelectedOption ' ' num2str(ev.SelectedOptionIndex)]));
    """)
    d = s.first.ui["dialogs"][0]
    assert d["kind"] == "confirm" and d["options"] == ["Save", "Discard"]
    r = s.ui_event({"id": d["id"], "event": "dialog", "value": "Discard"})
    assert r["update"]["dialogs"] == []
    assert prop(s, "l", "text") == "Discard 2"


def test_gcbo_gcbf_inside_callbacks():
    s = start("""
        f = uifigure('Name', 'W');
        uibutton(f, 'Tag', 'b', 'Text', 'me', 'ButtonPushedFcn', @(s, e) fprintf('%s %s %d\\n', ...
            get(gcbo, 'Text'), get(gcbf, 'Name'), isempty(gcbo)));
        disp(isempty(gcbo));
    """)
    assert s.first.output.strip() == "1"
    assert send(s, "b")["output"] == "me W 0\n"


def test_unknown_ids_and_events_are_ignored():
    s = start("f = uifigure; uibutton(f);")
    assert s.ui_event({"id": 9999, "event": "push"})["error"] is None
    assert s.ui_event({"id": "x", "event": "nothing"})["error"] is None
    assert s.ui_event({})["error"] is None
