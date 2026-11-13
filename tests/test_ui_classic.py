"""Classic GUIs: figure + uicontrol, guidata, guihandles, gcbo/gcbf, units."""

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


def props(s: Session, tag: str) -> dict:
    return ui.find(s.interp, tag=tag).serialize()["props"]


def test_uicontrol_styles_map_to_shared_component_types():
    s = start("""
        f = figure('Name', 'Classic', 'Position', [100 100 300 400], 'MenuBar', 'none', 'NumberTitle', 'off');
        uicontrol('Style', 'pushbutton', 'String', 'Go', 'Position', [10 10 80 25]);
        uicontrol(f, 'Style', 'edit', 'String', 'abc');
        uicontrol('Parent', f, 'Style', 'text', 'String', 'Label');
        uicontrol('Style', 'slider', 'Min', 0, 'Max', 10, 'Value', 4);
        uicontrol('Style', 'checkbox', 'String', 'Yes', 'Value', 1);
        uicontrol('Style', 'popupmenu', 'String', {'a', 'b', 'c'}, 'Value', 2);
        uicontrol('Style', 'listbox', 'String', 'x|y|z', 'Value', 3);
        uicontrol('Style', 'radiobutton', 'String', 'r');
        uicontrol('Style', 'togglebutton', 'String', 't');
        uicontrol('Style', 'edit', 'Max', 3, 'String', {'two'; 'lines'});
    """)
    assert s.first.figures == []  # the figure is an app window, not a plot
    fig = s.first.ui["figures"][0]
    assert fig["props"]["name"] == "Classic" and fig["props"]["classic"] is True
    assert (fig["props"]["width"], fig["props"]["height"]) == (300, 400)
    kids = fig["children"]
    assert [k["type"] for k in kids] == ["button", "editfield", "label", "slider", "checkbox", "dropdown",
                                         "listbox", "radiobutton", "statebutton", "textarea"]
    assert kids[0]["props"]["text"] == "Go" and kids[0]["props"]["pos"] == [9, 9, 80, 25]
    assert kids[1]["props"]["value"] == "abc"
    assert kids[3]["props"]["value"] == 4 and kids[3]["props"]["max"] == 10
    assert kids[4]["props"]["value"] is True
    assert kids[5]["props"]["items"] == ["a", "b", "c"] and kids[5]["props"]["selected"] == 1
    assert kids[6]["props"]["items"] == ["x", "y", "z"] and kids[6]["props"]["selected"] == [2]
    assert kids[9]["props"]["value"] == "two\nlines"


def test_classic_callbacks_get_and_set():
    s = start("""
        f = figure;
        e = uicontrol('Style', 'edit', 'Tag', 'in', 'String', '2');
        t = uicontrol('Style', 'text', 'Tag', 'out', 'String', '');
        uicontrol('Style', 'pushbutton', 'Tag', 'go', 'String', 'Square', 'Callback', {@square, e, t});
        uicontrol('Style', 'slider', 'Tag', 'sl', 'Min', 0, 'Max', 1, 'Callback', ...
            @(src, ~) set(t, 'String', sprintf('%.2f', get(src, 'Value'))));
        uicontrol('Style', 'popupmenu', 'Tag', 'pop', 'String', {'red', 'green'}, 'Callback', ...
            @(src, ~) set(t, 'String', src.String{src.Value}));
        uicontrol('Style', 'checkbox', 'Tag', 'chk', 'Callback', 'disp(get(gcbo, ''Value''))');
        function square(src, event, e, t)
            x = str2double(get(e, 'String'));
            set(t, 'String', num2str(x^2));
        end
    """)
    send(s, "in", "value", value="7")
    send(s, "go")
    assert props(s, "out")["text"] == "49"
    send(s, "sl", "value", value=0.25)
    assert props(s, "out")["text"] == "0.25"
    send(s, "pop", "value", value=1)
    assert props(s, "out")["text"] == "green"
    assert send(s, "chk", "value", value=True)["output"].strip() == "1"
    assert send(s, "chk", "value", value=False)["output"].strip() == "0"


def test_guidata_and_guihandles():
    s = start("""
        f = figure('Tag', 'main');
        uicontrol('Style', 'text', 'Tag', 'display', 'String', '0');
        uicontrol('Style', 'pushbutton', 'Tag', 'inc', 'String', '+', 'Callback', @increase);
        handles = guihandles(f);
        handles.count = 0;
        guidata(f, handles);
        disp(fieldnames(handles)');
        function increase(hObject, eventdata)
            handles = guidata(hObject);
            handles.count = handles.count + 1;
            set(handles.display, 'String', num2str(handles.count));
            guidata(hObject, handles);
        end
    """)
    assert "display" in s.first.output and "inc" in s.first.output and "main" in s.first.output
    send(s, "inc")
    send(s, "inc")
    assert props(s, "display")["text"] == "2"
    assert s.run("d = guidata(f); disp(d.count)").output.strip() == "2"


def test_gcbo_gcbf_and_figure_properties_of_classic_windows():
    s = start("""
        f = figure('Name', 'Win');
        uicontrol('Tag', 'b', 'String', 'B', 'Callback', ...
            @(~, ~) fprintf('%s|%s|%d\\n', get(gcbo, 'String'), get(gcbf, 'Name'), gcbf == f));
        b = findobj(f, 'Tag', 'b'); disp(get(b, 'Style')); disp(class(get(b, 'Parent')));
        disp(class(ancestor(b, 'figure')));
    """)
    assert s.first.output.split() == ["pushbutton", "matlab.ui.Figure", "matlab.ui.Figure"]
    assert send(s, "b")["output"] == "B|Win|1\n"


def test_normalized_units_and_panels():
    s = start("""
        f = figure('Position', [0 0 400 200]);
        p = uipanel('Title', 'Options', 'Position', [0 0 0.5 1], 'Tag', 'p');
        b = uicontrol(p, 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.2], 'Tag', 'b');
        c = uicontrol(p, 'Units', 'characters', 'Position', [1 1 10 2], 'Tag', 'c');
        disp(get(p, 'Units'));
        set(b, 'Units', 'pixels'); disp(round(get(b, 'Position')));
    """)
    assert s.first.output.split() == ["normalized", "21", "21", "160", "40"]
    assert props(s, "p")["npos"] == [0.0, 0.0, 0.5, 1.0] and props(s, "p")["title"] == "Options"
    assert props(s, "c")["pos"] == [6.0, 13.0, 60.0, 26.0]


def test_classic_button_group():
    s = start("""
        f = figure;
        bg = uibuttongroup('Tag', 'bg', 'SelectionChangedFcn', ...
            @(src, ev) fprintf('%s -> %s\\n', ev.OldValue.String, ev.NewValue.String));
        uicontrol(bg, 'Style', 'radiobutton', 'String', 'one', 'Tag', 'r1');
        uicontrol(bg, 'Style', 'radiobutton', 'String', 'two', 'Tag', 'r2');
        disp(bg.SelectedObject.String);
    """)
    assert s.first.output.strip() == "one"
    assert send(s, "r2", "value", value=True)["output"] == "one -> two\n"
    assert props(s, "r1")["value"] is False


def test_axes_in_a_classic_figure_is_drawn_behind_the_controls():
    s = start("""
        f = figure('Position', [0 0 500 300]);
        ax = axes('Parent', f, 'Units', 'normalized', 'Position', [0.1 0.3 0.8 0.6]);
        h = plot(ax, 1:5, (1:5).^2);
        uicontrol('Style', 'pushbutton', 'Tag', 'flip', 'String', 'Flip', 'Position', [20 20 80 30], ...
            'Callback', @(~, ~) set(h, 'YData', fliplr(get(h, 'YData'))));
    """)
    fig = s.first.ui["figures"][0]
    assert fig["props"]["image"].startswith("data:image/png;base64,")
    r = send(s, "flip")
    assert "image" in r["update"]["patch"][str(fig["id"])]
    assert s.run("disp(h.YData(1))").output.strip() == "25"


def test_axes_inside_a_panel():
    s = start("""
        f = figure;
        p = uipanel(f, 'Position', [0.3 0 0.7 1]);
        ax = axes('Parent', p);
        plot(ax, [1 2 3]);
        disp(get(ax, 'Units'));
    """)
    assert s.first.output.strip() == "normalized"
    panel = s.first.ui["figures"][0]["children"][0]
    assert panel["children"][0]["type"] == "axes"


def test_plain_figures_are_still_plots_and_closing_works():
    s = start("figure; plot(1:3); f2 = figure; uicontrol('String', 'x', 'Tag', 'x');")
    assert len(s.first.figures) == 1  # the plot; the window with the button is the app
    assert len(s.first.ui["figures"]) == 1
    r = s.run("close(f2);")
    assert r.ui is None
    s = start("f = figure; uicontrol('String', 'x'); delete(f);")
    assert s.first.ui is None


def test_close_button_runs_close_request_fcn():
    s = start("""
        f = figure('Tag', 'f', 'CloseRequestFcn', @(src, ~) disp('closing'));
        uicontrol('String', 'x', 'Tag', 'x');
    """)
    node = ui.find(s.interp, type="figure")
    r = s.ui_event({"id": node.uid, "event": "close"})
    assert r["output"] == "closing\n" and s.ui_snapshot() is not None
    s.run("set(f, 'CloseRequestFcn', 'closereq');")
    r = s.ui_event({"id": node.uid, "event": "close"})
    assert r["update"]["figures"] == []


def test_uiwait_returns_without_a_front_end_and_resumes_with_one():
    s = start("f = figure; uicontrol('String', 'OK', 'Tag', 'ok', 'Callback', 'uiresume(gcbf)'); "
              "uiwait(f); disp('after');")
    assert s.first.output.strip() == "after"

    # A front end that can keep the window open installs a hook; here it presses OK.
    s = Session()
    calls = []

    def hook(interp, fig, dialog):
        calls.append(fig.uid in ui.state(interp).waiting)
        ok = ui.find(interp, tag="ok")
        ui.dispatch(interp, {"id": ok.uid, "event": "push"})
        calls.append(fig.uid in ui.state(interp).waiting)

    ui.state(s.interp).wait_hook = hook
    r = s.run("f = figure; uicontrol('String', 'OK', 'Tag', 'ok', 'Callback', 'uiresume(gcbf)'); "
              "uiwait(f); disp('after');")
    assert r.error is None and calls == [True, False]


def test_appdata():
    s = start("f = figure; setappdata(f, 'n', 5); disp(getappdata(f, 'n')); disp(isappdata(f, 'n')); "
              "b = uicontrol; setappdata(b, 'k', 'v'); disp(getappdata(b, 'k')); rmappdata(f, 'n'); "
              "disp(isappdata(f, 'n'))")
    assert s.first.output.split() == ["5", "1", "v", "0"]
