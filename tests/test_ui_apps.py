"""Whole apps: class files in the style App Designer exports, and the example apps."""

import glob
import os
import textwrap

import pytest

from mylab import ui
from mylab.api import Session

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASS_APP = """
classdef TempApp < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure         matlab.ui.Figure
        GridLayout       matlab.ui.container.GridLayout
        CelsiusField     matlab.ui.control.NumericEditField
        ConvertButton    matlab.ui.control.Button
        ResultLabel      matlab.ui.control.Label
        UIAxes           matlab.ui.control.UIAxes
        UnitDropDown     matlab.ui.control.DropDown
    end

    properties (Access = private)
        History = [] % values converted so far
    end

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            app.ResultLabel.Text = 'Ready';
            plot(app.UIAxes, 0, 32, 'o');
        end

        % Button pushed function: ConvertButton
        function ConvertButtonPushed(app, event)
            c = app.CelsiusField.Value;
            if strcmp(app.UnitDropDown.Value, 'Kelvin')
                r = c + 273.15;
            else
                r = c * 9 / 5 + 32;
            end
            app.History(end + 1) = r;
            app.ResultLabel.Text = sprintf('%.1f %s', r, app.UnitDropDown.Value);
            plot(app.UIAxes, app.History, '-o');
            title(app.UIAxes, sprintf('%d conversions', numel(app.History)));
        end

        % Value changed function: UnitDropDown
        function UnitDropDownValueChanged(app, event)
            app.ResultLabel.Text = ['Unit: ' event.Value];
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 420 360];
            app.UIFigure.Name = 'MATLAB App';

            % Create GridLayout
            app.GridLayout = uigridlayout(app.UIFigure);
            app.GridLayout.ColumnWidth = {'1x', '1x', '1x'};
            app.GridLayout.RowHeight = {22, 22, '1x'};

            % Create CelsiusField
            app.CelsiusField = uieditfield(app.GridLayout, 'numeric');
            app.CelsiusField.Tag = 'celsius';
            app.CelsiusField.Layout.Row = 1;
            app.CelsiusField.Layout.Column = 1;

            % Create UnitDropDown
            app.UnitDropDown = uidropdown(app.GridLayout);
            app.UnitDropDown.Items = {'Fahrenheit', 'Kelvin'};
            app.UnitDropDown.ValueChangedFcn = createCallbackFcn(app, @UnitDropDownValueChanged, true);
            app.UnitDropDown.Tag = 'unit';
            app.UnitDropDown.Layout.Row = 1;
            app.UnitDropDown.Layout.Column = 2;
            app.UnitDropDown.Value = 'Fahrenheit';

            % Create ConvertButton
            app.ConvertButton = uibutton(app.GridLayout, 'push');
            app.ConvertButton.ButtonPushedFcn = createCallbackFcn(app, @ConvertButtonPushed, true);
            app.ConvertButton.Tag = 'convert';
            app.ConvertButton.Layout.Row = 1;
            app.ConvertButton.Layout.Column = 3;
            app.ConvertButton.Text = 'Convert';

            % Create ResultLabel
            app.ResultLabel = uilabel(app.GridLayout);
            app.ResultLabel.HorizontalAlignment = 'center';
            app.ResultLabel.FontWeight = 'bold';
            app.ResultLabel.Tag = 'result';
            app.ResultLabel.Layout.Row = 2;
            app.ResultLabel.Layout.Column = [1 3];
            app.ResultLabel.Text = '';

            % Create UIAxes
            app.UIAxes = uiaxes(app.GridLayout);
            title(app.UIAxes, 'Title')
            xlabel(app.UIAxes, 'X')
            ylabel(app.UIAxes, 'Y')
            zlabel(app.UIAxes, 'Z')
            app.UIAxes.Tag = 'axes';
            app.UIAxes.Layout.Row = 3;
            app.UIAxes.Layout.Column = [1 3];

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = TempApp

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end
"""


def send(s, tag, event="push", **more):
    n = ui.find(s.interp, tag=tag)
    r = s.ui_event({"id": n.uid, "event": event, **more})
    assert r["error"] is None, r["error"]
    return r


def text_of(s, tag):
    return ui.find(s.interp, tag=tag).serialize()["props"].get("text")


def test_app_designer_style_class_app(tmp_path):
    (tmp_path / "TempApp.m").write_text(CLASS_APP)
    s = Session(cwd=str(tmp_path))
    r = s.run("TempApp")
    assert r.error is None, r.error
    assert r.output == ""  # nargout == 0: nothing is displayed
    fig = r.ui["figures"][0]
    assert fig["props"]["name"] == "MATLAB App" and "visible" not in fig["props"]
    assert text_of(s, "result") == "Ready"
    send(s, "celsius", "value", value="100")
    send(s, "convert")
    assert text_of(s, "result") == "212.0 Fahrenheit"
    r = send(s, "unit", "value", value=1)
    assert text_of(s, "result") == "Unit: Kelvin"
    r = send(s, "convert")
    assert text_of(s, "result") == "373.1 Kelvin"
    ax = ui.find(s.interp, tag="axes")
    assert r["update"]["patch"][str(ax.uid)]["alt"].startswith("2 conversions")

    # The app object can also be kept and used from the command line.
    r = s.run("app = TempApp; app.CelsiusField.Value = 10; disp(class(app.UIFigure)); delete(app); "
              "disp(isvalid(app))")
    assert r.error is None, r.error
    assert r.output.split() == ["matlab.ui.Figure", "0"]
    assert len(r.ui["figures"]) == 1  # the first app is still open, the second was deleted


def test_handle_class_app_with_method_callbacks(tmp_path):
    (tmp_path / "Clicker.m").write_text(textwrap.dedent("""
        classdef Clicker < handle
            properties
                Fig
                Label
                Count = 0
            end
            methods
                function obj = Clicker()
                    obj.Fig = uifigure('Name', 'Clicker');
                    obj.Label = uilabel(obj.Fig, 'Tag', 'label', 'Text', '0');
                    uibutton(obj.Fig, 'Tag', 'a', 'ButtonPushedFcn', @(src, event) obj.bump(1));
                    uibutton(obj.Fig, 'Tag', 'b', 'ButtonPushedFcn', @obj.onPush);
                end
                function bump(obj, n)
                    obj.Count = obj.Count + n;
                    obj.Label.Text = num2str(obj.Count);
                end
                function onPush(obj, src, event)
                    obj.bump(10);
                end
            end
        end
    """))
    s = Session(cwd=str(tmp_path))
    r = s.run("c = Clicker;")
    assert r.error is None, r.error
    send(s, "a")
    send(s, "b")
    assert text_of(s, "label") == "11"
    assert s.run("disp(c.Count)").output.strip() == "11"


def test_mlapp_files_are_explained(tmp_path):
    (tmp_path / "thing.mlapp").write_bytes(b"PK\x03\x04binary")
    s = Session(cwd=str(tmp_path))
    r = s.run("run('thing.mlapp')")
    assert r.error is not None and "Export" in r.error and ".mlapp" in r.error


EXAMPLES = sorted(glob.glob(os.path.join(ROOT, "examples", "apps", "*.m")))


def test_there_are_example_apps():
    assert len(EXAMPLES) >= 3


@pytest.mark.parametrize("path", EXAMPLES, ids=[os.path.basename(p) for p in EXAMPLES])
def test_example_apps_start_and_describe_a_window(path):
    s = Session(cwd=os.path.dirname(path))
    r = s.run(os.path.splitext(os.path.basename(path))[0])
    assert r.error is None, r.error
    assert r.ui is not None and len(r.ui["figures"]) == 1
    assert r.ui["figures"][0]["children"]


def test_unit_converter_example():
    s = Session(cwd=os.path.join(ROOT, "examples", "apps"))
    with open(os.path.join(ROOT, "examples", "apps", "unit_converter.m"), encoding="utf-8") as fh:
        assert s.run(fh.read()).error is None
    send(s, "input", "value", value="10")
    assert text_of(s, "result").startswith("10 km = 6.2137 mi")
    send(s, "swap")
    assert text_of(s, "result").startswith("10 mi = 16.093 km")


def test_plot_explorer_example():
    s = Session(cwd=os.path.join(ROOT, "examples", "apps"))
    with open(os.path.join(ROOT, "examples", "apps", "plot_explorer.m"), encoding="utf-8") as fh:
        assert s.run(fh.read()).error is None
    ax = ui.find(s.interp, type="axes")
    r = send(s, "frequency", "value", value=5)
    assert "image" in r["update"]["patch"][str(ax.uid)]
    assert "5" in text_of(s, "readout")
    r = send(s, "frequency", "value", value=7, changing=True)
    assert "7" in text_of(s, "readout")


def test_todo_list_example():
    s = Session(cwd=os.path.join(ROOT, "examples", "apps"))
    assert s.run("todo_list").error is None
    table = ui.find(s.interp, type="table")
    before = len(table.serialize()["props"]["rows"])
    send(s, "newtask", "value", value="Write tests")
    send(s, "add")
    rows = table.serialize()["props"]["rows"]
    assert len(rows) == before + 1 and rows[-1][0] == "Write tests" and rows[-1][1] is False
    s.ui_event({"id": table.uid, "event": "cell", "row": before, "col": 1, "value": True})
    assert "1 of" in text_of(s, "status") or "done" in text_of(s, "status")
    send(s, "clear")
    assert len(table.serialize()["props"]["rows"]) == before
