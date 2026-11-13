"""Help text for app windows (uifigure, uicontrol and their components)."""

_COMMON = (" Every component also has Position ([left bottom width height] in pixels), Visible, Enable, Tooltip, "
           "Tag, UserData, FontSize, FontWeight, FontColor and BackgroundColor, and Layout.Row / Layout.Column "
           "inside a grid layout. See docs/APPS.md.")
_WHERE = (" The window is shown by the mylab web app (App panel), or in your browser when the file is run with "
          "the mylab command.")


def _comp(name: str, what: str, props: str, example: str, styles: str = "") -> tuple:
    usage = [f"h = {name}(parent)", f"h = {name}(parent, Name, Value, ...)", f"h = {name}"]
    if styles:
        usage.insert(1, f"h = {name}(parent, style)")
    return usage, f"{what} {props}{_COMMON} Example: {example}"


HELP = {
    "uifigure": (["fig = uifigure", "fig = uifigure(Name, Value, ...)", "fig = uifigure('Name', title)"],
                 "Creates an app window and returns its handle. Put a uigridlayout in it, or place components "
                 "with Position. Properties: Name (the window title), Position, Color, Visible, Resize, "
                 "CloseRequestFcn (runs when the window's close button is pressed; without it the window is "
                 "deleted), SizeChangedFcn, UserData, Tag." + _WHERE +
                 " Example: fig = uifigure('Name', 'Demo'); uibutton(fig, 'Text', 'Hello');"),
    "uigridlayout": (["g = uigridlayout(parent)", "g = uigridlayout(parent, [rows columns])",
                      "g = uigridlayout(parent, Name, Value, ...)"],
                     "A layout that arranges components in rows and columns and fills its parent. New "
                     "components take the next free cell, row by row; move one with h.Layout.Row = 2 and "
                     "h.Layout.Column = [1 3] (a range spans several cells). RowHeight and ColumnWidth are "
                     "cell arrays with one entry per row or column: a number of pixels, 'fit' (as large as "
                     "the content) or a weight such as '1x' or '2x' (a share of the space that is left). "
                     "Other properties: Padding ([left bottom right top]), RowSpacing, ColumnSpacing, "
                     "Scrollable. Example: g = uigridlayout(fig, [2 2]); g.RowHeight = {'fit', '1x'};"),
    "uipanel": (["p = uipanel(parent)", "p = uipanel(parent, 'Title', text, ...)", "p = uipanel(Name, Value, ...)"],
                "A box with an optional Title that holds other components. Components inside it are placed "
                "relative to the panel. In a classic figure the default Units are 'normalized' and the "
                "default parent is the current figure; BorderType 'none' hides the border."),
    "uitabgroup": (["tg = uitabgroup(parent)", "tg = uitabgroup(parent, Name, Value, ...)"],
                   "A set of tabs. Add pages with uitab(tg, 'Title', text). SelectedTab is the page that is "
                   "shown (set it to switch pages); SelectionChangedFcn runs when the user switches, with "
                   "event.OldValue and event.NewValue."),
    "uitab": (["t = uitab(tabgroup)", "t = uitab(tabgroup, 'Title', text)"],
              "One page of a tab group. It holds components like a panel does. Example: tg = "
              "uitabgroup(fig); t = uitab(tg, 'Title', 'Data'); uitable(t);"),
    "uilabel": _comp("uilabel", "A line of text (a cell array or string array gives several lines).",
                     "Properties: Text, HorizontalAlignment ('left', 'center', 'right'), VerticalAlignment, "
                     "WordWrap.", "uilabel(fig, 'Text', 'Name:')"),
    "uibutton": (_comp("uibutton", "A push button, or a state button (uibutton(parent, 'state')) that stays "
                       "pressed.", "Properties: Text, Icon (an image file), ButtonPushedFcn (push button), "
                       "Value and ValueChangedFcn (state button).",
                       "uibutton(fig, 'Text', 'Run', 'ButtonPushedFcn', @(src, event) disp('pressed'))", "x")),
    "uieditfield": (_comp("uieditfield", "A box for one line of text, or for a number "
                          "(uieditfield(parent, 'numeric')).", "Properties: Value (text, or a number), "
                          "Placeholder, Editable, ValueChangedFcn (runs after Enter or when the field loses "
                          "the focus; event.Value and event.PreviousValue), ValueChangingFcn (while typing); "
                          "numeric fields also have Limits, RoundFractionalValues and ValueDisplayFormat. A "
                          "number outside Limits is not accepted.",
                          "f = uieditfield(fig, 'numeric', 'Limits', [0 100], 'Value', 5)", "x")),
    "uitextarea": _comp("uitextarea", "A box for several lines of text.", "Properties: Value (a cell array "
                        "with one line per cell), Editable, Placeholder, ValueChangedFcn.",
                        "uitextarea(fig, 'Value', {'first line'; 'second line'})"),
    "uidropdown": _comp("uidropdown", "A list that drops down to choose one item.", "Properties: Items (cell "
                        "array of text), ItemsData (optional values that stand for the items), Value (the "
                        "chosen item, or its ItemsData element), ValueIndex, Editable (the user may type a "
                        "new value), ValueChangedFcn.",
                        "d = uidropdown(fig, 'Items', {'Red', 'Green'}, 'ValueChangedFcn', @(s, e) disp(e.Value))"),
    "uilistbox": _comp("uilistbox", "A list that shows several items at once.", "Properties: Items, ItemsData, "
                       "Value, Multiselect ('on' lets the user pick several; Value is then a cell array), "
                       "ValueChangedFcn.", "uilistbox(fig, 'Items', {'a', 'b', 'c'}, 'Value', 'b')"),
    "uicheckbox": _comp("uicheckbox", "A check box.", "Properties: Text, Value (true or false), "
                        "ValueChangedFcn.", "uicheckbox(fig, 'Text', 'Show grid', 'Value', true)"),
    "uiradiobutton": (["r = uiradiobutton(buttongroup)", "r = uiradiobutton(buttongroup, 'Text', text, ...)"],
                      "A radio button inside a uibuttongroup; only one button of the group is selected. "
                      "Properties: Text, Value. React to the choice with the group's SelectionChangedFcn."),
    "uitogglebutton": (["t = uitogglebutton(buttongroup)", "t = uitogglebutton(buttongroup, 'Text', text, ...)"],
                       "A button inside a uibuttongroup that stays pressed while it is the selected one. "
                       "Properties: Text, Value."),
    "uibuttongroup": (["bg = uibuttongroup(parent)", "bg = uibuttongroup(parent, Name, Value, ...)"],
                      "A panel whose radio buttons and toggle buttons exclude each other. Properties: Title, "
                      "SelectedObject (the selected button), Buttons, SelectionChangedFcn (event.OldValue and "
                      "event.NewValue are the buttons). Works with uiradiobutton, uitogglebutton and with "
                      "uicontrol('Style', 'radiobutton')."),
    "uiswitch": (_comp("uiswitch", "An on/off switch.", "Properties: Items (two texts, {'Off', 'On'}), "
                       "ItemsData, Value (one of the items), ValueChangedFcn. The styles 'slider', 'rocker' "
                       "and 'toggle' look the same in mylab.", "s = uiswitch(fig); s.Value = 'On';", "x")),
    "uislider": _comp("uislider", "A slider for a number in a range.", "Properties: Value, Limits ([low "
                      "high]), Orientation ('horizontal' or 'vertical'), ValueChangedFcn (when the handle is "
                      "released), ValueChangingFcn (while it is dragged; read event.Value). A Value outside "
                      "Limits is moved to the nearest limit.",
                      "uislider(fig, 'Limits', [0 10], 'ValueChangedFcn', @(s, e) disp(e.Value))"),
    "uispinner": _comp("uispinner", "A number field with up and down arrows.", "Properties: Value, Limits, "
                       "Step, RoundFractionalValues, ValueChangedFcn, ValueChangingFcn.",
                       "uispinner(fig, 'Limits', [1 10], 'Step', 1, 'Value', 3)"),
    "uiknob": (_comp("uiknob", "A knob for a number in a range. mylab shows it as a slider.", "Properties: "
                     "Value, Limits, ValueChangedFcn, ValueChangingFcn. uiknob(parent, 'discrete') chooses "
                     "one of Items instead and is shown as a drop-down list.", "uiknob(fig, 'Limits', [0 11])", "x")),
    "uigauge": (_comp("uigauge", "A gauge that shows a number in a range (read-only).", "Properties: Value, "
                      "Limits. The styles 'circular', 'linear', 'ninetydegree' and 'semicircular' are "
                      "all shown as a bar with the value next to it.", "g = uigauge(fig); g.Value = 40;", "x")),
    "uilamp": _comp("uilamp", "A small colored light.", "Properties: Color (a color name, hex code or [r g b]).",
                    "lamp = uilamp(fig); lamp.Color = 'red';"),
    "uitable": (["t = uitable(parent)", "t = uitable(parent, 'Data', data, ...)", "t = uitable(Name, Value, ...)"],
                "A table of values. Data is a numeric or logical matrix, a cell array, a string array or a "
                "table. Other properties: ColumnName (cell array of headings), RowName, ColumnEditable (true "
                "or false, or one value per column), ColumnFormat, CellEditCallback (runs after the user "
                "changes a cell: event.Indices is [row column], event.PreviousData, event.NewData and "
                "event.EditData the typed text), CellSelectionCallback and SelectionChangedFcn. Logical "
                "columns are shown as check boxes. An edit is stored in Data before the callback runs. "
                "Example: uitable(fig, 'Data', magic(3), 'ColumnEditable', true)"),
    "uiaxes": (["ax = uiaxes(parent)", "ax = uiaxes(parent, Name, Value, ...)"],
               "An axes inside an app window. Pass it as the first input of the plotting functions: plot(ax, "
               "x, y), title(ax, 'Signal'), xlabel(ax, 't'), hold(ax, 'on'), cla(ax), imagesc(ax, C). It has "
               "the usual axes properties (XLim, YLim, Title, ...) plus Position, Layout, Visible and Tag. "
               "The front end shows the drawing as an image that is refreshed after every callback."),
    "uiimage": _comp("uiimage", "A picture.", "Properties: ImageSource (a file name such as 'logo.png', or an "
                     "m-by-n-by-3 array), ScaleMethod ('fit', 'fill', 'none', 'stretch'), ImageClickedFcn.",
                     "uiimage(fig, 'ImageSource', 'logo.png')"),
    "uidatepicker": _comp("uidatepicker", "A field for choosing a date.", "Properties: Value (a datetime, NaT "
                          "when empty), ValueChangedFcn.", "d = uidatepicker(fig); d.Value = datetime(2024, 5, 17);"),
    "uihyperlink": _comp("uihyperlink", "A link that opens a web address.", "Properties: Text, URL, "
                         "HyperlinkClickedFcn.", "uihyperlink(fig, 'Text', 'Docs', 'URL', 'https://example.org')"),
    "uimenu": (["m = uimenu(fig, 'Text', text)", "item = uimenu(m, 'Text', text, 'MenuSelectedFcn', fcn)"],
               "A menu in the menu bar of a window, or an item of a menu. Properties: Text (classic name: "
               "Label), MenuSelectedFcn (classic name: Callback), Checked, Separator, Enable."),
    "uialert": (["uialert(fig, message, title)", "uialert(fig, message, title, 'Icon', icon, 'CloseFcn', fcn)"],
                "Shows a message box in front of an app window. The code continues at once; CloseFcn runs "
                "when the user closes the box. Icon is 'error' (default), 'warning', 'info' or 'success'."),
    "uiconfirm": (["answer = uiconfirm(fig, message, title)",
                   "answer = uiconfirm(fig, message, title, 'Options', {'Yes', 'No'}, 'DefaultOption', 1)",
                   "uiconfirm(fig, message, title, ..., 'CloseFcn', fcn)"],
                  "Asks a question with buttons. Run with the mylab command, the answer is the button the "
                  "user presses. In the web app and other places that cannot wait for a click, the call with "
                  "an output returns DefaultOption at once and prints a note; use the CloseFcn form there: "
                  "the dialog is shown and fcn(src, event) runs later with event.SelectedOption and "
                  "event.SelectedOptionIndex."),
    "uiprogressdlg": (["d = uiprogressdlg(fig, 'Title', text, 'Message', text)",
                       "d = uiprogressdlg(fig, 'Indeterminate', 'on', 'Cancelable', 'on')"],
                      "A progress box in front of an app window. Set d.Value (0 to 1) and d.Message as the "
                      "work goes on, read d.CancelRequested, and finish with close(d). The front end only "
                      "redraws after a callback returns, so a box that is opened and closed inside one "
                      "callback is never seen."),
    "uicontrol": (["h = uicontrol('Style', style, Name, Value, ...)", "h = uicontrol(parent, Name, Value, ...)"],
                  "A classic control in a figure window (or panel). Style is 'pushbutton' (default), "
                  "'togglebutton', 'checkbox', 'radiobutton', 'edit', 'text', 'slider', 'popupmenu' or "
                  "'listbox'. Properties: String (the text; the list of items as a cell array or 'a|b|c' for "
                  "pop-up menus and list boxes), Value (the slider value, the item number, or Max/Min for "
                  "check boxes and toggles), Min, Max, SliderStep, Callback, Position, Units ('pixels', "
                  "'normalized', 'characters', ...), FontSize, ForegroundColor, BackgroundColor, Enable, "
                  "Visible, Tag, UserData, TooltipString. Read and change them with get and set or with a "
                  "dot. Callback is a function handle called with (source, eventdata), a cell array {@fcn, "
                  "extra, ...} or text to evaluate. Without a parent the control goes into the current "
                  "figure. Example: uicontrol('Style', 'pushbutton', 'String', 'OK', 'Position', [20 20 80 "
                  "30], 'Callback', @(src, ~) disp('OK'))"),
    "guidata": (["data = guidata(h)", "guidata(h, data)"],
                "Stores one value (usually a struct) with the window that h belongs to, or returns it. h can "
                "be the window or any component in it, so callbacks use it to share data: handles = "
                "guidata(hObject); handles.count = handles.count + 1; guidata(hObject, handles);"),
    "guihandles": (["handles = guihandles(h)", "handles = guihandles"],
                   "A struct with one field per Tag in the window of h; each field holds the component with "
                   "that tag (several components when they share a tag). Tags that are not valid field names "
                   "are skipped."),
    "gcbo": (["h = gcbo", "[h, fig] = gcbo"],
             "The component whose callback is running, and its window; empty outside callbacks."),
    "gcbf": ("fig = gcbf", "The window that holds the component whose callback is running; empty outside "
                           "callbacks."),
    "closereq": ("closereq", "Deletes the window whose callback is running (the default CloseRequestFcn)."),
    "uiwait": (["uiwait(fig)", "uiwait"],
               "Waits until uiresume(fig) is called or the window is closed. This works when the app is run "
               "with the mylab command, where the window stays open while the script waits. In the web app, "
               "in notebooks and when there is no window, it returns at once: write the rest of the work in "
               "callbacks instead."),
    "uiresume": (["uiresume(fig)", "uiresume"], "Lets a waiting uiwait(fig) return."),
    "waitfor": (["waitfor(h)", "waitfor(h, property)"],
                "waitfor(h) behaves like uiwait(h). With a property name it returns at once."),
    "registerApp": ("registerApp(app, fig)",
                    "Part of the code App Designer exports: notes that fig is the window of the app object. "
                    "mylab runs exported app classes (classdef MyApp < matlab.apps.AppBase)."),
    "runStartupFcn": ("runStartupFcn(app, @startupFcn)",
                      "Part of the code App Designer exports: calls the app's startup method."),
    "createCallbackFcn": ("fcn = createCallbackFcn(app, @method, passEvent)",
                          "Part of the code App Designer exports: returns a callback that calls method(app, "
                          "event), or method(app) when passEvent is false."),
    "guide": ("guide", "Not available: GUIDE and its .fig files are not supported. Build the window in code "
                       "with figure and uicontrol, or with uifigure."),
    "appdesigner": ("appdesigner", "Not available. .mlapp files are binary and cannot be opened; export the "
                                   "app to a .m file in MATLAB and run that. The mylab web app has its own "
                                   "app builder (New app)."),
}

for _n in ("uitoolbar", "uipushtool", "uitoggletool", "uicontextmenu", "uitree", "uitreenode", "uihtml",
           "uistyle", "uirangeslider", "dialog"):
    HELP[_n] = (f"{_n}(...)", "Not available in mylab's app windows; calling it raises an error.")
