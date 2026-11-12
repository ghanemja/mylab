"""Help text for dialog functions."""

_HOW = ("In a terminal or notebook the question is asked as text; when nobody can answer (for example in a "
        "batch run) the default is used and a note is printed.")
_NO_GUI = ("Windows with buttons and other controls cannot be built in mylab: this raises the error "
           "'Graphical user interfaces are not supported in mylab'. Use input, menu, questdlg or inputdlg "
           "for simple interaction.")
_BOX = "Prints the message with the label '{0}' (there is no pop-up window) and returns a handle that close " \
       "and delete accept. The message can be text or a cell array of lines."

HELP = {
    "waitbar": (["h = waitbar(x, message)", "waitbar(x, h)", "waitbar(x, h, message)", "waitbar(x)"],
                "Shows progress x between 0 and 1. In a terminal a single text progress bar is redrawn in "
                "place; elsewhere nothing is printed. Update the bar by passing its handle h (or nothing, "
                "for the most recent bar), and finish with close(h) or delete(h). Example: h = waitbar(0, "
                "'Working'); for k = 1:10, waitbar(k/10, h); end; close(h)"),
    "msgbox": (["msgbox(message)", "msgbox(message, title)", "msgbox(message, title, icon)", "h = msgbox(...)"],
               _BOX.format("Message") + " icon 'error', 'warn' or 'help' changes the label."),
    "errordlg": (["errordlg(message)", "errordlg(message, title)", "h = errordlg(...)"],
                 _BOX.format("Error") + " The script continues; use error to stop it."),
    "warndlg": (["warndlg(message)", "warndlg(message, title)", "h = warndlg(...)"], _BOX.format("Warning")),
    "helpdlg": (["helpdlg(message)", "helpdlg(message, title)", "h = helpdlg(...)"], _BOX.format("Help")),
    "questdlg": (["answer = questdlg(question)", "answer = questdlg(question, title)",
                  "answer = questdlg(question, title, default)",
                  "answer = questdlg(question, title, btn1, btn2, default)",
                  "answer = questdlg(question, title, btn1, btn2, btn3, default)"],
                 "Asks a question with up to three choices ('Yes', 'No', 'Cancel' unless others are given) "
                 "and returns the chosen text. " + _HOW + " Type the number or the name of a choice; an "
                 "empty reply picks the default."),
    "inputdlg": (["answers = inputdlg(prompt)", "answers = inputdlg(prompts, title)",
                  "answers = inputdlg(prompts, title, dims, defaults)"],
                 "Asks for one line of text per prompt (prompts is text or a cell array) and returns the "
                 "answers as a column cell array of character vectors; use str2double for numbers. An empty "
                 "reply keeps that prompt's default. " + _HOW + " A front end that cancels returns {}."),
    "listdlg": (["[idx, ok] = listdlg('ListString', items)",
                 "[idx, ok] = listdlg('ListString', items, 'SelectionMode', 'single')",
                 "listdlg(..., 'InitialValue', k, 'PromptString', text, 'Name', title)"],
                "Lets the user choose from a numbered list and returns the chosen item numbers and ok = 1, "
                "or [] and ok = 0 after a cancel. The default selection is InitialValue (item 1). " + _HOW),
    "menu": (["k = menu(title, choice1, choice2, ...)", "k = menu(title, choices)"],
             "Shows numbered choices and returns the number picked, or 0 if no valid choice was made. "
             "When nobody can answer, the first choice is used."),
    "uigetfile": (["[file, folder] = uigetfile", "[file, folder] = uigetfile(filter)",
                   "[file, folder] = uigetfile(filter, title, defaultName)",
                   "[file, folder, idx] = uigetfile(..., 'MultiSelect', 'on')"],
                  "Asks for an existing file by having its path typed, and returns the file name and its "
                  "folder (ending with a file separator); join them with fullfile. Both are 0 after a "
                  "cancel (an empty reply), so test isequal(file, 0). When nobody can answer, the default "
                  "name is used if that file exists, otherwise the result is a cancel. With MultiSelect, "
                  "several paths separated by semicolons return a cell array of names."),
    "uiputfile": (["[file, folder] = uiputfile", "[file, folder] = uiputfile(filter, title, defaultName)"],
                  "Asks for the name of a file to save to (typed as a path) and returns the name and folder, "
                  "or 0 and 0 after a cancel. When nobody can answer, the default name is used if one was "
                  "given."),
    "uigetdir": (["folder = uigetdir", "folder = uigetdir(start, title)"],
                 "Asks for a folder (typed as a path) and returns it, or 0 after a cancel. An empty reply, or "
                 "nobody to answer, gives the start folder (the current folder when none is given)."),
    "uiwait": (["uiwait", "uiwait(h)"],
               "Returns immediately: mylab dialogs have already been answered when they return, so there is "
               "nothing to wait for."),
    "uiresume": (["uiresume", "uiresume(h)"], "Has no effect; see uiwait."),
    "waitfor": (["waitfor(h)", "waitfor(h, property)"], "Returns immediately; see uiwait."),
    "waitforbuttonpress": ("k = waitforbuttonpress",
                           "Waits for a mouse click (0) or key press (1) in a plot window. Without a live "
                           "window it returns 0 at once."),
    "ginput": (["[x, y] = ginput(n)", "[x, y] = ginput", "[x, y, button] = ginput(n)", "xy = ginput(n)"],
               "Reads n mouse clicks (or clicks until Enter) from the current figure window and returns "
               "their data coordinates as column vectors. It needs a live plot window; elsewhere it raises "
               "an error."),
}
for _n in ("uicontrol", "uipanel", "uimenu", "uifigure", "guide", "uitable", "uibuttongroup", "uitab",
           "uitabgroup", "uiaxes", "uibutton", "uilabel", "uieditfield", "uidropdown", "uislider", "uicheckbox",
           "uitoolbar", "uipushtool", "uicontextmenu", "appdesigner", "dialog", "guihandles"):
    HELP[_n] = (f"{_n}(...)", _NO_GUI)
