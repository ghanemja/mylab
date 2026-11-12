# mylab in VS Code

The mylab extension lets you run, check and debug `.m` files in
[Visual Studio Code](https://code.visualstudio.com). It also works in editors
built on VS Code, such as VSCodium and Cursor.

## Install

You need two things: mylab itself, and the extension.

### 1. Install mylab

mylab needs Python 3.10 or newer. In a terminal:

```bash
pip install mylab-lang
```

Check that it works:

```bash
mylab --version
```

### 2. Install the extension

**From the Marketplace** (when the extension is published there):

1. Open the Extensions view (`Ctrl+Shift+X`, or `Cmd+Shift+X` on a Mac).
2. Search for **mylab**.
3. Select **Install**.

**From a `.vsix` file** (always works):

1. Download `mylab-X.Y.Z.vsix` from the
   [latest release](https://github.com/ghanemja/mylab/releases/latest).
2. In the Extensions view, open the **…** menu at the top and choose
   **Install from VSIX…**.
3. Pick the file you downloaded.

Or, in a terminal:

```bash
code --install-extension mylab-X.Y.Z.vsix
```

### 3. Open a `.m` file

The language shown at the bottom right of the window should be
**MATLAB (mylab)**. If it shows something else, see
[Another extension opens my `.m` files](#another-extension-opens-my-m-files).

## Run code

| Action | Keys (Windows, Linux) | Keys (Mac) | Also |
|---|---|---|---|
| Run the file | `Ctrl+Alt+Enter` | `Cmd+Alt+Enter` | ▶ button at the top right of the editor |
| Run the selection, or the current section | `Shift+Enter` | `Shift+Enter` | ▶ menu, right-click menu |
| Debug the file | `Ctrl+Alt+Shift+Enter` | `Cmd+Alt+Shift+Enter` | ▶ menu |
| Show help for the name at the cursor | `Shift+F1` | `Shift+F1` | right-click menu |
| Open the interactive prompt | — | — | Command Palette → **mylab: Open Interactive Prompt** |

Every command is also in the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`)
under **mylab:**.

These keys were chosen so that they do not replace VS Code's own shortcuts.
(`Ctrl+Enter` and `F5` already mean something in VS Code.) To change them, open
**File → Preferences → Keyboard Shortcuts** and search for `mylab`.

### Run a file

The file is saved, then run with `mylab yourfile.m` in a terminal panel. The
program runs in the file's folder, so it can read data files and call other
`.m` files that are next to it. Plots open in their own windows.

### Run a section or a selection

A **section** is the part of a file between two lines that start with `%%`:

```matlab
%% Load the data
t = linspace(0, 1, 200);
y = sin(2*pi*5*t);

%% Plot it
plot(t, y)
```

Put the cursor in a section and press `Shift+Enter` to run that section only.
If you select text first, only the selection runs.

Things to know:

- Each run starts fresh. Variables from an earlier run are not kept. A section
  that needs `t` must create `t` itself. To keep variables between steps, use
  the interactive prompt, or a [Jupyter notebook](JUPYTER.md).
- Line numbers in error messages match your file.
- Functions defined at the end of the file are not part of a section. Put
  functions that sections share in their own `.m` files.
- The file does not need to be saved. The code is copied to a temporary file.

### Debug a file

**mylab: Debug File** starts mylab's debugger in a terminal. It pauses at the
first line. Type `h` and press Enter to list the debugger commands (step, run
to a line, show variables, and so on).

### Interactive prompt

**mylab: Open Interactive Prompt** opens a terminal with the `>>` prompt, in
the folder of the current file.

## Find problems before you run

The extension checks your file with `mylab --check`. This does not run your
code. Problems are underlined in the editor and listed in the Problems panel
(`Ctrl+Shift+M`):

- **Errors** (red): syntax errors, such as a missing `end` or bracket.
- **Warnings** (yellow): a call to a function that mylab does not have and that
  is not a `.m` file in the same folder. This catches spelling mistakes and
  functions from toolboxes that mylab does not provide.

Files are checked when you open them, when you save them, and a moment after
you stop typing. To check only on open and save, turn off the setting
`mylab.checkOnType`.

## Help, completion and editing

- **Hover**: point at a function name to see its calling forms and a
  description. This works offline for mylab's built-in functions. For a
  function in a `.m` file in the same folder, the hover shows the comment at
  the top of that file.
- **Help page**: `Shift+F1` opens the same help next to your code.
- **Completion**: start typing to see function names with a one-line summary,
  and keywords.
- **Snippets**: type `for`, `if`, `ifelse`, `while`, `switch`, `function`,
  `try` or `section`, and choose the snippet to insert the whole block.
- **Call hints**: after `(` and `,` the extension shows the function's arguments.
- **Folding**: fold `if`/`for`/`while`/`switch`/`try`/`function`/`classdef`
  blocks, `%%` sections and `%{ … %}` comments with the arrows next to the
  line numbers.
- **Outline**: the Outline view and `Ctrl+Shift+O` list functions and sections.
- **Comments**: `Ctrl+/` adds or removes `%`.

The single quote `'` is not closed automatically, because in this language it
also means "transpose" (`x'`).

## Settings

Open **File → Preferences → Settings** and search for `mylab`.

| Setting | Default | What it does |
|---|---|---|
| `mylab.executable` | `mylab` | The command that starts mylab. Use a name on your PATH, a full path, or a command line such as `python -m mylab`. |
| `mylab.checkOnType` | on | Check for problems while you type. When off, files are checked on open and save only. |
| `mylab.runInTerminal` | on | Run in a terminal panel. When off, output goes to the **mylab** channel of the Output panel. Programs that call `input` need the terminal. The debugger always uses a terminal. |

### How the extension finds mylab

It tries these in order and uses the first that works:

1. The `mylab.executable` setting, if you changed it.
2. `mylab` on your PATH.
3. `mylab` in a `.venv` or `venv` folder at the top of the open folder.
4. `python -m mylab` with the interpreter chosen by **Python: Select
   Interpreter** (needs the Python extension), or the
   `python.defaultInterpreterPath` setting.
5. `python3 -m mylab` and `python -m mylab` (on Windows: `py -m mylab` and
   `python -m mylab`).

The **mylab** channel of the Output panel (**View → Output**, then pick
**mylab** in the list) shows which one was chosen.

In a folder that you have not marked as trusted, steps 3 and 4 are skipped and
a `mylab.executable` value stored in the folder is ignored.

## Troubleshooting

### mylab not found

The message "mylab not found — install with: pip install mylab-lang" means
none of the commands above worked.

1. Open a terminal **inside VS Code** (**Terminal → New Terminal**) and run
   `mylab --version`.
2. If that prints a version, restart VS Code. It may have been started before
   mylab was installed.
3. If it says the command is not found, install mylab: `pip install mylab-lang`.
   If you use a virtual environment or conda, activate it first.
4. If mylab is installed but still not found, give the extension the full
   path. Find it with `which mylab` (Mac, Linux) or `where mylab` (Windows),
   then paste the result into the `mylab.executable` setting. Example values:

   | System | Example |
   |---|---|
   | Windows | `C:\Users\you\AppData\Local\Programs\Python\Python312\Scripts\mylab.exe` |
   | Mac, Linux | `/home/you/.venv/bin/mylab` |
   | Any | `python -m mylab` |

### Another extension opens my `.m` files

VS Code uses one language for each file. `.m` is also used by Objective-C, and
by the MathWorks MATLAB extension.

- **For one file:** click the language name at the bottom right of the window
  and choose **MATLAB (mylab)**.
- **For all `.m` files:** open the Command Palette, run **Preferences: Open
  User Settings (JSON)**, and add:

  ```json
  "files.associations": { "*.m": "mylab-matlab" }
  ```

**Using mylab together with the MathWorks extension.** You do not have to
choose. When a file's language is the MathWorks extension's **MATLAB**, the
mylab commands, shortcuts, problem checks, hover help and completion still
work. Colours and indentation then come from the MathWorks extension. You may
see two hovers or two sets of problems, one from each extension.

### No problems are shown

- Check that the language at the bottom right is **MATLAB (mylab)** or **MATLAB**.
- Open **View → Output** and pick **mylab**. If it says that
  `mylab --check` did not answer as expected, your mylab is too old. Update it:
  `pip install --upgrade mylab-lang`.
- A file that uses `eval`, `load` or `run` gets no unknown-function
  warnings, because those can create names while the program
  runs.

### A warning says a function is unknown, but it exists

The check knows mylab's built-in functions and the `.m` files in the same
folder as your file. Functions in other folders (added with `addpath`) are
reported as unknown. The program still runs correctly.

### Plots do not appear

Plots open in separate windows when the program ends. On a computer with no
display (for example over SSH or in a container) there is nowhere to show
them. Save the figure to a file instead: `saveas(gcf, 'plot.png')`.

### `Shift+Enter` does not add a new line any more

In `.m` files, `Shift+Enter` runs the current section. To remove that
shortcut, open **Keyboard Shortcuts**, search for
`mylab.runSelection`, right-click it and choose **Remove Keybinding**.

## For developers

The extension lives in [`editors/vscode`](../editors/vscode). It is plain
JavaScript with no build step and no dependencies.

```bash
cd editors/vscode
node --test                                          # unit tests (Node 20 or newer)
npx --yes @vscode/vsce package --no-dependencies     # build mylab-X.Y.Z.vsix
```

To try your changes, open `editors/vscode` in VS Code and press `F5`.

The hover and completion data is `editors/vscode/data/functions.json`. It is
generated from mylab's help entries. After changing `mylab/help_data.py`, run:

```bash
python scripts/gen_editor_data.py
```

CI fails if the file is out of date.
