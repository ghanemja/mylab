# mylab — run MATLAB-language code for free

Run, check and debug `.m` files in VS Code with
[mylab](https://github.com/ghanemja/mylab), a free, open-source interpreter
for MATLAB-language code. No licence, no account, no sign-in.

mylab is an independent project. It is not made or endorsed by MathWorks.

## Get started

1. **Install mylab** (needs Python 3.10 or newer):

   ```
   pip install mylab-lang
   ```

2. **Install this extension.**
3. **Open a `.m` file** and press the ▶ button at the top right of the editor,
   or press `Ctrl+Alt+Enter` (`Cmd+Alt+Enter` on a Mac).

If VS Code says "mylab not found", see
[the troubleshooting guide](https://github.com/ghanemja/mylab/blob/main/docs/VSCODE.md#mylab-not-found).

## What you get

| Feature | How to use it |
| --- | --- |
| **Run a file** | ▶ button, or `Ctrl+Alt+Enter` / `Cmd+Alt+Enter` |
| **Run a section or a selection** | `Shift+Enter`. Sections are the parts of a file between lines that start with `%%`. With text selected, only the selection runs. |
| **Debug a file** | `Ctrl+Alt+Shift+Enter` / `Cmd+Alt+Shift+Enter`. Opens mylab's step-by-step debugger in a terminal; type `h` there for help. |
| **Interactive prompt** | Command Palette → **mylab: Open Interactive Prompt** |
| **Problems as you type** | Syntax errors and misspelled function names appear underlined and in the Problems panel. Nothing is run to find them. |
| **Help on hover** | Point at a function name to see how to call it and what it does. Works offline for the 770+ built-in functions, and for your own functions in the same folder. |
| **Help page** | `Shift+F1` on a function name |
| **Completion** | Function names with a one-line summary, keywords, and snippets for `for`, `if`, `while`, `switch`, `function` and `try` |
| **Call hints** | Type `(` after a function name to see its arguments |
| **Editing** | Syntax colouring, automatic indentation, folding of blocks and `%%` sections, an outline of functions and sections |

## Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `mylab.executable` | `mylab` | The command that starts mylab. A name, a full path, or a command line such as `python -m mylab`. |
| `mylab.checkOnType` | on | Check for problems while you type. Turn it off to check only when a file is opened or saved. |
| `mylab.runInTerminal` | on | Run in a terminal panel (programs can ask for input). Turn it off to send output to the Output panel instead. |

When `mylab.executable` is left at its default and `mylab` is not on your
PATH, the extension also looks for mylab in the workspace's `.venv` folder and
tries `python -m mylab` with the Python interpreter you selected in VS Code.

## If another extension also opens `.m` files

VS Code can only use one language for `.m` files at a time. If your file opens
as Objective-C or as "MATLAB" (the MathWorks extension):

- **This file only:** click the language name at the bottom right of the
  window and choose **MATLAB (mylab)**.
- **Always:** add this to your settings (`Ctrl+Shift+P` → **Preferences: Open
  User Settings (JSON)**):

  ```json
  "files.associations": { "*.m": "mylab-matlab" }
  ```

You can also keep the MathWorks extension's "MATLAB" language. The mylab
commands, problems, hover help and completion work there too.

## More

- [Full guide and troubleshooting](https://github.com/ghanemja/mylab/blob/main/docs/VSCODE.md)
- [What mylab supports](https://github.com/ghanemja/mylab/blob/main/docs/COMPATIBILITY.md)
- [Report a problem](https://github.com/ghanemja/mylab/issues)

MIT licence. MATLAB is a registered trademark of The MathWorks, Inc.
