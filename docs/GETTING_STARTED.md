# Getting started

This guide assumes you have never installed a programming tool before.

## Option A: the browser (easiest)

Open the mylab web app. Wait for "Ready" in the top bar; the first visit
downloads Python into your browser, so it takes a little while. Then:

1. Click **welcome.m** in the Files panel and press **Run** (or Ctrl+Enter).
2. Type commands in the **Command Window** at the bottom, for example
   `x = 1:5` and press Enter.
3. Click a variable in the **Workspace** panel to see its value.
4. Plots appear in the **Plots** panel, where you can download them.

Your files are saved in your browser. Use the Files panel to download them
or to import `.m` files you already have.

## Option B: install on your computer

1. Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/).
   On Windows, tick **"Add Python to PATH"** in the installer.
2. Open a terminal (on Windows, "Command Prompt"; on macOS, "Terminal").
3. Run:

   ```bash
   pip install "git+https://github.com/ghanemja/mylab"
   ```

   If `pip` is not found, try `python -m pip install ...` (or `py -m pip ...`
   on Windows).

4. Start mylab:

   ```bash
   mylab
   ```

   You'll see a `>>` prompt. Type `exit` to quit.

5. To run a script file, go to its folder and run it:

   ```bash
   cd path/to/my/files
   mylab myscript.m
   ```

   Functions saved in other `.m` files in the same folder are found
   automatically. Plot windows open when the script finishes.

## Your first few commands

```matlab
a = 3;                 % a semicolon hides the output
b = [1 2 3]            % a row vector
M = [1 2; 3 4]         % a 2-by-2 matrix
M'                     % transpose
M * M                  % matrix product
M .* M                 % element-by-element product
v = linspace(0, 1, 5)  % 5 points from 0 to 1
v(2)                   % indexing starts at 1
v(end)                 % the last element
sum(v), mean(v), max(v)
help linspace          % help for any function
```

## Getting help

- `help name` shows the calling forms and a short description of a function.
- [FUNCTIONS.md](FUNCTIONS.md) lists every function by topic.
- In the browser app, the **Assistant** explains errors and fixes code. It can
  use a free local model (Ollama) or your own API key.
- [COMPATIBILITY.md](COMPATIBILITY.md) lists what isn't supported yet.
