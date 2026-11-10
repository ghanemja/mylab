# Jupyter kernel

You can use mylab in Jupyter notebooks, JupyterLab and VS Code notebooks.

```bash
pip install "mylab[jupyter]"
mylab-kernel install --user        # or --sys-prefix to install into the current environment
```

Then pick **mylab (MATLAB language)** when you create a notebook or change its
kernel.

- Each cell runs in one persistent workspace. Statements without a semicolon
  print their value.
- Figures appear inline, below the cell.
- **Tab** completes variable and function names.
- **Shift+Tab** (or `help name`) shows help for a function.
- `input()` asks for a value in the notebook.

To remove the kernel, run:

```bash
jupyter kernelspec uninstall mylab
```
