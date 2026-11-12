"""Jupyter kernel for mylab.

Install the kernel spec once, then pick "mylab" in Jupyter's kernel list::

    pip install "mylab-lang[jupyter]"
    mylab-kernel install --user

Figures are shown inline, Tab completes names, and Shift+Tab shows help.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import tempfile

from ipykernel.kernelbase import Kernel

from . import __version__
from .api import Session

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*$")


class MylabKernel(Kernel):
    implementation = "mylab"
    implementation_version = __version__
    language = "matlab"
    language_version = __version__
    language_info = {
        "name": "matlab",
        "mimetype": "text/x-matlab",
        "file_extension": ".m",
        "codemirror_mode": "octave",
        "pygments_lexer": "matlab",
    }
    banner = f"mylab {__version__} - free MATLAB-language kernel (NumPy backend)"
    help_links = [{"text": "mylab functions", "url": "https://github.com/ghanemja/mylab/blob/main/docs/FUNCTIONS.md"}]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.environ.setdefault("MPLBACKEND", "Agg")
        self.mylab = Session()
        self.mylab.interp.input_func = self._input

    def _input(self, prompt: str) -> str:
        return self.raw_input(prompt)

    def do_execute(self, code, silent, store_history=True, user_expressions=None, allow_stdin=False):
        result = self.mylab.run(code)
        if not silent:
            if result.output:
                self.send_response(self.iopub_socket, "stream", {"name": "stdout", "text": result.output})
            for png in result.figures:
                self.send_response(self.iopub_socket, "display_data", {
                    "data": {"image/png": base64.b64encode(png).decode("ascii"), "text/plain": "<Figure>"},
                    "metadata": {},
                })
        if result.error:
            where = f" (line {result.error_line})" if result.error_line else ""
            message = f"Error{where}: {result.error}"
            if not silent:
                self.send_response(self.iopub_socket, "stream", {"name": "stderr", "text": message + "\n"})
            return {"status": "error", "execution_count": self.execution_count, "ename": "MatlabError",
                    "evalue": result.error, "traceback": [message]}
        return {"status": "ok", "execution_count": self.execution_count, "payload": [], "user_expressions": {}}

    def _token_at(self, code: str, cursor_pos: int) -> tuple[str, int]:
        m = _IDENT.search(code[:cursor_pos])
        if not m:
            return "", cursor_pos
        return m.group(0), m.start()

    def do_complete(self, code, cursor_pos):
        token, start = self._token_at(code, cursor_pos)
        matches = self.mylab.complete(token) if token and "." not in token else []
        return {"status": "ok", "matches": matches, "cursor_start": start, "cursor_end": cursor_pos,
                "metadata": {}}

    def do_inspect(self, code, cursor_pos, detail_level=0, omit_sections=()):
        token, _ = self._token_at(code, cursor_pos)
        # Also accept the cursor sitting just inside the name.
        tail = re.match(r"[A-Za-z0-9_]*", code[cursor_pos:])
        if tail:
            token += tail.group(0)
        if not token:
            return {"status": "ok", "found": False, "data": {}, "metadata": {}}
        text = self.mylab.help(token)
        found = "not found" not in text
        return {"status": "ok", "found": found, "data": {"text/plain": text} if found else {}, "metadata": {}}

    def do_is_complete(self, code):
        from .cli import needs_more_input
        if needs_more_input(code if code.endswith("\n") else code + "\n"):
            return {"status": "incomplete", "indent": "    "}
        return {"status": "complete"}


# -- kernel spec installation ----------------------------------------------------------------

KERNEL_SPEC = {
    "argv": [sys.executable, "-m", "mylab.kernel", "-f", "{connection_file}"],
    "display_name": "mylab (MATLAB language)",
    "language": "matlab",
    "metadata": {"debugger": False},
}


def install(user: bool = True, prefix: str | None = None) -> str:
    from jupyter_client.kernelspec import KernelSpecManager
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "kernel.json"), "w") as fh:
            json.dump(KERNEL_SPEC, fh, indent=2)
        dest = KernelSpecManager().install_kernel_spec(td, "mylab", user=user and prefix is None, prefix=prefix)
    return dest


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "install":
        p = argparse.ArgumentParser(prog="mylab-kernel install", description="Install the mylab Jupyter kernel.")
        p.add_argument("--user", action="store_true", help="install for the current user (default)")
        p.add_argument("--sys-prefix", action="store_true", help="install into the current Python environment")
        p.add_argument("--prefix", help="install under this prefix")
        a = p.parse_args(argv[1:])
        prefix = a.prefix or (sys.prefix if a.sys_prefix else None)
        dest = install(user=prefix is None, prefix=prefix)
        print(f"Installed the mylab kernel spec in {dest}")
        return
    from ipykernel.kernelapp import IPKernelApp
    IPKernelApp.launch_instance(argv=argv, kernel_class=MylabKernel)


if __name__ == "__main__":
    main()
