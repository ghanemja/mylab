"""Help text for the debugger commands."""

HELP = {
    "keyboard": ("keyboard",
                 "Pauses the program at this point and opens the K>> debugger prompt, where you can inspect "
                 "variables, step (n, s) or continue (c). Only works in an interactive terminal; elsewhere it "
                 "does nothing."),
    "dbstop": (["dbstop in FILE at LINE", "dbstop in FILE", "dbstop if error"],
               "Sets a breakpoint: the program pauses with a K>> prompt when it reaches that line. Needs an "
               "interactive terminal. In the web app, use Step through instead."),
    "dbclear": ("dbclear all", "Removes all breakpoints."),
    "dbstatus": (["dbstatus", "s = dbstatus"], "Lists the breakpoints, or returns them as a struct array."),
    "dbstack": (["dbstack", "s = dbstack"],
                "Shows the functions currently being run, innermost first, or returns them as a struct array."),
    "dbcont": ("dbcont", "At the K>> prompt: continue running until the next breakpoint."),
    "dbstep": (["dbstep", "dbstep in", "dbstep out"],
               "At the K>> prompt: run the next line (stepping over calls), step into a function, or run "
               "until the current function returns."),
    "dbquit": ("dbquit", "At the K>> prompt: stop the program."),
    "dbup": ("dbup", "Not supported: the K>> prompt always works in the paused function's workspace."),
    "dbdown": ("dbdown", "Not supported: the K>> prompt always works in the paused function's workspace."),
}
