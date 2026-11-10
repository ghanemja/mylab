# Using mylab from AI assistants (MCP)

mylab includes an [MCP](https://modelcontextprotocol.io) server, so assistants
such as Claude Desktop, Cursor or any MCP-capable client can run
MATLAB-language code for you, look at your variables and get plots back as
images.

## Install

```bash
pip install "mylab[mcp]"
```

This installs the `mylab-mcp` command, which talks to the client over stdio.

## Configure your client

Add a server entry to your client's MCP configuration. For Claude Desktop
that is `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mylab": {
      "command": "mylab-mcp",
      "env": { "MYLAB_WORKDIR": "/path/to/your/m-files" }
    }
  }
}
```

If `mylab-mcp` isn't on the client's `PATH`, put the full path in
`command`. To find it, run `which mylab-mcp` (or `where mylab-mcp` on
Windows).

`MYLAB_WORKDIR` is optional. It sets the folder where the server looks for
your scripts and function files.

## Tools

| Tool | What it does |
|---|---|
| `run_code` | Runs code in a persistent workspace and returns the output, any error (with line number) and figures as PNG images. |
| `run_file` | Runs a `.m` file from disk. |
| `list_variables` | Lists the workspace variables with their size, class and a short preview. |
| `get_variable` | Shows one variable in full, as the command window prints it. |
| `function_help` | Shows help for a function: its calling forms and behaviour in mylab. |
| `search_functions` | Searches function names and help text, e.g. "regression" or "filter". |
| `reset_session` | Clears the workspace. |

## Privacy and safety

The server runs locally with your user's permissions. Code it runs can read and
write files like any program you run yourself. Your MCP client normally asks
before each tool call, so review the code before you approve it.
