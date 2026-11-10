# mylab in the browser

The mylab web app runs MATLAB-language code entirely in your web browser.
There is nothing to install and no account to create. It is built for students
who don't have a MATLAB license and for anyone on a Chromebook or a low-end
laptop.

![The mylab web app](images/web-app.png)

## How it works

- The mylab interpreter is Python. The web app runs it with
  [Pyodide](https://pyodide.org), a version of Python compiled to
  WebAssembly, together with NumPy, SciPy and Matplotlib.
- The engine runs in a **Web Worker**, so the page stays responsive while your
  code runs.
- Nothing runs on a server. The site is plain static files, hosted for free on
  GitHub Pages. Your code and files stay in your browser.

## Using the app

| Panel | What it does |
|-------|--------------|
| **Files** (left) | Your `.m` files, saved in the browser (IndexedDB). Create, rename, delete, import (button or drag-and-drop) and download single files or everything as a `.zip`. |
| **Editor** (center) | Tabs, syntax highlighting, bracket matching, comment toggling and name completion. Files save automatically. |
| **Command Window** (bottom) | Type a command and press Enter. Arrow keys walk through your history (typing a few letters first filters it). Tab completes names. `clc` clears the window. |
| **Workspace** (right) | Every variable with its size and class. Click one to see its full value. |
| **Plots** (right) | Figures drawn by your code. Click to enlarge, or download as PNG. |
| **Assistant** (optional) | An AI helper, see below. |

Running code:

- **Run** (Ctrl+Enter) runs the file in the editor.
- **Run selection** (Ctrl+Shift+Enter) runs the selected lines, or the line
  with the cursor.
- Before each run, all your files are copied into the engine's working folder,
  so a script can call functions saved in other `.m` files (a file
  `add_squares.m` that starts with `function s = add_squares(a, b)` can be
  called as `add_squares(3, 4)`).
- Errors are shown in red with their line number, and the line is highlighted
  in the editor.
- **Stop** stops runaway code (for example an endless loop). The browser has
  no safe way to interrupt Python inside a worker on GitHub Pages, so Stop
  restarts the engine: **all variables are cleared**. Your files are not
  affected.
- **Reset** clears all variables without restarting.

Other things: light and dark themes (following your system setting until you
pick one), a keyboard shortcut list (keyboard icon), and the About dialog.

The first visit downloads about 40 MB (Python, NumPy, SciPy, Matplotlib). The
browser caches these files, so later visits start much faster. On a slow
connection the first start can take a minute; you can read and edit your files
while it loads.

## The assistant

The assistant panel (Assistant button, top right) can explain an error, fix
your code, explain code, translate it to Python (NumPy/Matplotlib) or answer
questions. Code in its answers gets buttons to copy it, insert it into the
editor, replace the current file, or run it.

With each question, the app sends your current file (or selection), your last
error and a list of variable names, sizes and classes (not their values).

The assistant is optional. Choose a provider in its settings (gear icon):

### Ollama (free, runs on your computer)

1. Install Ollama from [ollama.com](https://ollama.com).
2. Download a model, for example `ollama pull qwen2.5-coder` (small machines:
   try `qwen2.5-coder:1.5b` or `llama3.2`).
3. Ollama only accepts requests from web pages it trusts. Start it with
   `OLLAMA_ORIGINS` set to the site's address (the settings dialog shows the
   exact command for the page you are on):

   ```bash
   # macOS / Linux
   OLLAMA_ORIGINS="https://<user>.github.io" ollama serve
   ```

   ```powershell
   # Windows PowerShell
   $env:OLLAMA_ORIGINS="https://<user>.github.io"; ollama serve
   ```

   If the Ollama desktop app is already running, quit it first (or set
   `OLLAMA_ORIGINS` as a system environment variable and restart it).
4. In the settings, choose **Ollama**, keep the URL `http://localhost:11434`,
   click **Find models** and pick one.

Browsers allow an `https://` page to talk to `http://localhost`. Chrome may ask
once for permission to access devices on your local network; allow it.

### Anthropic Claude (your own API key)

Choose **Anthropic Claude**, paste an API key from
[console.anthropic.com](https://console.anthropic.com) and keep the default
model (`claude-opus-5-5`) or type another model id. The app uses the official
Anthropic SDK, loaded from jsDelivr, and calls the API directly from your
browser.

### OpenAI-compatible (your own API key)

Choose **OpenAI-compatible** and enter a base URL, model and key. This works
with any service that offers `POST {base}/chat/completions` with streaming and
allows browser requests (CORS), for example OpenAI
(`https://api.openai.com/v1`), OpenRouter (`https://openrouter.ai/api/v1`),
Groq (`https://api.groq.com/openai/v1`) or LM Studio on your computer
(`http://localhost:1234/v1`, enable CORS in its server settings; no key
needed).

## Privacy

- Your files live in your browser's IndexedDB storage. They are not uploaded
  anywhere. If the browser blocks storage (some private modes do), the app
  keeps files in memory only and says so in the Files panel; download them
  before closing the tab.
- Clearing your browser's site data deletes your files. Use **Download all**
  (.zip) to keep a backup.
- API keys are saved in this browser's `localStorage` and sent only to the
  provider you chose, never to mylab. **Anyone who can use this browser
  profile can read a saved key**, so don't save keys on shared or public
  computers. Use **Forget saved keys** in the settings to remove them.
- With an online provider, the context described above is sent to that
  provider and handled under its terms. With Ollama, everything stays on your
  computer.

## Limitations

- `input()` is not available in the browser yet. It raises an error; set
  values in variables at the top of your script instead.
- First load is large (about 40 MB) and needs a reasonably modern browser
  with WebAssembly (Chrome, Edge, Firefox, Safari from the last few years).
- Stop restarts the engine and clears the workspace (see above).
- Code runs on a single thread, and Python in WebAssembly is slower than
  native Python. Very large computations are better run with the desktop
  `mylab` command.
- Plots are static images. There is no zooming or rotating of 3-D plots.
- `exit`/`quit`, system commands and files outside the app's working folder are
  not available. Files created by your code (for example with `saveas`) live
  only until the engine restarts.

## Local development

You need Python 3.10+ and `pip`.

```bash
python scripts/build_web.py --serve          # build the wheel, serve web/ on http://127.0.0.1:8000/
python scripts/build_web.py --serve --port 8123
```

`build_web.py` builds a pure-Python wheel of mylab, copies it to
`web/wheels/` and writes `web/wheels/manifest.json`. The worker reads the
manifest and installs the wheel with `micropip`. `web/wheels/` is not
committed; CI builds it.

The app has no build step: `web/index.html`, `web/styles.css`, ES modules in
`web/js/` and the worker `web/worker.js`. Third-party code is loaded from CDNs
with pinned versions: Pyodide 0.27.7 (jsDelivr), CodeMirror 5.65.16 and JSZip
3.10.1 (cdnjs), and the Anthropic SDK 1.3.0 (jsDelivr, only when that
provider is used).

### Tests

`scripts/test_web.mjs` builds and serves the app, then drives it in headless
Chromium with Playwright (needs network access to the CDNs):

```bash
npm install playwright   # once, if Playwright isn't available
node scripts/test_web.mjs
```

It checks the command window, workspace, plotting, error line numbers, calling
a function from another file, run selection, the assistant panel and settings
(AI providers are mocked; no real API is called), Stop, persistence, the phone
layout, and that there are no console errors. It also saves the screenshot at
`docs/images/web-app.png`.

## Deployment

`.github/workflows/pages.yml` runs on every push to `main`: it builds the
wheel with `scripts/build_web.py` and publishes `web/` to GitHub Pages. To
enable it, open the repository's **Settings → Pages** and set **Source** to
**GitHub Actions**.
