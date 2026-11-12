'use strict';
// Find a way to start mylab. No dependency on VS Code.

const path = require('path');

const DEFAULT_EXECUTABLE = 'mylab';
const INSTALL_HINT = 'mylab not found — install with: pip install mylab-lang';
const DOCS_URL = 'https://github.com/ghanemja/mylab/blob/main/docs/VSCODE.md#mylab-not-found';

/**
 * Split a command line into words. Double and single quotes group words;
 * backslashes are kept as they are, so Windows paths survive.
 *   parseCommandLine('"C:\\Program Files\\py\\python.exe" -m mylab')
 *     -> ['C:\\Program Files\\py\\python.exe', '-m', 'mylab']
 */
function parseCommandLine(text) {
  const words = [];
  let cur = '';
  let quote = null;
  let has = false;
  for (const c of String(text || '').trim()) {
    if (quote) {
      if (c === quote) quote = null;
      else cur += c;
    } else if (c === '"' || c === "'") {
      quote = c;
      has = true;
    } else if (c === ' ' || c === '\t') {
      if (cur || has) words.push(cur);
      cur = '';
      has = false;
    } else {
      cur += c;
    }
  }
  if (cur || has) words.push(cur);
  return words;
}

/**
 * The commands to try, best first. Each is {command, args, label}; the
 * mylab arguments are appended to `args` when it runs.
 *
 * @param {object} o
 * @param {string} [o.configured]       the mylab.executable setting
 * @param {string} [o.pythonPath]       the Python interpreter selected in VS Code, if known
 * @param {string[]} [o.workspaceFolders] folders that may hold a .venv
 * @param {string} [o.platform]         process.platform
 * @param {(p: string) => boolean} [o.exists] file test (for the setting and virtual environments)
 */
function candidateCommands(o) {
  const platform = o.platform || process.platform;
  const win = platform === 'win32';
  const exists = o.exists || (() => false);
  const out = [];
  const add = (command, args, label) => {
    if (!command) return;
    const key = JSON.stringify([command, args]);
    if (!out.some((c) => c.key === key)) out.push({ command, args, label, key });
  };

  const configured = String(o.configured || '').trim();
  if (configured && configured !== DEFAULT_EXECUTABLE) {
    if (exists(configured)) {
      add(configured, [], 'the mylab.executable setting'); // a path, possibly with spaces
    } else {
      const words = parseCommandLine(configured);
      add(words[0], words.slice(1), 'the mylab.executable setting');
    }
  }
  add(DEFAULT_EXECUTABLE, [], 'mylab on the PATH');

  const p = win ? path.win32 : path.posix;
  for (const folder of o.workspaceFolders || []) {
    for (const venv of ['.venv', 'venv']) {
      const exe = win ? p.join(folder, venv, 'Scripts', 'mylab.exe') : p.join(folder, venv, 'bin', 'mylab');
      if (exists(exe)) add(exe, [], `the ${venv} folder of the workspace`);
    }
  }
  if (o.pythonPath) add(o.pythonPath, ['-m', 'mylab'], 'the selected Python interpreter');
  if (win) {
    add('py', ['-m', 'mylab'], 'py -m mylab');
    add('python', ['-m', 'mylab'], 'python -m mylab');
  } else {
    add('python3', ['-m', 'mylab'], 'python3 -m mylab');
    add('python', ['-m', 'mylab'], 'python -m mylab');
  }
  return out.map(({ command, args, label }) => ({ command, args, label }));
}

/**
 * Try the candidates in order. `probe(candidate)` resolves to the version
 * text (for example "mylab 1.0.0") when the command works, or to a falsy
 * value / a rejection when it does not.
 * @returns {Promise<{command, args, label, version}|null>}
 */
async function resolveExecutable(candidates, probe) {
  for (const c of candidates) {
    let version = null;
    try {
      version = await probe(c);
    } catch (e) {
      version = null;
    }
    if (version) return { ...c, version: String(version).trim() };
  }
  return null;
}

/** True when the output of `<command> --version` comes from mylab. */
function looksLikeMylab(output) {
  return /^mylab \d/m.test(String(output || ''));
}

/** A readable command line for messages (not for a shell). */
function describeCommand(exe, extraArgs) {
  return [exe.command, ...exe.args, ...(extraArgs || [])]
    .map((w) => (/\s/.test(w) ? `"${w}"` : w))
    .join(' ');
}

module.exports = {
  DEFAULT_EXECUTABLE,
  INSTALL_HINT,
  DOCS_URL,
  parseCommandLine,
  candidateCommands,
  resolveExecutable,
  looksLikeMylab,
  describeCommand,
};
