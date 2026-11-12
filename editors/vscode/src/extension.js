'use strict';
// The VS Code layer. This is the only file that loads the `vscode` module;
// the logic lives in the other files of this folder, which plain Node can test.

const vscode = require('vscode');
const cp = require('child_process');
const fs = require('fs');
const path = require('path');

const { codeToRun } = require('./sections');
const { parseCheckOutput, toDiagnostics } = require('./diagnostics');
const exeLib = require('./executable');
const helpLib = require('./help');
const { foldingRanges, outline } = require('./folding');
const { codeOnly, wordAt, splitLines } = require('./text');
const { WorkCopies } = require('./workcopy');

const LANGUAGES = ['mylab-matlab', 'matlab'];
const SELECTOR = LANGUAGES.map((language) => ({ language }));
const HELP_SCHEME = 'mylab-help';

let output; // output channel
let functions = {}; // bundled reference: name -> {usage, description}
let completionCache = null;
let workCopies = null;
let diagnostics = null;
let exePromise = null; // cached result of the search for mylab
let warnedNotFound = false;
let runningChild = null;
const checkState = new Map(); // document uri -> {timer, seq}
const helpTexts = new Map(); // help document uri -> text
const helpCache = new Map(); // "folder\nname" -> {time, text}

function activate(context) {
  output = vscode.window.createOutputChannel('mylab');
  workCopies = new WorkCopies();
  diagnostics = vscode.languages.createDiagnosticCollection('mylab');
  try {
    functions = JSON.parse(fs.readFileSync(path.join(context.extensionPath, 'data', 'functions.json'), 'utf8'));
  } catch (e) {
    output.appendLine(`Could not read the bundled function reference: ${e.message}`);
  }

  const helpEmitter = new vscode.EventEmitter();
  const helpProvider = {
    onDidChange: helpEmitter.event,
    provideTextDocumentContent: (uri) => helpTexts.get(uri.toString()) || '',
  };

  context.subscriptions.push(
    output,
    diagnostics,
    { dispose: () => workCopies.dispose() },
    vscode.commands.registerCommand('mylab.runFile', () => guard(runFile(false))),
    vscode.commands.registerCommand('mylab.debugFile', () => guard(runFile(true))),
    vscode.commands.registerCommand('mylab.runSelection', () => guard(runSelection())),
    vscode.commands.registerCommand('mylab.openPrompt', () => guard(openPrompt())),
    vscode.commands.registerCommand('mylab.showHelp', () => guard(showHelp(helpEmitter))),
    vscode.commands.registerCommand('mylab.stop', stopRunning),
    vscode.commands.registerCommand('mylab.checkFile', () => {
      const doc = activeDocument();
      if (doc) scheduleCheck(doc, 0);
    }),
    vscode.workspace.registerTextDocumentContentProvider(HELP_SCHEME, helpProvider),
    vscode.languages.registerHoverProvider(SELECTOR, { provideHover }),
    vscode.languages.registerCompletionItemProvider(SELECTOR, { provideCompletionItems }),
    vscode.languages.registerSignatureHelpProvider(SELECTOR, { provideSignatureHelp }, '(', ','),
    vscode.languages.registerFoldingRangeProvider(SELECTOR, { provideFoldingRanges }),
    vscode.languages.registerDocumentSymbolProvider(SELECTOR, { provideDocumentSymbols }),
    vscode.workspace.onDidOpenTextDocument((doc) => scheduleCheck(doc, 0)),
    vscode.workspace.onDidSaveTextDocument((doc) => scheduleCheck(doc, 0)),
    vscode.workspace.onDidChangeTextDocument((e) => {
      if (config().get('checkOnType', true) && e.contentChanges.length) scheduleCheck(e.document, 500);
    }),
    vscode.workspace.onDidCloseTextDocument((doc) => {
      const key = doc.uri.toString();
      const state = checkState.get(key);
      if (state && state.timer) clearTimeout(state.timer);
      checkState.delete(key);
      diagnostics.delete(doc.uri);
    }),
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration('mylab') || e.affectsConfiguration('python.defaultInterpreterPath')) {
        exePromise = null;
        warnedNotFound = false;
        vscode.workspace.textDocuments.forEach((doc) => scheduleCheck(doc, 0));
      }
    })
  );
  if (vscode.workspace.onDidGrantWorkspaceTrust) {
    context.subscriptions.push(vscode.workspace.onDidGrantWorkspaceTrust(() => {
      exePromise = null; // the folder's own .venv may be used now
    }));
  }
  vscode.workspace.textDocuments.forEach((doc) => scheduleCheck(doc, 0));
}

function deactivate() {
  stopRunning();
  if (workCopies) workCopies.dispose();
}

// -- helpers -------------------------------------------------------------------------------

function config() {
  return vscode.workspace.getConfiguration('mylab');
}

function isOurs(doc) {
  return !!doc && LANGUAGES.includes(doc.languageId);
}

function activeDocument() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showInformationMessage('Open a .m file first.');
    return null;
  }
  return editor.document;
}

function guard(promise) {
  return Promise.resolve(promise).catch((e) => {
    output.appendLine(String((e && e.stack) || e));
    vscode.window.showErrorMessage(`mylab: ${(e && e.message) || e}`);
  });
}

function folderOf(doc) {
  if (doc && doc.uri.scheme === 'file') return path.dirname(doc.uri.fsPath);
  const folders = vscode.workspace.workspaceFolders;
  return folders && folders.length ? folders[0].uri.fsPath : undefined;
}

// -- finding mylab -------------------------------------------------------------------------

async function selectedPython() {
  // The interpreter chosen with "Python: Select Interpreter", when the Python extension is in use.
  try {
    const ext = vscode.extensions.getExtension('ms-python.python');
    if (ext && ext.isActive && ext.exports && ext.exports.environments) {
      const envs = ext.exports.environments;
      const active = envs.getActiveEnvironmentPath();
      const resolved = active && (await envs.resolveEnvironment(active));
      const exe = resolved && resolved.executable && resolved.executable.uri;
      if (exe && exe.fsPath) return exe.fsPath;
    }
  } catch (e) {
    // fall through to the setting
  }
  let p = vscode.workspace.getConfiguration('python').get('defaultInterpreterPath');
  if (!p || p === 'python') return undefined;
  const folders = vscode.workspace.workspaceFolders;
  if (folders && folders.length) p = p.replace(/\$\{workspaceFolder\}/g, folders[0].uri.fsPath);
  if (p.startsWith('~')) p = path.join(require('os').homedir(), p.slice(1));
  return p.includes('${') ? undefined : p;
}

function probe(candidate) {
  return new Promise((resolve) => {
    cp.execFile(candidate.command, [...candidate.args, '--version'], { timeout: 20000, windowsHide: true },
      (err, stdout, stderr) => {
        const text = `${stdout || ''}\n${stderr || ''}`;
        const m = /^mylab \d.*$/m.exec(text);
        resolve(!err && m ? m[0] : null);
      });
  });
}

/** The command that starts mylab, or null. `quiet` suppresses the "not found" message. */
async function findMylab(quiet) {
  if (!exePromise) {
    exePromise = (async () => {
      const trusted = vscode.workspace.isTrusted !== false;
      const candidates = exeLib.candidateCommands({
        configured: config().get('executable', exeLib.DEFAULT_EXECUTABLE),
        // In a folder the user has not trusted, never start a program that the folder supplies.
        pythonPath: trusted ? await selectedPython() : undefined,
        workspaceFolders: trusted ? (vscode.workspace.workspaceFolders || []).map((f) => f.uri.fsPath) : [],
        platform: process.platform,
        exists: (p) => {
          try {
            return fs.statSync(p).isFile();
          } catch (e) {
            return false;
          }
        },
      });
      const found = await exeLib.resolveExecutable(candidates, probe);
      if (found) output.appendLine(`Using ${exeLib.describeCommand(found)} (${found.version}, from ${found.label}).`);
      else output.appendLine(`mylab was not found. Tried: ${candidates.map((c) => exeLib.describeCommand(c)).join('; ')}`);
      return found;
    })();
  }
  const exe = await exePromise;
  if (!exe) {
    exePromise = null; // look again next time: the user may have installed it meanwhile
    if (!quiet || !warnedNotFound) {
      warnedNotFound = true;
      showNotFound();
    }
  }
  return exe;
}

function showNotFound() {
  const guide = 'Open Install Guide';
  const settings = 'Open Settings';
  vscode.window.showErrorMessage(exeLib.INSTALL_HINT, guide, settings).then((choice) => {
    if (choice === guide) vscode.env.openExternal(vscode.Uri.parse(exeLib.DOCS_URL));
    if (choice === settings) vscode.commands.executeCommand('workbench.action.openSettings', 'mylab.executable');
  });
}

function execMylab(exe, args, cwd, timeout) {
  return new Promise((resolve) => {
    cp.execFile(exe.command, [...exe.args, ...args],
      { cwd, timeout: timeout || 30000, maxBuffer: 8 * 1024 * 1024, windowsHide: true },
      (err, stdout, stderr) => {
        const code = err ? (typeof err.code === 'number' ? err.code : -1) : 0;
        resolve({ code, stdout: String(stdout || ''), stderr: String(stderr || '') });
      });
  });
}

// -- running -------------------------------------------------------------------------------

async function runFile(debug) {
  const doc = activeDocument();
  if (!doc) return;
  let file;
  if (doc.isUntitled || doc.uri.scheme !== 'file') {
    file = workCopies.tempScript(doc.getText(), 'untitled.m');
  } else {
    if (doc.isDirty && !(await doc.save())) return;
    file = doc.uri.fsPath;
  }
  const exe = await findMylab(false);
  if (!exe) return;
  const name = path.basename(file);
  await launch(exe, debug ? ['--debug', file] : [file], folderOf(doc), debug ? `Debug ${name}` : `Run ${name}`, debug);
}

async function runSelection() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showInformationMessage('Open a .m file first.');
    return;
  }
  const doc = editor.document;
  const sel = editor.selection;
  const picked = codeToRun(
    doc.getText(),
    sel.isEmpty ? null : {
      startLine: sel.start.line,
      startColumn: sel.start.character,
      endLine: sel.end.line,
      endColumn: sel.end.character,
    },
    sel.active.line
  );
  if (!picked.code.trim()) {
    vscode.window.showInformationMessage('Nothing to run here.');
    return;
  }
  const exe = await findMylab(false);
  if (!exe) return;
  // The temporary file keeps the file's name, and blank lines keep the line
  // numbers, so error messages point at the right place in the real file.
  const base = doc.uri.scheme === 'file' ? path.basename(doc.uri.fsPath) : 'untitled.m';
  const file = workCopies.tempScript(picked.code, base);
  const what = picked.kind === 'selection' ? 'selection' : 'section';
  const label = `Run ${what} of ${base} (lines ${picked.startLine + 1}-${picked.endLine + 1})`;
  await launch(exe, [file], folderOf(doc), label, false);
}

async function launch(exe, args, cwd, label, forceTerminal) {
  if (forceTerminal || config().get('runInTerminal', true)) {
    const scope = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders.length
      ? vscode.TaskScope.Workspace
      : vscode.TaskScope.Global;
    const execution = new vscode.ProcessExecution(exe.command, [...exe.args, ...args], cwd ? { cwd } : undefined);
    const task = new vscode.Task({ type: 'mylab' }, scope, label, 'mylab', execution, []);
    task.presentationOptions = {
      reveal: vscode.TaskRevealKind.Always,
      panel: vscode.TaskPanelKind.Dedicated,
      focus: !!forceTerminal, // the debugger needs the keyboard
      clear: true,
      showReuseMessage: false,
    };
    await vscode.tasks.executeTask(task);
    return;
  }
  stopRunning();
  output.clear();
  output.show(true);
  output.appendLine(`> ${label}`);
  const child = cp.spawn(exe.command, [...exe.args, ...args], { cwd, windowsHide: true });
  runningChild = child;
  child.stdout.on('data', (d) => output.append(d.toString()));
  child.stderr.on('data', (d) => output.append(d.toString()));
  child.on('error', (e) => output.appendLine(`Could not start mylab: ${e.message}`));
  child.on('close', (code) => {
    if (runningChild === child) runningChild = null;
    output.appendLine(code ? `\n[finished with an error, exit code ${code}]` : '\n[finished]');
  });
  if (child.stdin) child.stdin.end(); // no keyboard here: input() gets end-of-file instead of waiting forever
}

function stopRunning() {
  if (runningChild) {
    runningChild.kill();
    runningChild = null;
  }
}

async function openPrompt() {
  const exe = await findMylab(false);
  if (!exe) return;
  const editor = vscode.window.activeTextEditor;
  const terminal = vscode.window.createTerminal({
    name: 'mylab',
    shellPath: exe.command,
    shellArgs: exe.args,
    cwd: folderOf(editor && editor.document),
  });
  terminal.show();
}

// -- diagnostics ---------------------------------------------------------------------------

function scheduleCheck(doc, delay) {
  if (!isOurs(doc) || doc.uri.scheme === HELP_SCHEME) return;
  const key = doc.uri.toString();
  const state = checkState.get(key) || { timer: null, seq: 0 };
  checkState.set(key, state);
  if (state.timer) clearTimeout(state.timer);
  state.timer = setTimeout(() => {
    state.timer = null;
    runCheck(doc, state).catch((e) => output.appendLine(`Check failed: ${e.message}`));
  }, delay);
}

async function runCheck(doc, state) {
  const seq = ++state.seq;
  const exe = await findMylab(true);
  if (!exe || doc.isClosed) return;
  const text = doc.getText();
  const onDisk = doc.uri.scheme === 'file' && !doc.isDirty;
  const target = onDisk ? doc.uri.fsPath : workCopies.write(doc.uri.scheme === 'file' ? doc.uri.fsPath : null, text);
  const result = await execMylab(exe, ['--check', target], path.dirname(target), 30000);
  if (seq !== state.seq || doc.isClosed) return; // a newer check has started
  const parsed = parseCheckOutput(result.stdout);
  if (!parsed) {
    if (!state.complained) {
      state.complained = true;
      output.appendLine(`"mylab --check" did not answer as expected (is mylab up to date?). ${result.stderr.trim()}`);
    }
    return;
  }
  diagnostics.set(doc.uri, toDiagnostics(parsed.problems, text).map((d) => {
    const diag = new vscode.Diagnostic(
      new vscode.Range(d.startLine, d.startColumn, d.endLine, d.endColumn),
      d.message,
      d.severity === 'error' ? vscode.DiagnosticSeverity.Error : vscode.DiagnosticSeverity.Warning
    );
    diag.source = 'mylab';
    return diag;
  }));
}

// -- hover, completion, signatures ---------------------------------------------------------

function nameAt(doc, position) {
  const line = doc.lineAt(position.line).text;
  const code = codeOnly(line);
  if (code[position.character] === ' ' && line[position.character] !== ' ') return null; // comment or string
  const hit = wordAt(line, position.character, (n) => Object.prototype.hasOwnProperty.call(functions, n));
  if (!hit) return null;
  return { ...hit, range: new vscode.Range(position.line, hit.start, position.line, hit.end) };
}

async function userHelp(name, folder) {
  // Only names that have a NAME.m next to the file: no process is started for ordinary variables.
  if (!folder || !/^[A-Za-z_]\w*$/.test(name) || !fs.existsSync(path.join(folder, `${name}.m`))) return null;
  const key = `${folder}\n${name}`;
  const cached = helpCache.get(key);
  if (cached && Date.now() - cached.time < 15000) return cached.text;
  const exe = await findMylab(true);
  if (!exe) return null;
  const result = await execMylab(exe, ['--help-for', name], folder, 15000);
  const text = result.code === 0 ? result.stdout : null;
  helpCache.set(key, { time: Date.now(), text });
  return text;
}

async function provideHover(doc, position) {
  const hit = nameAt(doc, position);
  if (!hit) return null;
  if (Object.prototype.hasOwnProperty.call(functions, hit.word)) {
    return new vscode.Hover(new vscode.MarkdownString(helpLib.hoverMarkdown(hit.word, functions[hit.word])), hit.range);
  }
  if (helpLib.KEYWORDS.includes(hit.word)) return null;
  const text = await userHelp(hit.word, doc.uri.scheme === 'file' ? path.dirname(doc.uri.fsPath) : null);
  const md = text && helpLib.hoverFromHelpText(hit.word, text);
  return md ? new vscode.Hover(new vscode.MarkdownString(md), hit.range) : null;
}

function provideCompletionItems(doc, position) {
  const line = doc.lineAt(position.line).text.slice(0, position.character);
  const code = codeOnly(line);
  if (line.length && code[line.length - 1] === ' ' && line[line.length - 1] !== ' ') return []; // comment or string
  if (/\.\s*[A-Za-z_]\w*$/.test(line) && !/containers\.\w*$/.test(line)) return []; // a field name: s.fie|
  if (!completionCache) {
    completionCache = helpLib.completionEntries(functions).map((e) => {
      const item = new vscode.CompletionItem(
        e.label,
        e.kind === 'function' ? vscode.CompletionItemKind.Function : vscode.CompletionItemKind.Keyword
      );
      item.detail = e.detail;
      if (e.documentation) item.documentation = e.documentation;
      return item;
    });
  }
  const items = completionCache.slice();
  if (doc.uri.scheme === 'file') {
    try {
      const self = path.basename(doc.uri.fsPath);
      for (const f of fs.readdirSync(path.dirname(doc.uri.fsPath))) {
        const name = f.slice(0, -2);
        if (!f.endsWith('.m') || f === self || !/^[A-Za-z_]\w*$/.test(name)) continue;
        if (Object.prototype.hasOwnProperty.call(functions, name)) continue;
        const item = new vscode.CompletionItem(name, vscode.CompletionItemKind.Function);
        item.detail = `${f} (this folder)`;
        items.push(item);
      }
    } catch (e) {
      // the folder cannot be read: built-in names only
    }
  }
  return items;
}

function provideSignatureHelp(doc, position) {
  const before = doc.lineAt(position.line).text.slice(0, position.character);
  const ctx = helpLib.callContext(before);
  if (!ctx || !Object.prototype.hasOwnProperty.call(functions, ctx.name)) return null;
  const entry = functions[ctx.name];
  const sigs = helpLib.signatures(ctx.name, entry);
  if (!sigs.length) return null;
  const help = new vscode.SignatureHelp();
  help.signatures = sigs.map((s) => {
    const info = new vscode.SignatureInformation(s.label, helpLib.summary(entry.description));
    info.parameters = s.parameters.map((p) => new vscode.ParameterInformation([p.start, p.end]));
    return info;
  });
  const active = helpLib.activeSignature(sigs, ctx.argIndex);
  help.activeSignature = active.index;
  help.activeParameter = active.parameter;
  return help;
}

function provideFoldingRanges(doc) {
  return foldingRanges(splitLines(doc.getText())).map((r) => new vscode.FoldingRange(
    r.start,
    r.end,
    r.kind === 'comment' ? vscode.FoldingRangeKind.Comment
      : r.kind === 'section' ? vscode.FoldingRangeKind.Region : undefined
  ));
}

function provideDocumentSymbols(doc) {
  const kinds = {
    function: vscode.SymbolKind.Function,
    class: vscode.SymbolKind.Class,
    section: vscode.SymbolKind.Namespace,
  };
  return outline(splitLines(doc.getText())).map((s) => {
    const full = new vscode.Range(s.start, 0, s.end, doc.lineAt(s.end).text.length);
    const first = doc.lineAt(s.start).text;
    const nameEnd = s.kind === 'section' ? first.length : s.nameColumn + s.name.length;
    const selection = new vscode.Range(s.start, s.nameColumn, s.start, Math.max(nameEnd, s.nameColumn));
    return new vscode.DocumentSymbol(s.name, s.kind === 'section' ? 'section' : '', kinds[s.kind], full, selection);
  });
}

// -- help command --------------------------------------------------------------------------

async function showHelp(emitter) {
  const editor = vscode.window.activeTextEditor;
  let name = null;
  if (editor) {
    const hit = editor.selection.isEmpty
      ? nameAt(editor.document, editor.selection.active)
      : { word: editor.document.getText(editor.selection).trim() };
    if (hit && /^[A-Za-z_][\w.]*$/.test(hit.word)) name = hit.word;
  }
  if (!name) {
    name = await vscode.window.showInputBox({ prompt: 'Show help for which function?', placeHolder: 'for example: linspace' });
    if (!name) return;
    name = name.trim();
  }
  let text;
  if (Object.prototype.hasOwnProperty.call(functions, name)) {
    text = helpLib.helpPlainText(name, functions[name]);
  } else {
    const exe = await findMylab(false);
    if (!exe) return;
    const folder = folderOf(editor && editor.document);
    const result = await execMylab(exe, ['--help-for', name], folder, 15000);
    if (result.code !== 0 || !result.stdout.trim()) {
      vscode.window.showInformationMessage(`mylab has no help for "${name}".`);
      return;
    }
    text = `${name}\n\n${result.stdout}`;
  }
  const uri = vscode.Uri.parse(`${HELP_SCHEME}:${encodeURIComponent(name)}.txt`);
  helpTexts.set(uri.toString(), text);
  emitter.fire(uri);
  const helpDoc = await vscode.workspace.openTextDocument(uri);
  await vscode.window.showTextDocument(helpDoc, { viewColumn: vscode.ViewColumn.Beside, preview: true, preserveFocus: true });
}

module.exports = { activate, deactivate };
