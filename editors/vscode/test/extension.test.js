'use strict';
// Loads the real src/extension.js against a small fake of the `vscode` module
// (test-support/fake-vscode.js) and a fake mylab command, to check that the
// activation layer is wired up: commands exist, providers answer, problems
// reach the Problems panel. It is not a substitute for trying the extension
// in VS Code, but it catches broken wiring without needing VS Code or Python.

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { create, loadExtension } = require('../test-support/fake-vscode');

const ROOT = path.join(__dirname, '..');
const manifest = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'));
const FAKE_MYLAB = `"${process.execPath}" "${path.join(__dirname, '..', 'test-support', 'fake-mylab.js')}"`;

function setup(settings) {
  const fake = create({ 'mylab.executable': FAKE_MYLAB, ...settings });
  const extension = loadExtension(fake);
  const context = { extensionPath: ROOT, subscriptions: [] };
  extension.activate(context);
  return { ...fake, extension, context };
}

async function waitFor(predicate, what) {
  const start = Date.now();
  while (!predicate()) {
    if (Date.now() - start > 10000) throw new Error(`timed out waiting for ${what}`);
    await new Promise((r) => setTimeout(r, 20));
  }
}

function tempFolder(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'mylab-ext-test-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}

test('every command in package.json is registered, and nothing else', (t) => {
  const { state, extension } = setup();
  t.after(() => extension.deactivate());
  const declared = manifest.contributes.commands.map((c) => c.command).sort();
  assert.deepEqual([...state.commands.keys()].sort(), declared);
  for (const name of ['hover', 'completion', 'signature', 'folding', 'symbols', 'content']) {
    assert.ok(state.providers[name], `${name} provider`);
  }
});

test('opening a file reports problems with ranges; fixing it clears them', async (t) => {
  const { state, extension, makeDocument } = setup();
  t.after(() => extension.deactivate());
  const dir = tempFolder(t);
  const file = path.join(dir, 'main.m');
  fs.writeFileSync(file, 'x = 1;\ny = BAD + 2;\nz = unknownfn(3);\n');
  const doc = makeDocument(file, fs.readFileSync(file, 'utf8'));
  state.listeners.open.forEach((fn) => fn(doc));
  await waitFor(() => state.diagnostics.has(doc.uri.toString()), 'diagnostics');
  const list = state.diagnostics.get(doc.uri.toString());
  assert.equal(list.length, 2);
  assert.deepEqual([list[0].range.start.line, list[0].range.start.character, list[0].range.end.character], [1, 4, 7]);
  assert.equal(list[0].severity, 0);
  assert.equal(list[0].source, 'mylab');
  assert.deepEqual([list[1].range.start.line, list[1].range.start.character, list[1].range.end.character], [2, 4, 13]);
  assert.equal(list[1].severity, 1);

  // Unsaved edits are checked through a temporary copy; the file on disk is not touched.
  doc.text = 'x = 1;\ny = 2;\n';
  doc.isDirty = true;
  state.listeners.change.forEach((fn) => fn({ document: doc, contentChanges: [{}] }));
  await waitFor(() => state.diagnostics.get(doc.uri.toString()).length === 0, 'cleared diagnostics');
  assert.equal(fs.readFileSync(file, 'utf8'), 'x = 1;\ny = BAD + 2;\nz = unknownfn(3);\n');
  assert.deepEqual(fs.readdirSync(dir), ['main.m']);

  state.listeners.close.forEach((fn) => fn(doc));
  assert.equal(state.diagnostics.has(doc.uri.toString()), false);
});

test('unsaved text knows about the other .m files in its folder', async (t) => {
  const { state, extension, makeDocument } = setup();
  t.after(() => extension.deactivate());
  const dir = tempFolder(t);
  fs.writeFileSync(path.join(dir, 'unknownhelper.m'), 'function unknownhelper(x)\nend\n');
  const doc = makeDocument(path.join(dir, 'main.m'), 'unknownhelper(1);\nunknownother(2);\n', { dirty: true });
  state.listeners.open.forEach((fn) => fn(doc));
  await waitFor(() => state.diagnostics.has(doc.uri.toString()), 'diagnostics');
  const list = state.diagnostics.get(doc.uri.toString());
  assert.deepEqual(list.map((d) => d.range.start.line), [1]);
});

test('checkOnType off: typing does not check, saving does', async (t) => {
  const { state, extension, makeDocument } = setup({ 'mylab.checkOnType': false });
  t.after(() => extension.deactivate());
  const dir = tempFolder(t);
  const file = path.join(dir, 'a.m');
  fs.writeFileSync(file, 'BAD\n');
  const doc = makeDocument(file, 'BAD\n', { languageId: 'matlab' });
  state.listeners.change.forEach((fn) => fn({ document: doc, contentChanges: [{}] }));
  await new Promise((r) => setTimeout(r, 700));
  assert.equal(state.diagnostics.has(doc.uri.toString()), false);
  state.listeners.save.forEach((fn) => fn(doc));
  await waitFor(() => state.diagnostics.has(doc.uri.toString()), 'diagnostics after save');
});

test('files in other languages are left alone', async (t) => {
  const { state, extension, makeDocument } = setup();
  t.after(() => extension.deactivate());
  const doc = makeDocument('/tmp/x.py', 'BAD', { languageId: 'python' });
  state.listeners.open.forEach((fn) => fn(doc));
  await new Promise((r) => setTimeout(r, 300));
  assert.equal(state.diagnostics.size, 0);
});

test('Run File and Debug File start mylab as a task in the file\'s folder', async (t) => {
  const { state, extension, makeDocument, makeEditor } = setup();
  t.after(() => extension.deactivate());
  const dir = tempFolder(t);
  const file = path.join(dir, 'demo.m');
  fs.writeFileSync(file, 'disp(1)\n');
  state.activeTextEditor = makeEditor(makeDocument(file, 'disp(1)\n'), 0, 0);
  await state.commands.get('mylab.runFile')();
  await state.commands.get('mylab.debugFile')();
  assert.equal(state.tasks.length, 2);
  const [run, debug] = state.tasks;
  assert.equal(run.execution.process, process.execPath);
  assert.deepEqual(run.execution.args.slice(1), [file]);
  assert.equal(run.execution.options.cwd, dir);
  assert.deepEqual(debug.execution.args.slice(1), ['--debug', file]);
  assert.equal(debug.presentationOptions.focus, true);
});

test('Run Selection / Current Section runs a padded temporary copy', async (t) => {
  const { state, extension, makeDocument, makeEditor } = setup();
  t.after(() => extension.deactivate());
  const dir = tempFolder(t);
  const file = path.join(dir, 'demo.m');
  const text = '%% A\nx = 1;\n%% B\ny = 2;\ndisp(y)\n';
  state.activeTextEditor = makeEditor(makeDocument(file, text, { dirty: true }), 4, 2);
  await state.commands.get('mylab.runSelection')();
  const task = state.tasks[0];
  const temp = task.execution.args[task.execution.args.length - 1];
  assert.equal(path.basename(temp), 'demo.m');
  assert.notEqual(path.dirname(temp), dir);
  assert.equal(fs.readFileSync(temp, 'utf8'), '\n\n%% B\ny = 2;\ndisp(y)\n');
  assert.equal(task.execution.options.cwd, dir);
  assert.match(task.name, /section .*lines 3-5/);

  state.activeTextEditor = makeEditor(makeDocument(file, text), 1, 0, 1, 6);
  await state.commands.get('mylab.runSelection')();
  const temp2 = state.tasks[1].execution.args.at(-1);
  assert.equal(fs.readFileSync(temp2, 'utf8'), '\nx = 1;\n');
});

test('runInTerminal off sends the output to the mylab channel', async (t) => {
  const { state, extension, makeDocument, makeEditor } = setup({ 'mylab.runInTerminal': false });
  t.after(() => extension.deactivate());
  const dir = tempFolder(t);
  const file = path.join(dir, 'demo.m');
  fs.writeFileSync(file, 'disp(1)\n');
  state.activeTextEditor = makeEditor(makeDocument(file, 'disp(1)\n'), 0, 0);
  await state.commands.get('mylab.runFile')();
  assert.equal(state.tasks.length, 0);
  await waitFor(() => state.outputLines.some((l) => l.includes('[finished]')), 'the run to finish');
  assert.ok(state.outputLines.some((l) => l.includes(`ran ${file}`)));
});

test('Open Interactive Prompt starts mylab in a terminal', async (t) => {
  const { state, extension } = setup();
  t.after(() => extension.deactivate());
  await state.commands.get('mylab.openPrompt')();
  assert.equal(state.terminals.length, 1);
  assert.equal(state.terminals[0].shellPath, process.execPath);
  assert.equal(state.terminals[0].name, 'mylab');
});

test('when mylab is missing, a message says how to install it', async (t) => {
  const { state, extension, makeDocument, makeEditor } = setup({ 'mylab.executable': 'definitely-not-installed-mylab' });
  t.after(() => extension.deactivate());
  // Hide any real mylab or Python from the search.
  const savedPath = process.env.PATH;
  process.env.PATH = '';
  t.after(() => {
    process.env.PATH = savedPath;
  });
  state.activeTextEditor = makeEditor(makeDocument(null, 'disp(1)\n'), 0, 0);
  await state.commands.get('mylab.runFile')();
  assert.equal(state.tasks.length, 0);
  const error = state.messages.find((m) => m[0] === 'error');
  assert.ok(error, 'an error message is shown');
  assert.match(error[1], /mylab not found — install with: pip install mylab-lang/);
  assert.ok(error.includes('Open Install Guide'));
});

test('hover: built-in functions from the bundled data, user functions from mylab', async (t) => {
  const { state, extension, makeDocument, Position } = setup();
  t.after(() => extension.deactivate());
  const dir = tempFolder(t);
  fs.writeFileSync(path.join(dir, 'myhelper.m'), '% help\n');
  const doc = makeDocument(path.join(dir, 'main.m'), "y = linspace(0, 1); % zeros here\nz = myhelper(y) + other;\ns = 'zeros';\n");
  const builtin = await state.providers.hover.provideHover(doc, new Position(0, 6));
  assert.match(builtin.contents.value, /^```matlab\n.*linspace\(/);
  assert.deepEqual([builtin.range.start.character, builtin.range.end.character], [4, 12]);
  assert.equal(await state.providers.hover.provideHover(doc, new Position(0, 24)), null, 'no hover in comments');
  assert.equal(await state.providers.hover.provideHover(doc, new Position(2, 7)), null, 'no hover in text');
  const user = await state.providers.hover.provideHover(doc, new Position(1, 6));
  assert.match(user.contents.value, /MYHELPER {2}Help from the file/);
  assert.equal(await state.providers.hover.provideHover(doc, new Position(1, 20)), null, 'plain variables have no hover');
});

test('completion, signature help, folding and outline answer', (t) => {
  const { state, extension, makeDocument, Position } = setup();
  t.after(() => extension.deactivate());
  const dir = tempFolder(t);
  fs.writeFileSync(path.join(dir, 'myhelper.m'), '');
  const doc = makeDocument(path.join(dir, 'main.m'), 'function r = f(x)\n  r = linspace(0, \nend\n% lin\ns.fie\n');
  const items = state.providers.completion.provideCompletionItems(doc, new Position(1, 9));
  const labels = items.map((i) => i.label);
  for (const want of ['linspace', 'zeros', 'for', 'myhelper']) assert.ok(labels.includes(want), want);
  assert.ok(!labels.includes('main'));
  const linspace = items.find((i) => i.label === 'linspace');
  assert.match(linspace.detail, /linspace\(/);
  assert.ok(linspace.documentation.endsWith('.'));
  assert.deepEqual(state.providers.completion.provideCompletionItems(doc, new Position(3, 5)), [], 'none in comments');
  assert.deepEqual(state.providers.completion.provideCompletionItems(doc, new Position(4, 5)), [], 'none for field names');

  const sig = state.providers.signature.provideSignatureHelp(doc, new Position(1, 18));
  assert.ok(sig.signatures.length >= 1);
  assert.equal(sig.activeParameter, 1);
  assert.ok(sig.signatures[sig.activeSignature].parameters.length >= 2);
  assert.equal(state.providers.signature.provideSignatureHelp(doc, new Position(0, 5)), null);

  const folds = state.providers.folding.provideFoldingRanges(doc);
  assert.deepEqual([folds[0].start, folds[0].end], [0, 2]);
  const symbols = state.providers.symbols.provideDocumentSymbols(doc);
  assert.deepEqual(symbols.map((s) => s.name), ['f']);
});

test('Show Help opens the help text beside the editor', async (t) => {
  const { state, extension, makeDocument, makeEditor } = setup();
  t.after(() => extension.deactivate());
  state.activeTextEditor = makeEditor(makeDocument('/tmp/none/main.m', 'y = linspace(0, 1);\n'), 0, 7);
  await state.commands.get('mylab.showHelp')();
  assert.equal(state.shown.length, 1);
  const text = state.shown[0].getText();
  assert.ok(text.startsWith('linspace\n\n  '));
});
