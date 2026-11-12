'use strict';
// Checks on the files VS Code reads directly: package.json, the grammar, the
// language configuration and the snippets.

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.join(__dirname, '..');
const read = (rel) => JSON.parse(fs.readFileSync(path.join(ROOT, rel), 'utf8'));
const manifest = read('package.json');
const grammar = read('syntaxes/mylab.tmLanguage.json');

test('package.json points at files that exist', () => {
  const c = manifest.contributes;
  const files = [
    manifest.main,
    manifest.icon,
    ...c.languages.map((l) => l.configuration),
    ...c.grammars.map((g) => g.path),
    ...c.snippets.map((s) => s.path),
    'README.md', 'CHANGELOG.md', 'LICENSE', 'data/functions.json',
  ];
  for (const f of files) assert.ok(fs.existsSync(path.join(ROOT, f)), f);
  const png = fs.readFileSync(path.join(ROOT, manifest.icon));
  assert.equal(png.subarray(1, 4).toString(), 'PNG');
  assert.ok(png.readUInt32BE(16) >= 128, 'the Marketplace wants an icon of at least 128 pixels');
});

test('package.json basics', () => {
  assert.equal(manifest.name, 'mylab');
  assert.equal(manifest.license, 'MIT');
  assert.match(manifest.engines.vscode, /^\^1\.\d+\.0$/);
  assert.equal(manifest.repository.url, 'https://github.com/ghanemja/mylab.git');
  assert.equal(manifest.bugs.url, 'https://github.com/ghanemja/mylab/issues');
  assert.deepEqual(manifest.contributes.languages[0].extensions, ['.m']);
  assert.ok(manifest.contributes.languages[0].aliases.includes('MATLAB (mylab)'));
  assert.ok(manifest.activationEvents.includes('onLanguage:matlab'), 'also wakes up for the MathWorks language id');
  assert.equal(manifest.dependencies, undefined, 'no runtime dependencies');
  assert.equal(manifest.contributes.grammars[0].scopeName, grammar.scopeName);
});

test('menus and keybindings only use declared commands', () => {
  const c = manifest.contributes;
  const declared = new Set(c.commands.map((x) => x.command));
  for (const group of Object.values(c.menus)) for (const item of group) assert.ok(declared.has(item.command), item.command);
  for (const kb of c.keybindings) {
    assert.ok(declared.has(kb.command), kb.command);
    assert.match(kb.when, /editorLangId =~ \/\^\(mylab-matlab\|matlab\)\$\//, 'only active in .m files');
  }
  for (const want of ['mylab.runFile', 'mylab.runSelection', 'mylab.debugFile', 'mylab.openPrompt', 'mylab.showHelp']) {
    assert.ok(declared.has(want), want);
  }
  for (const key of ['mylab.executable', 'mylab.checkOnType', 'mylab.runInTerminal']) {
    assert.ok(c.configuration.properties[key], key);
  }
  assert.equal(c.configuration.properties['mylab.executable'].default, 'mylab');
});

function walk(node, visit) {
  if (Array.isArray(node)) node.forEach((n) => walk(n, visit));
  else if (node && typeof node === 'object') {
    visit(node);
    Object.values(node).forEach((n) => walk(n, visit));
  }
}

test('grammar: every include resolves and every pattern compiles', () => {
  let count = 0;
  walk(grammar, (node) => {
    if (typeof node.include === 'string' && node.include.startsWith('#')) {
      assert.ok(grammar.repository[node.include.slice(1)], `missing rule ${node.include}`);
    }
    for (const key of ['match', 'begin', 'end']) {
      if (typeof node[key] === 'string') {
        assert.doesNotThrow(() => new RegExp(node[key]), `${key}: ${node[key]}`);
        count += 1;
      }
    }
  });
  assert.ok(count > 30);
});

// The patterns below are written so that JavaScript reads them the same way VS Code's
// regular-expression engine does, which lets us try them here.
const rule = (name) => grammar.repository[name];
const re = (source) => new RegExp(source, 'g');
const all = (source, text) => [...text.matchAll(re(source))].map((m) => m[0]);

test('grammar: transpose versus text', () => {
  const transpose = rule('transpose').match;
  assert.deepEqual(all(transpose, "y = x' + [a' b']' + z.' + c(1)' + s{2}'"), ["'", "'", "'", "'", "'", "'", "'"]);
  assert.deepEqual(all(transpose, "disp('hi'); s = ['a', 'b']; t = {'c'};").length, 4, 'only closing quotes, which the text rule consumes first');
  assert.equal(new RegExp(transpose).test("= 'text"), false);
  assert.equal(new RegExp(transpose).test("('text"), false);
});

test('grammar: numbers', () => {
  const numbers = (text) => rule('number').patterns.flatMap((p) => all(p.match, text));
  assert.deepEqual(numbers('a = 42 + 3.14 - .5 + 1e3 + 2.5E-4 + 6.02d23;'), ['42', '3.14', '.5', '1e3', '2.5E-4', '6.02d23']);
  assert.deepEqual(numbers('z = 3i + 4.5j; h = 0xFF + 0x1Fu8 + 0b1010;'), ['0xFF', '0x1Fu8', '0b1010', '3i', '4.5j']);
  assert.deepEqual(numbers('x1 = y2.z3; v = a.^2 + 3.*b;'), ['2', '3'], 'digits inside names are not numbers; 3.*b keeps its operator');
  assert.deepEqual(numbers('r = 1:10;'), ['1', '10']);
});

test('grammar: keywords, comments, sections and definitions', () => {
  const control = rule('keyword').patterns[0].match;
  assert.deepEqual(all(control, 'if a, for k = 1:n, end, elseif b; s.end = 1; endless = 2; end'), ['if', 'for', 'end', 'elseif', 'end']);
  assert.equal(new RegExp(rule('section').match).test('%% Title'), true);
  assert.equal(new RegExp(rule('section').match).test('  %%'), true);
  assert.equal(new RegExp(rule('section').match).test('%%% banner'), false);
  assert.equal(new RegExp(rule('block-comment').begin).test('  %{'), true);
  assert.equal(new RegExp(rule('block-comment').begin).test('%{ not a block'), false);
  const def = new RegExp(rule('function-definition').match);
  assert.deepEqual(def.exec('function [a, b] = two(x)').slice(1, 5), ['function', '[a, b]', '=', 'two']);
  assert.deepEqual(def.exec('  function y = f(x)').slice(1, 5), ['function', 'y', '=', 'f']);
  assert.deepEqual(def.exec('function show(x)').slice(1, 5), ['function', undefined, undefined, 'show']);
  assert.deepEqual(def.exec('function obj = set.Value(obj, v)')[4], 'set.Value');
  assert.equal(new RegExp(rule('class-definition').match).exec('classdef (Sealed) Point < handle')[2], 'Point');
  assert.deepEqual(new RegExp(rule('continuation').match).exec('a = 1 + ... the rest').slice(1), ['...', ' the rest']);
});

test('language configuration: indentation rules', () => {
  const conf = read('language-configuration.json');
  assert.equal(conf.comments.lineComment, '%');
  assert.deepEqual(conf.comments.blockComment, ['%{', '%}']);
  const inc = new RegExp(conf.indentationRules.increaseIndentPattern);
  const dec = new RegExp(conf.indentationRules.decreaseIndentPattern);
  for (const line of ['if x > 0', '  for k = 1:3', 'while true', 'switch v', '  case 1', '  otherwise', 'try', 'catch err',
    'function y = f(x)', 'else', 'elseif y', 'classdef A', '  properties', '  methods (Static)', 'parfor i = 1:4']) {
    assert.ok(inc.test(line), `should indent after: ${line}`);
  }
  for (const line of ['x = 1;', 'if x, y = 1; end', 'for k = 1:3, disp(k); end', 'disp(ifelse)', 'format long', 'endless = 3;',
    'properties = 3;']) {
    assert.ok(!inc.test(line), `should not indent after: ${line}`);
  }
  for (const line of ['end', '  end;', 'else', 'elseif x', 'case 2', 'otherwise', 'catch', 'end % loop']) {
    assert.ok(dec.test(line), `should outdent: ${line}`);
  }
  for (const line of ['x = y(end);', 'endless = 1;', 'end_value = 2;', 'disp(end)', 'else_part = 1;']) {
    assert.ok(!dec.test(line), `should not outdent: ${line}`);
  }
  for (const key of ['wordPattern']) assert.doesNotThrow(() => new RegExp(conf[key]));
});

test('snippets cover the main blocks', () => {
  const snippets = read('snippets/mylab.json');
  const prefixes = Object.values(snippets).flatMap((s) => [].concat(s.prefix));
  for (const want of ['for', 'if', 'while', 'switch', 'function', 'try']) assert.ok(prefixes.includes(want), want);
  for (const [name, s] of Object.entries(snippets)) {
    assert.ok(Array.isArray(s.body) && s.body.length, name);
    assert.ok(s.description, name);
    const body = s.body.join('\n');
    if (/^(for|if|while|switch|function|try)/.test(body)) assert.match(body, /\nend$/, `${name} closes its block`);
  }
});

test('.vscodeignore keeps tests and tools out of the package', () => {
  const ignore = fs.readFileSync(path.join(ROOT, '.vscodeignore'), 'utf8');
  for (const want of ['test/**', 'test-support/**', 'tools/**']) assert.ok(ignore.includes(want), want);
});
