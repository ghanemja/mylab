'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const help = require('../src/help');

const functions = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'data', 'functions.json'), 'utf8'));

test('the bundled reference has the expected shape', () => {
  const names = Object.keys(functions);
  assert.ok(names.length > 500, `only ${names.length} functions`);
  for (const name of names) {
    const e = functions[name];
    assert.ok(Array.isArray(e.usage) && e.usage.length > 0, `${name}: usage`);
    assert.ok(e.usage.every((u) => typeof u === 'string' && u.trim()), `${name}: usage lines`);
    assert.ok(typeof e.description === 'string' && e.description.trim(), `${name}: description`);
  }
  for (const name of ['zeros', 'plot', 'linspace', 'containers.Map']) assert.ok(functions[name], name);
  assert.deepEqual(names, names.slice().sort(), 'names are sorted, as scripts/gen_editor_data.py writes them');
});

test('summary is the first sentence', () => {
  assert.equal(help.summary('Array filled with 0. With no size it returns 0.'), 'Array filled with 0.');
  assert.equal(help.summary('One sentence only.'), 'One sentence only.');
  assert.equal(help.summary('No full stop'), 'No full stop.');
});

test('hoverMarkdown shows the calling forms, then the description', () => {
  const md = help.hoverMarkdown('zeros', {
    usage: ['A = zeros(n)', 'A = zeros(m, n)'],
    description: 'Array filled with 0. Sizes such as [m n] and a_b * 2 work.',
  });
  assert.ok(md.startsWith('```matlab\nA = zeros(n)\nA = zeros(m, n)\n```\n\n'));
  assert.ok(md.includes('Array filled with 0.'));
  assert.ok(md.includes('\\[m n\\]'), 'square brackets are escaped');
  assert.ok(md.includes('a\\_b \\* 2'), 'underscores and stars are escaped');
  assert.ok(md.trimEnd().endsWith('*zeros — mylab built-in function*'));
});

test('hoverMarkdown works for every bundled function', () => {
  for (const name of Object.keys(functions)) {
    const md = help.hoverMarkdown(name, functions[name]);
    assert.ok(md.includes('```matlab\n'), name);
    assert.equal((md.match(/```/g) || []).length, 2, name);
  }
});

test('hoverFromHelpText wraps help for user functions and rejects "not found"', () => {
  const md = help.hoverFromHelpText('area2', 'AREA2  Area of a circle.\n  a = area2(r)\n');
  assert.ok(md.startsWith('```text\nAREA2  Area of a circle.\n  a = area2(r)\n```'));
  assert.ok(md.includes('function in this folder'));
  assert.equal(help.hoverFromHelpText('nope', 'nope not found.\n'), null);
  assert.equal(help.hoverFromHelpText('nope', ''), null);
});

test('helpPlainText lays out usage and wrapped description', () => {
  const text = help.helpPlainText('f', { usage: ['y = f(x)'], description: 'word '.repeat(40).trim() });
  const lines = text.split('\n');
  assert.equal(lines[0], 'f');
  assert.equal(lines[2], '  y = f(x)');
  assert.ok(lines.slice(4).every((l) => l.length <= 76));
  assert.ok(lines[4].startsWith('  word'));
});

test('completionEntries lists functions with summaries, and keywords', () => {
  const entries = help.completionEntries({ zeros: { usage: ['A = zeros(n)'], description: 'Array of 0. More.' } });
  assert.deepEqual(entries[0], { label: 'zeros', kind: 'function', detail: 'A = zeros(n)', documentation: 'Array of 0.' });
  const kws = entries.filter((e) => e.kind === 'keyword').map((e) => e.label);
  for (const kw of ['for', 'if', 'while', 'switch', 'function', 'try', 'end']) assert.ok(kws.includes(kw), kw);
});

test('parseUsage finds the parameters and where they are', () => {
  const s = help.parseUsage('[q, r] = deconv(b, a)', 'deconv');
  assert.equal(s.label, '[q, r] = deconv(b, a)');
  assert.deepEqual(s.parameters.map((p) => p.label), ['b', 'a']);
  for (const p of s.parameters) assert.equal(s.label.slice(p.start, p.end), p.label);
  assert.deepEqual(help.parseUsage("A = zeros(..., 'like', p)", 'zeros').parameters.map((p) => p.label), ['...', "'like'", 'p']);
  assert.deepEqual(help.parseUsage('A = zeros([m n ...])', 'zeros').parameters.map((p) => p.label), ['[m n ...]']);
  assert.deepEqual(help.parseUsage('y = f(g(a, b), c)', 'f').parameters.map((p) => p.label), ['g(a, b)', 'c']);
  assert.deepEqual(help.parseUsage('t = tic()', 'tic').parameters, []);
  assert.equal(help.parseUsage('hold on', 'hold'), null);
  assert.equal(help.parseUsage('x = NaN', 'NaN'), null);
});

test('signatures can be built for the whole reference without errors', () => {
  let withSigs = 0;
  for (const name of Object.keys(functions)) {
    const sigs = help.signatures(name, functions[name]);
    if (sigs.length) withSigs += 1;
    for (const s of sigs) for (const p of s.parameters) assert.equal(s.label.slice(p.start, p.end), p.label);
  }
  assert.ok(withSigs > 400, `${withSigs} functions have signatures`);
});

test('callContext finds the call and the argument being typed', () => {
  assert.deepEqual(help.callContext('y = linspace('), { name: 'linspace', argIndex: 0 });
  assert.deepEqual(help.callContext('y = linspace(0, 1, '), { name: 'linspace', argIndex: 2 });
  assert.deepEqual(help.callContext('y = max(abs(x), '), { name: 'max', argIndex: 1 });
  assert.deepEqual(help.callContext('y = max(abs(x'), { name: 'abs', argIndex: 0 });
  assert.deepEqual(help.callContext('plot(x, [1, 2, '), { name: 'plot', argIndex: 1 });
  assert.deepEqual(help.callContext("fprintf('a, b (c', "), { name: 'fprintf', argIndex: 1 });
  assert.deepEqual(help.callContext("z = size(x', "), { name: 'size', argIndex: 1 });
  assert.deepEqual(help.callContext('m = containers.Map('), { name: 'containers.Map', argIndex: 0 });
  assert.equal(help.callContext('y = linspace(0, 1);'), null);
  assert.equal(help.callContext('y = (1 + '), null);
  assert.equal(help.callContext('x = 1; % call f('), null);
  assert.equal(help.callContext('y = 3'), null);
});

test('activeSignature picks a form that has the argument being typed', () => {
  const sigs = help.signatures('zeros', {
    usage: ['A = zeros(n)', 'A = zeros(m, n)', 'A = zeros(m, n, p, ...)'],
  });
  assert.deepEqual(help.activeSignature(sigs, 0), { index: 0, parameter: 0 });
  assert.deepEqual(help.activeSignature(sigs, 1), { index: 1, parameter: 1 });
  assert.deepEqual(help.activeSignature(sigs, 2), { index: 2, parameter: 2 });
  assert.deepEqual(help.activeSignature(sigs, 7), { index: 2, parameter: 3 }); // the "..." parameter
  const fixed = help.signatures('f', { usage: ['y = f(a)', 'y = f(a, b)'] });
  assert.deepEqual(help.activeSignature(fixed, 5), { index: 1, parameter: 1 });
});
