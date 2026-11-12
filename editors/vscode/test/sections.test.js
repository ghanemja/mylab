'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { splitSections, sectionAt, codeToRun } = require('../src/sections');
const { codeOnly, commentStart, isSectionMarker, wordAt } = require('../src/text');

const FILE = [
  'x = 1;', //            0
  '', //                  1
  '%% Load data', //      2
  'a = [1 2 3];', //      3
  "s = 'not %% a marker';", // 4
  '   %% Plot', //        5
  'plot(a)', //           6
  '%%', //                7
  'disp(x)', //           8
].join('\n');

test('section markers need %% then a space or the end of the line', () => {
  assert.equal(isSectionMarker('%% Title'), true);
  assert.equal(isSectionMarker('%%'), true);
  assert.equal(isSectionMarker('  %%\tTitle'), true);
  assert.equal(isSectionMarker('%%% not a section'), false);
  assert.equal(isSectionMarker('%%Title'), false);
  assert.equal(isSectionMarker('x = 1; %% trailing'), false);
  assert.equal(isSectionMarker('% comment'), false);
});

test('splitSections finds every section, including the text before the first marker', () => {
  assert.deepEqual(splitSections(FILE), [
    { startLine: 0, endLine: 1, title: '' },
    { startLine: 2, endLine: 4, title: 'Load data' },
    { startLine: 5, endLine: 6, title: 'Plot' },
    { startLine: 7, endLine: 8, title: '' },
  ]);
});

test('a file without markers is one section', () => {
  assert.deepEqual(splitSections('a = 1;\nb = 2;'), [{ startLine: 0, endLine: 1, title: '' }]);
  assert.deepEqual(splitSections(''), [{ startLine: 0, endLine: 0, title: '' }]);
});

test('blank lines before the first marker are not a section', () => {
  assert.deepEqual(splitSections('\n\n%% A\nx = 1;'), [{ startLine: 2, endLine: 3, title: 'A' }]);
});

test('Windows line endings', () => {
  assert.deepEqual(splitSections('%% A\r\nx = 1;\r\n%% B\r\ny = 2;'), [
    { startLine: 0, endLine: 1, title: 'A' },
    { startLine: 2, endLine: 3, title: 'B' },
  ]);
});

test('sectionAt picks the section around a line', () => {
  assert.equal(sectionAt(FILE, 0).startLine, 0);
  assert.equal(sectionAt(FILE, 4).startLine, 2);
  assert.equal(sectionAt(FILE, 5).startLine, 5);
  assert.equal(sectionAt(FILE, 8).startLine, 7);
  assert.equal(sectionAt('\n%% A\nx = 1;', 0).startLine, 1); // above the first section: use the first
});

test('codeToRun without a selection runs the section and keeps line numbers', () => {
  const r = codeToRun(FILE, null, 6);
  assert.equal(r.kind, 'section');
  assert.equal(r.startLine, 5);
  assert.equal(r.endLine, 6);
  assert.equal(r.code, '\n\n\n\n\n   %% Plot\nplot(a)\n');
  assert.equal(r.code.split('\n')[6], 'plot(a)'); // same line number as in the file
});

test('codeToRun with an empty selection behaves like no selection', () => {
  const r = codeToRun(FILE, { startLine: 3, startColumn: 2, endLine: 3, endColumn: 2 });
  assert.equal(r.kind, 'section');
  assert.equal(r.startLine, 2);
});

test('codeToRun with a selection runs exactly the selected text', () => {
  const r = codeToRun(FILE, { startLine: 3, startColumn: 4, endLine: 3, endColumn: 11 }, 3);
  assert.equal(r.kind, 'selection');
  assert.equal(r.code, '\n\n\n    [1 2 3]\n');
});

test('a selection ending at the start of a line leaves that line out', () => {
  const r = codeToRun(FILE, { startLine: 3, startColumn: 0, endLine: 5, endColumn: 0 }, 5);
  assert.equal(r.code, "\n\n\na = [1 2 3];\ns = 'not %% a marker';\n");
  assert.equal(r.endLine, 4);
});

test('codeOnly blanks strings and comments but keeps columns', () => {
  assert.equal(codeOnly("x = 'a % b'; % note"), "x = '     ';       ");
  assert.equal(codeOnly("y = x' + 1; % t"), "y = x' + 1;    ");
  assert.equal(codeOnly('s = "if end"; z'), 's = "      "; z');
  assert.equal(codeOnly("a = b'; c = 'end';"), "a = b'; c = '   ';");
  assert.equal(codeOnly("q = 'it''s'; r"), "q = '     '; r");
  assert.equal(codeOnly('a = 1 + ... more'), 'a = 1 +         ');
  assert.equal(codeOnly("m = [a' b']';"), "m = [a' b']';");
  for (const line of ["x = 'abc", '%{', "f('a', \"b\") % c"]) assert.equal(codeOnly(line).length, line.length);
});

test('commentStart ignores percent signs inside text', () => {
  assert.equal(commentStart("fprintf('%d%%\\n', x) % show"), 21);
  assert.equal(commentStart("fprintf('%d\\n', x)"), -1);
  assert.equal(commentStart("y = x' % transpose"), 7);
});

test('wordAt finds names, and dotted names that are known', () => {
  const known = (n) => n === 'containers.Map';
  assert.deepEqual(wordAt('y = linspace(0, 1);', 6), { word: 'linspace', start: 4, end: 12 });
  assert.deepEqual(wordAt('y = linspace(0, 1);', 12), { word: 'linspace', start: 4, end: 12 });
  assert.equal(wordAt('y = linspace(0, 1);', 2), null);
  assert.deepEqual(wordAt('m = containers.Map();', 16, known), { word: 'containers.Map', start: 4, end: 18 });
  assert.deepEqual(wordAt('m = containers.Map();', 6, known), { word: 'containers.Map', start: 4, end: 18 });
  assert.deepEqual(wordAt('v = s.field;', 7, known), { word: 'field', start: 6, end: 11 });
});
