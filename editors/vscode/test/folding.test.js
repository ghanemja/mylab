'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { foldingRanges, outline } = require('../src/folding');
const { WorkCopies } = require('../src/workcopy');

const pairs = (lines, kind) => foldingRanges(lines).filter((r) => !kind || r.kind === kind).map((r) => [r.start, r.end]);

test('blocks fold from their first line to their end', () => {
  const lines = [
    'function y = f(x)', //   0
    '  if x > 0', //          1
    '    y = x(end);', //     2  end inside brackets is an index
    '  else', //              3
    "    y = 'end';", //      4  end inside text
    '  end', //               5
    '  for k = 1:3, y = y + k; end', // 6  one line: no fold
    '  while true % for', //  7
    '    break', //           8
    '  end', //               9
    'end', //                10
  ];
  assert.deepEqual(pairs(lines, 'block'), [[0, 10], [1, 5], [7, 9]]);
});

test('sections fold up to the next section, without trailing blank lines', () => {
  const lines = ['%% One', 'a = 1;', '', '%% Two', 'b = 2;', 'c = 3;', ''];
  assert.deepEqual(pairs(lines, 'section'), [[0, 1], [3, 5]]);
});

test('block comments fold and hide the keywords inside them', () => {
  const lines = ['%{', 'if this were code', 'for', '%}', 'x = 1;'];
  assert.deepEqual(foldingRanges(lines), [{ start: 0, end: 3, kind: 'comment' }]);
});

test('files whose functions have no end still fold per function', () => {
  const lines = [
    'function a()', //  0
    '  if true', //     1
    '    x = 1;', //    2
    '  end', //         3
    '', //              4
    'function b()', //  5
    '  y = 2;', //      6
  ];
  assert.deepEqual(pairs(lines, 'block'), [[0, 3], [1, 3], [5, 6]]);
});

test('classdef blocks fold', () => {
  const lines = [
    'classdef Point', //            0
    '  properties', //              1
    '    x', //                     2
    '  end', //                     3
    '  methods (Static)', //        4
    '    function p = origin()', // 5
    '      p = Point();', //        6
    '    end', //                   7
    '  end', //                     8
    'end', //                       9
  ];
  assert.deepEqual(pairs(lines, 'block'), [[0, 9], [1, 3], [4, 8], [5, 7]]);
});

test('a struct field called end, and switch blocks', () => {
  const lines = ['switch s.end', '  case 1', '    x = 1;', '  otherwise', '    x = 2;', 'end'];
  assert.deepEqual(pairs(lines, 'block'), [[0, 5]]);
});

test('outline lists functions, classes and sections', () => {
  const lines = [
    '%% Setup', //                       0
    'x = 1;', //                         1
    '%% Helpers', //                     2
    'function [a, b] = two(x)', //       3
    '  a = x; b = x;', //                4
    'end', //                            5
    'function show(x)', //               6
    '  disp(x)', //                      7
    'end', //                            8
  ];
  const got = outline(lines).map((s) => [s.kind, s.name, s.start, s.end]);
  assert.deepEqual(got, [
    ['section', 'Setup', 0, 1],
    ['section', 'Helpers', 2, 8],
    ['function', 'two', 3, 5],
    ['function', 'show', 6, 8],
  ]);
  const two = outline(lines).find((s) => s.name === 'two');
  assert.equal(lines[3].slice(two.nameColumn, two.nameColumn + 3), 'two');
});

test('WorkCopies mirrors the folder: the text plus empty stand-ins for the other .m files', () => {
  const src = fs.mkdtempSync(path.join(os.tmpdir(), 'mylab-test-src-'));
  const copies = new WorkCopies();
  try {
    fs.writeFileSync(path.join(src, 'main.m'), 'saved text');
    fs.writeFileSync(path.join(src, 'helper.m'), 'function helper()\nend\n');
    fs.writeFileSync(path.join(src, 'notes.txt'), 'x');
    const copy = copies.write(path.join(src, 'main.m'), 'unsaved text');
    assert.equal(path.basename(copy), 'main.m');
    assert.notEqual(path.dirname(copy), src);
    assert.equal(fs.readFileSync(copy, 'utf8'), 'unsaved text');
    assert.deepEqual(fs.readdirSync(path.dirname(copy)).sort(), ['helper.m', 'main.m']);
    assert.equal(fs.readFileSync(path.join(src, 'main.m'), 'utf8'), 'saved text', 'the real file is untouched');
    assert.deepEqual(fs.readdirSync(src).sort(), ['helper.m', 'main.m', 'notes.txt'], 'nothing is added to the folder');

    fs.rmSync(path.join(src, 'helper.m'));
    fs.writeFileSync(path.join(src, 'other.m'), '');
    const again = copies.write(path.join(src, 'main.m'), 'newer');
    assert.equal(again, copy);
    assert.deepEqual(fs.readdirSync(path.dirname(copy)).sort(), ['main.m', 'other.m']);

    const untitled = copies.write(null, 'x = 1;');
    assert.equal(path.basename(untitled), 'untitled.m');
    const script = copies.tempScript('\n\ny = 2;\n', 'main.m');
    assert.equal(fs.readFileSync(script, 'utf8'), '\n\ny = 2;\n');
    const root = copies.root;
    copies.dispose();
    assert.equal(fs.existsSync(root), false);
  } finally {
    copies.dispose();
    fs.rmSync(src, { recursive: true, force: true });
  }
});
