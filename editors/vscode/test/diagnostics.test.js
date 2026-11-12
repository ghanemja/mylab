'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { parseCheckOutput, toDiagnostics } = require('../src/diagnostics');

test('parseCheckOutput reads the JSON that mylab --check prints', () => {
  const out = '{"ok": false, "problems": [{"line": 2, "column": 13, "severity": "error", "message": "unexpected \';\'"}]}\n';
  assert.deepEqual(parseCheckOutput(out), {
    ok: false,
    problems: [{ line: 2, column: 13, severity: 'error', message: "unexpected ';'" }],
  });
  assert.deepEqual(parseCheckOutput('{"ok": true, "problems": []}'), { ok: true, problems: [] });
});

test('parseCheckOutput tolerates noise before the JSON and rejects anything else', () => {
  assert.deepEqual(parseCheckOutput('some warning\n{"ok": true, "problems": []}\n'), { ok: true, problems: [] });
  assert.equal(parseCheckOutput(''), null);
  assert.equal(parseCheckOutput('usage: mylab [-h] ...'), null);
  assert.equal(parseCheckOutput('{"ok": true}'), null);
  assert.equal(parseCheckOutput('{broken'), null);
});

test('parseCheckOutput fills in missing fields', () => {
  const r = parseCheckOutput('{"ok": false, "problems": [{"message": "bad"}, {"line": 3}]}');
  assert.deepEqual(r.problems, [{ line: 1, column: 1, severity: 'error', message: 'bad' }]);
  assert.equal(r.ok, false);
});

const TEXT = [
  'x = 1;', //                  line 1
  'y = foo(x) +;', //           line 2
  '   z = helper(3); % helper', // line 3
  "s = 'helper'; w = helper2(1);", // line 4
  'total = total1 + 2', //      line 5
  '', //                        line 6
].join('\n');

test('an error is underlined at its column', () => {
  const [d] = toDiagnostics([{ line: 2, column: 13, severity: 'error', message: "unexpected ';'" }], TEXT);
  assert.deepEqual(d, {
    startLine: 1, startColumn: 12, endLine: 1, endColumn: 13, severity: 'error', message: "Unexpected ';'",
  });
});

test('an error on a name underlines the whole name', () => {
  const [d] = toDiagnostics([{ line: 5, column: 11, severity: 'error', message: 'something odd' }], TEXT);
  assert.deepEqual([d.startColumn, d.endColumn], [8, 14]);
});

test('an unknown-function warning underlines the name, not comments or text', () => {
  const msg = (n) => `'${n}' is not a function mylab provides, and no ${n}.m was found in this folder.`;
  const [a] = toDiagnostics([{ line: 3, column: 1, severity: 'warning', message: msg('helper') }], TEXT);
  assert.deepEqual([a.startLine, a.startColumn, a.endColumn, a.severity], [2, 7, 13, 'warning']);
  const [b] = toDiagnostics([{ line: 4, column: 1, severity: 'warning', message: msg('helper2') }], TEXT);
  assert.deepEqual([b.startColumn, b.endColumn], [18, 25]);
});

test('a name that is not on the line falls back to the line text', () => {
  const [d] = toDiagnostics([{ line: 3, column: 1, severity: 'warning', message: "'nothere' is unknown" }], TEXT);
  assert.deepEqual([d.startLine, d.startColumn, d.endColumn], [2, 3, 26]); // the text of the line
});

test('positions outside the file are clamped', () => {
  const [past] = toDiagnostics([{ line: 99, column: 1, severity: 'error', message: "unbalanced '['" }], 'a = [1 2\nb = 2;');
  assert.deepEqual([past.startLine, past.startColumn, past.endColumn], [1, 0, 1]);
  const [eol] = toDiagnostics([{ line: 1, column: 40, severity: 'error', message: 'unexpected end of input' }], '  a = [1 2  ');
  assert.deepEqual([eol.startLine, eol.startColumn, eol.endColumn], [0, 2, 10]);
  const [blank] = toDiagnostics([{ line: 1, column: 1, severity: 'error', message: 'oops' }], '');
  assert.deepEqual([blank.startColumn, blank.endColumn], [0, 1]);
});
