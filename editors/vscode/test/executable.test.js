'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const exe = require('../src/executable');

const cmds = (list) => list.map((c) => [c.command, ...c.args].join(' '));

test('parseCommandLine splits words and respects quotes', () => {
  assert.deepEqual(exe.parseCommandLine('mylab'), ['mylab']);
  assert.deepEqual(exe.parseCommandLine('  python3   -m mylab '), ['python3', '-m', 'mylab']);
  assert.deepEqual(exe.parseCommandLine('"C:\\Program Files\\Python312\\python.exe" -m mylab'),
    ['C:\\Program Files\\Python312\\python.exe', '-m', 'mylab']);
  assert.deepEqual(exe.parseCommandLine("'/opt/my tools/mylab'"), ['/opt/my tools/mylab']);
  assert.deepEqual(exe.parseCommandLine(''), []);
});

test('default setting: PATH first, then the selected Python, then python3/python', () => {
  const list = exe.candidateCommands({ configured: 'mylab', pythonPath: '/home/u/env/bin/python', platform: 'linux' });
  assert.deepEqual(cmds(list), ['mylab', '/home/u/env/bin/python -m mylab', 'python3 -m mylab', 'python -m mylab']);
});

test('a custom setting is tried first, and the default still follows', () => {
  const list = exe.candidateCommands({ configured: 'python3.12 -m mylab', platform: 'linux' });
  assert.deepEqual(cmds(list), ['python3.12 -m mylab', 'mylab', 'python3 -m mylab', 'python -m mylab']);
  assert.match(list[0].label, /setting/);
});

test('a setting that is an existing path with spaces is used whole', () => {
  const p = '/Users/me/My Tools/bin/mylab';
  const list = exe.candidateCommands({ configured: p, platform: 'darwin', exists: (f) => f === p });
  assert.equal(list[0].command, p);
  assert.deepEqual(list[0].args, []);
});

test('virtual environments in the workspace are used when they contain mylab', () => {
  const list = exe.candidateCommands({
    configured: 'mylab',
    platform: 'linux',
    workspaceFolders: ['/work/proj'],
    exists: (f) => f === '/work/proj/.venv/bin/mylab',
  });
  assert.deepEqual(cmds(list), ['mylab', '/work/proj/.venv/bin/mylab', 'python3 -m mylab', 'python -m mylab']);
});

test('Windows uses Scripts\\mylab.exe and the py launcher', () => {
  const list = exe.candidateCommands({
    configured: '',
    platform: 'win32',
    workspaceFolders: ['C:\\work'],
    exists: (f) => f === 'C:\\work\\.venv\\Scripts\\mylab.exe',
  });
  assert.deepEqual(cmds(list), ['mylab', 'C:\\work\\.venv\\Scripts\\mylab.exe', 'py -m mylab', 'python -m mylab']);
});

test('duplicates are dropped', () => {
  const list = exe.candidateCommands({ configured: 'python3 -m mylab', pythonPath: 'python3', platform: 'linux' });
  assert.deepEqual(cmds(list), ['python3 -m mylab', 'mylab', 'python -m mylab']);
});

test('resolveExecutable returns the first candidate that answers', async () => {
  const list = exe.candidateCommands({ configured: 'mylab', pythonPath: '/env/bin/python', platform: 'linux' });
  const tried = [];
  const found = await exe.resolveExecutable(list, async (c) => {
    tried.push(c.command);
    if (c.command === 'mylab') throw new Error('ENOENT');
    return c.command === '/env/bin/python' ? 'mylab 1.0.0\n' : null;
  });
  assert.deepEqual(tried, ['mylab', '/env/bin/python']);
  assert.equal(found.command, '/env/bin/python');
  assert.deepEqual(found.args, ['-m', 'mylab']);
  assert.equal(found.version, 'mylab 1.0.0');
});

test('resolveExecutable returns null when nothing works', async () => {
  const list = exe.candidateCommands({ configured: 'mylab', platform: 'linux' });
  assert.equal(await exe.resolveExecutable(list, async () => null), null);
});

test('looksLikeMylab and describeCommand', () => {
  assert.equal(exe.looksLikeMylab('mylab 1.0.0\n'), true);
  assert.equal(exe.looksLikeMylab('Python 3.12.1'), false);
  assert.equal(exe.describeCommand({ command: '/a b/python', args: ['-m', 'mylab'] }, ['--check', 'f.m']),
    '"/a b/python" -m mylab --check f.m');
  assert.match(exe.INSTALL_HINT, /pip install mylab-lang/);
});
