'use strict';
// Pretends to be the mylab command for tests, so they need no Python.
//   node fake-mylab.js --version | --check FILE | --help-for NAME | [--debug] FILE
const fs = require('node:fs');
const path = require('node:path');

const args = process.argv.slice(2);
if (args[0] === '--version') {
  console.log('mylab 9.9.9');
} else if (args[0] === '--check') {
  const text = fs.readFileSync(args[1], 'utf8');
  const problems = [];
  text.split('\n').forEach((line, i) => {
    const col = line.indexOf('BAD');
    if (col >= 0) problems.push({ line: i + 1, column: col + 1, severity: 'error', message: "unexpected 'BAD'" });
    const m = /\b(unknown\w*)\(/.exec(line);
    // Like the real checker: a NAME.m next to the file makes NAME known.
    if (m && !fs.existsSync(path.join(path.dirname(args[1]), `${m[1]}.m`))) {
      problems.push({ line: i + 1, column: 1, severity: 'warning', message: `'${m[1]}' is not a function mylab provides.` });
    }
  });
  console.log(JSON.stringify({ ok: !problems.some((p) => p.severity === 'error'), problems }));
} else if (args[0] === '--help-for') {
  if (fs.existsSync(`${args[1]}.m`)) {
    console.log(`${args[1].toUpperCase()}  Help from the file in ${path.basename(process.cwd())}.`);
  } else {
    console.log(`${args[1]} not found.`);
    process.exitCode = 1;
  }
} else {
  console.log(`ran ${args.join(' ')}`);
}
