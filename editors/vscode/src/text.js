'use strict';
// Small text helpers shared by the other modules. No dependency on VS Code.

/**
 * Replace comments and the contents of strings with spaces, keeping every
 * other character at the same column. Lets callers search a line for
 * keywords, names and brackets without being fooled by text inside quotes.
 *
 * A single quote starts a string unless it follows a name, a number, a
 * closing bracket, a dot or another quote; there it means transpose.
 */
function codeOnly(line) {
  let out = '';
  let i = 0;
  const n = line.length;
  while (i < n) {
    const c = line[i];
    if (c === '%') break;
    if (c === '.' && line.startsWith('...', i)) break;
    if (c === '"' || (c === "'" && !isTransposeAt(line, i))) {
      const end = stringEnd(line, i);
      out += c + ' '.repeat(Math.max(0, end - i - 2)) + (end - i >= 2 ? c : '');
      i = end;
      continue;
    }
    out += c;
    i += 1;
  }
  return out + ' '.repeat(n - out.length);
}

/** Column where a comment (% or ...) starts on the line, or -1. Quotes are respected. */
function commentStart(line) {
  let i = 0;
  while (i < line.length) {
    const c = line[i];
    if (c === '%' || (c === '.' && line.startsWith('...', i))) return i;
    if (c === '"' || (c === "'" && !isTransposeAt(line, i))) i = stringEnd(line, i);
    else i += 1;
  }
  return -1;
}

function isTransposeAt(line, i) {
  if (i === 0) return false;
  return /[\w)\]}.']/.test(line[i - 1]);
}

/** Index just past the string that opens at `start` (a doubled quote is an escaped quote). */
function stringEnd(line, start) {
  const q = line[start];
  let i = start + 1;
  while (i < line.length) {
    if (line[i] === q) {
      if (line[i + 1] === q) {
        i += 2;
        continue;
      }
      return i + 1;
    }
    i += 1;
  }
  return line.length;
}

/** True for a section heading: a line that starts with %% followed by a space or nothing. */
function isSectionMarker(line) {
  return /^\s*%%(?:\s|$)/.test(line);
}

function splitLines(text) {
  return text.split(/\r\n|\r|\n/);
}

/**
 * The name under a column: {word, start, end} or null. Dotted names such as
 * containers.Map are returned whole when `knownDotted` says they exist.
 */
function wordAt(line, column, knownDotted) {
  const re = /[A-Za-z_][A-Za-z0-9_]*/g;
  let m;
  while ((m = re.exec(line))) {
    const start = m.index;
    const end = start + m[0].length;
    if (column < start) break;
    if (column > end) continue;
    if (knownDotted) {
      const dotted = /^[A-Za-z_][A-Za-z0-9_.]*/.exec(line.slice(leftDotted(line, start)));
      if (dotted) {
        const ds = leftDotted(line, start);
        let name = dotted[0].replace(/\.+$/, '');
        while (name.includes('.')) {
          if (knownDotted(name) && column <= ds + name.length) return { word: name, start: ds, end: ds + name.length };
          name = name.slice(0, name.lastIndexOf('.'));
        }
      }
    }
    return { word: m[0], start, end };
  }
  return null;
}

function leftDotted(line, start) {
  let s = start;
  while (s > 0 && /[A-Za-z0-9_.]/.test(line[s - 1])) s -= 1;
  while (s < start && !/[A-Za-z_]/.test(line[s])) s += 1;
  return s;
}

module.exports = { codeOnly, commentStart, isTransposeAt, stringEnd, isSectionMarker, splitLines, wordAt };
