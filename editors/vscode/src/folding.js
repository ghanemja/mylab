'use strict';
// Folding ranges and outline entries: blocks (if/for/function ... end),
// %% sections and %{ ... %} comments. No dependency on VS Code.

const { codeOnly, isSectionMarker } = require('./text');

const OPENERS = new Set(['if', 'for', 'parfor', 'while', 'switch', 'try', 'function', 'classdef']);
// These open a block only inside classdef, at the start of a line.
const CLASS_OPENERS = new Set(['properties', 'methods', 'events', 'enumeration']);

/**
 * Block keywords on each line, in order: [{line, word}] where word is an
 * opener or 'end'. An `end` inside brackets is an index, not a block end.
 * Lines inside %{ ... %} are skipped.
 */
function blockTokens(lines) {
  const tokens = [];
  const comments = [];
  let commentDepth = 0;
  let commentStartLine = 0;
  let inClass = false;
  for (let i = 0; i < lines.length; i++) {
    const trimmed = lines[i].trim();
    if (trimmed === '%{') {
      if (commentDepth === 0) commentStartLine = i;
      commentDepth += 1;
      continue;
    }
    if (commentDepth > 0) {
      if (trimmed === '%}') {
        commentDepth -= 1;
        if (commentDepth === 0) comments.push({ start: commentStartLine, end: i });
      }
      continue;
    }
    const code = codeOnly(lines[i]);
    let depth = 0;
    const re = /[A-Za-z_]\w*|[()[\]{}]/g;
    let m;
    let first = true;
    while ((m = re.exec(code))) {
      const w = m[0];
      if (w === '(' || w === '[' || w === '{') depth += 1;
      else if (w === ')' || w === ']' || w === '}') depth = Math.max(0, depth - 1);
      else {
        const afterDot = m.index > 0 && code[m.index - 1] === '.';
        if (!afterDot && depth === 0) {
          if (w === 'end') tokens.push({ line: i, word: 'end' });
          else if (OPENERS.has(w)) {
            if (w === 'classdef') inClass = true;
            tokens.push({ line: i, word: w });
          } else if (inClass && first && CLASS_OPENERS.has(w) && /^\s*(\(|$)/.test(code.slice(m.index + w.length))) {
            tokens.push({ line: i, word: w });
          }
        }
        first = false;
      }
    }
  }
  return { tokens, comments };
}

function matchBlocks(tokens, lineCount, functionsNeedEnd) {
  const ranges = [];
  const stack = [];
  const close = (endLine) => {
    const open = stack.pop();
    if (open && endLine > open.line) ranges.push({ start: open.line, end: endLine, kind: 'block', word: open.word });
  };
  for (const t of tokens) {
    if (t.word === 'end') {
      if (stack.length) close(t.line);
    } else {
      if (t.word === 'function' && !functionsNeedEnd) {
        // Functions are not closed with `end` in this file: a new function ends the previous one.
        while (stack.length) close(t.line - 1);
      }
      stack.push(t);
    }
  }
  const unclosed = stack.slice();
  while (stack.length) close(lineCount - 1);
  return { ranges, unclosedFunction: unclosed.some((t) => t.word === 'function') };
}

/**
 * Folding ranges for a file: [{start, end, kind}] with 0-based, inclusive
 * lines. kind is 'block', 'section' or 'comment'.
 */
function foldingRanges(lines) {
  const { tokens, comments } = blockTokens(lines);
  let matched = matchBlocks(tokens, lines.length, true);
  if (matched.unclosedFunction) matched = matchBlocks(tokens, lines.length, false);
  const out = matched.ranges.map(({ start, end }) => ({ start, end: trimBlank(lines, start, end), kind: 'block' }));

  for (const c of comments) out.push({ start: c.start, end: c.end, kind: 'comment' });

  const inComment = (i) => comments.some((c) => i > c.start && i < c.end);
  const marks = [];
  for (let i = 0; i < lines.length; i++) {
    if (isSectionMarker(lines[i]) && !inComment(i)) marks.push(i);
  }
  marks.forEach((start, k) => {
    const next = k + 1 < marks.length ? marks[k + 1] : lines.length;
    const end = trimBlank(lines, start, next - 1);
    if (end > start) out.push({ start, end, kind: 'section' });
  });
  return out.filter((r) => r.end > r.start).sort((a, b) => a.start - b.start || b.end - a.end);
}

function trimBlank(lines, start, end) {
  let e = end;
  while (e > start && lines[e].trim() === '') e -= 1;
  return e;
}

/**
 * Outline entries: functions, classes and %% sections.
 * [{name, kind: 'function' | 'class' | 'section', start, end, nameColumn}]
 */
function outline(lines) {
  const { tokens } = blockTokens(lines);
  let matched = matchBlocks(tokens, lines.length, true);
  if (matched.unclosedFunction) matched = matchBlocks(tokens, lines.length, false);
  const out = [];
  for (const r of matched.ranges) {
    if (r.word !== 'function' && r.word !== 'classdef') continue;
    const code = codeOnly(lines[r.start]);
    let name = null;
    if (r.word === 'function') {
      const m = /\bfunction\b\s*(?:[^=]*=\s*)?([A-Za-z_][\w.]*)/.exec(code);
      if (m) name = { text: m[1], column: m.index + m[0].length - m[1].length };
    } else {
      const m = /\bclassdef\b\s*(?:\([^)]*\)\s*)?([A-Za-z_]\w*)/.exec(code);
      if (m) name = { text: m[1], column: m.index + m[0].length - m[1].length };
    }
    if (name) {
      out.push({
        name: name.text,
        kind: r.word === 'function' ? 'function' : 'class',
        start: r.start,
        end: trimBlank(lines, r.start, r.end),
        nameColumn: name.column,
      });
    }
  }
  const marks = [];
  for (let i = 0; i < lines.length; i++) if (isSectionMarker(lines[i])) marks.push(i);
  marks.forEach((start, k) => {
    const title = lines[start].replace(/^\s*%%\s*/, '').trim() || '(untitled section)';
    const next = k + 1 < marks.length ? marks[k + 1] : lines.length;
    out.push({ name: title, kind: 'section', start, end: trimBlank(lines, start, next - 1), nameColumn: 0 });
  });
  return out.sort((a, b) => a.start - b.start || b.end - a.end);
}

module.exports = { foldingRanges, outline, blockTokens };
