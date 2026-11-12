'use strict';
// Sections ("cells"): parts of a file separated by lines that start with %%.

const { isSectionMarker, splitLines } = require('./text');

/**
 * Split a file into sections. Lines are numbered from 0 and both ends are
 * inclusive. A file with no %% line is one section. Text before the first
 * %% line is a section of its own (unless it is empty).
 * @returns {{startLine: number, endLine: number, title: string}[]}
 */
function splitSections(text) {
  const lines = splitLines(text);
  const starts = [];
  for (let i = 0; i < lines.length; i++) {
    if (isSectionMarker(lines[i])) starts.push(i);
  }
  if (starts.length === 0 || starts[0] !== 0) {
    const firstEnd = starts.length ? starts[0] : lines.length;
    if (starts.length === 0 || lines.slice(0, firstEnd).some((l) => l.trim() !== '')) starts.unshift(0);
  }
  return starts.map((start, k) => {
    const end = (k + 1 < starts.length ? starts[k + 1] : lines.length) - 1;
    const title = isSectionMarker(lines[start]) ? lines[start].replace(/^\s*%%\s*/, '').trim() : '';
    return { startLine: start, endLine: Math.max(start, end), title };
  });
}

/** The section that contains `line` (0-based). */
function sectionAt(text, line) {
  const sections = splitSections(text);
  let found = sections[0];
  for (const s of sections) {
    if (s.startLine <= line) found = s;
  }
  return found;
}

/**
 * The code to run for "Run Selection / Current Section".
 *
 * `selection` is {startLine, startColumn, endLine, endColumn} (0-based) or
 * null. With a non-empty selection, that text runs; otherwise the section
 * around the cursor runs.
 *
 * The result is padded with blank lines (and spaces) in front, so that line
 * numbers in error messages match the original file.
 * @returns {{code: string, startLine: number, endLine: number, kind: 'selection' | 'section'}}
 */
function codeToRun(text, selection, cursorLine) {
  const lines = splitLines(text);
  const empty = !selection ||
    (selection.startLine === selection.endLine && selection.startColumn === selection.endColumn);
  if (!empty) {
    const picked = lines.slice(selection.startLine, selection.endLine + 1);
    const last = picked.length - 1;
    picked[last] = picked[last].slice(0, selection.endColumn);
    picked[0] = ' '.repeat(selection.startColumn) + picked[0].slice(selection.startColumn);
    // A selection that ends at the start of a line does not include that line.
    let endLine = selection.endLine;
    if (picked.length > 1 && selection.endColumn === 0) {
      picked.pop();
      endLine -= 1;
    }
    return {
      code: '\n'.repeat(selection.startLine) + picked.join('\n') + '\n',
      startLine: selection.startLine,
      endLine,
      kind: 'selection',
    };
  }
  const line = cursorLine === undefined ? (selection ? selection.startLine : 0) : cursorLine;
  const s = sectionAt(text, line);
  let endLine = s.endLine;
  while (endLine > s.startLine && lines[endLine].trim() === '') endLine -= 1; // drop trailing blank lines
  return {
    code: '\n'.repeat(s.startLine) + lines.slice(s.startLine, endLine + 1).join('\n') + '\n',
    startLine: s.startLine,
    endLine,
    kind: 'section',
  };
}

module.exports = { splitSections, sectionAt, codeToRun };
