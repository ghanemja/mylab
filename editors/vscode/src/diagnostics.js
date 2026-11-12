'use strict';
// Turn the JSON printed by `mylab --check FILE` into editor diagnostics.
// No dependency on VS Code: ranges are plain numbers, counted from 0.

const { codeOnly, splitLines } = require('./text');

/**
 * Parse the output of `mylab --check`. Returns {ok, problems} or null when
 * the text is not the expected JSON (for example an older mylab without
 * --check, which prints a usage message instead).
 */
function parseCheckOutput(stdout) {
  const text = String(stdout || '').trim();
  if (!text) return null;
  // Tolerate stray lines printed before the JSON (warnings from Python, say).
  const start = text.lastIndexOf('\n{') >= 0 ? text.lastIndexOf('\n{') + 1 : text.indexOf('{');
  if (start < 0) return null;
  let data;
  try {
    data = JSON.parse(text.slice(start));
  } catch (e) {
    return null;
  }
  if (!data || !Array.isArray(data.problems)) return null;
  const problems = data.problems
    .filter((p) => p && typeof p.message === 'string')
    .map((p) => ({
      line: Number.isFinite(p.line) ? p.line : 1,
      column: Number.isFinite(p.column) ? p.column : 1,
      severity: p.severity === 'warning' ? 'warning' : 'error',
      message: p.message,
    }));
  return { ok: data.ok !== false && !problems.some((p) => p.severity === 'error'), problems };
}

/**
 * Map problems (1-based line and column) to ranges in `text`.
 * @returns {{startLine:number,startColumn:number,endLine:number,endColumn:number,
 *            severity:'error'|'warning',message:string}[]}
 */
function toDiagnostics(problems, text) {
  const lines = splitLines(text);
  return problems.map((p) => {
    const lineNo = clamp(p.line - 1, 0, lines.length - 1);
    const line = lines[lineNo] || '';
    const [start, end] = rangeInLine(line, p);
    return {
      startLine: lineNo,
      startColumn: start,
      endLine: lineNo,
      endColumn: end,
      severity: p.severity,
      message: tidy(p.message),
    };
  });
}

function rangeInLine(line, p) {
  const quoted = /'([^']+)'/.exec(p.message);
  const col = clamp(p.column - 1, 0, line.length);

  // Warnings about an unknown function give the line only: underline the name.
  if (quoted && /^[A-Za-z_]\w*$/.test(quoted[1]) && (p.column <= 1 || p.severity === 'warning')) {
    const m = new RegExp('(?<![\\w.])' + quoted[1] + '(?!\\w)').exec(codeOnly(line));
    if (m) return [m.index, m.index + quoted[1].length];
  }
  // The message names the text it stopped at, and that text is at the column.
  if (quoted && line.startsWith(quoted[1], col)) return [col, col + quoted[1].length];
  // A name or number under the column: underline all of it.
  if (col < line.length && /\w/.test(line[col])) {
    let s = col;
    let e = col;
    while (s > 0 && /\w/.test(line[s - 1])) s -= 1;
    while (e < line.length && /\w/.test(line[e])) e += 1;
    return [s, e];
  }
  if (col < line.length && line[col] !== ' ' && line[col] !== '\t') return [col, col + 1];
  // At or past the end of the line (or on a blank): underline the line's text.
  const first = line.search(/\S/);
  if (first < 0) return [0, Math.max(line.length, 1)];
  return [first, line.trimEnd().length];
}

function tidy(message) {
  const s = message.trim();
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

module.exports = { parseCheckOutput, toDiagnostics };
