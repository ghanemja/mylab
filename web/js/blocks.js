// Block structure of MATLAB-language code: which lines open and close if/for/while/function/…
// blocks. Used for auto-indent, "Format document", code folding and the outline.
// Everything here is plain text work; nothing is run.

const OPENERS = new Set(["if", "for", "parfor", "while", "switch", "try", "function", "classdef", "spmd"]);
const CLASS_PARTS = new Set(["properties", "methods", "events", "enumeration"]);
const MIDDLES = new Set(["else", "elseif", "catch"]);
const CASES = new Set(["case", "otherwise"]);
const ENDS = new Set(["end", "endif", "endfor", "endwhile", "endfunction", "endswitch", "end_try_catch"]);
export const KEYWORDS = [
  "break", "case", "catch", "classdef", "continue", "else", "elseif", "end", "for", "function", "global",
  "if", "otherwise", "parfor", "persistent", "return", "switch", "try", "while",
];

const isWordStart = (c) => /[A-Za-z_]/.test(c);
const isWord = (c) => /[A-Za-z0-9_]/.test(c);

/** Width of the leading whitespace of a line, counting a tab as `unit` columns. */
export function indentOf(line, unit = 4) {
  let n = 0;
  for (const c of line) {
    if (c === " ") n++;
    else if (c === "\t") n += unit - (n % unit);
    else break;
  }
  return n;
}

/**
 * Scan one line. `st` carries state between lines: { sq, par, cont, block }.
 * Returns { words: [{ word, col, first }], code: bool } where words are the names found
 * outside strings, comments and brackets. Updates st in place.
 */
export function scanLine(line, st) {
  const words = [];
  const trimmed = line.trim();
  if (st.block) {
    if (trimmed === "%}") st.block = false;
    return { words, code: false, comment: true };
  }
  if (trimmed === "%{") {
    st.block = true;
    return { words, code: false, comment: true };
  }
  if (!st.cont) st.par = 0;
  st.cont = false;
  let i = 0;
  let prev = ""; // previous character that is not white space
  let adjacent = false; // prev is directly before position i
  let code = false;
  const n = line.length;
  while (i < n) {
    const c = line[i];
    if (c === " " || c === "\t") {
      adjacent = false;
      i++;
      continue;
    }
    if (c === "%" || (c === "#" && !code)) break;
    if (c === "." && line.startsWith("...", i)) {
      st.cont = true;
      break;
    }
    code = true;
    if (c === '"') {
      i++;
      while (i < n) {
        if (line[i] === '"') {
          if (line[i + 1] === '"') i++;
          else break;
        }
        i++;
      }
      i++;
      prev = '"';
      adjacent = true;
      continue;
    }
    if (c === "'") {
      const transpose = adjacent && /[A-Za-z0-9_)\]}'.]/.test(prev);
      if (!transpose) {
        i++;
        while (i < n) {
          if (line[i] === "'") {
            if (line[i + 1] === "'") i++;
            else break;
          }
          i++;
        }
      }
      i++;
      prev = "'";
      adjacent = true;
      continue;
    }
    if (c === "(") st.par++;
    else if (c === ")") st.par = Math.max(0, st.par - 1);
    else if (c === "[" || c === "{") st.sq++;
    else if (c === "]" || c === "}") st.sq = Math.max(0, st.sq - 1);
    else if (/[0-9]/.test(c)) {
      while (i < n && /[0-9A-Za-z_.]/.test(line[i])) {
        if (line.startsWith("...", i)) break;
        i++;
      }
      prev = "0";
      adjacent = true;
      continue;
    } else if (isWordStart(c)) {
      let j = i + 1;
      while (j < n && isWord(line[j])) j++;
      if (st.par === 0 && st.sq === 0 && prev !== "." && prev !== "@") {
        words.push({ word: line.slice(i, j), col: i, first: words.length === 0 && line.slice(0, i).trim() === "" });
      }
      i = j;
      prev = "a";
      adjacent = true;
      continue;
    }
    prev = c;
    adjacent = true;
    i++;
  }
  return { words, code, comment: !code && trimmed !== "" };
}

function weight(block, opts) {
  if (block.kind === "switch") return 2;
  if (block.kind === "function") return opts.indentFunctions ? 1 : 0;
  return 1;
}

function pass(lines, opts, functionsHaveEnd) {
  const st = { sq: 0, par: 0, cont: false, block: false };
  const stack = [];
  const levels = new Array(lines.length).fill(null);
  const anchors = new Array(lines.length).fill(null); // for end/else/case lines: { line, kind } of their block
  const after = new Array(lines.length).fill(0);
  const ranges = [];
  let balanced = true;
  const level = () => stack.reduce((s, b) => s + weight(b, opts), 0);
  const closeFunction = (i) => {
    // Functions without "end": the function runs up to the line before the next one.
    while (stack.length) {
      const b = stack.pop();
      if (b.kind === "function") {
        let e = i - 1;
        while (e > b.line && (!lines[e].trim() || /^\s*%/.test(lines[e]))) e--;
        ranges.push({ start: b.line, end: e + 1, kind: "function", noEnd: true });
      }
    }
  };
  for (let i = 0; i < lines.length; i++) {
    const continued = st.cont || st.sq > 0;
    const wasBlock = st.block;
    const { words, code, comment } = scanLine(lines[i], st);
    if (wasBlock) {
      after[i] = level();
      continue; // inside a %{ … %} comment: left alone
    }
    if (!code && !comment) {
      levels[i] = 0; // blank line
      after[i] = level();
      continue;
    }
    const first = words.length && words[0].first && !continued ? words[0].word : "";
    if (!continued) {
      const top = stack[stack.length - 1];
      if (ENDS.has(first)) {
        levels[i] = top ? level() - weight(top, opts) : level();
        if (top) anchors[i] = { line: top.line, kind: top.kind, shift: 0 };
      } else if (MIDDLES.has(first)) {
        levels[i] = Math.max(0, level() - 1);
        if (top) anchors[i] = { line: top.line, kind: top.kind, shift: 0 };
      } else if (CASES.has(first)) {
        levels[i] = Math.max(0, level() - 1);
        if (top && top.kind === "switch") anchors[i] = { line: top.line, kind: top.kind, shift: 1 };
      } else if (first === "function" && !functionsHaveEnd) {
        levels[i] = 0;
      } else {
        levels[i] = level();
      }
    }
    for (const w of words) {
      const word = w.word;
      const inClass = stack.some((b) => b.kind === "classdef");
      if (word === "function" && !functionsHaveEnd) {
        closeFunction(i);
        stack.push({ kind: "function", line: i });
      } else if (OPENERS.has(word)) {
        stack.push({ kind: word, line: i });
      } else if (w.first && !continued && inClass && CLASS_PARTS.has(word)) {
        stack.push({ kind: word, line: i });
      } else if (w.first && !continued && word === "arguments" && stack.length && stack[stack.length - 1].kind === "function" && /^\s*arguments\s*(\(\s*\w+\s*\))?\s*(%.*)?$/.test(lines[i])) {
        stack.push({ kind: "arguments", line: i });
      } else if (ENDS.has(word)) {
        const b = stack.pop();
        if (b) ranges.push({ start: b.line, end: i, kind: b.kind });
        else balanced = false;
      }
    }
    after[i] = level();
  }
  if (functionsHaveEnd && stack.length) balanced = false;
  if (!functionsHaveEnd) closeFunction(lines.length);
  ranges.sort((a, b) => a.start - b.start || b.end - a.end);
  return { levels, anchors, after, ranges, balanced };
}

/** Does the first function in the file have an indented body? (Both styles are common.) */
function detectFunctionIndent(lines, unit) {
  const st = { sq: 0, par: 0, cont: false, block: false };
  let fnIndent = -1;
  for (let i = 0; i < lines.length; i++) {
    const continued = st.cont || st.sq > 0;
    const { words, code } = scanLine(lines[i], st);
    if (!code || continued) continue;
    const first = words.length && words[0].first ? words[0].word : "";
    if (first === "classdef") return true;
    if (fnIndent < 0) {
      if (first === "function") fnIndent = indentOf(lines[i], unit);
      continue;
    }
    if (ENDS.has(first) || first === "function") return true;
    return indentOf(lines[i], unit) > fnIndent;
  }
  return true;
}

/**
 * Analyze code. Returns {
 *   levels:  indent level per line (null: continuation or block-comment line, keep relative),
 *   after:   indent level of a new line added after each line,
 *   anchors: for end/else/case lines, the line of the block they belong to,
 *   ranges:  [{ start, end, kind }] blocks (0-based lines, `end` is the line of the "end"),
 *   functionsHaveEnd, indentFunctions }
 */
export function analyze(text, { unit = 4, indentFunctions = null } = {}) {
  const lines = Array.isArray(text) ? text : String(text).split("\n");
  const opts = { indentFunctions: indentFunctions === null ? detectFunctionIndent(lines, unit) : indentFunctions };
  let r = pass(lines, opts, true);
  let functionsHaveEnd = true;
  if (!r.balanced && lines.some((l) => /^\s*function\b/.test(l))) {
    const r2 = pass(lines, opts, false);
    if (r2.balanced || !r.balanced) {
      r = r2;
      functionsHaveEnd = false;
    }
  }
  return { ...r, lines, functionsHaveEnd, indentFunctions: opts.indentFunctions };
}

/** Re-indent code by its block structure. Only white space at line starts and ends changes. */
export function formatCode(text, { unit = 4 } = {}) {
  const lines = String(text).split("\n");
  const a = analyze(lines, { unit });
  const out = [];
  let delta = 0; // how far the current statement moved (for its continuation lines)
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (a.levels[i] === null) {
      if (!line.trim()) out.push("");
      else out.push(" ".repeat(Math.max(0, indentOf(line, unit) + delta)) + line.trim());
      continue;
    }
    if (!line.trim()) {
      out.push("");
      continue;
    }
    const want = a.levels[i] * unit;
    delta = want - indentOf(line, unit);
    out.push(" ".repeat(want) + line.trim());
  }
  return out.join("\n");
}

/** Indent (in columns) for a new line typed after `lineNo`, given the text before the cursor. */
export function indentAfter(lines, lineNo, { unit = 4 } = {}) {
  const upTo = lines.slice(0, lineNo + 1);
  const a = analyze(upTo, { unit, indentFunctions: analyze(lines, { unit }).indentFunctions });
  const cur = indentOf(lines[lineNo], unit);
  // Find the line that starts this statement (skip back over continuation lines).
  let startLine = lineNo;
  while (startLine > 0 && a.levels[startLine] === null) startLine--;
  const st = { sq: 0, par: 0, cont: false, block: false };
  for (let i = 0; i <= lineNo; i++) scanLine(upTo[i], st);
  if (st.block) return cur;
  if (st.cont || st.sq > 0) return startLine === lineNo ? cur + unit : cur;
  if (!upTo[lineNo].trim()) return cur;
  const base = indentOf(upTo[startLine], unit);
  const level = a.levels[startLine] === null ? 0 : a.levels[startLine];
  return Math.max(0, base + (a.after[lineNo] - level) * unit);
}

/**
 * Where a line that holds only end / else / elseif / catch / case / otherwise should start,
 * or null when it is not such a line or has no open block.
 */
export function dedentFor(lines, lineNo, { unit = 4 } = {}) {
  const word = lines[lineNo].trim();
  if (!ENDS.has(word) && !MIDDLES.has(word) && !CASES.has(word)) return null;
  const a = analyze(lines.slice(0, lineNo + 1), { unit, indentFunctions: true });
  const anchor = a.anchors[lineNo];
  if (!anchor) return null;
  if (anchor.kind === "function" && ENDS.has(word) && !a.functionsHaveEnd) return null;
  return indentOf(lines[anchor.line], unit) + anchor.shift * unit;
}

const FUNCTION_LINE = /^\s*function\s+(?:(\[[^\]]*\]|[A-Za-z_]\w*)\s*=\s*)?([A-Za-z_][\w.]*)\s*(\([^)]*\)?)?/;

/** Functions defined in a file: [{ name, line, signature, doc }] (0-based lines). */
export function findFunctions(text) {
  const lines = String(text).split("\n");
  const out = [];
  const st = { sq: 0, par: 0, cont: false, block: false };
  for (let i = 0; i < lines.length; i++) {
    const wasBlock = st.block;
    const { words } = scanLine(lines[i], st);
    if (wasBlock || !words.length || !words[0].first || words[0].word !== "function") continue;
    let head = lines[i];
    for (let j = i; /\.\.\.\s*(%.*)?$/.test(lines[j]) && j + 1 < lines.length; j++) head = head.replace(/\.\.\..*$/, " ") + lines[j + 1].trim();
    const m = FUNCTION_LINE.exec(head);
    if (!m) continue;
    const name = m[2].includes(".") ? m[2].split(".").pop() : m[2];
    const args = (m[3] || "").trim();
    const signature = (m[1] ? m[1].replace(/\s+/g, " ") + " = " : "") + name + (args ? (args.endsWith(")") ? args : args + ")") : "()");
    // The first comment line under the function line is its one-line description.
    let doc = "";
    let k = i + 1;
    while (k < lines.length && /\.\.\.\s*$/.test(lines[k - 1])) k++;
    if (k < lines.length && /^\s*%(?![%{])/.test(lines[k])) {
      doc = lines[k].replace(/^\s*%+\s?/, "").trim();
      const upper = new RegExp("^" + name + "\\b\\s*", "i");
      if (upper.test(doc)) doc = doc.replace(upper, "").trim();
    }
    out.push({ name, line: i, signature, doc, col: lines[i].indexOf(name, lines[i].indexOf("function") + 8) });
  }
  return out;
}

/** All comment lines under a function line (its help text). */
export function helpComments(text, line) {
  const lines = String(text).split("\n");
  const out = [];
  let k = line + 1;
  while (k < lines.length && /\.\.\.\s*$/.test(lines[k - 1])) k++;
  for (; k < lines.length && /^\s*%/.test(lines[k]); k++) out.push(lines[k].replace(/^\s*%+ ?/, ""));
  return out.join("\n");
}
