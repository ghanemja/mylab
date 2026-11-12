'use strict';
// Hover text, completion entries and call signatures, built from the bundled
// function reference (data/functions.json). No dependency on VS Code.

const { codeOnly, commentStart } = require('./text');

const KEYWORDS = [
  'if', 'elseif', 'else', 'end', 'for', 'parfor', 'while', 'switch', 'case', 'otherwise',
  'try', 'catch', 'break', 'continue', 'return', 'function', 'global', 'persistent',
  'classdef', 'properties', 'methods', 'events', 'enumeration',
];

/** The first sentence of a description (the same rule as mylab.help.summary). */
function summary(description) {
  return String(description || '').split('. ')[0].replace(/\.+$/, '') + '.';
}

/** Markdown for the hover of a built-in function. */
function hoverMarkdown(name, entry) {
  const usage = (entry.usage || []).join('\n');
  const parts = [];
  if (usage) parts.push('```matlab\n' + usage + '\n```');
  parts.push(escapeMarkdown(entry.description || ''));
  parts.push(`*${escapeMarkdown(name)} — mylab built-in function*`);
  return parts.join('\n\n');
}

/**
 * Markdown for the hover of a function that mylab found on disk, from the
 * text printed by `mylab --help-for NAME`. Returns null for "not found".
 */
function hoverFromHelpText(name, text) {
  const body = String(text || '').replace(/\s+$/, '');
  if (!body || new RegExp('^' + escapeRegExp(name) + ' not found\\.?$').test(body.trim())) return null;
  return '```text\n' + body.replace(/```/g, "'''") + '\n```\n\n*' + escapeMarkdown(name) + ' — function in this folder*';
}

/** Plain text for the help view, in the layout `mylab --help-for` prints. */
function helpPlainText(name, entry) {
  const usage = (entry.usage || []).map((u) => '  ' + u).join('\n');
  return `${name}\n\n${usage}\n\n${wrap(entry.description || '', 76, '  ')}\n`;
}

function wrap(text, width, indent) {
  return text
    .split('\n')
    .map((para) => {
      const lines = [];
      let cur = indent;
      for (const word of para.split(/\s+/).filter(Boolean)) {
        if (cur.length > indent.length && cur.length + 1 + word.length > width) {
          lines.push(cur);
          cur = indent + word;
        } else {
          cur += (cur.length > indent.length ? ' ' : '') + word;
        }
      }
      if (cur.trim()) lines.push(cur);
      return lines.join('\n');
    })
    .join('\n');
}

function escapeMarkdown(text) {
  return String(text).replace(/[\\`*_{}[\]<>#|~]/g, '\\$&');
}

function escapeRegExp(text) {
  return String(text).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Completion entries for every built-in function and keyword:
 * [{label, kind: 'function' | 'keyword', detail, documentation}]
 */
function completionEntries(functions) {
  const out = [];
  for (const name of Object.keys(functions)) {
    const entry = functions[name];
    out.push({
      label: name,
      kind: 'function',
      detail: (entry.usage && entry.usage[0]) || name,
      documentation: summary(entry.description),
    });
  }
  for (const kw of KEYWORDS) out.push({ label: kw, kind: 'keyword', detail: 'keyword', documentation: '' });
  return out;
}

/**
 * Read one calling form. "[q, r] = deconv(b, a)" for deconv gives
 * {label: '[q, r] = deconv(b, a)', parameters: [{label:'b',start,end}, {label:'a',start,end}]}
 * where start/end are character positions in the label. Forms that are not
 * calls with brackets ("hold on") give null.
 */
function parseUsage(usage, name) {
  const m = new RegExp('(?<![\\w.])' + escapeRegExp(name) + '\\s*\\(').exec(usage);
  if (!m) return null;
  const open = m.index + m[0].length - 1;
  const parameters = [];
  let depth = 0;
  let start = open + 1;
  let i = open;
  for (; i < usage.length; i++) {
    const c = usage[i];
    if (c === "'" || c === '"') {
      const close = usage.indexOf(c, i + 1);
      if (close > i) i = close;
      continue;
    }
    if (c === '(' || c === '[' || c === '{') depth += 1;
    else if (c === ')' || c === ']' || c === '}') {
      depth -= 1;
      if (depth === 0) break;
    } else if (c === ',' && depth === 1) {
      pushParameter(usage, start, i, parameters);
      start = i + 1;
    }
  }
  if (depth !== 0) return null;
  pushParameter(usage, start, i, parameters);
  return { label: usage, parameters };
}

function pushParameter(usage, start, end, parameters) {
  const raw = usage.slice(start, end);
  const text = raw.trim();
  if (!text) return;
  const s = start + raw.indexOf(text);
  parameters.push({ label: text, start: s, end: s + text.length });
}

/** All calling forms of a function that can be shown as signatures. */
function signatures(name, entry) {
  return (entry.usage || []).map((u) => parseUsage(u, name)).filter(Boolean);
}

/**
 * Which call is the cursor in? `before` is the text of the line up to the
 * cursor. Returns {name, argIndex} for the innermost unclosed name(...), or
 * null. Text in strings and comments is ignored.
 */
function callContext(before) {
  const raw = String(before);
  if (commentStart(raw) >= 0) return null; // typing a comment
  const code = codeOnly(raw);
  const stack = [];
  for (let i = 0; i < code.length; i++) {
    const c = code[i];
    if (c === '(' || c === '[' || c === '{') {
      let name = null;
      if (c === '(') {
        const m = /([A-Za-z_][\w.]*)\s*$/.exec(code.slice(0, i));
        if (m && !/^\.|\.$/.test(m[1])) name = m[1];
      }
      stack.push({ bracket: c, name, commas: 0 });
    } else if (c === ')' || c === ']' || c === '}') {
      stack.pop();
    } else if (c === ',' && stack.length) {
      stack[stack.length - 1].commas += 1;
    }
  }
  for (let k = stack.length - 1; k >= 0; k--) {
    if (stack[k].bracket === '(' && stack[k].name) {
      // Commas inside [..] or {..} nested in the call belong to those.
      return { name: stack[k].name, argIndex: stack[k].commas };
    }
    if (stack[k].bracket === '(') return null; // (a, b) without a name: not a call we know
  }
  return null;
}

/**
 * Pick the calling form to highlight: the first one with enough parameters
 * for the argument being typed (or with a "..." parameter), else the longest.
 */
function activeSignature(sigs, argIndex) {
  for (let i = 0; i < sigs.length; i++) {
    const ps = sigs[i].parameters;
    if (ps.length > argIndex) return { index: i, parameter: argIndex };
    const last = ps[ps.length - 1];
    if (last && last.label.includes('...') && ps.length) return { index: i, parameter: ps.length - 1 };
  }
  let best = 0;
  sigs.forEach((s, i) => {
    if (s.parameters.length > sigs[best].parameters.length) best = i;
  });
  return { index: best, parameter: Math.max(0, Math.min(argIndex, sigs[best].parameters.length - 1)) };
}

module.exports = {
  KEYWORDS,
  summary,
  hoverMarkdown,
  hoverFromHelpText,
  helpPlainText,
  completionEntries,
  parseUsage,
  signatures,
  callContext,
  activeSignature,
};
