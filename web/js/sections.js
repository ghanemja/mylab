// Code sections: a line that starts with "%%" begins a new section.

const HEADER = /^\s*%%(?:\s+(.*))?$/;

function firstCodeLine(lines) {
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i].trim();
    if (t && !t.startsWith("%") && !t.startsWith("#")) return i;
  }
  return -1;
}

/**
 * Split code into sections.
 * Returns {
 *   hasSections,                   true when the file has at least one "%%" line
 *   sections: [{ title, start, end, header }],   0-based line numbers, end exclusive;
 *                                  header is true when line `start` is a "%%" line
 *   tailStart,                     first line of the local functions at the end of a script
 *                                  (they are added to every section when it runs), or lines.length
 * }
 */
export function splitSections(code) {
  const lines = String(code).split("\n");
  let tailStart = lines.length;
  const first = firstCodeLine(lines);
  const isFunctionFile = first >= 0 && /^function\b/.test(lines[first].trim());
  if (!isFunctionFile) {
    for (let i = 0; i < lines.length; i++) {
      if (/^function\b/.test(lines[i])) {
        tailStart = i;
        break;
      }
    }
    // Comment lines directly above the first local function belong to it.
    while (tailStart < lines.length && tailStart > 0 && /^\s*%(?!%)/.test(lines[tailStart - 1])) tailStart--;
  }
  const sections = [];
  let current = null;
  for (let i = 0; i < tailStart; i++) {
    const m = HEADER.exec(lines[i]);
    if (m) {
      if (current) current.end = i;
      current = { title: (m[1] || "").trim(), start: i, end: tailStart, header: true };
      sections.push(current);
    } else if (!current) {
      if (!lines[i].trim()) continue;
      current = { title: "", start: i, end: tailStart, header: false };
      sections.push(current);
    }
  }
  const hasSections = sections.some((s) => s.header);
  if (!sections.length) sections.push({ title: "", start: 0, end: tailStart, header: false });
  return { hasSections, sections, tailStart, lines };
}

/** Index of the section that contains the 0-based line, or -1 (e.g. inside the local functions). */
export function sectionAt(split, line) {
  const { sections } = split;
  for (let i = 0; i < sections.length; i++) if (line >= sections[i].start && line < sections[i].end) return i;
  if (sections.length && line < sections[0].start) return 0;
  return -1;
}

/**
 * Pieces of one section for a report: the comment lines right under the "%%" line become
 * prose; the rest is code. codeStart is the 0-based line of the first code line.
 */
export function sectionParts(split, index) {
  const s = split.sections[index];
  const lines = split.lines.slice(s.start, s.end);
  let i = s.header ? 1 : 0;
  const prose = [];
  while (i < lines.length && /^\s*%(?!%)/.test(lines[i])) {
    prose.push(lines[i].replace(/^\s*% ?/, ""));
    i++;
  }
  // Decorative rules such as "% ------" are not prose.
  const text = prose.map((p) => (/^[-=*_\s]{4,}$/.test(p) ? "" : p)).join("\n").trim();
  let codeStart = i;
  while (codeStart < lines.length && !lines[codeStart].trim()) codeStart++;
  let codeEnd = lines.length;
  while (codeEnd > codeStart && !lines[codeEnd - 1].trim()) codeEnd--;
  return {
    title: s.title,
    prose: text,
    code: lines.slice(codeStart, codeEnd).join("\n"),
    codeStart: s.start + codeStart,
  };
}

/** Code to run for a section: its lines, at their original line numbers' offset. */
export function sectionCode(split, index) {
  const s = split.sections[index];
  return split.lines.slice(s.start, s.end).join("\n");
}

export function tailCode(split) {
  return split.tailStart < split.lines.length ? split.lines.slice(split.tailStart).join("\n") : "";
}
