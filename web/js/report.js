// Builds the standalone HTML document for "Publish report". The result has no scripts and no
// outside links to load: styles are inline and figures are embedded as data URLs.

import { escapeHtml, renderMarkdown } from "./markdown.js";

/** Syntax-highlighted HTML for MATLAB-language code (uses CodeMirror's tokenizer when loaded). */
export function highlight(code) {
  const CM = globalThis.CodeMirror;
  if (!CM || !CM.runMode) return escapeHtml(code);
  let html = "";
  CM.runMode(code, "text/x-octave", (text, style) => {
    const t = escapeHtml(text);
    html += style ? `<span class="${style.split(/\s+/).map((s) => "t-" + s).join(" ")}">${t}</span>` : t;
  });
  return html;
}

const CSS = `
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: #f4f6f8; color: #1c2730; font: 15px/1.6 system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans", sans-serif; }
main { max-width: 820px; margin: 28px auto; background: #fff; padding: 44px 52px 36px; border: 1px solid #dde3e8; border-radius: 10px; }
header.title { border-bottom: 2px solid #0f766e; padding-bottom: 14px; margin-bottom: 8px; }
h1 { font-size: 27px; line-height: 1.25; margin: 0 0 6px; letter-spacing: -.01em; }
.meta { color: #5d6b77; font-size: 13.5px; margin: 0; }
.meta b { color: #1c2730; }
nav.toc { margin: 18px 0 4px; font-size: 14px; }
nav.toc ol { margin: 4px 0 0; padding-left: 22px; }
nav.toc a { color: #0f766e; text-decoration: none; }
section { margin-top: 30px; }
h2 { font-size: 19px; margin: 0 0 8px; padding-top: 4px; }
.prose p { margin: 0 0 10px; }
.prose ul, .prose ol { margin: 0 0 10px; padding-left: 24px; }
.prose h3, .prose h4 { margin: 12px 0 6px; font-size: 15.5px; }
code, pre { font-family: ui-monospace, "SFMono-Regular", "Cascadia Mono", "Roboto Mono", "DejaVu Sans Mono", Menlo, Consolas, monospace; }
code { background: #f1f4f6; padding: .05em .35em; border-radius: 4px; font-size: .9em; }
pre { margin: 10px 0; padding: 11px 14px; border-radius: 6px; font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
pre.code { background: #f6f8f9; border: 1px solid #dde3e8; border-left: 3px solid #0f766e; }
pre.output { background: #fff; border: 1px solid #dde3e8; color: #26333d; }
pre.md-code { background: #f6f8f9; border: 1px solid #dde3e8; }
.label { font-size: 11px; font-weight: 650; letter-spacing: .07em; text-transform: uppercase; color: #5d6b77; margin: 12px 0 -4px; }
.error { background: #fdecec; border: 1px solid #f1b9bb; border-left: 3px solid #c0262d; color: #8f1c22; padding: 9px 14px; border-radius: 6px; margin: 10px 0; font-size: 14px; }
.skipped { color: #5d6b77; font-style: italic; font-size: 14px; }
figure { margin: 14px 0; text-align: center; }
figure img { max-width: 100%; height: auto; border: 1px solid #dde3e8; border-radius: 6px; }
figcaption { color: #5d6b77; font-size: 12.5px; margin-top: 4px; }
footer { margin-top: 36px; padding-top: 12px; border-top: 1px solid #dde3e8; color: #5d6b77; font-size: 12.5px; }
.t-keyword { color: #8a3ffc; font-weight: 600; }
.t-string, .t-string-2 { color: #a1460a; }
.t-number, .t-atom { color: #0b6bcb; }
.t-comment { color: #5f7a6a; font-style: italic; }
.t-builtin { color: #0f766e; }
.t-operator { color: #44515c; }
@media (max-width: 700px) { main { margin: 0; border: 0; border-radius: 0; padding: 24px 18px; } }
@media print {
  @page { margin: 18mm 16mm; }
  body { background: #fff; font-size: 11.5pt; }
  main { margin: 0; padding: 0; border: 0; max-width: none; }
  nav.toc a { color: inherit; }
  h2 { break-after: avoid; }
  .label { break-after: avoid; }
  pre, figure, .error { break-inside: avoid; }
  pre.code, pre.output, pre.md-code, .error { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
}
`;

/**
 * report = { title, file, author, date, sections: [{ title, prose, code, result }] }
 * result = { output, error, error_line, figures: [base64 png], animations: [base64 gif] } or { skipped: true }
 */
export function buildReport(report) {
  const esc = escapeHtml;
  const titled = report.sections.filter((s) => s.title);
  let figureNo = 0;
  const parts = [];
  report.sections.forEach((s, i) => {
    const r = s.result || {};
    const out = [];
    out.push(`<section id="s${i + 1}">`);
    if (s.title) out.push(`<h2>${esc(s.title)}</h2>`);
    if (s.prose) out.push(`<div class="prose">${renderMarkdown(s.prose)}</div>`);
    if (s.code) out.push(`<pre class="code">${highlight(s.code)}</pre>`);
    if (r.skipped) {
      if (s.code) out.push(`<p class="skipped">Not run, because an earlier section ended with an error.</p>`);
    } else {
      const text = (r.output || "").replace(/\s+$/, "");
      if (text) out.push(`<div class="label">Output</div><pre class="output">${esc(text)}</pre>`);
      if (r.error) {
        const where = r.error_line ? ` (line ${r.error_line}${r.error_file ? " of " + esc(r.error_file) : ""})` : "";
        out.push(`<div class="error"><strong>Error${where}:</strong> ${esc(r.error)}</div>`);
      }
      for (const f of r.figures || []) {
        figureNo++;
        out.push(`<figure><img alt="Figure ${figureNo}" src="data:image/png;base64,${f}"><figcaption>Figure ${figureNo}</figcaption></figure>`);
      }
      for (const a of r.animations || []) {
        figureNo++;
        out.push(`<figure><img alt="Animation ${figureNo}" src="data:image/gif;base64,${a}"><figcaption>Figure ${figureNo} (animation)</figcaption></figure>`);
      }
    }
    out.push(`</section>`);
    parts.push(out.join("\n"));
  });

  const toc =
    titled.length > 2
      ? `<nav class="toc"><strong>Contents</strong><ol>${report.sections
          .map((s, i) => (s.title ? `<li><a href="#s${i + 1}">${esc(s.title)}</a></li>` : ""))
          .join("")}</ol></nav>`
      : "";
  const meta = [report.author ? `<b>${esc(report.author)}</b>` : "", esc(report.file || ""), esc(report.date || "")].filter(Boolean).join(" &nbsp;·&nbsp; ");

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="mylab">
<title>${esc(report.title)}</title>
<style>${CSS}</style>
</head>
<body>
<main>
<header class="title">
<h1>${esc(report.title)}</h1>
<p class="meta">${meta}</p>
</header>
${toc}
${parts.join("\n")}
<footer>Published with mylab, a free environment for MATLAB-language code that runs in the browser.</footer>
</main>
</body>
</html>
`;
}
