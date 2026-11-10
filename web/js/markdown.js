// A small, safe Markdown renderer for assistant replies.
// Supports fenced code blocks, inline code, bold/italic, links, headings and lists.
// Everything is HTML-escaped first; no raw HTML from the model is ever inserted.

export function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function inline(text) {
  let s = escapeHtml(text);
  const codes = [];
  s = s.replace(/`([^`]+)`/g, (_, c) => {
    codes.push(c);
    return `\u0000${codes.length - 1}\u0000`;
  });
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[\s(])\*([^*\s][^*]*)\*(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>");
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  s = s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[Number(i)]}</code>`);
  return s;
}

/**
 * Split Markdown into blocks. Returns a list of
 *   {type:"code", lang, code, closed} or {type:"html", html}.
 * An unclosed fence (while streaming) is returned with closed=false.
 */
export function parseMarkdown(md) {
  const blocks = [];
  const lines = String(md).replace(/\r\n?/g, "\n").split("\n");
  let i = 0;
  let para = [];
  let list = null; // {ordered, items}

  const flushPara = () => {
    if (para.length) {
      blocks.push({ type: "html", html: `<p>${para.map(inline).join("<br>")}</p>` });
      para = [];
    }
  };
  const flushList = () => {
    if (list) {
      const tag = list.ordered ? "ol" : "ul";
      blocks.push({ type: "html", html: `<${tag}>${list.items.map((t) => `<li>${inline(t)}</li>`).join("")}</${tag}>` });
      list = null;
    }
  };

  while (i < lines.length) {
    const line = lines[i];
    const fence = line.match(/^\s*(```+|~~~+)\s*([\w+#.-]*)\s*$/);
    if (fence) {
      flushPara();
      flushList();
      const marker = fence[1];
      const lang = (fence[2] || "").toLowerCase();
      const body = [];
      i++;
      let closed = false;
      while (i < lines.length) {
        if (lines[i].trim().startsWith(marker[0].repeat(3)) && lines[i].trim().replace(/[`~]/g, "") === "") {
          closed = true;
          i++;
          break;
        }
        body.push(lines[i]);
        i++;
      }
      blocks.push({ type: "code", lang, code: body.join("\n"), closed });
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      flushPara();
      flushList();
      const level = Math.min(4, heading[1].length + 2);
      blocks.push({ type: "html", html: `<h${level}>${inline(heading[2])}</h${level}>` });
      i++;
      continue;
    }
    const li = line.match(/^\s*([-*+]|\d+[.)])\s+(.*)$/);
    if (li) {
      flushPara();
      const ordered = /\d/.test(li[1]);
      if (!list || list.ordered !== ordered) {
        flushList();
        list = { ordered, items: [] };
      }
      list.items.push(li[2]);
      i++;
      continue;
    }
    if (list && /^\s{2,}\S/.test(line)) {
      list.items[list.items.length - 1] += " " + line.trim();
      i++;
      continue;
    }
    if (!line.trim()) {
      flushPara();
      flushList();
      i++;
      continue;
    }
    flushList();
    para.push(line);
    i++;
  }
  flushPara();
  flushList();
  return blocks;
}
