// Packing code into links: JSON -> deflate (when the browser can) -> base64url.
// A payload starts with "1" (deflate-raw compressed) or "0" (plain UTF-8).

export function bytesToBase64(bytes) {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

export function base64ToBytes(b64) {
  const s = atob(b64);
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}

const toUrl = (b64) => b64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
const fromUrl = (s) => s.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (s.length % 4)) % 4);

async function pipe(bytes, stream) {
  const out = new Blob([bytes]).stream().pipeThrough(stream);
  return new Uint8Array(await new Response(out).arrayBuffer());
}

export async function encodePayload(value) {
  const raw = new TextEncoder().encode(JSON.stringify(value));
  if (typeof CompressionStream !== "undefined") {
    try {
      const packed = await pipe(raw, new CompressionStream("deflate-raw"));
      if (packed.length < raw.length) return "1" + toUrl(bytesToBase64(packed));
    } catch {
      /* fall through to the uncompressed form */
    }
  }
  return "0" + toUrl(bytesToBase64(raw));
}

export async function decodePayload(text) {
  const kind = text[0];
  let bytes;
  try {
    bytes = base64ToBytes(fromUrl(text.slice(1)));
  } catch {
    throw new Error("The link is damaged (it may have been cut off when it was copied).");
  }
  if (kind === "1") {
    if (typeof DecompressionStream === "undefined") throw new Error("This browser is too old to open compressed links. Please update it.");
    try {
      bytes = await pipe(bytes, new DecompressionStream("deflate-raw"));
    } catch {
      throw new Error("The link is damaged (it may have been cut off when it was copied).");
    }
  } else if (kind !== "0") {
    throw new Error("This link was made by a newer version of mylab.");
  }
  try {
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    throw new Error("The link is damaged (it may have been cut off when it was copied).");
  }
}

/** "#a=1&b=two" -> { a: "1", b: "two" } */
export function parseHash(hash) {
  const out = {};
  for (const part of String(hash || "").replace(/^#/, "").split("&")) {
    if (!part) continue;
    const i = part.indexOf("=");
    const key = i < 0 ? part : part.slice(0, i);
    let value = i < 0 ? "" : part.slice(i + 1);
    try {
      value = decodeURIComponent(value);
    } catch {
      /* keep the raw text */
    }
    out[key] = value;
  }
  return out;
}

/** Text if the bytes are valid UTF-8 without NUL characters, otherwise null. */
export function bytesToText(bytes) {
  try {
    const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    return text.includes("\u0000") ? null : text;
  } catch {
    return null;
  }
}

const TEXT_EXT = /\.(m|txt|csv|tsv|dat|json|md|xml|html|svg|log|tex|yaml|yml|ini|py)$/i;
const IMAGE_EXT = /\.(png|jpe?g|gif|bmp|webp|svg)$/i;

export const isTextName = (name) => TEXT_EXT.test(name);
export const isImageName = (name) => IMAGE_EXT.test(name);

export function mimeOf(name) {
  const ext = (/\.([A-Za-z0-9]+)$/.exec(name) || [, ""])[1].toLowerCase();
  return (
    { png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", gif: "image/gif", bmp: "image/bmp", webp: "image/webp", svg: "image/svg+xml",
      csv: "text/csv", json: "application/json", html: "text/html", zip: "application/zip" }[ext] ||
    (isTextName(name) ? "text/plain" : "application/octet-stream")
  );
}

/** File content as stored (string for text that decodes cleanly, Uint8Array otherwise). */
export function toStored(name, bytes) {
  if (/\.(png|jpe?g|gif|bmp|webp|mat|xlsx?|zip|pdf)$/i.test(name)) return bytes;
  const text = bytesToText(bytes);
  return text === null ? bytes : text;
}

export function formatSize(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(n < 10240 ? 1 : 0)} kB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
