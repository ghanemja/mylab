// Live collaboration: several people work on one project at the same time, browser to browser.
//
// Everyone in a session holds the same Yjs document: the project's files (a Y.Text per text file, the
// bytes of other files), the session's name and host, and the command windows of the people who
// share their session. Who is there, with their colour, file and cursor, travels as y-protocols
// awareness. y-webrtc connects the browsers directly (WebRTC); a signalling server only introduces
// them. The link carries a random session id and a 256-bit key after the "#", which browsers never
// send to a server:
//
//   #live=<session>.<key>[&s=<signalling servers, when not the default>]
//
// The signalling topic is a hash of both, so the server learns neither. y-webrtc encrypts every
// signalling message (and the messages between tabs of one browser) with AES-GCM, using a key derived
// from the link's key; the direct connections are encrypted by WebRTC itself (DTLS), and their
// fingerprints travel inside those encrypted messages. See docs/WEB.md, "Privacy".
//
// Everything received is untrusted: paths are checked like imported ones, sizes are capped, names and
// output are shown as text and figures only as PNG images. Code from the session never runs by itself;
// each person runs it with Run, in their own engine.

import { $, el, toast, copyText, promptDialog, confirmDialog } from "./ui.js";
import { formatSize } from "./codec.js";
import { chunkedPeerClass } from "./live-peer.js";
import { DocBinding } from "./live-binding.js";

const VENDOR_URL = new URL("../vendor/live-collab.mjs", import.meta.url).href;
export const DEFAULT_SIGNALING = ["wss://y-webrtc-eu.fly.dev"];
export const DEFAULT_ICE = [{ urls: ["stun:stun.l.google.com:19302", "stun:stun.cloudflare.com:3478"] }];
// White text has a contrast of at least 5.9:1 on each.
const COLORS = ["#b42318", "#1d5fb8", "#24702a", "#7a3fb0", "#9a4a00", "#00706a", "#ad2c66", "#5b4a3d"];
const LIMIT = {
  text: 1024 * 1024, // characters in a text file, when it is shared
  textReceived: 4 * 1024 * 1024, // a shared text may grow while people type; beyond this it is not taken
  binary: 2 * 1024 * 1024,
  total: 16 * 1024 * 1024,
  files: 500,
};
const ALONE_MS = 3000; // connected to the signalling server this long and no one else around: the room is empty
const NOBODY_MS = 15000;
const BLOCKED_MS = 15000;
const SIGNAL_MS = 10000;
const HOST_KEY = "mylab.liveHost."; // sessionStorage: which session this tab started (no key in it)

/* ------------------------------------------------------------------ helpers */

function b64url(bytes) {
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

const token = (n) => b64url(crypto.getRandomValues(new Uint8Array(n)));

/** The signalling topic: a hash of the session id and the key, so a signalling server learns neither. */
async function topicFor(room, key) {
  const h = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(`mylab live topic\n${room}\n${key}`));
  return "mylab-" + [...new Uint8Array(h)].slice(0, 16).map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** "<session>.<key>" from a link, or null. */
export function parseLiveLink(value) {
  const m = /^([A-Za-z0-9_-]{16})\.([A-Za-z0-9_-]{43})$/.exec(String(value || ""));
  return m ? { room: m[1], key: m[2] } : null;
}

/** A display name as text: no control or direction-changing characters, at most 40 characters. */
export function cleanName(s) {
  return String(s ?? "")
    .replace(/[\u0000-\u001f\u007f-\u009f\u00ad\u061c\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff]/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 40);
}

/** A path from the session if it is a safe project-relative file name (the rules of New file and import), else null. */
export function safePath(p) {
  if (typeof p !== "string" || !p || p.length > 160) return null;
  if (/[\u0000-\u001f\u007f\\:*?"<>|]/.test(p)) return null;
  const parts = p.split("/");
  if (parts.some((s) => !s || s === "." || s === ".." || s.startsWith(".") || s.length > 80 || !s.trim())) return null;
  return p;
}

const clip = (s, n) => {
  s = String(s ?? "");
  return s.length > n ? s.slice(0, n) + "…" : s;
};
const isPng = (b64) => typeof b64 === "string" && b64.length <= 2800000 && /^iVBORw0KGgo[A-Za-z0-9+/]*={0,2}$/.test(b64);
const sameBytes = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
const sameContent = (a, b) => a !== undefined && (typeof a === "string" ? a === b : typeof b !== "string" && sameBytes(a, b));
const initials = (name) => name.split(" ").filter(Boolean).slice(0, 2).map((w) => [...w][0]).join("").toUpperCase() || "?";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const listText = (names) => (names.length <= 2 ? names.join(" and ") : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`);

/* ------------------------------------------------------------------ module */

export function initLive(app) {
  const { state, prefs, store } = app;
  let lib = null; // the bundle, once loaded
  let S = null; // the session this tab is in
  let followId = null; // the person whose session is shown in the Live tab
  let unfollowed = false; // "Nobody" was picked: don't follow the next person who shares
  let renderQueued = false;
  let cursorTimer = null;
  const pending = { joined: [], left: [], timer: null };

  /* ---------- settings: signalling and STUN/TURN servers ---------- */

  const params = new URLSearchParams(location.search);
  const listParam = (name) => (params.get(name) || "").split(",").map((s) => s.trim()).filter(Boolean);
  const isSignalingUrl = (u) => typeof u === "string" && u.length < 300 && /^wss?:\/\/[^\s/?#]+[^\s]*$/i.test(u);
  const isIceUrl = (u) => typeof u === "string" && u.length < 300 && /^(stun|turns?):[^\s]+$/i.test(u);

  function savedSignaling() {
    const v = prefs.get("liveSignaling", []);
    return Array.isArray(v) ? v.filter(isSignalingUrl) : [];
  }

  /** Which signalling servers to use: ?signaling= (tests), then the link's, then Settings, then the default. */
  function signalingFor(fromLink = []) {
    const p = listParam("signaling").filter(isSignalingUrl);
    if (p.length) return p;
    const l = fromLink.filter(isSignalingUrl);
    if (l.length) return l;
    const s = savedSignaling();
    return s.length ? s : DEFAULT_SIGNALING;
  }

  /** Lines of "address [user name] [password]" from Settings. */
  function parseIce(text) {
    const out = [];
    for (const line of String(text || "").split("\n")) {
      const [url, username, credential] = line.trim().split(/\s+/);
      if (!url || !isIceUrl(url)) continue;
      out.push(username ? { urls: url, username, credential: credential || "" } : { urls: url });
    }
    return out;
  }

  function iceServers() {
    const p = listParam("ice").filter(isIceUrl);
    if (p.length) return [{ urls: p }];
    const s = parseIce(prefs.get("liveIce", ""));
    return s.length ? s : DEFAULT_ICE;
  }

  const myName = () => cleanName(prefs.get("liveName", "")) || "Guest";

  function setName(name) {
    name = cleanName(name);
    if (!name) return;
    prefs.set("liveName", name);
    $("set-live-name").value = name;
    $("live-name").value = name;
    if (S) {
      const st = S.awareness.getLocalState();
      if (st && st.user) S.awareness.setLocalStateField("user", { ...st.user, name });
    }
  }

  async function ensureName() {
    const saved = cleanName(prefs.get("liveName", ""));
    if (saved) return saved;
    const name = await promptDialog({
      title: "Your name in live sessions",
      text: "The others in the session see it next to your cursor and in the list of people. It is saved in this browser.",
      value: "",
      placeholder: "For example Sam",
      okLabel: "Continue",
      validate: (v) => (cleanName(v) ? { name: cleanName(v) } : { error: "Please enter a name." }),
    });
    if (name) setName(name);
    return name ? cleanName(name) : null;
  }

  /* ---------- the session ---------- */

  async function loadLib() {
    if (lib) return lib;
    if (typeof RTCPeerConnection === "undefined") {
      throw new Error("this browser can't connect directly to other browsers (WebRTC is missing or turned off).");
    }
    const mod = await import(VENDOR_URL);
    mod.setPeerClass(chunkedPeerClass(mod.SimplePeer));
    lib = mod;
    return lib;
  }

  async function openSession({ room, key, hostId, servers, seed }) {
    const L = await loadLib();
    const topic = await topicFor(room, key);
    const ydoc = new L.Y.Doc();
    const awareness = new L.awarenessProtocol.Awareness(ydoc);
    const s = {
      L, room, key, topic, ydoc, awareness, hostId, signaling: servers,
      files: ydoc.getMap("files"),
      meta: ydoc.getMap("meta"),
      shared: ydoc.getMap("sessions"),
      bindings: new Map(), // path -> DocBinding
      writes: new Map(), // path -> timer of a pending storage write
      skipped: new Map(), // path -> why this person's file is not shared
      ignored: new Map(), // path -> why a file from the session was not taken
      pendingMoves: [],
      queue: Promise.resolve(),
      projectId: null,
      ready: false, // the project and the shared files are in step
      adopting: false,
      canSeed: false,
      applying: 0,
      sharing: false,
      opened: Date.now(),
      signalSince: 0,
      peersSeen: false,
      known: new Map(), // clientId -> name, to say who left
      status: null,
    };
    S = s;
    if (seed) seed(s);
    awareness.setLocalState({ v: 1, user: { name: myName(), color: COLORS[ydoc.clientID % COLORS.length] }, file: null, cursor: null, sharing: false, host: false });
    s.files.observe((e, tr) => onFilesMap(s, e, tr));
    s.files.observeDeep((events, tr) => onFilesDeep(s, events, tr));
    s.meta.observe((e, tr) => onMeta(s, e, tr));
    s.shared.observeDeep(() => queueRender());
    awareness.on("change", (ch) => onAwareness(s, ch));
    s.provider = new L.WebrtcProvider(topic, ydoc, { signaling: servers, password: key, awareness, peerOpts: { config: { iceServers: iceServers() } } });
    s.provider.on("peers", (e) => {
      if (e.added.length || e.webrtcPeers.length || e.bcPeers.length) s.peersSeen = true;
      queueRender();
    });
    s.onHide = () => {
      if (s.sharing) s.shared.delete(String(ydoc.clientID));
    };
    window.addEventListener("pagehide", s.onHide);
    s.ticker = setInterval(() => tick(s), 1000);
    return s;
  }

  const amHost = (s) => !!s.hostId && s.meta.get("host") === s.hostId;
  const baseProjectName = () => state.project.name.replace(/^(Shared|Live): /, "");

  function fragment(s) {
    const custom = s.signaling.join(",") !== DEFAULT_SIGNALING.join(",");
    return `live=${s.room}.${s.key}` + (custom ? "&s=" + encodeURIComponent(s.signaling.join(",")) : "");
  }

  const linkFor = (s) => location.origin + location.pathname + "#" + fragment(s);

  function setHash(s) {
    // the address bar holds the link while the session runs, so a reload joins again
    history.replaceState(null, "", location.pathname + location.search + "#" + fragment(s));
  }

  /** Start a session on the current project: this person is its host. */
  async function startSession() {
    if (S) return;
    const typed = cleanName($("live-name").value);
    if (typed) setName(typed);
    const name = typed || (await ensureName());
    if (!name) return;
    $("live-start").disabled = true;
    try {
      await app.flushAll();
      const room = token(12);
      const key = token(32);
      const hostId = token(12);
      try {
        sessionStorage.setItem(HOST_KEY + room, hostId);
      } catch {
        /* the tab forgets that it is the host after a reload */
      }
      const s = await openSession({
        room, key, hostId, servers: signalingFor(),
        seed: (s) => s.ydoc.transact(() => {
          s.meta.set("v", 1);
          s.meta.set("name", baseProjectName());
          s.meta.set("host", hostId);
          seedFiles(s);
        }),
      });
      s.projectId = state.project.id;
      s.ready = true;
      await linkProject(s, true);
      bindOpenDocs(s);
      setSharing(s, true);
      updateLocalAwareness(s);
      setHash(s);
      renderAll();
      announce("Live session started. Copy the link and send it to the people you want to work with.");
      if ($("dlg-live").open) $("live-copy").focus();
    } catch (e) {
      console.warn(e);
      toast("Live collaboration couldn't start: " + (e.message || e), "error");
      if (S) await leave({ quiet: true });
    } finally {
      $("live-start").disabled = false;
    }
  }

  /** #live=… opened: join that session (asks for a name the first time). */
  async function joinFromLink(hash) {
    const link = parseLiveLink(hash.live);
    if (!link) {
      toast("This live session link is incomplete. The part after “#live=” must hold the session and its key: ask for the whole link.", "error");
      if (S) setHash(S); // the session this tab is in goes on
      else if (/^#live/.test(location.hash)) app.clearHash();
      return;
    }
    if (S && S.room === link.room && S.key === link.key) return openDialog();
    if (S) await leave({ quiet: true });
    const name = await ensureName();
    if (!name) {
      app.clearHash();
      return;
    }
    // a project of this browser that is already in this session (the host's, or a kept copy)
    const linked = state.projects.find((p) => p.meta && p.meta.live === link.room);
    if (linked && state.project.id !== linked.id) await app.switchProject(linked.id);
    let hostId = null;
    try {
      hostId = sessionStorage.getItem(HOST_KEY + link.room);
    } catch {
      /* not the host, then */
    }
    let s;
    try {
      s = await openSession({ room: link.room, key: link.key, hostId, servers: signalingFor(String(hash.s || "").split(",")) });
    } catch (e) {
      toast("Couldn't join the live session: " + (e.message || e), "error");
      app.clearHash();
      return;
    }
    s.projectId = linked && state.project.id === linked.id ? linked.id : null;
    s.canSeed = !!s.projectId;
    renderAll();
  }

  /** The files arrived from the others: put them in this person's project (a temporary one for a newcomer). */
  async function adoptRoom(s) {
    if (s.ready || s.adopting) return;
    s.adopting = true;
    try {
      if (s.meta.get("stopped")) {
        await leave({ reason: "ended", by: cleanName(s.meta.get("stoppedBy")) });
        return;
      }
      const name = cleanName(s.meta.get("name")) || "Live session";
      const files = roomFiles(s);
      await app.flushAll();
      if (s.projectId && state.projects.some((p) => p.id === s.projectId)) {
        if (state.project.id !== s.projectId) await app.switchProject(s.projectId);
        await replaceLocal(s, files);
      } else {
        const p = await app.createProject({ name: "Live: " + name, kind: "shared", temp: true, meta: { live: s.room, name }, files, open: app.pickMain(files) });
        s.projectId = p.id;
      }
      if (s !== S) return;
      s.ready = true;
      bindOpenDocs(s);
      await replaceLocal(s, roomFiles(s)); // anything that changed meanwhile
      updateLocalAwareness(s);
      const others = people(s).filter((p) => !p.me).map((p) => p.name);
      announce(`You joined the live session “${name}”.` + (others.length ? ` Here: ${listText(others)}.` : ""));
      toast(`You joined the live session “${name}”.`);
    } finally {
      s.adopting = false;
      queueRender();
    }
  }

  /** No one else is in the session and this browser has a copy: carry on with it. */
  function seedFromLocal(s) {
    if (s.ready || s.adopting || !s.projectId || state.project.id !== s.projectId) return;
    if (!s.hostId) {
      s.hostId = token(12);
      try {
        sessionStorage.setItem(HOST_KEY + s.room, s.hostId);
      } catch {
        /* see startSession */
      }
    }
    s.ydoc.transact(() => {
      if (!s.meta.get("v")) {
        s.meta.set("v", 1);
        s.meta.set("name", baseProjectName());
        s.meta.set("host", s.hostId);
      }
      seedFiles(s);
    });
    s.ready = true;
    bindOpenDocs(s);
    updateLocalAwareness(s);
    queueRender();
  }

  /** Leave the session. Everyone keeps their own copy of the files. */
  async function leave({ reason = "left", by = "", quiet = false } = {}) {
    const s = S;
    if (!s) return;
    S = null;
    clearInterval(s.ticker);
    clearTimeout(cursorTimer);
    window.removeEventListener("pagehide", s.onHide);
    if (reason === "ended" || reason === "stopped") {
      // The session is over for everyone, who are all closing their connections now: send nothing
      // more. (WebKit reports a send on a channel the other side is closing as an uncaught error,
      // even inside try/catch.)
      const room = s.provider.room;
      if (room) for (const conn of room.webrtcConns.values()) conn.destroy();
    } else {
      try {
        if (s.sharing) s.shared.delete(String(s.ydoc.clientID));
        s.awareness.setLocalState(null); // tells the others at once
      } catch {
        /* already gone */
      }
      await sleep(150); // let the goodbye reach them
    }
    for (const b of s.bindings.values()) {
      b.destroy();
      try {
        b.doc.clearHistory(); // the history holds other people's edits; undo starts afresh
      } catch {
        /* the document is gone */
      }
    }
    s.bindings.clear();
    try {
      s.provider.destroy();
    } catch {
      /* not connected */
    }
    s.ydoc.destroy();
    for (const [path, t] of s.writes) {
      clearTimeout(t);
      writeNow(s, path);
    }
    if (reason === "ended" || reason === "stopped") await linkProject(s, false);
    if (/^#live=/.test(location.hash)) app.clearHash();
    followId = null;
    unfollowed = false;
    if (document.querySelector(".console-panel").dataset.view === "live") app.showBottom("console");
    if ($("dlg-live").open) {
      $("dlg-live").close();
      $("btn-live").focus();
    }
    renderAll();
    if (quiet) return;
    const p = state.projects.find((x) => x.id === s.projectId);
    const msg = {
      ended: `${by || "The host"} ended the live session.`,
      stopped: "You ended the live session for everyone.",
      switched: "You left the live session because you opened another project.",
      deleted: "You left the live session because its project was closed.",
    }[reason] || "You left the live session.";
    const kept = p && reason !== "deleted" ? ` Your copy of the files is in “${p.name}”${p.temp ? " (temporary: use Keep a copy to save it)" : ""}.` : "";
    toast(msg + kept);
    announce(msg);
  }

  async function endForEveryone() {
    if (!S || !amHost(S)) return;
    const ok = await confirmDialog({
      title: "End the live session for everyone?",
      text: "Everyone is disconnected and keeps their own copy of the files. The link no longer works once no one is in the session.",
      okLabel: "End session",
    });
    if (!ok || !S) return;
    S.ydoc.transact(() => {
      S.meta.set("stoppedBy", myName());
      S.meta.set("stopped", true);
    });
    await sleep(150);
    await leave({ reason: "stopped" });
  }

  /** Remember (or forget) on the project which session it belongs to: the session id, never the key. */
  async function linkProject(s, on) {
    const p = state.projects.find((x) => x.id === s.projectId);
    if (!p) return;
    const meta = { ...(p.meta || {}) };
    if (on) meta.live = s.room;
    else if (meta.live === s.room) delete meta.live;
    else return;
    p.meta = meta;
    await store.putProject(p);
  }

  /* ---------- files: the project and the shared document ---------- */

  /** Put this person's files into the shared document (the start of a session). */
  function seedFiles(s) {
    const Y = s.L.Y;
    let total = 0;
    let count = 0;
    s.skipped.clear();
    for (const path of app.sortedPaths()) {
      const content = app.liveContent(path);
      const size = content.length;
      const why = !safePath(path)
        ? "its name can't be shared"
        : typeof content === "string" && size > LIMIT.text
          ? `a text file over ${formatSize(LIMIT.text)}`
          : typeof content !== "string" && size > LIMIT.binary
            ? `over ${formatSize(LIMIT.binary)}`
            : count >= LIMIT.files
              ? `more than ${LIMIT.files} files`
              : total + size > LIMIT.total
                ? `over the ${formatSize(LIMIT.total)} total`
                : "";
      if (why) {
        s.skipped.set(path, why);
        continue;
      }
      total += size;
      count++;
      s.files.set(path, typeof content === "string" ? new Y.Text(content) : content.slice());
    }
  }

  /** A file of the shared document, checked: its content, or null (and a note) when it is not taken. */
  function checkedValue(s, path, v) {
    const why = !safePath(path)
      ? "not a valid file name"
      : v instanceof s.L.Y.Text
        ? v.length > LIMIT.textReceived ? "too large" : ""
        : v instanceof Uint8Array
          ? v.length > LIMIT.binary ? "too large" : ""
          : "not a file";
    if (why) {
      s.ignored.set(String(path).slice(0, 160), why);
      return null;
    }
    s.ignored.delete(path);
    return v instanceof Uint8Array ? v.slice() : v.toString();
  }

  /** Every file of the shared document that may be taken, as [{ path, content }]. */
  function roomFiles(s) {
    const out = [];
    let total = 0;
    for (const [path, v] of [...s.files.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1))) {
      const content = checkedValue(s, path, v);
      if (content === null) continue;
      if (out.length >= LIMIT.files || total + content.length > 2 * LIMIT.total) {
        s.ignored.set(path, "over the size limit of a session");
        continue;
      }
      total += content.length;
      out.push({ path, content });
    }
    // a file where another needs a folder (a and a/b.m) can't be stored
    const paths = out.map((f) => f.path);
    return out.filter((f) => {
      const clash = paths.some((p) => p !== f.path && p.startsWith(f.path + "/"));
      if (clash) s.ignored.set(f.path, "a folder of the same name exists");
      return !clash;
    });
  }

  /** Make the project hold exactly these files (open documents are kept in step by their bindings). */
  async function replaceLocal(s, files) {
    if (state.project.id !== s.projectId) return;
    const want = new Map(files.map((f) => [f.path, f.content]));
    const remove = app.sortedPaths().filter((p) => !want.has(p));
    const write = files.filter((f) => !s.bindings.has(f.path) && !sameContent(state.files.has(f.path) ? app.liveContent(f.path) : undefined, f.content));
    s.applying++;
    try {
      if (remove.length) await app.removePaths(remove);
      if (write.length) await app.writeFiles(write);
    } finally {
      s.applying--;
    }
  }

  function bindDoc(s, path, doc) {
    if (!s.ready || s.bindings.has(path)) return;
    const yt = s.files.get(path);
    if (!(yt instanceof s.L.Y.Text)) return;
    const b = new DocBinding({
      Y: s.L.Y, ytext: yt, doc, awareness: s.awareness, path,
      userOf: (id) => userOf(s, id),
      onRemoteChange: (p) => {
        if (p !== state.active) app.scheduleSave(p); // the editor saves the file that is shown
      },
    });
    s.bindings.set(path, b);
  }

  function unbind(s, path) {
    const b = s.bindings.get(path);
    if (!b) return;
    b.destroy();
    s.bindings.delete(path);
  }

  function bindOpenDocs(s) {
    for (const [path, doc] of state.docs) bindDoc(s, path, doc);
  }

  /** A file this person wrote (new, imported, written by code): into the shared document. */
  function putFile(s, path, content) {
    const Y = s.L.Y;
    if (!safePath(path)) return;
    const size = content.length;
    const big = typeof content === "string" ? size > LIMIT.text : size > LIMIT.binary;
    if (big) {
      s.skipped.set(path, typeof content === "string" ? `a text file over ${formatSize(LIMIT.text)}` : `over ${formatSize(LIMIT.binary)}`);
      return;
    }
    s.skipped.delete(path);
    const cur = s.files.get(path);
    if (typeof content === "string") {
      if (!(cur instanceof Y.Text)) s.files.set(path, new Y.Text(content));
      else if (cur.toString() !== content) {
        // change only the part that differs, so that others' edits elsewhere in the file survive
        const old = cur.toString();
        let a = 0;
        while (a < old.length && a < content.length && old[a] === content[a]) a++;
        let b = 0;
        while (b < old.length - a && b < content.length - a && old[old.length - 1 - b] === content[content.length - 1 - b]) b++;
        if (old.length - a - b) cur.delete(a, old.length - a - b);
        if (content.length - a - b) cur.insert(a, content.slice(a, content.length - b));
      }
    } else if (!(cur instanceof Uint8Array) || !sameBytes(cur, content)) {
      s.files.set(path, content.slice());
    }
  }

  function flushMoves(s) {
    const moves = s.pendingMoves;
    s.pendingMoves = [];
    if (s !== S || !moves.length) return;
    const Y = s.L.Y;
    s.ydoc.transact(() => {
      for (const [from, to] of moves) {
        unbind(s, from);
        const v = s.files.get(from);
        if (v === undefined) {
          if (state.files.has(to)) putFile(s, to, app.liveContent(to));
          continue;
        }
        const copy = v instanceof Y.Text ? new Y.Text(v.toString()) : v instanceof Uint8Array ? v.slice() : null;
        s.files.delete(from);
        if (copy && safePath(to)) s.files.set(to, copy);
      }
    });
    for (const [, to] of moves) {
      const doc = state.docs.get(to);
      if (doc) bindDoc(s, to, doc);
    }
    queueRender();
  }

  /** Files added, replaced or removed by someone else. */
  function onFilesMap(s, event, tr) {
    if (s !== S || tr.local || !s.ready) return;
    const changes = [...event.changes.keys].map(([path, ch]) => ({ path, action: ch.action }));
    s.queue = s.queue.then(() => applyRemoteFiles(s, changes)).catch((e) => console.warn("mylab live:", e));
  }

  async function applyRemoteFiles(s, changes) {
    if (s !== S || !s.ready || state.project.id !== s.projectId) return;
    const dels = [];
    const puts = [];
    for (const { path, action } of changes) {
      if (action === "delete") {
        unbind(s, path);
        s.ignored.delete(path);
        if (state.files.has(path)) dels.push(path);
        continue;
      }
      const v = s.files.get(path);
      if (v === undefined) continue;
      const content = checkedValue(s, path, v);
      if (content !== null) puts.push({ path, content });
    }
    // a renamed file arrives as a removal and an addition with the same content
    const moves = [];
    for (const d of [...dels]) {
      const old = app.liveContent(d);
      const i = puts.findIndex((p) => !state.files.has(p.path) && sameContent(old, p.content));
      if (i < 0) continue;
      moves.push([d, puts[i].path]);
      dels.splice(dels.indexOf(d), 1);
      puts.splice(i, 1);
    }
    const clashes = (path) => [...state.files.keys()].some((o) => o !== path && (o.startsWith(path + "/") || path.startsWith(o + "/")));
    const okMoves = moves.filter(([, to]) => !clashes(to));
    s.applying++;
    try {
      if (okMoves.length) await app.movePaths(okMoves);
      if (dels.length) await app.removePaths(dels);
      // an open document is bound to the new text instead of being overwritten
      const writes = puts.filter((p) => !(typeof p.content === "string" && state.docs.has(p.path)) && !clashes(p.path));
      if (writes.length) await app.writeFiles(writes);
      for (const p of puts) if (state.docs.has(p.path)) unbind(s, p.path);
    } finally {
      s.applying--;
    }
    for (const path of [...okMoves.map(([, to]) => to), ...puts.map((p) => p.path)]) {
      const doc = state.docs.get(path);
      if (doc && s === S) bindDoc(s, path, doc);
    }
    queueRender();
  }

  /** Text typed by someone else in a file this person has not opened. */
  function onFilesDeep(s, events, tr) {
    if (s !== S || tr.local || !s.ready) return;
    for (const e of events) {
      if (e.path.length !== 1 || !(e.target instanceof s.L.Y.Text)) continue;
      const path = e.path[0];
      if (s.bindings.has(path)) continue;
      const f = state.files.get(path);
      if (!f || typeof f.content !== "string" || state.docs.has(path)) continue;
      const text = e.target.toString();
      if (text.length > LIMIT.textReceived) continue;
      f.content = text;
      f.modified = Date.now();
      clearTimeout(s.writes.get(path));
      s.writes.set(path, setTimeout(() => writeNow(s, path), 500));
    }
  }

  function writeNow(s, path) {
    s.writes.delete(path);
    const f = state.files.get(path);
    if (f && state.project && state.project.id === s.projectId) store.putFile(s.projectId, path, f.content);
  }

  /* ---------- the session's name, host, end ---------- */

  function onMeta(s, e, tr) {
    if (s !== S) return;
    if (!s.ready && !s.adopting && s.meta.get("v")) adoptRoom(s);
    if (e.keysChanged.has("stopped") && s.meta.get("stopped") && !tr.local) leave({ reason: "ended", by: cleanName(s.meta.get("stoppedBy")) });
    else if (e.keysChanged.has("host")) updateLocalAwareness(s);
    queueRender();
  }

  /* ---------- people ---------- */

  /** A person's name and colour, checked (awareness states come from the others). */
  function userOf(s, id) {
    const st = s.awareness.getStates().get(id);
    if (!st || typeof st !== "object" || !st.user || typeof st.user !== "object") return null;
    const name = cleanName(st.user.name) || "Someone";
    const color = COLORS.includes(st.user.color) ? st.user.color : COLORS[Math.abs(Number(id) || 0) % COLORS.length];
    return { name, color };
  }

  function people(s) {
    const me = s.ydoc.clientID;
    const out = [];
    for (const [id, st] of s.awareness.getStates()) {
      const u = userOf(s, id);
      if (!u) continue;
      const file = typeof st.file === "string" && safePath(st.file) ? st.file : null;
      out.push({ id, ...u, me: id === me, host: st.host === true, sharing: st.sharing === true && s.shared.get(String(id)) instanceof s.L.Y.Map, file });
    }
    out.sort((a, b) => b.me - a.me || b.host - a.host || a.name.localeCompare(b.name));
    return out;
  }

  function onAwareness(s, { added, updated, removed }) {
    if (s !== S) return;
    const me = s.ydoc.clientID;
    // two people with one colour: the one who came later (higher id) takes a free colour
    const mine = s.awareness.getLocalState();
    if (mine && mine.user) {
      const taken = new Set();
      let clash = false;
      for (const [id] of s.awareness.getStates()) {
        if (id === me) continue;
        const u = userOf(s, id);
        if (!u) continue;
        taken.add(u.color);
        if (u.color === mine.user.color && id < me) clash = true;
      }
      const free = clash && COLORS.find((c) => !taken.has(c));
      if (free) s.awareness.setLocalStateField("user", { ...mine.user, color: free });
    }
    // the people who are there when this person arrives are named once, when the files have arrived
    const settled = s.ready && !s.adopting;
    for (const id of [...added, ...updated]) {
      if (id === me) continue;
      const u = userOf(s, id);
      if (!u) continue;
      if (!s.known.has(id) && settled) queueAnnounce("joined", u.name);
      s.known.set(id, u.name);
    }
    for (const id of removed) {
      if (id === me || !s.known.has(id)) continue;
      queueAnnounce("left", s.known.get(id));
      s.known.delete(id);
    }
    queueRender();
  }

  function updateLocalAwareness(s) {
    if (s !== S) return;
    const st = s.awareness.getLocalState() || {};
    const b = s.ready && state.active ? s.bindings.get(state.active) : null;
    const shown = b && app.editor.getDoc() === b.doc && !app.editor.getWrapperElement().hidden;
    const next = {
      ...st,
      user: { name: myName(), color: (st.user && st.user.color) || COLORS[s.ydoc.clientID % COLORS.length] },
      file: s.ready && state.active && s.files.has(state.active) ? state.active : null,
      cursor: shown ? b.cursorState() : null,
      sharing: s.sharing,
      host: amHost(s),
    };
    if (JSON.stringify(next) !== JSON.stringify(st)) s.awareness.setLocalState(next);
  }

  function queueCursor() {
    if (!S || !S.ready) return;
    clearTimeout(cursorTimer);
    cursorTimer = setTimeout(() => S && updateLocalAwareness(S), 40);
  }

  function queueAnnounce(kind, name) {
    pending[kind].push(name);
    clearTimeout(pending.timer);
    pending.timer = setTimeout(() => {
      const parts = [];
      if (pending.joined.length) parts.push(`${listText(pending.joined)} joined the live session.`);
      if (pending.left.length) parts.push(`${listText(pending.left)} left the live session.`);
      pending.joined = [];
      pending.left = [];
      announce(parts.join(" "));
    }, 1200);
  }

  /** Tell screen readers (one polite message, joins and leaves gathered). */
  function announce(text) {
    const r = $("live-announce");
    r.textContent = "";
    setTimeout(() => (r.textContent = text), 60);
  }

  /* ---------- sharing a session: command window, figures, workspace ---------- */

  function setSharing(s, on) {
    const key = String(s.ydoc.clientID);
    s.sharing = on;
    if (on) {
      const m = new s.L.Y.Map();
      s.ydoc.transact(() => {
        s.shared.set(key, m);
        m.set("log", new s.L.Y.Array());
        m.set("vars", varsSummary(state.variables));
      });
    } else {
      s.shared.delete(key);
    }
    updateLocalAwareness(s);
    queueRender();
  }

  const varsSummary = (list) => (list || []).slice(0, 100).map((v) => ({ n: clip(v.name, 63), s: clip(v.size, 30), c: clip(v.class, 30), p: clip(v.preview, 120) }));

  function myFeed(s) {
    const m = s.shared.get(String(s.ydoc.clientID));
    return m instanceof s.L.Y.Map ? m : null;
  }

  function pushLog(s, entries) {
    const m = myFeed(s);
    const log = m && m.get("log");
    if (!log || !entries.length) return;
    s.ydoc.transact(() => {
      log.push(entries.map((e) => ({ ...e, at: Date.now() })));
      if (log.length > 200) log.delete(0, log.length - 200);
      const all = log.toArray();
      let figs = 0;
      for (let i = all.length - 1; i >= 0; i--) if (all[i].k === "fig" && ++figs > 6) log.delete(i, 1);
    });
  }

  app.on("runStarting", (ctx, code) => {
    if (!S || !S.sharing) return;
    pushLog(S, [ctx.kind === "command" ? { k: "in", t: clip(code, 4000) } : { k: "run", t: clip(ctx.label || "run", 400) }]);
  });

  app.on("runFinished", (res, err) => {
    if (!S || !S.sharing || !res) return;
    const entries = [];
    if (res.output) entries.push({ k: "out", t: clip(res.output, 20000) });
    if (res.error) entries.push({ k: "err", t: clip((err && err.where ? err.where + ": " : "") + res.error, 4000) });
    for (const b64 of (res.figures || []).slice(0, 4)) entries.push(isPng(b64) ? { k: "fig", png: b64 } : { k: "info", t: "A figure too large to share." });
    if ((res.animations || []).length) entries.push({ k: "info", t: "An animation (animations are not shared)." });
    pushLog(S, entries);
  });

  app.on("variables", (list) => {
    const m = S && S.sharing && myFeed(S);
    if (m) m.set("vars", varsSummary(list));
  });

  /* ---------- hooks into the app ---------- */

  app.on("docCreated", (path, doc) => S && bindDoc(S, path, doc));
  app.on("filesWritten", (files) => {
    const s = S;
    if (!s || !s.ready || s.applying) return;
    s.ydoc.transact(() => {
      for (const f of files) putFile(s, f.path, f.content);
    });
    queueRender();
  });
  app.on("moved", (from, to) => {
    const s = S;
    if (!s || !s.ready || s.applying) return;
    s.pendingMoves.push([from, to]);
    if (s.pendingMoves.length === 1) queueMicrotask(() => flushMoves(s));
  });
  app.on("removed", (paths) => {
    const s = S;
    if (!s || !s.ready || s.applying) return;
    s.ydoc.transact(() => {
      for (const p of paths) {
        unbind(s, p);
        s.skipped.delete(p);
        s.files.delete(p);
      }
    });
    queueRender();
  });
  app.on("projectChanged", (p) => {
    if (S && S.projectId && p.id !== S.projectId && !S.adopting) leave({ reason: "switched" });
    else queueCursor();
  });
  app.on("projectDeleted", (id) => {
    if (S && S.projectId === id) leave({ reason: "deleted" });
  });
  app.on("fileOpened", queueCursor);
  app.editor.on("cursorActivity", queueCursor);

  // Undo and redo in a shared file take back only this person's own edits.
  for (const name of ["undo", "redo"]) {
    const original = CodeMirror.commands[name];
    CodeMirror.commands[name] = (cm) => {
      const b = S && S.ready ? [...S.bindings.values()].find((x) => x.doc === cm.getDoc()) : null;
      return b ? b[name]() : original(cm);
    };
  }

  // y-webrtc decrypts signalling messages without catching a failure: one that was not encrypted with
  // this session's key (another client on the same topic) is dropped quietly instead of logged.
  window.addEventListener("unhandledrejection", (e) => {
    if (S && e.reason && e.reason.name === "OperationError") e.preventDefault();
  });

  /** "Keep a copy" in a session's temporary project: the project becomes a saved one and stays in the session. */
  async function keepLiveCopy(p) {
    if (!p.temp || !p.meta || !p.meta.live) return false;
    await app.flushAll();
    const files = app.sortedPaths().map((path) => ({ path, content: app.liveContent(path) }));
    p.name = app.uniqueProjectName(String(p.meta.name || p.name.replace(/^Live: /, "")));
    p.temp = false;
    p.kind = "files";
    await store.makePermanent(p, files);
    app.sortProjects();
    prefs.set("project", p.id);
    app.renderProjectHead();
    $("save-state").textContent = store.persistent ? "Saved" : "Not saved (no storage)";
    toast(`Saved as the project “${p.name}”. It now lives in this browser` + (S && S.projectId === p.id ? " and stays in the live session." : "."));
    queueRender();
    return true;
  }

  /* ---------- status ---------- */

  function tick(s) {
    if (s !== S) return;
    const now = Date.now();
    const room = s.provider.room;
    const signal = s.provider.signalingConns.some((c) => c.connected);
    if (signal && !s.signalSince) s.signalSince = now;
    if (!signal) s.signalSince = 0;
    const conns = room ? [...room.webrtcConns.values()] : [];
    const connected = conns.filter((c) => c.connected).length + (room ? room.bcConns.size : 0);
    if (connected && !s.connectedSince) s.connectedSince = now;
    if (!connected) s.connectedSince = 0;
    if (!signal && !s.signalLost) s.signalLost = now;
    if (signal) s.signalLost = 0;
    // a rejoining person with a copy of the files, and no one else around: carry on with that copy
    if (!s.ready && !s.adopting && s.canSeed && signal && !s.peersSeen && now - s.signalSince >= ALONE_MS) seedFromLocal(s);
    let kind;
    let text;
    if (connected) {
      kind = "ok";
      text = `Connected to ${connected} other ${connected === 1 ? "browser" : "browsers"}.`;
      if (!s.ready) text += " Receiving the files…";
    } else if (!signal && now - (s.signalLost || s.opened) > SIGNAL_MS && !connected) {
      kind = "problem";
      text = `Can't reach the signalling server that introduces the browsers to each other (${s.signaling.map((u) => u.replace(/^wss?:\/\//, "")).join(", ")}). Check your internet connection, or set another server in Settings.`;
    } else if (!signal) {
      kind = "wait";
      text = "Connecting…";
    } else if (s.peersSeen && now - s.signalSince > BLOCKED_MS) {
      kind = "problem";
      text = "Someone is in the session, but your browsers couldn't connect to each other directly. Strict firewalls (common on school, company and some mobile networks) block such connections; a TURN server can relay them: add one in Settings.";
    } else if (!s.ready && now - s.opened > NOBODY_MS) {
      kind = "problem";
      text = "No one else is in this session right now. The session may have ended, the link may be incomplete, or the others are offline: a live session works only while someone who has the files is online.";
    } else if (!s.ready) {
      kind = "wait";
      text = "Waiting for someone in the session…";
    } else {
      kind = "wait";
      text = "No one else is here yet. Send the link to the people you want to work with.";
    }
    const status = { kind, text, connected };
    if (!s.status || s.status.text !== status.text) {
      s.status = status;
      queueRender();
    }
  }

  /* ---------- user interface ---------- */

  function queueRender() {
    if (renderQueued) return;
    renderQueued = true;
    requestAnimationFrame(() => {
      renderQueued = false;
      renderAll();
    });
  }

  function avatar(p) {
    return el("span", { class: "live-avatar", style: `background:${p.color}`, "aria-hidden": "true" }, initials(p.name));
  }

  const describe = (p) => p.name + (p.me ? " (you)" : "") + (p.host ? ", host" : "") + (p.sharing ? ", sharing their session" : "");

  function renderAll() {
    renderBar();
    if ($("dlg-live").open) renderDialog();
    renderFollow();
  }

  function renderBar() {
    const s = S;
    const btn = $("btn-live");
    btn.classList.toggle("live-on", !!s);
    btn.dataset.status = s && s.status ? s.status.kind : "";
    $("live-bar").hidden = !s;
    $("tab-live").hidden = !s;
    if (!s) {
      btn.setAttribute("aria-label", "Live collaboration");
      btn.title = "Live collaboration: work on this project together with others, at the same time";
      $("live-people").replaceChildren();
      return;
    }
    const list = people(s);
    btn.setAttribute("aria-label", `Live collaboration: on, ${list.length} ${list.length === 1 ? "person" : "people"}`);
    btn.title = "Live session: " + (s.status ? s.status.text : "connecting…");
    const sig = list.map((p) => [p.id, p.name, p.color, p.host, p.sharing].join(":")).join("|");
    if ($("live-people").dataset.sig === sig) return;
    $("live-people").dataset.sig = sig;
    $("live-people").replaceChildren(
      ...list.slice(0, 6).map((p) => el("li", { class: "live-person" + (p.me ? " me" : ""), title: describe(p), "data-client": String(p.id) }, avatar(p), el("span", { class: "sr-only" }, describe(p)))),
      ...(list.length > 6 ? [el("li", { class: "live-person more", title: list.slice(6).map((p) => p.name).join(", ") }, `+${list.length - 6}`)] : [])
    );
  }

  function renderDialog() {
    const s = S;
    $("live-start-view").hidden = !!s;
    $("live-session-view").hidden = !s;
    $("live-start").hidden = !!s;
    $("live-leave").hidden = !s;
    $("live-end").hidden = !(s && amHost(s));
    if (!s) {
      if (document.activeElement !== $("live-name")) $("live-name").value = cleanName(prefs.get("liveName", ""));
      $("live-project").textContent = state.project.name;
      return;
    }
    const link = linkFor(s);
    if ($("live-link").value !== link) $("live-link").value = link;
    const st = $("live-conn");
    const text = s.status ? s.status.text : "Connecting…";
    if (st.textContent !== text) st.textContent = text;
    st.className = "small live-conn " + (s.status ? s.status.kind : "wait");
    $("live-seed").hidden = !(!s.ready && s.canSeed && s.status && s.status.kind === "problem");
    $("live-share").checked = s.sharing;
    $("live-share").disabled = !s.ready;
    const list = people(s);
    const sig = JSON.stringify(list) + followId;
    if ($("live-list").dataset.sig !== sig) {
      $("live-list").dataset.sig = sig;
      $("live-list").replaceChildren(
        ...list.map((p) =>
          el(
            "li",
            { class: "live-row", "data-client": String(p.id) },
            avatar(p),
            el("span", { class: "live-who" }, el("strong", {}, p.name), p.me ? " (you)" : "", p.host ? el("span", { class: "pill small" }, "host") : null, p.file ? el("span", { class: "muted small" }, ` in ${p.file}`) : null),
            p.sharing && !p.me
              ? el("button", { type: "button", class: "btn sm" + (followId === p.id ? " primary" : ""), "aria-pressed": String(followId === p.id), onclick: () => follow(p.id, true) }, followId === p.id ? "Following" : "Follow")
              : p.sharing ? el("span", { class: "muted small" }, "sharing") : null
          )
        )
      );
    }
    const notes = [];
    if (s.skipped.size) notes.push("Not shared from your project: " + [...s.skipped].slice(0, 6).map(([p, why]) => `${p} (${why})`).join(", ") + (s.skipped.size > 6 ? ` and ${s.skipped.size - 6} more` : "") + ".");
    if (s.ignored.size) notes.push("Not taken from the session: " + [...s.ignored].slice(0, 6).map(([p, why]) => `${p} (${why})`).join(", ") + (s.ignored.size > 6 ? ` and ${s.ignored.size - 6} more` : "") + ".");
    $("live-skipped").hidden = !notes.length;
    $("live-skipped").textContent = notes.join(" ");
  }

  function openDialog() {
    renderDialog();
    if (!$("dlg-live").open) $("dlg-live").showModal();
    $("live-copied").hidden = true;
    (S ? $("live-copy") : cleanName($("live-name").value) ? $("live-start") : $("live-name")).focus();
  }

  async function copyLink() {
    if (!S) return;
    const ok = await copyText(linkFor(S));
    const note = $("live-copied");
    note.hidden = false;
    note.textContent = ok ? "Copied! Anyone with this link can join and edit the files while the session runs." : "Couldn't copy automatically. Select the link above and copy it by hand.";
    note.classList.toggle("error-text", !ok);
    if (!ok) $("live-link").select();
  }

  function follow(id, show) {
    followId = id;
    unfollowed = id === null;
    if (show) {
      if ($("dlg-live").open) $("dlg-live").close();
      app.showBottom("live");
      $("tab-live").focus();
    }
    queueRender();
  }

  function renderFollow() {
    const s = S;
    if (!s) {
      $("live-follow-body").hidden = true;
      $("live-follow").replaceChildren();
      return;
    }
    const sharers = people(s).filter((p) => !p.me && p.sharing);
    let gone = "";
    if (followId !== null && !sharers.some((p) => p.id === followId)) {
      gone = s.known.has(followId) ? `${s.known.get(followId)} stopped sharing their session.` : "The person you followed left the session.";
      followId = null;
    }
    if (followId === null && !unfollowed && sharers.length) followId = sharers[0].id;
    const sel = $("live-follow");
    const sig = sharers.map((p) => p.id + p.name).join("|") + "#" + followId;
    if (sel.dataset.sig !== sig) {
      sel.dataset.sig = sig;
      sel.replaceChildren(el("option", { value: "" }, "Nobody"), ...sharers.map((p) => el("option", { value: String(p.id) }, p.name + (p.host ? " (host)" : ""))));
      sel.value = followId === null ? "" : String(followId);
    }
    const who = sharers.find((p) => p.id === followId);
    $("live-follow-empty").hidden = !!who;
    $("live-follow-empty").textContent = gone || (sharers.length ? "Pick a person to follow their command window, figures and workspace." : "No one else is sharing their session. When someone does, their command window, figures and workspace appear here.");
    $("live-follow-body").hidden = !who;
    if (!who) return;
    const m = s.shared.get(String(who.id));
    const log = m.get("log");
    const entries = log instanceof s.L.Y.Array ? log.toArray().slice(-200) : [];
    const feed = $("live-feed");
    feed.setAttribute("aria-label", `${who.name}'s command window`);
    const sigFeed = who.id + ":" + entries.length + ":" + (entries.length ? entries[entries.length - 1].at : "");
    if (feed.dataset.sig !== sigFeed) {
      feed.dataset.sig = sigFeed;
      const nearBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 60;
      feed.replaceChildren(...entries.map((e) => feedEntry(e, who.name)).filter(Boolean));
      if (nearBottom || !feed.dataset.scrolled) feed.scrollTop = feed.scrollHeight;
    }
    const vars = Array.isArray(m.get("vars")) ? m.get("vars").slice(0, 100) : [];
    $("live-vars-title").textContent = `${who.name}'s workspace`;
    $("live-vars").querySelector("tbody").replaceChildren(
      ...vars.filter((v) => v && typeof v === "object").map((v) => el("tr", {}, el("td", {}, clip(v.n, 63)), el("td", {}, clip(v.s, 30)), el("td", {}, clip(v.c, 30)), el("td", { title: clip(v.p, 120) }, clip(v.p, 120))))
    );
    $("live-vars-empty").hidden = vars.length > 0;
    $("live-vars").hidden = !vars.length;
  }

  /** One entry of someone's command window, as text (or a PNG figure). */
  function feedEntry(e, name) {
    if (!e || typeof e !== "object") return null;
    const t = clip(e.t, 20000);
    switch (e.k) {
      case "in":
        return el("div", { class: "entry echo" }, el("b", {}, ">> "), t);
      case "run":
        return el("div", { class: "entry echo" }, el("b", {}, "▶ "), t);
      case "out":
        return el("div", { class: "entry out" }, t.replace(/\n{3,}$/, "\n\n"));
      case "err":
        return el("div", { class: "entry err" }, el("strong", {}, "Error: "), t);
      case "info":
        return el("div", { class: "entry info" }, t);
      case "fig":
        return isPng(e.png) ? el("figure", { class: "live-fig" }, el("img", { src: "data:image/png;base64," + e.png, alt: `Figure from ${name}'s session` })) : null;
      default:
        return null;
    }
  }

  /* ---------- settings fields ---------- */

  function setupSettings() {
    const sig = $("set-live-signaling");
    const ice = $("set-live-ice");
    const err = $("set-live-error");
    sig.value = savedSignaling().join("\n");
    ice.value = String(prefs.get("liveIce", "") || "");
    $("set-live-name").value = cleanName(prefs.get("liveName", ""));
    const complain = (text) => {
      err.hidden = !text;
      err.textContent = text;
    };
    sig.addEventListener("change", () => {
      const lines = sig.value.split("\n").map((l) => l.trim()).filter(Boolean);
      const bad = lines.filter((l) => !isSignalingUrl(l));
      prefs.set("liveSignaling", lines.filter(isSignalingUrl));
      complain(bad.length ? `Not a signalling server address (they start with wss:// or ws://): ${bad.join(", ")}` : "");
    });
    ice.addEventListener("change", () => {
      const lines = ice.value.split("\n").map((l) => l.trim()).filter(Boolean);
      const bad = lines.filter((l) => !isIceUrl(l.split(/\s+/)[0]));
      prefs.set("liveIce", lines.join("\n"));
      complain(bad.length ? `Not a STUN or TURN address (they start with stun:, turn: or turns:): ${bad.join(", ")}` : "");
    });
    $("set-live-name").addEventListener("change", (e) => setName(e.target.value));
  }

  /* ---------- wiring ---------- */

  $("btn-live").addEventListener("click", openDialog);
  $("live-start").addEventListener("click", startSession);
  $("live-copy").addEventListener("click", copyLink);
  $("live-link").addEventListener("focus", (e) => e.target.select());
  $("live-leave").addEventListener("click", () => leave());
  $("live-end").addEventListener("click", endForEveryone);
  $("live-seed").addEventListener("click", () => S && seedFromLocal(S));
  $("live-share").addEventListener("change", (e) => S && S.ready && setSharing(S, e.target.checked));
  $("live-name").addEventListener("change", (e) => setName(e.target.value));
  $("live-name").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      startSession();
    }
  });
  $("live-follow").addEventListener("change", (e) => follow(e.target.value ? Number(e.target.value) : null, false));
  $("live-feed").addEventListener("scroll", (e) => (e.target.dataset.scrolled = "1"));
  setupSettings();

  app.commands.register([
    { id: "live.open", group: "Project", title: "Live collaboration…", run: openDialog },
    { id: "live.leave", group: "Project", title: "Leave the live session", run: () => leave(), when: () => !!S },
    { id: "live.follow", group: "Project", title: "Show the live session's shared command windows", run: () => app.showBottom("live"), when: () => !!S },
  ]);

  Object.assign(app, {
    openLiveLink: (hash) => {
      joinFromLink(hash).catch((e) => {
        console.warn(e);
        toast(e.message || String(e), "error");
      });
      return true;
    },
    keepLiveCopy,
    openLiveDialog: openDialog,
    // for the tests and for debugging (never the key)
    live: {
      info: () =>
        S && {
          ready: S.ready,
          host: amHost(S),
          sharing: S.sharing,
          projectId: S.projectId,
          status: S.status,
          people: people(S).map((p) => ({ name: p.name, color: p.color, me: p.me, host: p.host, sharing: p.sharing, file: p.file })),
          files: [...S.files.keys()].sort(),
        },
      leave: (opts) => leave(opts),
    },
  });
}
