// Binds a shared Y.Text to a CodeMirror 5 document (CodeMirror.Doc), for live collaboration.
//
// mylab keeps one CodeMirror.Doc per open file and swaps them in and out of a single editor, so the
// binding works on the document, not on the editor: a file in a background tab stays in sync, and
// switching tabs keeps its cursor and history. (y-codemirror binds to the editor and resets the
// document's content whenever it is bound, which does not fit that model.)
//
// - Local edits are read from "beforeChange", which CodeMirror fires synchronously before each single
//   change with positions in the text as it is at that moment, and applied to the Y.Text at once.
// - Remote edits arrive as Y.Text deltas and are applied with replaceRange (origin "live"). The local
//   cursors keep their place in the text: text someone else inserts exactly at a cursor appears after
//   it, so two people typing at the same spot each keep their own words together.
// - Undo and redo use a Y.UndoManager that only tracks this person's edits, so Ctrl+Z never takes back
//   what someone else typed (CodeMirror's own history would).
// - Other people's cursors and selections are shown with bookmarks and marks in their colour.

const LIVE_ORIGIN = "live";

/** Index-based positions of the selections of a document, and which one is the primary selection. */
function selectionIndexes(doc) {
  const ranges = doc.listSelections().map((r) => ({ anchor: doc.indexFromPos(r.anchor), head: doc.indexFromPos(r.head) }));
  return { ranges, primary: doc.sel && Number.isInteger(doc.sel.primIndex) ? doc.sel.primIndex : 0 };
}

/**
 * Move an index across a remote change. An insertion at the very index leaves it in front of the new
 * text: the other person's typing does not push this person's cursor along.
 */
function mapIndex(i, delta) {
  let pos = 0;
  for (const d of delta) {
    if (d.retain) pos += d.retain;
    else if (d.insert) {
      const n = typeof d.insert === "string" ? d.insert.length : 1;
      if (pos < i) i += n;
      pos += n;
    } else if (d.delete) {
      if (pos < i) i = Math.max(pos, i - d.delete);
    }
  }
  return i;
}

export class DocBinding {
  /**
   * @param {object} o
   * @param {any} o.Y          the Yjs module
   * @param {any} o.ytext      the shared text
   * @param {any} o.doc        the CodeMirror.Doc
   * @param {any} o.awareness  y-protocols Awareness, for other people's cursors
   * @param {string} o.path    the file's path in the project (cursors carry it)
   * @param {(path: string) => void} [o.onRemoteChange]  called after a remote edit was applied
   * @param {(clientId: number) => {name: string, color: string} | null} o.userOf  a peer's checked name and colour
   */
  constructor({ Y, ytext, doc, awareness, path, onRemoteChange, userOf }) {
    this.Y = Y;
    this.ytext = ytext;
    this.doc = doc;
    this.awareness = awareness;
    this.path = path;
    this.onRemoteChange = onRemoteChange || (() => {});
    this.userOf = userOf;
    this.applyingRemote = false;
    this.marks = new Map(); // clientId -> [marks]
    this.undoManager = new Y.UndoManager(ytext, { trackedOrigins: new Set([this]), captureTimeout: 600 });
    this.undoManager.on("stack-item-added", ({ stackItem }) => stackItem.meta.set("sel", this._relativeSelection()));
    this.undoManager.on("stack-item-popped", ({ stackItem }) => {
      const sel = stackItem.meta.get("sel");
      if (sel) this._restoreSelection(sel);
    });

    // start from the shared text (usually it is the same already)
    this._setText(ytext.toString());

    this._onBefore = (d, change) => this._local(change);
    this._onY = (event, tr) => this._remote(event, tr);
    this._onAwareness = ({ added, updated, removed }) => {
      for (const id of [...added, ...updated, ...removed]) if (id !== awareness.clientID) this._drawCursor(id);
    };
    doc.on("beforeChange", this._onBefore);
    ytext.observe(this._onY);
    awareness.on("change", this._onAwareness);
    for (const id of awareness.getStates().keys()) if (id !== awareness.clientID) this._drawCursor(id);
  }

  /** Make the document's text equal to `text`, changing only the part that differs. */
  _setText(text) {
    const doc = this.doc;
    const old = doc.getValue();
    if (old === text) return;
    let a = 0;
    while (a < old.length && a < text.length && old[a] === text[a]) a++;
    let b = 0;
    while (b < old.length - a && b < text.length - a && old[old.length - 1 - b] === text[text.length - 1 - b]) b++;
    const sel = selectionIndexes(doc);
    const delta = [{ retain: a }, { delete: old.length - a - b }, { insert: text.slice(a, text.length - b) }];
    this._applying(() => doc.replaceRange(text.slice(a, text.length - b), doc.posFromIndex(a), doc.posFromIndex(old.length - b), LIVE_ORIGIN));
    this._putSelection(sel, delta);
  }

  _applying(fn) {
    const was = this.applyingRemote;
    this.applyingRemote = true;
    try {
      const cm = this.doc.getEditor();
      if (cm) cm.operation(fn);
      else fn();
    } finally {
      this.applyingRemote = was;
    }
  }

  _putSelection(sel, delta) {
    const doc = this.doc;
    const ranges = sel.ranges.map((r) => ({ anchor: doc.posFromIndex(mapIndex(r.anchor, delta)), head: doc.posFromIndex(mapIndex(r.head, delta)) }));
    if (!ranges.length) return;
    this._applying(() => doc.setSelections(ranges, Math.min(sel.primary, ranges.length - 1), { scroll: false, origin: LIVE_ORIGIN }));
  }

  /** A local change, just before CodeMirror applies it: the same change on the shared text. */
  _local(change) {
    if (this.applyingRemote) return;
    const doc = this.doc;
    let from = doc.indexFromPos(change.from);
    let to = doc.indexFromPos(change.to);
    let text = change.text.join("\n");
    if (change.origin === "setValue") {
      // a whole new text (a file written by code, a restored version): pass on only what differs,
      // so that what others type meanwhile elsewhere in the file stays where it is
      const old = doc.getValue();
      let a = 0;
      while (a < old.length && a < text.length && old[a] === text[a]) a++;
      let b = 0;
      while (b < old.length - a && b < text.length - a && old[old.length - 1 - b] === text[text.length - 1 - b]) b++;
      [from, to, text] = [a, old.length - b, text.slice(a, text.length - b)];
    }
    if (to === from && !text) return;
    this.ytext.doc.transact(() => {
      if (to > from) this.ytext.delete(from, to - from);
      if (text) this.ytext.insert(from, text);
    }, this);
  }

  /** A change of the shared text made elsewhere (another person, or undo/redo): the same on the document. */
  _remote(event, tr) {
    if (tr.origin === this) return;
    const doc = this.doc;
    const delta = event.delta;
    const sel = selectionIndexes(doc);
    this._applying(() => {
      let index = 0;
      for (const d of delta) {
        if (d.retain) index += d.retain;
        else if (d.insert) {
          const text = typeof d.insert === "string" ? d.insert : "";
          const pos = doc.posFromIndex(index);
          doc.replaceRange(text, pos, pos, LIVE_ORIGIN);
          index += text.length;
        } else if (d.delete) {
          doc.replaceRange("", doc.posFromIndex(index), doc.posFromIndex(index + d.delete), LIVE_ORIGIN);
        }
      }
    });
    // undo/redo moves the cursor itself (stack-item-popped); a remote edit leaves it alone
    if (tr.origin !== this.undoManager) this._putSelection(sel, delta);
    if (doc.getValue() !== this.ytext.toString()) this._setText(this.ytext.toString()); // never drift apart
    this.onRemoteChange(this.path);
  }

  _relativeSelection() {
    const Y = this.Y;
    const doc = this.doc;
    return doc.listSelections().map((r) => ({
      anchor: Y.createRelativePositionFromTypeIndex(this.ytext, doc.indexFromPos(r.anchor)),
      head: Y.createRelativePositionFromTypeIndex(this.ytext, doc.indexFromPos(r.head)),
    }));
  }

  _restoreSelection(sel) {
    const Y = this.Y;
    const doc = this.doc;
    const at = (rel) => {
      const abs = Y.createAbsolutePositionFromRelativePosition(rel, this.ytext.doc);
      return doc.posFromIndex(abs && abs.type === this.ytext ? abs.index : 0);
    };
    try {
      this._applying(() => doc.setSelections(sel.map((r) => ({ anchor: at(r.anchor), head: at(r.head) })), 0, { origin: LIVE_ORIGIN }));
    } catch {
      /* the text changed too much: leave the cursor where it is */
    }
  }

  undo() {
    this.undoManager.undo();
  }

  redo() {
    this.undoManager.redo();
  }

  /** The cursor of this person in this file, as relative positions for the awareness state. */
  cursorState() {
    const Y = this.Y;
    const doc = this.doc;
    const r = doc.listSelections()[0];
    if (!r) return null;
    return {
      path: this.path,
      anchor: Y.relativePositionToJSON(Y.createRelativePositionFromTypeIndex(this.ytext, doc.indexFromPos(r.anchor))),
      head: Y.relativePositionToJSON(Y.createRelativePositionFromTypeIndex(this.ytext, doc.indexFromPos(r.head))),
    };
  }

  /** Draw (or remove) another person's cursor and selection in this document. */
  _drawCursor(clientId) {
    for (const m of this.marks.get(clientId) || []) m.clear();
    this.marks.delete(clientId);
    const state = this.awareness.getStates().get(clientId);
    const user = state && this.userOf(clientId);
    const cur = state && state.cursor;
    if (!user || !cur || cur.path !== this.path || !cur.anchor || !cur.head) return;
    const Y = this.Y;
    let anchor;
    let head;
    try {
      anchor = Y.createAbsolutePositionFromRelativePosition(Y.createRelativePositionFromJSON(cur.anchor), this.ytext.doc);
      head = Y.createAbsolutePositionFromRelativePosition(Y.createRelativePositionFromJSON(cur.head), this.ytext.doc);
    } catch {
      return; // not a position this document knows
    }
    if (!anchor || !head || anchor.type !== this.ytext || head.type !== this.ytext) return;
    const doc = this.doc;
    const len = this.ytext.length;
    const a = Math.min(len, anchor.index);
    const h = Math.min(len, head.index);
    const marks = [];
    if (a !== h) {
      marks.push(doc.markText(doc.posFromIndex(Math.min(a, h)), doc.posFromIndex(Math.max(a, h)), {
        className: "live-sel",
        // a faint tint and an underline in their colour: the text keeps its contrast
        css: `background-color: color-mix(in srgb, ${user.color} 6%, transparent); box-shadow: inset 0 -2px 0 ${user.color}`,
        attributes: { "data-client": String(clientId) },
      }));
    }
    const caret = document.createElement("span");
    caret.className = "live-caret";
    caret.style.borderColor = user.color;
    caret.dataset.client = String(clientId);
    caret.dataset.name = user.name;
    caret.setAttribute("aria-hidden", "true");
    const label = document.createElement("span");
    label.className = "live-caret-name";
    label.style.backgroundColor = user.color;
    label.textContent = user.name;
    caret.appendChild(label);
    setTimeout(() => caret.classList.add("quiet"), 2500);
    marks.push(doc.setBookmark(doc.posFromIndex(h), { widget: caret, insertLeft: true }));
    this.marks.set(clientId, marks);
  }

  destroy() {
    this.doc.off("beforeChange", this._onBefore);
    this.ytext.unobserve(this._onY);
    this.awareness.off("change", this._onAwareness);
    for (const marks of this.marks.values()) for (const m of marks) m.clear();
    this.marks.clear();
    this.undoManager.destroy();
  }
}
