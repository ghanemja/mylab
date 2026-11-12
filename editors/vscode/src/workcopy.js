'use strict';
// Checking unsaved text. `mylab --check` reads a file, and it treats every
// NAME.m next to that file as a known function. So for unsaved text we build
// a small mirror folder in the system's temporary directory: the text under
// the file's own name, plus an empty file for each neighbouring .m file.
// Nothing is ever written into the user's folder. No dependency on VS Code.

const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');

class WorkCopies {
  constructor(root) {
    this.root = root || null; // created on first use
    this.listings = new Map(); // mirror folder -> names of the stubs written
  }

  ensureRoot() {
    if (!this.root) this.root = fs.mkdtempSync(path.join(os.tmpdir(), 'mylab-vscode-'));
    return this.root;
  }

  /** The mirror folder for a source folder (stable for the life of this object). */
  mirrorFor(sourceDir) {
    const id = crypto.createHash('sha1').update(sourceDir || 'untitled').digest('hex').slice(0, 12);
    const dir = path.join(this.ensureRoot(), id);
    fs.mkdirSync(dir, { recursive: true });
    return dir;
  }

  /**
   * Write `text` as a temporary copy of `filePath` and return the copy's path.
   * `filePath` may be null for an untitled document.
   */
  write(filePath, text) {
    const sourceDir = filePath ? path.dirname(filePath) : null;
    const base = filePath ? path.basename(filePath) : 'untitled.m';
    const dir = this.mirrorFor(sourceDir);
    if (sourceDir) this.syncStubs(sourceDir, dir, base);
    const copy = path.join(dir, base);
    fs.writeFileSync(copy, text, 'utf8');
    return copy;
  }

  syncStubs(sourceDir, dir, keep) {
    let names;
    try {
      names = fs.readdirSync(sourceDir).filter((n) => n.endsWith('.m') && n !== keep).sort();
    } catch (e) {
      return;
    }
    const key = names.join('\n');
    if (this.listings.get(dir) === key) return;
    const wanted = new Set(names);
    for (const old of fs.readdirSync(dir)) {
      if (old !== keep && !wanted.has(old)) fs.rmSync(path.join(dir, old), { force: true });
    }
    for (const n of names) {
      const stub = path.join(dir, n);
      if (!fs.existsSync(stub)) fs.writeFileSync(stub, '');
    }
    this.listings.set(dir, key);
  }

  /** A temporary file for running a selection or section. */
  tempScript(text, name) {
    const dir = path.join(this.ensureRoot(), 'run');
    fs.mkdirSync(dir, { recursive: true });
    const file = path.join(dir, name || 'mylab_selection.m');
    fs.writeFileSync(file, text, 'utf8');
    return file;
  }

  dispose() {
    if (this.root) fs.rmSync(this.root, { recursive: true, force: true });
    this.root = null;
    this.listings.clear();
  }
}

module.exports = { WorkCopies };
