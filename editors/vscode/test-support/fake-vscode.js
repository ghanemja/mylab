'use strict';
// A very small stand-in for the `vscode` module, enough to load src/extension.js
// under plain Node and exercise its commands and providers. It records what the
// extension does instead of showing anything.

const path = require('node:path');

function create(settings) {
  const state = {
    commands: new Map(),
    providers: {},
    diagnostics: new Map(),
    messages: [],
    tasks: [],
    terminals: [],
    shown: [],
    outputLines: [],
    listeners: {},
    settings: settings || {},
    documents: [],
    activeTextEditor: undefined,
  };

  class Position {
    constructor(line, character) {
      this.line = line;
      this.character = character;
    }
  }
  class Range {
    constructor(a, b, c, d) {
      this.start = new Position(a, b);
      this.end = new Position(c, d);
    }
  }
  class Simple {
    constructor(...args) {
      this.args = args;
    }
  }
  class Diagnostic {
    constructor(range, message, severity) {
      Object.assign(this, { range, message, severity });
    }
  }
  class EventEmitter {
    constructor() {
      this.fired = [];
      this.event = () => ({ dispose() {} });
    }
    fire(x) {
      this.fired.push(x);
    }
  }
  class Task {
    constructor(definition, scope, name, source, execution) {
      Object.assign(this, { definition, scope, name, source, execution });
    }
  }
  class ProcessExecution {
    constructor(process, args, options) {
      Object.assign(this, { process, args, options });
    }
  }
  class CompletionItem {
    constructor(label, kind) {
      Object.assign(this, { label, kind });
    }
  }
  class SignatureHelp {}
  class SignatureInformation {
    constructor(label, documentation) {
      Object.assign(this, { label, documentation, parameters: [] });
    }
  }
  class ParameterInformation {
    constructor(label) {
      this.label = label;
    }
  }
  class MarkdownString {
    constructor(value) {
      this.value = value;
    }
  }
  class Hover {
    constructor(contents, range) {
      Object.assign(this, { contents, range });
    }
  }
  class FoldingRange {
    constructor(start, end, kind) {
      Object.assign(this, { start, end, kind });
    }
  }
  class DocumentSymbol {
    constructor(name, detail, kind, range, selectionRange) {
      Object.assign(this, { name, detail, kind, range, selectionRange });
    }
  }

  const Uri = {
    file: (fsPath) => ({ scheme: 'file', fsPath, toString: () => `file://${fsPath}` }),
    parse: (text) => ({ scheme: text.split(':')[0], fsPath: '', path: text, toString: () => text }),
  };

  const on = (name) => (fn) => {
    (state.listeners[name] = state.listeners[name] || []).push(fn);
    return { dispose() {} };
  };
  const register = (name) => (selector, provider) => {
    state.providers[name] = provider;
    return { dispose() {} };
  };

  const vscode = {
    Position, Range, Diagnostic, EventEmitter, Task, ProcessExecution, CompletionItem, SignatureHelp,
    SignatureInformation, ParameterInformation, MarkdownString, Hover, FoldingRange, DocumentSymbol, Uri,
    ShellExecution: Simple,
    DiagnosticSeverity: { Error: 0, Warning: 1 },
    CompletionItemKind: { Function: 2, Keyword: 13 },
    FoldingRangeKind: { Comment: 1, Imports: 2, Region: 3 },
    SymbolKind: { Namespace: 2, Class: 4, Function: 11 },
    TaskScope: { Global: 1, Workspace: 2 },
    TaskRevealKind: { Always: 1 },
    TaskPanelKind: { Dedicated: 2 },
    ViewColumn: { Beside: -2 },
    window: {
      get activeTextEditor() {
        return state.activeTextEditor;
      },
      createOutputChannel: () => ({
        appendLine: (s) => state.outputLines.push(s),
        append: (s) => state.outputLines.push(s),
        clear() {},
        show() {},
        dispose() {},
      }),
      showErrorMessage: (...args) => {
        state.messages.push(['error', ...args]);
        return Promise.resolve(undefined);
      },
      showInformationMessage: (...args) => {
        state.messages.push(['info', ...args]);
        return Promise.resolve(undefined);
      },
      showInputBox: () => Promise.resolve(undefined),
      createTerminal: (options) => {
        state.terminals.push(options);
        return { show() {} };
      },
      showTextDocument: (doc) => {
        state.shown.push(doc);
        return Promise.resolve();
      },
    },
    commands: {
      registerCommand: (id, fn) => {
        state.commands.set(id, fn);
        return { dispose() {} };
      },
      executeCommand: () => Promise.resolve(),
    },
    languages: {
      createDiagnosticCollection: () => ({
        set: (uri, list) => state.diagnostics.set(uri.toString(), list),
        delete: (uri) => state.diagnostics.delete(uri.toString()),
        dispose() {},
      }),
      registerHoverProvider: register('hover'),
      registerCompletionItemProvider: register('completion'),
      registerSignatureHelpProvider: register('signature'),
      registerFoldingRangeProvider: register('folding'),
      registerDocumentSymbolProvider: register('symbols'),
    },
    workspace: {
      isTrusted: true,
      workspaceFolders: undefined,
      get textDocuments() {
        return state.documents;
      },
      getConfiguration: (section) => ({
        get: (key, fallback) => {
          const full = `${section}.${key}`;
          return Object.prototype.hasOwnProperty.call(state.settings, full) ? state.settings[full] : fallback;
        },
      }),
      registerTextDocumentContentProvider: (scheme, provider) => {
        state.providers.content = provider;
        return { dispose() {} };
      },
      openTextDocument: (uri) => Promise.resolve({ uri, getText: () => state.providers.content.provideTextDocumentContent(uri) }),
      onDidOpenTextDocument: on('open'),
      onDidSaveTextDocument: on('save'),
      onDidChangeTextDocument: on('change'),
      onDidCloseTextDocument: on('close'),
      onDidChangeConfiguration: on('config'),
    },
    tasks: {
      executeTask: (task) => {
        state.tasks.push(task);
        return Promise.resolve();
      },
    },
    extensions: { getExtension: () => undefined },
    env: { openExternal: () => Promise.resolve(true) },
  };

  /** A document that behaves like vscode.TextDocument for what the extension uses. */
  function makeDocument(fsPath, text, options) {
    const o = options || {};
    const lines = () => doc.text.split(/\r\n|\r|\n/);
    const doc = {
      text,
      uri: fsPath ? Uri.file(fsPath) : { scheme: 'untitled', fsPath: '', toString: () => 'untitled:Untitled-1' },
      languageId: o.languageId || 'mylab-matlab',
      isDirty: !!o.dirty,
      isUntitled: !fsPath,
      isClosed: false,
      getText: (range) => {
        if (!range) return doc.text;
        const ls = lines().slice(range.start.line, range.end.line + 1);
        ls[ls.length - 1] = ls[ls.length - 1].slice(0, range.end.character);
        ls[0] = ls[0].slice(range.start.character);
        return ls.join('\n');
      },
      lineAt: (n) => ({ text: lines()[n] }),
      save: () => Promise.resolve(true),
      fileName: fsPath ? path.basename(fsPath) : 'Untitled-1',
    };
    return doc;
  }

  function makeEditor(doc, startLine, startCharacter, endLine, endCharacter) {
    const start = new Position(startLine, startCharacter);
    const end = new Position(endLine === undefined ? startLine : endLine, endCharacter === undefined ? startCharacter : endCharacter);
    return {
      document: doc,
      selection: { start, end, active: end, isEmpty: start.line === end.line && start.character === end.character },
    };
  }

  return { vscode, state, makeDocument, makeEditor, Position };
}

/** Make require('vscode') return the fake while `fn` loads the extension. */
function loadExtension(fake) {
  const Module = require('node:module');
  const originalLoad = Module._load;
  Module._load = function (request, ...rest) {
    if (request === 'vscode') return fake.vscode;
    return originalLoad.call(this, request, ...rest);
  };
  try {
    const file = require.resolve('../src/extension.js');
    delete require.cache[file];
    return require(file);
  } finally {
    Module._load = originalLoad;
  }
}

module.exports = { create, loadExtension };
