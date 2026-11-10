// The assistant panel: chat UI, quick actions and the three provider backends.
// API keys are kept in this browser's localStorage and sent only to the chosen provider.

import { parseMarkdown, escapeHtml } from "./markdown.js";
import { prefs } from "./storage.js";

const ANTHROPIC_SDK_URL = "https://cdn.jsdelivr.net/npm/@anthropic-ai/sdk@1.3.0/+esm";

export const DEFAULT_SETTINGS = {
  provider: "none",
  ollama: { url: "http://localhost:11434", model: "" },
  anthropic: { key: "", model: "claude-opus-5-5" },
  openai: { url: "https://api.openai.com/v1", model: "", key: "" },
};

export const SYSTEM_PROMPT = `You are the assistant built into mylab, a free, open-source environment that runs MATLAB-language code in the web browser. mylab is an interpreter written in Python on top of NumPy, SciPy and Matplotlib. It is not MATLAB and is not affiliated with The MathWorks. Most users are students learning to program and to do numerical work.

Your job is to help users write, understand and fix MATLAB-language code that runs in mylab.

How to answer:
- Be concise and practical. Give a short explanation, then code. Skip long introductions.
- Put MATLAB-language code in \`\`\`matlab fenced blocks and Python code in \`\`\`python blocks. The app adds "Insert into editor" and "Run" buttons to matlab blocks, so each block should run on its own.
- When you fix a whole file, return the complete corrected file in a single matlab block, then list the changes in a few bullets.
- When explaining an error, say what caused it, name the line, and show the corrected code.

What mylab supports:
- Prefer core language features and widely used functions: matrices and indexing, colon ranges, element-wise operators, if/elseif/else, for, while, switch, try/catch, functions (in their own .m files or as local functions at the end of a script), anonymous functions, cell arrays, structs, char arrays and strings, disp, fprintf, sprintf, num2str, zeros, ones, eye, rand, randn, linspace, size, numel, sum, prod, cumsum, mean, std, max, min, sort, find, any, all, abs, sqrt, exp, log, mod, round, linear algebra (inv, det, eig, rank, the backslash operator), polyfit, polyval, interp1, fzero, integral, ode45, and plotting with figure, plot, hold, subplot, bar, scatter, histogram, surf, mesh, contour, title, xlabel, ylabel, legend, grid and axis.
- Toolbox-specific functions, Simulink, GUI building (uicontrol, App Designer), file dialogs and live scripts are not available. Suggest a plain alternative written with basic operations.
- input() does not work in the browser version. Use variables at the top of the script instead.
- Plots are shown automatically after each run. Do not rely on pause, drawnow or animation loops.
- If you are not sure whether mylab supports a function, say so briefly and offer a fallback written with basic operations.

Each user message may start with a [mylab context] section holding the user's current file, their last error and a summary of their workspace. Use it to answer, but do not repeat it back.`;

const QUICK_PROMPTS = {
  error: "Explain this error in simple terms and show me how to fix it.",
  fix: "Find and fix the problems in my code. Give me the complete corrected code in one matlab block, then briefly list what you changed.",
  explain: "Explain what my code does, step by step, as if I were a student seeing it for the first time.",
  python:
    "Translate my code to Python using NumPy and Matplotlib. Give one complete python block, then note any differences in behavior (for example 0-based indexing).",
};

export function loadSettings() {
  const saved = prefs.get("ai", null) || {};
  return {
    provider: saved.provider || DEFAULT_SETTINGS.provider,
    ollama: { ...DEFAULT_SETTINGS.ollama, ...(saved.ollama || {}) },
    anthropic: { ...DEFAULT_SETTINGS.anthropic, ...(saved.anthropic || {}) },
    openai: { ...DEFAULT_SETTINGS.openai, ...(saved.openai || {}) },
  };
}

export function saveSettings(settings) {
  return prefs.set("ai", settings);
}

class FriendlyError extends Error {}

function trimSlash(u) {
  return String(u || "").trim().replace(/\/+$/, "");
}

/* ------------------------------------------------------------------ providers */

let anthropicSdk = null;
async function loadAnthropicSdk() {
  if (anthropicSdk) return anthropicSdk;
  try {
    const mod = await import(ANTHROPIC_SDK_URL);
    anthropicSdk = mod.default || mod.Anthropic;
    if (!anthropicSdk) throw new Error("missing export");
    return anthropicSdk;
  } catch (e) {
    console.warn("mylab: could not load the Anthropic SDK", e);
    throw new FriendlyError(
      "The Anthropic SDK couldn't be loaded from cdn.jsdelivr.net. Check your internet connection or any content blocker, then try again."
    );
  }
}

async function chatAnthropic(cfg, { system, messages, signal, onText }) {
  if (!cfg.key) throw new FriendlyError("Add your Anthropic API key in the assistant settings first.");
  const Anthropic = await loadAnthropicSdk();
  const client = new Anthropic({ apiKey: cfg.key.trim(), dangerouslyAllowBrowser: true });
  let stream;
  try {
    stream = client.beta.messages.stream({
      model: (cfg.model || DEFAULT_SETTINGS.anthropic.model).trim(),
      max_tokens: 16000,
      system,
      messages,
      betas: ["server-side-fallback-2026-07-01"],
      fallbacks: "default",
    });
    const onAbort = () => stream.abort();
    signal.addEventListener("abort", onAbort, { once: true });
    for await (const event of stream) {
      if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
        onText(event.delta.text);
      }
    }
    const msg = await stream.finalMessage();
    signal.removeEventListener("abort", onAbort);
    return { refusal: msg.stop_reason === "refusal", truncated: msg.stop_reason === "max_tokens", model: msg.model };
  } catch (e) {
    if (signal.aborted || e instanceof Anthropic.APIUserAbortError) return { aborted: true };
    if (e instanceof Anthropic.AuthenticationError)
      throw new FriendlyError("Anthropic rejected the API key. Check that it is correct and active in the assistant settings.");
    if (e instanceof Anthropic.PermissionDeniedError)
      throw new FriendlyError("This API key is not allowed to use that model. Check your Anthropic account or pick another model.");
    if (e instanceof Anthropic.NotFoundError)
      throw new FriendlyError(`Model "${cfg.model}" was not found. Check the model id in the assistant settings.`);
    if (e instanceof Anthropic.RateLimitError)
      throw new FriendlyError("Anthropic's rate limit was reached (or your credit ran out). Wait a moment and try again.");
    if (e instanceof Anthropic.BadRequestError)
      throw new FriendlyError(`Anthropic could not process the request: ${e.message}`);
    if (e instanceof Anthropic.InternalServerError)
      throw new FriendlyError("Anthropic's servers had a problem. Please try again in a moment.");
    if (e instanceof Anthropic.APIConnectionError)
      throw new FriendlyError("Couldn't connect to Anthropic. Check your internet connection and try again.");
    if (e instanceof Anthropic.APIError) throw new FriendlyError(`Anthropic error: ${e.message}`);
    throw e;
  }
}

async function readLines(response, signal, onLine) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).replace(/\r$/, "");
        buf = buf.slice(nl + 1);
        if (onLine(line) === false) return;
      }
      if (signal.aborted) return;
    }
    if (buf.trim()) onLine(buf);
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* ignore */
    }
  }
}

async function chatOllama(cfg, { system, messages, signal, onText }) {
  const base = trimSlash(cfg.url || DEFAULT_SETTINGS.ollama.url);
  if (!cfg.model) throw new FriendlyError("Choose an Ollama model in the assistant settings first.");
  let resp;
  try {
    resp = await fetch(base + "/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: cfg.model, messages: [{ role: "system", content: system }, ...messages], stream: true }),
      signal,
    });
  } catch (e) {
    if (signal.aborted) return { aborted: true };
    throw new FriendlyError(
      `Couldn't reach Ollama at ${base}. Make sure it is running and was started with OLLAMA_ORIGINS allowing ${location.origin} (see the assistant settings).`
    );
  }
  if (!resp.ok) {
    let detail = "";
    try {
      detail = (await resp.json()).error || "";
    } catch {
      /* ignore */
    }
    if (resp.status === 404)
      throw new FriendlyError(`Ollama doesn't have the model "${cfg.model}". Run: ollama pull ${cfg.model}`);
    if (resp.status === 403)
      throw new FriendlyError(`Ollama refused the request from this page. Start it with OLLAMA_ORIGINS=${location.origin}`);
    throw new FriendlyError(`Ollama error (${resp.status}): ${detail || resp.statusText}`);
  }
  let errorText = "";
  try {
    await readLines(resp, signal, (line) => {
      if (!line.trim()) return;
      let obj;
      try {
        obj = JSON.parse(line);
      } catch {
        return;
      }
      if (obj.error) {
        errorText = obj.error;
        return false;
      }
      if (obj.message && obj.message.content) onText(obj.message.content);
      if (obj.done) return false;
    });
  } catch (e) {
    if (signal.aborted) return { aborted: true };
    throw e;
  }
  if (errorText) throw new FriendlyError(`Ollama error: ${errorText}`);
  return signal.aborted ? { aborted: true } : {};
}

export async function listOllamaModels(url) {
  const base = trimSlash(url || DEFAULT_SETTINGS.ollama.url);
  const resp = await fetch(base + "/api/tags");
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const data = await resp.json();
  return (data.models || []).map((m) => m.name || m.model).filter(Boolean);
}

async function chatOpenAI(cfg, { system, messages, signal, onText }) {
  const base = trimSlash(cfg.url || DEFAULT_SETTINGS.openai.url);
  if (!cfg.model) throw new FriendlyError("Enter a model name in the assistant settings first.");
  const headers = { "Content-Type": "application/json" };
  if (cfg.key) headers.Authorization = "Bearer " + cfg.key.trim();
  let resp;
  try {
    resp = await fetch(base + "/chat/completions", {
      method: "POST",
      headers,
      body: JSON.stringify({ model: cfg.model, messages: [{ role: "system", content: system }, ...messages], stream: true }),
      signal,
    });
  } catch (e) {
    if (signal.aborted) return { aborted: true };
    throw new FriendlyError(
      `Couldn't reach ${base}. Check the URL and your connection. The server must also allow requests from web pages (CORS).`
    );
  }
  if (!resp.ok) {
    let detail = "";
    try {
      const j = await resp.json();
      detail = (j.error && (j.error.message || j.error)) || j.message || "";
    } catch {
      /* ignore */
    }
    if (resp.status === 401) throw new FriendlyError(`The server rejected the API key (401). ${detail}`);
    if (resp.status === 429) throw new FriendlyError(`Rate limit reached (429). Wait a moment and try again. ${detail}`);
    throw new FriendlyError(`Server error (${resp.status}): ${detail || resp.statusText}`);
  }
  let finish = null;
  try {
    await readLines(resp, signal, (line) => {
      const m = line.match(/^data:\s?(.*)$/);
      if (!m) return;
      const data = m[1].trim();
      if (data === "[DONE]") return false;
      let obj;
      try {
        obj = JSON.parse(data);
      } catch {
        return;
      }
      const choice = obj.choices && obj.choices[0];
      if (choice && choice.delta && choice.delta.content) onText(choice.delta.content);
      if (choice && choice.finish_reason) finish = choice.finish_reason;
    });
  } catch (e) {
    if (signal.aborted) return { aborted: true };
    throw e;
  }
  if (signal.aborted) return { aborted: true };
  return { refusal: finish === "content_filter", truncated: finish === "length" };
}

const PROVIDERS = {
  ollama: { label: "Ollama", chat: chatOllama, model: (s) => s.ollama.model },
  anthropic: { label: "Claude", chat: chatAnthropic, model: (s) => s.anthropic.model },
  openai: { label: "OpenAI-compatible", chat: chatOpenAI, model: (s) => s.openai.model },
};

/* ------------------------------------------------------------------ context */

function buildContext(ctx) {
  const parts = ["[mylab context]"];
  if (ctx.fileName) {
    let code = ctx.code || "";
    if (code.length > 16000) code = code.slice(0, 16000) + "\n% ... (file truncated)";
    parts.push(`Current file: ${ctx.fileName}\n\`\`\`matlab\n${code}\n\`\`\``);
  }
  if (ctx.selection) {
    parts.push(`Selected lines ${ctx.selection.from}-${ctx.selection.to}:\n\`\`\`matlab\n${ctx.selection.text}\n\`\`\``);
  }
  if (ctx.lastError) {
    const e = ctx.lastError;
    let s = `Last error${e.where ? " in " + e.where : ""}${e.line ? " (line " + e.line + ")" : ""}: ${e.message}`;
    if (e.code && e.where === "Command Window") s += `\nCommand: ${e.code}`;
    parts.push(s);
  } else {
    parts.push("Last error: none");
  }
  if (ctx.variables && ctx.variables.length) {
    const list = ctx.variables.slice(0, 40).map((v) => `${v.name} (${v.size} ${v.class})`);
    if (ctx.variables.length > 40) list.push(`and ${ctx.variables.length - 40} more`);
    parts.push("Workspace: " + list.join(", "));
  } else {
    parts.push("Workspace: empty");
  }
  parts.push("[end of mylab context]");
  return parts.join("\n\n");
}

/* ------------------------------------------------------------------ panel */

export class Assistant {
  /**
   * @param {object} hooks  { getContext(), insertCode(code), replaceFile(code), runCode(code), toast(msg, kind) }
   */
  constructor(hooks) {
    this.hooks = hooks;
    this.settings = loadSettings();
    this.history = []; // [{role, content}] as sent to the model
    this.controller = null;
    this.el = {
      panel: document.getElementById("ai-panel"),
      messages: document.getElementById("ai-messages"),
      form: document.getElementById("ai-form"),
      input: document.getElementById("ai-input"),
      send: document.getElementById("ai-send"),
      stop: document.getElementById("ai-stop"),
      explainError: document.getElementById("ai-explain-error"),
      label: document.getElementById("ai-provider-label"),
      dlg: document.getElementById("dlg-ai-settings"),
    };
    this._bind();
    this._updateLabel();
    this._welcome();
  }

  _bind() {
    const { form, input, stop, explainError } = this.el;
    form.addEventListener("submit", (ev) => {
      ev.preventDefault();
      const text = input.value.trim();
      if (!text || this.controller) return;
      input.value = "";
      this._autosize();
      this.ask(text);
    });
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) {
        ev.preventDefault();
        form.requestSubmit();
      }
    });
    input.addEventListener("input", () => this._autosize());
    stop.addEventListener("click", () => this.controller && this.controller.abort());
    explainError.addEventListener("click", () => this.quick("error"));
    document.querySelectorAll("[data-ai-action]").forEach((b) =>
      b.addEventListener("click", () => this.quick(b.dataset.aiAction))
    );
    document.getElementById("btn-ai-clear").addEventListener("click", () => this.clear());
    document.getElementById("btn-ai-settings").addEventListener("click", () => this.openSettings());
    this._bindSettings();
  }

  _autosize() {
    const t = this.el.input;
    t.style.height = "auto";
    t.style.height = Math.min(160, t.scrollHeight) + "px";
  }

  _updateLabel() {
    const p = PROVIDERS[this.settings.provider];
    this.el.label.textContent = p ? `${p.label}${p.model(this.settings) ? " · " + p.model(this.settings) : ""}` : "Off";
    this.el.label.title = this.el.label.textContent;
  }

  _welcome() {
    const div = document.createElement("div");
    div.className = "msg notice ai-welcome";
    const configured = !!PROVIDERS[this.settings.provider];
    div.innerHTML = configured
      ? `<h3>Ask about your code</h3><p>The assistant sees your current file, your last error and the names of your variables. Try the buttons below, or ask anything.</p>`
      : `<h3>Set up the assistant</h3>
         <p>The assistant can explain errors, fix code and translate it to Python. It is optional and needs one of:</p>
         <p>• <strong>Ollama</strong>: free, runs on your own computer<br>• <strong>Claude</strong> or an <strong>OpenAI-compatible</strong> service: your own API key</p>
         <p><button type="button" class="btn sm primary" data-open-settings>Open settings</button></p>`;
    const b = div.querySelector("[data-open-settings]");
    if (b) b.addEventListener("click", () => this.openSettings());
    this.el.messages.appendChild(div);
  }

  clear() {
    if (this.controller) this.controller.abort();
    this.history = [];
    this.el.messages.innerHTML = "";
    this._welcome();
  }

  setLastError(err) {
    this.el.explainError.hidden = !err;
  }

  quick(kind) {
    const text = QUICK_PROMPTS[kind];
    if (!text || this.controller) return;
    const ctx = this.hooks.getContext();
    if (kind !== "error" && !ctx.fileName && !ctx.selection) {
      this.hooks.toast("Open a file first.", "error");
      return;
    }
    this.ask(text);
  }

  _addNotice(text, isError = false) {
    const div = document.createElement("div");
    div.className = "msg notice" + (isError ? " error" : "");
    div.textContent = text;
    this.el.messages.appendChild(div);
    this._scroll();
    return div;
  }

  _scroll() {
    const m = this.el.messages;
    m.scrollTop = m.scrollHeight;
  }

  async ask(text) {
    const provider = PROVIDERS[this.settings.provider];
    const ctx = this.hooks.getContext();
    const userDiv = document.createElement("div");
    userDiv.className = "msg user";
    userDiv.textContent = text;
    const ctxNote = document.createElement("span");
    ctxNote.className = "ctx";
    const bits = [];
    if (ctx.selection) bits.push(`selection of ${ctx.fileName || "file"}`);
    else if (ctx.fileName) bits.push(ctx.fileName);
    if (ctx.lastError) bits.push("last error");
    if (ctx.variables && ctx.variables.length) bits.push(`${ctx.variables.length} variables`);
    ctxNote.textContent = bits.length ? "with " + bits.join(", ") : "";
    userDiv.appendChild(ctxNote);
    this.el.messages.appendChild(userDiv);
    this._scroll();

    if (!provider) {
      const n = this._addNotice("The assistant is off. Choose a provider in settings to start chatting. ");
      const b = document.createElement("button");
      b.className = "btn sm primary";
      b.type = "button";
      b.textContent = "Open settings";
      b.addEventListener("click", () => this.openSettings());
      n.appendChild(b);
      return;
    }

    const content = `${buildContext(ctx)}\n\n${text}`;
    const messages = [...this.history, { role: "user", content }];

    const out = document.createElement("div");
    out.className = "msg assistant";
    this.el.messages.appendChild(out);
    let acc = "";
    let pending = false;
    let finished = false;
    const render = (final = false) => {
      pending = false;
      if (finished) return;
      if (final) finished = true;
      this._renderAssistant(out, acc, !final);
      const m = this.el.messages;
      if (m.scrollHeight - m.scrollTop - m.clientHeight < 120 || final) this._scroll();
    };
    render();

    this.controller = new AbortController();
    this._setBusy(true);
    try {
      const result = await provider.chat(this.settings[this.settings.provider], {
        system: SYSTEM_PROMPT,
        messages,
        signal: this.controller.signal,
        onText: (t) => {
          acc += t;
          if (!pending) {
            pending = true;
            requestAnimationFrame(() => render());
          }
        },
      });
      render(true);
      if (acc.trim()) {
        this.history.push({ role: "user", content }, { role: "assistant", content: acc });
      }
      if (result && result.aborted) this._addNotice("Stopped.");
      else if (result && result.refusal)
        this._addNotice("The model declined this request. Try rephrasing it, or ask about a different part of your code.");
      else if (result && result.truncated) this._addNotice("The answer was cut off because it reached the length limit.");
      else if (!acc.trim()) this._addNotice("The model returned an empty answer.");
    } catch (e) {
      render(true);
      if (!acc.trim()) out.remove();
      const msg = e instanceof FriendlyError ? e.message : `Something went wrong: ${e && e.message ? e.message : e}`;
      if (!(e instanceof FriendlyError)) console.warn("mylab assistant error", e);
      this._addNotice(msg, true);
    } finally {
      this.controller = null;
      this._setBusy(false);
    }
  }

  _setBusy(busy) {
    this.el.send.hidden = busy;
    this.el.stop.hidden = !busy;
    document.querySelectorAll(".ai-quick .chip").forEach((c) => (c.disabled = busy));
  }

  _renderAssistant(el, text, streaming) {
    el.innerHTML = "";
    const blocks = parseMarkdown(text);
    for (const b of blocks) {
      if (b.type === "html") {
        const wrap = document.createElement("div");
        wrap.innerHTML = b.html;
        while (wrap.firstChild) el.appendChild(wrap.firstChild);
      } else {
        el.appendChild(this._codeBlock(b, streaming && !b.closed));
      }
    }
    if (streaming) {
      const t = document.createElement("span");
      t.className = "typing";
      (el.lastElementChild && el.lastElementChild.tagName === "P" ? el.lastElementChild : el).appendChild(t);
    }
  }

  _codeBlock(b, incomplete) {
    const box = document.createElement("div");
    box.className = "codeblock";
    const head = document.createElement("div");
    head.className = "cb-head";
    const isMatlab = ["matlab", "octave", "m", ""].includes(b.lang);
    const label = document.createElement("span");
    label.textContent = b.lang || "code";
    head.appendChild(label);
    const mk = (text, title, fn) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn sm ghost";
      btn.textContent = text;
      btn.title = title;
      btn.disabled = incomplete;
      btn.addEventListener("click", fn);
      head.appendChild(btn);
      return btn;
    };
    mk("Copy", "Copy to the clipboard", async () => {
      try {
        await navigator.clipboard.writeText(b.code);
        this.hooks.toast("Copied.");
      } catch {
        this.hooks.toast("Couldn't access the clipboard.", "error");
      }
    });
    if (isMatlab) {
      mk("Insert into editor", "Insert at the cursor (replaces the selection)", () => this.hooks.insertCode(b.code));
      mk("Replace file", "Replace the whole current file with this code", () => this.hooks.replaceFile(b.code));
      mk("Run", "Run this code in the command window", () => this.hooks.runCode(b.code));
    }
    const pre = document.createElement("pre");
    pre.innerHTML = escapeHtml(b.code);
    box.append(head, pre);
    return box;
  }

  /* ---------------- settings dialog ---------------- */

  _bindSettings() {
    const $ = (id) => document.getElementById(id);
    const sel = $("ai-provider");
    sel.addEventListener("change", () => this._showProviderFields(sel.value));
    $("ollama-refresh").addEventListener("click", () => this._refreshOllama());
    $("ai-forget").addEventListener("click", () => {
      $("anthropic-key").value = "";
      $("openai-key").value = "";
      this.settings.anthropic.key = "";
      this.settings.openai.key = "";
      saveSettings(this.settings);
      this.hooks.toast("Saved keys removed from this browser.");
    });
    this.el.dlg.addEventListener("close", () => {
      if (this.el.dlg.returnValue === "save") this._saveFromForm();
    });
    const origin = location.origin.startsWith("http") ? location.origin : "*";
    $("ollama-cmd").textContent = `OLLAMA_ORIGINS="${origin}" ollama serve`;
    $("ollama-cmd-win").textContent = `$env:OLLAMA_ORIGINS="${origin}"; ollama serve`;
  }

  _showProviderFields(p) {
    document.querySelectorAll(".provider-fields").forEach((f) => (f.hidden = f.dataset.provider !== p));
    document.getElementById("key-warning").hidden = !(p === "anthropic" || p === "openai");
  }

  openSettings() {
    const s = this.settings;
    const $ = (id) => document.getElementById(id);
    $("ai-provider").value = s.provider;
    $("ollama-url").value = s.ollama.url;
    $("ollama-model").value = s.ollama.model;
    $("anthropic-key").value = s.anthropic.key;
    $("anthropic-model").value = s.anthropic.model;
    $("openai-url").value = s.openai.url;
    $("openai-model").value = s.openai.model;
    $("openai-key").value = s.openai.key;
    $("ollama-status").textContent = "";
    this._showProviderFields(s.provider);
    this.el.dlg.returnValue = "";
    this.el.dlg.showModal();
    if (s.provider === "ollama" && !s.ollama.model) this._refreshOllama();
  }

  async _refreshOllama() {
    const $ = (id) => document.getElementById(id);
    const status = $("ollama-status");
    status.textContent = "Looking for Ollama…";
    try {
      const models = await listOllamaModels($("ollama-url").value);
      const dl = $("ollama-models");
      dl.innerHTML = "";
      for (const m of models) {
        const o = document.createElement("option");
        o.value = m;
        dl.appendChild(o);
      }
      if (models.length && !$("ollama-model").value) $("ollama-model").value = models[0];
      status.textContent = models.length
        ? `Found ${models.length} model${models.length > 1 ? "s" : ""}: ${models.slice(0, 6).join(", ")}${models.length > 6 ? "…" : ""}`
        : "Ollama is running but has no models. Run: ollama pull qwen2.5-coder";
    } catch {
      status.textContent = "Couldn't reach Ollama. Is it running and started with OLLAMA_ORIGINS (see below)?";
    }
  }

  _saveFromForm() {
    const $ = (id) => document.getElementById(id);
    this.settings = {
      provider: $("ai-provider").value,
      ollama: { url: $("ollama-url").value.trim() || DEFAULT_SETTINGS.ollama.url, model: $("ollama-model").value.trim() },
      anthropic: { key: $("anthropic-key").value.trim(), model: $("anthropic-model").value.trim() || DEFAULT_SETTINGS.anthropic.model },
      openai: {
        url: $("openai-url").value.trim() || DEFAULT_SETTINGS.openai.url,
        model: $("openai-model").value.trim(),
        key: $("openai-key").value.trim(),
      },
    };
    const ok = saveSettings(this.settings);
    this._updateLabel();
    if (!this.history.length && !this.controller) {
      this.el.messages.innerHTML = "";
      this._welcome();
    }
    this.hooks.toast(ok ? "Assistant settings saved." : "Settings apply to this visit only (this browser blocks local storage).");
  }
}
