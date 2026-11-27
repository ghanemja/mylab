#!/usr/bin/env node
// A signalling server for mylab's live collaboration, compatible with y-webrtc's: browsers subscribe
// to a topic and every message published to it is passed on to everyone subscribed. It only
// introduces browsers to each other. What they publish is encrypted with the key from the session's
// link, so the server sees topics (hashes), IP addresses, sizes and times, never content.
//
// Plain Node.js, no packages: a small WebSocket server (RFC 6455, text frames).
//
//   node scripts/signaling_server.mjs [--port 4444] [--host 127.0.0.1]
//
// Then put ws://<host>:<port> (wss:// behind a TLS proxy) under Settings → Live collaboration.
// scripts/test_web_live.mjs starts one for the browser tests (startSignalingServer below).

import http from "node:http";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

const GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
const MAX_MESSAGE = 1024 * 1024; // signalling messages are small; anything bigger closes the connection
const MAX_TOPICS = 64; // per connection

function frame(text) {
  const data = Buffer.from(text, "utf8");
  const n = data.length;
  const head = n < 126 ? Buffer.from([0x81, n]) : n < 65536 ? Buffer.from([0x81, 126, n >> 8, n & 255]) : Buffer.concat([Buffer.from([0x81, 127]), (() => { const b = Buffer.alloc(8); b.writeBigUInt64BE(BigInt(n)); return b; })()]);
  return Buffer.concat([head, data]);
}

/**
 * Start the server. onMessage(message, info) is called with every message a client sends (the tests
 * check that nothing readable passes through). Resolves to { port, url, close(), topics }.
 */
export function startSignalingServer({ port = 0, host = "127.0.0.1", onMessage = null, quiet = true } = {}) {
  const topics = new Map(); // topic -> Set of connections
  const conns = new Set();

  const server = http.createServer((req, res) => {
    res.writeHead(200, { "Content-Type": "text/plain" });
    res.end("mylab signalling server\n");
  });

  server.on("upgrade", (req, socket) => {
    const key = req.headers["sec-websocket-key"];
    if (!key || String(req.headers.upgrade).toLowerCase() !== "websocket") return socket.destroy();
    const accept = crypto.createHash("sha1").update(key + GUID).digest("base64");
    socket.write(`HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`);
    socket.setNoDelay(true);
    const conn = { socket, topics: new Set(), closed: false, address: req.socket.remoteAddress };
    conns.add(conn);
    let buf = Buffer.alloc(0);
    let parts = [];

    const send = (msg) => {
      if (!conn.closed) socket.write(frame(JSON.stringify(msg)));
    };
    conn.send = send;
    const close = () => {
      if (conn.closed) return;
      conn.closed = true;
      for (const t of conn.topics) {
        const subs = topics.get(t);
        if (subs) {
          subs.delete(conn);
          if (!subs.size) topics.delete(t);
        }
      }
      conns.delete(conn);
      socket.destroy();
    };

    const handle = (text) => {
      let msg;
      try {
        msg = JSON.parse(text);
      } catch {
        return;
      }
      if (!msg || typeof msg.type !== "string") return;
      if (onMessage) onMessage(JSON.parse(text), { address: conn.address }); // as the client sent it
      if (msg.type === "subscribe") {
        for (const t of Array.isArray(msg.topics) ? msg.topics : []) {
          if (typeof t !== "string" || t.length > 200 || conn.topics.size >= MAX_TOPICS) continue;
          if (!topics.has(t)) topics.set(t, new Set());
          topics.get(t).add(conn);
          conn.topics.add(t);
        }
      } else if (msg.type === "unsubscribe") {
        for (const t of Array.isArray(msg.topics) ? msg.topics : []) {
          const subs = topics.get(t);
          if (subs) {
            subs.delete(conn);
            if (!subs.size) topics.delete(t);
          }
          conn.topics.delete(t);
        }
      } else if (msg.type === "publish" && typeof msg.topic === "string") {
        const subs = topics.get(msg.topic);
        if (subs) {
          msg.clients = subs.size;
          for (const c of subs) c.send(msg);
        }
      } else if (msg.type === "ping") {
        send({ type: "pong" });
      }
    };

    socket.on("data", (chunk) => {
      buf = Buffer.concat([buf, chunk]);
      for (;;) {
        if (buf.length < 2) return;
        const fin = buf[0] & 0x80;
        const opcode = buf[0] & 0x0f;
        const masked = buf[1] & 0x80;
        let len = buf[1] & 0x7f;
        let at = 2;
        if (len === 126) {
          if (buf.length < 4) return;
          len = buf.readUInt16BE(2);
          at = 4;
        } else if (len === 127) {
          if (buf.length < 10) return;
          const big = buf.readBigUInt64BE(2);
          if (big > BigInt(MAX_MESSAGE)) return close();
          len = Number(big);
          at = 10;
        }
        if (len > MAX_MESSAGE || !masked) return close(); // clients must mask their frames
        if (buf.length < at + 4 + len) return;
        const mask = buf.subarray(at, at + 4);
        const data = Buffer.from(buf.subarray(at + 4, at + 4 + len));
        for (let i = 0; i < data.length; i++) data[i] ^= mask[i & 3];
        buf = buf.subarray(at + 4 + len);
        if (opcode === 0x8) return close();
        if (opcode === 0x9) {
          socket.write(Buffer.concat([Buffer.from([0x8a, data.length]), data.subarray(0, 125)]));
          continue;
        }
        if (opcode === 0xa) continue;
        if (opcode === 0x1 || opcode === 0x2 || opcode === 0x0) {
          parts.push(data);
          if (parts.reduce((n, p) => n + p.length, 0) > MAX_MESSAGE) return close();
          if (fin) {
            const text = Buffer.concat(parts).toString("utf8");
            parts = [];
            handle(text);
          }
        }
      }
    });
    socket.on("error", close);
    socket.on("close", close);
  });

  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, host, () => {
      const p = server.address().port;
      if (!quiet) console.log(`mylab signalling server on ws://${host}:${p}`);
      resolve({
        port: p,
        url: `ws://${host}:${p}`,
        topics,
        close: () =>
          new Promise((r) => {
            for (const c of [...conns]) c.socket.destroy();
            server.close(() => r());
          }),
      });
    });
  });
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const arg = (name, fallback) => {
    const i = process.argv.indexOf(name);
    return i > 0 ? process.argv[i + 1] : fallback;
  };
  startSignalingServer({ port: Number(arg("--port", process.env.PORT || 4444)), host: arg("--host", "127.0.0.1"), quiet: false }).catch((e) => {
    console.error(e.message);
    process.exit(1);
  });
}
