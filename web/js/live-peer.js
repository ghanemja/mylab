// WebRTC peers for live collaboration that can send messages of any size.
//
// A data channel refuses a message bigger than the limit the other browser announces (256 KiB in
// Chromium and Safari), and a full send buffer closes the channel. y-webrtc sends the whole shared
// project to a newcomer in one message and does not catch either case (the message is lost and the
// newcomer never gets in sync), so the peers it creates (see scripts/vendor.mjs, setPeerClass)
// split each message into frames and send them only while the channel's buffer has room.
//
// A frame starts with one byte: 0 = a whole message, 1 = the first part of a longer one (followed by
// its total length as 4 bytes), 2 = a further part. The channel is reliable and ordered, and frames
// are queued one message after the other, so the parts of two messages never mix.

const FRAME = 60 * 1024; // safe in every browser
const HIGH_WATER = 1024 * 1024; // stop sending while this much waits in the channel's buffer
export const MAX_MESSAGE = 48 * 1024 * 1024; // a peer may not make us collect more than this

/** A subclass of simple-peer's Peer (the class y-webrtc uses) with framing and flow control. */
export function chunkedPeerClass(SimplePeer) {
  return class ChunkedPeer extends SimplePeer {
    constructor(opts) {
      super(opts);
      this._frames = [];
      this._pumpTimer = null;
      this._incoming = null; // { buf, filled }
      // A connection that fails ends with an "error" event (y-webrtc then looks for the peer again). An
      // event emitter throws an "error" that no one listens to: make sure a late one never does.
      this.on("error", () => {});
    }

    send(data) {
      if (this.destroying || this.destroyed) return super.send(data); // simple-peer reports this
      if (!this._channel || this._channel.readyState !== "open") throw new Error("The connection is not open yet.");
      const bytes = data instanceof Uint8Array ? data : typeof data === "string" ? new TextEncoder().encode(data) : new Uint8Array(data);
      if (bytes.length + 1 <= FRAME) {
        const f = new Uint8Array(bytes.length + 1);
        f.set(bytes, 1);
        this._frames.push(f);
      } else {
        const first = new Uint8Array(FRAME);
        first[0] = 1;
        new DataView(first.buffer).setUint32(1, bytes.length);
        first.set(bytes.subarray(0, FRAME - 5), 5);
        this._frames.push(first);
        for (let at = FRAME - 5; at < bytes.length; at += FRAME - 1) {
          const part = bytes.subarray(at, Math.min(bytes.length, at + FRAME - 1));
          const f = new Uint8Array(part.length + 1);
          f[0] = 2;
          f.set(part, 1);
          this._frames.push(f);
        }
      }
      this._pump();
    }

    _pump() {
      clearTimeout(this._pumpTimer);
      this._pumpTimer = null;
      const ch = this._channel;
      while (this._frames.length && !this.destroyed && !this.destroying && ch && ch.readyState === "open") {
        if (ch.bufferedAmount > HIGH_WATER) {
          // "bufferedamountlow" brings us back; the timer covers a browser that misses it
          this._pumpTimer = setTimeout(() => this._pump(), 100);
          return;
        }
        try {
          ch.send(this._frames.shift());
        } catch {
          // The other side closed the channel a moment ago (WebKit still says "open" then). The
          // connection is over: close it without an "error", which y-webrtc would only log.
          this._frames = [];
          this.destroy();
          return;
        }
      }
      if (!ch || ch.readyState !== "open") this._frames = [];
    }

    _onChannelBufferedAmountLow() {
      super._onChannelBufferedAmountLow();
      this._pump();
    }

    _onChannelMessage(event) {
      if (this.destroyed) return;
      const f = new Uint8Array(event.data instanceof ArrayBuffer ? event.data : new TextEncoder().encode(String(event.data)));
      if (!f.length) return;
      if (f[0] === 0) {
        this._incoming = null;
        return super._onChannelMessage({ data: f.slice(1).buffer });
      }
      if (f[0] === 1 && f.length > 5) {
        const total = new DataView(f.buffer, f.byteOffset).getUint32(1);
        if (total > MAX_MESSAGE) {
          this._incoming = null;
          return this.destroy(new Error("A peer sent a message that is too large."));
        }
        this._incoming = { buf: new Uint8Array(total), filled: 0 };
        return this._take(f.subarray(5));
      }
      if (f[0] === 2 && this._incoming) return this._take(f.subarray(1));
      this._incoming = null; // not a frame of ours: ignore it
    }

    _take(part) {
      const m = this._incoming;
      if (m.filled + part.length > m.buf.length) {
        this._incoming = null;
        return;
      }
      m.buf.set(part, m.filled);
      m.filled += part.length;
      if (m.filled === m.buf.length) {
        this._incoming = null;
        super._onChannelMessage({ data: m.buf.buffer });
      }
    }

    destroy(err) {
      clearTimeout(this._pumpTimer);
      this._frames = [];
      this._incoming = null;
      return super.destroy(err);
    }
  };
}
