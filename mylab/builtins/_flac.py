"""FLAC audio in pure Python and NumPy, used when the optional soundfile package is not installed.

The format (published by Xiph.Org): the marker ``fLaC``, metadata blocks (the first, STREAMINFO,
gives rate, channels, sample size and length), then frames. A frame codes each channel as a
subframe: a constant, the plain samples, or a predictor (a fixed polynomial of order 0 to 4, or
linear prediction) followed by its residuals in Rice code.

The encoder here uses the fixed predictors with one Rice parameter per subframe. The decoder
reads every kind of subframe; linear prediction, which other encoders use, is restored by a
plain Python loop and is therefore slow for long recordings.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np

from ..errors import MatlabError

BLOCK = 4096


def _crc_table(poly: int, bits: int) -> np.ndarray:
    top = 1 << (bits - 1)
    mask = (1 << bits) - 1
    out = []
    for i in range(256):
        c = i << (bits - 8)
        for _ in range(8):
            c = ((c << 1) ^ poly) & mask if c & top else (c << 1) & mask
        out.append(c)
    return np.array(out, dtype=np.uint32)


_CRC8 = _crc_table(0x07, 8)
_CRC16 = _crc_table(0x8005, 16)


def _crc8(data: bytes) -> int:
    c = 0
    for b in data:
        c = int(_CRC8[c ^ b])
    return c


def _crc16(data: bytes) -> int:
    c = 0
    for b in data:
        c = ((c << 8) & 0xFFFF) ^ int(_CRC16[(c >> 8) ^ b])
    return c


def _utf8_number(n: int) -> bytes:
    """The frame number in the extended UTF-8 coding FLAC uses (up to 36 bits)."""
    if n < 0x80:
        return bytes([n])
    out = []
    count = 1
    while n >= (1 << (6 - count)):
        out.append(0x80 | (n & 0x3F))
        n >>= 6
        count += 1
    lead = ((0xFF << (8 - count)) & 0xFF) | n
    return bytes([lead] + out[::-1])


def _bits_of(values: np.ndarray, width: int) -> np.ndarray:
    """Each value as ``width`` bits, most significant first (two's complement), as a flat 0/1 array."""
    v = values.astype(np.int64) & ((1 << width) - 1)
    shifts = np.arange(width - 1, -1, -1, dtype=np.int64)
    return ((v[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)


def _rice_bits(res: np.ndarray, k: int) -> np.ndarray:
    u = ((res << 1) ^ (res >> 63)).astype(np.int64)  # zigzag: 0, -1, 1, -2, ... -> 0, 1, 2, 3, ...
    q = u >> k
    lengths = q + 1 + k
    ends = np.cumsum(lengths)
    starts = ends - lengths
    out = np.zeros(int(ends[-1]) if ends.size else 0, dtype=np.uint8)
    stop = starts + q
    out[stop] = 1
    for j in range(k):
        out[stop + 1 + j] = (u >> (k - 1 - j)) & 1
    return out


def _subframe(x: np.ndarray, bps: int) -> np.ndarray:
    """The bits of one subframe for the samples x (int64)."""
    n = x.size
    if np.all(x == x[0]):
        return np.concatenate([np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8), _bits_of(x[:1], bps)])
    best = None
    res = x
    for order in range(0, 5):
        if order:
            res = np.diff(res)
        if n <= order:
            break
        r = res if order == 0 else res
        u = (r << 1) ^ (r >> 63)
        total = int(u.sum())
        mean = total / max(r.size, 1)
        k0 = max(int(np.floor(np.log2(mean))) if mean >= 1 else 0, 0)
        for k in {max(k0 - 1, 0), k0, min(k0 + 1, 30)}:
            if k > 30:
                continue
            cost = r.size * (k + 1) + int((u >> k).sum()) + order * bps
            if best is None or cost < best[0]:
                best = (cost, order, k, r)
    cost, order, k, r = best
    if cost + 10 >= n * bps:  # no gain over the plain samples
        return np.concatenate([np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=np.uint8), _bits_of(x, bps)])
    head = np.array([0, 0, 0, 1, (order >> 2) & 1, (order >> 1) & 1, order & 1, 0], dtype=np.uint8)
    if k < 15:
        method = np.array([0, 0], dtype=np.uint8)
        param = _bits_of(np.array([k]), 4)
    else:
        method = np.array([0, 1], dtype=np.uint8)
        param = _bits_of(np.array([k]), 5)
    return np.concatenate([head, _bits_of(x[:order], bps), method, np.zeros(4, dtype=np.uint8), param,
                           _rice_bits(r.astype(np.int64), k)])


def encode(samples: np.ndarray, rate: int, bps: int) -> bytes:
    """A FLAC file for integer samples (frames by channels) with ``bps`` bits each."""
    samples = np.asarray(samples, dtype=np.int64)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    n, channels = samples.shape
    if not 1 <= channels <= 8:
        raise MatlabError("A FLAC file holds between 1 and 8 channels.")
    if not 1 <= rate < 1 << 20:
        raise MatlabError("The sample rate of a FLAC file must be below 1048576 Hz.")
    frames = []
    sizes = []
    for number, start in enumerate(range(0, n, BLOCK)):
        block = samples[start:start + BLOCK]
        size = block.shape[0]
        header = bytes([0xFF, 0xF8, 0x70, ((channels - 1) << 4)]) + _utf8_number(number) + \
            struct.pack(">H", size - 1)
        header += bytes([_crc8(header)])
        bits = np.concatenate([_subframe(np.ascontiguousarray(block[:, c]), bps) for c in range(channels)])
        body = np.packbits(bits).tobytes()
        frame = header + body
        frame += struct.pack(">H", _crc16(frame))
        frames.append(frame)
        sizes.append(len(frame))
    width = (bps + 7) // 8
    raw = samples.astype("<i4").reshape(-1)
    pcm = raw.astype("<i2").tobytes() if width == 2 else raw.astype("<i1").tobytes() if width == 1 else \
        raw.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :width].tobytes()
    block_size = BLOCK if n >= BLOCK else max(n, 16)
    info = struct.pack(">HH", block_size, block_size)
    info += (min(sizes) if sizes else 0).to_bytes(3, "big") + (max(sizes) if sizes else 0).to_bytes(3, "big")
    info += ((rate << 44) | ((channels - 1) << 41) | ((bps - 1) << 36) | n).to_bytes(8, "big")
    info += hashlib.md5(pcm).digest()
    # the last frame may be shorter than the stream's block size: its header carries its own size
    return b"fLaC" + bytes([0x80]) + len(info).to_bytes(3, "big") + info + b"".join(frames)


# -- decoding ------------------------------------------------------------------------------------

class _Bits:
    def __init__(self, data: bytes):
        self.bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        self.pos = 0

    def read(self, n: int) -> int:
        if n == 0:
            return 0
        chunk = self.bits[self.pos:self.pos + n]
        if chunk.size < n:
            raise MatlabError("The FLAC file ends in the middle of a frame.")
        self.pos += n
        value = 0
        for b in chunk.tolist():
            value = (value << 1) | b
        return value

    def signed(self, n: int) -> int:
        v = self.read(n)
        return v - (1 << n) if n and v >> (n - 1) else v

    def many(self, count: int, width: int) -> np.ndarray:
        """``count`` signed values of ``width`` bits each."""
        chunk = self.bits[self.pos:self.pos + count * width]
        if chunk.size < count * width:
            raise MatlabError("The FLAC file ends in the middle of a frame.")
        self.pos += count * width
        weights = (1 << np.arange(width - 1, -1, -1, dtype=np.int64))
        v = chunk.reshape(count, width).astype(np.int64) @ weights
        return np.where(v >> (width - 1), v - (1 << width), v) if width else v

    def unary(self) -> int:
        window = self.bits[self.pos:self.pos + 64]
        hit = np.flatnonzero(window)
        while hit.size == 0:
            if window.size == 0:
                raise MatlabError("The FLAC file ends in the middle of a frame.")
            self.pos += window.size
            window = self.bits[self.pos:self.pos + 4096]
            hit = np.flatnonzero(window)
        q = int(hit[0])
        self.pos += q + 1
        return q


def _residual(r: _Bits, block: int, order: int) -> np.ndarray:
    method = r.read(2)
    if method > 1:
        raise MatlabError("The FLAC file uses a residual coding that is not defined.")
    width = 4 if method == 0 else 5
    parts = 1 << r.read(4)
    out = np.empty(block - order, dtype=np.int64)
    done = 0
    for p in range(parts):
        count = (block // parts) - (order if p == 0 else 0)
        k = r.read(width)
        if k == (1 << width) - 1:  # escape: plain samples of a given width
            raw_bits = r.read(5)
            out[done:done + count] = r.many(count, raw_bits) if raw_bits else 0
            done += count
            continue
        bits = r.bits
        # find where each code's run of zeros stops: walk the positions of the one-bits
        end_guess = min(bits.size, r.pos + count * (k + 40) + 4096)
        ones = np.flatnonzero(bits[r.pos:end_guess]) + r.pos
        stops = np.empty(count, dtype=np.int64)
        pos = r.pos
        i = 0
        total = ones.size
        ones_list = ones.tolist()
        for c in range(count):
            while i < total and ones_list[i] < pos:
                i += 1
            if i >= total:  # the window was too short: extend it
                more = np.flatnonzero(bits[end_guess:]) + end_guess
                if more.size == 0:
                    raise MatlabError("The FLAC file ends in the middle of a frame.")
                ones_list = ones_list[i:] + more.tolist()
                end_guess = bits.size
                total = len(ones_list)
                i = 0
                while i < total and ones_list[i] < pos:
                    i += 1
            stop = ones_list[i]
            stops[c] = stop
            pos = stop + 1 + k
            i += 1
        starts = np.concatenate([[r.pos], stops[:-1] + 1 + k])
        q = stops - starts
        u = q << k
        for j in range(k):
            u |= bits[stops + 1 + j].astype(np.int64) << (k - 1 - j)
        out[done:done + count] = (u >> 1) ^ -(u & 1)
        done += count
        r.pos = pos
    return out


def _decode_subframe(r: _Bits, block: int, bps: int) -> np.ndarray:
    if r.read(1):
        raise MatlabError("The FLAC file is damaged (bad subframe header).")
    kind = r.read(6)
    wasted = 0
    if r.read(1):
        wasted = r.unary() + 1
        bps -= wasted
    if kind == 0:
        out = np.full(block, r.signed(bps), dtype=np.int64)
    elif kind == 1:
        out = r.many(block, bps)
    elif 8 <= kind <= 12:
        order = kind - 8
        warm = r.many(order, bps) if order else np.zeros(0, dtype=np.int64)
        res = _residual(r, block, order)
        out = np.empty(block, dtype=np.int64)
        out[:order] = warm
        # undo the order-th difference: integrate once per order, starting from the warm-up values
        levels = [warm]
        for _ in range(order):
            levels.append(np.diff(levels[-1]))
        seq = res
        for j in range(order - 1, -1, -1):
            seq = levels[j][-1] + np.cumsum(seq)
        out[order:] = seq
    elif kind >= 32:
        order = (kind & 31) + 1
        warm = r.many(order, bps)
        precision = r.read(4) + 1
        if precision == 16:
            raise MatlabError("The FLAC file is damaged (bad predictor precision).")
        shift = r.signed(5)
        coefs = [int(c) for c in r.many(order, precision)]
        res = _residual(r, block, order).tolist()
        x = [int(v) for v in warm]
        for n in range(order, block):
            acc = 0
            for j in range(order):
                acc += coefs[j] * x[n - 1 - j]
            x.append(res[n - order] + (acc >> shift))
        out = np.array(x, dtype=np.int64)
    else:
        raise MatlabError("The FLAC file uses a subframe type that is not defined.")
    return out << wasted if wasted else out


_BLOCK_SIZES = {1: 192, 2: 576, 3: 1152, 4: 2304, 5: 4608, 8: 256, 9: 512, 10: 1024, 11: 2048, 12: 4096,
                13: 8192, 14: 16384, 15: 32768}
_SAMPLE_SIZES = {1: 8, 2: 12, 4: 16, 5: 20, 6: 24, 7: 32}


def read_info(data: bytes) -> dict:
    if data[:4] != b"fLaC":
        if data[:3] == b"ID3":  # an ID3 tag may come before the stream
            size = ((data[6] & 127) << 21) | ((data[7] & 127) << 14) | ((data[8] & 127) << 7) | (data[9] & 127)
            return read_info(data[10 + size:])
        raise MatlabError("The file is not a FLAC file.")
    pos = 4
    info = None
    while True:
        if pos + 4 > len(data):
            raise MatlabError("The FLAC file is damaged (metadata).")
        last = data[pos] >> 7
        kind = data[pos] & 127
        size = int.from_bytes(data[pos + 1:pos + 4], "big")
        if kind == 0:
            body = data[pos + 4:pos + 4 + size]
            packed = int.from_bytes(body[10:18], "big")
            info = {"rate": packed >> 44, "channels": ((packed >> 41) & 7) + 1, "bits": ((packed >> 36) & 31) + 1,
                    "total": packed & ((1 << 36) - 1), "block": int.from_bytes(body[2:4], "big")}
        pos += 4 + size
        if last:
            break
    if info is None:
        raise MatlabError("The FLAC file has no stream information.")
    info["audio"] = pos
    info["data"] = data
    return info


def decode(data: bytes) -> tuple[np.ndarray, int, int]:
    """(samples as frames by channels int64, sample rate, bits per sample)."""
    info = read_info(data)
    data = info["data"]
    channels, bps = info["channels"], info["bits"]
    r = _Bits(data[info["audio"]:])
    out = []
    got = 0
    total = info["total"]
    while r.pos + 16 <= r.bits.size and (total == 0 or got < total):
        if r.read(14) != 0x3FFE:
            if got:
                break  # padding or tags after the last frame
            raise MatlabError("The FLAC file is damaged (frame sync not found).")
        r.read(2)
        size_code = r.read(4)
        rate_code = r.read(4)
        assign = r.read(4)
        bits_code = r.read(3)
        r.read(1)
        lead = r.read(8)  # the frame or sample number: only its length matters here
        extra = 0
        while lead & 0x80 and extra < 7:
            lead = (lead << 1) & 0xFF
            extra += 1
        for _ in range(max(extra - 1, 0)):
            r.read(8)
        if size_code == 6:
            block = r.read(8) + 1
        elif size_code == 7:
            block = r.read(16) + 1
        elif size_code in _BLOCK_SIZES:
            block = _BLOCK_SIZES[size_code]
        else:
            raise MatlabError("The FLAC file is damaged (bad block size).")
        if rate_code == 12:
            r.read(8)
        elif rate_code in (13, 14):
            r.read(16)
        frame_bps = _SAMPLE_SIZES.get(bits_code, bps)
        r.read(8)  # header check value
        if assign < 8:
            if assign + 1 != channels:
                raise MatlabError("The FLAC file is damaged (channel count changes).")
            chans = [_decode_subframe(r, block, frame_bps) for _ in range(channels)]
        elif assign == 8:  # left and side
            left = _decode_subframe(r, block, frame_bps)
            side = _decode_subframe(r, block, frame_bps + 1)
            chans = [left, left - side]
        elif assign == 9:  # side and right
            side = _decode_subframe(r, block, frame_bps + 1)
            right = _decode_subframe(r, block, frame_bps)
            chans = [side + right, right]
        elif assign == 10:  # mid and side
            mid = _decode_subframe(r, block, frame_bps)
            side = _decode_subframe(r, block, frame_bps + 1)
            mid = (mid << 1) | (side & 1)
            chans = [(mid + side) >> 1, (mid - side) >> 1]
        else:
            raise MatlabError("The FLAC file is damaged (bad channel assignment).")
        r.pos = (r.pos + 7) & ~7
        r.read(16)  # frame check value
        out.append(np.stack(chans, axis=1))
        got += block
    if not out:
        return np.zeros((0, channels), dtype=np.int64), info["rate"], bps
    samples = np.concatenate(out)
    if total:
        samples = samples[:total]
    return samples, info["rate"], bps
