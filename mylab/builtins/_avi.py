"""AVI video files in pure Python: Motion JPEG and uncompressed frames, written and read.

An AVI file is a RIFF container: a header list ('hdrl') with the main header and one stream
header and format per stream, a 'movi' list with one chunk per frame ('00dc' compressed,
'00db' uncompressed), and an index ('idx1') of where the frames are. JPEG coding is Pillow's.
"""

from __future__ import annotations

import io
import struct
from fractions import Fraction
from typing import Any, Iterable

import numpy as np

from ..errors import MatlabError


def _chunk(tag: bytes, data: bytes) -> bytes:
    return tag + struct.pack("<I", len(data)) + data + (b"\0" if len(data) % 2 else b"")


def _list(kind: bytes, data: bytes) -> bytes:
    return b"LIST" + struct.pack("<I", len(data) + 4) + kind + data


def write(path: str, frames: Iterable[np.ndarray], shape: tuple, fps: float, quality: float = 75.0,
          mode: str = "mjpeg") -> int:
    """Write RGB frames (height-by-width-by-3 uint8). ``mode`` is 'mjpeg', 'raw' or 'gray'. Returns the count."""
    from PIL import Image
    h, w = int(shape[0]), int(shape[1])
    rate = Fraction(float(fps)).limit_denominator(100000)
    row = (w * 3 + 3) & ~3
    chunks: list[tuple[bytes, int]] = []  # tag and size of each frame, for the index
    parts: list[bytes] = []
    biggest = 0
    for f in frames:
        f = np.asarray(f, dtype=np.uint8)
        if mode == "mjpeg":
            buf = io.BytesIO()
            Image.fromarray(f).save(buf, "JPEG", quality=int(min(max(round(quality), 1), 100)), subsampling=0
                                    if quality >= 90 else 2)
            data, tag = buf.getvalue(), b"00dc"
        elif mode == "gray":
            gray = f if f.ndim == 2 else np.round(f[..., 0] * 0.2989 + f[..., 1] * 0.587 + f[..., 2] * 0.114)
            rows = np.zeros((h, (w + 3) & ~3), dtype=np.uint8)
            rows[:, :w] = gray.astype(np.uint8)[::-1]
            data, tag = rows.tobytes(), b"00db"
        else:
            rows = np.zeros((h, row), dtype=np.uint8)
            rows[:, :w * 3] = f[::-1, :, ::-1].reshape(h, w * 3)  # bottom-up rows of blue-green-red
            data, tag = rows.tobytes(), b"00db"
        chunks.append((tag, len(data)))
        parts.append(_chunk(tag, data))
        biggest = max(biggest, len(data))
    n = len(chunks)
    compression = b"MJPG" if mode == "mjpeg" else b"\0\0\0\0"
    bits = 8 if mode == "gray" else 24
    image_size = biggest if mode == "mjpeg" else (h * ((w + 3) & ~3) if mode == "gray" else h * row)
    strf = struct.pack("<IiiHH4sIiiII", 40, w, h, 1, bits, compression, image_size, 0, 0,
                       256 if mode == "gray" else 0, 0)
    if mode == "gray":
        strf += b"".join(bytes([k, k, k, 0]) for k in range(256))
    strh = struct.pack("<4s4sIHHIIIIIIII4h", b"vids", b"MJPG" if mode == "mjpeg" else b"DIB ", 0, 0, 0, 0,
                       rate.denominator, rate.numerator, 0, n, biggest, 0xFFFFFFFF, 0, 0, 0, w, h)
    avih = struct.pack("<14I", int(round(1e6 / float(fps))), int(biggest * float(fps)), 0, 0x10, n, 0, 1, biggest,
                       w, h, 0, 0, 0, 0)
    hdrl = _list(b"hdrl", _chunk(b"avih", avih) + _list(b"strl", _chunk(b"strh", strh) + _chunk(b"strf", strf)))
    movi_body = b"".join(parts)
    index = []
    offset = 4  # offsets in the index count from the 'movi' tag
    for (tag, size), part in zip(chunks, parts):
        index.append(tag + struct.pack("<III", 0x10, offset, size))
        offset += len(part)
    body = b"AVI " + hdrl + _list(b"movi", movi_body) + _chunk(b"idx1", b"".join(index))
    with open(path, "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", len(body)) + body)
    return n


class AviFile:
    """The video stream of an AVI file: frame positions and how to decode them."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as fh:
            head = fh.read(12)
            if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"AVI ":
                raise MatlabError(f"'{path}' is not an AVI file.")
            self.data = head + fh.read()
        self.width = self.height = 0
        self.bits = 24
        self.compression = b""
        self.palette: np.ndarray | None = None
        self.rate = 0.0
        self.micro = 0
        self.frames: list[tuple[int, int]] = []  # offset and size of each frame's data
        self.stream = -1
        self._streams = 0
        self._pending = False  # the stream header just read is the video stream: its format follows
        self._walk(12, len(self.data), None)
        if self.stream < 0:
            raise MatlabError(f"'{path}' has no video stream.")
        if not self.rate and self.micro:
            self.rate = 1e6 / self.micro

    def _walk(self, pos: int, end: int, parent: bytes | None) -> None:
        d = self.data
        while pos + 8 <= end:
            tag = d[pos:pos + 4]
            size = struct.unpack_from("<I", d, pos + 4)[0]
            body = pos + 8
            if tag in (b"LIST", b"RIFF"):  # further RIFF chunks hold the frames of long files
                kind = d[body:body + 4]
                self._walk(body + 4, min(body + size, end), kind)
            elif tag == b"avih" and size >= 40:
                self.micro = struct.unpack_from("<I", d, body)[0]
            elif tag == b"strh" and size >= 48:
                if d[body:body + 4] == b"vids" and self.stream < 0:
                    self.stream = self._streams
                    scale, rate = struct.unpack_from("<II", d, body + 20)
                    self.rate = rate / scale if scale else 0.0
                    self._pending = True
                else:
                    self._pending = False
                self._streams += 1
            elif tag == b"strf" and self._pending:
                self._pending = False
                (_, w, h, _, bits, comp) = struct.unpack_from("<IiiHH4s", d, body)
                self.width, self.height, self.bits, self.compression = w, h, bits, comp
                if bits <= 8:
                    count = struct.unpack_from("<I", d, body + 32)[0] or (1 << bits)
                    pal = np.frombuffer(d, dtype=np.uint8, count=4 * count, offset=body + 40).reshape(count, 4)
                    self.palette = pal[:, 2::-1].copy()
            elif parent in (b"movi", b"rec ") and len(tag) == 4 and tag[2:] in (b"dc", b"db") \
                    and tag[:2].isdigit() and int(tag[:2]) == max(self.stream, 0):
                if size:
                    self.frames.append((body, size))
                elif self.frames:
                    self.frames.append(self.frames[-1])  # an empty chunk repeats the frame before it
            pos = body + size + (size & 1)

    def frame(self, k: int) -> np.ndarray:
        """Frame k (from zero) as a height-by-width-by-3 uint8 array (height-by-width when grayscale)."""
        off, size = self.frames[k]
        raw = self.data[off:off + size]
        h, w = abs(self.height), self.width
        comp = self.compression.upper()
        if comp in (b"MJPG", b"JPEG", b"AVRN", b"LJPG") or raw[:2] == b"\xff\xd8":
            from PIL import Image
            try:
                img = Image.open(io.BytesIO(raw))
                img.load()
            except Exception as e:  # Pillow raises several types for damaged frames
                raise MatlabError(f"Frame {k + 1} of '{self.path}' cannot be decoded ({e}).") from None
            return np.asarray(img.convert("RGB") if img.mode != "L" else img)
        if comp in (b"\0\0\0\0", b"DIB ", b"RAW ", b"RGB "):
            if self.bits == 24:
                row = (w * 3 + 3) & ~3
                a = np.frombuffer(raw, dtype=np.uint8, count=h * row).reshape(h, row)[:, :w * 3].reshape(h, w, 3)
                out = a[:, :, ::-1]
            elif self.bits == 32:
                a = np.frombuffer(raw, dtype=np.uint8, count=h * w * 4).reshape(h, w, 4)
                out = a[:, :, 2::-1]
            elif self.bits == 8:
                row = (w + 3) & ~3
                a = np.frombuffer(raw, dtype=np.uint8, count=h * row).reshape(h, row)[:, :w]
                pal = self.palette
                gray = pal is None or bool(np.all(pal[:, 0] == pal[:, 1]) and np.all(pal[:, 1] == pal[:, 2]))
                out = a if pal is None else (pal[a][..., 0] if gray else pal[a])
            else:
                raise MatlabError(f"Uncompressed AVI frames with {self.bits} bits per pixel are not supported.")
            return np.ascontiguousarray(out[::-1] if self.height > 0 else out)  # stored bottom row first
        raise UnsupportedCodec(self.compression.decode("latin-1").strip() or "unknown")


class UnsupportedCodec(Exception):
    pass


def decode_with_ffmpeg(path: str, exe: str) -> tuple[Any, float]:
    """All frames of a video as a memory-mapped frames-by-height-by-width-by-3 array, and the frame rate."""
    import re
    import subprocess
    import tempfile
    try:
        probe = subprocess.run([exe, "-hide_banner", "-i", path], stdin=subprocess.DEVNULL, capture_output=True,
                               text=True, errors="replace", timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        raise MatlabError(f"VideoReader: ffmpeg could not open '{path}' ({e}).") from None
    info = probe.stderr
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", info)
    if not m:
        raise MatlabError(f"VideoReader: '{path}' has no video stream that ffmpeg can read.")
    w, h = int(m.group(1)), int(m.group(2))
    fps = 0.0
    mf = re.search(r"([\d.]+)\s*fps", info) or re.search(r"([\d.]+)\s*tbr", info)
    if mf:
        fps = float(mf.group(1))
    tmp = tempfile.NamedTemporaryFile(suffix=".rgb", delete=False)
    tmp.close()
    try:
        run = subprocess.run([exe, "-y", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24",
                              tmp.name], stdin=subprocess.DEVNULL, capture_output=True, timeout=3600)
    except (OSError, subprocess.SubprocessError) as e:
        raise MatlabError(f"VideoReader: ffmpeg could not decode '{path}' ({e}).") from None
    import os
    size = os.path.getsize(tmp.name)
    if run.returncode != 0 or size < w * h * 3:
        os.remove(tmp.name)
        raise MatlabError(f"VideoReader: ffmpeg could not decode '{path}'.")
    n = size // (w * h * 3)
    return np.memmap(tmp.name, dtype=np.uint8, mode="r", shape=(n, h, w, 3)), fps or 30.0
