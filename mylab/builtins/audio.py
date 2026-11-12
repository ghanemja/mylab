"""Audio files: audioread, audiowrite, audioinfo (WAV), and sound/audioplayer stand-ins.

There is no sound device here, so sound, soundsc and audioplayer check their
arguments and play nothing.
"""

from __future__ import annotations

import os
import wave
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import Struct, char
from ._registry import builtin
from ._util import bool_scalar, fnum, fscalar, is_text, nargs, text, to_int

_WAV_ONLY = ("Only WAV files (.wav) are supported. Convert other formats first, for example with "
             "ffmpeg -i input.mp3 output.wav.")


def _check_file(name: str, what: str) -> str:
    if not os.path.isfile(name):
        raise MatlabError(f"{what}: the file '{name}' was not found.", "MATLAB:audiovideo:audioread:fileNotFound")
    ext = os.path.splitext(name)[1].lower()
    if ext not in (".wav", ".wave"):
        raise MatlabError(f"{what}: cannot read '{name}'. {_WAV_ONLY}")
    return name


def _read_wav(name: str, what: str) -> tuple[int, np.ndarray, int]:
    """Sample rate, samples (frames-by-channels, native type) and bits per sample."""
    import warnings

    from scipy.io import wavfile
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rate, data = wavfile.read(name)
    except (ValueError, EOFError) as e:
        raise MatlabError(f"{what}: '{name}' is not a readable WAV file ({e}).") from None
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    bits = data.dtype.itemsize * 8
    try:
        with wave.open(name, "rb") as w:
            bits = w.getsampwidth() * 8
    except (wave.Error, EOFError):
        pass  # floating-point WAV: the stdlib reader refuses it; the dtype tells the size
    return int(rate), data, bits


def _to_double(data: np.ndarray) -> np.ndarray:
    if data.dtype.kind == "f":
        return data.astype(float)
    if data.dtype == np.uint8:
        return (data.astype(float) - 128.0) / 128.0
    return data.astype(float) / float(2 ** (data.dtype.itemsize * 8 - 1))


@builtin("audioread")
def _audioread(interp, args, nargout):
    nargs(args, 1, 3, "audioread")
    name = _check_file(text(args[0], "file name"), "audioread")
    rate, data, bits = _read_wav(name, "audioread")
    native = False
    for a in args[1:]:
        if is_text(a):
            kind = text(a).lower()
            if kind not in ("native", "double"):
                raise MatlabError("audioread: the data type must be 'double' or 'native'.")
            native = kind == "native"
        else:
            rng = fnum(a).flatten()
            if rng.size != 2:
                raise MatlabError("audioread: the sample range must be [first last].")
            first = int(rng[0])
            last = data.shape[0] if np.isinf(rng[1]) else int(rng[1])
            if first < 1 or last > data.shape[0] or first > last:
                raise MatlabError(f"audioread: the sample range must lie within 1 and {data.shape[0]}.")
            data = data[first - 1:last]
    if native:
        if bits == 24:  # stored left-justified in 32 bits
            data = data >> 8
        y: np.ndarray = data.astype(np.float32) if data.dtype == np.float32 else data.copy()
    else:
        y = _to_double(data)
    return y, fscalar(rate)


@builtin("wavread")
def _wavread(interp, args, nargout):
    nargs(args, 1, 2, "wavread")
    name = text(args[0], "file name")
    if not os.path.splitext(name)[1]:
        name += ".wav"
    rate, data, bits = _read_wav(_check_file(name, "wavread"), "wavread")
    if len(args) > 1 and is_text(args[1]) and text(args[1]).lower() == "size":
        return np.array([[float(data.shape[0]), float(data.shape[1])]]), fscalar(rate), fscalar(bits)
    return _to_double(data), fscalar(rate), fscalar(bits)


@builtin("audioinfo")
def _audioinfo(interp, args, nargout):
    nargs(args, 1, 1, "audioinfo")
    name = _check_file(text(args[0], "file name"), "audioinfo")
    rate, data, bits = _read_wav(name, "audioinfo")
    empty = np.zeros((0, 0))
    return Struct.scalar({
        "Filename": char(os.path.abspath(name)), "CompressionMethod": char("Uncompressed"),
        "NumChannels": fscalar(data.shape[1]), "SampleRate": fscalar(rate),
        "TotalSamples": fscalar(data.shape[0]), "Duration": fscalar(data.shape[0] / rate if rate else 0.0),
        "Title": empty, "Comment": empty, "Artist": empty, "BitsPerSample": fscalar(bits)})


def _write_wav(interp, name: str, y: Any, rate: float, bits: int | None, what: str) -> None:
    if os.path.splitext(name)[1].lower() not in (".wav", ".wave"):
        raise MatlabError(f"{what}: cannot write '{name}'. {_WAV_ONLY}")
    if not isinstance(y, np.ndarray) or y.dtype.kind not in "fiu" or y.ndim > 2:
        raise MatlabError(f"{what}: the audio data must be a numeric matrix with one column per channel.")
    if rate <= 0 or rate != int(rate):
        raise MatlabError(f"{what}: the sample rate must be a positive integer (it is {rate:g}).")
    data = y.reshape(1, -1).T if y.shape[0] == 1 and y.shape[1] > 2 else y
    if data.shape[1] > 256:
        raise MatlabError(f"{what}: the data has {data.shape[1]} columns. Each column is a channel; "
                          "pass a column vector for mono sound.")
    if bits is None:
        bits = {"uint8": 8, "int16": 16, "int32": 32, "float32": 32, "float64": 16}.get(data.dtype.name, 16)
    if bits not in (8, 16, 24, 32, 64):
        raise MatlabError(f"{what}: BitsPerSample must be 8, 16, 24, 32 or 64.")
    if data.dtype.kind == "f":
        if (bits == 32 and data.dtype == np.float32) or bits == 64:
            from scipy.io import wavfile
            wavfile.write(name, int(rate), data.astype(np.float32 if bits == 32 else np.float64))
            return
        if data.size and float(np.max(np.abs(data))) > 1.0:
            interp.warn("Data clipped when writing file: sample values must be between -1 and 1.",
                        "MATLAB:audiovideo:audiowrite:dataClipped")
        x = np.clip(data.astype(float), -1.0, 1.0)
        if bits == 8:
            ints = np.clip(np.round(x * 128.0 + 128.0), 0, 255).astype(np.uint8)
        else:
            full = float(2 ** (bits - 1))
            ints = np.clip(np.round(x * full), -full, full - 1).astype(np.int64)
    else:
        need = {"uint8": 8, "int16": 16, "int32": 32}.get(data.dtype.name)
        if need is None:
            raise MatlabError(f"{what}: integer audio data must be uint8, int16 or int32.")
        if need != bits:
            raise MatlabError(f"{what}: {data.dtype.name} data is written with {need} bits per sample.")
        ints = data.astype(np.int64) if bits != 8 else data
    width = bits // 8
    if bits == 8:
        raw = ints.astype(np.uint8).tobytes()
    elif bits == 24:
        b = ints.astype("<i4").reshape(-1, 1).view(np.uint8).reshape(-1, 4)[:, :3]
        raw = b.tobytes()
    else:
        raw = ints.astype(f"<i{width}").tobytes()
    with wave.open(name, "wb") as w:
        w.setnchannels(data.shape[1])
        w.setsampwidth(width)
        w.setframerate(int(rate))
        w.writeframes(raw)


@builtin("audiowrite")
def _audiowrite(interp, args, nargout):
    if len(args) < 3:
        raise MatlabError("audiowrite needs a file name, the samples and the sample rate: "
                          "audiowrite('tone.wav', y, 44100).")
    bits = None
    k = 3
    while k < len(args):
        key = text(args[k], "option name").lower()
        if k + 1 >= len(args):
            raise MatlabError(f"audiowrite: the option '{text(args[k])}' needs a value.")
        if key == "bitspersample":
            bits = to_int(args[k + 1], "BitsPerSample")
        elif key not in ("bitrate", "quality", "title", "artist", "comment"):
            raise MatlabError(f"audiowrite: unknown option '{text(args[k])}'.")
        k += 2
    _write_wav(interp, text(args[0], "file name"), args[1], float(fnum(args[2]).flat[0]), bits, "audiowrite")
    return None


@builtin("wavwrite")
def _wavwrite(interp, args, nargout):
    nargs(args, 2, 4, "wavwrite")
    name = text(args[-1], "file name")
    if not os.path.splitext(name)[1]:
        name += ".wav"
    rate = float(fnum(args[1]).flat[0]) if len(args) > 2 else 8000.0
    bits = to_int(args[2], "bits") if len(args) > 3 else 16
    _write_wav(interp, name, args[0], rate, bits, "wavwrite")
    return None


# -- playback stand-ins -----------------------------------------------------------------------------

def _no_sound_note(interp) -> None:
    if not interp.state.get("sound_note_shown"):
        interp.state["sound_note_shown"] = True
        interp.out("(sound: there is no audio device here, so nothing is played; use audiowrite to save "
                   "the sound as a WAV file)\n")


def _check_sound(args: list, name: str) -> None:
    if not args:
        raise MatlabError(f"{name} needs the samples to play, e.g. {name}(y, 8000).")
    y = args[0]
    if not isinstance(y, np.ndarray) or y.dtype.kind not in "fiu":
        raise MatlabError(f"{name}: the audio data must be numeric.")
    if y.ndim > 2 or (min(y.shape) > 2 and y.size):
        raise MatlabError(f"{name}: the audio data must have one or two channels (columns).")
    if len(args) > 1:
        fs = float(fnum(args[1]).flat[0])
        if not 1000 <= fs <= 384000:
            raise MatlabError(f"{name}: the sample rate must be between 1000 and 384000 Hz (it is {fs:g}).")


@builtin("sound")
def _sound(interp, args, nargout):
    _check_sound(args, "sound")
    _no_sound_note(interp)
    return None


@builtin("soundsc")
def _soundsc(interp, args, nargout):
    _check_sound(args, "soundsc")
    _no_sound_note(interp)
    return None


class AudioPlayer(MObject):
    """audioplayer(y, Fs): holds the samples; play and playblocking return at once."""

    is_handle = True
    _METHODS = {"play", "playblocking", "stop", "pause", "resume", "isplaying"}

    def __init__(self, y: np.ndarray, fs: float, bits: float):
        self.y, self.fs, self.bits = y, fs, bits
        self.props: dict[str, Any] = {"Tag": char(""), "UserData": np.zeros((0, 0))}

    def class_name(self) -> str:
        return "audioplayer"

    def has_method(self, name: str) -> bool:
        return name in self._METHODS

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name == "isplaying":
            return [bool_scalar(False)]
        if name in ("play", "playblocking"):
            _no_sound_note(interp)
        return []

    def get_field(self, interp, name: str) -> Any:
        frames = self.y.shape[0] if self.y.ndim == 2 and self.y.shape[0] > 1 else self.y.size
        values = {"SampleRate": fscalar(self.fs), "BitsPerSample": fscalar(self.bits),
                  "NumChannels": fscalar(self.y.shape[1] if self.y.shape[0] > 1 else 1),
                  "NumberOfChannels": fscalar(self.y.shape[1] if self.y.shape[0] > 1 else 1),
                  "TotalSamples": fscalar(frames), "CurrentSample": fscalar(1), "Running": char("off"),
                  "Type": char("audioplayer"), "DeviceID": fscalar(-1)}
        if name in values:
            return values[name]
        if name in self.props:
            return self.props[name]
        if name in self._METHODS:
            return super().get_field(interp, name)
        raise MatlabError(f"Unrecognized property '{name}' for class 'audioplayer'.")

    def set_field(self, interp, name: str, value: Any) -> "AudioPlayer":
        if name == "SampleRate":
            self.fs = float(fnum(value).flat[0])
        else:
            self.props[name] = value
        return self

    def display_lines(self) -> list[str]:
        return ["  audioplayer with properties:", "", f"       SampleRate: {self.fs:g}",
                f"    BitsPerSample: {self.bits:g}", "          Running: 'off'"]


@builtin("audioplayer")
def _audioplayer(interp, args, nargout):
    _check_sound(args, "audioplayer")
    if len(args) < 2:
        raise MatlabError("audioplayer needs the samples and the sample rate: audioplayer(y, Fs).")
    bits = float(fnum(args[2]).flat[0]) if len(args) > 2 else 16.0
    return AudioPlayer(args[0], float(fnum(args[1]).flat[0]), bits)
