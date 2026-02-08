# main_remote_hud_foveated.py
from __future__ import annotations

import argparse
import base64
import ctypes
import ctypes.wintypes as w
import json
import struct
import threading
import time
import urllib.request
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final

# ============================================================
# FRANZ (Windows 11 + Python 3.12 only)
# - foveated HUD memory (remote)
# - multi-call tool chains (1..4)
# - no annotate/attend overlays
# - GDI StretchBlt(HALFTONE)+SetBrushOrgEx downsample
# - FIX: force PNG alpha=255 (GDI capture alpha is often 0 -> transparent screenshots)
# - FIX: ThreadingHTTPServer + no-store caching headers (mobile stability)
# ============================================================

API_URL: Final[str] = "http://localhost:1234/v1/chat/completions"
MODEL_NAME: Final[str] = "qwen3-vl-2b-instruct-1m"

DUMP_ROOT: Final[Path] = Path("dump")

RES_PRESETS: Final[dict[str, tuple[int, int]]] = {
    "low": (512, 288),
    "med": (1024, 576),
    "high": (1536, 864),
}

TOOL_CHOICE_DEFAULT: Final[str] = "auto"  # "auto" or "required"

SAMPLING: Final[dict[str, object]] = {
    "temperature": 1.2,
    "top_p": 0.8,
    "top_k": 20,
    "max_tokens": 700,
    "stream": False,
    "stop": [],
    "presence_penalty": 1.2,
    "frequency_penalty": 0.2,
    "logit_bias": {},
    "repeat_penalty": 1.2,
    "seed": 42,
}

SYSTEM_PROMPT: Final[str] = """You are FRANZ, operating a Windows desktop using ONLY the provided tools.

REMOTE HUD MEMORY
- Your narrative memory is stored in a REMOTE HUD (browser), not on-screen.
- The screenshot may or may not show that browser; treat any FRANZ HUD page as off-limits (do not click/type into it).

TIME MODEL
- The screenshot shows NOW (T).
- The HUD memory text provided in the prompt is your narrative memory from (T-1).
- The prompt includes LAST_ACTION (T-1). Use it to align your memory with the screenshot.

TOOL OUTPUT RULES
- Output a CHAIN of 1 to 4 tool calls per step (ordered).
- If your CHAIN_CONFIDENCE is < 0.65, output EXACTLY 1 tool call (a safe exploratory action or wait).
- If you output multiple tool calls, set story to an empty string for calls 1..N-1.
- Only the FINAL tool call (call N) may include a non-empty story.

JSON STRICTNESS
- Tool-call arguments MUST be a single JSON object.
- MUST contain ONLY keys defined in the tool schema (no extra keys).
- Coordinates are normalized integers 0..1000 (0,0 top-left; 1000,1000 bottom-right).

ACTION POLICY
- Prefer drag for drawing/painting strokes instead of repeated clicking.
- Keep chains short unless the UI state is stable.
- Use wait when nothing meaningful is changing, or to let UI settle.

FOVEATED HUD MEMORY (only in the FINAL tool call)
Rewrite the FULL story every step using EXACTLY these lines (<= 1400 characters total):
[PAST] Long-term mission + constraints + stable facts. (edge / long-term)
[NOW] What you see and what matters right now in the screenshot. (center / immediate)
[WHERE] Last action location and intent: tool + consideration + expected effect. (center / orientation)
[DELTA] What changed since last action; or UNKNOWN if you cannot verify. (center)
[NEXT] Next chain plan in tool terms (1..4 actions). (edge / near-future)
[CONF] 0.00..1.00
""".strip()

DEFAULT_HUD_TEXT: Final[str] = (
    "[PAST] FRANZ is a desktop control agent inside a Windows 11 sandbox. The remote HUD is his memory. "
    "He must complete tasks by using mouse/keyboard tools and keep a stable narrative state.\n"
    "[NOW] Initial pause.\n"
    "[WHERE] None.\n"
    "[DELTA] UNKNOWN.\n"
    "[NEXT] Resume and begin cautious exploration.\n"
    "[CONF] 0.50"
)


def _xy_schema() -> dict[str, Any]:
    return {"type": "integer", "minimum": 0, "maximum": 1000}


TOOLS: Final[list[dict[str, Any]]] = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Mouse: left click at normalized (x,y). story must be empty except for the FINAL tool call in a chain.",
            "parameters": {
                "type": "object",
                "properties": {"x": _xy_schema(), "y": _xy_schema(), "story": {"type": "string"}},
                "required": ["x", "y", "story"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "double_click",
            "description": "Mouse: double click at normalized (x,y). story must be empty except for the FINAL tool call in a chain.",
            "parameters": {
                "type": "object",
                "properties": {"x": _xy_schema(), "y": _xy_schema(), "story": {"type": "string"}},
                "required": ["x", "y", "story"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "right_click",
            "description": "Mouse: right click at normalized (x,y). story must be empty except for the FINAL tool call in a chain.",
            "parameters": {
                "type": "object",
                "properties": {"x": _xy_schema(), "y": _xy_schema(), "story": {"type": "string"}},
                "required": ["x", "y", "story"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drag",
            "description": "Mouse: drag from (x1,y1) to (x2,y2). story must be empty except for the FINAL tool call in a chain.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x1": _xy_schema(),
                    "y1": _xy_schema(),
                    "x2": _xy_schema(),
                    "y2": _xy_schema(),
                    "story": {"type": "string"},
                },
                "required": ["x1", "y1", "x2", "y2", "story"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Keyboard: type the given text. story must be empty except for the FINAL tool call in a chain.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "story": {"type": "string"}},
                "required": ["text", "story"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Mouse wheel: scroll by dy (positive=down, negative=up). story must be empty except for the FINAL tool call in a chain.",
            "parameters": {
                "type": "object",
                "properties": {"dy": {"type": "integer"}, "story": {"type": "string"}},
                "required": ["dy", "story"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "No-op: wait for ms milliseconds. Use when you need time for UI changes. story must be empty except for the FINAL tool call in a chain.",
            "parameters": {
                "type": "object",
                "properties": {"ms": {"type": "integer", "minimum": 1, "maximum": 10000}, "story": {"type": "string"}},
                "required": ["ms", "story"],
                "additionalProperties": False,
            },
        },
    },
]
TOOL_NAME_SET: Final[set[str]] = {str(t["function"]["name"]).strip().lower() for t in TOOLS}

# ============================================================
# Win32 (ctypes)
# ============================================================

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

try:
    ctypes.WinDLL("Shcore").SetProcessDpiAwareness(2)
except Exception:
    pass

INPUT_MOUSE: Final[int] = 0
INPUT_KEYBOARD: Final[int] = 1

WHEEL_DELTA: Final[int] = 120

MOUSEEVENTF_MOVE: Final[int] = 0x0001
MOUSEEVENTF_ABSOLUTE: Final[int] = 0x8000
MOUSEEVENTF_LEFTDOWN: Final[int] = 0x0002
MOUSEEVENTF_LEFTUP: Final[int] = 0x0004
MOUSEEVENTF_RIGHTDOWN: Final[int] = 0x0008
MOUSEEVENTF_RIGHTUP: Final[int] = 0x0010
MOUSEEVENTF_WHEEL: Final[int] = 0x0800

KEYEVENTF_UNICODE: Final[int] = 0x0004
KEYEVENTF_KEYUP: Final[int] = 0x0002

SRCCOPY: Final[int] = 0x00CC0020
CAPTUREBLT: Final[int] = 0x40000000

SM_CXSCREEN: Final[int] = 0
SM_CYSCREEN: Final[int] = 1

HALFTONE: Final[int] = 4

gdi32.CreateCompatibleDC.argtypes = [w.HDC]
gdi32.CreateCompatibleDC.restype = w.HDC
gdi32.DeleteDC.argtypes = [w.HDC]
gdi32.DeleteDC.restype = w.BOOL
gdi32.SelectObject.argtypes = [w.HDC, w.HGDIOBJ]
gdi32.SelectObject.restype = w.HGDIOBJ
gdi32.DeleteObject.argtypes = [w.HGDIOBJ]
gdi32.DeleteObject.restype = w.BOOL
gdi32.BitBlt.argtypes = [w.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, w.HDC, ctypes.c_int, ctypes.c_int, w.DWORD]
gdi32.BitBlt.restype = w.BOOL
gdi32.StretchBlt.argtypes = [
    w.HDC,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    w.HDC,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    w.DWORD,
]
gdi32.StretchBlt.restype = w.BOOL
gdi32.SetStretchBltMode.argtypes = [w.HDC, ctypes.c_int]
gdi32.SetStretchBltMode.restype = ctypes.c_int
gdi32.SetBrushOrgEx.argtypes = [w.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
gdi32.SetBrushOrgEx.restype = w.BOOL

user32.GetDC.argtypes = [w.HWND]
user32.GetDC.restype = w.HDC
user32.ReleaseDC.argtypes = [w.HWND, w.HDC]
user32.ReleaseDC.restype = ctypes.c_int
user32.GetSystemMetrics.argtypes = [ctypes.c_int]
user32.GetSystemMetrics.restype = ctypes.c_int
user32.SendInput.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_int]
user32.SendInput.restype = ctypes.c_uint


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", w.DWORD),
        ("biWidth", w.LONG),
        ("biHeight", w.LONG),
        ("biPlanes", w.WORD),
        ("biBitCount", w.WORD),
        ("biCompression", w.DWORD),
        ("biSizeImage", w.DWORD),
        ("biXPelsPerMeter", w.LONG),
        ("biYPelsPerMeter", w.LONG),
        ("biClrUsed", w.DWORD),
        ("biClrImportant", w.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", w.DWORD * 3)]


gdi32.CreateDIBSection.argtypes = [w.HDC, ctypes.POINTER(BITMAPINFO), ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), w.HANDLE, w.DWORD]
gdi32.CreateDIBSection.restype = w.HBITMAP

ULONG_PTR = getattr(w, "ULONG_PTR", ctypes.c_size_t)


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", w.LONG),
        ("dy", w.LONG),
        ("mouseData", w.DWORD),
        ("dwFlags", w.DWORD),
        ("time", w.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", w.WORD),
        ("wScan", w.WORD),
        ("dwFlags", w.DWORD),
        ("time", w.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _INPUTunion(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", w.DWORD), ("union", _INPUTunion)]


@dataclass(slots=True)
class Coord:
    sw: int
    sh: int

    def to_screen(self, xn: int, yn: int) -> tuple[int, int]:
        x = int(max(0, min(1000, xn)) / 1000.0 * (self.sw - 1))
        y = int(max(0, min(1000, yn)) / 1000.0 * (self.sh - 1))
        return x, y

    def to_win32_abs(self, x: int, y: int) -> tuple[int, int]:
        ax = int(x * 65535 / max(1, self.sw - 1))
        ay = int(y * 65535 / max(1, self.sh - 1))
        return ax, ay


def _raise_last_error(msg: str) -> None:
    raise ctypes.WinError(ctypes.get_last_error(), msg)


def capture_screen_bgra(sw: int, sh: int) -> bytes:
    sdc = user32.GetDC(0)
    if not sdc:
        _raise_last_error("GetDC")

    mdc = gdi32.CreateCompatibleDC(sdc)
    if not mdc:
        user32.ReleaseDC(0, sdc)
        _raise_last_error("CreateCompatibleDC")

    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = sw
    bmi.bmiHeader.biHeight = -sh
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = 0

    bits = ctypes.c_void_p()
    hbmp = gdi32.CreateDIBSection(sdc, ctypes.byref(bmi), 0, ctypes.byref(bits), None, 0)
    if not hbmp or not bits:
        gdi32.DeleteDC(mdc)
        user32.ReleaseDC(0, sdc)
        _raise_last_error("CreateDIBSection")

    old = gdi32.SelectObject(mdc, hbmp)
    ok = gdi32.BitBlt(mdc, 0, 0, sw, sh, sdc, 0, 0, SRCCOPY | CAPTUREBLT)
    if not ok:
        gdi32.SelectObject(mdc, old)
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(mdc)
        user32.ReleaseDC(0, sdc)
        _raise_last_error("BitBlt")

    out = ctypes.string_at(bits, sw * sh * 4)

    gdi32.SelectObject(mdc, old)
    gdi32.DeleteObject(hbmp)
    gdi32.DeleteDC(mdc)
    user32.ReleaseDC(0, sdc)

    return out


def downsample_bgra_stretchblt(src: bytes, sw: int, sh: int, dw: int, dh: int) -> bytes:
    if (sw, sh) == (dw, dh):
        return src
    if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
        return b""
    if len(src) < sw * sh * 4:
        return b""

    sdc = user32.GetDC(0)
    if not sdc:
        _raise_last_error("GetDC")

    try:
        src_dc = gdi32.CreateCompatibleDC(sdc)
        dst_dc = gdi32.CreateCompatibleDC(sdc)
        if not src_dc or not dst_dc:
            _raise_last_error("CreateCompatibleDC")

        bmi_src = BITMAPINFO()
        bmi_src.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi_src.bmiHeader.biWidth = sw
        bmi_src.bmiHeader.biHeight = -sh
        bmi_src.bmiHeader.biPlanes = 1
        bmi_src.bmiHeader.biBitCount = 32
        bmi_src.bmiHeader.biCompression = 0

        src_bits = ctypes.c_void_p()
        src_bmp = gdi32.CreateDIBSection(sdc, ctypes.byref(bmi_src), 0, ctypes.byref(src_bits), None, 0)
        if not src_bmp or not src_bits:
            _raise_last_error("CreateDIBSection(src)")

        old_src = gdi32.SelectObject(src_dc, src_bmp)
        ctypes.memmove(src_bits, src, sw * sh * 4)

        bmi_dst = BITMAPINFO()
        bmi_dst.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi_dst.bmiHeader.biWidth = dw
        bmi_dst.bmiHeader.biHeight = -dh
        bmi_dst.bmiHeader.biPlanes = 1
        bmi_dst.bmiHeader.biBitCount = 32
        bmi_dst.bmiHeader.biCompression = 0

        dst_bits = ctypes.c_void_p()
        dst_bmp = gdi32.CreateDIBSection(sdc, ctypes.byref(bmi_dst), 0, ctypes.byref(dst_bits), None, 0)
        if not dst_bmp or not dst_bits:
            gdi32.SelectObject(src_dc, old_src)
            gdi32.DeleteObject(src_bmp)
            _raise_last_error("CreateDIBSection(dst)")

        old_dst = gdi32.SelectObject(dst_dc, dst_bmp)

        gdi32.SetStretchBltMode(dst_dc, HALFTONE)
        gdi32.SetBrushOrgEx(dst_dc, 0, 0, None)

        ok = gdi32.StretchBlt(dst_dc, 0, 0, dw, dh, src_dc, 0, 0, sw, sh, SRCCOPY)
        if not ok:
            gdi32.SelectObject(src_dc, old_src)
            gdi32.SelectObject(dst_dc, old_dst)
            gdi32.DeleteObject(src_bmp)
            gdi32.DeleteObject(dst_bmp)
            gdi32.DeleteDC(src_dc)
            gdi32.DeleteDC(dst_dc)
            _raise_last_error("StretchBlt")

        out = ctypes.string_at(dst_bits, dw * dh * 4)

        gdi32.SelectObject(src_dc, old_src)
        gdi32.SelectObject(dst_dc, old_dst)
        gdi32.DeleteObject(src_bmp)
        gdi32.DeleteObject(dst_bmp)
        gdi32.DeleteDC(src_dc)
        gdi32.DeleteDC(dst_dc)

        return out
    finally:
        user32.ReleaseDC(0, sdc)


def encode_png(bgra: bytes, w_: int, h_: int) -> bytes:
    # FIX: force alpha=255 (GDI capture alpha often 0, which makes PNG fully transparent in browsers)
    rgba = bytearray(len(bgra))
    for i in range(0, len(bgra), 4):
        b = bgra[i]
        g = bgra[i + 1]
        r = bgra[i + 2]
        rgba[i] = r
        rgba[i + 1] = g
        rgba[i + 2] = b
        rgba[i + 3] = 255

    stride = w_ * 4
    raw = bytearray()
    for y in range(h_):
        raw.append(0)
        start = y * stride
        raw += rgba[start : start + stride]

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w_, h_, 8, 6, 0, 0, 0)  # RGBA
    idat = zlib.compress(bytes(raw), level=6)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _send_inputs(inputs: list[INPUT]) -> None:
    if not inputs:
        return
    arr = (INPUT * len(inputs))(*inputs)
    sent = user32.SendInput(len(arr), arr, ctypes.sizeof(INPUT))
    if sent != len(arr):
        _raise_last_error("SendInput")


def mouse_move_abs(x: int, y: int, conv: Coord) -> None:
    ax, ay = conv.to_win32_abs(x, y)
    _send_inputs(
        [
            INPUT(
                type=INPUT_MOUSE,
                union=_INPUTunion(
                    mi=MOUSEINPUT(dx=ax, dy=ay, mouseData=0, dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, time=0, dwExtraInfo=0)
                ),
            )
        ]
    )


def mouse_click(x: int, y: int, conv: Coord) -> None:
    mouse_move_abs(x, y, conv)
    _send_inputs(
        [
            INPUT(type=INPUT_MOUSE, union=_INPUTunion(mi=MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTDOWN, time=0, dwExtraInfo=0))),
            INPUT(type=INPUT_MOUSE, union=_INPUTunion(mi=MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTUP, time=0, dwExtraInfo=0))),
        ]
    )


def mouse_double_click(x: int, y: int, conv: Coord) -> None:
    mouse_click(x, y, conv)
    time.sleep(0.05)
    mouse_click(x, y, conv)


def mouse_right_click(x: int, y: int, conv: Coord) -> None:
    mouse_move_abs(x, y, conv)
    _send_inputs(
        [
            INPUT(type=INPUT_MOUSE, union=_INPUTunion(mi=MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_RIGHTDOWN, time=0, dwExtraInfo=0))),
            INPUT(type=INPUT_MOUSE, union=_INPUTunion(mi=MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_RIGHTUP, time=0, dwExtraInfo=0))),
        ]
    )


def mouse_drag(x1: int, y1: int, x2: int, y2: int, conv: Coord) -> None:
    mouse_move_abs(x1, y1, conv)
    _send_inputs([INPUT(type=INPUT_MOUSE, union=_INPUTunion(mi=MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTDOWN, time=0, dwExtraInfo=0)))])
    time.sleep(0.03)
    mouse_move_abs(x2, y2, conv)
    time.sleep(0.03)
    _send_inputs([INPUT(type=INPUT_MOUSE, union=_INPUTunion(mi=MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTUP, time=0, dwExtraInfo=0)))])


def scroll(dy: int) -> None:
    ticks = max(-100, min(100, int(dy)))
    md = ticks * WHEEL_DELTA
    _send_inputs([INPUT(type=INPUT_MOUSE, union=_INPUTunion(mi=MOUSEINPUT(dx=0, dy=0, mouseData=md, dwFlags=MOUSEEVENTF_WHEEL, time=0, dwExtraInfo=0)))])


def type_text(text: str) -> None:
    if not text:
        return
    utf16 = text.encode("utf-16le", errors="surrogatepass")
    inputs: list[INPUT] = []
    for i in range(0, len(utf16), 2):
        code = utf16[i] | (utf16[i + 1] << 8)
        inputs.append(INPUT(type=INPUT_KEYBOARD, union=_INPUTunion(ki=KEYBDINPUT(wVk=0, wScan=code, dwFlags=KEYEVENTF_UNICODE, time=0, dwExtraInfo=0))))
        inputs.append(
            INPUT(type=INPUT_KEYBOARD, union=_INPUTunion(ki=KEYBDINPUT(wVk=0, wScan=code, dwFlags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, time=0, dwExtraInfo=0)))
        )
    _send_inputs(inputs)


# ============================================================
# Remote HUD
# ============================================================

@dataclass(slots=True)
class RemoteHUDState:
    story: str
    dump_dir: Path
    paused: bool = True
    stop: bool = False
    latest_image: str = ""
    image_token: int = 0
    last_action: str = "None"
    step: int = 0
    recent_actions: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def set_story(self, s: str) -> None:
        s = (s or "").strip()
        if not s:
            s = DEFAULT_HUD_TEXT
        with self._lock:
            self.story = s

    def get_story(self) -> str:
        with self._lock:
            return self.story

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self.paused = paused

    def is_paused(self) -> bool:
        with self._lock:
            return self.paused

    def request_stop(self) -> None:
        with self._lock:
            self.stop = True

    def should_stop(self) -> bool:
        with self._lock:
            return self.stop

    def update_latest_image(self, name: str) -> None:
        with self._lock:
            self.latest_image = name
            self.image_token += 1

    def latest_image_path(self) -> Path | None:
        with self._lock:
            if not self.latest_image:
                return None
            return self.dump_dir / self.latest_image

    def update_runtime_state(self, *, step: int | None = None, last_action: str | None = None, recent_actions: list[str] | None = None) -> None:
        with self._lock:
            if step is not None:
                self.step = step
            if last_action is not None:
                self.last_action = last_action
            if recent_actions is not None:
                self.recent_actions = recent_actions[-20:]

    def state_json(self) -> dict[str, Any]:
        with self._lock:
            return {
                "paused": self.paused,
                "stop": self.stop,
                "story": self.story,
                "dump_dir": str(self.dump_dir),
                "latest_image": self.latest_image,
                "image_token": self.image_token,
                "step": self.step,
                "last_action": self.last_action,
                "recent_actions": self.recent_actions[-8:],
            }


class HTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            html = b"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>FRANZ HUD</title>
<style>
body { margin: 0; padding: 10px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: #000; color: #0f0; }
.row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
button { background: #000; color: #0f0; border: 1px solid #0f0; padding: 6px 10px; cursor: pointer; }
textarea { width: 100%; height: 320px; background: #000; color: #0f0; border: 1px solid #0f0; padding: 10px; box-sizing: border-box; }
#screenshot { width: 100%; border: 1px solid #0f0; margin-top: 10px; }
.small { color: #0a0; font-size: 12px; }
.list { white-space: pre-wrap; color: #0a0; font-size: 12px; }
</style>
</head>
<body>
<div class="row">
  <div>Status: <span id="status">?</span></div>
  <button id="toggle" onclick="toggleRun()">...</button>
  <button onclick="saveStory()">SAVE STORY</button>
  <button onclick="stopAgent()">STOP</button>
</div>
<div class="small">Step: <span id="step">0</span> | Last action: <span id="last">(none)</span></div>
<div class="small">Latest image: <span id="imgname">(none)</span> | Dump: <span id="dumpdir">(?)</span></div>
<textarea id="story" spellcheck="false"></textarea>
<div class="list" id="recent"></div>
<img id="screenshot" src="/screenshot?ts=0" />
<script>
const statusEl = document.getElementById('status');
const toggleBtn = document.getElementById('toggle');
const storyEl = document.getElementById('story');
const imgEl = document.getElementById('screenshot');
const imgNameEl = document.getElementById('imgname');
const dumpDirEl = document.getElementById('dumpdir');
const stepEl = document.getElementById('step');
const lastEl = document.getElementById('last');
const recentEl = document.getElementById('recent');

let lastToken = -1;
let lastPaused = true;
let inFlight = false;

async function tick() {
  if (inFlight) return;
  inFlight = true;
  try {
    const d = await fetch('/state', {cache: 'no-store'}).then(r => r.json());

    lastPaused = !!d.paused;
    statusEl.textContent = lastPaused ? 'PAUSED' : 'RUNNING';
    toggleBtn.textContent = lastPaused ? 'RESUME' : 'PAUSE';
    storyEl.readOnly = !lastPaused;

    stepEl.textContent = String(d.step ?? 0);
    lastEl.textContent = String(d.last_action ?? '');
    dumpDirEl.textContent = String(d.dump_dir ?? '');
    imgNameEl.textContent = d.latest_image || '(none)';

    const ra = Array.isArray(d.recent_actions) ? d.recent_actions : [];
    recentEl.textContent = ra.length ? ("Recent:\\n" + ra.join("\\n")) : "";

    if (document.activeElement !== storyEl) {
      storyEl.value = d.story || '';
    }

    if (typeof d.image_token === 'number' && d.image_token !== lastToken) {
      lastToken = d.image_token;
      imgEl.src = '/screenshot?ts=' + Date.now();
    }
  } catch (e) {
    // keep looping even if a request fails
  } finally {
    inFlight = false;
  }
}

async function toggleRun() {
  await fetch(lastPaused ? '/resume' : '/pause', {method: 'POST'});
  await tick();
}

async function saveStory() {
  await fetch('/story', {method: 'POST', body: storyEl.value});
  await tick();
}

async function stopAgent() {
  await fetch('/stop', {method: 'POST'});
}

tick();
setInterval(tick, 1000);
</script>
</body>
</html>"""
            self._send(200, "text/html; charset=utf-8", html)
            return

        if self.path.startswith("/state"):
            payload = self.server.hud_state.state_json()  # type: ignore[attr-defined]
            self._send(200, "application/json; charset=utf-8", json.dumps(payload).encode("utf-8"))
            return

        if self.path.startswith("/screenshot"):
            p = self.server.hud_state.latest_image_path()  # type: ignore[attr-defined]
            if p and p.exists():
                try:
                    data = p.read_bytes()
                except Exception:
                    data = b""
                if data:
                    self._send(200, "image/png", data)
                    return
            self._send(404, "text/plain; charset=utf-8", b"no screenshot")
            return

        self._send(404, "text/plain; charset=utf-8", b"not found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/pause":
            self.server.hud_state.set_paused(True)  # type: ignore[attr-defined]
            self._send(200, "text/plain; charset=utf-8", b"ok")
            return
        if self.path == "/resume":
            self.server.hud_state.set_paused(False)  # type: ignore[attr-defined]
            self._send(200, "text/plain; charset=utf-8", b"ok")
            return
        if self.path == "/stop":
            self.server.hud_state.request_stop()  # type: ignore[attr-defined]
            self._send(200, "text/plain; charset=utf-8", b"ok")
            return
        if self.path == "/story":
            length = int(self.headers.get("Content-Length", "0") or "0")
            data = self.rfile.read(length) if length > 0 else b""
            text = data.decode("utf-8", errors="replace")
            self.server.hud_state.set_story(text)  # type: ignore[attr-defined]
            self._send(200, "text/plain; charset=utf-8", b"ok")
            return
        self._send(404, "text/plain; charset=utf-8", b"not found")


class RemoteServer:
    def __init__(self, *, host: str, port: int, hud_state: RemoteHUDState):
        self._srv = ThreadingHTTPServer((host, port), HTTPHandler)
        self._srv.hud_state = hud_state  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._srv.shutdown()


# ============================================================
# Agent loop
# ============================================================

def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _as_int(x: Any) -> int | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            if "." in s:
                return int(float(s))
            return int(s)
        except ValueError:
            return None
    return None


def sanitize_tool_args(tool: str, args: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    t = tool.lower()

    def get_story() -> str:
        s = args.get("story", "")
        return s if isinstance(s, str) else ""

    if t in ("click", "double_click", "right_click"):
        x = _as_int(args.get("x"))
        y = _as_int(args.get("y"))
        if x is None or y is None:
            return None, "missing x/y"
        return {"x": _clamp_int(x, 0, 1000), "y": _clamp_int(y, 0, 1000), "story": get_story()}, ""

    if t == "drag":
        x1 = _as_int(args.get("x1"))
        y1 = _as_int(args.get("y1"))
        x2 = _as_int(args.get("x2"))
        y2 = _as_int(args.get("y2"))
        if None in (x1, y1, x2, y2):
            return None, "missing x1/y1/x2/y2"
        return {
            "x1": _clamp_int(x1, 0, 1000),
            "y1": _clamp_int(y1, 0, 1000),
            "x2": _clamp_int(x2, 0, 1000),
            "y2": _clamp_int(y2, 0, 1000),
            "story": get_story(),
        }, ""

    if t == "type_text":
        txt = args.get("text", "")
        if not isinstance(txt, str):
            txt = str(txt)
        return {"text": txt, "story": get_story()}, ""

    if t == "scroll":
        dy = _as_int(args.get("dy"))
        if dy is None:
            return None, "missing dy"
        return {"dy": _clamp_int(dy, -100, 100), "story": get_story()}, ""

    if t == "wait":
        ms = _as_int(args.get("ms"))
        if ms is None:
            return None, "missing ms"
        return {"ms": _clamp_int(ms, 1, 10_000), "story": get_story()}, ""

    return None, f"unknown tool {tool!r}"


def summarize_action(tool: str, args: dict[str, Any]) -> str:
    t = tool.lower()
    if t in ("click", "double_click", "right_click"):
        return f"{t}({args.get('x')},{args.get('y')})"
    if t == "drag":
        return f"drag({args.get('x1')},{args.get('y1')})->({args.get('x2')},{args.get('y2')})"
    if t == "scroll":
        return f"scroll(dy={args.get('dy')})"
    if t == "type_text":
        txt = str(args.get("text") or "")
        if len(txt) > 40:
            txt = txt[:40] + "…"
        return f"type_text({txt!r})"
    if t == "wait":
        return f"wait(ms={args.get('ms')})"
    return f"{t}(...)"


def trim_story(story: str, limit_chars: int = 1400) -> str:
    s = (story or "").strip().replace("\r", "")
    if len(s) <= limit_chars:
        return s
    return s[-limit_chars:]


def log_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write("\n")


def call_vlm(
    *,
    png: bytes,
    hud_story_tminus1: str,
    last_action_tminus1: str,
    recent_actions_tminusN: list[str],
    tool_choice: str,
) -> tuple[list[tuple[str, dict[str, Any]]], str]:
    recent_block = "\n".join(f"- {a}" for a in recent_actions_tminusN[-4:]) or "- (none)"

    user_text = (
        "You must control the desktop using tools.\n"
        "Execute a CHAIN of 1..4 tool calls. If CHAIN_CONFIDENCE < 0.65 then output EXACTLY 1 tool call.\n"
        "Intermediate calls must use story=\"\". Only the FINAL call includes the full HUD story in the fixed foveated format.\n"
        "Do NOT add extra JSON keys.\n\n"
        "HUD_MEMORY (T-1):\n"
        f"{hud_story_tminus1}\n\n"
        "LAST_ACTION (T-1):\n"
        f"{last_action_tminus1}\n\n"
        "RECENT_ACTIONS (most recent last):\n"
        f"{recent_block}\n"
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}"}},
                ],
            },
        ],
        "tools": TOOLS,
        "tool_choice": tool_choice,
        **SAMPLING,
    }

    req = urllib.request.Request(API_URL, json.dumps(payload).encode("utf-8"), {"Content-Type": "application/json"})
    data = json.load(urllib.request.urlopen(req, timeout=120))

    choices = data.get("choices") or []
    if not choices:
        raise ValueError("No choices in VLM response")

    msg = choices[0].get("message") or {}
    raw_content = str(msg.get("content") or "")

    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        return [], raw_content

    out: list[tuple[str, dict[str, Any]]] = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = str(fn.get("name") or "").strip().lower()
        if name not in TOOL_NAME_SET:
            continue

        args_raw = fn.get("arguments", "")
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw) if args_raw.strip() else {}
            except json.JSONDecodeError:
                continue
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            continue

        if isinstance(args, dict):
            out.append((name, args))

    return out, raw_content


def execute(tool: str, args: dict[str, Any], conv: Coord) -> None:
    match tool:
        case "click":
            x, y = conv.to_screen(int(args["x"]), int(args["y"]))
            mouse_click(x, y, conv)
        case "right_click":
            x, y = conv.to_screen(int(args["x"]), int(args["y"]))
            mouse_right_click(x, y, conv)
        case "double_click":
            x, y = conv.to_screen(int(args["x"]), int(args["y"]))
            mouse_double_click(x, y, conv)
        case "drag":
            x1, y1 = conv.to_screen(int(args["x1"]), int(args["y1"]))
            x2, y2 = conv.to_screen(int(args["x2"]), int(args["y2"]))
            mouse_drag(x1, y1, x2, y2, conv)
        case "type_text":
            type_text(str(args["text"]))
        case "scroll":
            scroll(int(args["dy"]))
        case "wait":
            time.sleep(int(args["ms"]) / 1000.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="FRANZ agent: foveated memory + multi-call + remote HUD")
    parser.add_argument("--res", choices=["low", "med", "high"], default="high", help="Model input resolution preset")
    parser.add_argument("--tool-choice", choices=["auto", "required"], default=TOOL_CHOICE_DEFAULT, help="tool_choice sent to the model server")
    parser.add_argument("--host", default="0.0.0.0", help="Remote HUD bind host")
    parser.add_argument("--port", type=int, default=8080, help="Remote HUD port")
    args = parser.parse_args()

    model_w, model_h = RES_PRESETS[args.res]
    sw, sh = user32.GetSystemMetrics(SM_CXSCREEN), user32.GetSystemMetrics(SM_CYSCREEN)
    conv = Coord(sw=sw, sh=sh)

    dump = DUMP_ROOT / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dump.mkdir(parents=True, exist_ok=True)
    events_path = dump / "events.jsonl"
    story_path = dump / "story.txt"

    hud_state = RemoteHUDState(story=DEFAULT_HUD_TEXT, dump_dir=dump, paused=True)
    hud_state.set_story(DEFAULT_HUD_TEXT)

    remote = RemoteServer(host=args.host, port=args.port, hud_state=hud_state)
    remote.start()

    print(f"FRANZ | Screen: {sw}x{sh} | Model input: {model_w}x{model_h} | Model: {MODEL_NAME} | tool_choice: {args.tool_choice}")
    print(f"Dump: {dump}")
    print(f"PAUSED - open http://<PC_IP>:{args.port}/ and click RESUME")

    last_story = DEFAULT_HUD_TEXT
    last_action = "None"
    recent_actions: list[str] = []

    step = 0
    try:
        while not hud_state.should_stop():
            while hud_state.is_paused() and not hud_state.should_stop():
                time.sleep(0.1)
            if hud_state.should_stop():
                break

            step += 1
            ts = datetime.now().strftime("%H:%M:%S")

            last_story = hud_state.get_story().strip() or DEFAULT_HUD_TEXT

            bgra = capture_screen_bgra(sw, sh)
            small = downsample_bgra_stretchblt(bgra, sw, sh, model_w, model_h)
            png = encode_png(small, model_w, model_h)

            img_name = f"step{step:03d}.png"
            (dump / img_name).write_bytes(png)
            hud_state.update_latest_image(img_name)

            hud_state.update_runtime_state(step=step, last_action=last_action, recent_actions=recent_actions)

            log_jsonl(
                events_path,
                {
                    "ts": ts,
                    "step": step,
                    "event": "screenshot",
                    "file": img_name,
                    "hud_tminus1": last_story,
                    "last_action_tminus1": last_action,
                },
            )

            try:
                calls_raw, raw_content = call_vlm(
                    png=png,
                    hud_story_tminus1=last_story,
                    last_action_tminus1=last_action,
                    recent_actions_tminusN=recent_actions,
                    tool_choice=args.tool_choice,
                )
            except Exception as e:
                err = str(e)
                print(f"[{ts}] {step:03d} | VLM ERROR: {err}")
                log_jsonl(events_path, {"ts": ts, "step": step, "event": "vlm_error", "error": err})
                time.sleep(0.5)
                continue

            if not calls_raw:
                preview = (raw_content or "").strip().replace("\r", "")
                if len(preview) > 800:
                    preview = preview[:800]
                print(f"[{ts}] {step:03d} | NO TOOL_CALLS PARSED | content_preview={preview!r}")
                log_jsonl(events_path, {"ts": ts, "step": step, "event": "no_tool_calls", "content_preview": preview})
                time.sleep(0.25)
                continue

            calls: list[tuple[str, dict[str, Any]]] = []
            for tool, raw_args in calls_raw[:4]:
                clean, reason = sanitize_tool_args(tool, raw_args)
                if clean is None:
                    log_jsonl(
                        events_path,
                        {
                            "ts": ts,
                            "step": step,
                            "event": "bad_tool_args",
                            "tool": tool,
                            "reason": reason,
                            "raw_args": raw_args,
                        },
                    )
                    continue
                calls.append((tool, clean))

            if not calls:
                print(f"[{ts}] {step:03d} | ALL TOOL_CALLS INVALID")
                log_jsonl(events_path, {"ts": ts, "step": step, "event": "all_invalid"})
                time.sleep(0.25)
                continue

            story_candidate = ""
            for _tool, _args in calls:
                s = _args.get("story")
                if isinstance(s, str) and s.strip():
                    story_candidate = s.strip()

            for idx, (tool, tool_args) in enumerate(calls, 1):
                action_summary = summarize_action(tool, tool_args)
                print(f"[{ts}] {step:03d}.{idx:02d} | {action_summary}")

                ok = True
                err = ""
                try:
                    execute(tool, tool_args, conv)
                except Exception as e:
                    ok = False
                    err = str(e)
                    print(f"[{ts}] {step:03d}.{idx:02d} | EXEC ERROR: {err} | args={tool_args}")

                if ok:
                    last_action = action_summary

                recent_actions.append(f"{step:03d}.{idx:02d} {action_summary} {'OK' if ok else 'ERR'}")
                recent_actions = recent_actions[-20:]
                hud_state.update_runtime_state(step=step, last_action=last_action, recent_actions=recent_actions)

                log_jsonl(
                    events_path,
                    {
                        "ts": ts,
                        "step": step,
                        "idx": idx,
                        "event": "tool",
                        "tool": tool,
                        "args": {k: v for k, v in tool_args.items() if k != "story"},
                        "ok": ok,
                        "error": err,
                        "last_action": last_action,
                    },
                )

                if not ok:
                    break
                time.sleep(0.05)

            if story_candidate:
                last_story = trim_story(story_candidate, 1400)
                hud_state.set_story(last_story)
                try:
                    story_path.write_text(last_story, encoding="utf-8", newline="\n")
                except Exception:
                    pass

            time.sleep(0.2)

    finally:
        remote.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nFRANZ stops.")
