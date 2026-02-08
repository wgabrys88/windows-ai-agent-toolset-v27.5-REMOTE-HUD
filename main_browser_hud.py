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
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Final

# Windows 11 + Python 3.12 only.
ULONG_PTR = getattr(w, "ULONG_PTR", ctypes.c_size_t)

API_URL: Final[str] = "http://localhost:1234/v1/chat/completions"
MODEL_NAME: Final[str] = "qwen3-vl-4b-instruct"

DUMP_FOLDER: Final[Path] = Path("dump")

RES_PRESETS: Final[dict[str, tuple[int, int]]] = {
    "low": (512, 288),
    "med": (1024, 576),
    "high": (1536, 864),
}

COLOR_PALETTE: Final[dict[str, int]] = {
    "red": 0x000000FF,
    "green": 0x0000FF00,
    "blue": 0x00FF0000,
    "yellow": 0x0000FFFF,
    "cyan": 0x00FFFF00,
    "magenta": 0x00FF00FF,
    "orange": 0x000099FF,
    "pink": 0x00CBC0FF,
}

ATTEND_WINDOW_WIDTH: Final[float] = 0.20
ATTEND_WINDOW_HEIGHT: Final[float] = 0.15
ATTEND_WINDOW_COLOR: Final[str] = "blue"
ATTEND_WINDOW_TRANSPARENCY: Final[int] = 60
ATTEND_MAX_WINDOWS: Final[int] = 6

STORY_MIN_LENGTH: Final[int] = 300
STORY_MAX_LENGTH: Final[int] = 2000

SAMPLING: Final[dict[str, object]] = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_tokens": 800,
    "stream": False,
    "stop": [],
    "presence_penalty": 1.5,
    "frequency_penalty": 0.0,
    "logit_bias": {},
    "repeat_penalty": 1.0,
    "seed": 42,
}

REMOTE_PORT_DEFAULT: Final[int] = 8080

OBS_NORM_W, OBS_NORM_H = ATTEND_WINDOW_WIDTH, ATTEND_WINDOW_HEIGHT
OBS_MAX_TARGETS: Final[int] = ATTEND_MAX_WINDOWS

MIN_REPORT_LENGTH = STORY_MIN_LENGTH
MAX_REPORT_LENGTH = STORY_MAX_LENGTH


def _get_color(name: str) -> int:
    return COLOR_PALETTE.get(name.lower(), COLOR_PALETTE["blue"])


OBS_DEFAULT_COLOR: Final[int] = _get_color(ATTEND_WINDOW_COLOR)
OBS_OPACITY: Final[int] = ATTEND_WINDOW_TRANSPARENCY


SYSTEM_PROMPT: Final[str] = """
You are FRANZ. You control the computer using tools only.

Coordinate system:
All x/y are integers in [0..1000]. (0,0)=top-left. (1000,1000)=bottom-right.
drag uses (x1,y1)->(x2,y2). scroll uses dy (positive=up, negative=down).

Tool categories:
Mouse: click, double_click, right_click, drag
Keyboard: type_text
Screen: scroll, attend

Rules:
- You receive TWO inputs each step: (1) the screenshot, (2) the current HUD story text (provided in user text).
- You must continue the story. For every tool call you output, include a "story" field.
- The story MUST be a rewritten full narrative, not just a delta.
- Use the screenshot to ground your actions (spot at least 5 areas correlated with continuation).
- Use blue attend windows as optional visual anchors when you are highly confident and want to observe.

This is your protocol. To ACT!
""".strip()

DEFAULT_HUD_TEXT: Final[str] = "FRANZ is exploring the environment and its own capabilities."

TOOLS: Final[list[dict[str, Any]]] = [
    {"type": "function", "function": {"name": "click", "description": "Mouse: click", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "story": {"type": "string"}}, "required": ["x", "y", "story"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "double_click", "description": "Mouse: double click", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "story": {"type": "string"}}, "required": ["x", "y", "story"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "right_click", "description": "Mouse: right click", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "story": {"type": "string"}}, "required": ["x", "y", "story"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "drag", "description": "Mouse: drag", "parameters": {"type": "object", "properties": {"x1": {"type": "integer"}, "y1": {"type": "integer"}, "x2": {"type": "integer"}, "y2": {"type": "integer"}, "story": {"type": "string"}}, "required": ["x1", "y1", "x2", "y2", "story"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "type_text", "description": "Keyboard: type", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "story": {"type": "string"}}, "required": ["text", "story"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "scroll", "description": "Screen: scroll", "parameters": {"type": "object", "properties": {"dy": {"type": "integer"}, "story": {"type": "string"}}, "required": ["dy", "story"], "additionalProperties": False}}},
    {
        "type": "function",
        "function": {
            "name": "attend",
            "description": "Screen: attend (show blue overlays).",
            "parameters": {
                "type": "object",
                "properties": {
                    "targets": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": OBS_MAX_TARGETS,
                        "items": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "label": {"type": "string"}}, "required": ["x", "y", "label"], "additionalProperties": False},
                    },
                    "story": {"type": "string"},
                },
                "required": ["targets", "story"],
                "additionalProperties": False,
            },
        },
    },
]
TOOL_NAME_SET: Final[set[str]] = {str(t["function"]["name"]).strip().lower() for t in TOOLS}

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

kernel32.GetCurrentProcessId.restype = w.DWORD
user32.GetWindowThreadProcessId.argtypes = [w.HWND, ctypes.POINTER(w.DWORD)]
user32.GetWindowThreadProcessId.restype = w.DWORD

ctypes.WinDLL("Shcore").SetProcessDpiAwareness(2)
kernel32.LoadLibraryW("Msftedit.dll")

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

WS_POPUP: Final[int] = 0x80000000
WS_VISIBLE: Final[int] = 0x10000000
WS_CHILD: Final[int] = 0x40000000
ES_MULTILINE: Final[int] = 0x0004
ES_AUTOVSCROLL: Final[int] = 0x0040
ES_READONLY: Final[int] = 0x0800
WS_EX_TOPMOST: Final[int] = 0x00000008
WS_EX_LAYERED: Final[int] = 0x00080000
WS_EX_TOOLWINDOW: Final[int] = 0x00000080

WM_SETFONT: Final[int] = 0x0030
WM_CLOSE: Final[int] = 0x0010
WM_DESTROY: Final[int] = 0x0002
WM_MOUSEWHEEL: Final[int] = 0x020A

SW_SHOWNOACTIVATE: Final[int] = 4
SWP_NOMOVE: Final[int] = 0x0002
SWP_NOSIZE: Final[int] = 0x0001
SWP_NOACTIVATE: Final[int] = 0x0010
SWP_SHOWWINDOW: Final[int] = 0x0040
HWND_TOPMOST: Final[int] = -1
SRCCOPY: Final[int] = 0x00CC0020
CAPTUREBLT: Final[int] = 0x40000000
LWA_ALPHA: Final[int] = 0x00000002
CS_HREDRAW: Final[int] = 0x0002
CS_VREDRAW: Final[int] = 0x0001
IDC_ARROW: Final[int] = 32512

MAKEINTRESOURCEW = lambda i: ctypes.cast(ctypes.c_void_p(i & 0xFFFF), w.LPCWSTR)

# GDI stretch mode constant: HALFTONE = 4 (higher quality scaling).
HALFTONE: Final[int] = 4


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", w.LONG), ("dy", w.LONG), ("mouseData", w.DWORD), ("dwFlags", w.DWORD), ("time", w.DWORD), ("dwExtraInfo", ULONG_PTR)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", w.WORD), ("wScan", w.WORD), ("dwFlags", w.DWORD), ("time", w.DWORD), ("dwExtraInfo", ULONG_PTR)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", w.DWORD), ("wParamL", w.WORD), ("wParamH", w.WORD)]


class _INPUTunion(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", w.DWORD), ("union", _INPUTunion)]


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


class MSG(ctypes.Structure):
    _fields_ = [("hwnd", w.HWND), ("message", ctypes.c_uint), ("wParam", w.WPARAM), ("lParam", w.LPARAM), ("time", w.DWORD), ("pt", w.POINT)]


class RECT(ctypes.Structure):
    _fields_ = [("left", w.LONG), ("top", w.LONG), ("right", w.LONG), ("bottom", w.LONG)]


WNDPROC = ctypes.WINFUNCTYPE(w.LPARAM, w.HWND, ctypes.c_uint, w.WPARAM, w.LPARAM)
WNDENUMPROC = ctypes.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint),
        ("style", ctypes.c_uint),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", w.HINSTANCE),
        ("hIcon", w.HANDLE),
        ("hCursor", w.HANDLE),
        ("hbrBackground", w.HANDLE),
        ("lpszMenuName", w.LPCWSTR),
        ("lpszClassName", w.LPCWSTR),
        ("hIconSm", w.HANDLE),
    ]


_SIGNATURES: Final[list[tuple[Any, list[tuple[str, list[Any], Any]]]]] = [
    (
        gdi32,
        [
            ("DeleteObject", [w.HGDIOBJ], w.BOOL),
            ("CreateCompatibleDC", [w.HDC], w.HDC),
            ("CreateDIBSection", [w.HDC, ctypes.POINTER(BITMAPINFO), ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), w.HANDLE, w.DWORD], w.HBITMAP),
            ("SelectObject", [w.HDC, w.HGDIOBJ], w.HGDIOBJ),
            ("BitBlt", [w.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, w.HDC, ctypes.c_int, ctypes.c_int, w.DWORD], w.BOOL),
            ("StretchBlt", [w.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, w.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, w.DWORD], w.BOOL),
            ("SetStretchBltMode", [w.HDC, ctypes.c_int], ctypes.c_int),
            ("DeleteDC", [w.HDC], w.BOOL),
            ("CreateFontW", [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, w.DWORD, w.DWORD, w.DWORD, w.DWORD, w.DWORD, w.DWORD, w.DWORD, w.DWORD, w.LPCWSTR], w.HFONT),
            ("CreateSolidBrush", [w.COLORREF], w.HANDLE),
        ],
    ),
    (
        user32,
        [
            ("CreateWindowExW", [w.DWORD, w.LPCWSTR, w.LPCWSTR, w.DWORD, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, w.HWND, w.HMENU, w.HINSTANCE, w.LPVOID], w.HWND),
            ("ShowWindow", [w.HWND, ctypes.c_int], w.BOOL),
            ("SetWindowPos", [w.HWND, w.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint], w.BOOL),
            ("DestroyWindow", [w.HWND], w.BOOL),
            ("SendInput", [ctypes.c_uint, ctypes.POINTER(INPUT), ctypes.c_int], ctypes.c_uint),
            ("GetSystemMetrics", [ctypes.c_int], ctypes.c_int),
            ("GetDC", [w.HWND], w.HDC),
            ("ReleaseDC", [w.HWND, w.HDC], ctypes.c_int),
            ("PostMessageW", [w.HWND, ctypes.c_uint, w.WPARAM, w.LPARAM], w.BOOL),
            ("GetMessageW", [ctypes.POINTER(MSG), w.HWND, ctypes.c_uint, ctypes.c_uint], w.BOOL),
            ("TranslateMessage", [ctypes.POINTER(MSG)], w.BOOL),
            ("DispatchMessageW", [ctypes.POINTER(MSG)], w.LPARAM),
            ("SetLayeredWindowAttributes", [w.HWND, w.COLORREF, ctypes.c_ubyte, w.DWORD], w.BOOL),
            ("DefWindowProcW", [w.HWND, ctypes.c_uint, w.WPARAM, w.LPARAM], w.LPARAM),
            ("RegisterClassExW", [ctypes.POINTER(WNDCLASSEXW)], w.ATOM),
            ("LoadCursorW", [w.HINSTANCE, w.LPCWSTR], w.HANDLE),
            ("GetAsyncKeyState", [ctypes.c_int], ctypes.c_short),
            ("EnumWindows", [WNDENUMPROC, w.LPARAM], w.BOOL),
            ("EnumChildWindows", [w.HWND, WNDENUMPROC, w.LPARAM], w.BOOL),
            ("IsWindowVisible", [w.HWND], w.BOOL),
            ("GetClassNameW", [w.HWND, w.LPWSTR, ctypes.c_int], ctypes.c_int),
            ("GetWindowRect", [w.HWND, ctypes.POINTER(RECT)], w.BOOL),
        ],
    ),
    (kernel32, [("GetModuleHandleW", [w.LPCWSTR], w.HMODULE)]),
]

for dll, funcs in _SIGNATURES:
    for name, args, res in funcs:
        fn = getattr(dll, name)
        fn.argtypes = args
        fn.restype = res


@dataclass(slots=True)
class Coord:
    sw: int
    sh: int

    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        nx = max(0.0, min(1000.0, x)) * self.sw / 1000
        ny = max(0.0, min(1000.0, y)) * self.sh / 1000
        return int(nx), int(ny)

    def to_win32(self, x: int, y: int) -> tuple[int, int]:
        wx = (x * 65535 // self.sw) if self.sw > 0 else 0
        wy = (y * 65535 // self.sh) if self.sh > 0 else 0
        return wx, wy


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def send_input(inputs: list[INPUT]) -> None:
    arr = (INPUT * len(inputs))(*inputs)
    if user32.SendInput(len(arr), arr, ctypes.sizeof(INPUT)) != len(inputs):
        raise ctypes.WinError(ctypes.get_last_error())
    time.sleep(0.05)


def make_mouse_input(dx: int, dy: int, flags: int, data: int = 0) -> INPUT:
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.union.mi = MOUSEINPUT(dx=dx, dy=dy, mouseData=data, dwFlags=flags, time=0, dwExtraInfo=0)
    return inp


def mouse_click(x: int, y: int, conv: Coord) -> None:
    wx, wy = conv.to_win32(x, y)
    send_input([make_mouse_input(wx, wy, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE), make_mouse_input(0, 0, MOUSEEVENTF_LEFTDOWN), make_mouse_input(0, 0, MOUSEEVENTF_LEFTUP)])


def mouse_right_click(x: int, y: int, conv: Coord) -> None:
    wx, wy = conv.to_win32(x, y)
    send_input([make_mouse_input(wx, wy, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE), make_mouse_input(0, 0, MOUSEEVENTF_RIGHTDOWN), make_mouse_input(0, 0, MOUSEEVENTF_RIGHTUP)])


def mouse_double_click(x: int, y: int, conv: Coord) -> None:
    wx, wy = conv.to_win32(x, y)
    send_input([make_mouse_input(wx, wy, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE), make_mouse_input(0, 0, MOUSEEVENTF_LEFTDOWN), make_mouse_input(0, 0, MOUSEEVENTF_LEFTUP)])
    time.sleep(0.05)
    send_input([make_mouse_input(0, 0, MOUSEEVENTF_LEFTDOWN), make_mouse_input(0, 0, MOUSEEVENTF_LEFTUP)])


def mouse_drag(x1: int, y1: int, x2: int, y2: int, conv: Coord) -> None:
    wx1, wy1 = conv.to_win32(x1, y1)
    wx2, wy2 = conv.to_win32(x2, y2)
    send_input([make_mouse_input(wx1, wy1, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE), make_mouse_input(0, 0, MOUSEEVENTF_LEFTDOWN)])
    time.sleep(0.05)
    for i in range(1, 11):
        ix = int(wx1 + (wx2 - wx1) * i / 10)
        iy = int(wy1 + (wy2 - wy1) * i / 10)
        send_input([make_mouse_input(ix, iy, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)])
        time.sleep(0.01)
    send_input([make_mouse_input(0, 0, MOUSEEVENTF_LEFTUP)])


def type_text(text: str) -> None:
    if not text:
        return
    utf16 = text.encode("utf-16le")
    inputs: list[INPUT] = []
    for i in range(0, len(utf16), 2):
        code = utf16[i] | (utf16[i + 1] << 8)
        d = INPUT()
        d.type = INPUT_KEYBOARD
        d.union.ki = KEYBDINPUT(wVk=0, wScan=code, dwFlags=KEYEVENTF_UNICODE, time=0, dwExtraInfo=0)
        inputs.append(d)
        u = INPUT()
        u.type = INPUT_KEYBOARD
        u.union.ki = KEYBDINPUT(wVk=0, wScan=code, dwFlags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, time=0, dwExtraInfo=0)
        inputs.append(u)
    send_input(inputs)


def scroll(dy: float) -> None:
    direction = 1 if dy > 0 else -1
    count = max(1, int(abs(dy) / WHEEL_DELTA))
    send_input([make_mouse_input(0, 0, MOUSEEVENTF_WHEEL, WHEEL_DELTA * direction) for _ in range(count)])


def _get_class_name(hwnd: w.HWND) -> str:
    buf = ctypes.create_unicode_buffer(256)
    n = user32.GetClassNameW(hwnd, buf, 256)
    return buf.value[:n] if n > 0 else ""


def append_execution_log(dump: Path, image_name: str, sw: int, sh: int, story: str) -> None:
    """
    Replaces the old "HUD window WM_GETTEXT capture" by writing the story directly.
    Still enumerates all visible windows belonging to this process (mostly OBS overlays).
    """
    log_path = dump / "execution-log.txt"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    pid = int(kernel32.GetCurrentProcessId())
    lines: list[str] = [f"=== {ts} | pid={pid} | image={image_name} ===\n"]

    if story:
        st = (story or "").replace("\r", "")
        lines.append("HUD_STORY:\n")
        for ln in st.split("\n"):
            lines.append("    " + ln + "\n")
        lines.append("\n")

    windows: list[w.HWND] = []

    @WNDENUMPROC
    def _enum_proc(hwnd: w.HWND, lparam: w.LPARAM) -> w.BOOL:
        try:
            if not user32.IsWindowVisible(hwnd):
                return True
            pid_out = w.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid_out))
            if int(pid_out.value) != pid:
                return True
            r = RECT()
            if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
                return True
            if r.right <= 0 or r.bottom <= 0 or r.left >= sw or r.top >= sh:
                return True
            windows.append(hwnd)
        except Exception:
            pass
        return True

    user32.EnumWindows(_enum_proc, 0)

    for i, hwnd in enumerate(windows, 1):
        try:
            r = RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(r))
            cls = _get_class_name(hwnd)
            lines.append(f"[{i:03d}] hwnd=0x{int(hwnd):016X} class={cls} rect=({r.left},{r.top},{r.right},{r.bottom})\n")
        except Exception as e:
            lines.append(f"[{i:03d}] hwnd=0x{int(hwnd):016X} log_error={e}\n")

    lines.append("\n")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace", newline="\n") as f:
        f.write("".join(lines))


def capture_screen(sw: int, sh: int) -> bytes:
    sdc = user32.GetDC(0)
    if not sdc:
        raise ctypes.WinError(ctypes.get_last_error())

    mdc = gdi32.CreateCompatibleDC(sdc)
    if not mdc:
        user32.ReleaseDC(0, sdc)
        raise ctypes.WinError(ctypes.get_last_error())

    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth, bmi.bmiHeader.biHeight = sw, -sh
    bmi.bmiHeader.biPlanes, bmi.bmiHeader.biBitCount = 1, 32

    bits = ctypes.c_void_p()
    hbm = gdi32.CreateDIBSection(sdc, ctypes.byref(bmi), 0, ctypes.byref(bits), None, 0)
    if not hbm:
        gdi32.DeleteDC(mdc)
        user32.ReleaseDC(0, sdc)
        raise ctypes.WinError(ctypes.get_last_error())

    gdi32.SelectObject(mdc, hbm)

    if not gdi32.BitBlt(mdc, 0, 0, sw, sh, sdc, 0, 0, SRCCOPY | CAPTUREBLT):
        gdi32.DeleteObject(hbm)
        gdi32.DeleteDC(mdc)
        user32.ReleaseDC(0, sdc)
        raise ctypes.WinError(ctypes.get_last_error())

    out = ctypes.string_at(bits, sw * sh * 4)
    user32.ReleaseDC(0, sdc)
    gdi32.DeleteDC(mdc)
    gdi32.DeleteObject(hbm)
    return out


def downsample(src: bytes, sw: int, sh: int, dw: int, dh: int) -> bytes:
    """High-quality downsample using StretchBlt(HALFTONE). BGRA32 in/out."""
    if (sw, sh) == (dw, dh):
        return src
    if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0 or len(src) < sw * sh * 4:
        return b""

    sdc = user32.GetDC(0)
    if not sdc:
        raise ctypes.WinError(ctypes.get_last_error())

    try:
        src_dc = gdi32.CreateCompatibleDC(sdc)
        dst_dc = gdi32.CreateCompatibleDC(sdc)
        if not src_dc or not dst_dc:
            raise ctypes.WinError(ctypes.get_last_error())

        bmi_src = BITMAPINFO()
        bmi_src.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi_src.bmiHeader.biWidth, bmi_src.bmiHeader.biHeight = sw, -sh
        bmi_src.bmiHeader.biPlanes, bmi_src.bmiHeader.biBitCount = 1, 32

        src_bits = ctypes.c_void_p()
        src_bmp = gdi32.CreateDIBSection(sdc, ctypes.byref(bmi_src), 0, ctypes.byref(src_bits), None, 0)
        if not src_bmp or not src_bits:
            raise ctypes.WinError(ctypes.get_last_error())

        old_src = gdi32.SelectObject(src_dc, src_bmp)
        ctypes.memmove(src_bits, src, sw * sh * 4)

        bmi_dst = BITMAPINFO()
        bmi_dst.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi_dst.bmiHeader.biWidth, bmi_dst.bmiHeader.biHeight = dw, -dh
        bmi_dst.bmiHeader.biPlanes, bmi_dst.bmiHeader.biBitCount = 1, 32

        dst_bits = ctypes.c_void_p()
        dst_bmp = gdi32.CreateDIBSection(sdc, ctypes.byref(bmi_dst), 0, ctypes.byref(dst_bits), None, 0)
        if not dst_bmp or not dst_bits:
            raise ctypes.WinError(ctypes.get_last_error())

        old_dst = gdi32.SelectObject(dst_dc, dst_bmp)
        gdi32.SetStretchBltMode(dst_dc, HALFTONE)

        if not gdi32.StretchBlt(dst_dc, 0, 0, dw, dh, src_dc, 0, 0, sw, sh, SRCCOPY):
            raise ctypes.WinError(ctypes.get_last_error())

        result = ctypes.string_at(dst_bits, dw * dh * 4)

        gdi32.SelectObject(src_dc, old_src)
        gdi32.SelectObject(dst_dc, old_dst)
        gdi32.DeleteObject(src_bmp)
        gdi32.DeleteObject(dst_bmp)
        gdi32.DeleteDC(src_dc)
        gdi32.DeleteDC(dst_dc)
        return result
    finally:
        user32.ReleaseDC(0, sdc)


def encode_png(bgra: bytes, width: int, height: int) -> bytes:
    raw = bytearray((width * 3 + 1) * height)
    for y in range(height):
        raw[y * (width * 3 + 1)] = 0
        row = bgra[y * width * 4 : (y + 1) * width * 4]
        for x in range(width):
            raw[y * (width * 3 + 1) + 1 + x * 3 : y * (width * 3 + 1) + 1 + x * 3 + 3] = [
                row[x * 4 + 2],
                row[x * 4 + 1],
                row[x * 4 + 0],
            ]
    comp = zlib.compress(bytes(raw), 6)
    ihdr = struct.pack(">2I5B", width, height, 8, 2, 0, 0, 0)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", comp) + chunk(b"IEND", b"")


def create_shared_fonts() -> tuple[w.HFONT, w.HFONT]:
    mono = gdi32.CreateFontW(-14, 0, 0, 0, 400, 0, 0, 0, 1, 0, 0, 0, 0, "Consolas")
    ui = gdi32.CreateFontW(-14, 0, 0, 0, 700, 0, 0, 0, 1, 0, 0, 0, 0, "Segoe UI")
    return mono, ui


@dataclass(slots=True)
class LabeledObsWindow:
    hwnd: w.HWND | None = None
    edit: w.HWND | None = None
    thread: threading.Thread | None = None
    ready: threading.Event = field(default_factory=threading.Event)
    stop: threading.Event = field(default_factory=threading.Event)
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    label: str = ""
    color: int = OBS_DEFAULT_COLOR
    font: w.HFONT = 0
    zoom_num: int = 180
    zoom_den: int = 100
    _wndproc_ref: WNDPROC | None = None

    def _wndproc(self, hwnd: w.HWND, msg: int, wparam: w.WPARAM, lparam: w.LPARAM) -> w.LPARAM:
        try:
            if msg == WM_MOUSEWHEEL:
                delta = ctypes.c_short(wparam >> 16).value
                ctrl = bool(user32.GetAsyncKeyState(0x11) & 0x8000)
                if ctrl:
                    if delta > 0:
                        self.zoom_num = min(400, int(self.zoom_num * 1.1))
                    else:
                        self.zoom_num = max(20, int(self.zoom_num * 0.9))
                    # no edit zoom; overlays are read-only labels
                    return 0
            if msg in (WM_CLOSE, WM_DESTROY):
                self.stop.set()
                if msg == WM_CLOSE:
                    user32.DestroyWindow(hwnd)
                return 0
        except Exception:
            pass
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _thread(self) -> None:
        hinst = kernel32.GetModuleHandleW(None)
        self._wndproc_ref = WNDPROC(self._wndproc)
        brush = gdi32.CreateSolidBrush(self.color)

        wc = WNDCLASSEXW(
            cbSize=ctypes.sizeof(WNDCLASSEXW),
            style=CS_HREDRAW | CS_VREDRAW,
            lpfnWndProc=self._wndproc_ref,
            cbClsExtra=0,
            cbWndExtra=0,
            hInstance=hinst,
            hIcon=None,
            hCursor=user32.LoadCursorW(None, MAKEINTRESOURCEW(IDC_ARROW)),
            hbrBackground=brush,
            lpszMenuName=None,
            lpszClassName="FRANZLabeledObs",
            hIconSm=None,
        )
        if not user32.RegisterClassExW(ctypes.byref(wc)) and ctypes.get_last_error() != 1410:
            self.ready.set()
            return

        self.hwnd = user32.CreateWindowExW(
            WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TOOLWINDOW,
            "FRANZLabeledObs",
            "",
            WS_POPUP | WS_VISIBLE,
            self.x,
            self.y,
            self.w,
            self.h,
            None,
            None,
            hinst,
            None,
        )
        if not self.hwnd:
            self.ready.set()
            return

        user32.SetLayeredWindowAttributes(self.hwnd, self.color, ctypes.c_ubyte(int(255 * OBS_OPACITY / 100)), LWA_ALPHA)

        # simple label via window title is enough for overlays; keep a child RichEdit like original for readability
        self.edit = user32.CreateWindowExW(
            0,
            "RICHEDIT50W",
            "",
            WS_CHILD | WS_VISIBLE | ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY,
            5,
            5,
            max(10, self.w - 10),
            max(10, self.h - 10),
            self.hwnd,
            None,
            hinst,
            None,
        )
        if self.edit:
            if self.font:
                user32.SendMessageW(self.edit, WM_SETFONT, self.font, 1)
            # SetWindowTextW is enough to show label
            user32.SetWindowTextW(self.edit, self.label)

        user32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE)
        user32.SetWindowPos(self.hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW)
        self.ready.set()

        msg = MSG()
        while not self.stop.is_set():
            if user32.GetMessageW(ctypes.byref(msg), None, 0, 0) in (0, -1):
                break
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    def show(self, cx: int, cy: int, sw: int, sh: int, label: str, color: int, font: w.HFONT) -> None:
        self.w = max(80, int(sw * OBS_NORM_W))
        self.h = max(60, int(sh * OBS_NORM_H))
        self.x = clamp(cx - self.w // 2, 0, max(0, sw - self.w))
        self.y = clamp(cy - self.h // 2, 0, max(0, sh - self.h))
        self.label = label
        self.color = color
        self.font = font
        self.ready.clear()
        self.stop.clear()
        self.thread = threading.Thread(target=self._thread, daemon=True)
        self.thread.start()
        self.ready.wait(timeout=1.0)

    def hide(self) -> None:
        self.stop.set()
        if self.hwnd:
            user32.PostMessageW(self.hwnd, WM_CLOSE, 0, 0)
        if self.thread:
            self.thread.join(timeout=0.5)
        self.hwnd = None
        self.edit = None
        self.thread = None


@dataclass(slots=True)
class ObsManager:
    windows: list[LabeledObsWindow] = field(default_factory=list)

    def show_multiple(self, targets: list[dict[str, Any]], sw: int, sh: int, conv: Coord, font: w.HFONT, color: int = OBS_DEFAULT_COLOR) -> None:
        self.hide_all()
        for t in targets[:OBS_MAX_TARGETS]:
            x_norm = float(t.get("x", 500))
            y_norm = float(t.get("y", 500))
            label = str(t.get("label", "")).strip() or f"({int(x_norm)},{int(y_norm)})"
            ox, oy = conv.to_screen(x_norm, y_norm)
            obs = LabeledObsWindow()
            obs.show(ox, oy, sw, sh, label, color, font)
            self.windows.append(obs)

    def hide_all(self) -> None:
        for obs in self.windows:
            obs.hide()
        self.windows.clear()


class RemoteHUDState:
    def __init__(self, *, story: str, dump_dir: Path):
        self._lock = threading.Lock()
        self._story = story
        self._paused = True
        self._run_event = threading.Event()
        self._stop = False
        self._dump_dir = dump_dir
        self._latest_image_name: str = ""
        self._image_token = 0

    def pause(self) -> None:
        with self._lock:
            self._paused = True
            self._run_event.clear()

    def resume(self) -> None:
        with self._lock:
            self._paused = False
            self._run_event.set()

    def stop(self) -> None:
        with self._lock:
            self._stop = True
            self._run_event.set()

    def should_stop(self) -> bool:
        with self._lock:
            return self._stop

    def wait_unpaused(self) -> None:
        while True:
            if self.should_stop():
                return
            with self._lock:
                if not self._paused:
                    return
            self._run_event.wait(timeout=0.1)

    def set_story(self, story: str) -> None:
        with self._lock:
            self._story = story

    def get_story(self) -> str:
        with self._lock:
            return self._story

    def set_latest_image(self, name: str) -> None:
        with self._lock:
            self._latest_image_name = name
            self._image_token += 1

    def state_json(self) -> dict[str, Any]:
        with self._lock:
            return {
                "paused": self._paused,
                "story": self._story,
                "latest_image": self._latest_image_name,
                "image_token": self._image_token,
                "dump_dir": str(self._dump_dir),
            }

    def latest_image_path(self) -> Path | None:
        with self._lock:
            if not self._latest_image_name:
                return None
            return self._dump_dir / self._latest_image_name


class HTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A003
        return

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/":
            html = b"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>FRANZ HUD</title>
<style>
body { margin: 0; padding: 10px; font-family: monospace; background: #000; color: #0f0; }
.row { display: flex; gap: 10px; align-items: center; }
button { background: #000; color: #0f0; border: 1px solid #0f0; padding: 6px 10px; cursor: pointer; }
textarea { width: 100%; height: 320px; background: #000; color: #0f0; border: 1px solid #0f0; padding: 10px; box-sizing: border-box; }
#screenshot { width: 100%; border: 1px solid #0f0; margin-top: 10px; }
.small { color: #0a0; font-size: 12px; }
</style>
</head>
<body>
<div class="row">
  <div>Status: <span id="status">?</span></div>
  <button id="toggle" onclick="toggleRun()">...</button>
  <button onclick="saveStory()">SAVE STORY</button>
  <button onclick="stopAgent()">STOP</button>
</div>
<div class="small">Latest image: <span id="imgname">(none)</span></div>
<textarea id="story" spellcheck="false"></textarea>
<img id="screenshot" src="/screenshot" />
<script>
const statusEl = document.getElementById('status');
const toggleBtn = document.getElementById('toggle');
const storyEl = document.getElementById('story');
const imgEl = document.getElementById('screenshot');
const imgNameEl = document.getElementById('imgname');

let lastToken = -1;
let lastPaused = true;

async function tick() {
  const d = await fetch('/state').then(r => r.json());
  lastPaused = !!d.paused;
  statusEl.textContent = lastPaused ? 'PAUSED' : 'RUNNING';
  toggleBtn.textContent = lastPaused ? 'RESUME' : 'PAUSE';
  storyEl.readOnly = !lastPaused;

  if (document.activeElement !== storyEl) {
    storyEl.value = d.story || '';
  }

  imgNameEl.textContent = d.latest_image || '(none)';
  if (typeof d.image_token === 'number' && d.image_token !== lastToken) {
    lastToken = d.image_token;
    imgEl.src = '/screenshot?ts=' + Date.now();
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

    def do_POST(self):  # noqa: N802
        if self.path == "/pause":
            self.server.hud_state.pause()  # type: ignore[attr-defined]
            self._send(200, "text/plain; charset=utf-8", b"ok")
            return
        if self.path == "/resume":
            self.server.hud_state.resume()  # type: ignore[attr-defined]
            self._send(200, "text/plain; charset=utf-8", b"ok")
            return
        if self.path == "/stop":
            self.server.hud_state.stop()  # type: ignore[attr-defined]
            self._send(200, "text/plain; charset=utf-8", b"ok")
            return
        if self.path == "/story":
            length = int(self.headers.get("Content-Length", "0") or "0")
            data = self.rfile.read(length) if length > 0 else b""
            text = data.decode("utf-8", errors="replace")
            self.server.hud_state.set_story(text.strip() or DEFAULT_HUD_TEXT)  # type: ignore[attr-defined]
            self._send(200, "text/plain; charset=utf-8", b"ok")
            return

        self._send(404, "text/plain; charset=utf-8", b"not found")


class RemoteServer:
    def __init__(self, *, port: int, hud_state: RemoteHUDState):
        self._srv = HTTPServer(("0.0.0.0", port), HTTPHandler)
        self._srv.hud_state = hud_state  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._srv.shutdown()


def call_vlm(png: bytes, hud_story: str) -> list[tuple[str, dict[str, Any]]]:
    prompt = (
        "Analyze the screenshot and the current HUD story.\n"
        "Return tool_calls (1..N). Each tool call must include a 'story' field containing the FULL rewritten narrative.\n\n"
        "CURRENT_HUD_STORY:\n"
        f"{hud_story}\n"
    )
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}"}},
                ],
            },
        ],
        "tools": TOOLS,
        "tool_choice": "required",
        **SAMPLING,
    }
    req = urllib.request.Request(API_URL, json.dumps(payload).encode("utf-8"), {"Content-Type": "application/json"})
    data = json.load(urllib.request.urlopen(req, timeout=120))
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("No choices in VLM response")
    msg = choices[0].get("message") or {}
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        raise ValueError("No tool calls in VLM response")
    out: list[tuple[str, dict[str, Any]]] = []
    for tc in tool_calls:
        fn = (tc.get("function") or {})
        name = str(fn.get("name") or "").strip().lower()
        if name not in TOOL_NAME_SET:
            raise ValueError(f"Unknown tool: {name!r}")
        args_raw = fn.get("arguments", "")
        if isinstance(args_raw, str):
            args = json.loads(args_raw) if args_raw.strip() else {}
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            raise ValueError(f"Invalid arguments type: {type(args_raw).__name__}")
        if not isinstance(args, dict):
            raise ValueError("Tool arguments must be an object")
        out.append((name, args))
    return out


def call_vlm_test(step: int) -> list[tuple[str, dict[str, Any]]]:
    ts = datetime.now().strftime("%H:%M:%S")
    story = f"FRANZ TEST\nCuriosity: high\nBoredom: low\nTS: {ts}\nSTEP: {step}\n"
    if step % 5 == 2:
        return [("click", {"x": 500, "y": 500, "story": story}), ("type_text", {"text": "TEST", "story": story})]
    if step % 5 == 3:
        return [("drag", {"x1": 300, "y1": 300, "x2": 700, "y2": 700, "story": story}), ("scroll", {"dy": -480, "story": story})]
    if step % 5 == 4:
        return [("right_click", {"x": 500, "y": 500, "story": story}), ("double_click", {"x": 500, "y": 500, "story": story})]
    if step % 5 == 1:
        return [("attend", {"targets": [{"x": 200, "y": 200, "label": "A"}], "story": story})]
    return [("attend", {"targets": [{"x": 500, "y": 500, "label": "Center"}], "story": story})]


def execute(tool: str, args: dict[str, Any], conv: Coord) -> None:
    match tool:
        case "click":
            mouse_click(*conv.to_screen(float(args["x"]), float(args["y"])), conv)
        case "right_click":
            mouse_right_click(*conv.to_screen(float(args["x"]), float(args["y"])), conv)
        case "double_click":
            mouse_double_click(*conv.to_screen(float(args["x"]), float(args["y"])), conv)
        case "drag":
            mouse_drag(*conv.to_screen(float(args["x1"]), float(args["y1"])), *conv.to_screen(float(args["x2"]), float(args["y2"])), conv)
        case "type_text":
            type_text(str(args["text"]))
        case "scroll":
            scroll(float(args["dy"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="FRANZ narrative-persistent AI agent (browser HUD)")
    parser.add_argument("--test", action="store_true", help="Enable test mode (simulated VLM responses)")
    parser.add_argument("--res", choices=["low", "med", "high"], default="high", help="Screen resolution preset")
    parser.add_argument("--port", type=int, default=REMOTE_PORT_DEFAULT, help="Browser HUD port")
    cli_args = parser.parse_args()

    test_mode = cli_args.test
    screen_w, screen_h = RES_PRESETS[cli_args.res]

    sw, sh = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    conv = Coord(sw=sw, sh=sh)

    dump = DUMP_FOLDER / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dump.mkdir(parents=True, exist_ok=True)

    print(f"FRANZ | Screen: {sw}x{sh} | Model: {screen_w}x{screen_h} | TEST_MODE={test_mode}")
    print(f"Dump: {dump}")
    print(f"PAUSED - open http://<machine-ip>:{cli_args.port} and click RESUME")
    print("Browser HUD: edit story while paused, click SAVE STORY")

    font_mono, _font_ui = create_shared_fonts()

    hud_state = RemoteHUDState(story=DEFAULT_HUD_TEXT, dump_dir=dump)
    hud_state.pause()  # start paused
    remote = RemoteServer(port=cli_args.port, hud_state=hud_state)
    remote.start()

    step = 0
    obs_mgr = ObsManager()

    try:
        while not hud_state.should_stop():
            hud_state.wait_unpaused()
            if hud_state.should_stop():
                break

            step += 1
            ts = datetime.now().strftime("%H:%M:%S")

            bgra = capture_screen(sw, sh)
            small = downsample(bgra, sw, sh, screen_w, screen_h)
            png = encode_png(small, screen_w, screen_h)

            img_name = f"step{step:03d}.png"
            (dump / img_name).write_bytes(png)
            hud_state.set_latest_image(img_name)

            current_story = hud_state.get_story()
            append_execution_log(dump, img_name, sw, sh, current_story)

            obs_mgr.hide_all()

            try:
                calls = call_vlm_test(step) if test_mode else call_vlm(png, current_story)
            except Exception as e:
                print(f"[{ts}] {step:03d} | VLM ERROR: {e}")
                time.sleep(1.0)
                continue

            obs_mgr.hide_all()
            last_story = current_story or DEFAULT_HUD_TEXT

            for idx, (tool, args) in enumerate(calls, 1):
                story = str(args.get("story") or "").strip()
                if story:
                    last_story = story

                print(f"[{ts}] {step:03d}.{idx:02d} | {tool}")

                if tool == "attend":
                    targets = args.get("targets")
                    if not isinstance(targets, list) or not targets:
                        targets = [{"x": 500, "y": 500, "label": "Default"}]
                    obs_mgr.show_multiple(targets, sw, sh, conv, font_mono, OBS_DEFAULT_COLOR)
                    continue

                try:
                    execute(tool, args, conv)
                except Exception as e:
                    print(f"[{ts}] {step:03d}.{idx:02d} | EXEC ERROR ({tool}): {e} | args={args}")
                    obs_mgr.show_multiple([{"x": 500, "y": 500, "label": "exec_error"}], sw, sh, conv, font_mono, OBS_DEFAULT_COLOR)
                    break

                time.sleep(0.05)

            hud_state.set_story(last_story)

            time.sleep(0.25)

        obs_mgr.hide_all()
    finally:
        remote.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nFRANZ stops.")
