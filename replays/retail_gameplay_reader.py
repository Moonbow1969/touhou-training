"""Read-only stock TH10 1.00a / TH12 1.00b replay telemetry on Windows."""

from dataclasses import dataclass
import ctypes
from ctypes import wintypes
from pathlib import Path
from struct import Struct

import psutil
import win32api
import win32con
import win32process

from .collect_replay_pixels import digest


@dataclass(frozen=True)
class Layout:
    executable_sha256: str
    stage: int
    player_pointer: int
    replay_pointer: int
    playback_word: int
    clock_offset: int
    replay_stage_offset: int
    float_xy_offset: int
    fixed_xy_offset: int
    coordinate_scale: int
    focus_bit: int


LAYOUTS = {
    10: Layout(
        "3bdb72cf3d7c33c183359d368c801490dbcf54e6b3b2f060b95d72250b6866a3",
        0x474C7C,
        0x477834,
        0x477838,
        0x474E5C,
        0x1C8,
        0x1D0,
        0x3C0,
        0x3CC,
        100,
        4,
    ),
    12: Layout(
        "eb275a8c285fd8450fa5ac79fa730a759aeca8b2904aeb16d30e65859372a459",
        0x4B0CB0,
        0x4B4514,
        0x4B4518,
        0x4D49D0,
        0x1D0,
        0x1D8,
        0x97C,
        0x988,
        128,
        8,
    ),
}

U32, I32, U16 = Struct("<I"), Struct("<i"), Struct("<H")
FLOAT_XY, FIXED_XY = Struct("<ff"), Struct("<ii")


@dataclass(frozen=True)
class PlaybackState:
    tick: int
    word: int
    stage: int
    replay_stage: int
    xy: tuple[float, float]
    fixed_xy: tuple[float, float]


class StockReplayReader:
    def __init__(self, title: int, pid: int):
        self.layout = LAYOUTS[title]
        expected = f"th{title}.exe"
        image = Path(psutil.Process(pid).exe())
        if (
            image.name.lower() != expected
            or digest(image) != self.layout.executable_sha256
        ):
            raise ValueError("Unrecognized original retail build.")
        self.handle = win32api.OpenProcess(
            win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, False, pid
        )
        base = next(
            module
            for module in win32process.EnumProcessModules(self.handle)
            if Path(win32process.GetModuleFileNameEx(self.handle, module)).name.lower()
            == expected
        )
        self.relocation = base - 0x400000
        self.read_memory = ctypes.WinDLL(
            "kernel32", use_last_error=True
        ).ReadProcessMemory
        self.read_memory.argtypes = [
            wintypes.HANDLE,
            wintypes.LPCVOID,
            wintypes.LPVOID,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        self.read_memory.restype = wintypes.BOOL

    def read(self, address, layout):
        buffer = ctypes.create_string_buffer(layout.size)
        if not self.read_memory(int(self.handle), address, buffer, layout.size, None):
            raise OSError(
                ctypes.get_last_error(), "Original game scene changed during read."
            )
        return layout.unpack(buffer.raw)

    def snapshot(self):
        layout = self.layout
        (stage,) = self.read(self.relocation + layout.stage, I32)
        (player,) = self.read(self.relocation + layout.player_pointer, U32)
        (replay,) = self.read(self.relocation + layout.replay_pointer, U32)
        if stage < 1 or stage > 7 or not player or not replay:
            return None
        (tick,) = self.read(replay + layout.clock_offset, I32)
        (replay_stage,) = self.read(replay + layout.replay_stage_offset, I32)
        (word,) = self.read(self.relocation + layout.playback_word, U16)
        xy = self.read(player + layout.float_xy_offset, FLOAT_XY)
        fixed = self.read(player + layout.fixed_xy_offset, FIXED_XY)
        return PlaybackState(
            tick,
            word,
            stage,
            replay_stage,
            xy,
            tuple(value / layout.coordinate_scale for value in fixed),
        )

    def close(self):
        self.handle.Close()
