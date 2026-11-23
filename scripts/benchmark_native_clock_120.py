"""Run the native clock benchmark at 120 Hz for the verified Steam TH14 binary."""

import ctypes as C
from ctypes import wintypes as W
import hashlib
from pathlib import Path
import struct
import sys

from scripts.benchmark_native_clock import main


def run_variant():
    from environment import interface as game

    kernel = C.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (W.DWORD, W.BOOL, W.DWORD)
    kernel.OpenProcess.restype = W.HANDLE
    kernel.VirtualAllocEx.argtypes = (W.HANDLE, C.c_void_p, C.c_size_t, W.DWORD, W.DWORD)
    kernel.VirtualAllocEx.restype = C.c_void_p
    kernel.VirtualFreeEx.argtypes = (W.HANDLE, C.c_void_p, C.c_size_t, W.DWORD)
    kernel.VirtualProtectEx.argtypes = (W.HANDLE, C.c_void_p, C.c_size_t, W.DWORD, C.POINTER(W.DWORD))
    kernel.WriteProcessMemory.argtypes = (
        W.HANDLE, C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t),
    )
    kernel.FlushInstructionCache.argtypes = (W.HANDLE, C.c_void_p, C.c_size_t)
    kernel.CloseHandle.argtypes = (W.HANDLE,)

    hash_expected = "4609e701b677b8317e43be92041f0c7075a292182848a7c11215c3a226034f90"
    process = kernel.OpenProcess(0x0438, False, game._game_pid)
    if not process:
        raise C.WinError(C.get_last_error())
    remote = None
    pointer_restored = True
    operand = game._base_address + 0x6A792
    try:
        module = game.win32process.GetModuleFileNameEx(process, game._base_address)
        if hashlib.sha256(Path(module).read_bytes()).hexdigest() != hash_expected:
            raise RuntimeError("Unsupported original executable.")
        original = game._read_game_memory(0x6A792, 4)
        original_pointer = struct.unpack("<I", original)[0]
        if game._base_address != 0x400000 or original_pointer != 0x4C1A18:
            raise RuntimeError("The stock limiter is already modified.")
        if game._read_game_memory(0xD9159, 1) != bytes([3]):
            raise RuntimeError("Enable the stock Fast latency mode before testing.")
        original_period = struct.unpack("<d", game._read_game_memory(original_pointer, 8, False))[0]
        if original_period != 1 / 60:
            raise RuntimeError("The stock limiter period is not 1/60 seconds.")

        def write(address, data):
            count = C.c_size_t()
            buffer = C.create_string_buffer(data)
            if not kernel.WriteProcessMemory(process, address, buffer, len(data), C.byref(count)):
                raise C.WinError(C.get_last_error())
            if count.value != len(data):
                raise RuntimeError("Partial process write.")

        def replace_pointer(target):
            old = W.DWORD()
            if not kernel.VirtualProtectEx(process, operand, 4, 0x40, C.byref(old)):
                raise C.WinError(C.get_last_error())
            try:
                write(operand, target)
                if not kernel.FlushInstructionCache(process, operand, 4):
                    raise C.WinError(C.get_last_error())
            finally:
                unused = W.DWORD()
                if not kernel.VirtualProtectEx(process, operand, 4, old.value, C.byref(unused)):
                    raise C.WinError(C.get_last_error())

        remote = kernel.VirtualAllocEx(process, None, 8, 0x3000, 0x04)
        if not remote or remote > 0xFFFFFFFF:
            raise RuntimeError("The game could not allocate a 32-bit limiter constant.")
        write(remote, struct.pack("<d", 1 / 120))
        pointer_restored = False
        try:
            replace_pointer(struct.pack("<I", remote))
            game.require_game_focus()
            if game.read_game_val("game_state") == 1:
                game._press_and_release("z")
            main()
        finally:
            replace_pointer(original)
            pointer_restored = game._read_game_memory(0x6A792, 4) == original
            if not pointer_restored:
                raise RuntimeError("The stock limiter pointer was not restored.")
            print("Restored and verified the original 60 Hz limiter.", flush=True)
    finally:
        if remote and pointer_restored:
            if not kernel.VirtualFreeEx(process, remote, 0, 0x8000):
                print("The detached probe allocation will be freed when TH14 exits.", file=sys.stderr)
        kernel.CloseHandle(process)


if __name__ == "__main__":
    run_variant()
