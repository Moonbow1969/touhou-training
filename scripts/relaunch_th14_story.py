"""Inspect or gracefully relaunch a verified TH14 Steam process after an ending."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import struct
import subprocess
from time import monotonic, sleep

import psutil

from environment.structured import load_parakit_offsets
from scripts.checkpoint_storage import digest


STEAM_ROOT = Path("C:/Program Files (x86)/Steam")
STOCK_SHA256 = "4609e701b677b8317e43be92041f0c7075a292182848a7c11215c3a226034f90"
WINDOW_TITLE = "Double Dealing Character. ver 1.00b"
APP_ID = "1043240"


def process_identity(process, expected_executable):
    executable = Path(process.exe()).resolve(strict=True)
    if executable != expected_executable:
        raise RuntimeError("The process is outside the verified TH14 Steam installation.")
    fingerprint = digest(executable)
    if fingerprint != STOCK_SHA256:
        raise RuntimeError("The process executable does not match the stock TH14 1.00b build.")
    return dict(pid=process.pid, created_at=process.create_time(), exe_sha256=fingerprint)


def native_window(pid, win32gui, win32process):
    matches = []

    def collect(hwnd, _):
        if (win32gui.IsWindowVisible(hwnd)
                and WINDOW_TITLE in win32gui.GetWindowText(hwnd)
                and win32process.GetWindowThreadProcessId(hwnd)[1] == pid):
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            if (right - left, bottom - top) == (646, 509):
                matches.append(hwnd)
        return True

    win32gui.EnumWindows(collect, None)
    if len(matches) != 1:
        raise RuntimeError("Expected one verified 640×480 TH14 window on the claimed process.")
    return matches[0]


def native_state(pid, parakit, executable, win32api, win32con, win32process):
    handle = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, False, pid)
    try:
        bases = [base for base in win32process.EnumProcessModules(handle)
                 if Path(win32process.GetModuleFileNameEx(handle, base)).resolve() == executable]
        if len(bases) != 1:
            raise RuntimeError("Unable to identify the installed stock module in the claimed process.")

        def read(offset, signed=False):
            raw = ctypes.create_string_buffer(4)
            length = ctypes.c_size_t()
            ok = ctypes.windll.kernel32.ReadProcessMemory(
                ctypes.c_void_p(int(handle)), ctypes.c_void_p(bases[0] + offset),
                raw, ctypes.sizeof(raw), ctypes.byref(length),
            )
            if not ok or length.value != 4:
                raise RuntimeError("The claimed native process state could not be read.")
            return struct.unpack("<i" if signed else "<I", raw.raw)[0]

        data = load_parakit_offsets(parakit)
        return dict(stage=read(data.environment.stage), screen=read(data.statics.game_screen),
                    pause=read(data.statics.pause_state), continues=read(data.statics.continues),
                    lives=read(data.statics.lives, signed=True), bombs=read(data.statics.bombs),
                    life_fragments=read(data.statics.life_pieces), bomb_fragments=read(data.statics.bomb_pieces),
                    power=read(data.statics.power), score=read(data.statics.score),
                    difficulty=read(data.environment.difficulty), character=read(data.environment.character),
                    subshot=read(data.environment.subshot))
    finally:
        handle.Close()


def is_ending(state):
    return (state["stage"], state["screen"], state["continues"], state["difficulty"],
            state["character"], state["subshot"]) == (6, 15, 0, 1, 0, 0) and state["lives"] >= 0


def stock_processes(executable):
    for process in psutil.process_iter(["exe"]):
        path = process.info["exe"]
        if path and Path(path).resolve() == executable:
            yield process


def wait_new_window(executable, old_pid, timeout, win32gui, win32process):
    deadline = monotonic() + timeout
    previous = None
    while monotonic() < deadline:
        matches = [process for process in stock_processes(executable) if process.pid != old_pid]
        if len(matches) > 1:
            raise RuntimeError("Multiple stock TH14 processes started; none was selected.")
        if matches:
            process = matches[0]
            try:
                window = native_window(process.pid, win32gui, win32process)
                identity = process_identity(process, executable)
            except (RuntimeError, psutil.NoSuchProcess):
                previous = None
            else:
                signature = (window, identity)
                if signature == previous:
                    return process, identity
                previous = signature
        sleep(.5)
    raise TimeoutError("Steam did not supply a stable verified stock TH14 window.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--created-at", type=float, help="Exact process creation time returned by inspection.")
    parser.add_argument("--steam-root", type=Path, default=STEAM_ROOT)
    parser.add_argument("--parakit", type=Path, default=Path(__file__).resolve().parents[2] / "parakit")
    parser.add_argument("--timeout", type=float, default=30.)
    parser.add_argument("--apply", action="store_true",
                        help="Gracefully close the claimed completed TH14 process and relaunch it through Steam.")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("This tool supports the stock Windows Steam installation.")
    if args.pid < 1 or not 0 < args.timeout <= 120 or args.apply and args.created_at is None:
        parser.error("Apply requires the claimed PID and its inspected process creation time.")
    import win32api
    import win32con
    import win32gui
    import win32process

    steam = args.steam_root / "steam.exe"
    executable = (args.steam_root / "steamapps/common/th14/th14.exe").resolve(strict=True)
    process = psutil.Process(args.pid)
    identity = process_identity(process, executable)
    hwnd = native_window(args.pid, win32gui, win32process)
    state = native_state(args.pid, args.parakit, executable, win32api, win32con, win32process)
    if not args.apply:
        print(json.dumps(dict(process=identity, native_state=state, relaunch_eligible=is_ending(state))))
        return
    if identity["created_at"] != args.created_at or not is_ending(state):
        raise RuntimeError("The claimed process is not on the verified zero-Continue ending.")
    if not steam.is_file() or [candidate.pid for candidate in stock_processes(executable)] != [args.pid]:
        raise RuntimeError("Expected the claimed process and a single stock Steam install.")
    if (process_identity(process, executable) != identity
            or native_window(args.pid, win32gui, win32process) != hwnd
            or not is_ending(native_state(args.pid, args.parakit, executable, win32api, win32con, win32process))):
        raise RuntimeError("The claimed process changed before the graceful close.")
    win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
    try:
        process.wait(timeout=min(args.timeout, 15.))
    except psutil.TimeoutExpired as error:
        raise RuntimeError("The verified game did not close gracefully. No force-close was attempted.") from error
    subprocess.Popen([str(steam), "-applaunch", APP_ID], cwd=str(args.steam_root))
    new_process, new_identity = wait_new_window(executable, args.pid, args.timeout, win32gui, win32process)
    opened = native_state(new_process.pid, args.parakit, executable, win32api, win32con, win32process)
    fresh = ((opened["screen"], opened["pause"], opened["stage"], opened["continues"], opened["difficulty"],
              opened["character"], opened["subshot"]) == (7, 2, 1, 0, 1, 0, 0)
             and all(opened[key] == value for key, value in dict(
                 lives=2, bombs=3, life_fragments=0, bomb_fragments=0, power=100, score=0,
             ).items()))
    print(json.dumps(dict(status="verified_fresh_credit" if fresh else "verified_window_relaunched",
                          old_process=identity, new_process=new_identity, native_state=opened)))


if __name__ == "__main__":
    main()
