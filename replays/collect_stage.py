import argparse
import ctypes
from ctypes import wintypes
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter, sleep

import keyboard
import numpy as np
import psutil
import win32api
import win32con
import win32gui
import win32process


from environment.structured import StateReader, load_parakit_offsets
from .inspect_rpy import DECODER_VERSION, PAIRED_DIRECTORY, ROOT, action_streams


CONTROL_BITS = 0xFB


class AlignmentError(RuntimeError):
    def __init__(self, message, summary):
        super().__init__(message)
        self.summary = summary


class ReplayProcess:
    def __init__(self, title, pid=None):
        image = f"th{title}.exe"
        matches = [proc for proc in psutil.process_iter(["name"]) if proc.info["name"] == image
                   and (pid is None or proc.pid == pid)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one running {image}, found {len(matches)}.")
        self.pid = matches[0].pid
        self.offsets = load_parakit_offsets(Path(__file__).resolve().parents[2] / "parakit", image)
        self.handle = win32api.OpenProcess(
            win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, False, self.pid
        )
        self.base = next(
            module for module in win32process.EnumProcessModules(self.handle)
            if Path(win32process.GetModuleFileNameEx(self.handle, module)).name.lower() == image
        )
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.memory = self.kernel.ReadProcessMemory
        self.memory.argtypes = [wintypes.HANDLE, wintypes.LPCVOID, wintypes.LPVOID,
                                ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
        self.memory.restype = wintypes.BOOL
        self.frozen = False

    def _read_game_memory(self, address, size, *, rel):
        buffer = ctypes.create_string_buffer(size)
        address = self.base + address if rel else address
        if not self.memory(int(self.handle), address, buffer, size, None):
            raise OSError(ctypes.get_last_error(), f"Cannot read game state at {address:#x}.")
        return buffer.raw

    def read_game_val(self, field):
        address = getattr(self.offsets.statics, field)
        return int.from_bytes(self._read_game_memory(address, 4, rel=True), "little", signed=True)

    def freeze(self):
        if not self.kernel.DebugActiveProcess(self.pid):
            raise RuntimeError("Could not pause the replay process.")
        self.frozen = True
        self.kernel.DebugSetProcessKillOnExit(False)

    def resume(self):
        if self.frozen:
            self.kernel.DebugActiveProcessStop(self.pid)
            self.frozen = False

    def clock(self, reader):
        misc = self.offsets.misc
        manager = reader.read_uint(misc.ascii_manager_ptr, relative=True)
        return reader.read_uint(manager + misc.global_timer)

    def start_selected_replay(self):
        _, foreground_pid = win32process.GetWindowThreadProcessId(win32gui.GetForegroundWindow())
        if foreground_pid != self.pid:
            raise RuntimeError("The replay menu must be foreground to send its start input.")
        keyboard.press("z")
        sleep(0.05)
        keyboard.release("z")

    def close(self):
        self.resume()
        self.handle.Close()


def wait_for_stage(game, reader, stage, timeout):
    deadline = perf_counter() + timeout
    while perf_counter() < deadline:
        if game.read_game_val("pause_state") == 2:
            current = reader.read_uint(reader.offsets.environment.stage, relative=True)
            if current == stage:
                return
            if current > stage:
                raise RuntimeError(f"Playback has already passed stage {stage}.")
        sleep(0.025)
    raise TimeoutError(f"Playback did not enter stage {stage} in {timeout}s.")


def has_control(reader, thread_pointer, player_state):
    gui = reader.offsets.gui
    thread = reader.offsets.game_thread
    gui_pointer = reader.read_uint(gui.gui_pointer, relative=True)
    in_dialogue = gui_pointer and reader.read_uint(gui_pointer + gui.zGui_dialogue)
    ending = reader.read_uint(thread_pointer + thread.zGameThread_flags) & thread.zGameThreadFlags_ending
    return player_state == 1 and not (in_dialogue or ending)


def collect_stage(game, reader, expected, stage, rank, stride):
    snapshots, labels, times, native = [], [], [], []
    controllable = []
    last_active = perf_counter()
    first_active = None
    last_frame = -1
    while True:
        start = game.clock(reader)
        game.resume()
        until = perf_counter() + 4
        while game.clock(reader) < start + stride:
            if perf_counter() >= until:
                raise TimeoutError("Replay clock stopped advancing.")
            sleep(0.0005)
        game.freeze()
        active = game.read_game_val("pause_state") == 2
        if not active:
            if perf_counter() - last_active > 6:
                break
            continue
        current_stage = reader.read_uint(reader.offsets.environment.stage, relative=True)
        if current_stage != stage:
            if times:
                break
            if perf_counter() - last_active > 6:
                break
            continue
        thread = reader.offsets.game_thread
        pointer = reader.read_uint(thread.game_thread_pointer, relative=True)
        if not pointer:
            continue
        frame = reader.read_uint(pointer + thread.zGameThread_stage_timer)
        if frame <= last_frame:
            if perf_counter() - last_active > 6:
                break
            continue
        if frame > len(expected) + 50:
            break
        snapshot = reader.snapshot()
        if snapshot.difficulty != rank:
            raise RuntimeError(f"Difficulty mismatch {snapshot.difficulty} vs expected {rank}.")
        times.append(frame)
        native.append(reader.read_uint(reader.offsets.statics.input, relative=True))
        snapshots.append(snapshot.observation)
        labels.append((snapshot.player_state, snapshot.bullets))
        controllable.append(has_control(reader, pointer, snapshot.player_state))
        last_active = perf_counter()
        last_frame = frame
        if first_active is None:
            first_active = last_active
    if len(times) < 100:
        raise RuntimeError("Replay did not emit enough snapshots.")
    return {
        "bullets": np.stack([row["bullets"] for row in snapshots]),
        "context": np.stack([row["context"] for row in snapshots]),
        "native": np.asarray(native, dtype=np.uint16),
        "stage_frame": np.asarray(times),
        "player_state": np.asarray(labels)[:, 0],
        "bullet_count": np.asarray(labels)[:, 1],
        "controllable": np.asarray(controllable, dtype=np.bool_),
    }, last_active - first_active


def align(filename, arrays, words, stride, elapsed):
    live = arrays["native"] & CONTROL_BITS
    frames = arrays["stage_frame"]
    eligible = arrays.get("controllable", arrays["player_state"] == 1)
    scores = []
    for offset in range(-100, 101):
        indices = frames + offset
        legal = (indices >= 0) & (indices + stride < len(words))
        candidate = words[indices[legal]] & CONTROL_BITS
        motion = (((candidate | live[legal]) & 0xF0) != 0) & eligible[legal]
        if motion.sum() > 100:
            agreement = float(np.mean(candidate[motion] == live[legal][motion]))
            scores.append((agreement, int(motion.sum()), offset))
    if not scores:
        raise AlignmentError(f"Could not match decoded actions for {filename}.",
                             {"replay": filename, "snapshots": len(frames)})
    agreement, moving, offset = max(scores)
    indices = frames + offset
    legal = (indices >= 0) & (indices + stride < len(words))
    keep = legal.copy()
    keep[legal] &= (words[indices[legal]] & CONTROL_BITS) == live[legal]
    keep &= eligible
    coverage = (int(frames[-1]) - int(frames[0])) / len(words)
    summary = {
        "replay": filename, "snapshots": len(frames), "pairs": int(keep.sum()),
        "moving_alignment": agreement, "moving_frames": moving, "offset": offset,
        "controllable_snapshots": int(eligible.sum()),
        "coverage": coverage, "first_frame": int(frames[0]), "last_frame": int(frames[-1]),
        "elapsed": elapsed,
    }
    print(json.dumps(summary), flush=True)
    if agreement < 0.95 or coverage < 0.9:
        raise AlignmentError("Replay alignment or complete-stage coverage failed its check.", summary)
    paired = {name: values[keep] for name, values in arrays.items()}
    paired["expert_action_chunk"] = words[indices[keep, None] + np.arange(1, stride + 1)[None, :]]
    return paired, summary


def save_stage(prefix, paired, metadata):
    data_path = prefix.with_suffix(".npz")
    metadata_path = prefix.with_suffix(".json")
    if data_path.is_file() and metadata_path.is_file():
        stored = json.loads(metadata_path.read_text(encoding="utf-8"))
        if stored["sha256"] != metadata["sha256"]:
            raise ValueError(f"Existing stage has a different replay hash: {metadata_path.name}")
        return
    suffix = f".{os.getpid()}.tmp"
    data_tmp = data_path.with_suffix(suffix)
    metadata_tmp = metadata_path.with_suffix(f".json{suffix}")
    with data_tmp.open("wb") as stream:
        np.savez_compressed(stream, **paired)
    metadata_tmp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    data_tmp.replace(data_path)
    metadata_tmp.replace(metadata_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--stage", type=int)
    stages.add_argument("--all-stages", action="store_true",
                        help="Record the replay from its first stage through the final stage.")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--start-selected", action="store_true")
    parser.add_argument("--from-stage", type=int, help="First stage to record with --all-stages.")
    parser.add_argument("--wait-for-stage", type=int, default=0,
                        help="Seconds to wait for the first requested stage during existing playback.")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / PAIRED_DIRECTORY)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("The stride must be positive.")
    if args.from_stage is not None and not args.all_stages:
        parser.error("--from-stage requires --all-stages.")
    manifest_path = args.manifest or ROOT / "expanded-corpus.json"
    if args.manifest is None and not manifest_path.exists():
        manifest_path = ROOT / "multigame-corpus.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = next(entry for entry in manifest if entry["name"] == args.filename)
    levels = ["Easy", "Normal", "Hard", "Lunatic", "Extra"]
    rank = levels.index(row["difficulty"])
    available = sorted(map(int, row["frames_by_stage"]))
    requested = available if args.all_stages else [args.stage or available[0]]
    if args.from_stage is not None:
        requested = [stage for stage in requested if stage >= args.from_stage]
        if not requested or requested[0] != args.from_stage:
            parser.error(f"Choose from the replay's stages {available}.")
    if not set(requested).issubset(available):
        parser.error(f"Choose from the replay's stages {available}.")
    path = ROOT / row["name"]
    if sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError("Replay SHA-256 differs from the corpus manifest.")
    words = {stage: np.frombuffer(body, dtype="<u2").reshape(-1, 3)[:, 0]
             for stage, body in action_streams(path, row["title"]) if stage in requested}
    if {stage: len(labels) for stage, labels in words.items()} != {
            stage: row["frames_by_stage"][str(stage)] for stage in requested}:
        raise ValueError("Decoded stage list or frame counts differ from the corpus manifest.")
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    failures = []
    game = ReplayProcess(row["title"], args.pid)
    try:
        if args.start_selected:
            game.start_selected_replay()
        reader = StateReader(game, game.offsets, 96)
        if args.wait_for_stage:
            wait_for_stage(game, reader, requested[0], args.wait_for_stage)
        print(f"Recording TH{row['title']} replay {row['name']}, stages {requested}", flush=True)
        for stage in requested:
            observations, elapsed = collect_stage(game, reader, words[stage], stage, rank, args.stride)
            try:
                paired, summary = align(args.filename, observations, words[stage], args.stride, elapsed)
            except AlignmentError as error:
                failures.append(stage)
                rejected = ROOT / "rejected"
                rejected.mkdir(exist_ok=True)
                prefix = rejected / f"{Path(args.filename).stem}-stage{stage}"
                metadata = dict(row, **error.summary, stage=stage, stride=args.stride,
                                decoder=DECODER_VERSION, failure=str(error))
                save_stage(prefix, observations, metadata)
                print(f"Stage {stage} rejected. Raw evidence saved to {prefix.name}; continuing replay.", flush=True)
                continue
            prefix = output / f"{Path(args.filename).stem}-stage{stage}"
            save_stage(prefix, paired, dict(row, **summary, stage=stage, stride=args.stride, decoder=DECODER_VERSION))
    finally:
        game.close()
    if failures:
        raise SystemExit(f"Replay completed with rejected stages: {failures}")


if __name__ == "__main__":
    main()
