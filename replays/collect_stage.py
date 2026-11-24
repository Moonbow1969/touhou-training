import argparse
import ctypes
from collections import Counter
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
TH14_STEAM_100B = "4609e701b677b8317e43be92041f0c7075a292182848a7c11215c3a226034f90"
FAST_FORWARD_KEY = "left ctrl"
TH14_REPLAY_MANAGER = 0xDB688
TH14_REPLAY_MODE_OFFSET = 16


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
        if not self.kernel.DebugSetProcessKillOnExit(False):
            error = ctypes.get_last_error()
            self.resume()
            raise OSError(error, "Could not set replay debugger detach behavior.")

    def resume(self):
        if self.frozen:
            if not self.kernel.DebugActiveProcessStop(self.pid):
                raise OSError(ctypes.get_last_error(), "Could not resume the replay process.")
            self.frozen = False

    def clock(self, reader):
        misc = self.offsets.misc
        manager = reader.read_uint(misc.ascii_manager_ptr, relative=True)
        return reader.read_uint(manager + misc.global_timer)

    def require_foreground(self):
        _, foreground_pid = win32process.GetWindowThreadProcessId(win32gui.GetForegroundWindow())
        if foreground_pid != self.pid:
            raise RuntimeError("The exact replay window must be in the foreground.")

    def require_stock_th14(self):
        executable = Path(psutil.Process(self.pid).exe())
        if sha256(executable.read_bytes()).hexdigest() != TH14_STEAM_100B:
            raise RuntimeError("This capture requires the verified stock Steam TH14 executable.")

    def require_th14_playback(self):
        pointer = int.from_bytes(self._read_game_memory(TH14_REPLAY_MANAGER, 4, rel=True), "little")
        if not pointer:
            raise RuntimeError("TH14 has no active replay manager.")
        mode = self._read_game_memory(pointer + TH14_REPLAY_MODE_OFFSET, 4, rel=False)
        if int.from_bytes(mode, "little") != 1:
            raise RuntimeError("TH14 must be playing a replay for the fast-forward capture.")

    def start_selected_replay(self):
        self.require_foreground()
        try:
            keyboard.press("z")
            sleep(0.05)
        finally:
            keyboard.release("z")

    def close(self):
        try:
            self.resume()
        finally:
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


def collect_stage(game, reader, expected, stage, rank, stride, progress_every=0, *,
                  fast_forward=False, record_clock=False):
    snapshots, labels, times, native = [], [], [], []
    controllable = []
    global_ticks = []
    last_active = perf_counter()
    first_active = None
    last_frame = -1
    while True:
        if fast_forward:
            game.require_foreground()
        start = game.clock(reader)
        game.resume()
        until = perf_counter() + 4
        while game.clock(reader) < start + stride:
            if perf_counter() >= until:
                raise TimeoutError("Replay clock stopped advancing.")
            if not fast_forward:
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
        if record_clock:
            global_ticks.append(game.clock(reader))
        native.append(reader.read_uint(reader.offsets.statics.input, relative=True))
        snapshots.append(snapshot.observation)
        labels.append((snapshot.player_state, snapshot.bullets))
        controllable.append(has_control(reader, pointer, snapshot.player_state))
        last_active = perf_counter()
        last_frame = frame
        if first_active is None:
            first_active = last_active
        if progress_every and len(times) % progress_every == 0:
            print(json.dumps({'stage': stage, 'native_frame': frame, 'snapshots': len(times),
                              'replay_frames': len(expected), 'elapsed': round(last_active - first_active, 3)}),
                  flush=True)
    if len(times) < 100:
        raise RuntimeError("Replay did not emit enough snapshots.")
    observations = {
        "bullets": np.stack([row["bullets"] for row in snapshots]),
        "context": np.stack([row["context"] for row in snapshots]),
        "native": np.asarray(native, dtype=np.uint16),
        "stage_frame": np.asarray(times),
        "player_state": np.asarray(labels)[:, 0],
        "bullet_count": np.asarray(labels)[:, 1],
        "controllable": np.asarray(controllable, dtype=np.bool_),
    }
    if record_clock:
        observations["global_tick"] = np.asarray(global_ticks, dtype=np.int64)
    return observations, last_active - first_active


def capture_timing(observations, stride, fast_forward):
    stage = np.diff(observations["stage_frame"])
    clock = np.diff(observations["global_tick"])
    exact = int(np.count_nonzero((stage == stride) & (clock == stride)))
    return {
        "capture_mode": "stock_th14_replay_fast_forward" if fast_forward else "stock_th14_replay",
        "consecutive_observations": len(stage),
        "exact_four_tick_observations": exact,
        "exact_four_tick_fraction": exact / len(stage),
        "stage_delta_histogram": dict(Counter(stage.tolist())),
        "global_delta_histogram": dict(Counter(clock.tolist())),
    }


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
    parser.add_argument("--progress-every", type=int, default=1000,
                        help="Log capture progress after this many sampled frames. Use zero to disable.")
    parser.add_argument("--start-selected", action="store_true")
    parser.add_argument("--from-stage", type=int, help="First stage to record with --all-stages.")
    parser.add_argument("--wait-for-stage", type=int, default=0,
                        help="Seconds to wait for the first requested stage during existing playback.")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fast-forward", action="store_true",
                        help="Measure stock TH14 replay fast-forward in a fresh, separate capture directory.")
    parser.add_argument("--fast-forward-through-stage", type=int,
                        help="Return to stock speed after this stage while keeping the replay running.")
    args = parser.parse_args()
    if args.progress_every < 0:
        parser.error("Capture progress interval must be nonnegative.")
    if args.stride < 1:
        parser.error("The stride must be positive.")
    if args.from_stage is not None and not args.all_stages:
        parser.error("--from-stage requires --all-stages.")
    manifest_path = args.manifest or ROOT / "expanded-corpus.json"
    if args.manifest is None and not manifest_path.exists():
        manifest_path = ROOT / "multigame-corpus.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = next(entry for entry in manifest if entry["name"] == args.filename)
    if args.fast_forward and (row["title"] != 14 or row["version"] != "1.00b" or args.stride != 4
                              or args.out is None or args.out.exists() or args.wait_for_stage <= 0):
        parser.error("Fast-forward requires stock TH14 1.00b, a four-tick stride, a fresh output and a stage wait.")
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
    if args.fast_forward_through_stage is not None and (
            not args.fast_forward or args.fast_forward_through_stage not in requested):
        parser.error("The fast-forward cutoff must be one of the requested stages.")
    path = ROOT / row["name"]
    if sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError("Replay SHA-256 differs from the corpus manifest.")
    words = {stage: np.frombuffer(body, dtype="<u2").reshape(-1, 3)[:, 0]
             for stage, body in action_streams(path, row["title"]) if stage in requested}
    if {stage: len(labels) for stage, labels in words.items()} != {
            stage: row["frames_by_stage"][str(stage)] for stage in requested}:
        raise ValueError("Decoded stage list or frame counts differ from the corpus manifest.")
    output = args.out or ROOT / PAIRED_DIRECTORY
    output.mkdir(parents=True, exist_ok=True)
    failures = []
    game = ReplayProcess(row["title"], args.pid)
    holding_fast_key = False
    try:
        if args.fast_forward:
            game.require_stock_th14()
        if args.start_selected:
            game.start_selected_replay()
        reader = StateReader(game, game.offsets, 96)
        if args.wait_for_stage:
            wait_for_stage(game, reader, requested[0], args.wait_for_stage)
        if args.fast_forward:
            game.require_foreground()
            game.require_th14_playback()
            holding_fast_key = True
            keyboard.press(FAST_FORWARD_KEY)
        print(f"Recording TH{row['title']} replay {row['name']}, stages {requested}", flush=True)
        for stage in requested:
            if holding_fast_key and args.fast_forward_through_stage is not None and stage > args.fast_forward_through_stage:
                keyboard.release(FAST_FORWARD_KEY)
                holding_fast_key = False
                print(f"Continuing the replay at stock speed from Stage {stage}.", flush=True)
            observations, elapsed = collect_stage(game, reader, words[stage], stage, rank, args.stride,
                                                  args.progress_every, fast_forward=holding_fast_key,
                                                  record_clock=args.fast_forward)
            timing = capture_timing(observations, args.stride, holding_fast_key) if args.fast_forward else {}
            try:
                paired, summary = align(args.filename, observations, words[stage], args.stride, elapsed)
                if holding_fast_key and timing["exact_four_tick_fraction"] < .99:
                    raise AlignmentError("Fast-forward capture failed the 99% exact-clock gate.", summary)
            except AlignmentError as error:
                failures.append(stage)
                rejected = output / "rejected" if args.fast_forward else ROOT / "rejected"
                rejected.mkdir(exist_ok=True)
                prefix = rejected / f"{Path(args.filename).stem}-stage{stage}"
                metadata = dict(row, **error.summary, **timing, stage=stage, stride=args.stride,
                                decoder=DECODER_VERSION, failure=str(error))
                save_stage(prefix, observations, metadata)
                print(f"Stage {stage} rejected. Raw evidence saved to {prefix.name}; continuing replay.", flush=True)
                continue
            prefix = output / f"{Path(args.filename).stem}-stage{stage}"
            save_stage(prefix, paired, dict(row, **summary, **timing, stage=stage, stride=args.stride,
                                           decoder=DECODER_VERSION))
    finally:
        try:
            if holding_fast_key:
                keyboard.release(FAST_FORWARD_KEY)
        finally:
            game.close()
    if failures:
        raise SystemExit(f"Replay completed with rejected stages: {failures}")


if __name__ == "__main__":
    main()
