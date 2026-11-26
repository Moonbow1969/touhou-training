"""Measure concurrent TH14 0.01b native rollouts over line-oriented worker pipes."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import json
from pathlib import Path
from queue import Empty, Queue
from statistics import median, quantiles
import subprocess
from threading import Barrier, Thread
from time import perf_counter


class TrialWorker:
    def __init__(self, command, shutdown_timeout=6):
        self.command = command
        self.shutdown_timeout = shutdown_timeout
        self.output = Queue()

    def __enter__(self):
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, encoding="utf-8", bufsize=1)
        Thread(target=self._drain, daemon=True).start()
        return self

    def _drain(self):
        try:
            for line in self.process.stdout:
                self.output.put(line)
        finally:
            self.output.put(None)

    def read(self, timeout=15):
        try:
            line = self.output.get(timeout=timeout)
        except Empty as error:
            raise RuntimeError("Native worker response timed out.") from error
        if line is None:
            raise RuntimeError("Native worker closed its response stream.")
        return json.loads(line)

    def request(self, command):
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        return self.read()

    def __exit__(self, *_):
        try:
            self.process.stdin.write("Q\n")
            self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=self.shutdown_timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()


def rates(seconds):
    return {"p50_ms": round(1000 * median(seconds), 2),
            "p95_ms": round(1000 * quantiles(seconds, n=20, method="inclusive")[-1], 2)}


def require_stage(state):
    if state["mode"] != 2 or state["stage"] != 1 or state["lives"] < 0:
        raise RuntimeError("Trial left the active first-stage benchmark.")


def benchmark(worker, episodes, decisions, barrier):
    warmup = perf_counter()
    try:
        state = worker.read(timeout=90)
        require_stage(state)
    except BaseException:
        barrier.abort()
        raise
    initial_frame = state["frame"]
    startup = perf_counter() - warmup
    barrier.wait(timeout=100)
    started = perf_counter()
    step_times, reset_times, reset_frames = [], [], []
    tick_deltas, stage_deltas = Counter(), Counter()
    horizontal, horizontal_responses, maximum_entities = 0, 0, 0
    for episode in range(episodes):
        if episode:
            reset_started = perf_counter()
            old_frame = state["frame"]
            state = worker.request("R")
            require_stage(state)
            if state["frame"] >= min(90, old_frame):
                raise RuntimeError("The original stage timer did not rewind on Retry.")
            reset_times.append(perf_counter() - reset_started)
            reset_frames.append(state["frame"])
        for index in range(decisions):
            left = (index // 20) % 2 == 0
            command = "S 49" if left else "S 50"  # Focus, Shoot, and one horizontal direction.
            start = perf_counter()
            following = worker.request(command)
            step_times.append(perf_counter() - start)
            require_stage(following)
            tick_deltas[following["delta"]] += 1
            stage_deltas[following["frame"] - state["frame"]] += 1
            maximum_entities = max(maximum_entities, following["bullet_count"])
            x = state["player"][0]
            # Exclude the opening movement lock, life-loss animation, and commands into an edge.
            if state["frame"] >= 100 and -183 < x < 183 and state["lives"] == following["lives"]:
                horizontal += 1
                dx = following["player"][0] - x
                horizontal_responses += int((-dx if left else dx) > .05)
            state = following
    elapsed = perf_counter() - started
    return {
        "decisions": len(step_times), "episodes": episodes, "elapsed_seconds": elapsed,
        "startup_seconds": startup, "initial_stage_frame": initial_frame,
        "decisions_per_second": len(step_times) / elapsed,
        "step_latency": rates(step_times), "reset_seconds": reset_times, "reset_frames": reset_frames,
        "global_tick_deltas": dict(sorted(tick_deltas.items())),
        "stage_frame_deltas": dict(sorted(stage_deltas.items())),
        "horizontal_commands": horizontal, "horizontal_responses": horizontal_responses,
        "max_projectile_count": maximum_entities,
        "observed_movement_and_projectiles": bool(maximum_entities and horizontal_responses),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=Path, required=True,
                        help="Private JSON mapping worker labels to their command argument arrays.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--worker-episodes", action="append", default=[], metavar="WORKER=COUNT",
                        help="Override a worker's episode budget to match heterogeneous sampling rates.")
    parser.add_argument("--decisions", type=int, default=96)
    args = parser.parse_args()
    commands = json.loads(args.workers.read_text(encoding="utf-8"))
    if not commands or not 1 <= args.episodes <= 100 or not 60 <= args.decisions <= 150:
        parser.error("Supply worker commands, 1–100 episodes, and 60–150 decisions per episode.")
    quotas = dict.fromkeys(commands, args.episodes)
    for override in args.worker_episodes:
        label, _, count = override.partition("=")
        if label not in quotas or not count.isdecimal() or not 1 <= int(count) <= 100:
            parser.error("Each worker episode quota must name a configured worker and request 1–100 episodes.")
        quotas[label] = int(count)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    barrier = Barrier(len(commands))
    rows, errors = {}, {}
    with ExitStack() as stack:
        workers = {label: stack.enter_context(TrialWorker(command)) for label, command in commands.items()}
        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            tasks = {pool.submit(benchmark, worker, quotas[label], args.decisions, barrier): label
                     for label, worker in workers.items()}
            for task in as_completed(tasks):
                label = tasks[task]
                try:
                    rows[label] = task.result()
                except Exception as error:
                    errors[label] = {"type": type(error).__name__, "message": str(error)}
                    print(f"{label}: {type(error).__name__}: {error}", flush=True)
    wall_time = max((row["elapsed_seconds"] for row in rows.values()), default=0)
    total_decisions = sum(row["decisions"] for row in rows.values())
    report = {"game": "TH14 0.01b", "task": "Normal Stage 1", "requested_frames_per_decision": 4,
              "episode_quotas": quotas, "decisions_per_episode": args.decisions,
              "parallel_seconds": wall_time if not errors else None,
              "aggregate_decisions_per_second": total_decisions / wall_time if not errors else None,
              "status": "incomplete" if errors else "complete",
              "workers": {label: rows[label] for label in commands if label in rows}, "errors": errors}
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
