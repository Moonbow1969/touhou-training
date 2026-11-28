"""Wait for the two signed original-source ledgers and run their registered TH07 PPO fit."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def read(path):
    return json.loads(path.read_bytes())


def ledger(path, scenes, out):
    data = read(path)
    rows = []
    seen = set()
    for entry in sorted(data["rows"], key=lambda row: row["seed"]):
        seed = entry["seed"]
        if seed in seen or seed not in scenes:
            raise ValueError(f"Unexpected or duplicated seed in {path.name}: {seed}")
        seen.add(seed)
        if not entry["qualified"]:
            continue
        scene = scenes[seed]
        if entry.get("scene", scene) != scene:
            raise ValueError(f"Source scene differs from its original reservation: {seed}")
        row = dict(entry, scene=scene)
        for key in ("file", "receipt"):
            row[key] = os.path.relpath(path.parent / entry[key], out).replace(os.sep, "/")
        rows.append(row)
    return rows


def wait_for_sources(paths, hours):
    deadline = time.monotonic() + 3600 * hours
    last = None
    while time.monotonic() < deadline:
        data = [read(path) if path.is_file() else {} for path in paths]
        status = [(d.get("complete", False), d.get("qualified_sources", 0), len(d.get("rows", ()))) for d in data]
        if status != last:
            print("Original source ledgers (complete, qualified, total rows):", status, flush=True)
            last = status
        if all(d.get("complete") is True for d in data):
            return
        time.sleep(10)
    raise TimeoutError("The registered original-game input ledgers did not seal within the collection window")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cold", "practice", "protocol", "release", "parent", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--hours", type=float, default=4.)
    args = parser.parse_args()
    sources = (args.cold.resolve(), args.practice.resolve())
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "policy").exists() or (out / "manifest.json").exists():
        raise FileExistsError(f"The fitter already has a source manifest in {out}")
    protocol, release = read(args.protocol), read(args.release)
    if release["branch"] != "normal-practice":
        raise ValueError("Expected the released Normal-practice branch")
    wait_for_sources(sources, args.hours)
    roster = release["roster"]
    first, last = roster["first_train_seed"], roster["last_train_seed"]
    scenes = release["practices"]
    practice = {seed: scenes[(seed-first) % len(scenes)] for seed in range(first, last+1)}
    cold = dict.fromkeys(protocol["common_normal_train_seeds"], "normal")
    cold.update(dict.fromkeys(range(21749117, 21749121), "normal"))
    all_rows = ledger(sources[0], cold, out)[:len(protocol["common_normal_train_seeds"])]
    all_rows.extend(ledger(sources[1], practice, out))
    held = [seed for interval in release["held_seed_reservations"].values()
            for seed in range(interval[0], interval[1]+1)]
    manifest = out / "manifest.json"
    freeze(manifest, dict(mode=release["branch"], protocol_sha256=digest(args.protocol),
        launch_sha256=digest(args.release), original_ledger_sha256=[digest(path) for path in sources],
        held_seeds=held, rows=all_rows))
    subprocess.run([sys.executable, "-u", "-m", "scripts.train_th07_curriculum_ppo", "--manifest", str(manifest),
                    "--parent", str(args.parent.resolve()), "--out", str(out / "policy")], check=True)


if __name__ == "__main__":
    main()
