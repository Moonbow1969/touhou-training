"""Run the registered two-phase TH07 PPO once the original source ledgers seal."""

import argparse
from pathlib import Path
import subprocess
import sys

from scripts.checkpoint_storage import digest
from scripts.run_th07_curriculum_when_ready import ledger, read, wait_for_sources
from scripts.train_five_original_games import freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("intake_root", "protocol", "parent", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--hours", type=float, default=3.)
    args = parser.parse_args()
    root, out = args.intake_root.resolve(), args.out.resolve()
    protocol = read(args.protocol)
    sources = [root / name / "intake-index.json" for name in (
        "letty-nonspell2", "letty-nonspell2-extension", "letty-final", "letty-final-extension")]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "policy").exists() or (out / "manifest.json").exists():
        raise FileExistsError(f"The fitter already has a source manifest in {out}")
    wait_for_sources(sources[:1], args.hours)
    screen = read(sources[0])["rows"]
    successes = [row for row in screen if row.get("outcome") == "practice_stage_clear"]
    if {row["seed"] for row in screen} != set(range(21749701,21749709)):
        raise ValueError("The eight fixed frontier sources were not independently accounted for")
    if not 2 <= len(successes) <= 6 or not any(row["qualified"] for row in successes):
        print(f"This frontier fit was not released by the eight-start screen ({len(successes)} objective completions).", flush=True)
        return
    wait_for_sources(sources, args.hours)
    scenes = {seed: row["scene"] for row in protocol["allocation"]
              for first, last in row["train_seed_ranges"] for seed in range(first, last+1)}
    all_rows = []
    seen = set()
    for path in sources:
        keys = {row["seed"] for row in read(path)["rows"]}
        if keys & seen:
            raise ValueError(f"Training original was accounted for twice: {path}")
        seen.update(keys)
        all_rows.extend(ledger(path, scenes, out))
    if set(scenes) != seen:
        raise ValueError("The conditional original trajectory roster does not match its sealed ledgers")
    first, last = protocol["comparison"]["held_seed_range"]
    manifest = out / "manifest.json"
    freeze(manifest, dict(mode="second-nonspell", protocol_sha256=digest(args.protocol),
        original_ledger_sha256=[digest(path) for path in sources], held_seeds=list(range(first,last+1)), rows=all_rows))
    subprocess.run([sys.executable, "-u", "-m", "scripts.train_th07_curriculum_ppo", "--manifest", str(manifest),
                    "--parent", str(args.parent.resolve()), "--out", str(out / "policy")], check=True)


if __name__ == "__main__":
    main()
