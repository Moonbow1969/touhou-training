"""Seal two original-game actor cohorts at the same policy version and run one joint update."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def combine(protocol_path, parent, cohorts, output):
    protocol = json.loads(protocol_path.read_bytes())
    with np.load(parent, allow_pickle=False) as model:
        parent_hash = hashlib.sha256(model["weights"].astype("<f8").tobytes()).hexdigest()
    if parent_hash != protocol["common_actor_sha256"]:
        raise ValueError("Initial weights differ from the sampling protocol.")
    records = []
    for title, receipt in cohorts.items():
        rows = json.loads(receipt.read_bytes())
        if len(rows) != protocol["credits_per_title"]:
            raise ValueError(f"Incomplete TH{title} cohort.")
        seeds = [row["seed"] for row in rows]
        if sorted(seeds) != sorted(protocol["seeds"][str(title)]):
            raise ValueError(f"Missing or repeated TH{title} controller seeds.")
        for row in rows:
            if row["title"] != title or row["actor_sha256"] != parent_hash:
                raise ValueError(f"Unexpected title or behavior policy in {receipt.name}")
            source = (receipt.parent/row["behavior"]).resolve(strict=True)
            if hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Changed behavior in {row['episode']}")
            records.append({**row, "behavior": Path(os.path.relpath(source, output)).as_posix()})
    # An existing output is immutable, including if a failed update left a sealed batch.
    output.mkdir(parents=True, exist_ok=False)
    manifest = output/"credits.json"
    manifest.write_text(json.dumps(records, indent=2)+"\n", encoding="utf-8")
    subprocess.run([sys.executable, str(Path(__file__).with_name("joint_update.py")),
                    "--initial", str(parent), "--credits", str(manifest),
                    "--credits-per-title", str(protocol["credits_per_title"]),
                    "--out", str(output/"trained")], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol","parent","th12","th13","out"):
        parser.add_argument("--"+name, required=True, type=Path)
    args = parser.parse_args()
    combine(args.protocol, args.parent, {12:args.th12,13:args.th13}, args.out)
