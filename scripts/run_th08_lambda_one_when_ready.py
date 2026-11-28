"""Supervise the approved one-iteration TH08 learner without a live chat session."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def read_optional(path):
    try:
        return json.loads(path.read_bytes())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "repository", "parent"):
        parser.add_argument("--"+name, required=True, type=Path)
    args = parser.parse_args()
    intake = args.study / "first-eight-intake.json"
    manifest = args.study / "qualified-train.json"
    deadline = time.monotonic() + 70*60
    print("Waiting for the verified 8-credit intake and 32 bound normal originals",flush=True)
    while time.monotonic() < deadline:
        guard = read_optional(intake)
        if guard is not None and guard.get("passed") is False:
            raise RuntimeError("The frozen first-eight native intake gate failed; no optimizer ran")
        batch = read_optional(manifest)
        if guard and guard.get("passed") is True and batch and len(batch["rows"]) == 32:
            break
        time.sleep(10)
    else:
        raise RuntimeError("The approved collection did not complete within the supervised window")
    subprocess.run([sys.executable,"-u","-m","scripts.train_th08_progress_lambda_one",
                    "--manifest",str(manifest),"--intake",str(intake),"--behavior",str(args.parent/"policy.ts"),
                    "--state",str(args.parent/"policy.pt"),"--out",str(args.study/"policy")],cwd=args.repository,check=True)
    print("Exactly one approved continuation update completed; awaiting external post-update review",flush=True)


if __name__ == "__main__":
    main()
