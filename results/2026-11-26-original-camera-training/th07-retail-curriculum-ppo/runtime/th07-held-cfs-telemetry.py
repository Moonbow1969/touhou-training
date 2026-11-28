"""Bounded read-only cgroup timing witness for the active held practice containers."""

import argparse
import json
from pathlib import Path
import subprocess
import time


def study_groups():
    names = subprocess.check_output([
        "docker", "ps", "--filter", "name=moonbow-th07-practice-", "--format", "{{.Names}}"
    ], text=True).splitlines()
    if not names:
        return {}
    result = subprocess.run(["docker", "inspect", *names], text=True, capture_output=True)
    rows = {}
    if not result.stdout.strip():
        return rows
    for obj in json.loads(result.stdout):
        source = next((m["Source"] for m in obj["Mounts"] if m["Destination"] == "/capture"), "")
        if "/th07-practice-actor-held/raw/" not in source:
            continue
        pid = obj["State"]["Pid"]
        try:
            relative = Path(f"/proc/{pid}/cgroup").read_text().strip().split(":", 2)[2]
            directory = Path("/sys/fs/cgroup") / relative.lstrip("/")
            quota = (directory / "cpu.max").read_text().strip()
        except FileNotFoundError:
            continue
        rows[Path(source).name] = (directory, quota, obj["HostConfig"]["CpusetCpus"])
    return rows


def sample(groups):
    instant = time.monotonic_ns()
    rows = {}
    for name, (directory, quota, cpus) in groups.items():
        try:
            counters = (directory / "cpu.stat").read_text().splitlines()
        except FileNotFoundError:
            continue
        stats = {key: int(value) for key, value in (line.split() for line in counters)}
        rows[name] = dict(cpu_max=quota, cpus=cpus, stat=stats)
    return dict(monotonic_ns=instant, groups=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", required=True, type=float)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    deadline = time.monotonic() + min(args.seconds, 900.)
    next_map = 0.
    groups = {}
    with args.out.open("x", buffering=65536) as log:
        while time.monotonic() < deadline:
            start = time.monotonic()
            if start >= next_map:
                groups = study_groups()
                next_map = start + 1.5
            print(json.dumps(sample(groups), separators=(",", ":")), file=log)
            time.sleep(max(0., .1 - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
