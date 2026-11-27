"""Summarize original TH12 worker timings from measured source clocks and SSH byte counts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median


def readings(root):
    rows = []
    for receipt in sorted((root / "credits").glob("*.json")):
        credit = json.loads(receipt.read_bytes())
        raw = root / "raw" / credit["episode"]
        timing = credit.get("engineering_timing", {})
        start_clock = raw / "credit-start-clock.txt"
        first_photo = raw / "evidence/clock.txt"
        launch_to_photo = None
        if start_clock.is_file() and first_photo.is_file():
            ts = first_photo.read_text().splitlines()
            if ts:
                launch_to_photo = float(ts[0].split()[0]) - float(start_clock.read_text())
        reset = (raw / "credit.reset.log").read_text().splitlines()
        rows.append(dict(episode=credit["episode"], seed=credit["seed"], role=credit["mode"],
                         arm=credit["policy"], host=credit["host"], lane=credit["lane"],
                         decisions=credit["actions"], gameplay_seconds=credit["seconds"],
                         captures_after_launch_seconds=launch_to_photo,
                         first_action_after_launch_seconds=timing.get("setup_until_first_action_seconds"),
                         original_reset_entry=reset[0].split()[-1] if reset else None,
                         raw_directory_bytes=timing.get("raw_directory_bytes"), gzip_wire_bytes=timing.get("wire_gzip_bytes"),
                         remote_tar_seconds=timing.get("remote_pack_seconds"),
                         request_to_wire_header_seconds=timing.get("ssh_request_to_header_seconds"),
                         receiving_archive_seconds=timing.get("archive_receive_seconds"),
                         local_unpack_and_verify_seconds=timing.get("local_unpack_verify_npz_seconds")))
    return rows


def summarize(rows, groups):
    by=defaultdict(list)
    for row in rows:
        by[" / ".join(str(row[k]) for k in groups)].append(row)
    fields=("gameplay_seconds","captures_after_launch_seconds","first_action_after_launch_seconds",
            "raw_directory_bytes","gzip_wire_bytes","remote_tar_seconds","request_to_wire_header_seconds",
            "receiving_archive_seconds","local_unpack_and_verify_seconds")
    return {label:dict(credits=len(group),decisions=sum(r["decisions"] for r in group),
                       **{k:dict(median=median([r[k] for r in group if r[k] is not None]),
                                 n=sum(r[k] is not None for r in group))
                          for k in fields if any(r[k] is not None for r in group)}) for label,group in by.items()}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    rows=readings(a.root)
    report=dict(definition="Launch is acquisition of the worker's exclusive credit lock before original menu reset. Color capture uses the monotonic timestamp immediately before scrot, and first action uses the actor's original applied-key clock. The SSH body measurement starts after its declared byte-count header. Remote tar uses private tmpfs. Earlier uninstrumented components have null durations.",
                credits=len(rows),by_host_lane=summarize(rows,("host","lane")),
                by_native_reset_entry=summarize(rows,("host","original_reset_entry")),episodes=rows)
    a.out.write_text(json.dumps(report,indent=2)+"\n")
