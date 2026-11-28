"""Independently reverify a retail credit after identifying redundant split-boundary reads."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adjudicate(folder):
    tools = folder.parent / "tool"
    sys.path.insert(0, str(tools))
    from verify_record import verify

    source = folder / "native-input.csv"
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = list(reader)
    camera_ns = [int(row["camera_ns"]) for row in csv.DictReader((folder / "controls.csv").read_text().splitlines())]
    excluded = []
    for i, row in enumerate(rows):
        tick, slot, clock = (int(row[k]) for k in ("tick", "event_index", "stageclock"))
        if tick == slot and abs(clock-tick) <= 1:
            continue
        if not 0 < i < len(rows)-1:
            raise ValueError("Cannot reconcile an endpoint read")
        before, after = rows[i-1], rows[i+1]
        same_stage = before["stage"] == row["stage"] == after["stage"]
        identical_resource_fields = all(before[k] == row[k] == after[k] for k in
            ("rank", "shot", "continues", "lives", "bombs", "misses", "bombs_used", "retry", "flags"))
        known_before = all(int(before[k]) == tick for k in ("tick", "event_index", "stageclock"))
        known_after = all(int(after[k]) == tick+1 for k in ("tick", "event_index", "stageclock"))
        intact_words = before["word"] == before["record_slot"] and after["word"] == after["record_slot"]
        no_camera_affected = not any(int(row["ns"]) <= ns < int(after["ns"]) for ns in camera_ns)
        if not (slot == clock == tick+1 and same_stage and identical_resource_fields and known_before
                and known_after and intact_words and no_camera_affected):
            raise ValueError(f"Native data cannot resolve row {i} without an assumption")
        excluded.append(i)
    receipt = dict(native_sha256=sha(source), official_rpy_sha256=sha(folder / "official-policy.rpy"),
        rows=len(rows), excluded_redundant_zero_based_rows=excluded,
        boundary_records=[rows[i-1:i+2] for i in excluded])
    with tempfile.TemporaryDirectory(dir=folder.parent) as temp:
        scratch = Path(temp)
        for file in folder.iterdir():
            if file.is_file() and file.name not in ("native-input.csv", "verified.json"):
                (scratch / file.name).symlink_to(file.resolve())
        with (scratch / source.name).open("w", newline="") as handle:
            writer=csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(row for i,row in enumerate(rows) if i not in excluded)
        checked = verify(scratch, (folder / "ui-verified.json").is_file())
        receipt["independent_verification"] = checked
        receipt["source_mask_gate"] = (checked["passed"] and not checked["stale"]
            and all(length > 0 for index, length in checked["partial"]) and checked["terminal_continues"] == 0)
    return receipt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("credit", type=Path)
    parser.add_argument("output", type=Path)
    args=parser.parse_args()
    args.output.write_text(json.dumps(adjudicate(args.credit), indent=2)+"\n")
