"""Use one signed credit to check that redundant-read removal cannot replace missing action proof."""
import argparse
import csv
import json
from pathlib import Path
import tempfile
from reconcile_duplicate import adjudicate


def run(folder):
    original = adjudicate(folder)
    assert original["source_mask_gate"] and len(original["excluded_redundant_zero_based_rows"]) == 1
    with tempfile.TemporaryDirectory(dir=folder.parent) as temp:
        case = Path(temp)
        for p in folder.iterdir():
            if p.is_file() and p.name not in ("native-input.csv", "verified.json"):
                (case / p.name).symlink_to(p.resolve())
        with (folder / "native-input.csv").open(newline="") as handle:
            reader = csv.DictReader(handle)
            cols = reader.fieldnames
            native = list(reader)
        with (folder / "controls.csv").open() as f:
            decision = next(r for r in csv.DictReader(f) if int(r["tick"]) == 2080)
        control = int(decision["requested"])
        changed = 0
        for row in native:
            if 2081 <= int(row["tick"]) <= 2084:
                assert int(row["word"]) & 247 == control
                row["word"] = row["record_slot"] = str(control ^ 1)
                changed += 1
        assert changed == 4
        with (case / "native-input.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(native)
        broken = adjudicate(case)
        assert not broken["source_mask_gate"]
        assert int(decision["seq"]) in broken["independent_verification"]["stale"]
        assert original["official_rpy_sha256"] == broken["official_rpy_sha256"]
    return dict(genuine_duplicate_verified=True, missing_native_action_witness_rejected=True,
                synthetic_missing_native_ticks=[2081,2082,2083,2084], official_rpy_sha256=original["official_rpy_sha256"],
                native_source_sha256=original["native_sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("credit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.credit), indent=2)+"\n")
