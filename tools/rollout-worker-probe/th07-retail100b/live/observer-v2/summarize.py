"""Report resource use and independent acceptance for all four registered attempts."""
import hashlib
import json
from pathlib import Path
import sys


def measure(study):
    plan=json.loads((study/"protocol.json").read_bytes())
    completed=json.loads((study/"engineering-results.json").read_bytes())
    rows=[]
    for spec in plan["lanes"]:
        seed=spec["seed"]
        f=study/f"{seed}-{spec['arm']}"
        receipt=next(r for r in completed if r["seed"]==seed)
        row=dict(seed=seed,role="engineering",arm=spec["arm"],physical_cpu_ids=spec["physical_cpu_ids"],
                 hardware="Momentum",accepted=receipt["qualified"])
        if (f/"verified.json").exists():
            q=json.loads((f/"verified.json").read_bytes())
            assert q["passed"]==row["accepted"] and q["sha256"]==spec["sha256"]
            assert (f/"build.sha256").read_text().split()[0]==plan["game_exe_sha256"]
            row.update(stage1_ticks=q["stage1_native_ticks"],stage2=q["stage2"],actions=q["actions"],
                complete=q["complete_windows"],exact=q["strict_exact"],fraction=q["exact_fraction"],stale=q["stale"],
                partial=q["partial"],omitted=q["skipped_decisions"],outcome=q["outcome"],continues=q["terminal_continues"],
                terminal_stock=q["terminal_stock"],bombs=q["actual_bombs"],score=q["score_proof"],
                p999_upper_read_ms=q["observer_p999_upper_ms"],max_read_ms=q["max_synchronous_observer_ms"],
                max_frozen_ms=q["max_freeze_ms"],polls=int(q["read_distribution"]["polls"]),
                rejected_polls=int(q["read_distribution"]["rejected"]),
                archival_records=q["archive"]["records"],duplicate_rows=q["archive"]["stable_duplicate_rows"],
                physical_ticks_per_gameplay_wall_sec=q["stage1_native_ticks_per_wall_second"],
                journal_bytes=(f/"events.bin").stat().st_size,
                receipt_sha256=hashlib.sha256((f/"verified.json").read_bytes()).hexdigest())
        else:
            row["error"]=receipt.get("error","No verified original receipt")
        if (f/"cgroup.jsonl").exists():
            lines=(f/"cgroup.jsonl").read_text().splitlines()
            a,b=(json.loads(lines[i]) for i in (0,-1))
            stats=[dict(line.split() for line in x["cpu"].splitlines()) for x in (a,b)]
            cpu_usec=int(stats[1]["usage_usec"])-int(stats[0]["usage_usec"])
            row["game_container_mean_cpu_cores"]=cpu_usec*1000/(b["ns"]-a["ns"])
            row["cgroup_throttled_periods"]=int(stats[1]["nr_throttled"])-int(stats[0]["nr_throttled"])
        rows.append(row)
    accepted=sum(r["accepted"] for r in rows)
    return dict(protocol_sha256=hashlib.sha256((study/"protocol.json").read_bytes()).hexdigest(),
                scope="Acquisition engineering only. Fixed parent, four natural fresh starts, no model evaluation.",
                attempts=len(rows),accepted=accepted,acceptance_rate=accepted/len(rows),
                registered_engineering_gate_passed=accepted>=plan["quality"]["minimum_valid_credits"],rows=rows)


if __name__=="__main__":
    root=Path(sys.argv[1])
    (root/"summary.json").write_text(json.dumps(measure(root),indent=2)+"\n")
