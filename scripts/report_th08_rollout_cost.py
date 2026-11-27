"""Measure original-game time, post-credit transfer and evidence bytes on one TH08 host."""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from statistics import median


def load(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def stamp(path):
    return datetime.fromisoformat(path.read_text().strip().replace(",",".")).timestamp()


def measure(home,role):
    rows=[]
    for source in sorted((home/("training-receipts" if role=="train" else "receipts")).glob("*.json")):
        rec=load(source)
        if not rec["physical_valid"]:
            raise ValueError(f"Unvalidated original credit requires separate accounting: {source.name}")
        raw=home/rec["raw"];qualified=home/rec["recovered_source"];rank=rec["rank"]
        original=load(raw/"receipt.json")
        assert original["seed"]==rec["seed"] and original["policy_sha256"]==rec["policy_sha256"]
        packed=home/rec["record"]["file"];transport=load(raw/"transport-audit.json")
        assert digest(packed)==rec["record"]["sha256"]
        assert digest(raw/"original-observations.mp4")==rec["evidence"]["original-observations.mp4"]
        wall0,wall1=[stamp(raw/name) for name in ("setup-start.txt","setup-end.txt")]
        timing=[json.loads(line) for line in (qualified/"timing.jsonl").read_bytes().splitlines()]
        onset=next(t for t in timing if t["phase"]=="policy")
        end=timing[-1]
        if end["phase"]!="terminal" or sum(t["phase"]=="policy" for t in timing)!=rec["audit"]["decisions"]:
            raise ValueError(f"Missing qualified terminal or action timestamps: {source.name}")
        played=(end["monotonic_ns"]-onset["monotonic_ns"])/1e9
        assert 0<played<=wall1-wall0 and rec["audit"]["decisions"]>0
        rows.append(dict(seed=rec["seed"],arm=rec["arm"],rank=rank,lane=rec["lane"],host=rec["physical_host"],
                         receipt_sha256=digest(source),decisions=rec["audit"]["decisions"],start_utc_unix=wall0,end_utc_unix=wall1,
                         original_job_seconds=wall1-wall0,first_policy_until_end_observation_seconds=played,
                         nonpolicy_seconds=wall1-wall0-played,applied_action_latency_median_ms=rec["audit"]["applied_ms_median"],
                         credited_native_duration_seconds=rec["audit"]["observed_duration_s"],
                         source_ended_as=end["phase"],ssh_pack_receive_seconds=transport["remote_pack_ssh_and_receive_seconds"],
                         ssh_archive_bytes=transport["compressed_bytes"],optimizer_bytes=packed.stat().st_size,
                         movie_bytes=(raw/"original-observations.mp4").stat().st_size))
    if not rows or len({r["host"] for r in rows})!=1:raise ValueError("Requires a nonempty single-host family")
    boundaries=sorted({t for r in rows for t in [r["start_utc_unix"],r["end_utc_unix"]]})
    concurrency=defaultdict(float)
    for a,b in zip(boundaries,boundaries[1:]):
        active=[r["lane"] for r in rows if r["start_utc_unix"]<=a and b<=r["end_utc_unix"]]
        if len(active)!=len(set(active)):raise ValueError("Two games assigned to one physical lane at once")
        concurrency[len(active)]+=b-a
    wall=boundaries[-1]-boundaries[0]
    sums={k:sum(r[k] for r in rows) for k in ("decisions","ssh_archive_bytes","optimizer_bytes","movie_bytes","ssh_pack_receive_seconds")}
    medians={k:median(r[k] for r in rows) for k in ("original_job_seconds","first_policy_until_end_observation_seconds","nonpolicy_seconds","ssh_pack_receive_seconds","applied_action_latency_median_ms")}
    return dict(credits=len(rows),host=rows[0]["host"],physical_start_to_last_end_seconds=wall,
                within_cohort_aggregate_decisions_per_second=sums["decisions"]/wall,whole_games_per_hour=3600*len(rows)/wall,
                concurrent_native_jobs_seconds=dict(sorted(concurrency.items())),totals=sums,per_credit_medians=medians,
                scope="The native job includes the cold container and menu round trip. The clock measures observations at input cadence, excluding menus from the first-policy-to-last-observation interval. Native observer calls, policy inference and key submission are local to this host. The SSH archive includes color movies and verification images after gameplay. The aggregate denominator includes gaps between assigned credits, including waiting on checkpoints or other studies. Pre-input technical retries are outside the measured signed job intervals but their elapsed time within the overall span is included.",
                rows=rows)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native",type=Path,required=True);p.add_argument("--role",choices=("train","held"),required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();report=measure(args.native,args.role)
    args.out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
