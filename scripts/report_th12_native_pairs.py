"""Report matched unassisted TH12 original credits with signed native stage and stock."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_credit(root, receipt, bosses):
    raw = root / "raw" / receipt["episode"]
    telemetry = receipt["stage2_review"]["native_telemetry"]
    stage_path = raw / "stage-native.tsv"
    verified = telemetry.get("qualified") and sha(stage_path) == telemetry["trace_sha256"]
    stage = receipt["stage2_reached"]
    with stage_path.open(newline="") as source:
        original = list(csv.DictReader(source, delimiter="\t"))
    stage1 = [int(r["tick"]) for r in original if r["mode"]=="0" and
              r["stage"]==r["manager_stage"]=="1" and int(r["tick"])>=0]
    if verified and stage is not (telemetry["max_stage"] >= 2):
        raise ValueError(f"Conflicting signed original stage for {receipt['episode']}")
    trace = root / "credits" / receipt["trajectory"]
    if sha(trace) != receipt["trajectory_sha256"]:
        raise ValueError(f"Modified trajectory for {receipt['episode']}")
    with np.load(trace,allow_pickle=False) as z:
        if not (np.array_equal(z["keys"],z["observed_keys"])
                and z["terminated"].sum()==1 and z["terminated"][-1]):
            raise ValueError(f"Invalid behavior or terminal in {receipt['episode']}")
        loss = np.flatnonzero(z["reward_parts"][:,1]<0)
        first = float((z["observed_ns"][loss[0]]-z["onset_ns"][0])/1e9) if len(loss) else None
    return dict(episode=receipt["episode"],seed=receipt["seed"],host=receipt["host"],lane=receipt["lane"],
                arm=receipt["policy"],rank=receipt.get("difficulty",1),
                confirmed_stage2=stage,sensor_verified=bool(verified),stage1_last_native_tick=max(stage1) if verified else None,
                seconds=receipt["seconds"],first_spare_loss_seconds=first,
                final_spares=receipt["remaining_spare_lives"],final_bombs=receipt["remaining_bombs"],
                entry_spares=receipt["stage2_remaining_spare_lives"],entry_bombs=receipt["stage2_remaining_bombs"],
                effective_bombs=sum(e["prior"]-e["current"] for e in receipt["event_history"]
                                    if e["event"]=="bombs" and e.get("cost",0)<0),
                timed_boss_visible=bosses.get(receipt["episode"]),actions=receipt["actions"],
                nonneutral=receipt["actions"]-receipt["move_counts"][4],
                vertical=sum(receipt["move_counts"][:3])+sum(receipt["move_counts"][6:]),
                focus=receipt["key_positive_counts"][0],shoot=receipt["key_positive_counts"][1],
                bomb_requests=receipt["key_positive_counts"][2],trajectory_sha256=receipt["trajectory_sha256"],
                stage_source_sha256=telemetry.get("trace_sha256"))


def stats(rows):
    counts = len(rows)
    totals = sum(r["actions"] for r in rows)
    out = dict(credits=counts,stage2_reaches=sum(r["confirmed_stage2"] is True for r in rows),
               stage2_unknown=sum(r["confirmed_stage2"] is None for r in rows),
               boss_visible=sum(r["timed_boss_visible"] is True for r in rows),
               boss_readable=sum(r["timed_boss_visible"] is not None for r in rows),actions=totals)
    known = counts - out["stage2_unknown"]
    if known:
        z = 1.959963984540054
        rate = out["stage2_reaches"] / known
        center = rate + z*z/(2*known)
        radius = z*math.sqrt(rate*(1-rate)/known + z*z/(4*known*known))
        scale = 1 + z*z/known
        out["stage2_rate_known_wilson95"] = [max(0.0,(center-radius)/scale), min(1.0,(center+radius)/scale)]
    for k in ("seconds","stage1_last_native_tick","first_spare_loss_seconds","final_spares","final_bombs","effective_bombs"):
        observed=[r[k] for r in rows if r[k] is not None]
        out[k]=dict(mean=float(np.mean(observed)),median=float(np.median(observed)),count=len(observed)) if observed else None
    for k in ("nonneutral","vertical","focus","shoot","bomb_requests"):
        out[k+"_fraction"]=sum(r[k] for r in rows)/totals
    out["stage2_entry_stocks"]=[dict(seed=r["seed"],spares=r["entry_spares"],bombs=r["entry_bombs"])
                                for r in rows if r["confirmed_stage2"] is True]
    return out


def compare(rows,reference,candidate):
    pairs=defaultdict(dict)
    for r in rows:
        key=r["seed"]
        if r["arm"] in pairs[key]:raise ValueError(f"Duplicate native behavior {key}/{r['arm']}")
        pairs[key][r["arm"]]=r
    matched=[(d[reference],d[candidate]) for _,d in sorted(pairs.items())]
    if any((a["host"],a["lane"],a["rank"])!=(b["host"],b["lane"],b["rank"]) for a,b in matched):
        raise ValueError("Physical worker or native rank differs within one seed")
    out={}
    for k in ("confirmed_stage2","stage1_last_native_tick","seconds","first_spare_loss_seconds","final_bombs"):
        by_host=defaultdict(list)
        for a,b in matched:
            if a[k] is not None and b[k] is not None:by_host[a["host"]].append(float(b[k])-float(a[k]))
        if not by_host:continue
        rng=np.random.default_rng(261126)
        draws=[]
        for values in by_host.values():
            x=np.array(values)
            draws.append(x[rng.integers(len(x),size=(20000,len(x)))])
        v=np.array([x for xs in by_host.values() for x in xs])
        out[k]=dict(pairs=len(v),mean=float(v.mean()),wins=int((v>0).sum()),ties=int((v==0).sum()),
                    losses=int((v<0).sum()),percentile95=np.quantile(np.concatenate(draws,axis=1).mean(1),[.025,.975]).tolist())
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root",type=Path)
    ap.add_argument("mode",choices=("train","held"))
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    r=json.loads((a.root/f"{a.mode}-receipt.json").read_bytes())
    if r["status"]!="complete":raise ValueError("Registered collection is incomplete")
    rank_plan=json.loads((a.root/"protocol.json").read_bytes())["plan"]
    expected={(p["host"],p["lane"],p["seed"],p["policy"]) for p in rank_plan if p["mode"]==a.mode}
    actual={(p["host"],p["lane"],p["seed"],p["policy"]) for p in r["credits"]}
    if len(r["credits"])!=len(expected) or expected!=actual:raise ValueError("Credits differ from the registered schedule")
    boss_file=a.root/f"{a.mode}-boss-hud.json"
    bosses=({q["episode"]:q["timed_boss_bar_seen"] for q in json.loads(boss_file.read_bytes())["episodes"]}
            if boss_file.is_file() else {})
    rows=[read_credit(a.root,row,bosses) for row in r["credits"]]
    protocol=json.loads((a.root.parent/"camera-protocol.json").read_bytes())
    cells={}
    for rank in sorted({r["rank"] for r in rows}):
        records=[r for r in rows if r["rank"]==rank]
        cells[rank]={"arms":{arm:stats([r for r in records if r["arm"]==arm]) for arm in sorted({r["arm"] for r in records})}}
        if a.mode=="held":cells[rank]["paired_candidate_minus_reference"]=compare(records,protocol["reference_arm"],protocol["candidate"])
    report=dict(scope="TH12 official trial, true original Reimu A story whole credits. Controller seeds are fixed. Original game RNG is uncontrolled. Stage 2 entry reserves are conditional on reaching it. Secondary native ticks and wall duration are distinct measures. Percentile intervals resample matched seeds within each physical host. Degenerate bootstrap bounds for zero successes do not bound population probability.",
                mode=a.mode,cells=cells,episodes=rows)
    a.out.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
