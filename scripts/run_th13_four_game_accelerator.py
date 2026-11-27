"""Run the preassigned Accelerator pairs in the four-game original TH13 evaluation."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import numpy as np

from scripts.collect_th10_full_action import save
from scripts.read_th13_trial_ranking import scan
from scripts.run_th10_camera_cohort import digest
from scripts.run_th13_accelerator_camera import run as collect


def run(root,workers):
    root=root.resolve();study=root/"four-game-online";native=study/"native13"
    allocation=json.loads((study/"th13-held-host-allocation.json").read_bytes())
    plan=json.loads((study/"camera-protocol.json").read_bytes())
    assert allocation["study_protocol_sha256"]==digest(study/"camera-protocol.json")
    seeds=[s for s in plan["eval_seeds"]["13"] if allocation["seed_host"][str(s)]=="accelerator"]
    models={"four-sft":root/"ranked-multigame-sft/four-title/policy.ts","four-ppo":study/"policy/policy.ts"}
    hashes={a:digest(p) for a,p in models.items()}
    assert hashes["four-sft"]=="510132b2cd0cd04b728d066fe73e8703440c7e3cce1dcd165393da691ce76220"
    assert hashes["four-ppo"]==json.loads((study/"policy/report.json").read_bytes())["policy_sha256"]
    assert seeds==list(range(1211363,1211375))
    for order_index in (0,1):
        for group in (seeds[:6],seeds[6:]):
            arm=allocation["accelerator_order"][str(group[0])][order_index]
            for rank in (2,3):
                assigned=[s for s in group if plan["rank_by_seed"]["13"][str(s)]==rank]
                assert len(assigned)==3 and all(allocation["accelerator_order"][str(s)][order_index]==arm for s in assigned)
                missing=[s for s in assigned if not (native/f"held-{arm}-{s}/receipt.json").is_file()]
                if missing:
                    collect(SimpleNamespace(qualification=root/"engineering/th13-accelerator",connection_config=workers.resolve(),
                        checkpoint=models[arm],output=native,role="held",arm=arm,seeds=missing,workers=3,difficulty=rank))
    readings={Path(row["path"]).parent.name:row for row in scan(native)["rows"]}
    rows=[];archive=study/"th13-accelerator-held-native-logs.zip";tmp=archive.with_suffix(".inprogress")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for seed in seeds:
            for arm in models:
                folder=native/f"held-{arm}-{seed}";f=folder/"receipt.json";r=json.loads(f.read_bytes())
                steps=np.fromfile(folder/"steps.bin","<f4").reshape(-1,143)
                rank=plan["rank_by_seed"]["13"][str(seed)];reading=readings[folder.name]
                assert r["host"]=="accelerator" and r["policy_sha256"]==hashes[arm] and r["difficulty"]==rank
                assert r["terminal"] and r["continues"]==0 and r["seed"]==seed and r["steps"]==len(steps)
                assert (steps[:,141]==rank).all() and list(steps[0,[125,127,129]])==[1,2,2]
                assert r["native_difficulty_trace_sha256"]==digest(folder/"steps.bin")
                rows.append(dict(seed=seed,arm=arm,difficulty=rank,host=r["host"],stage2=r["stage2_reached"],
                        last_observed_stage1_tick=int(steps[steps[:,125]==1,135].max()),
                        terminal_spares=max(0,int(r["remaining_lives"])),terminal_bombs=r["remaining_bombs"],
                        native_rank_rows=len(steps),native_key_matches=r["native_key_match"],casts=r["effective_bombs"],
                        ranking_readout=reading["readout"],ranking_agrees=reading["matches_native"],
                        policy_sha256=r["policy_sha256"],native_trace_sha256=r["native_difficulty_trace_sha256"]))
                for name in ("receipt.json","steps.bin","ranking.png","selected-difficulty.png","events.tsv","final-stock.bin"):
                    z.write(folder/name,f"{folder.name}/{name}")
        save(study/"th13-accelerator-held.json",rows)
        for p in (study/"th13-held-host-allocation.json",study/"th13-accelerator-held.json"):z.write(p,p.name)
    with zipfile.ZipFile(tmp) as z:assert z.testzip() is None
    tmp.replace(archive)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--workers",type=Path,required=True)
    args=parser.parse_args();run(args.root,args.workers)
