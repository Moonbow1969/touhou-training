"""Publish native post-handoff and opening-control eligibility and event coverage."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import zipfile

import numpy as np

from scripts.collect_th10_full_action import save
from scripts.run_th10_camera_cohort import digest


def read(path):
    return json.loads(path.read_bytes())


def summarize(study):
    protocol=read(study/"protocol.json")
    rows=[]
    for group, dirname, arm, seeds in (("prefix","native13","humanprefix-visual",protocol["attempt_seeds"]),
                                       ("opening","opening-control","four-ppo",protocol["opening_control_seeds"])):
        for seed in seeds:
            folder=study/dirname/f"train-{arm}-{seed}"
            r=dict(group=group,seed=seed,relative_source=folder.relative_to(study).as_posix())
            gate=folder/"qualification.json"
            if (folder/"eligible.json").is_file():
                el=read(folder/"eligible.json"); receipt=read(folder/"receipt.json");p=folder/"rollout.npz"
                assert digest(p)==receipt["sha256"]==el["rollout_sha256"] and receipt["policy_sha256"]==protocol["source_policy_sha256"]
                raw=np.fromfile(folder/"steps.bin","<f4").reshape(-1,143)
                with np.load(p) as z:
                    assert len(z["frames"])==receipt["steps"] and not z["terminated"][:-1].any() and z["terminated"][-1]
                r.update(eligible=True,decisions=receipt["steps"],stage2=receipt["stage2"],terminal_spares=max(0,receipt["remaining_lives"]),
                         terminal_bombs=receipt["remaining_bombs"],effective_casts=receipt["effective_bombs"],native_actions_matched=receipt["native_key_match"],
                         start_tick=int(raw[0,134]),last_stage1_tick=int(raw[raw[:,126]==1,135].max()),initial_spares=int(raw[0,127]),
                         initial_bombs=int(raw[0,129]),life_losses=receipt["confirmed_losses"],extends=receipt["life_earned"],
                         decisions_at_or_after_tick3000=int(((raw[:,134]>=3000)&(raw[:,125]==1)).sum()),
                         distinct_non_survival_event_decisions=int((raw[:,136:140].sum(1)>0).sum()),rollout_sha256=el["rollout_sha256"],
                         max_logp_error=el["max_logp_abs_error"],max_value_error=el["max_value_abs_error"])
                if group=="prefix":
                    q=read(gate);r.update(prefix_ticks=q["distinct_observed_prefix_ticks"],human_keys_matched=el["prefix_raw_native_input_matches"],
                        prefix_real_seconds=(int(q["handoff"]["monotonic_ns"])-int(q["prefix_opening"]["monotonic_ns"]))/1e9,
                        first_policy_observation_tick=el["first_native_before_tick"])
            elif (folder/"collection-error.json").is_file() or (gate.is_file() and read(gate).get("validated_handoff")==False):
                r.update(eligible=False,decisions=0,eligibility_failure=read(gate) if gate.exists() else read(folder/"collection-error.json"))
            else:return False
            rows.append(r)
    archives=[]
    # Each six-attempt group retains its exact native frames, limiting individual assets to well below GitHub's object cap.
    for group in ("prefix","opening"):
        selected=[row for row in rows if row["group"]==group]
        for ordinal,start in enumerate((0,6),1):
            dest=study/f"native-train-{group}-{ordinal}.zip";tmp=dest.with_suffix(".inprogress")
            keep=("protocol.json","qualification.json","eligible.json","receipt.json","rollout.npz","steps.bin","final-stock.bin","ranking.png","events.tsv",
                  "prefix.csv","handoff-original.png","prefix-opening.png","collection-error.json")
            with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED,compresslevel=3) as z:
                for row in selected[start:start+6]:
                    f=study/row["relative_source"]
                    for name in keep:
                        if (f/name).is_file():z.write(f/name,row["relative_source"]+"/"+name)
            with zipfile.ZipFile(tmp) as z:assert z.testzip() is None
            assert tmp.stat().st_size<99_000_000;tmp.replace(dest)
            archives.append(dict(file=dest.name,bytes=dest.stat().st_size,sha256=digest(dest)))
    save(study/"train-native-acquisition.json",dict(protocol_sha256=digest(study/"protocol.json"),attempts=rows,archives=archives))
    return True


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("study",type=Path);parser.add_argument("--wait",action="store_true")
    a=parser.parse_args()
    while not summarize(a.study):
        if not a.wait:raise SystemExit("Waiting for registered outcomes")
        time.sleep(12)
