"""Verify newly sampled merged-SFT trajectories against the earlier G1 controller-seed ledger."""
import argparse
import json
from pathlib import Path

import numpy as np
from scripts.collect_th10_full_action import save
from scripts.run_th10_camera_cohort import digest


def check(root):
    work=root/"four-game-merged-online";old=root/"four-game-online"
    current_plan=json.loads((work/"camera-protocol.json").read_bytes());old_plan=json.loads((old/"camera-protocol.json").read_bytes())
    rows=[]
    for title in (10,13):
        reg=json.loads((work/f"th{title}-held-host-allocation.json").read_bytes())
        for seed in current_plan["train_seeds"][str(title)]:
            assert seed in old_plan["train_seeds"][str(title)]
            rank=current_plan["rank_by_seed"][str(title)][str(seed)];label="hard" if rank==2 else "lunatic"
            if title==10:
                source=old/"native10"/label/f"momentum-{seed}";target=work/"native10"/label/f"momentum-{seed}"
                file="report.json";data="behavior.npz";adc="behavior_sha256"
            else:
                source=old/"native13"/f"train-four-sft-{seed}";target=work/"native13"/f"train-merged-sft-{seed}"
                file="receipt.json";data="rollout.npz";adc="sha256"
            known=json.loads((source/file).read_bytes());sample=json.loads((target/file).read_bytes())
            assert sample["seed"]==known["seed"]==seed and sample["host"]==known["host"]==reg["train_seed_host"][str(seed)]
            assert known["policy_sha256"]==old_plan["behavior_sha256"] and sample["policy_sha256"]==current_plan["behavior_sha256"]!=known["policy_sha256"]
            assert digest(source/data)==known[adc] and digest(target/data)==sample[adc]!=known[adc]
            with np.load(target/data) as z,np.load(source/data) as b:
                assert z["terminated"].sum()==1 and z["terminated"][-1] and z["action"].shape==z["logp"].shape==(sample["steps"],4)
                assert len(z["value"])==sample["steps"] and not z["value"].any() and np.isfinite(z["logp"]).all()
                assert not np.array_equal(z["action"],b["action"])
                clocks=z["seconds"] if "seconds" in z else (z["clock_ns"]-z["clock_ns"][0])/1e9
                before=b["seconds"] if "seconds" in b else (b["clock_ns"]-b["clock_ns"][0])/1e9
                cad=[float(np.median(np.diff(v))*1000) for v in (before,clocks)]
            rows.append(dict(title=title,seed=seed,rank=rank,same_host=sample["host"],old_joint_sft_policy_sha256=known["policy_sha256"],merged_sft_policy_sha256=sample["policy_sha256"],
                old_original_behavior_sha256=known[adc],fresh_original_behavior_sha256=sample[adc],old_and_new_decisions=[known["steps"],sample["steps"]],
                old_and_new_median_control_interval_ms=cad,all_new_critic_predictions_zero=True,action_sequences_differ=True,
                fresh_native_trace_sha256=sample.get("native_difficulty_trace_sha256") or digest(target/"native.csv.gz"),
                fresh_held_out_of_batch=True))
    save(work/"th10-th13-training-provenance.json",dict(protocol_sha256=digest(work/"camera-protocol.json"),
        note="Matched controller seeds and physical hosts, with newly sampled original games. Original game RNG is independently initialized by each launch.",
        registered_source=16,credits=rows));print(len(rows),sum(r["old_and_new_decisions"][1] for r in rows))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);check(p.parse_args().root)
