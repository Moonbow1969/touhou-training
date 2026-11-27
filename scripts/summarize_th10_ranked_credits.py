"""Compare whole-credit TH10 outcomes separately for each verified native difficulty."""
import argparse
import json
from pathlib import Path
import numpy as np


def run(study):
    analyses={}
    for rank,label in [(2,"hard"),(3,"lunatic")]:
        native=study/"native10"/label
        data=json.loads((native/"paired.json").read_bytes())
        ledger=native/"qualified-stage-receipts.json"
        if not ledger.exists():ledger=native/"held-qualified-native-ranks.json"
        receipts=json.loads(ledger.read_bytes())
        arms=list(data["group"])
        groups={name:{r["seed"]:r for r in data["credits"][name]} for name in arms}
        seeds=sorted(groups[arms[0]])
        assert len(seeds)>=2 and seeds==sorted(groups[arms[1]])
        assert len(receipts)==len(arms)*len(seeds) and all(row["difficulty"]==rank for row in receipts)
        measures={};rng=np.random.default_rng(20261126+rank)
        for metric, extract in (
                ("native_score",lambda row:row["raw_score"]),("settled_power",lambda row:row["final_power"]),
                ("confirmed_casts",lambda row:len(row["native_bomb_spends"])),("wall_s",lambda row:row["duration_s"]),
                ("first_life_loss_wall_s",lambda row:row["loss_events"][0]["seconds"])):
            arr=np.array([[extract(groups[a][seed]) for a in arms] for seed in seeds])
            delta=arr[:,1]-arr[:,0]
            means=rng.choice(delta,(20000,len(delta))).mean(1)
            measures[metric]=dict(mean=arr.mean(0).tolist(),paired_mean_change=float(delta.mean()),
                                  paired_bootstrap95=np.quantile(means,[.025,.975]).tolist())
        if all("stage_clock_sha256" in r for r in receipts):
            from scripts.th10_native_stage_clock import verify
            times={}
            for row in receipts:
                d=native/row["arm"]/f"{row['host']}-{row['seed']}"
                record=json.loads((d/"report.json").read_bytes());log=record["native_stage_clock"]
                finish=verify(d,record)
                assert finish==row["last_observed_stage1_tick"] and log["data_sha256"]==row["stage_clock_sha256"]
                times[row["arm"],row["seed"]]=(finish,log["first_life_loss_tick"] if log["first_life_loss_stage"]==1 else None)
            for i,label_ in enumerate(("final_stage1_native_tick","first_loss_stage1_native_tick")):
                arr=np.array([[times[a,s][i] for a in arms] for s in seeds if all(times[a,s][i] is not None for a in arms)])
                if len(arr):
                    delta=arr[:,1]-arr[:,0];means=rng.choice(delta,(20000,len(delta))).mean(1)
                    measures[label_]=dict(n=len(delta),mean=arr.mean(0).tolist(),paired_mean_change=float(delta.mean()),
                                          paired_bootstrap95=np.quantile(means,[.025,.975]).tolist())
        analyses[label]=dict(arms=arms,matched_seeds=seeds,difficulty=rank,
                native_rank_readings=sum(r["rank_samples"] for r in receipts),
                verified_key_counts=[[data["group"][a]["decisions_matching_native_input"],data["group"][a]["decisions"]] for a in arms],
                whole_credit_game_overs=[data["group"][a]["native_game_overs"] for a in arms],
                stage2=[data["group"][a]["stage2_reached"] for a in arms],
                stage2_entry_resources=[data["group"][a]["stage2_resources"] for a in arms],
                terminal_lives=[data["group"][a]["final_spare_life_counts"] for a in arms],
                movement_switch_rate=[data["movement_switches"][a]["pooled_rate"] for a in arms],
                native_first_life_both_axes=[data["group"][a]["native_first_life_movement"]["credits_moving_on_both_axes"] for a in arms],
                total_casts=[data["group"][a]["confirmed_bomb_spends"] for a in arms],secondary=measures)
    out=study/"native10/ranked-native-comparison.json";out.write_text(json.dumps(analyses,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("study",type=Path)
    run(p.parse_args().study)
