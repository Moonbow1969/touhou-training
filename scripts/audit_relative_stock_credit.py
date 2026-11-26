"""Audit one finished native student or teacher credit on its actual frozen state sequence."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.checkpoint_storage import digest
from scripts.distill_state_route import C,ROOT,SCALE,Teacher,labels
from scripts.relative_phase_student import RelativeActor
from scripts.train_relative_phase_student import measure,observed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    path=args.report.resolve()
    source=json.loads(path.read_bytes())
    if source["status"]!="complete" or len(source["completed"])!=1:raise ValueError("Expected one completed credit.")
    record=source["completed"][0];result=record["result"];trial=record["trial"];seed=trial["policy_seed"]
    if seed<2402:raise ValueError("This audit is for newly reserved or student-native credits.")
    base=ROOT/"outputs/th14-relative-phase-student-261125"
    fingerprint=json.loads((base/"training.json").read_bytes())["hash"]
    if digest(base/"student.pt")!=fingerprint:raise ValueError("Student weights changed.")
    shard=path.parent/result["transition_shard"]["path"]
    if digest(shard)!=result["transition_shard"]["sha256"]:raise ValueError("Native transition archive changed.")
    student=source["specification"]["policies"][trial["arm"]].get("sha256")==fingerprint
    torch.set_num_threads(2)
    model=RelativeActor().eval();model.load_state_dict(torch.load(base/"student.pt",weights_only=True)["state_dict"])
    teacher=Teacher();d=observed(shard,teacher)
    reference=labels(teacher,d,d["x"][:,:2]*SCALE)
    actual=d["actual"]
    stage_metrics={}
    for stage in np.unique(d["stage"]):
        mask=d["stage"]==stage;sub={k:v[mask] for k,v in d.items()}
        behavioral=measure(model,sub,teacher,"cpu",against_actual=True)
        sub["actual"]=reference[mask]
        stage_metrics[str(stage)]=dict(behavior=behavioral,against_privileged_relabel=measure(model,sub,teacher,"cpu",against_actual=True),
            privileged_teacher_vs_behavior=(reference[mask]==actual[mask]).mean(0).tolist())
    with np.load(shard) as z:
        b=z["native_before"];a=z["native_after"]
        event=(b[:,C["player_state"]]==1)&(a[:,C["player_state"]]==4)
        hitstate=(b[:,C["player_state"]]==4)&(b[:,C["bombs"]]>0)
        evidence=dict(decisions=len(b),four_exact_ticks=int(z["four_tick_stage_transition"].sum()),
            native_hits=int(event.sum()),hit_state_with_stock=int(hitstate.sum()),
            bomb_requested_during_stock_hit=int(z["action"][hitstate,3].sum()),
            native_life_loss=int(z["native_life_losses"].sum()),bombs_consumed=int(z["actual_bombs_consumed"].sum()),
            max_stage=int(np.max(b[:,C["stage"]])))
    payload=dict(policy=trial["arm"],policy_seed=seed,student_on_policy=student,outcome=result["outcome"],
        continues=result["native_continues"],stage_records=result["stage_records"],student_weights=fingerprint,shard=digest(shard),
        measurements=evidence,stages=stage_metrics,interpretation="Behavior agreement means agreement with the actor that generated these states. A student-native credit supplies an actual survival result and an offline privileged-teacher label comparison on the student's state distribution.")
    if args.out.exists():raise FileExistsError(args.out)
    args.out.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf8")
    print(json.dumps({k:payload[k] for k in ("policy","policy_seed","student_on_policy","outcome","continues","measurements")}))


if __name__=="__main__":main()
