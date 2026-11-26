"""Evaluate a frozen student’s waypoint error and live-wrapper data parity without controlling a game."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.distill_state_route import C,HOLD_NATIVE,PAIR,ROOT,SCALE,WORLD,Teacher,labels,scene
from scripts.state_route_student import StateActor,StockActor


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    out=ROOT/"outputs/th14-state-only-route-student-261125"
    teacher=Teacher();weights=out/"student.pt"
    model=StateActor().eval();model.load_state_dict(torch.load(weights,weights_only=True)["state_dict"])
    info=json.loads((out/"report.json").read_bytes())
    support={}
    for record in info["train"]:
        path=ROOT/record["file"]
        if path.parent.name!="transition-shards":continue
        with np.load(path) as z:
            b=z["native_before"];stage=b[:,C["stage"]];spell=b[:,C["spell_id"]];v=b[:,C["boss_vulnerable"]]==1
            support[path.name]=dict(decisions=len(b),stages=np.unique(stage).tolist(),
                stage1_vulnerable_spell8=int(((stage==1)&v&(spell==8)).sum()),hit_with_stock=int(((b[:,C["player_state"]]==4)&(b[:,C["bombs"]]>0)).sum()),
                stock_range=[int(b[:,C["bombs"]].min()),int(b[:,C["bombs"]].max())])
    result={}
    for directory in HOLD_NATIVE:
        path=next((WORLD/directory/"transition-shards").glob("*.npz"))
        d=scene(path,teacher,True)
        x,c,r=(torch.tensor(d[k]) for k in ("x","clock","raw"))
        *heads,g=model(x,c,r)
        direct=np.stack([v.argmax(1).numpy() for v in heads],1)
        predicted_target=g.numpy()*SCALE
        distance=np.linalg.norm(predicted_target-d["target"],axis=1)
        learned_goal=dict(d,target=predicted_target.astype("float32"))
        output_at_predicted_goal=labels(teacher,learned_goal,d["x"][:,:2]*SCALE)
        wrapper=StockActor(weights);env=SimpleNamespace(reset_count=1)
        wrapper.environment=env
        agreement=0;count=0;differences=[]
        with np.load(path) as z:
            b=z["native_before"];ctx=z["context"];bul=z["bullets"]
            for i,row in enumerate(b):
                if row[C["snapshot_valid"]]!=1:continue
                st,hp,spell,player_state=(int(row[C[k]]) for k in ("stage","boss_max_hp","spell_id","player_state"))
                env.snapshot=SimpleNamespace(stage=st,boss=SimpleNamespace(max_hp=hp,vulnerable=bool(row[C["boss_vulnerable"]]==1)) if hp>=0 else None,
                                           spell_id=spell if spell>=0 else None,player_state=player_state)
                env.held={"x"} if row[C["previous_x"]] else set()
                env.last_frame=int(row[C["stage_frame"]])
                env.game=SimpleNamespace(read_game_val=lambda key:int(row[C["bombs"]]))
                acted,_=wrapper.predict(dict(context=ctx[i],bullets=bul[i]))
                if not np.array_equal(acted,direct[count]):
                    single=model(x[count:count+1],c[count:count+1],r[count:count+1])[:4]
                    batch_margins=[float(torch.topk(v[count],2).values.diff().abs().item()) for v in heads]
                    differences.append(dict(index=i,batched=direct[count].tolist(),wrapper=acted.tolist(),
                        rowwise=[v.argmax(1).item() for v in single],batch_margin=batch_margins))
                agreement+=int(np.array_equal(acted,direct[count]));count+=1
        slices={}
        for name,msk in (("stage1",d["stage"]==1),("stage2",d["stage"]==2),("stage1_spell8",d["spell"]==8),("near",d["danger"])):
            if not msk.any():continue
            slices[name]=dict(n=int(msk.sum()),median_target_error_game_units=float(np.median(distance[msk])),
                p90_target_error_game_units=float(np.quantile(distance[msk],.9)),
                favorable_counterfactual_move_accuracy=float((output_at_predicted_goal[msk,0]==d["actual"][msk,0]).mean()),
                idle_accuracy_reference=float((d["actual"][msk,0]==4).mean()))
        result[path.name]=dict(wrapper_exact_joint=agreement,wrapper_rows=count,rowwise_disagreements=differences,slices=slices)
    payload=dict(training_native_support=support,observed_holdouts=result,
        counterfactual="Teacher motion decoder sees the student's predicted waypoint, true position and privileged demonstrator button preference. Diagnostic only. Repeated frozen inference has no training or model-selection effect.",
        decision="Reject native evaluation. Both held-out hazard and pre-hit motion thresholds failed.")
    (out/"diagnostic.json").write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf8")
    print(json.dumps(payload["observed_holdouts"]))


if __name__=="__main__":main()
