"""Independent TH10 Lunatic replication with randomized counterbalanced pairs on two Momentum lanes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scripts.collect_th10_full_action import save,pin,install_weights,start_camera_service,one
from scripts.run_th10_camera_cohort import checkpoint,digest,verify
from scripts.run_th10_high_rank_online import native_ledger
from scripts.package_th10_credit_logs import package
from scripts.render_th10_matched_credit import render

def run(root,workers):
    study=root/"th10-lunatic-merged-replication";native=study/"native10";native.mkdir(parents=True,exist_ok=True)
    ref,alt="merged-sft","merged-ppo";models={ref:root/"specialist-vs-shared/merged-equal/policy.ts",alt:root/"four-game-merged-online/policy/policy.ts"}
    shas={arm:digest(model) for arm,model in models.items()}
    assert list(shas.values())==["1f7f1a6b73fee215c0142a9a27be8d1523d38baa9ad99fba3d37d4839a792b3a","e69e667987fed0fceb721cbabefbb1e03f956af0a99f763de341d64298e59c18"]
    seeds=list(range(1411051,1411075));rng=np.random.default_rng(2026112701);allocation=[]
    for lane in (0,1):
        group=seeds[lane::2];order=rng.permutation([ref]*6+[alt]*6).tolist()
        for seed,first in zip(group,order):
            allocation.append(dict(seed=seed,host="momentum",queue=lane,first=first,second=alt if first==ref else ref))
    protocol=dict(name="Independent TH10 Lunatic replication",policy_sha256=shas,seeds=seeds,allocation=allocation,rng_seed=2026112701,
        control="Official TH10 0.02a trial, Lunatic menu and original per-row difficulty3, full ReimuA start, both maximum native lives, all four policy actions, 120ms control, no assistance or Continue.",
        pairing="Two bounded, sequential Momentum queues. Each pair uses the same logical queue, physical machine, build and controller seed and runs consecutive credits. OS CPU affinity and the original hidden game RNG are uncontrolled. Within each queue exactly six first runs for each arm.",
        primary="Paired mean candidate minus reference last observed native Stage1 tick with 95% percentile paired bootstrap, fixed 50,000 draws sampled on complete pairs. No pooling with discovery sample.",
        stage2="Same independently qualified stage sensor; Stage2 reach count and paired binary change. First Stage2 reserves and Power if reached, settled zero spares and Power plus confirmed Power-backed X activations.",first_video_seed=seeds[0])
    out=study/"protocol.json"
    if out.exists():assert json.loads(out.read_bytes())==protocol
    else:assert not any(native.rglob("report.json"));save(out,protocol)
    client=json.loads(workers.read_bytes())["momentum"];build=pin(client,camera=True,difficulty=3,stage_sensor="qualified-v2-clock")
    for arm,file in models.items():
        install_weights(client,file,shas[arm]);start_camera_service(client,build,shas[arm]);(native/arm).mkdir(exist_ok=True)
    def group(slot):
        for a in allocation:
            if a["queue"]!=slot:continue
            for arm in (a["first"],a["second"]):
                one((client,build,shas[arm],"momentum",a["seed"],arm,native/arm,120,False))
    with ThreadPoolExecutor(2) as pool:list(pool.map(group,(0,1)))
    torch.set_num_threads(2)
    for arm,model in models.items():verify(native/arm,model,checkpoint(model),seeds,"held",host="momentum",difficulty=3)
    rows=native_ledger([native/ref,native/alt],seeds,3,"momentum")
    save(study/"qualified-native-credits.json",rows)
    lookup={(r["seed"],r["arm"]):r for r in rows}
    def contrast(k):
        matrix=np.array([[lookup[s,arm][k] for arm in (ref,alt)] for s in seeds],dtype=float)
        delta=matrix[:,1]-matrix[:,0];boot=np.random.default_rng(2026112702).choice(delta,(50000,len(seeds))).mean(1)
        return dict(n=len(seeds),mean=matrix.mean(0).tolist(),paired_difference=float(delta.mean()),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),pairs=[dict(seed=s,reference=float(v[0]),candidate=float(v[1])) for s,v in zip(seeds,matrix)])
    result=dict(arms=[ref,alt],primary=contrast("last_observed_stage1_tick"),stage2=contrast("stage2"),
                final_power=contrast("terminal_power"),first_loss=contrast("first_loss_tick"),confirmed_casts=contrast("effective_strikes"),
                scores=contrast("score"),terminal_spares=contrast("terminal_spares"),entries=[r for r in rows if r["stage2"]],
                stage2_counts=[sum(r["stage2"] for r in rows if r["arm"]==arm) for arm in (ref,alt)],protocol_sha256=digest(out))
    save(study/"replication.json",result)
    package(study/"native-credits.zip",[native/ref,native/alt])
    arms=[native/arm/f"momentum-{seeds[0]}" for arm in (ref,alt)]
    render(SimpleNamespace(arms=arms,expected_arms=[ref,alt],labels=["Merged human SFT","Four-game merged PPO"],out=study/"video/th10-lunatic-1411051.mp4"))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("workers",type=Path);a=p.parse_args();run(a.root.resolve(),a.workers.resolve())

