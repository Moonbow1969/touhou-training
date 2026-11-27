"""Score the registered, two-host TH10 native-clock replication of the large PPO batch."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.collect_th10_full_action import save
from scripts.package_th10_credit_logs import package
from scripts.render_th10_matched_credit import render
from scripts.score_original_camera_run import verify
from scripts.summarize_th10_full_credit import read_arm,summarize


def contrast(pairs,metric):
    hosts=sorted({r["host"] for r in pairs})
    rng=np.random.default_rng(26112610)
    groups={};draws=[];pooled=[]
    for host in hosts:
        arr=np.array([[r["baseline"][metric],r["candidate"][metric]] for r in pairs if r["host"]==host
                      and r["baseline"][metric] is not None and r["candidate"][metric] is not None])
        if not len(arr):continue
        delta=arr[:,1]-arr[:,0]
        dist=rng.choice(delta,(20000,len(delta)))
        draws.append(dist);pooled.append(arr)
        groups[host]=dict(n=len(delta),means=arr.mean(0).tolist(),paired_mean_change=float(delta.mean()),
                          bootstrap95=np.quantile(dist.mean(1),[.025,.975]).tolist())
    if not pooled:return dict(n=0,hosts=groups)
    arr=np.concatenate(pooled);delta=arr[:,1]-arr[:,0]
    return dict(n=len(delta),means=arr.mean(0).tolist(),paired_mean_change=float(delta.mean()),
                bootstrap95=np.quantile(np.concatenate(draws,axis=1).mean(1),[.025,.975]).tolist(),hosts=groups)


def run(study):
    plan=json.loads((study/"protocol.json").read_bytes());native=study/"native10";out=study/"evaluation"
    for name in ("action-traces","endpoints"):(out/name).mkdir(parents=True,exist_ok=True)
    arms=[plan["reference_arm"],plan["candidate"]];folders={a:[] for a in arms};pairs=[]
    for host,seeds in plan["seeds_by_physical_host"].items():
        assert len(seeds)==12 and plan["reference_first"][host]==seeds[:6] and plan["candidate_first"][host]==seeds[6:]
        local=native/host;local_plan=json.loads((local/"paired-protocol.json").read_bytes())
        assert local_plan["first_reference"]==seeds[:6] and local_plan["first_candidate"]==seeds[6:]
        for arm in arms:folders[arm].append(local/("held-"+arm))
        for seed in seeds:
            pair=dict(seed=seed,host=host)
            for arm,name in zip(arms,("baseline","candidate")):
                folder=local/("held-"+arm)/f"{host}-{seed}";f=folder/"report.json";r=json.loads(f.read_bytes())
                assert r["host"]==host and r["seed"]==seed and r["policy_sha256"]==plan["policy_sha256"][arm]
                assert r["difficulty"]==1 and r["game_sha256"]==plan["original_game_sha256"] and r["continues"]==0
                v=verify(10,f,r,folder/"behavior.npz",arm,out);clock=r["native_stage_clock"]
                assert clock["source_sha256"]==plan["stage_clock_source_sha256"] and v["latest_observed_stage1_native_tick"] is not None
                v.update(first_loss_native_stage=clock["first_life_loss_stage"],first_loss_stage1_tick=(
                    clock["first_life_loss_tick"] if clock["first_life_loss_stage"]==1 else None),
                    native_clock_rows=clock["rows"],stage_clock_trace_sha256=clock["data_sha256"],original_score=r["raw_score"],
                    clock_minus_terminal_ms=clock["sidecar_minus_stock_terminal_ms"],native_action_matches=r["native_inputs_matching"])
                pair[name]=v
            pairs.append(pair)
    assert sorted(r["seed"] for r in pairs)==plan["eval_seeds"]["10"]
    analysis={metric:contrast(pairs,metric) for metric in ("latest_observed_stage1_native_tick","first_loss_stage1_tick",
                  "original_score","terminal_power","effective_bombs","wall_seconds")}
    raw=summarize({a:read_arm(folders[a],plan["eval_seeds"]["10"]) for a in arms},plan["name"],arms[0])
    stage={a:{k:raw["group"][a][k] for k in ("n","stage2_reached","stage2_rate","stage2_wilson_95pct","stage2_resources",
        "native_game_overs","final_spare_life_counts","mean_final_power","confirmed_bomb_spends")} for a in arms}
    discordant=raw["paired_versus_"+arms[0]][arms[1]]
    summary=dict(arms=arms,measure="Observed stage-local tick at the last native sample tagged Stage 1.",
                 interpretation="The counter measures elapsed original simulation time within Stage 1. Boss duration can vary across credits. The native stage transition independently measures whether Stage 1 was cleared.",
                 seed_pairs=len(pairs),metrics=analysis,stage_and_resources=stage,
                 paired_stage2={k:discordant[k] for k in ("stage2_gained","stage2_lost","stage2_delta_rate","stage2_mcnemar_exact_p")})
    save(out/"native-clock-comparison.json",summary);save(out/"matched-credits.json",pairs)
    fig,axes=plt.subplots(1,2,figsize=(9,4.1),layout="constrained")
    for ax,metric,label in zip(axes,("latest_observed_stage1_native_tick","first_loss_stage1_tick"),
                               ("Final observed Stage 1 tick","Stage 1 tick at first life loss")):
        for host,color in (("accelerator","#346885"),("momentum","#ad643d")):
            for j,row in enumerate(r for r in pairs if r["host"]==host):
                y=[row[k][metric] for k in ("baseline","candidate")]
                if any(v is None for v in y):continue
                x=np.array([0.,1.])+(j-5.5)*.008
                ax.plot(x,y,color=color,alpha=.35,lw=.9);ax.scatter(x,y,color=color,s=11,label=host if j==0 else None)
        ax.set_xticks([0,1],["Prior PPO","Large-batch PPO"]);ax.set_xlim(-.15,1.15);ax.set_ylabel(label);ax.grid(axis="y",alpha=.18)
        ax.legend(fontsize=8)
    fig.suptitle("TH10 Normal · matched original-game credits");fig.savefig(out/"native-clock-pairs.png",dpi=170);plt.close(fig)
    package(study/"th10-held-native-logs.zip",[f for a in arms for f in folders[a]])
    seed=plan["preregistered_movie_seed"];host=next(r["host"] for r in pairs if r["seed"]==seed)
    video=study/f"video-{seed}/th10-native-clock-{seed}.mp4"
    render(SimpleNamespace(arms=[native/host/("held-"+a)/f"{host}-{seed}" for a in arms],out=video,
                           expected_arms=["held-"+a for a in arms],labels=["Prior PPO","Large-batch PPO"]))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("study",type=Path)
    parser.add_argument("--wait",action="store_true")
    args=parser.parse_args()
    if args.wait:
        deadline=time.monotonic()+2*3600
        for host in ("accelerator","momentum"):
            while not (args.study/"native10"/host/"paired.json").is_file():
                if time.monotonic()>=deadline:raise TimeoutError(f"No sealed {host} arm comparison")
                time.sleep(4)
    run(args.study)
