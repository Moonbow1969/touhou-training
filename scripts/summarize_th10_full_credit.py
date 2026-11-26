"""Summarize paired whole-credit TH10 arms with resources and original game endpoints."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_arm(directories, expected_seeds=range(711001,711013)):
    out={}
    for directory in directories:
        for p in Path(directory).glob("*/report.json"):
            row=json.loads(p.read_bytes())
            if row["seed"] in out or not row["terminal"] or row["continues"] or row["policy_sha256"] is None:
                raise ValueError("Duplicate or incomplete frozen credit")
            with gzip.open(p.parent/"native.csv.gz","rt") as f:
                states=list(csv.DictReader(f))
            with np.load(p.parent/"behavior.npz") as behaviors:
                utc=behaviors["action_utc_ms"];actions=behaviors["action"]
            losses=[]
            for a,b in zip(states,states[1:]):
                if int(b["lives"])>=int(a["lives"]):continue
                time=int(b['utc_ms'])
                idx=(utc<=time)&(utc>time-500)
                losses.append(dict(before_power=int(a['power_units'])/20.,
                                   seconds=(time-utc[0]).item()/1000.,
                                   bomb_in_preceding_half_second=bool(actions[idx,3].any())))
            d={k:row[k] for k in ("host","seed","policy_sha256","behavior_sha256","terminal_screenshot_sha256",
                "steps","native_outcome","stage2_reached","stage2_lives","stage2_power","stage_entries",
                "final_stage","final_lives","final_power","power_at_final_life_loss","final_power_source",
                "raw_score","duration_s","native_inputs_matching","native_inputs_with_sample","median_control_ms",
                "shoot_on","focus_on","bomb_on","moves")}
            d.update(native_bomb_spends=row["successful_bombs"],loss_events=losses)
            out[row['seed']]=d
    if sorted(out)!=list(expected_seeds):raise ValueError("Incomplete frozen arm or unexpected controller seeds")
    return out


def summarize(arms,design=None):
    groups={}
    paired={}
    base=arms['SFT']
    for name,items in arms.items():
        rows=[items[s] for s in sorted(items)]
        if len(set(r['policy_sha256'] for r in rows))!=1:raise ValueError("Changed policy within evaluation")
        event=[s for r in rows for s in r["loss_events"]]
        groups[name]=dict(n=len(rows),stage2_reached=sum(r['stage2_reached'] for r in rows),
            native_game_overs=sum(r['native_outcome']=='native_final_life_lost' for r in rows),
            stage2_resources=[dict(seed=r['seed'],lives=r['stage2_lives'],power=r['stage2_power']) for r in rows if r['stage2_reached']],
            mean_final_power=float(np.mean([r['final_power'] for r in rows])),
            mean_wall_survival_s=float(np.mean([r['duration_s'] for r in rows])),
            mean_raw_score=float(np.mean([r['raw_score'] for r in rows])),
            confirmed_bomb_spends=sum(len(r['native_bomb_spends']) for r in rows),
            bomb_held_decisions=sum(r['bomb_on'] for r in rows),losses=len(event),
            losses_with_at_least_one_power=sum(s['before_power']>=1 for s in event),
            losses_with_power_and_no_recent_bomb=sum(s['before_power']>=1 and not s['bomb_in_preceding_half_second'] for s in event),
            decisions=sum(r['steps'] for r in rows),decisions_matching_native_input=sum(r['native_inputs_matching'] for r in rows))
        if name!='SFT':
            blocks=[]
            for r in rows:
                parent=base[r['seed']]
                if parent['host']!=r['host']:raise ValueError("Frozen pair ran on different physical machines")
                blocks.append(dict(seed=r['seed'],host=r['host'],
                    delta_wall_survival_s=r['duration_s']-parent['duration_s'],
                    delta_raw_score=r['raw_score']-parent['raw_score'],
                    delta_settled_power=r['final_power']-parent['final_power']))
            v=np.array([r['delta_wall_survival_s'] for r in blocks]);rng=np.random.default_rng(10860)
            paired[name]=dict(pairs=blocks,mean_survival_delta_s=float(v.mean()),positive=int((v>0).sum()),
                 paired_resampled_95pct_interval_for_mean_s=np.quantile(rng.choice(v,(10000,len(v))).mean(1),[.025,.975]).tolist())
    return dict(design=design or '12 unique controller seeds per frozen arm. Same native TH10 Normal trial and host within each pair; native RNG uncontrolled. All lives and no Continue. First six controller seeds played SFT before PPO1, final six PPO1 before SFT. TH10 has Power-paid spirit strikes; spare count -1 means last life was lost. Real-time survival is sensitive to game speed and is a secondary measure.',n_per_arm=len(base),
        group=groups,paired_versus_sft=paired,credits={name:[group[s] for s in sorted(group)] for name,group in arms.items()})


def draw(arms,out):
    fig,axs=plt.subplots(1,3,figsize=(10.8,3.9),layout="constrained")
    names=list(arms);colors=["#2275a0","#d47537","#497a56"]
    for ax,key,label in zip(axs,("duration_s","raw_score","final_power"),("Whole-credit wall seconds","Original score (× 10,000)","Power at Game Over")):
        table=np.array([[arms[g][s][key] for g in names] for s in sorted(arms[names[0]])])
        if key=="raw_score":table=table/1000.
        for i,row in enumerate(table):
            jitter=(i-(len(table)-1)/2)*.009;x=np.arange(len(names))+jitter
            ax.plot(x,row,color="#a4afba",alpha=.56,lw=.8)
            for j,v in enumerate(row):ax.scatter(x[j],v,color=colors[j],s=14)
        ax.scatter(range(len(names)),table.mean(0),color="#26242b",marker="_",s=270,lw=2.3,zorder=4)
        ax.set_xticks(range(len(names)),names);ax.set_ylabel(label);ax.set_xlim(-.35,len(names)-.65);ax.grid(axis="y",alpha=.18)
        ax.spines[["right","top"]].set_visible(False)
    fig.suptitle(f"TH10 original Normal  ·  {len(arms[names[0]])} matched full credits per policy  ·  no Continues")
    fig.supxlabel("Lines pair controller seeds within the same physical host. Bars show arm means.",fontsize=9)
    fig.savefig(out,dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft",type=Path,nargs="+",required=True);parser.add_argument("--ppo1",type=Path,required=True)
    parser.add_argument("--ppo2",type=Path);parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    arms={"SFT":read_arm(args.sft),"PPO1":read_arm([args.ppo1])}
    if args.ppo2:arms['PPO2']=read_arm([args.ppo2])
    args.out.with_suffix('.json').write_text(json.dumps(summarize(arms),indent=2)+'\n',encoding="utf-8")
    draw(arms,args.out.with_suffix('.png'))
