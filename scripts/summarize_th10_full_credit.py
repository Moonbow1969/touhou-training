"""Summarize paired whole-credit TH10 arms with resources and original game endpoints."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
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
            qualified=row.get('stage_sensor_validated') is True
            if qualified:
                proof=row['stage2_evidence']
                assert proof['stage_register']=='0x472b40' and proof['legacy_register']=='0x472b44'
                assert hashlib.sha256((p.parent/'native.csv.gz').read_bytes()).hexdigest()==proof['native_trace_sha256']
                assert proof['native_rows']==len(states) and all('legacy_start_field' in s for s in states)
                assert (proof['first_stage2'] is not None)==any(int(s['stage'])==2 for s in states)==row['stage2_reached']
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
            d.update(native_bomb_spends=row["successful_bombs"],loss_events=losses,stage_sensor_validated=qualified)
            # Native first-life coordinates exclude the recentering animation on respawn.
            first_loss=next((i for i,s in enumerate(states) if int(s['lives'])<int(states[0]['lives'])),len(states))
            xy=np.array([(float(s['x']),float(s['y'])) for s in states[:first_loss]])
            d['first_life_coordinate_span']=(xy.max(axis=0)-xy.min(axis=0)).tolist()
            out[row['seed']]=d
    if sorted(out)!=list(expected_seeds):raise ValueError("Incomplete frozen arm or unexpected controller seeds")
    return out


def summarize(arms,design=None,baseline='SFT'):
    groups={}
    paired={}
    base=arms[baseline]
    for name,items in arms.items():
        rows=[items[s] for s in sorted(items)]
        if len(set(r['policy_sha256'] for r in rows))!=1:raise ValueError("Changed policy within evaluation")
        event=[s for r in rows for s in r["loss_events"]]
        hits=sum(r['stage2_reached'] for r in rows)
        n=len(rows);proportion=hits/n;z=1.959963984540054
        center=(proportion+z*z/(2*n))/(1+z*z/n)
        radius=z*math.sqrt(proportion*(1-proportion)/n+z*z/(4*n*n))/(1+z*z/n)
        groups[name]=dict(n=n,stage2_reached=hits,stage2_rate=proportion,
            stage2_wilson_95pct=[max(0.,center-radius) if hits else 0.,min(1.,center+radius) if hits<n else 1.],
            native_game_overs=sum(r['native_outcome']=='native_final_life_lost' for r in rows),
            stage2_resources=[dict(seed=r['seed'],lives=r['stage2_lives'],power=r['stage2_power']) for r in rows if r['stage2_reached']],
            final_spare_life_counts=dict(Counter(r['final_lives'] for r in rows)),
            mean_final_power=float(np.mean([r['final_power'] for r in rows])),
            mean_wall_survival_s=float(np.mean([r['duration_s'] for r in rows])),
            mean_raw_score=float(np.mean([r['raw_score'] for r in rows])),
            confirmed_bomb_spends=sum(len(r['native_bomb_spends']) for r in rows),
            bomb_held_decisions=sum(r['bomb_on'] for r in rows),losses=len(event),
            movement_decisions=sum(r['steps']-r['moves'][4] for r in rows),
            moves_by_direction=np.array([r['moves'] for r in rows]).sum(0).tolist(),
            losses_with_at_least_one_power=sum(s['before_power']>=1 for s in event),
            losses_with_power_and_no_recent_bomb=sum(s['before_power']>=1 and not s['bomb_in_preceding_half_second'] for s in event),
            decisions=sum(r['steps'] for r in rows),decisions_matching_native_input=sum(r['native_inputs_matching'] for r in rows))
        spans=np.array([r['first_life_coordinate_span'] for r in rows])
        groups[name]['native_first_life_movement']=dict(
            credits_moving_on_both_axes=int(np.all(spans>0,axis=1).sum()),
            median_xy_span_game_units=np.median(spans,axis=0).tolist(),
            min_xy_span_game_units=spans.min(axis=0).tolist())
        if name!=baseline:
            blocks=[]
            for r in rows:
                parent=base[r['seed']]
                if parent['host']!=r['host']:raise ValueError("Frozen pair ran on different physical machines")
                blocks.append(dict(seed=r['seed'],host=r['host'],
                    stage2_baseline=bool(parent['stage2_reached']),stage2_candidate=bool(r['stage2_reached']),
                    delta_wall_survival_s=r['duration_s']-parent['duration_s'],
                    delta_raw_score=r['raw_score']-parent['raw_score'],
                    delta_settled_power=r['final_power']-parent['final_power']))
            v=np.array([r['delta_wall_survival_s'] for r in blocks]);rng=np.random.default_rng(10860)
            gained=sum(b['stage2_candidate'] and not b['stage2_baseline'] for b in blocks)
            lost=sum(b['stage2_baseline'] and not b['stage2_candidate'] for b in blocks)
            discordant=gained+lost
            exact=min(1.,2*sum(math.comb(discordant,i) for i in range(min(gained,lost)+1))/(2**discordant))
            paired[name]=dict(pairs=blocks,mean_survival_delta_s=float(v.mean()),positive=int((v>0).sum()),
                 stage2_gained=gained,stage2_lost=lost,stage2_delta_rate=(gained-lost)/len(rows),stage2_mcnemar_exact_p=exact,
                 paired_resampled_95pct_interval_for_mean_s=np.quantile(rng.choice(v,(10000,len(v))).mean(1),[.025,.975]).tolist())
    credits={name:[dict(group[s]) for s in sorted(group)] for name,group in arms.items()}
    for name,rows in credits.items():
        qualified=all(r.get('stage_sensor_validated') is True for r in rows)
        groups[name]['stage2_sensor_validated']=qualified
        if not qualified:
            for k in ('stage2_reached','stage2_rate','stage2_wilson_95pct','stage2_resources'):groups[name][k]=None
            for r in rows:
                if r.get('stage_sensor_validated'):continue
                r['legacy_stage2_flag']=r['stage2_reached']
                for k in ('stage2_reached','stage2_lives','stage2_power','stage_entries','final_stage'):r[k]=None
    for name,diff in paired.items():
        if groups[name]['stage2_sensor_validated'] and groups[baseline]['stage2_sensor_validated']:continue
        for k in ('stage2_gained','stage2_lost','stage2_delta_rate','stage2_mcnemar_exact_p'):diff[k]=None
        for pair in diff['pairs']:pair['stage2_candidate']=pair['stage2_baseline']=None
    return dict(design=design or '12 unique controller seeds per frozen arm. Same native TH10 Normal trial and host within each pair; native RNG uncontrolled. All lives and no Continue. First six controller seeds played SFT before PPO1, final six PPO1 before SFT. TH10 has Power-paid spirit strikes; spare count -1 means last life was lost. Real-time survival is sensitive to game speed and is a secondary measure.',n_per_arm=len(base),
        group=groups,**{f"paired_versus_{baseline.lower()}":paired},credits=credits)


def draw(arms,out,difficulty_name="Normal"):
    fig,axs=plt.subplots(1,3,figsize=(10.8,3.9),layout="constrained")
    names=list(arms);colors=["#2275a0","#d47537","#497a56"]
    for ax,key,label in zip(axs,("duration_s","raw_score","final_power"),("Whole-credit wall seconds","Original score (× 10,000)","Power at Game Over")):
        table=np.array([[arms[g][s][key] for g in names] for s in sorted(arms[names[0]])])
        if key=="raw_score":table=table/10000.
        for i,row in enumerate(table):
            jitter=(i-(len(table)-1)/2)*.009;x=np.arange(len(names))+jitter
            ax.plot(x,row,color="#a4afba",alpha=.56,lw=.8)
            for j,v in enumerate(row):ax.scatter(x[j],v,color=colors[j],s=14)
        ax.scatter(range(len(names)),table.mean(0),color="#26242b",marker="_",s=270,lw=2.3,zorder=4)
        ax.set_xticks(range(len(names)),names);ax.set_ylabel(label);ax.set_xlim(-.35,len(names)-.65);ax.grid(axis="y",alpha=.18)
        ax.spines[["right","top"]].set_visible(False)
    fig.suptitle(f"TH10 original {difficulty_name}  ·  {len(arms[names[0]])} matched full credits per policy  ·  no Continues")
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
