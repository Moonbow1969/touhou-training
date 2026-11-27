"""Summarize paired endpoints, observed physical motion and action cadence for the native TH10 2×2."""
import argparse
from collections import Counter
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.check_th10_native_motion import audit as motion
from scripts.collect_th10_full_action import save
from scripts.estimate_th10_motion_ticks import main as cadence
from scripts.package_th10_credit_logs import package
from scripts.run_th10_camera_pair import switches
from scripts.summarize_th10_full_credit import read_arm,summarize


def summarize_factorial(base):
    plan=json.loads((base/'protocol.json').read_bytes());seeds=plan['seeds'];host=plan['host']
    arms={a:read_arm([base/a],seeds) for a in plan['settings']}
    report=summarize(arms,plan['design'],'four-120')
    metric={}
    for a,rows in arms.items():
        folder=base/a
        times=cadence(folder,'held')
        changes=switches(folder,host,seeds)
        physical=[motion(folder/f'{host}-{seed}') for seed in seeds]
        whole={key:sum((r['counts'][key] for r in physical),Counter()) for key in ('horizontal','vertical','diagonal')}
        report['group'][a].update(movement_switches=changes,physical_displacement=whole,
             physical_displacement_per_credit=physical,motion_clock=times,
             switches_per_second=sum(r['switches'] for r in changes['credits'])/sum(r['duration_s'] for r in rows.values()))
        vals={}
        for i,seed in enumerate(seeds):
            r=rows[seed];s=changes['credits'][i];t=times['credits'][i];assert s['seed']==t['seed']==seed
            vals[seed]=dict(stage2=float(r['stage2_reached']),duration_s=r['duration_s'],raw_score=r['raw_score'],
                final_spares=r['final_lives'],final_power=r['final_power'],confirmed_spirit_strikes=len(r['native_bomb_spends']),
                movement_switch_rate=s['rate'],switches_per_second=s['switches']/r['duration_s'],
                median_control_ms=t['median_control_ms'],estimated_motion_ticks_per_decision=t['estimated_motion_increments_per_decision'])
        metric[a]=vals
    contrasts={'fast_minus_slow_four':{'four-62':1,'four-120':-1},
               'fast_minus_slow_multi':{'multi-62':1,'multi-120':-1},
               'multi_minus_four_fast':{'multi-62':1,'four-62':-1},
               'multi_minus_four_slow':{'multi-120':1,'four-120':-1},
               'interaction_multi_minus_four_of_fast_minus_slow':{'multi-62':1,'multi-120':-1,'four-62':-1,'four-120':1}}
    result={}
    for label,weights in contrasts.items():
        output={}
        for key in next(iter(metric['four-120'].values())):
            valid=[s for s in seeds if all(metric[a][s][key] is not None for a in weights)]
            if not valid:continue
            delta=np.asarray([sum(w*metric[a][s][key] for a,w in weights.items()) for s in valid])
            boot=np.random.default_rng(759001).choice(delta,(10000,len(valid))).mean(1)
            output[key]=dict(n=len(valid),mean=float(delta.mean()),paired_bootstrap_95pct=np.quantile(boot,[.025,.975]).tolist(),positive=int((delta>0).sum()))
        result[label]=output
    report['factorial_paired_contrasts']=result; report['per_seed_metrics']=metric
    save(base/'factorial.json',report)
    fig,axes=plt.subplots(1,4,figsize=(12,3.7),layout='constrained')
    for ax,key,title in zip(axes,['estimated_motion_ticks_per_decision','duration_s','raw_score','movement_switch_rate'],
                             ['Native motion increments','Time to Game Over (s)','Original score (×10,000)','Movement change (%)']):
        for model,color,label in [('four','#206eaa','4-tick training'),('multi','#b55f26','4/8-tick training')]:
            values=np.array([[metric[f'{model}-{h}'][s][key] for h in (62,120)] for s in seeds],dtype=float)
            if key=='raw_score':values/=1000
            if key=='movement_switch_rate':values*=100
            ax.plot([0,1],values.T,color=color,alpha=.18,linewidth=.9)
            ax.plot([0,1],np.nanmean(values,0),'-o',color=color,label=label,linewidth=2.4,markersize=4)
        ax.set_title(title,fontsize=10);ax.set_xticks([0,1],['62 ms','120 ms'])
        ax.set_xlim(-.15,1.15);ax.grid(axis='y',alpha=.18);ax.spines[['right','top']].set_visible(False)
    axes[0].legend(fontsize=8);fig.suptitle('TH10 Normal · eight matched full credits per condition')
    fig.supxlabel('Configured key hold. Thin lines connect one controller seed. Thick lines show condition means.',fontsize=9)
    fig.savefig(base/'factorial.png',dpi=175);plt.close(fig)
    package(base/'native-logs.zip',[base/a for a in arms])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('base',type=Path)
    summarize_factorial(p.parse_args().base)
