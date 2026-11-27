"""Run counterbalanced blocks of three independent original TH13 credits for frozen supervised policies."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.collect_th10_full_action import save
from scripts.read_th13_trial_ranking import scan
from scripts.run_th13_accelerator_camera import run as collect
from scripts.score_native_stage_clock import summarize as clock_score
from scripts.score_original_camera_run import verify
from scripts.run_th10_camera_cohort import digest
from scripts.read_th13_trial_ranking import TRIAL_SHA256


def frozen(root, arm, path):
    report_path=path.parent/'report.json'
    if not report_path.is_file() or not path.is_file():
        return None
    report=json.loads(report_path.read_bytes())
    actual=digest(path)
    if report['torchscript_sha256']!=actual:
        raise ValueError(f'{arm} checkpoint SHA is absent from its frozen training report')
    if arm.startswith('new-'):
        milestone=root/'human-train-incremental/registered-024-unique-train.json'
        protocol=path.parent.parent/'protocol.json'
        if not milestone.is_file() or not protocol.is_file():
            return None
        if json.loads(protocol.read_bytes())['milestone_sha256']!=digest(milestone):
            raise ValueError(f'{arm} belongs to a different registered training milestone')
    return actual


def score(out, plan):
    native=out/'native13'
    public=out/'evaluation'
    for name in ('action-traces','endpoints'):(public/name).mkdir(parents=True,exist_ok=True)
    photos=scan(native)
    photo={Path(r['path']).parent.name:r for r in photos['rows']}
    seeds=plan['eval_seeds']['13'];rows=[];groups={}
    archive=out/'th13-held-native-logs.zip';temporary=archive.with_suffix('.staging')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for arm in plan['arms']:
            arm_rows=[]
            for seed in seeds:
                tag=f'held-{arm}-{seed}';folder=native/tag;record=folder/'receipt.json'
                receipt=json.loads(record.read_bytes());reading=photo[tag]
                assert receipt['host']=='accelerator' and receipt['arm']==arm and receipt['seed']==seed
                assert receipt['game_sha256']==TRIAL_SHA256 and receipt['continues']==0
                if receipt['outcome']=='game_over':assert receipt['remaining_lives']==-1
                assert receipt['terminal'] and receipt['policy_sha256']==frozen(out.parents[1],arm,out.parents[1]/plan['arms'][arm])
                credit=verify(13,record,receipt,folder/'rollout.npz',arm,public)
                native_states=np.fromfile(folder/'steps.bin','<f4').reshape(-1,143)
                assert len(native_states)==credit['decisions'] and list(native_states[0,[125,127,129,141]])==[1,2,2,1]
                first=np.flatnonzero(native_states[:,136]>0)
                credit.update(first_native_loss_tick=int(native_states[first[0],135]) if len(first) else None,
                    score_original=int(native_states[-1,131])*10,rankscreen_stage=reading['readout'],
                    ranking_matches_native=reading['matches_native'],native_key_matches=receipt['native_key_match'],
                    native_steps_sha256=digest(folder/'steps.bin'),actual_start=receipt['start'],actual_end=receipt['finish'])
                for name in ('receipt.json','ranking.png','steps.bin','events.tsv','actor.log','final-stock.bin'):
                    z.write(folder/name,f'{tag}/{name}')
                arm_rows.append(credit)
            rows.extend(arm_rows)
            groups[arm]=dict(credits=len(arm_rows),stage2=sum(r['stage2_reached'] for r in arm_rows),
                independently_corroborated_rankings=sum(r['ranking_matches_native'] is True for r in arm_rows),
                unresolved_rankings=sum(r['rankscreen_stage'] is None for r in arm_rows),
                mean_last_stage1_tick=float(np.mean([r['latest_observed_stage1_native_tick'] for r in arm_rows])),
                mean_first_loss_tick=float(np.mean([r['first_native_loss_tick'] for r in arm_rows if r['first_native_loss_tick'] is not None])),
                mean_original_score=float(np.mean([r['score_original'] for r in arm_rows])),
                casts=sum(r['effective_bombs'] for r in arm_rows),mean_terminal_bombs=float(np.mean([r['terminal_bombs'] for r in arm_rows])),
                zero_remaining_lives=sum(r['terminal_spares']==0 for r in arm_rows),
                decisions=sum(r['decisions'] for r in arm_rows),native_key_matches=sum(r['native_key_matches'] for r in arm_rows))
        save(public/'credits.json',rows); save(public/'ranking-readout.json',photos)
        comparisons={}
        for label in plan['contrasts']:
            new,old=label.split(' minus ')
            comparisons[label]=clock_score(rows,13,(old,new))
        save(public/'paired.json',dict(primary=plan['primary'],groups=groups,contrasts=comparisons))
        names=list(groups)
        fig,axes=plt.subplots(1,2,figsize=(10.2,4.4),layout='constrained')
        for credit in rows:
            credit['plot_order']=names.index(credit['arm'])
        for seed in seeds:
            sample=sorted((r for r in rows if r['seed']==seed),key=lambda r:r['plot_order'])
            axes[0].plot(range(4),[r['latest_observed_stage1_native_tick'] for r in sample],color='#74828e',alpha=.45,lw=.8)
        axes[0].scatter(range(4),[groups[n]['mean_last_stage1_tick'] for n in names],color='#1b5e78',zorder=3)
        axes[0].set_xticks(range(4),['14-player\nCNN','14-player\nattention','38-player\nCNN','38-player\nattention'])
        axes[0].set_ylabel('Last observed Stage 1 script tick')
        for i,(label,stats) in enumerate(comparisons.items()):
            lower,upper=stats['paired_fixed_host_bootstrap95']
            axes[1].hlines(i,lower,upper,color='#1b5e78',lw=2)
            axes[1].scatter(stats['mean_paired_change'],i,color='#1b5e78',zorder=3)
        axes[1].axvline(0,color='#738290',ls='--',lw=.9)
        axes[1].set_yticks(range(len(comparisons)),['38 − 14 players\nCNN','38 − 14 players\nattention','Attention − CNN\n38 players'])
        axes[1].set_xlabel('Paired change in native ticks')
        for ax in axes:ax.spines[['right','top']].set_visible(False)
        fig.suptitle('TH13 Normal original game · 12 full credits per frozen policy')
        fig.supxlabel('Lines on the left pair controller seeds. Bars on the right show unadjusted 95% paired bootstrap intervals.',fontsize=9)
        fig.savefig(public/'paired-stage-clock.png',dpi=150);plt.close(fig)
        for p in [out/'protocol.json',out/'dispatch.jsonl',public/'credits.json',public/'ranking-readout.json',public/'paired.json',public/'paired-stage-clock.png']:
            z.write(p,p.relative_to(out).as_posix())
    with zipfile.ZipFile(temporary) as z:assert z.testzip() is None
    temporary.replace(archive)


def run(args):
    root=args.root.resolve();out=root/'scaled-human-sft/native-24';native=out/'native13'
    plan=json.loads((out/'protocol.json').read_bytes());seeds=plan['eval_seeds']['13']
    names=list(plan['arms'])
    if names!=['base-cnn','base-attention','new-cnn','new-attention'] or len(seeds)!=12:
        raise ValueError('Unexpected experiment registration')
    schedule=[(names[i],seeds[((i+j)%4)*3:((i+j)%4+1)*3]) for j in range(4) for i in ((0,1,2,3) if j%2==0 else (3,2,1,0))]
    def record(**items):
        with (out/'dispatch.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),**items))+'\n')
    if args.wait_initial:
        # The initial three were started by a standalone invocation while this supervisor was being staged.
        while not all((native/f'held-base-cnn-{seed}/receipt.json').exists() for seed in seeds[:3]):time.sleep(3)
        first=[json.loads((native/f'held-base-cnn-{s}/receipt.json').read_bytes()) for s in seeds[:3]]
        record(event='standalone_first_block',arm=names[0],seeds=seeds[:3],policy_sha256=first[0]['policy_sha256'],
               native_starts=[r['start'] for r in first])
    unchanged={}
    while True:
        pending=False
        for arm,block in schedule:
            if all((native/f'held-{arm}-{s}/receipt.json').is_file() for s in block):continue
            pending=True;path=root/plan['arms'][arm];fingerprint=frozen(root,arm,path)
            if fingerprint is None:continue
            if unchanged.setdefault(arm,fingerprint)!=fingerprint:raise ValueError('Weights changed within held arm')
            record(event='dispatch',arm=arm,seeds=block,policy_sha256=fingerprint)
            collect(SimpleNamespace(qualification=root/'engineering/th13-accelerator',connection_config=args.workers.resolve(),
                checkpoint=path,output=native,role='held',arm=arm,seeds=block,workers=3))
            record(event='complete_block',arm=arm,seeds=block,policy_sha256=fingerprint)
            break
        else:
            if not pending:break
            time.sleep(6)
    score(out,plan)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=Path,required=True)
    p.add_argument('--wait-initial',action='store_true')
    p.add_argument('--score-only',action='store_true')
    args=p.parse_args()
    if args.score_only:
        out=args.root.resolve()/'scaled-human-sft/native-24';score(out,json.loads((out/'protocol.json').read_bytes()))
    else:run(args)
