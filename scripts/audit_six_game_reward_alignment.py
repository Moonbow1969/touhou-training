"""Audit whether six-title physical training returns rank progress within each title and difficulty."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scripts.audit_th10_reward_alignment import ranks
from scripts.checkpoint_storage import digest
from scripts.score_original_camera_run import records
from scripts.th10_native_stage_clock import verify as th10_tick
from scripts.th12_playable_actions import playable_length
from scripts.train_five_original_games import freeze
from scripts.verify_th17_original_credit import inspect as th17_native


def original_progress(t,path,r):
    if t==10:return th10_tick(path.parent,r),'last playable stage 1 native tick'
    if t==11:
        native=path.parent/'native.bin'
        if digest(native)!=r['evidence_sha256']['native.bin']:raise ValueError('TH11 score evidence changed')
        last=np.fromfile(native,'<i4').reshape(-1,12)[-1]
        if last[5]!=-1 or last[8]!=0:raise ValueError('TH11 final score is not settled')
        return int(last[1])*10,'settled original score'
    if t==12:return r['covered_native_events']['death_native_tick'],'fatal native tick'
    if t==13:
        f=path.parent/r['file']
        if digest(f)!=r['sha256']:raise ValueError('TH13 stage clock changed')
        with np.load(f,allow_pickle=False) as z:return int(z['native_frame'][z['stage']==1].max()),'last observed stage 1 native tick'
    if t==17:return th17_native(path,r,r['difficulty'])['last_playable_stage1_tick'],'last playable stage 1 native tick'
    raise ValueError('Unexpected original game')


def correlation(x,y):
    a=ranks(x);b=ranks(y)
    a-=a.mean();b-=b.mean()
    norm=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/norm) if norm else None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True,type=Path)
    d=p.parse_args().data.resolve();s=d/'six-game-online';root=s/'joint-six/reward-audit';root.mkdir(exist_ok=True)
    manifest=s/'joint-six/optimizer-input/credits.json';rows=json.loads(manifest.read_bytes())
    if digest(manifest)!=json.loads((s/'joint-six/policy/report.json').read_bytes())['manifest_sha256']:raise ValueError('Unverified training records')
    plan=dict(manifest=digest(manifest),gamma=.999,
        objective='At the frozen on-policy six-title training batch, compare the ranking of actual first-step discounted or undiscounted return to an independently observed original progress metric. Report within-title/rank Spearman correlation and the separate association with decision count. TH08 uses observed physical gameplay duration without an independently sampled stage tick. TH11 uses its settled native score. Other titles use native stage 1 progress. Resample complete games within recorded host and report percentile intervals; n=16 per cell. Descriptive and outcome-blind to held games, not a causal reward-component ablation. All sources precede the six-title update.')
    freeze(root/'protocol.json',plan)
    native={digest(f):(t,f) for t,f in records(d/'five-game-generation3',True,True)}
    eight={}
    for f in (s/'selected-training').glob('*.json'):
        v=json.loads(f.read_bytes())
        if 'record' in v:eight[v['record']['source_log_sha256']]=v
    groups=defaultdict(list)
    for row in rows:
        f=manifest.parent/row['file']
        if row['role']!='train' or digest(f)!=row['sha256']:raise ValueError('Changed source')
        with np.load(f,allow_pickle=False) as z:
            reward=z['reward'].reshape(-1)
            n=playable_length(row['title'],z.get('native_action_eligible'),reward,row.get('postterminal_input_boundary'))
            reward=reward[:n].astype(float)
        if row['title']==8:
            v=eight[row['source_log_sha256']];progress=v['audit']['observed_duration_s'];metric='original-camera elapsed seconds to first Continue';host='Gamer'
        else:
            t,physical=native[row['source_log_sha256']]
            if t!=row['title']:raise ValueError('Wrong original title')
            source=json.loads(physical.read_bytes());progress,metric=original_progress(t,physical,source);host=source['host']
        groups[(row['title'],row['difficulty'])].append(dict(seed=row['seed'],host=host,progress_metric=metric,progress=progress,decisions=n,
            undiscounted=float(reward.sum()),discounted=float(reward@np.power(.999,np.arange(n))),nonzero_reward_rows=int(np.count_nonzero(reward))))
    final={}
    for (t,rank),rs in sorted(groups.items()):
        if len(rs)!=16:raise ValueError('Expected sixteen original credits per cell')
        x=np.array([v['progress'] for v in rs]);rng=np.random.default_rng(171030+t*10+rank)
        hosts=sorted({v['host'] for v in rs});indices=[np.flatnonzero([v['host']==h for v in rs]) for h in hosts]
        sampled=np.column_stack([rng.choice(ix,(1500,len(ix))) for ix in indices]);analysis={}
        for key in ('discounted','undiscounted','decisions'):
            y=np.array([v[key] for v in rs]);point=correlation(x,y)
            bx=np.stack([ranks(v) for v in x[sampled]])
            by=np.stack([ranks(v) for v in y[sampled]])
            bx-=bx.mean(1,keepdims=True);by-=by.mean(1,keepdims=True)
            norm=np.sqrt((bx*bx).sum(1)*(by*by).sum(1));good=norm>0;vals=(bx[good]*by[good]).sum(1)/norm[good]
            analysis[key]=dict(spearman=point,within_host_bootstrap95=np.quantile(vals,[.025,.975]).tolist() if len(vals) else None,
                defined_draws=len(vals),minimum=float(y.min()),maximum=float(y.max()))
        final[f'TH{t:02}-{rank}']=dict(metrics=analysis,records=rs)
        print(t,rank,'discounted return/progress',analysis['discounted']['spearman'],flush=True)
    freeze(root/'report.json',dict(registration=digest(root/'protocol.json'),cells=final))

if __name__=='__main__':
    main()
