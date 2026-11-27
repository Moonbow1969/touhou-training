"""Independently recheck original controls and summarize paired TH12 time-control cohorts."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/rollout-worker-probe/th12-full'))
from native_control_quality import evaluate
from report_th12_native_pairs import read_credit

def build(directory,split):
    registration=json.loads((directory/'clock-protocol.json').read_bytes())
    planned=[x for x in registration['plan'] if x['mode']==split]
    groups=defaultdict(list);paired=defaultdict(dict);details=[]
    for job in planned:
        d=directory/job['cohort']/'native12'
        receipt=json.loads((d/'credits'/(job['tag']+'.json')).read_bytes())
        raw=d/'raw'/job['tag']
        checks=evaluate(raw,job['execution_clock'])
        if checks!=receipt['control_clock_verification'] or any(receipt[k]!=job[m] for k,m in [('episode','tag'),('host','host'),('lane','lane'),('seed','seed'),('difficulty','rank'),('policy','policy')]):
            raise ValueError(f'Signed control or allocation mismatch {job["tag"]}')
        source=read_credit(d,receipt,{})
        if not source['sensor_verified']:raise ValueError(f'Missing original native stage proof {job["tag"]}')
        arm=job['execution_clock'] if split=='train' else receipt['policy']
        groups[job['host'],arm].append(checks)
        paired[receipt['seed']][arm]=source
        details.append(dict(tag=job['tag'],arm=arm,rank=job['rank'],host=job['host'],lane=job['lane'],trajectory_sha256=receipt['trajectory_sha256'],stock_stage=source,game_clock=checks))
    by_host={}
    for (host,arm),samples in groups.items():
        histogram=Counter()
        for r in samples:histogram.update(r['native_gap_histogram'])
        n=sum(x['decisions'] for x in samples);matches=sum(x['native_key_matches'] for x in samples);eligible=sum(x['eligible_native_end_of_hold'] for x in samples)
        by_host.setdefault(host,{})[arm]=dict(credits=len(samples),actions=n,native_key_matches=matches,eligible=eligible,match_probability=matches/eligible,coverage=eligible/n,
            native_interdecision_gap_histogram=dict(sorted(histogram.items(),key=lambda kv:int(kv[0]))),fraction_exactly_four=histogram['4']/sum(histogram.values()),mean_native_ticks_per_observation=sum(int(g)*v for g,v in histogram.items())/sum(histogram.values()))
    a,b=('wall120','tick4') if split=='train' else ('wall120-ppo','tick4-ppo')
    ranks={}
    for rank in (2,3):
        pairs=[(p[a],p[b]) for p in paired.values() if p[a]['rank']==rank]
        if any((x['seed'],x['host'],x['lane'],x['rank'])!=(y['seed'],y['host'],y['lane'],y['rank']) for x,y in pairs):raise ValueError('Mismatched pair')
        cells={}
        for key in ('confirmed_stage2','stage1_last_native_tick','first_spare_loss_seconds','final_spares','final_bombs'):
            x=np.asarray([p[0][key] for p in pairs],float);y=np.asarray([p[1][key] for p in pairs],float)
            if not np.all(np.isfinite(x+y)):raise ValueError((rank,key))
            hosts=np.array([p[0]['host'] for p in pairs]);resampled=[];rng=np.random.default_rng(261126)
            for host in np.unique(hosts):
                delta=(y-x)[hosts==host];resampled.append(rng.choice(delta,(20000,len(delta)),replace=True))
            cells[key]=dict(reference_mean=float(x.mean()),candidate_mean=float(y.mean()),mean_delta=float((y-x).mean()),host_stratified_bootstrap95=np.quantile(np.concatenate(resampled,axis=1).mean(axis=1),[.025,.975]).tolist())
        ranks[str(rank)]=dict(pairs=len(pairs),endpoints=cells)
    return dict(split=split,reference=a,candidate=b,note='TRAIN pairs sample the same frozen human SFT under two native controller clocks. HELD compares independently fitted PPO policies at the identical four-native-tick deployment cadence. A paired controller seed does not fix the original game RNG. Eligible input checks require a post-input tick with a stable original stage manager.',registration_sha256=hashlib.sha256((directory/'clock-protocol.json').read_bytes()).hexdigest(),by_host=by_host,by_rank=ranks,credits=details)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('split',choices=['train','held']);p.add_argument('output',type=Path)
    a=p.parse_args();r=build(a.root,a.split);a.output.write_text(json.dumps(r,indent=2)+'\n')
