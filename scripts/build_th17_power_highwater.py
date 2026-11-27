"""Build a signed, decision-aligned TH17 power supplement from frozen G2 native training rows."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/rollout-worker-probe/th17-full'))
from timeline import coordinates

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save_json(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def table(path):
    return np.genfromtxt(path,names=True,delimiter='\t',dtype=None,encoding='utf8',ndmin=1)

def build(folder,out):
    witness=json.loads((folder/'receipt.json').read_bytes())
    if witness['split']!='train' or witness['arm']!='four-ppo' or witness['observer_version']!=2 or witness['terminal']!='game_over':raise ValueError(folder)
    sources={k:sha(folder/f) for k,f in [('trajectory_sha256','behavior.npz'),('native_trace_sha256','native.tsv'),('decisions_sha256','decisions.tsv')]}
    if any(sources[k]!=witness[k] for k in sources):raise ValueError('Source changed')
    t=table(folder/'decisions.tsv');z=table(folder/'native.tsv')
    n=len(t);seed=witness['seed']
    with np.load(folder/'behavior.npz',allow_pickle=False) as behavior:
        assert n==witness['actions'] and behavior['terminated'].sum()==1 and behavior['terminated'][-1]
        assert np.array_equal(behavior['action'],np.stack([t[k] for k in ('move','focus','shoot','bomb')],axis=1))
        old_reward=behavior['reward'].astype('f4')
    if not np.array_equal(t['action_t'],np.arange(n)) or not np.all(t['observed_keys']==t['keys']):raise ValueError('Incomplete actual action sequence')
    valid=z['stage']>=1
    coordinates(z['stage'][valid],z['tick'][valid])
    # Match the full native identity returned by the shared-state observer at
    # both boundaries. Its stage, music tick and stock are evidence of identity.
    # Native chronology is the file row order; the music clock is never sorted.
    lookup=defaultdict(list)
    for i,row in enumerate(z):
        key=tuple(int(row[k]) for k in ('stage','tick','total_tick','spares','bombs'))
        lookup[key].append(i)
    def match(cols):
        rows=[]
        for row in t:
            keys=tuple(int(row[k]) for k in cols)
            matches=lookup[keys]
            if not matches or len({int(z['power'][i]) for i in matches})!=1:raise ValueError(('Missing/ambiguous native power',seed,keys))
            rows.append(matches[0])
        return np.asarray(rows,dtype='i4')
    pre=match(('stage','tick','total_tick','spares_before','bombs_before'))
    post=match(('after_stage','after_tick','after_total_tick','spares','bombs'))
    if np.any(pre>post) or np.any(post[:-1]>pre[1:]):raise ValueError('Native action boundary order reversed')
    lo,hi=int(pre[0]),int(post[-1])
    if tuple(int(z[lo][k]) for k in ('stage','spares','bombs','power'))!=(1,2,3,100) or z[hi]['spares']!=-1:raise ValueError('Unexpected native endpoint')
    highest=100;increments=np.zeros(n,'i2');events=[];ignored_recollection=0
    for i in range(lo+1,hi+1):
        old,row=z[i-1],z[i]
        power=int(row['power'])
        if power<0 or power>400:raise ValueError('Power outside original 0..4.00')
        if row['spares']<0:continue
        if power<=highest:
            if row['spares']==old['spares']:ignored_recollection+=max(0,int(power-old['power']))
            continue
        if row['spares']!=old['spares'] or old['power']<100:raise ValueError('New high is confounded by stock initialization')
        action=int(np.searchsorted(pre,i,side='left')-1)
        if not 0<=action<n or not pre[action]<i:raise ValueError('Power event preceding actual decision')
        units=power-highest;increments[action]+=units
        ignored_recollection+=max(0,highest-int(old['power']))
        gap=bool(i>post[action])
        if gap and (action+1>=n or i>pre[action+1]):raise ValueError('Uncovered native action interval')
        events.append(dict(action_t=action,native_row_zero_based=int(i),native_sensor_ms=int(row['ms']),stage=int(row['stage']),tick=int(row['tick']),
            spares=int(row['spares']),bombs=int(row['bombs']),power=power,previous_high=highest,added_units=units,
            action_onset_ns=int(t['onset_ns'][action]),action_observed_ns=int(t['observed_ns'][action]),during_held_key_gap=gap))
        highest=power
    high=100+np.cumsum(increments,dtype='i4')
    total=np.minimum(.15,(high-100)*.0005)
    bonus=np.diff(np.r_[0.,total]).astype('f4')
    if np.any(bonus<0) or highest!=int(max(z['power'][lo:hi+1])) or abs(float(bonus.sum())-min(.15,.0005*(highest-100)))>1e-6:raise ValueError('Supplement failed to telescope')
    dest=out/f'{seed}.npz'
    temp=out/f'{seed}.tmp.npz'
    np.savez_compressed(temp,action_t=t['action_t'].astype('i4'),onset_ns=t['onset_ns'],observed_ns=t['observed_ns'],
        stage_before=t['stage'].astype('i2'),tick_before=t['tick'].astype('i4'),stage_after=t['after_stage'].astype('i2'),tick_after=t['after_tick'].astype('i4'),
        native_row_before=pre,native_row_after=post,power_before=z['power'][pre].astype('i2'),power_after=z['power'][post].astype('i2'),
        spares_before=t['spares_before'].astype('i2'),spares_after=t['spares'].astype('i2'),bombs_before=t['bombs_before'].astype('i2'),bombs_after=t['bombs'].astype('i2'),
        power_highwater_after=high.astype('i2'),power_record_gain_units=increments,reward_original=old_reward,reward_bonus=bonus,
        trajectory_sha256=np.array(sources['trajectory_sha256']),native_trace_sha256=np.array(sources['native_trace_sha256']),decisions_sha256=np.array(sources['decisions_sha256']))
    temp.replace(dest)
    def stock(i):return {k:int(z[i][k]) for k in ('stage','tick','spares','bombs','power')}
    receipt=dict(seed=seed,rank=witness['difficulty'],title=17,actions=n,steps=n,policy_sha256=witness['policy_sha256'],**sources,
        behavior_sha256=sources['trajectory_sha256'],source_receipt_sha256=sha(folder/'receipt.json'),file=dest.name,npz=dest.name,
        overlay_sha256=sha(dest),initial=stock(lo),terminal=stock(hi),highwater_power=highest,
        power_record_gain_units=int(increments.sum()),ignored_nonrecord_positive_units=ignored_recollection,
        reward_bonus_sum=float(bonus.sum()),source_decision_endpoints_uniquely_power_matched=2*n,
        transitions_in_held_key_gap=sum(e['during_held_key_gap'] for e in events),events=events)
    save_json(out/f'{seed}.json',receipt)
    return {k:v for k,v in receipt.items() if k!='events'}|dict(event_receipt=f'{seed}.json',event_receipt_sha256=sha(out/f'{seed}.json'))

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('native',type=Path);a.add_argument('output',type=Path);args=a.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((args.native.parent/'camera-protocol.json').read_bytes())
    rows=[build(args.native/f'train-four-ppo-{s}',args.output) for s in protocol['train_seeds']['17']]
    if len(rows)!=16 or {r['seed'] for r in rows}!=set(range(1251701,1251717)):raise ValueError('Missing original G2 native credits')
    manifest=dict(schema='th17-power-highwater-v1',source_study='four-game-ppo-generation2',source_protocol_sha256=sha(args.native.parent/'camera-protocol.json'),
        source_policy_sha256=rows[0]['policy_sha256'],native_unit='one hundredth of displayed power',coefficient_per_power=0.05,cap_per_credit=0.15,
        attribution='Raw native rows from after the first pre-action state through the final after-action state. Record gains map to the last earlier native pre-action state; between an observed after-state and the next pre-state the mechanical keys are still held. Native row numbers are zero based, excluding TSV header. Source rows fully resolve observed power at every action boundary.',
        training_credits=len(rows),actions=sum(r['actions'] for r in rows),total_bonus=sum(r['reward_bonus_sum'] for r in rows),credits=rows)
    save_json(args.output/'manifest.json',manifest)
    print('Native original high-water overlay',len(rows),'credits',manifest['actions'],'decisions',manifest['total_bonus'],'reward units')

if __name__=='__main__':main()
