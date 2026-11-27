"""Train a matched PPO2 ablation with only the original TH10 spirit-strike spend cost removed."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from scripts.collect_th10_full_action import save
from scripts.package_th10_credit_logs import package
from scripts.run_th10_camera_pair import run as paired


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_strikes(folder,utc):
    with gzip.open(folder/'native.csv.gz','rt') as f:
        rows=list(csv.DictReader(f))
    prev=rows[0];last_loss=-10**15;last_press=-10**15;events=[]
    for row in rows:
        t=int(row['utc_ms']);power=int(row['power_units']);last_power=int(prev['power_units'])
        spare=int(row['lives']);last_spare=int(prev['lives']);word=int(row['native_input'])
        if spare<last_spare:last_loss=t
        if word&2 and not int(prev['native_input'])&2:last_press=t
        if last_power-power==20 and spare==last_spare and word&2 and t-last_press<=320 and t-last_loss>=500:
            i=int(np.clip(np.searchsorted(utc,t,side='right')-1,0,len(utc)-1))
            events.append(dict(step=i,power_before=last_power/20,power_after=power/20,lives=spare))
        prev=row
    return events


def prepare(root,out):
    source=root/'camera-round2/camera-batch';original=source/'credits.json'
    control=json.loads((root/'camera-ppo2/report.json').read_bytes())
    policy=root/'camera-ppo/policy.ts';parent=root/'camera-ppo/camera-ppo.pt'
    protocol=root/'camera-round2/camera-protocol.json'
    assert sha(original)==control['manifest_sha256'] and sha(protocol)==control['protocol_sha256']
    assert sha(parent)==control['parent_actor_critic_sha256'] and sha(policy)==control['native_behavior_sha256']
    old=json.loads(original.read_bytes());assert len(old)==48
    counts={t:sum(r['title']==t for r in old) for t in (10,12,13)};assert set(counts.values())=={16}
    batch=out/'batch';batch.mkdir(exist_ok=True)
    records=[];proof=[]
    for r in old:
        src=source/r['file'];dst=batch/r['file'];assert sha(src)==r['sha256']
        alt=r.copy()
        if r['title']!=10:
            if not dst.exists():
                try:os.link(src,dst)
                except OSError:shutil.copyfile(src,dst)
            assert sha(dst)==r['sha256'];records.append(alt);continue
        d=root/'camera-round2/native10'/f"momentum-{r['seed']}"
        native=json.loads((d/'report.json').read_bytes())
        assert sha(d/'report.json')==r['source_log_sha256'] and sha(d/'behavior.npz')==r['source_actions_sha256']
        with np.load(src) as z,np.load(d/'behavior.npz') as b,np.load(d/'original-pixels.npz') as pix:
            data={k:z[k] for k in z.files}
            assert np.array_equal(data['frames'],pix['obs'][:-1])
            for k in ('action','logp','value','reward','terminated'):assert np.array_equal(data[k],b[k])
            events=native_strikes(d,b['action_utc_ms'])
            assert events==native['successful_bombs']
            expected=np.zeros(len(data['reward']),dtype=np.float32)
            for e in events:expected[e['step']]+=np.float32(.03)
            assert np.array_equal(-b['reward_bomb'],expected)
            before=data['reward'].copy();new=before+expected
            assert np.allclose(new,b['reward_life']+b['reward_survival']+b['reward_stage'],atol=2e-7)
            changed=np.flatnonzero(new!=before)
            assert changed.tolist()==np.flatnonzero(expected).tolist()
            data['reward']=new
        if len(changed):
            temp=dst.with_suffix('.tmp')
            with temp.open('wb') as f:np.savez_compressed(f,**data)
            temp.replace(dst)
        elif not dst.exists():
            try:os.link(src,dst)
            except OSError:shutil.copyfile(src,dst)
        with np.load(src) as old_data,np.load(dst) as new_data:
            assert old_data.files==new_data.files
            assert all(np.array_equal(old_data[k],new_data[k]) for k in old_data if k!='reward')
            assert np.array_equal(new_data['reward'],data['reward'])
        alt['sha256']=sha(dst);records.append(alt)
        proof.append(dict(seed=r['seed'],steps=len(before),old_file_sha256=r['sha256'],new_file_sha256=alt['sha256'],
            native_sensor_sha256=sha(d/'native.csv.gz'),confirmed_original_events=events,changed_reward_steps=changed.tolist(),
            old_return=float(before.sum(dtype=np.float64)),new_return=float(new.sum(dtype=np.float64)),expected_addback=float(expected.sum(dtype=np.float64))))
    save(batch/'credits.json',records)
    total_events=sum(len(r['confirmed_original_events']) for r in proof)
    delta=sum(r['new_return']-r['old_return'] for r in proof)
    assert abs(delta-.03*total_events)<2e-5
    save(out/'reward-audit.json',dict(original_manifest_sha256=sha(original),variant_manifest_sha256=sha(batch/'credits.json'),
        native_behavior_sha256=sha(policy),parent_actor_critic_sha256=sha(parent),unchanged_training_protocol_sha256=sha(protocol),
        credits_by_title=counts,total_steps=control['steps'],th10_events=total_events,aggregate_return_addback=delta,
        untouched_non_th10_credits=sum(r['title']!=10 and r['sha256']==s['sha256'] for r,s in zip(records,old)),
        change='Add 0.03 at each previous TH10 reward step with a verified native spirit strike (1.00 Power, matching Bomb input and game stock). All observations, actions, timestamps, original terminal flags and all other rewards remain bitwise identical.',th10_credits=proof))
    return batch,protocol,parent,policy


def compare_probabilities(root,out):
    import torch
    torch.set_num_threads(2)
    plain=torch.jit.load(str(root/'camera-ppo2/policy.ts'),map_location='cpu').eval()
    variant=torch.jit.load(str(out/'policy/policy.ts'),map_location='cpu').eval()
    audit=json.loads((out/'reward-audit.json').read_bytes())
    subsets={k:[] for k in ('all_native_train_decisions','spirit_strike_steps_from_parent','other_steps')}
    with torch.inference_mode():
        for row in audit['th10_credits']:
            f=out/'batch'/f"th10-{row['seed']}.npz"
            with np.load(f) as b:frames=torch.from_numpy(b['frames']).float()/255
            diff=torch.zeros_like(frames);diff[1:]=(frames[1:]-frames[:-1]).abs()
            record=[]
            for batch in torch.stack((frames,diff),1).split(32):
                old=plain(batch);new=variant(batch)
                p=old[3].softmax(1)[:,1];q=new[3].softmax(1)[:,1]
                movement=.5*(new[0].softmax(1)-old[0].softmax(1)).abs().sum(1)
                record.append(torch.stack((p,q,movement),1))
            values=torch.cat(record).numpy();cast=np.zeros(len(values),bool);cast[row['changed_reward_steps']]=True
            for group,mask in [('all_native_train_decisions',np.ones(len(values),bool)),('spirit_strike_steps_from_parent',cast),('other_steps',~cast)]:
                subsets[group].append(values[mask])
    result={}
    for name,parts in subsets.items():
        sample=np.concatenate(parts);p,q,tv=sample.mean(0)
        result[name]=dict(n=len(sample),mean_plain_bomb_probability=float(p),mean_variant_bomb_probability=float(q),
            mean_variant_minus_plain_pp=float(100*(sample[:,1]-sample[:,0]).mean()),mean_movement_total_variation=float(tv))
    save(out/'fixed-training-state-probabilities.json',dict(scope='Both updated policies on identical TH10 training-state images. Cast rows were selected by original native events under the common PPO1 behavior. This does not measure the ablation gameplay effect.',groups=result))


def run(args):
    root=args.root.resolve();out=root/'th10-spend-ablation';out.mkdir(exist_ok=True)
    seeds=list(range(881001,881013))
    save(out/'protocol.json',dict(train='Exactly the original PPO2 48-credit three-title batch; only remove the 0.03 TH10 confirmed-spirit-strike cost.',
        optimizer_seed=261127,learner='scripts.train_full_resolution_camera_ppo',comparison='Existing plain camera PPO2 vs zero-TH10-spend-cost PPO2',
        held=seeds,host='accelerator',game='Official original TH10 Normal trial 0.02a, all starting lives, all four actions, no Continue',
        nominal_control_hold_ms=120,design='First six prespecified seeds see plain PPO2 first, last six see the reward variant first. Record both native endpoints and full sensor traces. Highest reached native stage, entry stocks, power-paid strikes, original score, and movement switching.'))
    batch,protocol,parent,policy=prepare(root,out)
    while not (root/'cadence-native10/factorial.json').is_file():time.sleep(3)
    trained=out/'policy'
    if not (trained/'report.json').is_file():
        repo=Path(__file__).resolve().parents[1]
        cmd=[sys.executable,'-u','-m','scripts.train_full_resolution_camera_ppo','--manifest',str(batch/'credits.json'),
             '--protocol',str(protocol),'--parent-actor-critic',str(parent),'--behavior',str(policy),'--out',str(trained)]
        with (out/'training.log').open('ab') as log:subprocess.run(cmd,cwd=repo,stdout=log,stderr=log,check=True)
    result=json.loads((trained/'report.json').read_bytes())
    assert result['manifest_sha256']==sha(batch/'credits.json') and result['parent_actor_critic_sha256']==sha(parent)
    assert any(e['accepted'] for e in result['history'])
    compare_probabilities(root,out)
    paired(SimpleNamespace(out=out/'native10',workers=args.workers.resolve(),host='accelerator',
        reference_name='plain-ppo2',candidate_name='no-spend-cost',reference=root/'camera-ppo2/policy.ts',
        candidate=trained/'policy.ts',candidate_history=False,concurrency=3,seeds=seeds))
    package(out/'th10-native-logs.zip',[out/'native10/plain-ppo2',out/'native10/no-spend-cost'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=Path,required=True)
    run(p.parse_args())
